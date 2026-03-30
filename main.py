from __future__ import annotations

import sys

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import QApplication
from qfluentwidgets import FluentIcon as FIF

from main_window import MainWindow
from modules.app_settings import AppSettings
from modules.bili_videos import BiliVideos
from modules.dicom_to_imgs import DicomToImage
from modules.ECG_handler import ECGHandler
from modules.gen_subtitles import GenSubtitles
from modules.mac_poop_scooper import MacPoopScooper
from modules.merge_colors import MergeColors
from modules.split_colors import SplitColors
from modules.sum_subtitles import SumSubtitles
from modules.twist_shape import TwistImgs
from ui import TaskDescriptor, apply_app_theme


class BatchFilesBinding(QThread):
    result_signal = Signal(str, str)
    running_changed = Signal(str, bool)
    completed = Signal(str, bool, str)

    def __init__(self, handler_object, descriptor: TaskDescriptor, file_window):
        super().__init__()
        self.work_folder = ''
        self.wanted_items: list[str] = []
        self.task_key = descriptor.key
        self.display_name = descriptor.title
        self.handler_object = handler_object
        self.file_window = file_window
        self._run_error: str | None = None

        self.handler_object.result_signal.connect(self._forward_result, Qt.QueuedConnection)
        self.finished.connect(self._handle_finished)

    def _forward_result(self, message):
        self.result_signal.emit(self.task_key, message)

    def _handle_finished(self):
        success = self._run_error is None
        self.running_changed.emit(self.task_key, False)
        if success:
            message = '任务执行完成。'
        else:
            message = f'任务执行失败: {self._run_error}'
        self.completed.emit(self.task_key, success, message)

    def update_setting(self, object_name, attribute, value):
        if self.handler_object.__class__.__name__ == object_name and hasattr(self.handler_object, attribute):
            setattr(self.handler_object, attribute, value)
            print(f'From BatchFilesBinding:\n\tUpdated {attribute} to {value} in {object_name}\n')

    def run(self):
        self._run_error = None
        try:
            self.handler_object.set_work_folder(self.work_folder)
            self.handler_object.selected_dirs_handler(self.wanted_items)
        except Exception as exc:
            self._run_error = str(exc)
            self.result_signal.emit(self.task_key, f'Error: {self._run_error}')

    def handler_binding(self):
        if self.isRunning():
            self.file_window.append_operation_log(self.task_key, '任务仍在运行，请稍候。')
            return

        work_folder, wanted_items = self.file_window.get_selected_directories()
        if not work_folder:
            message = '请先选择工作目录。'
            self.file_window.set_selection_status(message, is_error=True)
            self.file_window.append_operation_log(self.task_key, f'未执行：{message}')
            self.file_window.notify_blocking_issue(message)
            return

        if not wanted_items:
            message = '请至少勾选一个待处理目录。'
            self.file_window.set_selection_status(message, is_error=True)
            self.file_window.append_operation_log(self.task_key, f'未执行：{message}')
            self.file_window.notify_blocking_issue(message)
            return

        self.work_folder = work_folder
        self.wanted_items = wanted_items
        self.file_window.set_selection_status(f'已准备执行，当前勾选了 {len(wanted_items)} 个目录。')
        self.file_window.log_operation_start(self.task_key, work_folder, wanted_items)
        self.running_changed.emit(self.task_key, True)
        self.start()


def register_batch_operation(window: MainWindow, bindings: list[BatchFilesBinding], descriptor: TaskDescriptor):
    params = window.settings.get_class_params(descriptor.operation_cls.__name__)
    for key, value in descriptor.default_params.items():
        params.setdefault(key, value)

    operation_object = descriptor.operation_cls(**params)
    binding = BatchFilesBinding(operation_object, descriptor, window.FileWindow)
    binding.result_signal.connect(window.FileWindow.append_operation_log, Qt.QueuedConnection)
    binding.running_changed.connect(window.FileWindow.set_task_running, Qt.QueuedConnection)
    binding.completed.connect(window.FileWindow.finish_task, Qt.QueuedConnection)
    window.FileWindow.register_task(descriptor, binding.handler_binding)
    window.settings.changed_signal.connect(binding.update_setting)
    bindings.append(binding)


def build_task_descriptors() -> list[TaskDescriptor]:
    return [
        TaskDescriptor(
            key='merge-colors',
            title='颜色通道合成',
            description='按文件名前缀配对 R/G/B 图像，并合成新的彩色结果图。',
            icon=FIF.APPLICATION,
            operation_cls=MergeColors,
            default_params={'colors': ['R', 'G']},
        ),
        TaskDescriptor(
            key='dicom-processing',
            title='DICOM处理',
            description='批量读取 DICOM 序列，导出图片并在需要时生成视频。',
            icon=FIF.FOLDER_ADD,
            operation_cls=DicomToImage,
        ),
        TaskDescriptor(
            key='split-colors',
            title='分离颜色通道',
            description='把输入图片拆分为独立的 R/G/B 通道输出。',
            icon=FIF.SYNC,
            operation_cls=SplitColors,
            default_params={'colors': ['R', 'G']},
        ),
        TaskDescriptor(
            key='twist-images',
            title='图片视角变换',
            description='按预设四边形参数对图片做透视变换。',
            icon=FIF.EDIT,
            operation_cls=TwistImgs,
            default_params={'twisted_corner': [[0, 0], [430, 82], [432, 268], [0, 276]]},
        ),
        TaskDescriptor(
            key='bilibili-export',
            title='B站视频导出',
            description='批量修复并合并 Bilibili 缓存视频为可播放 MP4。',
            icon=FIF.PLAY,
            operation_cls=BiliVideos,
        ),
        TaskDescriptor(
            key='ecg-handler',
            title='ECG信号处理',
            description='分析 ECG CSV 数据，生成原始、滤波和高级分析图表。',
            icon=FIF.IOT,
            operation_cls=ECGHandler,
        ),
        TaskDescriptor(
            key='subtitle-generation',
            title='字幕生成',
            description='对音视频文件批量抽取音频并生成 SRT 字幕。',
            icon=FIF.DOCUMENT,
            operation_cls=GenSubtitles,
        ),
        TaskDescriptor(
            key='subtitle-summary',
            title='字幕总结',
            description='读取字幕内容，调用大模型生成摘要并导出 PDF。',
            icon=FIF.LIBRARY,
            operation_cls=SumSubtitles,
        ),
        TaskDescriptor(
            key='mac-cleaner',
            title='Mac铲屎官',
            description='批量清理指定目录下的系统垃圾文件。',
            icon=FIF.FOLDER,
            operation_cls=MacPoopScooper,
        ),
    ]


def main():
    app = QApplication(sys.argv)
    settings = AppSettings()
    settings.load_settings()
    apply_app_theme(settings.theme, app)

    window = MainWindow(settings)
    bindings: list[BatchFilesBinding] = []

    for descriptor in build_task_descriptors():
        register_batch_operation(window, bindings, descriptor)

    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
