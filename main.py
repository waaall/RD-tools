from __future__ import annotations

import sys

from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtWidgets import QApplication
from qfluentwidgets import FluentIcon as FIF

from core import MessageLevel, TaskLoader, TaskMessage
from main_window import MainWindow
from modules.app_settings import AppSettings
from modules.files_basic import FilesBasic
from ui import TaskDescriptor, apply_app_theme
from widgets.confirm_dialog import TaskExecutionConfirmDialog
from widgets.setting_page import humanize_setting_label


class BatchFilesBinding(QThread):
    result_signal = Signal(str, object)
    running_changed = Signal(str, bool)
    completed = Signal(str, bool, str)

    def __init__(self, descriptor: TaskDescriptor, settings: AppSettings, work_folder: str, wanted_items: list[str]):
        super().__init__()
        self.work_folder = work_folder
        self.wanted_items = wanted_items
        self.descriptor = descriptor
        self.task_key = descriptor.key
        self.display_name = descriptor.title
        self.settings = settings
        self._run_error: str | None = None

        self.finished.connect(self._handle_finished)

    def _forward_result(self, message):
        self.result_signal.emit(self.task_key, message)

    def _build_handler_params(self) -> dict[str, object]:
        params = self.settings.get_class_params(self.descriptor.settings_group)
        for key, value in self.descriptor.default_params.items():
            if params.get(key) in (None, {}):
                params[key] = value
        return params

    def _handle_finished(self):
        success = self._run_error is None
        self.running_changed.emit(self.task_key, False)
        if success:
            message = '任务执行完成。'
        else:
            message = f'任务执行失败: {self._run_error}'
        self.completed.emit(self.task_key, success, message)

    def run(self):
        self._run_error = None
        handler_object = None
        try:
            operation_cls = TaskLoader.load_class(self.descriptor.module_path, self.descriptor.class_name)
            params = self._build_handler_params()

            FilesBasic.set_bootstrap_reporter(self._forward_result)
            try:
                handler_object = operation_cls(**params)
            finally:
                FilesBasic.clear_bootstrap_reporter()

            handler_object.result_signal.connect(self._forward_result, Qt.QueuedConnection)
            handler_object.set_work_folder(self.work_folder)
            handler_object.selected_dirs_handler(self.wanted_items)
        except Exception as exc:
            self._run_error = str(exc)
            self.result_signal.emit(
                self.task_key,
                TaskMessage.build(f'任务执行失败: {self._run_error}', level=MessageLevel.ERROR),
            )
        finally:
            if handler_object is not None:
                handler_object.close_log_session()


def build_task_setting_lines(settings: AppSettings, descriptor: TaskDescriptor) -> list[str]:
    entries = settings.get_setting_entries('Batch_Files', group_name=descriptor.settings_group)
    return [
        f"{humanize_setting_label(entry['path'][-1])}: {format_setting_value(entry['value'])}"
        for entry in entries
    ]


def format_setting_value(value) -> str:
    if isinstance(value, bool):
        return 'True' if value else 'False'
    return str(value)


def release_active_binding(active_bindings: dict[str, BatchFilesBinding], task_key: str, binding: BatchFilesBinding):
    if active_bindings.get(task_key) is binding:
        active_bindings.pop(task_key, None)
    binding.deleteLater()


def launch_batch_operation(window: MainWindow, active_bindings: dict[str, BatchFilesBinding], descriptor: TaskDescriptor):
    existing_binding = active_bindings.get(descriptor.key)
    if existing_binding is not None and existing_binding.isRunning():
        window.FileWindow.append_operation_log(
            descriptor.key,
            TaskMessage.build('任务仍在运行，请稍候。', level=MessageLevel.WARNING),
        )
        return

    work_folder, wanted_items = window.FileWindow.get_selected_directories()
    if not work_folder:
        message = '请先选择工作目录。'
        window.FileWindow.set_selection_status(message, is_error=True)
        window.FileWindow.append_operation_log(
            descriptor.key,
            TaskMessage.build(f'未执行：{message}', level=MessageLevel.WARNING),
        )
        window.FileWindow.notify_blocking_issue(message)
        return

    if not wanted_items:
        message = '请至少勾选一个待处理目录。'
        window.FileWindow.set_selection_status(message, is_error=True)
        window.FileWindow.append_operation_log(
            descriptor.key,
            TaskMessage.build(f'未执行：{message}', level=MessageLevel.WARNING),
        )
        window.FileWindow.notify_blocking_issue(message)
        return

    if not TaskExecutionConfirmDialog.confirm(
        task_title=descriptor.title,
        selected_dirs=wanted_items,
        settings_lines=build_task_setting_lines(window.settings, descriptor),
        parent=window.FileWindow,
    ):
        window.FileWindow.append_operation_log(
            descriptor.key,
            TaskMessage.build('未执行：已取消执行确认。', level=MessageLevel.INFO),
        )
        return

    binding = BatchFilesBinding(descriptor, window.settings, work_folder, wanted_items)
    binding.result_signal.connect(window.FileWindow.append_operation_log, Qt.QueuedConnection)
    binding.running_changed.connect(window.FileWindow.set_task_running, Qt.QueuedConnection)
    binding.completed.connect(window.FileWindow.finish_task, Qt.QueuedConnection)
    binding.completed.connect(
        lambda *_args, task_key=descriptor.key, current_binding=binding: release_active_binding(
            active_bindings,
            task_key,
            current_binding,
        ),
        Qt.QueuedConnection,
    )

    active_bindings[descriptor.key] = binding
    window.FileWindow.set_selection_status(f'已准备执行，当前勾选了 {len(wanted_items)} 个目录。')
    window.FileWindow.log_operation_start(descriptor.key, work_folder, wanted_items)
    window.FileWindow.append_operation_log(
        descriptor.key,
        TaskMessage.build('正在加载任务实现...', level=MessageLevel.INFO),
    )
    binding.running_changed.emit(descriptor.key, True)
    binding.start()


def register_batch_operation(window: MainWindow, active_bindings: dict[str, BatchFilesBinding], descriptor: TaskDescriptor):
    window.FileWindow.register_task(
        descriptor,
        lambda current_descriptor=descriptor: launch_batch_operation(window, active_bindings, current_descriptor),
        has_settings=window.SettingWindow.has_task_settings(descriptor.key),
        open_settings_callback=lambda task_key=descriptor.key: window.open_task_settings(task_key),
    )


def order_task_descriptors(settings: AppSettings, descriptors: list[TaskDescriptor]) -> list[TaskDescriptor]:
    descriptor_map = {descriptor.key: descriptor for descriptor in descriptors}
    ordered_keys = settings.get_task_order([descriptor.key for descriptor in descriptors])
    return [
        descriptor_map[key]
        for key in ordered_keys
        if key in descriptor_map
    ]


def build_task_descriptors() -> list[TaskDescriptor]:
    return [
        TaskDescriptor(
            key='merge-colors',
            title='颜色通道合成',
            description='按文件名前缀配对 R/G/B 图像，并合成新的彩色结果图。',
            icon=FIF.APPLICATION,
            module_path='modules.merge_colors',
            class_name='MergeColors',
            settings_group='MergeColors',
            default_params={'colors': ['R', 'G']},
        ),
        TaskDescriptor(
            key='dicom-processing',
            title='DICOM处理',
            description='批量读取 DICOM 序列，导出图片并在需要时生成视频。',
            icon=FIF.FOLDER_ADD,
            module_path='modules.dicom_to_imgs',
            class_name='DicomToImage',
            settings_group='DicomToImage',
        ),
        TaskDescriptor(
            key='split-colors',
            title='分离颜色通道',
            description='把输入图片拆分为独立的 R/G/B 通道输出。',
            icon=FIF.SYNC,
            module_path='modules.split_colors',
            class_name='SplitColors',
            settings_group='SplitColors',
            default_params={'colors': ['R', 'G']},
        ),
        TaskDescriptor(
            key='twist-images',
            title='图片视角变换',
            description='按预设四边形参数对图片做透视变换。',
            icon=FIF.EDIT,
            module_path='modules.twist_shape',
            class_name='TwistImgs',
            settings_group='TwistImgs',
            default_params={'twisted_corner': [[0, 0], [430, 82], [432, 268], [0, 276]]},
        ),
        TaskDescriptor(
            key='bilibili-export',
            title='B站视频导出',
            description='批量修复并合并 Bilibili 缓存视频为可播放 MP4。',
            icon=FIF.PLAY,
            module_path='modules.bili_videos',
            class_name='BiliVideos',
            settings_group='BiliVideos',
        ),
        TaskDescriptor(
            key='ecg-handler',
            title='ECG信号处理',
            description='分析 ECG CSV 数据，生成原始、滤波和高级分析图表。',
            icon=FIF.IOT,
            module_path='modules.ECG_handler',
            class_name='ECGHandler',
            settings_group='ECGHandler',
        ),
        TaskDescriptor(
            key='subtitle-generation',
            title='字幕生成',
            description='对音视频文件批量抽取音频并生成 SRT 字幕。',
            icon=FIF.DOCUMENT,
            module_path='modules.gen_subtitles',
            class_name='GenSubtitles',
            settings_group='GenSubtitles',
        ),
        TaskDescriptor(
            key='files-renamer',
            title='批量重命名',
            description='按 prefix / all / body / between 规则批量重命名文件。',
            icon=FIF.EDIT,
            module_path='modules.files_renamer',
            class_name='FilesRenamer',
            settings_group='FilesRenamer',
        ),
        TaskDescriptor(
            key='mac-cleaner',
            title='Mac铲屎官',
            description='批量清理指定目录下的系统垃圾文件。',
            icon=FIF.FOLDER,
            module_path='modules.mac_poop_scooper',
            class_name='MacPoopScooper',
            settings_group='MacPoopScooper',
        ),
    ]


def main():
    app = QApplication(sys.argv)
    settings = AppSettings()
    settings.load_settings()
    task_descriptors = order_task_descriptors(settings, build_task_descriptors())
    apply_app_theme(settings.theme, app)

    window = MainWindow(settings, task_descriptors)
    active_bindings: dict[str, BatchFilesBinding] = {}

    for descriptor in task_descriptors:
        register_batch_operation(window, active_bindings, descriptor)

    window.show_for_launch()
    for warning in settings.consume_startup_warnings():
        QTimer.singleShot(0, lambda message=warning: window.show_notification('warning', '任务顺序', message, duration=5000))
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
