import sys

from widgets import *
from main_window import MainWindow
from modules import *


# =========================================================
# =======          绑定文件批量处理modules          =========
# =========================================================
class BatchFilesBinding(QThread):
    result_signal = Signal(str, str)

    def __init__(self, handler_object, bind_name, file_window):
        super().__init__()
        self.work_folder = ''
        self.wanted_items = []
        self.bind_name = bind_name
        self.handler_object = handler_object
        self.file_window = file_window

        self.handler_object.result_signal.connect(self._forward_result, Qt.QueuedConnection)

    def _forward_result(self, message):
        self.result_signal.emit(self.bind_name, message)

    def update_setting(self, object_name, attribute, value):
        if self.handler_object.__class__.__name__ == object_name and hasattr(self.handler_object,
                                                                             attribute):
            setattr(self.handler_object, attribute, value)
            print(f"From BatchFilesBinding:\n\tUpdated {attribute} to {value} in {object_name}\n")

    def run(self):
        try:
            self.handler_object.set_work_folder(self.work_folder)
            self.handler_object.selected_dirs_handler(self.wanted_items)
        except Exception as exc:
            self.result_signal.emit(self.bind_name, f"Error: {str(exc)}")

    def handler_binding(self):
        if self.isRunning():
            self.file_window.append_operation_log(self.bind_name, "任务仍在运行，请稍候。")
            return

        work_folder, wanted_items = self.file_window.get_selected_directories()
        if not work_folder:
            self.file_window.set_selection_status("请先选择工作目录。", is_error=True)
            self.file_window.append_operation_log(self.bind_name, "未执行：请先选择工作目录。")
            return
        if not wanted_items:
            self.file_window.set_selection_status("请至少勾选一个待处理目录。", is_error=True)
            self.file_window.append_operation_log(self.bind_name, "未执行：请至少勾选一个待处理目录。")
            return

        self.work_folder = work_folder
        self.wanted_items = wanted_items
        self.file_window.set_selection_status(f"已准备执行，当前勾选了 {len(wanted_items)} 个目录。")
        self.file_window.log_operation_start(self.bind_name, work_folder, wanted_items)
        self.start()


def register_batch_operation(window, bindings, operation_cls, display_name, description, default_params=None):
    params = window.SettingWindow.settings.get_class_params(operation_cls.__name__)
    for key, value in (default_params or {}).items():
        params.setdefault(key, value)

    operation_object = operation_cls(**params)
    binding = BatchFilesBinding(operation_object, display_name, window.FileWindow)
    binding.result_signal.connect(window.FileWindow.append_operation_log, Qt.QueuedConnection)
    window.FileWindow.add_file_operation(display_name, description, binding.handler_binding)
    window.SettingWindow.settings.changed_signal.connect(binding.update_setting)
    bindings.append(binding)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.SettingWindow.settings.load_settings()
    bindings = []

    operation_specs = [
        (MergeColors, '颜色通道合成', '按文件名前缀配对 R/G/B 图像，并合成新的彩色结果图。', {"colors": ["R", "G"]}),
        (DicomToImage, 'DICOM处理', '批量读取 DICOM 序列，导出图片并在需要时生成视频。', None),
        (SplitColors, '分离颜色通道', '把输入图片拆分为独立的 R/G/B 通道输出。', {"colors": ["R", "G"]}),
        (TwistImgs, '图片视角变换', '按预设四边形参数对图片做透视变换。', {"twisted_corner": [[0, 0], [430, 82], [432, 268], [0, 276]]}),
        (BiliVideos, 'B站视频导出', '批量修复并合并 Bilibili 缓存视频为可播放 MP4。', None),
        (ECGHandler, 'ECG信号处理', '分析 ECG CSV 数据，生成原始、滤波和高级分析图表。', None),
        (GenSubtitles, '字幕生成', '对音视频文件批量抽取音频并生成 SRT 字幕。', None),
        (SumSubtitles, '字幕总结', '读取字幕内容，调用大模型生成摘要并导出 PDF。', None),
        (MacPoopScooper, 'Mac铲屎官', '批量清理指定目录下的系统垃圾文件。', None),
    ]

    for operation_cls, display_name, description, default_params in operation_specs:
        register_batch_operation(window,
                                 bindings,
                                 operation_cls,
                                 display_name,
                                 description,
                                 default_params=default_params)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
