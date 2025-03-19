import sys

from widgets import *
from main_window import MainWindow
from modules import *
import resources_rc


# =========================================================
# =======          绑定文件批量处理modules          =========
# =========================================================
class BatchFilesBinding(QThread):
    def __init__(self, object, bind_name):
        super().__init__()
        self.work_folder = ''
        self.wanted_items = []
        self.bind_name = bind_name
        self.handler_object = object  # 直接接收处理对象实例

    def update_setting(self, object_name, attribute, value):
        # 检查处理对象是否与传递的 object_name 匹配，并且对象有这个属性
        if self.handler_object.__class__.__name__ == object_name and hasattr(self.handler_object,
                                                                             attribute):
            setattr(self.handler_object, attribute, value)
            print(f"From BatchFilesBinding:\n\tUpdated {attribute} to {value} in {object_name}\n")

    def update_user_select(self, work_folder: str, wanted_items: str):
        self.work_folder = work_folder
        self.wanted_items = wanted_items

    def run(self):
        # 在线程中执行耗时操作
        self.handler_object.set_work_folder(self.work_folder)
        self.handler_object.selected_dirs_handler(self.wanted_items)

    def handler_binding(self):
        # 启动线程
        self.start()


# =========================================================
# =======                main函数                 =========
# =========================================================
def main():
    # 初始化app
    app = QApplication(sys.argv)

    # 初始化主窗口
    window = MainWindow()

    # ==========================绑定mergecolors==========================
    # 初始化绑定,初始化可以直接设置需要的颜色通道: colors=["R", "G", "B"]
    merge_colors_bind = BatchFilesBinding(MergeColors(colors=["R", "G"]), '颜色通道合成')

    # FileWindow信号数据selected_signal传入绑定对象merge_colors_bind
    window.FileWindow.selected_signal.connect(merge_colors_bind.update_user_select)

    # dicom对象信息信号绑定FileWindow显示
    merge_colors_bind.handler_object.result_signal.connect(window.FileWindow.set_operation_result)

    # FileWindow创建对应的操作按钮绑定modules的操作
    window.FileWindow.add_file_operation(merge_colors_bind.bind_name,
                                         merge_colors_bind.handler_binding)

    # 设置的参数被修改的信号数据绑定update_setting函数
    window.SettingWindow.settings.changed_signal.connect(merge_colors_bind.update_setting)

    # =============================绑定dicom=============================
    dicom_bind = BatchFilesBinding(DicomToImage(), 'DICOM处理')
    window.FileWindow.selected_signal.connect(dicom_bind.update_user_select)
    dicom_bind.handler_object.result_signal.connect(window.FileWindow.set_operation_result)
    window.FileWindow.add_file_operation(dicom_bind.bind_name, dicom_bind.handler_binding)
    window.SettingWindow.settings.changed_signal.connect(dicom_bind.update_setting)

    # =============================绑定SplitColors=============================
    # 初始化绑定,初始化可以直接设置需要的颜色通道: colors=["R", "G", "B"]
    split_color_bind = BatchFilesBinding(SplitColors(colors=["R", "G"]), '分离颜色通道')
    window.FileWindow.selected_signal.connect(split_color_bind.update_user_select)
    split_color_bind.handler_object.result_signal.connect(window.FileWindow.set_operation_result)
    window.FileWindow.add_file_operation(split_color_bind.bind_name,
                                         split_color_bind.handler_binding)
    window.SettingWindow.settings.changed_signal.connect(split_color_bind.update_setting)

    # =============================绑定TwistImgs=============================
    twisted_corner = [[0, 0], [430, 82], [432, 268], [0, 276]]
    twist_shape_bind = BatchFilesBinding(TwistImgs(twisted_corner=twisted_corner), '图片视角变换')
    window.FileWindow.selected_signal.connect(twist_shape_bind.update_user_select)
    twist_shape_bind.handler_object.result_signal.connect(window.FileWindow.set_operation_result)
    window.FileWindow.add_file_operation(twist_shape_bind.bind_name,
                                         twist_shape_bind.handler_binding)
    window.SettingWindow.settings.changed_signal.connect(twist_shape_bind.update_setting)

    # =============================绑定BiliVideo=============================
    split_color_bind = BatchFilesBinding(BiliVideos(), 'B站视频导出')
    window.FileWindow.selected_signal.connect(split_color_bind.update_user_select)
    split_color_bind.handler_object.result_signal.connect(window.FileWindow.set_operation_result)
    window.FileWindow.add_file_operation(split_color_bind.bind_name,
                                         split_color_bind.handler_binding)
    window.SettingWindow.settings.changed_signal.connect(split_color_bind.update_setting)

    # =============================绑定ECGHandler=============================
    ecg_bind = BatchFilesBinding(ECGHandler(), 'ECG信号处理')
    window.FileWindow.selected_signal.connect(ecg_bind.update_user_select)
    ecg_bind.handler_object.result_signal.connect(window.FileWindow.set_operation_result)
    window.FileWindow.add_file_operation(ecg_bind.bind_name,
                                         ecg_bind.handler_binding)
    window.SettingWindow.settings.changed_signal.connect(ecg_bind.update_setting)

    # =============================绑定GenSubtitles=============================
    gen_subtitles_bind = BatchFilesBinding(GenSubtitles(), '视频字幕生成')
    window.FileWindow.selected_signal.connect(gen_subtitles_bind.update_user_select)
    gen_subtitles_bind.handler_object.result_signal.connect(window.FileWindow.set_operation_result)
    window.FileWindow.add_file_operation(gen_subtitles_bind.bind_name,
                                         gen_subtitles_bind.handler_binding)
    window.SettingWindow.settings.changed_signal.connect(gen_subtitles_bind.update_setting)

    # =============================app运行=============================
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
