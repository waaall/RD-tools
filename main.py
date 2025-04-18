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

    # 确保SettingWindow完全加载
    window.SettingWindow.settings.load_settings()

    # ==========================绑定mergecolors==========================
    # 获取MergeColors的初始化参数
    merge_colors_params = window.SettingWindow.settings.get_class_params("MergeColors")
    # 添加额外的必要参数
    if "colors" not in merge_colors_params:
        merge_colors_params["colors"] = ["R", "G"]
    # 初始化对象
    merge_colors_obj = MergeColors(**merge_colors_params)
    merge_colors_bind = BatchFilesBinding(merge_colors_obj, '颜色通道合成')
    # 绑定信号和槽
    window.FileWindow.selected_signal.connect(merge_colors_bind.update_user_select)
    merge_colors_bind.handler_object.result_signal.connect(window.FileWindow.set_operation_result)
    window.FileWindow.add_file_operation(merge_colors_bind.bind_name,
                                         merge_colors_bind.handler_binding)
    window.SettingWindow.settings.changed_signal.connect(merge_colors_bind.update_setting)

    # =============================绑定dicom=============================
    dicom_params = window.SettingWindow.settings.get_class_params("DicomToImage")
    dicom_obj = DicomToImage(**dicom_params)
    dicom_bind = BatchFilesBinding(dicom_obj, 'DICOM处理')
    window.FileWindow.selected_signal.connect(dicom_bind.update_user_select)
    dicom_bind.handler_object.result_signal.connect(window.FileWindow.set_operation_result)
    window.FileWindow.add_file_operation(dicom_bind.bind_name, dicom_bind.handler_binding)
    window.SettingWindow.settings.changed_signal.connect(dicom_bind.update_setting)

    # =============================绑定SplitColors=============================
    split_colors_params = window.SettingWindow.settings.get_class_params("SplitColors")
    # 添加额外的必要参数
    if "colors" not in split_colors_params:
        split_colors_params["colors"] = ["R", "G"]
    split_color_obj = SplitColors(**split_colors_params)
    split_color_bind = BatchFilesBinding(split_color_obj, '分离颜色通道')
    window.FileWindow.selected_signal.connect(split_color_bind.update_user_select)
    split_color_bind.handler_object.result_signal.connect(window.FileWindow.set_operation_result)
    window.FileWindow.add_file_operation(split_color_bind.bind_name,
                                         split_color_bind.handler_binding)
    window.SettingWindow.settings.changed_signal.connect(split_color_bind.update_setting)

    # =============================绑定TwistImgs=============================
    twist_params = window.SettingWindow.settings.get_class_params("TwistImgs")
    # 添加默认的twisted_corner参数
    if "twisted_corner" not in twist_params:
        twist_params["twisted_corner"] = [[0, 0], [430, 82], [432, 268], [0, 276]]
    twist_shape_obj = TwistImgs(**twist_params)
    twist_shape_bind = BatchFilesBinding(twist_shape_obj, '图片视角变换')
    window.FileWindow.selected_signal.connect(twist_shape_bind.update_user_select)
    twist_shape_bind.handler_object.result_signal.connect(window.FileWindow.set_operation_result)
    window.FileWindow.add_file_operation(twist_shape_bind.bind_name,
                                         twist_shape_bind.handler_binding)
    window.SettingWindow.settings.changed_signal.connect(twist_shape_bind.update_setting)

    # =============================绑定BiliVideo=============================
    bili_params = window.SettingWindow.settings.get_class_params("BiliVideos")
    bili_videos_obj = BiliVideos(**bili_params)
    bili_videos_bind = BatchFilesBinding(bili_videos_obj, 'B站视频导出')
    window.FileWindow.selected_signal.connect(bili_videos_bind.update_user_select)
    bili_videos_bind.handler_object.result_signal.connect(window.FileWindow.set_operation_result)
    window.FileWindow.add_file_operation(bili_videos_bind.bind_name,
                                         bili_videos_bind.handler_binding)
    window.SettingWindow.settings.changed_signal.connect(bili_videos_bind.update_setting)

    # =============================绑定ECGHandler=============================
    ecg_params = window.SettingWindow.settings.get_class_params("ECGHandler")
    ecg_obj = ECGHandler(**ecg_params)
    ecg_bind = BatchFilesBinding(ecg_obj, 'ECG信号处理')
    window.FileWindow.selected_signal.connect(ecg_bind.update_user_select)
    ecg_bind.handler_object.result_signal.connect(window.FileWindow.set_operation_result)
    window.FileWindow.add_file_operation(ecg_bind.bind_name,
                                         ecg_bind.handler_binding)
    window.SettingWindow.settings.changed_signal.connect(ecg_bind.update_setting)

    # =============================绑定GenSubtitles=============================
    gen_subtitles_params = window.SettingWindow.settings.get_class_params("GenSubtitles")
    gen_subtitles_obj = GenSubtitles(**gen_subtitles_params)
    gen_subtitles_bind = BatchFilesBinding(gen_subtitles_obj, '字幕生成')
    window.FileWindow.selected_signal.connect(gen_subtitles_bind.update_user_select)
    gen_subtitles_bind.handler_object.result_signal.connect(window.FileWindow.set_operation_result)
    window.FileWindow.add_file_operation(gen_subtitles_bind.bind_name,
                                         gen_subtitles_bind.handler_binding)
    window.SettingWindow.settings.changed_signal.connect(gen_subtitles_bind.update_setting)

    # =============================绑定SumSubtitles=============================
    sum_subtitles_params = window.SettingWindow.settings.get_class_params("SumSubtitles")
    sum_subtitles_obj = SumSubtitles(**sum_subtitles_params)
    sum_subtitles_bind = BatchFilesBinding(sum_subtitles_obj, '字幕总结')
    window.FileWindow.selected_signal.connect(sum_subtitles_bind.update_user_select)
    sum_subtitles_bind.handler_object.result_signal.connect(window.FileWindow.set_operation_result)
    window.FileWindow.add_file_operation(sum_subtitles_bind.bind_name,
                                         sum_subtitles_bind.handler_binding)
    window.SettingWindow.settings.changed_signal.connect(sum_subtitles_bind.update_setting)

    # =============================绑定MacPoopScooper=============================
    mac_poop_params = window.SettingWindow.settings.get_class_params("MacPoopScooper")
    mac_poop_obj = MacPoopScooper(**mac_poop_params)
    mac_poop_bind = BatchFilesBinding(mac_poop_obj, 'Mac铲屎官')
    window.FileWindow.selected_signal.connect(mac_poop_bind.update_user_select)
    mac_poop_bind.handler_object.result_signal.connect(window.FileWindow.set_operation_result)
    window.FileWindow.add_file_operation(mac_poop_bind.bind_name,
                                         mac_poop_bind.handler_binding)
    window.SettingWindow.settings.changed_signal.connect(mac_poop_bind.update_setting)

    # =============================app运行=============================
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
