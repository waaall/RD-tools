"""
    ==========================README===========================
    create date:    20240828
    change date:    20240902
    creator:        zhengxu
    function:       设置参数的实时修改与保存

    version:        beta 2.0
    updates:        提升了模块化

    details:
                    1. 「设置项变量名称」要唯一,
                    2. 参数path[-1]的名字要与对应接收类的参数名一致
                    3. 参数path[-2]的名字要与对应接收类的类名一致
                    4. 增加设置**需要在 settings.json 和 AppSettings类中增加 **_Settingmap
"""
import json
import os
import shutil
from pathlib import Path
from PySide6.QtCore import QObject, Signal


# =========================================================
# =======               软件设置参数类              =========
# =========================================================
class AppSettings(QObject):
    changed_signal = Signal(str, str, object)

    def __init__(self):
        super().__init__()
        self._default_settings_json = {}
        """
            1. 定义「设置项变量名称」到设置路径的映射, 附加选项在value第一项(如果有)
            2. **_Settingmap 命名要与 json key 和 value 的 path[0] 一致
            3. 如果是类相关的设置, 要与类名一致和对应的变量名一致，如"DicomToImage", "fps"
        """
        self.General_Settingmap = {
            "language": (["English", "French", "Spanish"], "General", "language"),
            "launch_maximized": ([True, False], "General", "Display", "launch_maximized"),
            "theme": (["Light", "Dark", "Auto"], "General", "Display", "theme"),
        }
        self.Network_Settingmap = {
            "serial_baud_rate": ([800, 1200, 2400, 4800, 9600, 14400, 19200, 38400],
                                 "Network", "Serial", "baud_rate"),
            "serial_data_bits": ([4, 8], "Network", "Serial", "data_bits"),
            "serial_stop_bits": ([1, 2, 4], "Network", "Serial", "stop_bits"),
            "serial_parity": (["None", "Even", "Odd"], "Network", "Serial", "parity"),
            "use_proxy": ("Network", "Internet", "use_proxy"),
            "proxy_address": ("Network", "Internet", "proxy_address"),
            "proxy_port": ("Network", "Internet", "proxy_port")
        }
        self.Batch_Files_Settingmap = {
            "dicom_log_folder_name": ("Batch_Files", "DicomToImage", "log_folder_name"),
            "dicom_fps": ([10, 20, 30], "Batch_Files", "DicomToImage", "fps"),
            "dicom_frame_dpi": ([100, 200, 400, 800], "Batch_Files", "DicomToImage", "frame_dpi"),
            "dicom_out_dir_prefix": ("Batch_Files", "DicomToImage", "out_dir_prefix"),
            "bili_log_folder_name": ("Batch_Files", "BiliVideos", "log_folder_name"),
            "bili_out_dir_prefix": ("Batch_Files", "BiliVideos", "out_dir_prefix"),
            "bili_add_group_title": ([True, False], "Batch_Files", "BiliVideos", "AddGroupTitle"),
            "bili_group_title_max_length": ([5, 10, 15, 20], "Batch_Files", "BiliVideos", "GroupTitleMaxLength"),
            "mergecolor_log_folder_name": ("Batch_Files", "MergeColors", "log_folder_name"),
            "mergecolor_out_dir_prefix": ("Batch_Files", "MergeColors", "out_dir_prefix"),
            "gen_subtitle_log_folder_name": ("Batch_Files", "GenSubtitles", "log_folder_name"),
            "gen_subtitle_model_path": ("Batch_Files", "GenSubtitles", "model_path"),
            "gen_subtitle_parallel": ([True, False], "Batch_Files", "GenSubtitles", "parallel"),
            "ecg_log_folder_name": ("Batch_Files", "ECGHandler", "log_folder_name"),
            "ecg_sampling_rate": ([100, 200, 500, 1000, 2000],
                                  "Batch_Files", "ECGHandler", "sampling_rate"),
            "ecg_parallel": ([True, False],
                             "Batch_Files", "ECGHandler", "parallel"),
            "ecg_out_dir_prefix": ("Batch_Files", "ECGHandler", "out_dir_prefix"),
            "ecg_filter_low_cut": ([0.1, 0.5, 0.67, 0.8, 1.0, 2.0],
                                   "Batch_Files", "ECGHandler", "filter_low_cut"),
            "ecg_filter_high_cut": ([5.0, 15.0, 30.0, 50.0, 100.0],
                                    "Batch_Files", "ECGHandler", "filter_high_cut"),
            "ecg_filter_order": ([1, 2, 4],
                                 "Batch_Files", "ECGHandler", "filter_order"),
            "ecg_drop_raw_zero": ([True, False],
                                  "Batch_Files", "ECGHandler", "drop_raw_zero"),
            "ecg_trim_raw_data": ([True, False], "Batch_Files", "ECGHandler", "trim_raw_data"),
            "ecg_trim_filtered_data": ([True, False],
                                       "Batch_Files", "ECGHandler", "trim_filtered_data"),
            "ecg_trim_percentage": ([5.0, 6.0, 8.0, 10.0, 15.0, 20.0],
                                    "Batch_Files", "ECGHandler", "trim_percentage"),
            "ecg_time_range_short": ([1.0, 2.0, 3.0, 4.0],
                                     "Batch_Files", "ECGHandler", "time_range_short"),
            "ecg_time_range_medium": ([10.0, 15.0, 20.0, 30.0],
                                      "Batch_Files", "ECGHandler", "time_range_medium"),
            "rename_log_folder_name": ("Batch_Files", "FilesRenamer", "log_folder_name"),
            "rename_mode": (
                ["prefix", "all", "body", "between"],
                "Batch_Files",
                "FilesRenamer",
                "mode",
            ),
            "rename_pattern": ("Batch_Files", "FilesRenamer", "pattern"),
            "rename_start_pattern": ("Batch_Files", "FilesRenamer", "start_pattern"),
            "rename_end_pattern": ("Batch_Files", "FilesRenamer", "end_pattern"),
            "rename_replace_with": ("Batch_Files", "FilesRenamer", "replace_with"),
            "rename_include_extension": ([True, False], "Batch_Files", "FilesRenamer", "include_extension"),
            "rename_case_sensitive": ([True, False], "Batch_Files", "FilesRenamer", "case_sensitive"),
            "rename_recursive": ([True, False], "Batch_Files", "FilesRenamer", "recursive"),
            "rename_max_threads": ([1, 2, 4, 8], "Batch_Files", "FilesRenamer", "max_threads"),
        }
        # 在初始化时加载设置
        self.__settings_json = None
        self._load_settings()

    # 加载设置到成员变量
    def _load_settings(self):
        """
        加载设置到成员变量，优先从用户目录下的配置文件加载，
        如果不存在则从项目配置目录加载，并创建用户目录下的配置文件
        """
        # 获取项目根目录
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_settings_file = os.path.join(base_dir, 'configs', 'settings.json')

        # 用户目录下的配置文件路径
        user_home = os.environ.get("HOME", "")
        if not user_home and os.name == 'nt':  # Windows系统
            user_home = os.environ.get("USERPROFILE", "")

        user_config_dir = os.path.join(user_home, "Develop", "RD-tools-configs")
        user_settings_file = os.path.join(user_config_dir, "settings.json")

        # 检查用户配置目录是否存在，不存在则创建
        if not os.path.exists(user_config_dir):
            try:
                os.makedirs(user_config_dir, exist_ok=True)
                print(f"From AppSettings:\n\t创建用户配置目录: {user_config_dir}\n")
            except Exception as e:
                print(f"From AppSettings:\n\t创建用户配置目录失败: {e}\n")
                # 如果创建失败，使用默认配置文件
                self.settings_file = default_settings_file

        # 如果用户配置文件不存在，但目录存在，则拷贝默认配置文件
        if not os.path.exists(user_settings_file) and os.path.exists(user_config_dir):
            try:
                shutil.copy2(default_settings_file, user_settings_file)
                print(f"From AppSettings:\n\t拷贝默认配置文件到用户目录: {user_settings_file}\n")
                self.settings_file = user_settings_file
            except Exception as e:
                print(f"From AppSettings:\n\t拷贝配置文件失败: {e}\n")
                # 如果拷贝失败，使用默认配置文件
                self.settings_file = default_settings_file
        elif os.path.exists(user_settings_file):
            # 用户配置文件存在，使用用户配置文件
            self.settings_file = user_settings_file
            print(f"From AppSettings:\n\t使用用户配置文件: {user_settings_file}\n")
        else:
            # 其他情况，使用默认配置文件
            self.settings_file = default_settings_file
            print(f"From AppSettings:\n\t使用默认配置文件: {default_settings_file}\n")

        # 打开文件加载json数据
        try:
            with open(default_settings_file, 'r', encoding='utf-8') as file:
                self._default_settings_json = json.load(file)
            with open(self.settings_file, 'r') as file:
                self.__settings_json = json.load(file)
        except Exception as e:
            print(f"From AppSettings:\n\t加载配置文件失败: {e}\n")
            # 如果加载失败，尝试加载默认配置文件
            if self.settings_file != default_settings_file:
                print("From AppSettings:\n\t尝试加载默认配置文件\n")
                self.settings_file = default_settings_file
                with open(self.settings_file, 'r', encoding='utf-8') as file:
                    self.__settings_json = json.load(file)
                if not self._default_settings_json:
                    self._default_settings_json = self.__settings_json

        # 提取第一级键作为main_categories
        self.__main_categories = list(self.__settings_json.keys())

        # 将设置的json数据加载到具体的变量中
        for category in self.__main_categories:
            setting_map = self.get_setting_map(category)
            for name, options_path in setting_map.items():
                _, path = self._extract_options_path(options_path)
                value = self.get_value_from_path(path)
                setattr(self, name, value)
                # 发送信号通知设置加载
                self.changed_signal.emit(path[-2], path[-1], value)

    # 根据 category_name 动态获取对应的 Settingmap
    def get_setting_map(self, category_name: str):
        setting_map_name = f"{category_name}_Settingmap"
        return getattr(self, setting_map_name, {})

    # 从 options_path 中提取 path 和 options
    def _extract_options_path(self, options_path: str):
        if isinstance(options_path[0], list):
            options = options_path[0]
            path = options_path[1:]
        else:
            options = None
            path = options_path
        return options, path

    def get_main_categories(self):
        return self.__main_categories

    def get_setting_entries(self, category_name: str, group_name: str | None = None):
        entries = []
        setting_map = self.get_setting_map(category_name)
        for name, options_path in setting_map.items():
            options, path = self._extract_options_path(options_path)
            if group_name is not None and (len(path) < 2 or path[1] != group_name):
                continue

            value = getattr(self, name, self.get_value_from_path(path))
            entries.append({
                'name': name,
                'options': options,
                'path': path,
                'value': value,
            })

        return entries

    def get_setting_groups(self, category_name: str):
        groups = []
        seen = set()
        for entry in self.get_setting_entries(category_name):
            path = entry['path']
            if len(path) < 2:
                continue

            group_name = path[1]
            if group_name in seen:
                continue

            seen.add(group_name)
            groups.append(group_name)

        return groups

    def get_value_from_path(self, path):
        d = self._lookup_path(self.__settings_json, path)
        if d is None:
            d = self._lookup_path(self._default_settings_json, path)

        # 确保返回正确的数据类型
        if d is None:
            return None

        # 处理布尔值
        if isinstance(d, str) and d.lower() in ('true', 'false'):
            return d.lower() == 'true'

        # 处理数字
        if isinstance(d, str):
            # 尝试转换为整数
            try:
                if d.isdigit():
                    return int(d)
            except (ValueError, AttributeError):
                pass

            # 尝试转换为浮点数
            try:
                return float(d)
            except (ValueError, AttributeError):
                pass

        return d

    @staticmethod
    def _lookup_path(source: dict | None, path):
        d = source
        for key in path:
            if not isinstance(d, dict) or key not in d:
                return None
            d = d[key]
        return d

    def _write_settings_file(self):
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as file:
                json.dump(self.__settings_json, file, indent=4)
                return True
        except Exception as e:
            print(f"From AppSettings:\n\t写入配置文件失败: {e}\n")
            return False

    # 根据类名获取参数
    def get_class_params(self, class_name):
        """
        根据类名查找对应的参数，返回可用于初始化该类的参数字典
        Args:
            class_name: 类的名称，如'GenSubtitles'
        Returns:
            dict: 包含类参数的字典，可直接用于类的初始化
        """
        # 确保设置已加载
        if self.__settings_json is None:
            self._load_settings()

        # 存储找到的参数
        params = {}

        # 遍历所有设置映射
        for category in self.__main_categories:
            setting_map = self.get_setting_map(category)
            # 查找与指定类相关的设置
            for name, options_path in setting_map.items():
                _, path = self._extract_options_path(options_path)
                # 检查路径中是否包含类名
                if len(path) >= 2 and path[1] == class_name:
                    # 获取参数名和值
                    param_name = path[-1]
                    param_value = self.get_value_from_path(path)
                    if param_value is None:
                        continue
                    params[param_name] = param_value
        return params

    # 手动加载设置
    def load_settings(self):
        """手动加载设置文件，用于在初始化时不立即加载的情况"""
        if self.__settings_json is None:
            self._load_settings()
        return True

    # 保存设置到文件
    def save_settings(self, name: str, value):
        # 查找设置对应的路径
        path = None
        for category in self.__main_categories:
            setting_map = self.get_setting_map(category)
            options_path = setting_map.get(name)
            if options_path:
                _, path = self._extract_options_path(options_path)
                break
        if path:
            # 获取最后一级之前的dict对象
            d = self.__settings_json
            for key in path[:-1]:
                if key not in d or not isinstance(d[key], dict):
                    d[key] = {}
                d = d[key]
            # 设置最后一级的值
            d[path[-1]] = value
            print(f"From AppSettings:\n\tUpdating setting: {path} = {value}\n")
        else:
            print(f"From AppSettings:\n\tSetting '{name}' not found\n")
            return False

        # 发送信号(类名,参数名和值), 通知设置修改
        self.changed_signal.emit(path[-2], path[-1], value)
        return self._write_settings_file()
