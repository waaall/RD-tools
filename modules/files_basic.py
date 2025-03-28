"""
    ===========================README============================
    create date:    20240801
    change date:    20241019
    creator:        zhengxu
    function:       用于批量文件处理的基类, 提供一些基本功能

    version:        beta 4.0
    updates:        实现多线程:
                        「I/O密集型多个独立子任务,很适合用多线程,即使python多线程无法调用多核」
                    模块化更进一步:
                        很多情况下之需要重写single_file_handler方法就OK.
                    pair函数提高实用性移到FilesBasic类
"""
# =========================用到的库==========================
import os
import queue
import platform
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QObject, Signal


# =========================================================
# =======              文件批量处理基类             =========
# =========================================================
class FilesBasic(QObject):
    result_signal = Signal(str)

    def __init__(self,
                 log_folder_name: str = 'handle_log',
                 out_dir_prefix: str = 'Out-',
                 max_threads: int = 3,
                 parallel: bool = True):

        super().__init__()
        # 设置消息队列(初始化顺序不是随意的)
        self.result_queue = queue.Queue()
        self.max_threads = max_threads
        self.parallel = parallel  # 保存并行处理标志

        # work_folder是dicom文件夹的上一级文件夹, 之后要通过set_work_folder改
        self._work_folder = os.getcwd()

        # 自类重定义此量,用于查找指定后缀的文件
        self.suffixs = ['.txt']

        # 之后会根据函数确定
        self.possble_dirs = None
        self._selected_dirs = []
        self._data_dir = None

        # 设置导出文件夹的前缀名 & log文件夹名字
        self.log_folder_name = log_folder_name
        self.out_dir_prefix = out_dir_prefix

        # 用户目录下的配置文件路径
        self.user_home_path = os.environ.get("HOME", "")
        if not self.user_home_path and os.name == 'nt':  # Windows系统
            self.user_home_path = os.environ.get("USERPROFILE", "")

    # ======================设置workfolder======================
    def set_work_folder(self, work_folder: str):
        """设置工作目录, 并确保目录存在"""
        if os.path.exists(work_folder) and os.path.isdir(work_folder):
            self._work_folder = work_folder
            os.chdir(work_folder)
            self.possble_dirs = [f for f in os.listdir(work_folder) if not f.startswith('.')]
            self.send_message(f"工作目录已设置为: {os.getcwd()}")
        else:
            raise ValueError(f"The directory {work_folder} does not exist or is not a directory.")
            # 遍历所有文件夹

    # ===========用户选择workfolder内的selected_dirs处理===========
    def selected_dirs_handler(self, indexs_list):
        # 接收的indexs_list可以是indexs也可以是文件夹名(字符串数组)
        if indexs_list[0] in self.possble_dirs:
            self._selected_dirs = indexs_list
        else:
            for index in indexs_list:
                if index in range(len(self.possble_dirs)):
                    self._selected_dirs.append(self.possble_dirs[index])
        if not self._selected_dirs:
            return False

        # 根据self.parallel决定是否使用并行处理
        if self.parallel:
            # 使用 ThreadPoolExecutor 并发处理每个选定的文件夹
            max_works = min(self.max_threads, os.cpu_count(), len(self._selected_dirs))
            with ThreadPoolExecutor(max_workers=max_works) as executor:
                # 将每个文件夹的处理, 提交给线程池 (直接同步调用, 不再使用异步包装)
                futures = [executor.submit(self._data_dir_handler, _data_dir)
                           for _data_dir in self._selected_dirs]
                # 等待所有任务完成
                for future in futures:
                    try:
                        future.result()  # 获取任务结果, 如果有异常会在这里抛出
                    except Exception as e:
                        self.send_message(f"处理文件夹时出错: {str(e)}")
        else:
            # 串行处理每个文件夹
            for _data_dir in self._selected_dirs:
                try:
                    self._data_dir_handler(_data_dir)
                except Exception as e:
                    self.send_message(f"处理文件夹时出错: {str(e)}")

        self._save_log()
        self.send_message('SUCCESS! log file saved.')
        return True

    # =====================处理单个数据文件夹函数======================
    def _data_dir_handler(self, _data_dir: str):
        # 检查_data_dir,为空则终止,否则创建输出文件夹,继续执行
        file_list = self._get_filenames_by_suffix(_data_dir)
        # file_list这种在每个线程不同的量(且不大),就不用self,避免多线程出问题
        if not file_list:
            self.send_message(f"Error: No file in {_data_dir}")
            return
        outfolder_name = self.out_dir_prefix + _data_dir
        os.makedirs(outfolder_name, exist_ok=True)

        # 根据self.parallel决定是否使用并行处理
        if self.parallel:
            # 多线程处理单个文件
            max_works = min(self.max_threads, os.cpu_count(), len(file_list))
            with ThreadPoolExecutor(max_workers=max_works) as executor:
                for file_name in file_list:
                    abs_input_path = os.path.join(self._work_folder, _data_dir, file_name)
                    abs_outfolder_path = os.path.join(self._work_folder, outfolder_name)
                    executor.submit(self.single_file_handler, abs_input_path, abs_outfolder_path)
        else:
            # 串行处理单个文件
            for file_name in file_list:
                abs_input_path = os.path.join(self._work_folder, _data_dir, file_name)
                abs_outfolder_path = os.path.join(self._work_folder, outfolder_name)
                self.single_file_handler(abs_input_path, abs_outfolder_path)

    # =================准确的说是每个不可拆分的子任务,子类需重写==================
    def single_file_handler(self, abs_input_path: str, abs_outfolder_path: str):
        # 检查文件路径格式
        if not self.check_file_path(abs_input_path, abs_outfolder_path):
            self.send_message("Error: failed to check_file_path")
            return

        # file_name = os.path.basename(abs_input_path)
        # abs_out_path = os.path.join(abs_outfolder_path, f"o-{file_name}")
        self.send_message("From FilesBasic: 这是基类, 请在子类中重写该方法.")

    # ========================发送log信息========================
    def send_message(self, message):
        print(f"From FilesBasic: \n\t{message}\n")
        self.result_queue.put(message)
        self.result_signal.emit(message)

    # ========================保存文件==========================
    def _save_to_file(self, content: str, abs_file_path: str, suffix: str = '.md') -> Optional[str]:
        """保存内容到文件"""
        try:
            output_path = Path(abs_file_path).with_suffix(suffix)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.send_message(f"已保存文件: {output_path}")
            return str(output_path)
        except Exception as e:
            self.send_message(f"Error: 保存文件失败: {e}")
            return None

    # =======================保存log信息========================
    def _save_log(self):
        os.makedirs(self.log_folder_name, exist_ok=True)
        # 获取当前时间并格式化为文件名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{current_time}.log"
        log_file_path = os.path.join(self.log_folder_name, log_filename)
        # 打开文件并写入队列中的内容
        with open(log_file_path, 'w') as log_file:
            while not self.result_queue.empty():
                log_entry = self.result_queue.get()
                log_file.write(f"{log_entry}\n")

    # ====================获取符合后缀的所有文件====================
    def _get_filenames_by_suffix(self, path: str):
        if not os.path.isdir(path):
            self.send_message(f"Error: Folder「{path}」does not exist.")
            return None

        # not f.startswith('.')不包括隐藏文件
        return [f for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
                and not f.startswith('.') and
                any(f.endswith(suffix) for suffix in self.suffixs)]

    # ======================检查文件路径======================
    def check_file_path(self, input_path: str, output_path: str):
        """
        符合 return True
        不符 return False
        """
        # 检查输入路径是否为文件
        if not os.path.isfile(input_path):
            self.send_message(f"Error: Input file does not exist: {input_path}")
            return False

        # 检查文件后缀是否合法
        file_name = os.path.basename(input_path)
        if file_name.startswith('.') or not any(file_name.endswith(suffix)
                                                for suffix in self.suffixs):
            self.send_message(f"Error: Invalid file name: {file_name}")
            return False

        # 检查输出路径是否为文件夹, 如果不存在则创建
        if not os.path.exists(output_path):
            return False

        # 如果一切正常, 返回 True
        return True

    # ==========================配对文件==========================
    def pair_files(self, pairs_label,
                   file_names: str,
                   pre_or_suffix: bool = True,
                   split_symbl: str = '-'):
        """
            param: pairs_label - 需要配对的一对字符串(除去相同部分)
            param: pre_or_suffix - 前缀or后缀, True表示前缀;

            return: pairs - 配对好的二维数组
        """
        max_pairs_lenth = len(pairs_label)
        if max_pairs_lenth < 2:
            self.send_messag("Error: pairs_label length is too short")
            return []

        if file_names is None:
            return []

        # 对文件名排序, 这样配对能将O(n^2)优化到O(nlogn)
        file_names.sort()
        pairs = []
        i = 0
        while i < len(file_names):
            # 分割文件名
            file_name = file_names[i]
            current_label, base_name = self._extract_label_and_base(file_name,
                                                                    pre_or_suffix,
                                                                    split_symbl)

            # 检查文件是否符合命名规则
            if current_label not in pairs_label:
                i += 1
                self.send_message(f"Warning: wrong naming convention「{file_name}」skipped ")
                continue

            # # 直接到核心, 配合下面j = i - max_pairs_lenth, 实现确定的一个找多种可能的其他
            if current_label != pairs_label[0]:
                i += 1
                continue

            pair = [file_name]

            # 子循环(在排序后的位置上寻找配对的图像)
            expected_label = [c for c in pairs_label if c != current_label]
            j = i - max_pairs_lenth     # 有可能在前面
            while j < len(file_names) and expected_label:
                co_label, co_base_name = self._extract_label_and_base(file_names[j],
                                                                      pre_or_suffix,
                                                                      split_symbl)

                # 检查下一个文件的前缀是否符合期望的颜色, 并匹配 base_name
                if co_label in expected_label and co_base_name == base_name:
                    pair.append(file_names[j])
                    expected_label.remove(co_label)  # 匹配到一个颜色后, 从期望集合中移除
                j += 1

            # 检查配对结果, 并根据情况输出提示信息
            if len(pair) == 1:
                # 没有找到任何配对文件
                self.send_message(f"Warning: Unmatched with {base_name}")
            else:
                pairs.append(pair)

            i += 1
        return pairs

    # ======================提取文件名的标签和基础名称=====================
    def _extract_label_and_base(self, file_name: str, pre_or_suffix: bool, split_symbl: str):
        """
        提取文件名的标签和基础名称（去掉扩展名的部分）。
        Args:
            file_name (str): 文件名字符串, 包含要解析的标签和文件名。
            pre_or_suffix (bool): 如果为 True, 提取文件名前缀作为标签；如果为 False, 提取文件名后缀作为标签。
            split_symbl (str): 用于分割文件名的分隔符, 决定从哪个符号处分割。
        Returns:
            Tuple[str, str]:
                - current_label (str): 提取的标签（前缀或后缀, 取决于 pre_or_suffix 的值）。
                - base_name (str): 文件名去掉扩展名后的基础名称。
        """

        if pre_or_suffix:
            # 提取前缀的情况
            parts = file_name.split(split_symbl, 1)
            current_label = parts[0]  # 取分隔符前面的部分作为标签
            base_name, _, _ = parts[1].rpartition('.')  # 提取文件名去掉扩展名
        else:
            # 提取后缀的情况
            parts = file_name.rsplit(split_symbl, 1)  # 使用 rsplit 从右边分割
            current_label = parts[-1]  # 取最后一个部分作为标签
            base_name = parts[0]  # 提取去掉扩展名的部分

        return current_label, base_name

    # ======================获取系统中可用的中文字体=====================
    def _get_system_cjk_font(self):
        """
        获取系统中可用的中文字体
        适用于Windows、MacOS和Linux系统
        Returns:
            str: 找到的中文字体名称, 如果未找到则返回None
        """
        # 记录操作系统信息
        os_type = platform.system()
        self.send_message(f"当前操作系统: {os_type}")

        # Windows系统
        if os_type == "Windows":
            # Windows系统常见中文字体
            windows_fonts = ["SimSun", "SimHei", "Microsoft YaHei", "FangSong",
                             "KaiTi", "NSimSun", "DengXian", "YouYuan"]
            # 检查Windows字体文件夹
            try:
                font_dir = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "Fonts")
                if os.path.exists(font_dir):
                    files = os.listdir(font_dir)
                    for font in windows_fonts:
                        # 检查相关字体文件是否存在
                        font_file = f"{font}.ttf"
                        if font_file in files or f"{font}.ttc" in files:
                            self.send_message(f"找到Windows中文字体: {font}")
                            return font
                        # 检查文件名是否包含字体名称（不区分大小写）
                        for file in files:
                            if file.lower().endswith(".ttf") or file.lower().endswith(".ttc"):
                                if font.lower() in file.lower():
                                    self.send_message(f"找到Windows中文字体: {font}")
                                    return font
            except Exception as e:
                self.send_message(f"检查Windows字体时出错: {e}")
        # macOS系统
        elif os_type == "Darwin":
            # macOS常见中文字体
            mac_fonts = ["PingFang SC", "Heiti SC", "Hiragino Sans GB", "Songti SC",
                         "Kaiti SC", "STHeiti", "STFangsong", "STSong", "STKaiti"]
            # 尝试使用macOS的字体列表命令
            try:
                # 尝试检查系统字体目录
                font_dirs = ["/System/Library/Fonts",
                             "/Library/Fonts",
                             os.path.expanduser("~/Library/Fonts")]
                for font_dir in font_dirs:
                    if os.path.exists(font_dir):
                        for font in os.listdir(font_dir):
                            for mac_font in mac_fonts:
                                if (mac_font.lower().replace(" ", "").replace("-", "") in
                                    font.lower().replace(" ", "").replace("-", "")):
                                    self.send_message(f"找到macOS中文字体: {mac_font}")
                                    return mac_font
                # 如果直接检查文件失败, 尝试使用命令行工具
                try:
                    result = subprocess.run(
                        ["system_profiler", "SPFontsDataType"],
                        capture_output=True,
                        text=True,
                        timeout=3
                    )
                    output = result.stdout
                    for font in mac_fonts:
                        if font in output:
                            self.send_message(f"找到macOS中文字体: {font}")
                            return font
                except Exception as e:
                    self.send_message(f"检查macOS字体列表时出错: {e}")
            except Exception as e:
                self.send_message(f"检查macOS字体时出错: {e}")
        # Linux系统
        elif os_type == "Linux":
            # Linux常见中文字体
            linux_fonts = [
                "Noto Sans CJK SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", 
                "Droid Sans Fallback", "Source Han Sans CN", "Source Han Serif CN"
            ]

            # 尝试使用fc-list命令
            try:
                result = subprocess.run(
                    ["fc-list", ":lang=zh"],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                output = result.stdout
                # 先检查已知的字体
                for font in linux_fonts:
                    if font.lower() in output.lower():
                        self.send_message(f"找到Linux中文字体: {font}")
                        return font

                # 提取第一个中文字体（如果有）
                font_matches = re.findall(r":(.*?):", output)
                if font_matches:
                    font = font_matches[0].strip()
                    self.send_message(f"找到Linux中文字体: {font}")
                    return font
                # 检查常见字体目录
                font_dirs = [
                    "/usr/share/fonts",
                    "/usr/local/share/fonts",
                    os.path.expanduser("~/.fonts")
                ]
                for font_dir in font_dirs:
                    if os.path.exists(font_dir):
                        for root, _, files in os.walk(font_dir):
                            for file in files:
                                if file.endswith(".ttf") or file.endswith(".ttc") or file.endswith(".otf"):
                                    for linux_font in linux_fonts:
                                        if (linux_font.lower().replace(" ", "").replace("-", "") in 
                                            file.lower().replace(" ", "").replace("-", "")):
                                            self.send_message(f"找到Linux中文字体文件: {linux_font}")
                                            return linux_font
            except Exception as e:
                self.send_message(f"检查Linux字体时出错: {e}")
        # 返回系统默认字体
        self.send_message("未找到系统中文字体, 将使用系统默认字体")
        return "sans"


# ====================main(单独执行时使用)(示范)====================
def main():
    # 获取用户输入的路径
    input_path = input("请复制实验文件夹所在目录的绝对路径(若Python代码在同一目录, 请直接按Enter): \n")

    # 判断用户是否直接按Enter, 设置为当前工作目录
    if not input_path:
        work_folder = os.getcwd()
    elif os.path.isdir(input_path):
        work_folder = input_path

    files_handler = FilesBasic()
    files_handler.set_work_folder(work_folder)
    possble_dirs = files_handler.possble_dirs

    # 给用户显示, 请用户输入index
    number = len(possble_dirs)
    files_handler.send_message('\n')
    for i in range(number):
        print(f"{i}: {possble_dirs[i]}")
    user_input = input("\n请选择要处理的序号(用空格分隔多个序号): \n")

    # 解析用户输入
    try:
        indices = user_input.split()
        index_list = [int(index) for index in indices]
    except ValueError:
        files_handler.send_message("输入错误, 必须输入数字")

    RESULT = files_handler.selected_dirs_handler(index_list)
    if not RESULT:
        files_handler.send_message("输入数字不在提供范围, 请重新运行")


# =========================调试用============================
if __name__ == '__main__':
    main()
