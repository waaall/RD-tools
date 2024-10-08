##===========================README============================##
""" 
    create date:    20240801 
    change date:    20240913
    creator:        zhengxu
    function:       用于批量文件处理的基类, 提供一些基本功能
    
    version:        beta 3.0
    updates:        实现多线程:
                        「I/O密集型多个独立子任务,很适合用多线程,即使python多线程无法调用多核」
                    模块化更进一步:
                        很多情况下之需要重写single_file_handler方法就OK.
"""

##=========================用到的库==========================
import os
import queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QObject, Signal
##=========================================================
##=======              文件批量处理基类             =========
##=========================================================
class FilesBasic(QObject):
    result_signal = Signal(str)
    def __init__(self,
                 log_folder_name :str = 'handle_log',
                 out_dir_suffix :str = 'Out-', 
                 max_threads :int = 3):
        
        super().__init__()
        # 设置消息队列(初始化顺序不是随意的)
        self.result_queue = queue.Queue()
        self.max_threads = max_threads
        
        # work_folder是dicom文件夹的上一级文件夹，之后要通过set_work_folder改
        self._work_folder = os.getcwd()

        # 自类重定义此量,用于查找指定后缀的文件
        self.suffixs = ['.txt']

        # 之后会根据函数确定
        self.possble_dirs = None
        self._selected_dirs = []
        self._data_dir = None

        # 设置导出文件夹的前缀名 & log文件夹名字
        self.log_folder_name = log_folder_name
        self.out_dir_suffix = out_dir_suffix

    ##======================设置workfolder======================##
    def set_work_folder(self, work_folder:str):
        """设置工作目录, 并确保目录存在"""
        if os.path.exists(work_folder) and os.path.isdir(work_folder):
            self._work_folder = work_folder
            os.chdir(work_folder)
            self.possble_dirs = [f for f in os.listdir(work_folder) if not f.startswith('.')]
            self.send_message(f"工作目录已设置为: {os.getcwd()}")
        else:
            raise ValueError(f"The directory {work_folder} does not exist or is not a directory.")
            # 遍历所有文件夹

    ##===========用户选择workfolder内的selected_dirs处理===========##
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
        
        # 使用 ThreadPoolExecutor 并发处理每个选定的文件夹
        max_works = min(self.max_threads, os.cpu_count(), len(self._selected_dirs))
        with ThreadPoolExecutor(max_workers=max_works) as executor:
            # 将每个文件夹的处理, 提交给线程池 (直接同步调用, 不再使用异步包装)
            futures = [executor.submit(self._data_dir_handler, _data_dir)  
                                    for _data_dir in self._selected_dirs]
            # 等待所有任务完成
            for future in futures:
                try:
                    future.result()  # 获取任务结果，如果有异常会在这里抛出
                except Exception as e:
                    self.send_message(f"处理文件夹时出错: {str(e)}")

        self._save_log()
        self.send_message('SUCCESS! log file saved.')
        return True

    ##=====================处理单个数据文件夹函数======================##
    def _data_dir_handler(self, _data_dir:str):
        # 检查_data_dir,为空则终止,否则创建输出文件夹,继续执行
        file_list = self._get_filenames_by_suffix(_data_dir)
        # file_list这种在每个线程不同的量(且不大),就不用self,避免多线程出问题
        if not file_list:
            self.send_message(f"Error: No file in {_data_dir}")
            return
        outfolder_name = self.out_dir_suffix + _data_dir
        os.makedirs(outfolder_name, exist_ok=True)
        
        # 多线程处理单个文件
        max_works = min(self.max_threads, os.cpu_count(), len(file_list))
        with ThreadPoolExecutor(max_workers=max_works) as executor:
            for file_name in file_list:
                abs_input_path = os.path.join(self._work_folder, _data_dir, file_name)
                abs_outfolder_path = os.path.join(self._work_folder, outfolder_name)
                executor.submit(self.single_file_handler, abs_input_path, abs_outfolder_path)

    ##=================准确的说是每个不可拆分的子任务,子类需重写==================##
    def single_file_handler(self, abs_input_path:str, abs_outfolder_path:str):
        # 检查文件路径格式
        if not self.check_file_path(abs_input_path, abs_outfolder_path):
            self.send_message("Error: failed to check_file_path")
            return
        
        file_name = os.path.basename(abs_input_path)
        abs_out_path = os.path.join(abs_outfolder_path, f"o-{file_name}")
        self.send_message("From FilesBasic: 这是基类, 请在子类中重写该方法.")
        
    ##========================发送log信息========================##
    def send_message(self, message):
        print(f"From FilesBasic: \n\t{message}\n")
        self.result_queue.put(message)
        self.result_signal.emit(message)

    ##=======================保存log信息========================##
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

    ##====================获取符合后缀的所有文件====================##
    def _get_filenames_by_suffix(self, path:str):
        if not os.path.isdir(path):
            self.send_message(f"Error: Folder「{path}」does not exist.")
            return None
        
        # not f.startswith('.')不包括隐藏文件
        return [f for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f)) 
                and not f.startswith('.') and
                any(f.endswith(suffix) for suffix in self.suffixs)]

    ##======================检查文件路径======================##
    def check_file_path(self, input_path:str, output_path:str):
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
        if file_name.startswith('.') or not any(file_name.endswith(suffix) for suffix in self.suffixs):
            self.send_message(f"Error: Invalid file name: {file_name}")
            return False

        # 检查输出路径是否为文件夹，如果不存在则创建
        if not os.path.exists(output_path):
            return False

        # 如果一切正常, 返回 True
        return True

##====================main(单独执行时使用)(示范)====================
def main():
    # 获取用户输入的路径
    input_path = input("请复制实验文件夹所在目录的绝对路径(若Python代码在同一目录, 请直接按Enter): \n")
    
    # 判断用户是否直接按Enter，设置为当前工作目录
    if not input_path:
        work_folder = os.getcwd()
    elif os.path.isdir(input_path):
        work_folder = input_path
    
    files_handler = FilesBasic()
    files_handler.set_work_folder(work_folder)
    possble_dirs = files_handler.possble_dirs
    
    # 给用户显示，请用户输入index
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

##=========================调试用============================
if __name__ == '__main__':
    main()