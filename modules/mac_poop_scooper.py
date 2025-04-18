"""
    ===========================README============================
    create date:    20241022
    creator:        zhengxu
    function:       批量清理 macOS 系统创建的_PoopPrefix开头隐藏文件

    version:        beta 1.0
    details:        当 macOS 系统在非原生文件系统上创建文件时, 会自动创建以 ._ 开头的隐藏文件
                    这个工具用于清理这些隐藏文件, 如果有同名文件（去掉前缀后）, 则删除隐藏文件
"""
# =========================用到的库==========================
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# 获取当前脚本所在目录的父目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.files_basic import FilesBasic


# =========================================================
# =======           清理 macOS 隐藏文件            =========
# =========================================================
class MacPoopScooper(FilesBasic):
    def __init__(self,
                 log_folder_name: str = 'handle_log',
                 parallel: bool = True):
        super().__init__(
            log_folder_name=log_folder_name,
            parallel=parallel
        )

        # 统计信息
        self.files_found = 0
        self.files_deleted = 0
        # 定义隐藏文件前缀
        self._PoopPrefix = '._'

    # =======================处理单个数据文件夹函数=======================
    def _data_dir_handler(self, _data_dir: str):
        """
        重写的数据文件夹处理方法, 递归遍历文件夹查找并处理开头的文件
        """

        # 获取全路径
        full_path = os.path.join(self._work_folder, _data_dir)

        # 创建要处理的任务列表
        real_poop_files = []

        # 递归遍历所有文件
        for root, _, files in os.walk(full_path):
            # 筛选出开头的文件
            poop_files = [f for f in files if f.startswith(self._PoopPrefix)]

            if poop_files:
                for poop_file in poop_files:
                    # 构建完整文件路径
                    poop_file_path = os.path.join(root, poop_file)
                    # 获取对应的正常文件名（去掉前缀）
                    normal_file = poop_file[len(self._PoopPrefix):]

                    # 只有当对应的正常文件存在时, 才添加到任务列表
                    if os.path.exists(os.path.join(root, normal_file)):
                        real_poop_files.append(poop_file_path)

        self.files_found += len(real_poop_files)

        # 根据 parallel 参数决定是否使用并行处理
        if self.parallel and real_poop_files:
            max_works = min(self.max_threads, os.cpu_count(), len(real_poop_files))
            with ThreadPoolExecutor(max_workers=max_works) as executor:
                # 将每个文件的处理提交给线程池
                futures = [executor.submit(self._delete_file, poop_path)
                           for poop_path in real_poop_files]
                # 等待所有任务完成
                for future in futures:
                    try:
                        future.result()  # 获取任务结果, 如果有异常会在这里抛出
                    except Exception as e:
                        self.send_message(f"处理文件时出错: {str(e)}")
        else:
            # 串行处理每个文件
            for poop_path in real_poop_files:
                try:
                    self._delete_file(poop_path)
                except Exception as e:
                    self.send_message(f"处理文件时出错: {str(e)}")

        # 输出处理结果
        self.send_message(f"文件夹处理完成: {_data_dir}")
        self.send_message(f"累积找到了{self.files_found}个poop文件, 删除了{self.files_deleted}个")


# =====================main(单独执行时使用)=====================
def main():
    # 获取用户输入的路径
    input_path = input("请复制要清理的文件夹绝对路径(若脚本在同一目录, 请直接按Enter):\n")

    # 判断用户是否直接按Enter, 设置为当前工作目录
    if not input_path:
        work_folder = os.getcwd()
    elif os.path.isdir(input_path):
        work_folder = input_path
    else:
        print(f"错误:路径 {input_path} 不存在或不是文件夹")
        return

    cleaner = MacPoopScooper()
    cleaner.set_work_folder(work_folder)
    possble_dirs = cleaner.possble_dirs

    # 给用户显示, 请用户输入index
    number = len(possble_dirs)
    cleaner.send_message('\n')
    for i in range(number):
        print(f"{i}: {possble_dirs[i]}")
    user_input = input("\n请选择要处理的序号(用空格分隔多个序号):\n")

    # 解析用户输入
    try:
        indices = user_input.split()
        index_list = [int(index) for index in indices]
    except ValueError:
        cleaner.send_message("输入错误, 必须输入数字")
        return

    RESULT = cleaner.selected_dirs_handler(index_list)
    if not RESULT:
        cleaner.send_message("输入数字不在提供范围, 请重新运行")


# =========================调试用============================
if __name__ == '__main__':
    main()
