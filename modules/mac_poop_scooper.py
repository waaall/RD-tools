"""
    ===========================README============================
    create date:    20241022
    creator:        zhengxu
    function:       批量清理 macOS 系统创建的隐藏元数据文件

    version:        beta 1.0
    details:        当 macOS 系统在非原生文件系统上创建文件时, 会自动创建以 ._ 开头的隐藏文件
                    这个工具会递归清理 .DS_Store；对于 ._ 文件，只有存在对应原文件时才删除
"""
# =========================用到的库==========================
import os
from concurrent.futures import ThreadPoolExecutor

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
        # 可以安全直接清理的 macOS 元数据文件
        self._JunkFileNames = {'.DS_Store'}

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
            for file_name in files:
                file_path = os.path.join(root, file_name)

                # Finder 目录元数据不依赖对应的正常文件，可以直接清理
                if file_name in self._JunkFileNames:
                    real_poop_files.append(file_path)
                    continue

                if not file_name.startswith(self._PoopPrefix):
                    continue

                # AppleDouble 文件仍采用保守策略：只有对应的正常文件存在时才清理
                normal_file = file_name[len(self._PoopPrefix):]
                if os.path.exists(os.path.join(root, normal_file)):
                    real_poop_files.append(file_path)

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


if __name__ == '__main__':
    from core.task_cli import run_task_cli

    raise SystemExit(run_task_cli('mac-cleaner', operation_cls=MacPoopScooper))
