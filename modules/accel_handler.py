"""
    ===========================README============================
    create date:    2024-05-01
    change date:    2024-05-01
    creator:        AI Assistant

    function:       三轴加速度数据处理与分析
                    1. 支持txt和csv格式数据文件读取
                    2. 数据预处理（去除错误行、格式化）
                    3. 基本统计分析和可视化功能

    version:        2.0
    updates:        继承FilesBasic类,实现多线程
"""
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，避免多线程问题
import matplotlib.pyplot as plt
import pandas as pd

# 获取当前文件所在目录,并加入系统环境变量(临时)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from modules.files_basic import FilesBasic


class AccelHandler(FilesBasic):
    """三轴加速度数据处理与分析类"""

    def __init__(self,
                 log_folder_name: str = 'accel_handler_log',
                 parallel: bool = False,
                 out_dir_prefix: str = 'accel-',
                 window_size: int = 5,
                 filter_method: str = 'moving_avg',
                 motion_threshold: float = None):
        """
        初始化加速度数据处理器
        Args:
            log_folder_name: 日志文件夹名称
            parallel: 是否使用并行处理
            out_dir_prefix: 输出文件夹前缀
            window_size: 滤波窗口大小
            filter_method: 滤波方法，'moving_avg'或'median'
            motion_threshold: 活动检测阈值，默认为None（使用均值+1个标准差）
        """
        super().__init__(log_folder_name=log_folder_name, out_dir_prefix=out_dir_prefix)
        self.parallel = parallel
        self.suffixs = ['.txt', '.csv']  # 设置要处理的文件后缀
        self.window_size = window_size
        self.filter_method = filter_method
        self.motion_threshold = motion_threshold

    def _data_dir_handler(self, _data_dir: str):
        """处理单个数据文件夹, 支持串行和并行处理"""
        # 检查_data_dir,为空则终止,否则继续执行
        file_list = self._get_filenames_by_suffix(_data_dir)
        if not file_list:
            self.send_message(f"Error: No data files in {_data_dir}")
            return

        if self.parallel:
            # 多线程处理单个文件
            max_works = min(self.max_threads, os.cpu_count(), len(file_list))
            with ThreadPoolExecutor(max_workers=max_works) as executor:
                for file_name in file_list:
                    abs_input_path = os.path.join(self._work_folder, _data_dir, file_name)
                    executor.submit(self.single_file_handler, _data_dir, abs_input_path)
        else:
            # 串行处理单个文件
            for file_name in file_list:
                abs_input_path = os.path.join(self._work_folder, _data_dir, file_name)
                self.single_file_handler(_data_dir, abs_input_path)

    def single_file_handler(self, _data_dir: str, abs_input_path: str):
        """处理单个数据文件: 生成统计信息和图表"""
        # 检查文件路径格式
        if not os.path.isfile(abs_input_path):
            self.send_message(f"Error: Input file does not exist: {abs_input_path}")
            return

        # 获取文件名（不含扩展名）并创建对应的输出文件夹
        file_name = os.path.basename(abs_input_path)
        base_name = os.path.splitext(file_name)[0]
        abs_outfolder_path = os.path.join(self._work_folder, _data_dir,
                                          self.out_dir_prefix + base_name)
        os.makedirs(abs_outfolder_path, exist_ok=True)

        self.send_message(f"Processing file: {file_name}")

        try:
            # 加载数据
            data = self._load_data(abs_input_path)
            if data is None:
                return

            # 预处理数据
            self._preprocess_data(data)

            # 计算统计信息
            stats = self._calculate_stats(data)

            # 保存统计信息
            stats_file = os.path.join(abs_outfolder_path, "statistics.txt")
            with open(stats_file, 'w') as f:
                f.write(f"数据源: {file_name}\n")
                f.write(f"样本数量: {len(data)}\n\n")
                for axis in ['x', 'y', 'z', 'total']:
                    f.write(f"{axis.upper()}轴统计:\n")
                    for stat_name, value in stats[axis].items():
                        f.write(f"  {stat_name}: {value:.2f}\n")
                    f.write("\n")

            # 生成图表
            self._plot_data(data, abs_outfolder_path)

            # 检测活动周期
            motion_periods = self._detect_motion_periods(data, stats)
            if motion_periods:
                # 保存活动周期信息
                motion_file = os.path.join(abs_outfolder_path, "motion_periods.txt")
                with open(motion_file, 'w') as f:
                    f.write("检测到的活动周期:\n")
                    for start, end in motion_periods:
                        f.write(f"开始: {start}, 结束: {end}, 持续时间: {end-start} 样本\n")

            # 导出处理后的数据
            export_path = os.path.join(abs_outfolder_path, "processed_data.csv")
            data.to_csv(export_path, index=False)

            self.send_message(f"Processing completed: {file_name}")

        except Exception as e:
            self.send_message(f"Error in processing file: {str(e)}")
            self.send_message(traceback.format_exc())
            return

    def _load_data(self, file_path: str) -> pd.DataFrame:
        """加载数据文件"""
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext == '.txt':
                return self._load_txt_data(file_path)
            elif file_ext == '.csv':
                return self._load_csv_data(file_path)
            else:
                self.send_message(f"错误: 不支持的文件格式 {file_ext}, 目前仅支持 .txt 和 .csv")
                return None
        except Exception as e:
            self.send_message(f"加载数据时出错: {e}")
            return None

    def _load_txt_data(self, file_path: str) -> pd.DataFrame:
        """加载txt格式的加速度数据"""
        raw_data = []

        with open(file_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line or not line.startswith('$'):
                    continue

                # 去除开头的$符号
                line = line[1:]

                # 检查是否有错误格式行
                if line.count(',') != 2:
                    continue

                try:
                    # 分割并转换为整数
                    parts = line.split(',')
                    x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
                    raw_data.append([x, y, z])
                except ValueError:
                    continue

        return pd.DataFrame(raw_data, columns=['x', 'y', 'z'])

    def _load_csv_data(self, file_path: str) -> pd.DataFrame:
        """加载csv格式的加速度数据"""
        df = pd.read_csv(file_path)

        # 查找加速度列
        accel_cols = []
        for col in df.columns:
            if 'accelerometer' in col.lower() or 'accel' in col.lower():
                accel_cols.append(col)

        if len(accel_cols) >= 3:
            x_col = [col for col in accel_cols if 'x' in col.lower()][0]
            y_col = [col for col in accel_cols if 'y' in col.lower()][0]
            z_col = [col for col in accel_cols if 'z' in col.lower()][0]

            result_df = df[[x_col, y_col, z_col]].copy()
            result_df.columns = ['x', 'y', 'z']

            if 'timestamp' in df.columns:
                result_df['timestamp'] = df['timestamp']
            return result_df
        else:
            result_df = pd.DataFrame()
            result_df['x'] = df.iloc[:, 1]
            result_df['y'] = df.iloc[:, 2]
            result_df['z'] = df.iloc[:, 3]

            if len(df.columns) > 0 and ('time' in df.columns[0].lower() or 'date' in df.columns[0].lower()):
                result_df['timestamp'] = df.iloc[:, 0]
            return result_df

    def _preprocess_data(self, data: pd.DataFrame) -> None:
        """数据预处理"""
        original_count = len(data)

        # 删除重复行
        data.drop_duplicates(inplace=True)

        # 删除包含NaN的行
        data.dropna(subset=['x', 'y', 'z'], inplace=True)

        # 确保所有值都是数值类型
        for col in ['x', 'y', 'z']:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # 再次删除可能产生的NaN
        data.dropna(subset=['x', 'y', 'z'], inplace=True)

        # 重置索引
        data.reset_index(drop=True, inplace=True)

        # 计算向量幅度
        data['magnitude'] = np.sqrt(
            data['x']**2 + data['y']**2 + data['z']**2
        )

        # 应用滤波
        if self.filter_method == 'moving_avg':
            for col in ['x', 'y', 'z']:
                data[col] = data[col].rolling(window=self.window_size, center=True).mean()
        elif self.filter_method == 'median':
            for col in ['x', 'y', 'z']:
                data[col] = data[col].rolling(window=self.window_size, center=True).median()

        # 填充可能出现的NaN值
        data.fillna(method='bfill', inplace=True)
        data.fillna(method='ffill', inplace=True)

        # 重新计算幅度
        data['magnitude'] = np.sqrt(
            data['x']**2 + data['y']**2 + data['z']**2
        )

        processed_count = len(data)
        if original_count > processed_count:
            self.send_message(f"预处理: 移除了 {original_count - processed_count} 行无效数据")

    def _calculate_stats(self, data: pd.DataFrame) -> dict:
        """计算数据的基本统计信息"""
        stats = {
            'x': {},
            'y': {},
            'z': {},
            'total': {}
        }

        for col in ['x', 'y', 'z', 'magnitude']:
            if col == 'magnitude':
                target = 'total'
            else:
                target = col

            stats[target] = {
                'min': data[col].min(),
                'max': data[col].max(),
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std(),
                'range': data[col].max() - data[col].min()
            }

        return stats

    def _plot_data(self, data: pd.DataFrame, output_dir: str) -> None:
        """绘制三轴加速度数据图表"""
        plt.figure(figsize=(12, 8))

        # 创建三个子图
        plt.subplot(4, 1, 1)
        plt.plot(data.index, data['x'], 'r-', label='X轴')
        plt.title('三轴加速度数据')
        plt.ylabel('X轴')
        plt.grid(True)
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(data.index, data['y'], 'g-', label='Y轴')
        plt.ylabel('Y轴')
        plt.grid(True)
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(data.index, data['z'], 'b-', label='Z轴')
        plt.ylabel('Z轴')
        plt.grid(True)
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(data.index, data['magnitude'], 'k-', label='合成幅度')
        plt.xlabel('样本索引')
        plt.ylabel('幅度')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        # 保存图表
        save_path = os.path.join(output_dir, "acceleration_plot.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _detect_motion_periods(self, data: pd.DataFrame, stats: dict) -> list:
        """检测活动周期"""
        if self.motion_threshold is None:
            threshold = stats['total']['mean'] + stats['total']['std']
        else:
            threshold = self.motion_threshold

        motion_periods = []
        start_idx = None

        for i, mag in enumerate(data['magnitude']):
            if mag > threshold and start_idx is None:
                start_idx = i
            elif mag <= threshold and start_idx is not None:
                motion_periods.append((start_idx, i))
                start_idx = None

        if start_idx is not None:
            motion_periods.append((start_idx, len(data) - 1))

        return motion_periods


# =====================main(单独执行时使用)=====================
def main():
    # 获取用户输入的路径
    input_path = input("请复制实验文件夹所在目录的绝对路径(若Python代码在同一目录, 请直接按Enter): \n")

    # 判断用户是否直接按Enter, 设置为当前工作目录
    if not input_path:
        work_folder = os.getcwd()
    elif os.path.isdir(input_path):
        work_folder = input_path

    accel_handler = AccelHandler(parallel=True)

    accel_handler.set_work_folder(work_folder)
    possble_dirs = accel_handler.possble_dirs

    # 给用户显示, 请用户输入index
    number = len(possble_dirs)
    accel_handler.send_message('\n')
    for i in range(number):
        print(f"{i}: {possble_dirs[i]}")
    user_input = input("\n请选择要处理的序号(用空格分隔多个序号): \n")

    # 解析用户输入
    try:
        indices = user_input.split()
        index_list = [int(index) for index in indices]
    except ValueError:
        accel_handler.send_message("输入错误, 必须输入数字")

    RESULT = accel_handler.selected_dirs_handler(index_list)
    if not RESULT:
        accel_handler.send_message("输入数字不在提供范围, 请重新运行")


# =========================调试用============================
if __name__ == '__main__':
    main()
