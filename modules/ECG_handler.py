"""
    ===========================README============================
    create date:    20250318
    change date:    20250319
    creator:        zhengxu
    function:       批量处理ECG数据并生成图表
    details:        _data_dir为CSV文件夹,内部的文件夹为子任务

    version:        beta 2.0
    updates:        继承FilesBasic类,实现多线程
"""
# =========================用到的库==========================
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，避免多线程问题
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq

# 获取当前文件所在目录,并加入系统环境变量(临时)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from modules.files_basic import FilesBasic


# =========================================================
# =======                ECG数据处理类              =========
# =========================================================
class ECGHandler(FilesBasic):
    def __init__(self,
                 log_folder_name: str = 'ecg_handler_log',
                 sampling_rate: int = 1000,  # 1kHz (1ms sampling)
                 parallel: bool = False,
                 out_dir_prefix: str = 'ecg-'):
        """
        初始化ECG处理
        Args:
            log_folder_name: 日志文件夹名称
            sampling_rate: 采样率, 默认1000Hz
            parallel: 是否使用并行处理
            out_dir_prefix: 输出文件夹前缀
        """
        super().__init__(log_folder_name=log_folder_name, out_dir_prefix=out_dir_prefix)
        self.sampling_rate = sampling_rate
        self.parallel = parallel
        self.suffixs = ['.csv']  # 设置要处理的文件后缀

    def _data_dir_handler(self, _data_dir: str):
        """处理单个数据文件夹, 支持串行和并行处理"""
        # 检查_data_dir,为空则终止,否则继续执行
        file_list = self._get_filenames_by_suffix(_data_dir)
        if not file_list:
            self.send_message(f"Error: No CSV file in {_data_dir}")
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
        """处理单个CSV文件: 生成时域和频域图表"""
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

        self.send_message(f"处理CSV文件: {file_name}")

        try:
            # 加载数据
            df = pd.read_csv(abs_input_path)
            data = df.iloc[:, 1].values

            # 生成时域图
            self._plot_time_domain(data, abs_outfolder_path)

            # 生成频域图
            self._plot_frequency_domain(data, abs_outfolder_path)

            self.send_message(f"处理完成: {file_name}")

        except Exception as e:
            self.send_message(f"处理文件时出错: {str(e)}")

    def _plot_time_domain(self, data: np.ndarray, output_dir: str):
        """绘制时域信号图"""
        time = np.arange(len(data)) / self.sampling_rate

        # 定义不同时间范围
        time_ranges = [
            (0, 3, "3s"),
            (0, 10, "10s"),
            (0, len(data) / self.sampling_rate, "full")
        ]

        for start_time, end_time, suffix in time_ranges:
            # 创建新的图形对象
            fig = plt.figure(figsize=(12, 6))
            # 计算对应的样本点索引
            start_idx = int(start_time * self.sampling_rate)
            end_idx = int(min(end_time * self.sampling_rate, len(data)))

            plt.plot(time[start_idx:end_idx], data[start_idx:end_idx])
            plt.title(f'ECG Time Domain Signal ({suffix})')
            plt.xlabel('Time (s)')
            plt.ylabel('ADC Value')
            plt.grid(True)

            # 保存图片
            save_path = os.path.join(output_dir, f"time_{suffix}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # 明确关闭图形对象

    def _plot_frequency_domain(self, data: np.ndarray, output_dir: str):
        """绘制频域信号图"""
        # 计算FFT
        n = len(data)
        yf = fft(data)
        xf = fftfreq(n, 1 / self.sampling_rate)

        # 只取正频率部分
        positive_freq = xf[:n // 2]
        amplitude = 2.0 / n * np.abs(yf[:n // 2])

        # 定义不同频率范围
        freq_ranges = [
            (0.01, 1, "0.01-1Hz"),
            (0.5, 5, "0.5-5Hz"),
            (0.5, 100, "0.5-100Hz"),
            (0.5, 500, "0.5-500Hz")
        ]

        for start_freq, end_freq, suffix in freq_ranges:
            # 创建新的图形对象
            fig = plt.figure(figsize=(12, 6))
            # 获取频率范围内的索引
            mask = (positive_freq >= start_freq) & (positive_freq <= end_freq)

            plt.plot(positive_freq[mask], amplitude[mask])
            plt.title(f'ECG Frequency Spectrum ({suffix})')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.grid(True)

            # 保存图片
            save_path = os.path.join(output_dir, f"freq_{suffix}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # 明确关闭图形对象


# =====================main(单独执行时使用)=====================
def main():
    # 获取用户输入的路径
    input_path = input("请复制实验文件夹所在目录的绝对路径(若Python代码在同一目录, 请直接按Enter): \n")

    # 判断用户是否直接按Enter, 设置为当前工作目录
    if not input_path:
        work_folder = os.getcwd()
    elif os.path.isdir(input_path):
        work_folder = input_path

    ecg_handler = ECGHandler()
    ecg_handler.set_work_folder(work_folder)
    possble_dirs = ecg_handler.possble_dirs

    # 给用户显示, 请用户输入index
    number = len(possble_dirs)
    ecg_handler.send_message('\n')
    for i in range(number):
        print(f"{i}: {possble_dirs[i]}")
    user_input = input("\n请选择要处理的序号(用空格分隔多个序号): \n")

    # 解析用户输入
    try:
        indices = user_input.split()
        index_list = [int(index) for index in indices]
    except ValueError:
        ecg_handler.send_message("输入错误, 必须输入数字")

    RESULT = ecg_handler.selected_dirs_handler(index_list)
    if not RESULT:
        ecg_handler.send_message("输入数字不在提供范围, 请重新运行")


# =========================调试用============================
if __name__ == '__main__':
    main()
