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
from scipy.signal import butter, filtfilt  # 添加滤波器相关函数

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
                 sampling_rate: int = 1000,         # 1kHz (1ms sampling)
                 parallel: bool = False,
                 out_dir_prefix: str = 'ecg-',
                 filter_low_cut: float = 0.5,       # 滤波器低频截止
                 filter_high_cut: float = 30.0,     # 滤波器高频截止
                 filter_order: int = 4,             # 滤波器阶数
                 trim_raw_data: bool = False,       # 是否截取原始数据
                 trim_filtered_data: bool = True,   # 是否截取滤波后数据
                 trim_percentage: float = 10.0,     # 前后截取的百分比0-20
                 time_range_short: float = 3.0,     # 短时域图范围(秒)
                 time_range_medium: float = 10.0):  # 中时域图范围(秒)
        """
        初始化ECG处理Args:
            log_folder_name: 日志文件夹名称
            sampling_rate: 采样率, 默认1000Hz
            parallel: 是否使用并行处理
            out_dir_prefix: 输出文件夹前缀
            filter_low_cut: 滤波器低频截止(Hz)
            filter_high_cut: 滤波器高频截止(Hz)
            filter_order: 滤波器阶数
            trim_raw_data: 是否截取原始数据
            trim_filtered_data: 是否截取滤波后数据
            trim_percentage: 前后截取的百分比(%)
            time_range_short: 短时域图范围(秒)
            time_range_medium: 中时域图范围(秒)
        """
        super().__init__(log_folder_name=log_folder_name, out_dir_prefix=out_dir_prefix)
        self.sampling_rate = sampling_rate
        self.parallel = parallel
        self.suffixs = ['.csv']  # 设置要处理的文件后缀

        # 滤波器参数
        self.filter_low_cut = filter_low_cut
        self.filter_high_cut = filter_high_cut
        self.filter_order = filter_order

        # 数据截取参数 - 分别控制原始数据和滤波后数据
        self.trim_raw_data = trim_raw_data
        self.trim_filtered_data = trim_filtered_data
        self.trim_percentage = min(max(0.0, trim_percentage), 30.0)

        # 时域图时间范围
        self.time_range_short = time_range_short
        self.time_range_medium = time_range_medium

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

        # 创建滤波后数据的输出文件夹
        abs_filtering_path = os.path.join(self._work_folder, _data_dir,
                                          self.out_dir_prefix + base_name + "_filtered")
        os.makedirs(abs_filtering_path, exist_ok=True)

        self.send_message(f"Processing CSV file: {file_name}")

        try:
            # 加载数据
            df = pd.read_csv(abs_input_path)
            data = df.iloc[:, 1].values
            time_axis = df.iloc[:, 0].values

            # 如果需要，截取原始数据前后部分
            if self.trim_raw_data and len(data) > 0:
                trim_samples = int(len(data) * (self.trim_percentage / 100.0))
                data = data[trim_samples:-trim_samples]
                time_axis = time_axis[trim_samples:-trim_samples]
                self.send_message(f"Trimmed raw data by {self.trim_percentage}%")

            # 生成时域图
            self._plot_time_domain(data, abs_outfolder_path)

            # 生成频域图
            self._plot_frequency_domain(data, abs_outfolder_path)

            # 应用带通滤波器处理ECG信号
            filtered_data = self._apply_bandpass_filter(data,
                                                        lowcut=self.filter_low_cut,
                                                        highcut=self.filter_high_cut,
                                                        order=self.filter_order)

            # 滤波后的数据时间轴
            filtered_time_axis = time_axis.copy()

            # 如果需要，单独截取滤波后数据
            if self.trim_filtered_data and len(filtered_data) > 0:
                trim_samples = int(len(filtered_data) * (self.trim_percentage / 100.0))
                aligned_raw_data = data[trim_samples:-trim_samples]
                filtered_data = filtered_data[trim_samples:-trim_samples]
                filtered_time_axis = filtered_time_axis[trim_samples:-trim_samples]
                self.send_message(f"Trimmed filtered data by {self.trim_percentage}%")
            else:
                aligned_raw_data = data     # 原始数据和filtered_data裁减后对齐, 用于对比图

            # 保存滤波后的数据为CSV文件
            filtered_df = pd.DataFrame({
                'Time(ms)': filtered_time_axis,
                'ECG_Filtered': filtered_data
            })
            filtered_csv_path = os.path.join(abs_filtering_path, "filtered.csv")
            filtered_df.to_csv(filtered_csv_path, index=False)
            self.send_message(f"Filtered data saved to: {filtered_csv_path}")

            # 生成滤波后的时域图
            self._plot_time_domain(filtered_data, abs_filtering_path)

            # 生成滤波后的频域图，用于验证滤波效果
            self._plot_frequency_domain(filtered_data, abs_filtering_path)

            # 生成原始与滤波后的对比图
            self._plot_comparison(aligned_raw_data, filtered_data, abs_filtering_path)

            self.send_message(f"Processing completed: {file_name}")

        except Exception as e:
            self.send_message(f"Error processing file: {str(e)}")
            import traceback
            self.send_message(traceback.format_exc())

    def _apply_bandpass_filter(self,
                               data: np.ndarray,
                               lowcut: float = None,
                               highcut: float = None,
                               order: int = None) -> np.ndarray:
        """
        应用带通滤波器处理ECG信号
        Args:
            data: 输入的ECG数据
            lowcut: 低截止频率 (Hz)，默认使用类属性
            highcut: 高截止频率 (Hz)，默认使用类属性
            order: 滤波器阶数，默认使用类属性
        Returns:
            滤波后的ECG数据
        """
        # 使用传入参数或默认类属性
        lowcut = lowcut if lowcut is not None else self.filter_low_cut
        highcut = highcut if highcut is not None else self.filter_high_cut
        order = order if order is not None else self.filter_order

        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist

        # 设计巴特沃斯带通滤波器
        b, a = butter(order, [low, high], btype='band')

        # 应用零相位滤波（不引入相位延迟）
        filtered_data = filtfilt(b, a, data)

        self.send_message(f"Applied bandpass filter: {lowcut}-{highcut}Hz, order: {order}")
        return filtered_data

    def _plot_comparison(self,
                         original_data: np.ndarray,
                         filtered_data: np.ndarray,
                         output_dir: str):
        """绘制原始信号与滤波后信号的对比图"""
        time = np.arange(len(original_data)) / self.sampling_rate
        filtered_time = np.arange(len(filtered_data)) / self.sampling_rate

        # 使用可配置的时间范围
        time_ranges = [
            (0, self.time_range_short, f"{self.time_range_short}s"),
            (0, self.time_range_medium, f"{self.time_range_medium}s"),
            (0, min(len(original_data), len(filtered_data)) / self.sampling_rate, "full")
        ]

        for start_time, end_time, suffix in time_ranges:
            # 创建新的图形对象
            fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # 计算对应的样本点索引 - 原始数据
            start_idx_orig = int(start_time * self.sampling_rate)
            end_idx_orig = int(min(end_time * self.sampling_rate, len(original_data)))

            # 计算对应的样本点索引 - 滤波后数据
            start_idx_filt = int(start_time * self.sampling_rate)
            end_idx_filt = int(min(end_time * self.sampling_rate, len(filtered_data)))

            # 原始信号
            ax[0].plot(time[start_idx_orig:end_idx_orig],
                       original_data[start_idx_orig: end_idx_orig], 'b-')
            ax[0].set_title(f'Original ECG Signal{" (Trimmed)" if self.trim_raw_data else ""}')
            ax[0].set_ylabel('ADC Value')
            ax[0].grid(True)

            # 滤波后信号
            ax[1].plot(filtered_time[start_idx_filt:end_idx_filt],
                       filtered_data[start_idx_filt:end_idx_filt], 'r-')
            ax[1].set_title(f'Filtered ECG Signal ({self.filter_low_cut}-{self.filter_high_cut}Hz)'+
                           (f' (Trimmed)' if self.trim_filtered_data else ''))
            ax[1].set_xlabel('Time (s)')
            ax[1].set_ylabel('ADC Value')
            ax[1].grid(True)

            plt.tight_layout()

            # 保存图片
            save_path = os.path.join(output_dir, f"comparison_{suffix}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # 明确关闭图形对象

    def _plot_time_domain(self, data: np.ndarray, output_dir: str, is_trimmed: bool = None):
        """绘制时域信号图"""
        # 根据目录判断是原始数据还是滤波后数据
        if is_trimmed is None:
            is_trimmed = self.trim_raw_data if "filtered" not in output_dir else self.trim_filtered_data

        time = np.arange(len(data)) / self.sampling_rate

        # 使用可配置的时间范围
        time_ranges = [
            (0, self.time_range_short, f"{self.time_range_short}s"),
            (0, self.time_range_medium, f"{self.time_range_medium}s"),
            (0, len(data) / self.sampling_rate, "full")
        ]

        for start_time, end_time, suffix in time_ranges:
            # 创建新的图形对象
            fig = plt.figure(figsize=(12, 6))
            # 计算对应的样本点索引
            start_idx = int(start_time * self.sampling_rate)
            end_idx = int(min(end_time * self.sampling_rate, len(data)))

            plt.plot(time[start_idx:end_idx], data[start_idx:end_idx])
            plt.title(f'ECG Time Domain Signal ({suffix}){" (Trimmed)" if is_trimmed else ""}')
            plt.xlabel('Time (s)')
            plt.ylabel('ADC Value')
            plt.grid(True)

            # 保存图片
            save_path = os.path.join(output_dir, f"time_{suffix}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # 明确关闭图形对象

    def _plot_frequency_domain(self, data: np.ndarray, output_dir: str, is_trimmed: bool = None):
        """绘制频域信号图"""
        # 根据目录判断是原始数据还是滤波后数据
        if is_trimmed is None:
            is_trimmed = self.trim_raw_data if "filtered" not in output_dir else self.trim_filtered_data

        # 计算FFT
        n = len(data)
        yf = fft(data)
        xf = fftfreq(n, 1 / self.sampling_rate)

        # 只取正频率部分
        positive_freq = xf[:n // 2]
        amplitude = 2.0 / n * np.abs(yf[:n // 2])

        # 获取奈奎斯特频率 (采样率的一半)
        nyquist_freq = self.sampling_rate / 2.0
        # 添加采样率最大分析频率
        max_freq = min(nyquist_freq * 0.95, 500)  # 最高不超过500Hz,留5%余量

        # 根据采样率动态确定频率范围, 仅当采样率足够高时添加更高的频率范围
        freq_ranges = [
            (0.01, 1, "0.01-1Hz"),
            (0.5, 5, "0.5-5Hz"),
        ]
        if nyquist_freq < 100:
            freq_ranges.append((0.5, max_freq, f"0.5-{max_freq}Hz"))
        if nyquist_freq >= 100:
            freq_ranges.append((0.5, 100, "0.5-100Hz"))
        if nyquist_freq >= 200:
            freq_ranges.append((0.5, max_freq, f"0.5-{max_freq:.1f}Hz"))

        for start_freq, end_freq, suffix in freq_ranges:
            # 创建新的图形对象
            fig = plt.figure(figsize=(12, 6))
            # 获取频率范围内的索引
            mask = (positive_freq >= start_freq) & (positive_freq <= end_freq)

            if np.any(mask):  # 确保此频率范围有数据点
                plt.plot(positive_freq[mask], amplitude[mask])
                plt.title(f'ECG Frequency Spectrum ({suffix}){" (Trimmed)" if is_trimmed else ""}')
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
