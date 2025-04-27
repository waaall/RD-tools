"""
    ===========================README============================
    create date:    20250318
    change date:    20250319
    creator:        zhengxu
    function:       批量处理ECG数据并生成图表
    details:        _data_dir为CSV文件夹,内部的文件夹为子任务

    version:        beta 3.0
    updates:        继承FilesBasic类,实现多线程
                    实现数字滤波
"""
# =========================用到的库==========================
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，避免多线程问题
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

# 获取当前文件所在目录,并加入系统环境变量(临时)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from modules.files_basic import FilesBasic


# =========================================================
# =======                ECG数据处理类             =========
# =========================================================
class ECGHandler(FilesBasic):
    def __init__(self,
                 log_folder_name: str = 'ecg_handler_log',
                 sampling_rate: int = 1000,         # 1kHz(1ms)需要大于100Hz
                 parallel: bool = False,
                 out_dir_prefix: str = 'ecg-',
                 filter_low_cut: float = 0.5,       # 滤波器低频截止
                 filter_high_cut: float = 30.0,     # 滤波器高频截止
                 filter_order: int = 2,             # 滤波器阶数
                 drop_raw_zero: bool = False,       # 是否删除原始数据中的零值
                 trim_raw_data: bool = False,       # 是否截取原始数据
                 trim_filtered_data: bool = True,   # 是否截取滤波后数据
                 trim_percentage: float = 10.0,     # 前后截取的百分比0-20
                 time_range_short: float = 3.0,     # 短时域图范围(秒)
                 time_range_medium: float = 10.0    # 中时域图范围(秒)
                 ):
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
            drop_raw_zero: 是否删除原始数据中的零值
        """
        super().__init__(log_folder_name=log_folder_name, out_dir_prefix=out_dir_prefix)
        # NeuroKit2库并行有问题
        if parallel:
            parallel = False
            self.send_message("Warning: NeuroKit2库并行有问题, 已自动设置为False")

        self.parallel = parallel
        self.suffixs = ['.csv']  # 设置要处理的文件后缀
        try:
            self.__sampling_rate = int(sampling_rate)
        except ValueError:
            print("Error: sampling_rate参数类型错误，使用默认值1000")
            self.__sampling_rate = 1000

        # 计算与采样率相关的频率参数
        self.__nyquist_freq = self.__sampling_rate / 2.0
        self.__max_freq = min(self.__nyquist_freq * 0.95, 500)  # 最高不超过500Hz,留5%余量

        # 验证并设置滤波参数
        self.__validate_filter_params(filter_low_cut, filter_high_cut, filter_order)

        # 数据截取参数 - 分别控制原始数据和滤波后数据
        self.trim_raw_data = trim_raw_data
        self.trim_filtered_data = trim_filtered_data
        self.trim_percentage = min(max(0.0, trim_percentage), 30.0)

        # 时域图时间范围
        self.time_range_short = time_range_short
        self.time_range_medium = time_range_medium

        # 是否删除原始数据中的零值
        self.drop_raw_zero = drop_raw_zero

    def __validate_filter_params(self, filter_low_cut, filter_high_cut, filter_order):
        """
        验证并设置滤波器参数，确保参数在有效范围内
        Args:
            filter_low_cut: 低频截止
            filter_high_cut: 高频截止
            filter_order: 滤波器阶数
        """
        # 验证低频截止
        try:
            low_cut = float(filter_low_cut)
            # 确保低频截止值在0.1-1.0之间
            if low_cut < 0.1:
                print(f"Warning: filter_low_cut值过小: {low_cut}, 已调整为0.1")
                self.__filter_low_cut = 0.1
            elif low_cut > 1.0:
                self.__filter_low_cut = 1.0
                self.send_message(f"Warning: filter_low_cut值过大: {low_cut}, 已调整为1.0")
            else:
                self.__filter_low_cut = low_cut
        except (ValueError, TypeError):
            print(f"Error: filter_low_cut无法转换为浮点数: {filter_low_cut}, 使用默认值0.5")
            self.__filter_low_cut = 0.5

        # 验证高频截止
        try:
            high_cut = float(filter_high_cut)
            # 确保高频截止不超过奈奎斯特频率
            if high_cut > self.__nyquist_freq:
                print(f"Warning: filter_high_cut值过大: {high_cut}, 已调整为{self.__nyquist_freq * 0.9}")
                self.__filter_high_cut = self.__nyquist_freq * 0.9
            else:
                self.__filter_high_cut = high_cut
        except (ValueError, TypeError):
            print(f"Error: filter_high_cut无法转换为浮点数: {filter_high_cut}, 使用默认值30.0")
            self.__filter_high_cut = 30.0

        # 验证滤波器阶数
        try:
            order = int(filter_order)
            valid_orders = [1, 2, 4]
            if order not in valid_orders:
                # 计算与有效选项的差值，选择差值最小的选项
                closest_order = min(valid_orders, key=lambda x: abs(x - order))
                self.__filter_order = closest_order
                self.send_message(f"Warning: filter_order:{order}不在[1,2,4]中，已修正为{closest_order}")
            else:
                self.__filter_order = order
        except ValueError:
            self.__filter_order = 2
            self.send_message("Warning: filter_order参数类型错误，使用默认值2")

    @property
    def nyquist_freq(self):
        """获取奈奎斯特频率，确保返回浮点数类型"""
        try:
            return float(self.__nyquist_freq)
        except (TypeError, ValueError):
            self.send_message("Warning: 奈奎斯特频率类型错误，重新计算")
            return float(self.sampling_rate) / 2.0

    @property
    def max_freq(self):
        """获取最大频率，确保返回浮点数类型"""
        try:
            return float(self.__max_freq)
        except (TypeError, ValueError):
            self.send_message("Warning: 最大频率类型错误，重新计算")
            nyquist = float(self.sampling_rate) / 2.0
            return min(nyquist * 0.95, 500.0)

    @property
    def filter_order(self):
        """获取滤波器阶数，确保返回整数类型"""
        try:
            return int(self.__filter_order)
        except (TypeError, ValueError):
            self.send_message("Warning: 滤波器阶数类型错误，返回默认值4")
            return 4

    @property
    def sampling_rate(self):
        """获取采样率，确保返回整数类型"""
        try:
            return int(self.__sampling_rate)
        except (TypeError, ValueError):
            self.send_message("Warning: 采样率类型错误，返回默认值1000")
            return 1000

    @property
    def filter_low_cut(self):
        """获取低频截止，确保返回浮点数类型"""
        try:
            return float(self.__filter_low_cut)
        except (TypeError, ValueError):
            self.send_message("Warning: 低频截止类型错误，返回默认值0.5")
            return 0.5

    @property
    def filter_high_cut(self):
        """获取高频截止，确保返回浮点数类型"""
        try:
            return float(self.__filter_high_cut)
        except (TypeError, ValueError):
            self.send_message("Warning: 高频截止类型错误，返回默认值30.0")
            return 30.0

    def calculate_min_frequency(self, data_length: int) -> float:
        """
        计算数据的最低频率
        Args:
            data_length: 数据长度
        Returns:
            最低频率(Hz)
        """
        if data_length <= 0:
            return 0.1  # 防止除零错误，返回默认最小值

        # 计算数据时长(秒)
        data_duration = data_length / self.sampling_rate
        # 最低频率是时长一半的倒数
        min_freq = 1.0 / (data_duration / 2.0)
        return min_freq

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
                                          self.out_dir_prefix + base_name + "_raw")
        os.makedirs(abs_outfolder_path, exist_ok=True)

        # 创建滤波后数据的输出文件夹
        abs_filtering_path = os.path.join(self._work_folder, _data_dir,
                                          self.out_dir_prefix + base_name + "_filtered")
        os.makedirs(abs_filtering_path, exist_ok=True)

        self.send_message(f"Processing CSV file: {file_name}")

        # 第一部分：数据加载和预处理
        try:
            # 加载数据
            df = pd.read_csv(abs_input_path)
            data = df.iloc[:, 1].values

            # 如果需要，删除数据中的零值
            if self.drop_raw_zero and len(data) > 0:
                # 找出非零值的索引
                non_zero_indices = np.where(data != 0)[0]
                if len(non_zero_indices) < len(data):
                    # 记录原始数据长度用于消息显示
                    original_length = len(data)
                    data = data[non_zero_indices]
                    self.send_message(f"删除了{original_length - len(data)}个零值 in {abs_input_path}")

            # 如果需要，截取原始数据前后部分
            if self.trim_raw_data and len(data) > 0:
                trim_samples = int(len(data) * (self.trim_percentage / 100.0))
                data = data[trim_samples:-trim_samples]
                self.send_message(f"Trimmed raw data by {self.trim_percentage}%")

            # 计算最低频率
            min_freq = self.calculate_min_frequency(len(data))
        except Exception as e:
            self.send_message(f"Error in data preprocessing: {str(e)}")
            self.send_message(traceback.format_exc())
            return

        # 第二部分：原始数据绘图
        try:
            # 生成时域图
            self._plot_time_domain(data, abs_outfolder_path)
            # 生成频域图
            self._plot_frequency_domain(data, abs_outfolder_path,
                                        min_freq=min_freq, is_trimmed=self.trim_raw_data)
        except Exception as e:
            self.send_message(f"Error in plotting original data: {str(e)}")
            self.send_message(traceback.format_exc())
            return

        # 第三部分：滤波和滤波后数据处理
        try:
            # 应用带通滤波器处理ECG信号
            filtered_data = self._apply_bandpass_filter(data, min_freq)
            if filtered_data is None:
                self.send_message("Error: 滤波失败, 不进行后续处理")
                return

            # 如果需要，单独截取滤波后数据
            if self.trim_filtered_data and len(filtered_data) > 0:
                trim_samples = int(len(filtered_data) * (self.trim_percentage / 100.0))
                aligned_raw_data = data[trim_samples:-trim_samples]
                filtered_data = filtered_data[trim_samples:-trim_samples]
                self.send_message(f"Trimmed filtered data by {self.trim_percentage}%")
            else:
                aligned_raw_data = data     # 原始数据和filtered_data裁减后对齐, 用于对比图

            # 生成滤波后的时间轴（在需要的时候生成一次就好）
            filtered_time_axis = np.arange(len(filtered_data)) / self.__sampling_rate

            # 保存滤波后的数据为CSV文件
            filtered_df = pd.DataFrame({
                'Time(ms)': filtered_time_axis,
                'ECG_Filtered': filtered_data
            })
            filtered_csv_path = os.path.join(abs_filtering_path, "filtered.csv")
            filtered_df.to_csv(filtered_csv_path, index=False)
            self.send_message(f"Filtered data saved to: {filtered_csv_path}")
        except Exception as e:
            self.send_message(f"Error in filtering and saving filtered data: {str(e)}")
            self.send_message(traceback.format_exc())
            return

        # 第四部分：滤波后数据绘图
        try:
            # 生成滤波后的时域图
            self._plot_time_domain(filtered_data, abs_filtering_path)
            # 生成滤波后的频域图，用于验证滤波效果
            self._plot_frequency_domain(filtered_data, abs_filtering_path, min_freq=min_freq)
            # 生成原始与滤波后的对比图
            self._plot_comparison(aligned_raw_data, filtered_data, abs_filtering_path)
        except Exception as e:
            self.send_message(f"Error in plotting filtered data: {str(e)}")
            self.send_message(traceback.format_exc())
            return

        # 第五部分：高级ECG分析
        adv_process_path = os.path.join(self._work_folder, _data_dir,
                                        self.out_dir_prefix + base_name + "_advanced")
        os.makedirs(adv_process_path, exist_ok=True)
        # 检查数据的最大和最小值，如果最小值的绝对值大于最大值，则心电信号正负取反
        pic_suffix = ''
        if np.abs(np.min(filtered_data)) > np.max(filtered_data):
            filtered_data = -filtered_data
            pic_suffix = '_reverted'
            self.send_message(f"心电信号为负, 取反后标记为'reverted':{base_name}")
        try:
            self._advanced_process(filtered_data, adv_process_path, suffix=pic_suffix)

            # 如果时长大于10秒，则裁剪前10秒数据进行处理
            data_duration = len(filtered_data) / self.sampling_rate
            if data_duration > 10.0:
                # 计算10秒对应的数据点数
                samples_10s = int(10.0 * self.sampling_rate)
                # 裁剪前10秒数据
                trimmed_data = filtered_data[:samples_10s]
                self.send_message(f"数据时长为{data_duration:.2f}秒，裁剪前10秒数据进行额外分析")
                # 额外处理裁剪后的数据
                self._advanced_process(trimmed_data, adv_process_path, suffix=pic_suffix + '_10s')
        except Exception as e:
            self.send_message(f"Error in advanced ECG analysis: {str(e)}")
            self.send_message(traceback.format_exc())
            return

        self.send_message(f"Processing completed: {file_name}")

    def _apply_bandpass_filter(self, data: np.ndarray, min_freq: float = None) -> np.ndarray:
        """
        应用带通滤波器处理ECG信号
        Args:
            data: 输入的ECG数据
            min_freq: 最低频率，如果为None则使用默认值
        Returns:
            滤波后的ECG数据
        """
        # 确保有最低频率
        if min_freq is None:
            min_freq = self.calculate_min_frequency(len(data))

        # 限制最低频率，确保不小于0.1Hz
        min_freq = max(min_freq * 2, 0.1)

        # 确定低频截止值（使用计算得到的最低频率和设置的低频截止中的较大值）
        lowcut = max(min_freq, self.filter_low_cut)

        # 如果使用了数据计算的最低频率而不是实例设置的低频截止值，发送消息说明
        if min_freq > self.filter_low_cut:
            self.send_message(f"Warning：由于数据时长过短{min_freq:.3f}Hz替代设置的低频截止{self.filter_low_cut}Hz")

        highcut = self.filter_high_cut
        order = self.filter_order

        # 检查低频截止是否大于高频截止
        if lowcut >= highcut:
            self.send_message(f"Error: 低频截止({lowcut}Hz)>=高频截止({highcut}Hz)，无法带通滤波")
            return None

        # 设计巴特沃斯带通滤波器
        try:
            nyquist = self.nyquist_freq
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
        except Exception as e:
            self.send_message(f"Error: 滤波器设计失败: {e}")
            return None

        # 应用零相位滤波（不引入相位延迟）
        try:
            filtered_data = filtfilt(b, a, data)
            self.send_message(f"Applied bandpass filter: {lowcut}-{highcut}Hz, order: {order}")
            return filtered_data
        except Exception as e:
            self.send_message(f"Error: 滤波失败: {e}，返回原始数据")
            return None

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
            ax[1].set_title(f"Filtered ECG Signal ({self.filter_low_cut:.3f}-{self.filter_high_cut:.3f}Hz)" +
                            (" (Trimmed)" if self.trim_filtered_data else ''))
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

    def _plot_frequency_domain(self,
                               data: np.ndarray,
                               output_dir: str,
                               min_freq: float,
                               is_trimmed: bool = None):
        """
        绘制频域信号图
        Args:
            data: 输入的ECG数据
            output_dir: 输出目录
            is_trimmed: 是否为裁剪后的数据
            min_freq: 最低频率，如果为None则计算
        """
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

        # 使用类属性获取奈奎斯特频率和最大频率
        nyquist_freq = self.nyquist_freq
        max_freq = self.max_freq
        # 检查最低频率是否大于最大频率，如果是则无法绘制频谱图
        if min_freq >= max_freq:
            self.send_message(f"Warning: 最低频率({min_freq:.3f}Hz)>最大频率({max_freq:.1f}Hz)，无法绘制频谱图")
            return

        # 根据采样率和计算的最低频率动态确定频率范围
        freq_ranges = []
        if min_freq < 1:
            freq_ranges.append((min_freq, 1, f"{min_freq:.3f}-1Hz"))
        if min_freq < 5:
            freq_ranges.append((min_freq, 5, f"{min_freq:.3f}-5Hz"))
        if min_freq < 30:
            freq_ranges.append((min_freq, 30, f"{min_freq:.3f}-30Hz"))
        if nyquist_freq < 100 and nyquist_freq >= 60:
            freq_ranges.append((min_freq, max_freq, f"{min_freq:.3f}-{max_freq}Hz"))
        if nyquist_freq >= 100:
            freq_ranges.append((min_freq, 100, f"{min_freq:.3f}-100Hz"))
        if nyquist_freq >= 200:
            freq_ranges.append((min_freq, max_freq, f"{min_freq:.3f}-{max_freq:.1f}Hz"))

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

    def _advanced_process(self, data: np.ndarray, output_dir: str, suffix: str = None):
        """
        处理ECG数据，应用滤波和高级分析
        Args:
            data: 输入的ECG数据
            output_dir: 输出目录路径
            suffix: 可选的文件名后缀
        """
        # 尝试导入neurokit2
        try:
            import neurokit2 as nk
        except ImportError:
            self.send_message("Warning: neurokit2库未安装，跳过高级ECG分析")
            return

        # 根据最低频率确定method
        min_freq = self.calculate_min_frequency(len(data))
        method = 'neurokit' if min_freq <= 0.5 else 'pantompkins'
        if method != 'neurokit':
            self.send_message(f"Warning: 数据时长过短，改用{method}方法对ECG进一步滤波")
        try:
            # 确保数据是numpy数组
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=float)

            # 使用neurokit2进行ECG处理
            signals, info = nk.ecg_process(data, sampling_rate=self.sampling_rate, method=method)

            # 准备文件名后缀
            file_suffix = "orig" if suffix is None else suffix

            # 保存neurokit2的处理结果
            nk.ecg_plot(signals, info)
            fig = plt.gcf()
            fig.set_size_inches(30, 9)
            save_path = os.path.join(output_dir, f"nk_ecg_{file_suffix}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.send_message(f"Neurokit2处理结果已保存至: {save_path}")

        except Exception as e:
            self.send_message(f"ECG高级处理失败: {str(e)}")
            self.send_message(traceback.format_exc())


# =====================main(单独执行时使用)=====================
def main():
    # 获取用户输入的路径
    input_path = input("请复制实验文件夹所在目录的绝对路径(若Python代码在同一目录, 请直接按Enter): \n")

    # 判断用户是否直接按Enter, 设置为当前工作目录
    if not input_path:
        work_folder = os.getcwd()
    elif os.path.isdir(input_path):
        work_folder = input_path

    ecg_handler = ECGHandler(filter_low_cut=0.2, filter_high_cut=30.0, filter_order=4)

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
