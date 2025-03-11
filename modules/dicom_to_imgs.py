"""
    ==========================README===========================
    create date:    20240805
    change date:    20240913
    creator:        zhengxu
    function:       批量读取DICOM序列,并转换成图片和视频
"""
# =========================用到的库==========================
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import pydicom
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 获取当前文件所在目录,并加入系统环境变量(临时)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from modules.files_basic import FilesBasic

# 添加 DLL 搜索路径
if os.name == 'nt':  # 仅在 Windows 系统上执行
    # 设置 DLL 搜索路径，libs 文件夹位于 modules 的上一级目录
    libs_dir = os.path.join(current_dir, '..', 'libs')
    try:
        os.add_dll_directory(libs_dir)
        print(f"Added DLL directory: {libs_dir}")
    except AttributeError:
        # Python < 3.8 没有 os.add_dll_directory 方法，直接修改 PATH
        os.environ['PATH'] = f"{libs_dir};" + os.environ['PATH']


# =========================================================
# =======               DICOM导出图片              =========
# =========================================================
class DicomToImage(FilesBasic):
    def __init__(self,
                 log_folder_name: str = 'dicom_handle_log',
                 fps: int = 10,
                 frame_dpi: int = 800,
                 out_dir_prefix: str = 'Img-'):
        super().__init__()

        # 重写父类suffixs,为dicom文件可能的后缀
        self.suffixs = ['.dcm', '']

        # 设置导出图片dpi & 导出图片文件夹的前缀名 & log文件夹名字
        self.fps = fps
        self.frame_dpi = frame_dpi
        self.log_folder_name = log_folder_name
        self.out_dir_prefix = out_dir_prefix

    # =====================处理(单个数据文件夹)函数======================
    def _data_dir_handler(self, _data_dir: str):
        # 检查_data_dir,为空则终止,否则创建输出文件夹,继续执行
        seq_dirs = self.__check_dicomdir(_data_dir)
        if not seq_dirs:
            self.send_message(f"Error: empty dicomdir「{_data_dir}」skipped")
            return
        outfolder_name = self.out_dir_prefix + _data_dir
        os.makedirs(outfolder_name, exist_ok=True)

        max_works = min(self.max_threads, os.cpu_count(), len(seq_dirs) * 2)
        with ThreadPoolExecutor(max_workers=max_works) as executor:
            # 获取dicom序列文件名list,并多线程调用处理每个dicom序列
            futures = []  # 保存所有提交的任务
            for seq_dir in seq_dirs:
                seqs_list = self._get_filenames_by_suffix(os.path.join(_data_dir, seq_dir))
                if not seqs_list:
                    self.send_message(f"Warning: empty seq dir「{seq_dir}」skipped")
                    continue
                for seq in seqs_list:
                    abs_input_path = os.path.join(self._work_folder, _data_dir, seq_dir, seq)
                    abs_outfolder_path = os.path.join(self._work_folder, outfolder_name)
                    futures.append(executor.submit(self.single_file_handler,
                                   abs_input_path,
                                   abs_outfolder_path,
                                   seq_dir))

            # 等待所有任务完成
            for future in futures:
                future.result()

        # 所有线程处理完成后，使用FFmpeg生成视频
        abs_outfolder_path = os.path.join(self._work_folder, outfolder_name)
        self.send_message("所有DICOM文件处理完成, 开始生成视频...")
        self._generate_videos_with_ffmpeg(abs_outfolder_path)
        self.send_message("视频生成完成")

        return True

    # ======================DICOM序列保存图片======================
    def single_file_handler(self, abs_input_path: str,
                            abs_outfolder_path: str,
                            seq_dir_name: str = 'none'):
        # 检查文件路径格式
        if not self.check_file_path(abs_input_path, abs_outfolder_path):
            self.send_message("Error: failed to check_file_path")
            return
        try:
            # 尝试读取 DICOM 文件
            ds = pydicom.dcmread(abs_input_path)
            # self.send_message(f"检测到DICOM文件: {abs_input_path}, 正在处理")
        except Exception:
            # 如果读取失败, 不抛出异常, 直接返回
            self.send_message(f"Error: failed to read the dicom file「{abs_input_path}」")
            return
        # 检查是否有多帧图像
        num_frames = ds.get('NumberOfFrames', 1)

        # 读取所有帧的图像数据
        pixel_array = ds.pixel_array

        # 构建输出文件名
        seq_file_name = os.path.basename(abs_input_path)
        seq_name, _ = os.path.splitext(seq_file_name)    # 去掉后缀

        # 视频写入初始化
        ref_height, ref_width = pixel_array[0].shape if num_frames > 1 else pixel_array.shape
        video_filename = os.path.join(abs_outfolder_path, f'seq_{seq_dir_name}-{seq_name}.mp4')

        # 检测视频文件是否存在，如果存在则删除
        if os.path.exists(video_filename):
            os.remove(video_filename)

        # 初始化视频写入对象
        if num_frames > 1:
            # 设置保存帧图片的目录，用于后续FFmpeg处理
            frames_dir = os.path.join(abs_outfolder_path, f'seq_{seq_dir_name}-{seq_name}_frames')
            os.makedirs(frames_dir, exist_ok=True)

            # 记录视频信息，稍后由FFmpeg处理
            video_info = {
                'frames_dir': frames_dir,
                'output_video': video_filename,
                'fps': self.fps,
                'width': ref_width,
                'height': ref_height
            }
            # 将信息保存到文件中，以便后续处理
            with open(os.path.join(frames_dir, 'video_info.json'), 'w') as f:
                import json
                json.dump(video_info, f)

        # 遍历每一帧, 保存为 PNG 图片
        for i in range(num_frames):
            # 提取当前帧的图像数据 # 如果视频帧为1, 则pixel_array不是一个数组, 所以要直接赋值
            frame_data = pixel_array[i] if num_frames > 1 else pixel_array

            if frame_data.dtype != np.uint8:
                # 归一化数据到 [0, 255] 范围
                min_bit = np.min(frame_data)
                max_bit = np.max(frame_data)
                if max_bit > min_bit:  # 防止除零错误
                    frame_data = (frame_data - min_bit) / (max_bit - min_bit) * 255
                frame_data = frame_data.astype(np.uint8)

            # 创建图像对象
            image = Image.fromarray(frame_data)
            if num_frames == 1:
                image_filename = os.path.join(abs_outfolder_path, f'seq_{seq_dir_name}-{seq_name}.png')
                image.save(image_filename, dpi=(self.frame_dpi, self.frame_dpi))
                self.send_message(f'单帧 DICOM 图像已保存到 {image_filename}')
            else:
                # 修改帧图片保存路径
                image_filename = os.path.join(frames_dir, f'frame_{i + 1:04d}.png')
                image.save(image_filename, dpi=(self.frame_dpi, self.frame_dpi))
                self.send_message(f'视频帧已保存到 {frames_dir}，将在处理完成后生成视频')

    # =====================找到DICOM序列文件夹列表======================
    def __check_dicomdir(self, _data_dir):
        try:
            items = os.listdir(_data_dir)
            dicomdir_found = any(item == 'DICOMDIR'
                                 and os.path.isfile(os.path.join(_data_dir, item))
                                 for item in items)

            folder_list = [item for item in items
                           if os.path.isdir(os.path.join(_data_dir, item))
                           and item != 'seq_imgs']

            if dicomdir_found:
                return folder_list
            else:
                self.send_message("DICOMDIR not found.")
                return folder_list
        except Exception as e:
            self.send_message(f"Error checking DICOMDIR: {e}")
            return None

    # =====================使用FFmpeg生成所有待处理的视频======================
    def _generate_videos_with_ffmpeg(self, output_folder):
        """使用FFmpeg生成所有待处理的视频"""
        import json
        import subprocess
        import shutil

        # 首先检查FFmpeg是否可用
        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            self.send_message("错误: 系统中未找到FFmpeg。请安装FFmpeg后再试。")
            return

        self.send_message(f"使用系统FFmpeg: {ffmpeg_path}")

        # 查找所有保存了视频信息的目录
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                if file == 'video_info.json':
                    info_path = os.path.join(root, file)
                    try:
                        with open(info_path, 'r') as f:
                            video_info = json.load(f)

                        frames_dir = video_info['frames_dir']
                        output_video = video_info['output_video']
                        fps = video_info['fps']

                        # 检查帧图像是否存在
                        frame_pattern = os.path.join(frames_dir, 'frame_*.png')
                        import glob
                        frames = glob.glob(frame_pattern)
                        if not frames:
                            self.send_message(f"警告: 未找到帧图像: {frame_pattern}")
                            continue

                        self.send_message(f"找到 {len(frames)} 个帧图像，开始生成视频")
                    except Exception as e:
                        self.send_message(f'处理视频信息时出错: {str(e)}')

                    # 执行FFmpeg命令生成视频
                    try:
                        shell_cmd = f'{ffmpeg_path} -y -framerate {fps} -i "{frames_dir}/frame_%04d.png" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p "{output_video}"'
                        result = subprocess.run(
                            shell_cmd,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=False
                        )
                        if result.returncode == 0:
                            self.send_message(f'视频生成成功: {output_video}')
                        else:
                            self.send_message(f'视频生成失败: {result.stderr}')
                    except Exception as e:
                        self.send_message(f'执行FFmpeg命令时出错: {str(e)}')


# =====================main(单独执行时使用)=====================
def main():
    # 获取用户输入的路径
    input_path = input("请复制实验文件夹所在目录的绝对路径(若Python代码在同一目录, 请直接按Enter): \n")

    # 判断用户是否直接按Enter，设置为当前工作目录
    if not input_path:
        work_folder = os.getcwd()
    elif os.path.isdir(input_path):
        work_folder = input_path

    DicomHandler = DicomToImage()
    DicomHandler.set_work_folder(work_folder)
    possble_dirs = DicomHandler.possble_dirs

    # 给用户显示，请用户输入index
    number = len(possble_dirs)
    DicomHandler.send_message('\n')
    for i in range(number):
        print(f"{i}: {possble_dirs[i]}")
    user_input = input("\n请选择要处理的序号(用空格分隔多个序号): \n")

    # 解析用户输入
    try:
        indices = user_input.split()
        index_list = [int(index) for index in indices]
    except ValueError:
        DicomHandler.send_message("输入错误, 必须输入数字")

    RESULT = DicomHandler.selected_dirs_handler(index_list)
    if not RESULT:
        print("输入数字不在提供范围, 请重新运行")


# =========================调试用============================
if __name__ == '__main__':
    main()
