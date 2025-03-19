"""
    ===========================README============================
    create date:    20250201
    change date:    20250319
    creator:        zhengxu
    function:       批量生成视频字幕
    details:        _data_dir为视频文件夹,内部的文件夹为子任务

    version:        beta 2.0
    updates:        继承FilesBasic类,实现多线程
"""
# =========================用到的库==========================
import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

# 尝试导入faster-whisper库，但不强制要求
try:
    from faster_whisper import WhisperModel
    has_faster_whisper = True
except ImportError:
    has_faster_whisper = False

# 获取当前文件所在目录,并加入系统环境变量(临时)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from modules.files_basic import FilesBasic

home = os.environ.get("HOME")  # 获取 $HOME 环境变量
global_model_path = os.path.join(home, "Develop/whisper_models/ggml-large-v3-turbo-q5_0.bin")


# =========================================================
# =======               视频字幕生成类              =========
# =========================================================
class GenSubtitles(FilesBasic):
    def __init__(self,
                 log_folder_name: str = 'gen_subtitles_log',
                 model_path: str = global_model_path,
                 compute_type: str = "auto",
                 parallel: bool = False):
        """
        初始化视频字幕生成器

        Args:
            log_folder_name: 日志文件夹名称
            model_path: Whisper模型名称或路径
            compute_type: 计算类型，可选 "auto", "int8", "int8_float16", "int16", "float16", "float32"
            parallel: 是否使用并行处理
            out_dir_prefix: 输出文件夹前缀
        """
        super().__init__(log_folder_name=log_folder_name)
        self.model_path = model_path
        self.compute_type = compute_type
        self.parallel = parallel
        self.video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
        self.suffixs = self.video_extensions

        # 检查是否有whisper-cli
        self.has_whisper_cli = shutil.which("whisper-cli") is not None

        # 检查工具是否存在
        self._check_requirements()

    def _check_requirements(self) -> None:
        """检查必需的工具是否存在"""
        # 检查ffmpeg
        if not shutil.which("ffmpeg"):
            raise RuntimeError("错误: 未找到ffmpeg命令. 请先安装ffmpeg. ")

        # 如果既没有whisper-cli也没有faster-whisper库，则报错
        if not self.has_whisper_cli and not has_faster_whisper:
            raise RuntimeError("错误: 未找到whisper-cli命令或faster-whisper库. 请至少安装其中一个. ")

        # 如果没有whisper-cli，则加载faster-whisper模型
        if not self.has_whisper_cli and has_faster_whisper:
            self.send_message("未检测到whisper-cli, 使用faster-whisper库")
            self.send_message(f"正在加载Whisper模型 '{self.model_path}'...")
            try:
                self.model = WhisperModel(self.model_path, compute_type=self.compute_type)
                self.send_message("模型加载完成")
            except Exception as e:
                raise RuntimeError(f"加载模型失败: {e}")

    def _data_dir_handler(self, _data_dir: str):
        """处理单个数据文件夹，支持串行和并行处理"""
        # 检查_data_dir,为空则终止,否则创建输出文件夹,继续执行
        file_list = self._get_filenames_by_suffix(_data_dir)
        if not file_list:
            self.send_message(f"Error: No video file in {_data_dir}")
            return
        outfolder_name = _data_dir  # 字幕输出到视频文件夹
        # os.makedirs(outfolder_name, exist_ok=True)

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

    def single_file_handler(self, abs_input_path: str, abs_outfolder_path: str):
        """处理单个视频文件: 提取音频、生成字幕"""
        # 检查文件路径格式
        if not self.check_file_path(abs_input_path, abs_outfolder_path):
            self.send_message("Error: failed to check_file_path")
            return

        self.send_message(f"处理视频: {os.path.basename(abs_input_path)}")

        # 提取音频
        audio_path = Path(abs_input_path).parent / f"{Path(abs_input_path).stem}_audio.wav"
        success = self._extract_audio(Path(abs_input_path), audio_path)
        if not success:
            return

        # 生成字幕
        success = self._generate_subtitle(audio_path, Path(abs_input_path))

        # 清理临时音频文件
        if audio_path.exists():
            try:
                os.remove(audio_path)
            except OSError as e:
                self.send_message(f"警告: 无法删除临时音频文件 '{audio_path}': {e}")

        if success:
            self.send_message(f"字幕生成完成: {Path(abs_input_path).stem}.srt")

    def _extract_audio(self, video_path: Path, audio_path: Path) -> bool:
        """从视频中提取音频"""
        if not video_path.exists():
            self.send_message(f"错误: 视频文件 '{video_path}' 不存在！")
            return False

        cmd = ["ffmpeg", "-i", str(video_path),
               "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
               "-hide_banner", "-loglevel", "error",
               str(audio_path)]

        try:
            import subprocess
            subprocess.run(cmd, check=True)
            # 更改音频文件权限
            os.chmod(audio_path, 0o777)
            return True
        except subprocess.CalledProcessError:
            self.send_message(f"错误: '{video_path.name}' 的音频提取失败！")
            return False

    def _generate_subtitle(self, audio_path: Path, original_video: Path) -> bool:
        """使用whisper-cli或faster-whisper库生成字幕"""
        if not audio_path.exists():
            self.send_message(f"错误: 音频文件 '{audio_path}' 不存在！")
            return False

        basename = original_video.stem

        # 优先使用whisper-cli
        if self.has_whisper_cli:
            self.send_message("使用whisper-cli生成字幕")
            output_path = original_video.parent / basename

            cmd = [
                "whisper-cli",
                "--model", str(self.model_path),
                "--file", str(audio_path),
                "-osrt",
                "-of", str(output_path),
                "--language", "auto"
            ]

            try:
                import subprocess
                subprocess.run(cmd, check=True)
                return True
            except subprocess.CalledProcessError:
                self.send_message(f"错误: '{original_video.name}' 的字幕生成失败！")
                return False

        # 如果没有whisper-cli，使用faster-whisper库
        elif has_faster_whisper:
            self.send_message("使用faster-whisper库生成字幕")
            output_srt_path = original_video.parent / f"{basename}.srt"

            try:
                # 使用faster-whisper进行转写
                self.send_message(f"开始转写 '{basename}'...")
                segments, info = self.model.transcribe(
                    str(audio_path),
                    language="auto",  # 自动检测语言
                    vad_filter=True,  # 使用语音活动检测过滤
                    word_timestamps=True  # 获取单词级时间戳
                )

                self.send_message(f"检测到语言: {info.language}, 概率: {info.language_probability:.2f}")

                # 将segments转换为SRT格式并写入文件
                with open(output_srt_path, "w", encoding="utf-8") as srt_file:
                    for i, segment in enumerate(segments, start=1):
                        # 格式化时间（从秒转为SRT格式 HH:MM:SS,mmm）
                        start_time = self._format_timestamp(segment.start)
                        end_time = self._format_timestamp(segment.end)

                        # 写入SRT格式
                        srt_file.write(f"{i}\n")
                        srt_file.write(f"{start_time} --> {end_time}\n")
                        srt_file.write(f"{segment.text.strip()}\n\n")

                self.send_message(f"字幕已保存到 '{output_srt_path}'")
                return True

            except Exception as e:
                self.send_message(f"错误: '{original_video.name}' 的字幕生成失败！原因: {e}")
                return False
        else:
            self.send_message("错误: 没有可用的字幕生成工具")
            return False

    def _format_timestamp(self, seconds: float) -> str:
        """将秒数转换为SRT时间戳格式 (HH:MM:SS,mmm)"""
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)

        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"


# =====================main(单独执行时使用)=====================
def main():
    # 获取用户输入的路径
    input_path = input("请复制实验文件夹所在目录的绝对路径(若Python代码在同一目录, 请直接按Enter): \n")

    # 判断用户是否直接按Enter，设置为当前工作目录
    if not input_path:
        work_folder = os.getcwd()
    elif os.path.isdir(input_path):
        work_folder = input_path

    subtitles_generator = GenSubtitles()
    subtitles_generator.set_work_folder(work_folder)
    possble_dirs = subtitles_generator.possble_dirs

    # 给用户显示，请用户输入index
    number = len(possble_dirs)
    subtitles_generator.send_message('\n')
    for i in range(number):
        print(f"{i}: {possble_dirs[i]}")
    user_input = input("\n请选择要处理的序号(用空格分隔多个序号): \n")

    # 解析用户输入
    try:
        indices = user_input.split()
        index_list = [int(index) for index in indices]
    except ValueError:
        subtitles_generator.send_message("输入错误, 必须输入数字")

    RESULT = subtitles_generator.selected_dirs_handler(index_list)
    if not RESULT:
        subtitles_generator.send_message("输入数字不在提供范围, 请重新运行")


# =========================调试用============================
if __name__ == '__main__':
    main()
