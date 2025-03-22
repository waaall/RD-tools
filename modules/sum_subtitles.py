"""
    ===========================README============================
    create date:    20250319
    change date:    20250321
    creator:        zhengxu
    function:       批量处理视频字幕，生成总结和关键节点截图
    details:        _data_dir为视频文件夹,内部的文件夹为子任务

    version:        beta 2.0
"""
# =========================用到的库==========================
import os
import sys
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

import ollama
from PIL import Image
import fitz  # PyMuPDF

# 获取当前文件所在目录,并加入系统环境变量(临时)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from modules.files_basic import FilesBasic


# =========================================================
# =======               字幕总结类              =========
# =========================================================
class SumSubtitles(FilesBasic):
    def __init__(self,
                 log_folder_name: str = 'sum_subtitles_log',
                 model_name: str = "deepseek-r1:1.5b",
                 parallel: bool = False):
        """
        初始化字幕总结器
        Args:
            log_folder_name: 日志文件夹名称
            model_name: 使用的模型名称
            parallel: 是否使用并行处理
        """
        super().__init__(log_folder_name=log_folder_name)
        self.model_name = model_name
        self.parallel = parallel
        self.suffixs = ['.srt']  # 只处理srt字幕文件

        # 检查必需工具
        self._check_requirements()

    def _check_requirements(self) -> None:
        """检查必需的工具是否存在"""
        # 检查ffmpeg
        if not shutil.which("ffmpeg"):
            raise RuntimeError("Error: 未找到ffmpeg命令. 请先安装ffmpeg. ")

        # 检查ollama服务是否可用
        try:
            ollama.list()
        except Exception as e:
            self.send_message(f"Error: Ollama不可用: {e}")

    def _data_dir_handler(self, _data_dir: str):
        """处理单个数据文件夹，支持串行和并行处理"""
        file_list = self._get_filenames_by_suffix(_data_dir)
        if not file_list:
            self.send_message(f"Error: No subtitle file in {_data_dir}")
            return

        if self.parallel:
            max_works = min(self.max_threads, os.cpu_count(), len(file_list))
            with ThreadPoolExecutor(max_workers=max_works) as executor:
                for file_name in file_list:
                    abs_input_path = os.path.join(self._work_folder, _data_dir, file_name)
                    executor.submit(self.single_file_handler, abs_input_path)
        else:
            for file_name in file_list:
                abs_input_path = os.path.join(self._work_folder, _data_dir, file_name)
                self.single_file_handler(abs_input_path)

    def single_file_handler(self, abs_input_path: str):
        """处理单个字幕文件"""
        if not self.check_file_path(abs_input_path, abs_input_path):
            self.send_message("Error: failed to check_file_path")
            return

        self.send_message(f"处理字幕文件: {os.path.basename(abs_input_path)}")

        # 读取字幕内容
        subtitle_content = self._read_subtitle(abs_input_path)
        if not subtitle_content:
            return

        # 调用Ollama生成总结
        summary = self._generate_summary(subtitle_content)
        if not summary:
            return

        # 保存Markdown文件
        md_path = self._save_markdown(summary, abs_input_path)
        if not md_path:
            return

        # 提取时间戳并截图
        timestamps = self._extract_timestamps(summary)
        if timestamps:
            self._capture_frames(abs_input_path, timestamps)

        # 生成PDF
        self._generate_pdf(md_path, timestamps)

    def _read_subtitle(self, subtitle_path: str) -> Optional[str]:
        """读取字幕文件内容"""
        try:
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.send_message(f"Error: 无法读取字幕文件 '{subtitle_path}': {e}")
            return None

    def _generate_summary(self, subtitle_content: str) -> Optional[str]:
        """调用Ollama生成总结"""
        prompt = f"""请分析以下字幕内容，并生成一份总结。在总结中，请：
1. 提取主要信息和关键点
2. 在关键节点处标注时间戳(格式: HH:MM:SS)

字幕内容：
{subtitle_content}
"""

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False
            )
            return response['response']
        except Exception as e:
            self.send_message(f"Error: 调用Ollama服务失败: {e}")
            return None

    def _save_markdown(self, summary: str, subtitle_path: str) -> Optional[str]:
        """保存总结为Markdown文件"""
        try:
            md_path = Path(subtitle_path).with_suffix('.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            self.send_message(f"已保存Markdown文件: {md_path}")
            return str(md_path)
        except Exception as e:
            self.send_message(f"Error: 保存Markdown文件失败: {e}")
            return None

    def _extract_timestamps(self, summary: str) -> List[str]:
        """从总结中提取时间戳"""
        pattern = r'\d{2}:\d{2}:\d{2}'
        return re.findall(pattern, summary)

    def _capture_frames(self, subtitle_path: str, timestamps: List[str]):
        """根据时间戳截取视频帧"""
        video_path = Path(subtitle_path).with_suffix('.mp4')  # 假设视频是mp4格式
        if not video_path.exists():
            self.send_message(f"Error: 未找到对应的视频文件: {video_path}")
            return

        output_dir = Path(subtitle_path).parent / "frames"
        output_dir.mkdir(exist_ok=True)

        for timestamp in timestamps:
            output_path = output_dir / f"{timestamp}.jpg"
            cmd = [
                "ffmpeg", "-ss", timestamp,
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                str(output_path)
            ]
            try:
                subprocess.run(cmd, check=True)
                self.send_message(f"已保存截图: {output_path}")
            except subprocess.CalledProcessError:
                self.send_message(f"Error: 截图失败 {timestamp}")

    def _generate_pdf(self, md_path: str, timestamps: List[str]):
        """生成PDF文档"""
        try:
            pdf_path = Path(md_path).with_suffix('.pdf')
            doc = fitz.open()

            # 添加Markdown内容
            page = doc.new_page()
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            page.insert_text((50, 50), content)

            # 添加截图
            frames_dir = Path(md_path).parent / "frames"
            for timestamp in timestamps:
                img_path = frames_dir / f"{timestamp}.jpg"
                if img_path.exists():
                    page = doc.new_page()
                    img = Image.open(img_path)
                    img_bytes = img.tobytes()
                    page.insert_image((50, 50), stream=img_bytes)

            doc.save(pdf_path)
            doc.close()
            self.send_message(f"已生成PDF文件: {pdf_path}")
        except Exception as e:
            self.send_message(f"Error: 生成PDF文件失败: {e}")


# =====================main(单独执行时使用)=====================
def main():
    # 获取用户输入的路径
    input_path = input("请复制实验文件夹所在目录的绝对路径(若Python代码在同一目录, 请直接按Enter): \n")

    # 判断用户是否直接按Enter，设置为当前工作目录
    if not input_path:
        work_folder = os.getcwd()
    elif os.path.isdir(input_path):
        work_folder = input_path

    subtitles_summarizer = SumSubtitles()
    subtitles_summarizer.set_work_folder(work_folder)
    possble_dirs = subtitles_summarizer.possble_dirs

    # 给用户显示，请用户输入index
    number = len(possble_dirs)
    subtitles_summarizer.send_message('\n')
    for i in range(number):
        print(f"{i}: {possble_dirs[i]}")
    user_input = input("\n请选择要处理的序号(用空格分隔多个序号): \n")

    # 解析用户输入
    try:
        indices = user_input.split()
        index_list = [int(index) for index in indices]
    except ValueError:
        subtitles_summarizer.send_message("输入Error, 必须输入数字")

    RESULT = subtitles_summarizer.selected_dirs_handler(index_list)
    if not RESULT:
        subtitles_summarizer.send_message("输入数字不在提供范围, 请重新运行")


# =========================调试用============================
if __name__ == '__main__':
    main()
