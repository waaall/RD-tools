"""
    ===========================README============================
    create date:    20250319
    change date:    20250322
    creator:        zhengxu
    function:       批量处理视频字幕, 生成总结和关键节点截图的PDF
    details:        _data_dir为视频文件夹,内部的文件夹为子任务

    version:        beta 3.0
"""
# =========================用到的库==========================
import os
import sys
import re
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from PIL import Image
import fitz  # PyMuPDF

# 获取当前文件所在目录,并加入系统环境变量(临时)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from modules.files_basic import FilesBasic
from modules.AI_chat import create_chat_instance, AIChatBase


# =========================================================
# =======               字幕总结类              =========
# =========================================================
class SumSubtitles(FilesBasic):
    def __init__(self,
                 log_folder_name: str = 'sum_subtitles_log',
                 api_provider: str = "ollama",
                 model_name: str = "qwq",
                 api_key: str = None,
                 temperature: float = 0.5,
                 max_tokens: int = 4096,
                 parallel: bool = False,
                 CAPTURE_FRAME: bool = True):
        """
        初始化字幕总结器
        Args:
            log_folder_name: 日志文件夹名称
            api_provider: API提供商, 可选值: "openai", "ollama", "deepseek", "ali", "siliconflow"
            model_name: 使用的模型名称
            api_key: API密钥（对于需要密钥的API提供商）
            temperature: 温度参数, 控制输出随机性
            max_tokens: 最大生成token数
            parallel: 是否使用并行处理
        """
        super().__init__(log_folder_name=log_folder_name)

        self.api_provider = api_provider
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.parallel = parallel
        self.out_dir_prefix = ''
        self.suffixs = ['.srt']  # 只处理srt字幕文件
        self.video_extensions = ['.mp4', '.mkv', '.avi', '.mov']
        self.CAPTURE_FRAME = CAPTURE_FRAME
        # AI聊天实例将在需要时创建
        self.ai_chat = None

        # 检查必需工具
        self._check_requirements()

    def _check_requirements(self) -> None:
        """检查必需的工具是否存在"""
        # 检查ffmpeg
        if not shutil.which("ffmpeg"):
            raise RuntimeError("Error: 未找到ffmpeg命令. 请先安装ffmpeg. ")

        # 检查依赖库
        try:
            import markdown
            import bs4
        except ImportError as e:
            self.send_message(f"Error: 缺少依赖库: {e}. 请安装: pip install markdown beautifulsoup4")

    def _create_ai_chat(self) -> AIChatBase:
        """按需创建或更新AI聊天实例"""
        # 检查实例是否已存在, 以及是否需要更新
        if self.ai_chat is None or self.ai_chat.__class__.__name__ != f"{self.api_provider.capitalize()}Chat":
            # 如果存在旧实例, 先清理资源
            if self.ai_chat:
                try:
                    self.ai_chat.clear_history()
                    self.ai_chat = None
                except Exception as e:
                    self.send_message(f"Warning: 释放旧AI聊天实例时出错: {e}")

            # 创建新的AI聊天实例
            try:
                self.ai_chat = create_chat_instance(
                    provider=self.api_provider,
                    model_name=self.model_name,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                self.send_message(f"已初始化 {self.api_provider} API, 模型: {self.model_name}")
            except Exception as e:
                self.send_message(f"Error: 初始化AI聊天实例失败: {e}")
                raise
        else:
            # 检查参数是否需要更新
            if (
                self.ai_chat.model_name != self.model_name
                or self.ai_chat.api_key != self.api_key
                or self.ai_chat.temperature != self.temperature
                or self.ai_chat.max_tokens != self.max_tokens
            ):
                # 更新实例参数
                self.ai_chat.model_name = self.model_name
                self.ai_chat.api_key = self.api_key
                self.ai_chat.temperature = self.temperature
                self.ai_chat.max_tokens = self.max_tokens
                self.send_message(f"已更新AI聊天实例参数, 模型: {self.model_name}")

        return self.ai_chat

    def single_file_handler(self, abs_input_path: str, abs_outfolder_path: str):
        """处理单个字幕文件"""
        if not os.path.exists(abs_input_path):
            self.send_message(f"Error: 文件不存在: {abs_input_path}")
            return

        if not self.check_file_path(abs_input_path, Path(abs_input_path).parent):
            self.send_message("Error: 文件检查失败")
            return

        # 读取字幕内容
        self.send_message(f"处理字幕文件: {os.path.basename(abs_input_path)}")
        subtitle_content = self._read_subtitle(abs_input_path)
        if not subtitle_content:
            return

        # 调用AI API生成总结
        summary = self._generate_summary(subtitle_content)
        if not summary:
            return

        # 提取Markdown摘要和JSON部分
        md_summary, json_content = self._extract_summary_and_json(summary)

        # 保存Markdown文件
        md_path = self._save_to_file(md_summary, abs_input_path, '.md')
        if not md_path:
            return

        # 保存JSON文件
        if json_content:
            json_path = self._save_to_file(json_content, abs_input_path, '.json')
            self.send_message(f"已保存JSON文件: {json_path}")

        # 提取时间戳
        if self.CAPTURE_FRAME:
            timestamps_data = self._extract_timestamps(json_content) if json_content else []
            if not timestamps_data:
                self.send_message("Warning: 未找到有效的时间戳数据, 将生成不包含截图的PDF")
        else:
            self.send_message("Warning: 未开启截图功能, 将生成不包含截图的PDF")
            timestamps_data = []

        # 执行视频截图
        if timestamps_data:
            self.send_message("正在截取视频关键帧...")
            frames_dir = self._capture_frames(abs_input_path, timestamps_data)
            if not frames_dir:
                self.send_message("Warning: 截图失败, 将生成不包含截图的PDF")

        # 生成PDF
        self.send_message("正在生成PDF文档...")
        pdf_path = Path(md_path).with_suffix('.pdf')
        frames_dir = Path(md_path).parent / "frames"
        self._generate_pdf(md_summary, pdf_path, frames_dir, timestamps_data)
        self.send_message(f"完成处理: {os.path.basename(abs_input_path)}")

    def _read_subtitle(self, subtitle_path: str) -> Optional[str]:
        """读取字幕文件内容"""
        try:
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # 尝试使用不同的编码
            try:
                with open(subtitle_path, 'r', encoding='gbk') as f:
                    return f.read()
            except Exception as e:
                self.send_message(f"Error: 无法读取字幕文件 '{subtitle_path}': {e}")
                return None
        except Exception as e:
            self.send_message(f"Error: 无法读取字幕文件 '{subtitle_path}': {e}")
            return None

    def _generate_summary(self, subtitle_content: str) -> Optional[str]:
        """调用AI API生成总结"""
        self._create_ai_chat()
        # 每次生成新的总结时清空历史对话
        self.ai_chat.clear_history()

        prompt = f"""请详细分析以下字幕内容, 并生成一份完整的视频总结。请严格按照以下两部分格式要求输出:

## 第一部分:视频摘要
使用Markdown格式提供视频的详细摘要, 包括:
- 视频的主要内容和主题
- 核心观点和结论
- 视频的整体结构和逻辑

## 第二部分:关键节点时间戳
必须使用以下严格的JSON格式提供关键内容的时间戳:
```json
[
  {{
    "time": "00:12:34",
    "content": "这里是关键内容的详细描述"
  }},
  {{
    "time": "00:23:45",
    "content": "另一个关键时刻的详细描述"
  }}
]
```

请注意以下严格要求:
1. 时间戳格式必须为"小时:分钟:秒"（HH:MM:SS）, 例如"01:23:45"
2. JSON必须完全符合上述格式, 不要有任何额外字符
3. 不要在JSON中使用Markdown语法
4. 至少提供5个关键时间节点（如果内容足够）
5. content字段请提供详细的内容描述, 不少于15个字

请分析字幕内容, 识别最重要的时间点, 并确保JSON可以被标准解析器正确解析。

字幕内容:
{subtitle_content}
"""
        try:
            # 使用AI聊天实例生成回复
            response = self.ai_chat.generate(prompt=prompt)
            return response
        except Exception as e:
            self.send_message(f"Error: 调用AI API失败: {e}")
            return None

    def _extract_summary_and_json(self, summary: str) -> Tuple[str, Optional[str]]:
        """将总结文本分离为Markdown摘要和JSON部分"""
        # 提取JSON部分
        json_match = re.search(r'```(?:json)?\s*(\[\s*\{\s*"time".*?\}\s*\])', summary, re.DOTALL)
        if not json_match:
            return summary, None

        json_content = json_match.group(1)
        # 尝试解析JSON以验证其有效性
        try:
            json.loads(json_content)
            # 从摘要中删除JSON部分, 保留纯摘要
            md_summary = re.sub(r'```(?:json)?.*?```', '', summary, flags=re.DOTALL)
            md_summary = md_summary.strip()
            return md_summary, json_content
        except json.JSONDecodeError as e:
            self.send_message(f"Warning: JSON解析失败: {e}")
            return summary, None

    def _extract_timestamps(self, json_content: Optional[str]) -> List[Dict[str, str]]:
        """从JSON内容中提取时间戳和内容"""
        if not json_content:
            self.send_message("Warning: 未提供JSON数据")
            return []
        try:
            timestamps_data = json.loads(json_content)

            # 验证JSON格式
            if not isinstance(timestamps_data, list):
                self.send_message("Warning: JSON数据不是列表格式")
                return []

            valid_timestamps = []
            for item in timestamps_data:
                if not isinstance(item, dict):
                    continue

                if "time" not in item or "content" not in item:
                    continue

                # 验证时间戳格式
                time_pattern = r'^\d{2}:\d{2}:\d{2}$'
                if not re.match(time_pattern, item["time"]):
                    self.send_message(f"Warning: 时间戳格式不正确: {item['time']}")
                    continue
                valid_timestamps.append(item)

            if valid_timestamps:
                self.send_message(f"成功解析到{len(valid_timestamps)}个时间戳")
                return valid_timestamps
            else:
                self.send_message("Warning: 未找到有效的时间戳数据")
        except json.JSONDecodeError as e:
            self.send_message(f"Warning: 解析JSON时间戳数据失败: {e}")
        except Exception as e:
            self.send_message(f"Warning: 解析时间戳数据时出错: {e}")

        self.send_message("Warning: 无法找到任何有效的时间戳")
        return []

    def _find_video_file(self, subtitle_path: str) -> Optional[Path]:
        """查找与字幕文件匹配的视频文件"""
        base_name = Path(subtitle_path).stem
        parent_dir = Path(subtitle_path).parent

        # 尝试所有支持的视频扩展名
        for ext in self.video_extensions:
            video_path = parent_dir / f"{base_name}{ext}"
            if video_path.exists():
                return video_path

        # 如果没有找到直接匹配的文件名, 尝试在同一目录下搜索相似名称的视频文件
        for file in parent_dir.glob('*'):
            if file.suffix in self.video_extensions and base_name in file.stem:
                return file
        return None

    def _capture_frames(self, subtitle_path: str, timestamps_data: List[Dict[str, str]]) -> Optional[Path]:
        """根据时间戳截取视频帧"""
        if not timestamps_data:
            self.send_message("Warning: 没有时间戳数据, 跳过截图")
            return None

        video_path = self._find_video_file(subtitle_path)
        if not video_path:
            self.send_message(f"Error: 未找到对应的视频文件, 已尝试扩展名: {', '.join(self.video_extensions)}")
            return None

        output_dir = Path(subtitle_path).parent / "frames"
        output_dir.mkdir(exist_ok=True)

        successful_frames = 0
        for item in timestamps_data:
            timestamp = item["time"]
            # 将时间戳中的冒号替换为下划线, 以便在Windows系统中创建有效的文件名
            safe_filename = timestamp.replace(':', '_')
            output_path = output_dir / f"{safe_filename}.png"  # 使用PNG格式
            cmd = [
                "ffmpeg", "-ss", timestamp,
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "1",  # 使用最高质量设置
                "-vf", "scale=800:-1",  # 固定宽度800像素，高度自适应
                "-pix_fmt", "rgb24",  # 使用标准RGB24像素格式
                "-y",  # 覆盖已存在的文件
                str(output_path)
            ]
            try:
                # 使用capture_output=True捕获输出, stderr=subprocess.PIPE抑制错误输出
                subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                successful_frames += 1
            except subprocess.CalledProcessError as e:
                self.send_message(f"Error: 截图失败 {timestamp}: {e}")

        if successful_frames > 0:
            self.send_message(f"成功截取了 {successful_frames}/{len(timestamps_data)} 个视频帧")
            return output_dir
        else:
            self.send_message("Warning: 所有截图都失败了")
            return None

    def _generate_pdf(self, md_content: str, pdf_path: Path, frames_dir: Path, timestamps_data: List[Dict[str, str]]):
        """生成PDF文档, 先将markdown转为PDF作为第一页, 然后添加截图和内容"""
        # 导入必要的库
        try:
            import markdown
            from bs4 import BeautifulSoup
        except ImportError as e:
            self.send_message(f"Error: 缺少必要的库: {e}")
            return None

        # 创建PDF文档
        try:
            doc = fitz.open()
        except Exception as e:
            self.send_message(f"Error: 无法创建PDF文档: {e}")
            return None

        # 获取系统中文字体
        default_fontname = 'china-s'
        self.send_message(f"使用字体: {default_fontname}")

        # 将Markdown转换为文本
        try:
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text()
        except Exception as e:
            self.send_message(f"Error: Markdown转换失败: {e}")
            doc.close()
            return None

        # 创建标题页
        try:
            current_page = doc.new_page()
            self._safe_insert_text(current_page, (50, 50), "视频摘要", 18, default_fontname)
        except Exception as e:
            self.send_message(f"创建标题页失败: {e}")
            doc.close()
            return None

        # 处理正文内容
        self._add_content_pages(doc, current_page, text, default_fontname)

        # 添加时间戳和截图
        if timestamps_data:
            self._add_timestamp_pages(doc, timestamps_data, frames_dir, default_fontname)

        # 保存PDF
        try:
            doc.save(pdf_path)
            doc.close()
            self.send_message(f"已生成PDF文件: {pdf_path}")
            return pdf_path
        except Exception as e:
            self.send_message(f"Error: 保存PDF文件失败: {e}")
            if doc:
                doc.close()
            return None

    def _add_content_pages(self, doc, current_page, text, default_fontname):
        """添加内容页面"""
        y_pos = 80
        paragraphs = text.split('\n\n')

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            # 处理段落文本，分割为适合页面宽度的行
            words = paragraph.split()
            lines = []
            current_line = []
            line_length = 0

            for word in words:
                if line_length + len(word) + 1 <= 80:  # +1 for space
                    current_line.append(word)
                    line_length += len(word) + 1
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    line_length = len(word)

            if current_line:
                lines.append(' '.join(current_line))

            # 检查是否需要创建新页面
            if y_pos + 15 * len(lines) > 800:
                current_page = doc.new_page()
                y_pos = 50

            # 插入段落内容
            if paragraph.startswith('#'):
                # 处理标题
                title = paragraph.split('\n')[0]
                if self._safe_insert_text(current_page, (50, y_pos), title, 14, default_fontname):
                    y_pos += 25
                else:
                    y_pos += 10

            # 插入段落文本
            for line in lines:
                if line.strip():
                    if self._safe_insert_text(current_page, (50, y_pos), line, 11, default_fontname):
                        y_pos += 15
                    else:
                        y_pos += 15  # 即使插入失败也移动位置

            y_pos += 10  # 段落间距
            # 检查是否需要新页面
            if y_pos > 800:
                current_page = doc.new_page()
                y_pos = 50

    def _add_timestamp_pages(self, doc, timestamps_data, frames_dir, default_fontname):
        """添加时间戳和截图页面"""
        for item in timestamps_data:
            timestamp = item["time"]
            content = item.get("content", "未提供内容描述")

            # 使用与_capture_frames方法相同的文件名转换逻辑
            safe_filename = timestamp.replace(':', '_')
            img_path = frames_dir / f"{safe_filename}.png"  # 使用PNG格式

            # 如果找不到PNG格式，尝试找JPG格式（兼容旧版本生成的文件）
            if not img_path.exists():
                self.send_message(f"Warning: 未找到时间点 {timestamp} 的截图")
                continue

            # 创建新页面
            page = doc.new_page()

            # 添加时间戳作为标题
            self._safe_insert_text(page, (50, 40), f"时间点: {timestamp}", 14, default_fontname)

            # 添加内容描述
            y_text = 70
            content_lines = [content[i:i + 80] for i in range(0, len(content), 80)]

            for line in content_lines:
                if line.strip():
                    if self._safe_insert_text(page, (50, y_text), line, 11, default_fontname):
                        y_text += 15
                    else:
                        y_text += 15  # 即使插入失败也移动位置

            # 确保有足够空间放置图片
            y_image = y_text + 15

            # 改进的图片插入逻辑
            try:
                # 创建临时文件处理图片
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name

                # 使用PIL打开和处理图片
                img = Image.open(img_path)

                # 确保图片为RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 计算适当的图像尺寸以适应页面
                max_width = 500
                max_height = 700 - y_image
                width, height = img.size
                aspect = width / height

                if width > max_width:
                    width = max_width
                    height = width / aspect

                if height > max_height:
                    height = max_height
                    width = height * aspect

                # 调整大小并保存为临时文件
                img_resized = img.resize((int(width), int(height)), Image.LANCZOS)
                img_resized.save(tmp_path, format='PNG')

                # 使用fitz插入图片
                rect = fitz.Rect(50, y_image, 50 + width, y_image + height)
                try:
                    # 首选使用文件名方式插入
                    page.insert_image(rect, filename=tmp_path)
                except Exception as e1:
                    self.send_message(f"Warning: 使用文件名插入失败，尝试其他方式: {e1}")
                    try:
                        # 尝试使用二进制数据
                        with open(tmp_path, 'rb') as f:
                            img_data = f.read()
                        page.insert_image(rect, stream=img_data)
                    except Exception as e2:
                        self.send_message(f"Warning: 无法插入图片: {e2}")
                        self._safe_insert_text(page, (50, y_image + 50), f"无法插入图片: {e2}", 10, "helv")

                # 删除临时文件
                import os
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    self.send_message(f"Warning: 无法删除临时文件: {e}")

            except Exception as e:
                self.send_message(f"Warning: 无法处理图片 {img_path}: {e}")
                # 添加错误说明
                self._safe_insert_text(page, (50, y_image + 50), f"无法加载图片: {e}", 10, "helv")

    def _safe_insert_text(self, page, point, text, fontsize, fontname, color=None):
        """安全地在PDF页面插入文本, 出错时尝试备用字体"""
        try:
            # 尝试使用首选字体
            kwargs = {"fontsize": fontsize, "fontname": fontname}
            if color:
                kwargs["color"] = color
            page.insert_text(point, text, **kwargs)
            return True
        except Exception as e:
            self.send_message(f"插入文本失败 [{fontname}]: {e}")
            try:
                # 尝试使用备用字体
                page.insert_text(point, text, fontsize=fontsize, fontname="helv")
                return True
            except Exception as e2:
                self.send_message(f"使用备用字体也失败: {e2}")
                return False


# =====================main(单独执行时使用)=====================
def main():
    # 获取用户输入的路径
    input_path = input("请复制实验文件夹所在目录的绝对路径(若Python代码在同一目录, 请直接按Enter): \n")

    # 判断用户是否直接按Enter, 设置为当前工作目录
    if not input_path:
        work_folder = os.getcwd()
    elif os.path.isdir(input_path):
        work_folder = input_path
    else:
        print(f"Error: 路径不存在或不是目录: {input_path}")
        return

    # 选择AI API提供商
    print("\n请选择要使用的AI API提供商:")
    providers = {
        "1": "ollama (本地部署, 默认)",
        "2": "openai (需API密钥)",
        "3": "deepseek (需API密钥)",
        "4": "ali (阿里通义千问, 需API密钥)",
        "5": "siliconflow (硅流智能, 需API密钥)"
    }

    for key, value in providers.items():
        print(f"{key}: {value}")

    provider_choice = input("\n请输入选项编号(默认1): ")
    provider_map = {
        "1": "ollama",
        "2": "openai",
        "3": "deepseek",
        "4": "ali",
        "5": "siliconflow"
    }
    api_provider = provider_map.get(provider_choice, "ollama")

    # 如果不是本地Ollama, 询问API密钥
    api_key = None
    if api_provider != "ollama":
        api_key = input(f"请输入{api_provider} API密钥: ")

    # 选择模型 (根据提供商提供默认值)
    default_models = {
        "ollama": "gemma3:12b",
        "openai": "gpt-3.5-turbo",
        "deepseek": "deepseek-chat",
        "ali": "qwen-max",
        "siliconflow": "Qwen/Qwen2.5-72B-Instruct-128K"
    }

    model_name = input(f"\n请输入模型名称 (默认: {default_models[api_provider]}): ")
    if not model_name:
        model_name = default_models[api_provider]

    # 创建SumSubtitles实例
    subtitles_summarizer = SumSubtitles(
        api_provider=api_provider,
        model_name=model_name,
        api_key=api_key
    )
    subtitles_summarizer.set_work_folder(work_folder)
    possble_dirs = subtitles_summarizer.possble_dirs

    # 显示可用目录, 让用户选择
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
        return

    RESULT = subtitles_summarizer.selected_dirs_handler(index_list)
    if not RESULT:
        subtitles_summarizer.send_message("输入数字不在提供范围, 请重新运行")


def _test_cjk_pdf():
    """测试中文PDF字体渲染问题，创建包含中文内容的PDF文件"""
    # 获取用户输入的路径
    input_path = input("请复制实验文件夹所在目录的绝对路径(若Python代码在同一目录, 请直接按Enter): \n")

    # 判断用户是否直接按Enter, 设置为当前工作目录
    if not input_path:
        work_folder = os.getcwd()
    elif os.path.isdir(input_path):
        work_folder = input_path
    else:
        print(f"Error: 路径不存在或不是目录: {input_path}")
        return

    # 在work_folder中创建测试PDF文件
    print(f"将在 {work_folder} 创建测试PDF文件")

    # 创建PDF文件
    test_pdf_path = os.path.join(work_folder, "test_cjk_pdf.pdf")
    try:
        # 创建PDF文档
        doc = fitz.open()
        current_page = doc.new_page()

        # 准备中文测试文本
        test_text = """# 中文字体渲染测试
                    这是一段测试文本，用于验证PDF中的中文字体渲染是否正常。
                    汉字测试：你好，世界！
                    数字和标点：123，456。789；
                    特殊符号：【】「」『』（）
                    繁体字：漢語測試，繁體中文。This is english"""

        # 创建SumSubtitles类的实例来使用其方法
        subtitles = SumSubtitles()

        # 添加内容
        subtitles._add_content_pages(doc, current_page, test_text, "china-s")

        # 保存PDF
        doc.save(test_pdf_path)
        doc.close()

        print(f"测试PDF已创建 {test_pdf_path}")
        print("请检查PDF中的中文是否正常显示, 没有乱码")

    except Exception as e:
        print(f"创建测试PDF时出错: {e}")


# =========================调试用============================
if __name__ == '__main__':
    main()
    # _test_cjk_pdf()
