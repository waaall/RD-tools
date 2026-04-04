"""
    ===========================README===========================
    create date:    20240908
    change date:    20240913
    creator:        zhengxu
    function:       批量分离图片色彩

    version:        beta 3.0
    updates:        修改了基类FilesBasic,做出了相应调整

"""

# =========================用到的库==========================
import os
from PIL import Image

from modules.files_basic import FilesBasic


# =========================================================
# =======                分离色彩通道              =========
# =========================================================
class SplitColors(FilesBasic):
    def __init__(self,
                 log_folder_name: str = 'split_colors_log',
                 frame_dpi: int = 200,
                 colors=None,
                 out_dir_prefix: str = 'split-'):

        super().__init__()
        self.init_colors(colors)

        # 需要处理的图片类型
        self.suffixs = ['.jpg', '.png', '.jpeg']

        # 设置导出图片文件夹的前缀名 & log文件夹名字
        self.log_folder_name = log_folder_name
        self.out_dir_prefix = out_dir_prefix

        # 设置导出图片dpi
        self.frame_dpi = (frame_dpi, frame_dpi)

    def init_colors(self, colors):
        # 定义可能的色彩通道, RGB顺序不能变, 跟pillow处理有关
        self.__default_colors = ['R', 'G', 'B']

        # 设置分离的色彩，如果 colors 不为 None 且合法则使用，否则使用默认的 RGB
        if colors is None:
            self._colors = self.__default_colors
        else:
            # 检查传入的 colors 是否有效
            invalid_colors = [c for c in colors if c not in self.__default_colors]
            if invalid_colors:
                self.send_message(f"Warning: can not init colors, set to {self.__default_colors}")
                self._colors = self.__default_colors
            else:
                self._colors = colors

    # ======================分离单张图片色彩======================
    def single_file_handler(self, abs_input_path: str, abs_outfolder_path: str):
        # 检查文件路径格式
        if not self.check_file_path(abs_input_path, abs_outfolder_path):
            self.send_message("Error: failed to check_file_path")
            return
        # 打开图片并分离色彩通道
        with Image.open(abs_input_path) as img:
            # 分离出红、绿、蓝通道;img_clrs 是一个包含 (R, G, B) 的元组
            try:
                if img.mode != 'RGB':
                    self.send_message(f"Warning: try to convert to RGB for「{img.mode}」")
                    img = img.convert('RGB')

                img_clrs = img.split()
            except Exception as e:
                self.send_message(f"Error: failed to convert to RGB for :{str(e)}")
                return

            # 创建空图像模板,全黑的灰度图像,大小与原图相同
            black = Image.new('L', img.size)

            for color in self._colors:
                # 获取通道索引
                color_index = self.__default_colors.index(color)

                # 创建RGB组合, 将其他通道设为黑色
                channels = [(channel if i == color_index else black)
                            for i, channel in enumerate(img_clrs)]

                # 合并通道并保存
                out_img = Image.merge('RGB', tuple(channels))
                img_name = os.path.basename(abs_input_path)
                abs_out_path = os.path.join(abs_outfolder_path, f"{color}-{img_name}")
                out_img.save(abs_out_path, dpi=self.frame_dpi)

            self.send_message(f"Saved splited images: {img_name}")


if __name__ == '__main__':
    from core.task_cli import run_task_cli

    raise SystemExit(run_task_cli('split-colors', operation_cls=SplitColors))
