"""
    ==========================README===========================
    create date:    20241008
    change date:
    creator:        zhengxu
    function:       批量读取bilibili缓存,转换成可以观看的视频
    details:        _data_dir为bilibili缓存视频的文件夹,内部的文件夹为子任务


    #===========本代码子函数参考下面网址=================
    https://github.com/molihuan/BilibiliCacheVideoMergePython
    ===============================================
"""
# =========================用到的库==========================
import os
import sys
from concurrent.futures import ThreadPoolExecutor


from ffmpy import FFmpeg
from json import load

# 获取当前文件所在目录,并加入系统环境变量(临时)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from modules.files_basic import FilesBasic


# =========================================================
# =======               DICOM导出图片              =========
# =========================================================
class BiliVideos(FilesBasic):
    def __init__(self,
                 log_folder_name: str = 'bili_video_handle_log',
                 middle_file_prefix: str = 'out-',
                 out_dir_prefix: str = 'videos-'):
        super().__init__()
        self.middle_file_prefix = middle_file_prefix
        self.suffixs = ['.m4s']

    # 修复音视频文件
    def __fix_m4s(self, path: str, name: str, bufsize: int = 256 * 1024 * 1024) -> None:
        assert bufsize > 0
        file = f"{path}/{name}"
        out_file = f"{path}/{self.middle_file_prefix}{name}"

        try:
            media = open(file, 'rb')
            header = media.read(32)

            # 判断头文件是否符合预期
            if b'000000000' not in header:
                self.send_message(f"Warning: 文件 {file} 的头文件不符合预期")
                return
            new_header = header.replace(b'000000000', b'')

            # 替换完成的文件写入
            out_media = open(out_file, 'wb')
            out_media.write(new_header)

            buf = media.read(bufsize)
            while buf:
                out_media.write(buf)
                buf = media.read(bufsize)

            self.send_message(f"文件修复完成并保存为「{out_file}」")
        except Exception as e:
            self.send_message(f"Error: 修复文件时发生错误: {str(e)}")
        finally:
            media.close()
            out_media.close()

    # 解析json文件, 获取标题, 将其返回
    def _get_title(self, info):
        if not os.path.exists(info):
            self.send_message(f"Warning: info 文件「{info}」不存在")
            return None

        try:
            with open(info, 'r', encoding='utf8') as f:
                info_data = load(f)

            # 检查 info 文件中是否有 'title' 字段
            if 'title' not in info_data:
                self.send_message(f"Warning: info 文件「{info}」中缺少 'title' 字段")
                return None

            title = info_data['title']
            return title
        except Exception as e:
            self.send_message(f"Warning: info文件解析错误: {str(e)}")
            return None

    # 转换合并函数
    def _transform(self, v, a, o):
        if not os.path.exists(v) or not os.path.exists(a):
            self.send_message(f"Error: 视频文件「{v}」或音频文件「{a}」不存在，无法进行转换")
            return False

        try:
            ff = FFmpeg(inputs={v: None, a: None}, outputs={o: '-vcodec copy -acodec copy'})
            print(f"执行转换命令：{ff.cmd}")
            ff.run()
            return True
        except Exception as e:
            self.send_message(f"Error: 音视频合并时发生错误: {str(e)}")
            return False

    # 获取m4s文件名
    def _get_file_name(self, path, suffix):
        files = [f for f in os.listdir(path) if f.endswith(suffix)
                 and not f.startswith(self.middle_file_prefix)]

        if len(files) < 2:
            self.send_message(f"Error: m4s获取文件失败「{path}」")
            return None
        elif len(files) > 2:
            self.send_message(f"Error: 存在多于2个m4s文件「{path}」")
            return None

        if files[0].endswith('-1-30280.m4s'):  # audio文件后缀 '-1-30280.m4s'
            return files
        elif files[1].endswith('-1-30280.m4s'):
            files[0], files[1] = files[1], files[0]
            return files
        else:
            self.send_message(f"Error: 未找到符合条件的音频文件「{path}」")
            return None

    # =====================处理(bilibili缓存文件夹)函数======================
    def _data_dir_handler(self, _data_dir: str):
        outfolder_name = self.out_dir_prefix + _data_dir
        os.makedirs(outfolder_name, exist_ok=True)

        paths = os.listdir(_data_dir)
        # 删除无关文件，仅保留视频所在文件夹
        folders = [p for p in paths if os.path.isdir(os.path.join(_data_dir, p))]
        # 多线程处理单个文件
        max_works = min(self.max_threads, os.cpu_count(), len(folders))
        with ThreadPoolExecutor(max_workers=max_works) as executor:
            for folder in folders:
                abs_input_path = os.path.join(self._work_folder, _data_dir, folder)
                abs_outfolder_path = os.path.join(self._work_folder, outfolder_name)
                executor.submit(self.single_video_handler, abs_input_path, abs_outfolder_path)

    # ========处理单个文件文件夹内的视频和音频，生成一个可播放的视频========
    def single_video_handler(self, abs_input_path: str, abs_outfolder_path: str):
        names = self._get_file_name(abs_input_path, '.m4s')
        if not names:
            return
        self._one_video_handler(abs_input_path, abs_outfolder_path, names)

    # ================处理一对视频&音频，生成一个可播放的视频================
    def _one_video_handler(self, abs_input_path: str, abs_outfolder_path: str, names):
        if len(names) != 2:
            self.send_message("Error: 配对音视频文件个数不对")
            return
        self.__fix_m4s(abs_input_path, names[1])    # 改视频文件
        self.send_message(f"正在处理视频文件：{names[1]}")

        self.__fix_m4s(abs_input_path, names[0])    # 改音频文件
        self.send_message(f"正在处理音频文件：{names[0]}")

        # 名字要与__fix_m4s函数中out_file一致
        video = f"{abs_input_path}/{self.middle_file_prefix}{names[1]}"
        audio = f"{abs_input_path}/{self.middle_file_prefix}{names[0]}"

        info = abs_input_path + '/videoInfo.json'

        # 获取视频名称
        title = self._get_title(info)
        if title is None:
            # 如果 _get_title 返回 None，使用 names[1] 去掉后缀作为备用名称
            title = os.path.splitext(names[1])[0]
        else:
            print(f"视频名称解析成功: {title}")
        out_video = os.path.join(abs_outfolder_path, title + '.mp4')

        # 合成音视频
        SUCCESS = self._transform(video, audio, out_video)
        if SUCCESS is True:
            self.send_message(f"SUCCESS: 「{title}」")


# =====================main(单独执行时使用)=====================
def main():
    # 获取用户输入的路径
    input_path = input("请复制实验文件夹所在目录的绝对路径(若Python代码在同一目录, 请直接按Enter): \n")

    # 判断用户是否直接按Enter，设置为当前工作目录
    if not input_path:
        work_folder = os.getcwd()
    elif os.path.isdir(input_path):
        work_folder = input_path

    bili_videos_generator = BiliVideos()
    bili_videos_generator.set_work_folder(work_folder)
    possble_dirs = bili_videos_generator.possble_dirs

    # 给用户显示，请用户输入index
    number = len(possble_dirs)
    bili_videos_generator.send_message('\n')
    for i in range(number):
        print(f"{i}: {possble_dirs[i]}")
    user_input = input("\n请选择要处理的序号(用空格分隔多个序号): \n")

    # 解析用户输入
    try:
        indices = user_input.split()
        index_list = [int(index) for index in indices]
    except ValueError:
        bili_videos_generator.send_message("输入错误, 必须输入数字")

    RESULT = bili_videos_generator.selected_dirs_handler(index_list)
    if not RESULT:
        bili_videos_generator.send_message("输入数字不在提供范围, 请重新运行")


# =====================文件失效了,=====================
def pair_mode():

    work_folder = "/Volumes/zxTF256/neoScience_work/stm32"

    bili_videos_generator = BiliVideos()
    bili_videos_generator.set_work_folder(work_folder)

    abs_in_path = '/Volumes/zxTF256/neoScience_work/stm32/m4s'
    abs_out_path = '/Volumes/zxTF256/neoScience_work/stm32/stm32教程铁头山羊第四版'
    file_names = bili_videos_generator._get_filenames_by_suffix(abs_in_path)

    pairs_label = ['30280.m4s', '30080.m4s', '100050.m4s', '30640.m4s']
    pairs = bili_videos_generator.pair_files(pairs_label, file_names, False)

    max_works = min(8, os.cpu_count(), len(pairs))
    with ThreadPoolExecutor(max_workers=max_works) as executor:
        for pair in pairs:
            parts = pair[0].rsplit('-', 1)
            base_name = parts[0]
            video = os.path.join(abs_in_path, pair[1])
            audio = os.path.join(abs_in_path, pair[0])
            out_v = os.path.join(abs_out_path, base_name + '.mp4')
            executor.submit(bili_videos_generator._transform, video, audio, out_v)


# =========================调试用============================
if __name__ == '__main__':
    # main()
    pair_mode()
