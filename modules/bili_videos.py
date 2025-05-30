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
                 out_dir_prefix: str = 'fixed-',
                 AddGroupTitle: bool = True,
                 GroupTitleMaxLength: int = 10):
        super().__init__()

        # 设置导出文件夹的前缀名 & log文件夹名字
        self.log_folder_name = log_folder_name
        self.out_dir_prefix = out_dir_prefix
        self.AddGroupTitle = AddGroupTitle

        # 验证GroupTitleMaxLength参数范围
        if not isinstance(GroupTitleMaxLength, int):
            GroupTitleMaxLength = 10
            self.send_message("GroupTitleMaxLength不是整数类型, 已设为10")
        if GroupTitleMaxLength <= 5 or GroupTitleMaxLength > 20:
            GroupTitleMaxLength = 10
            self.send_message("GroupTitleMaxLength不在5到20之间, 已设为10")
        self.GroupTitleMaxLength = GroupTitleMaxLength

        self.suffixs = ['.m4s']
        # Windows 文件系统不支持的字符: \ / : * ? " < > |
        self._invalid_chars = ['\\', '/', ':', '*', '?', '？', '。', '，',
                               '"', '<', '>', '|', '"', '"', '：', '`', '·']

    # 修复音视频文件
    def __fix_m4s(self, path: str, name: str, bufsize: int = 256 * 1024 * 1024) -> None:
        assert bufsize > 0
        file = f"{path}/{name}"
        out_file = f"{path}/{self.out_dir_prefix}{name}"

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
        except Exception as e:
            self.send_message(f"Warning: info文件解析错误: {str(e)}")
            return None

        title = ''

        # 检查 info 文件中是否有 'groupTitle' 字段
        if 'groupTitle' not in info_data:
            self.send_message(f"Warning: info 文件「{info}」中缺少 'groupTitle' 字段")
        elif self.AddGroupTitle:
            # 1. 删除不能作为文件名的特殊字符
            group_title = info_data['groupTitle']
            for char in self._invalid_chars:
                group_title = group_title.replace(char, '')

            # 2. 限制groupTitle长度
            if len(group_title) > self.GroupTitleMaxLength:
                group_title = group_title[:self.GroupTitleMaxLength]
                self.send_message(f"提示: groupTitle 过长，已截断为 '{group_title}'")

            title += group_title
            # 检查 info 文件中是否有 'p' 字段
            if 'p' not in info_data:
                self.send_message(f"Warning: info 文件「{info}」中缺少 'p' 字段")
            else:
                p_str = '-' + str(info_data['p']) + '-'
                title += p_str
        else:
            self.send_message("按照设置不添加groupTitle")

        # 检查 info 文件中是否有 'title' 字段
        if 'title' not in info_data:
            self.send_message(f"Warning: info 文件「{info}」中缺少 'title' 字段")
        else:
            # 获取title
            item_title = info_data['title']

            # 2. 删除不能作为文件名的特殊字符
            for char in self._invalid_chars:
                item_title = item_title.replace(char, '')

            # # 3. 删除title中与groupTitle重复三个以上的字符
            # if 'groupTitle' in info_data and len(info_data['groupTitle']) >= 3:
            #     group_title = info_data['groupTitle']
            #     # 查找group_title中长度超过3的子串在item_title中的位置
            #     for i in range(len(group_title) - 2):
            #         substr = group_title[i:i+3]  # 取三个字符的子串
            #         if substr in item_title:
            #             item_title = item_title.replace(substr, '')
            #             self.send_message(f"提示: 移除了title中与groupTitle重复的子串 '{substr}'")

            title += item_title
        return title

    # 转换合并函数
    def _transform(self, v, a, o):
        if not os.path.exists(v) or not os.path.exists(a):
            self.send_message(f"Error: 视频文件「{v}」或音频文件「{a}」不存在，无法进行转换")
            return False

        try:
            # 添加-y参数以自动覆盖已存在的文件
            # 添加-loglevel warning以减少输出
            # 添加-threads 0让FFmpeg自动决定线程数
            ff = FFmpeg(
                inputs={v: None, a: None},
                outputs={o: '-vcodec copy -acodec copy -y -loglevel warning -threads 0'},
                global_options='-hide_banner'
            )
            print(f"执行转换命令：{ff.cmd}")
            ff.run()
            return True
        except Exception as e:
            self.send_message(f"Error: 音视频合并时发生错误: {str(e)}")
            return False

    # 获取m4s文件名
    def _get_file_name(self, path, suffix):
        files = [f for f in os.listdir(path) if f.endswith(suffix)
                 and not f.startswith(self.out_dir_prefix)]

        if len(files) < 2:
            self.send_message(f"Error: m4s获取文件失败「{path}」")
            return None
        elif len(files) > 2:
            self.send_message(f"Error: 存在多于2个m4s文件「{path}」")
            return None

        if files[0].endswith('-1-30280.m4s'):   # audio文件后缀 '-1-30280.m4s'
            return files
        elif files[1].endswith('-1-30280.m4s'):
            files[0], files[1] = files[1], files[0]
            return files
        elif files[0].endswith('-1-30216.m4s'):  # 或者audio文件后缀 '-1-30216.m4s'
            return files
        elif files[1].endswith('-1-30216.m4s'):
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

        if not folders:
            self.send_message(f"Error: No video folders found in {_data_dir}")
            return

        self.send_message(f"找到{len(folders)}个视频文件夹，开始处理")

        # 计算合适的线程数 - 视频处理比较消耗I/O和CPU，调整线程数
        max_works = min(min(self.max_threads, 6), os.cpu_count(), len(folders))
        with ThreadPoolExecutor(max_workers=max_works) as executor:
            futures = []
            for folder in folders:
                abs_input_path = os.path.join(self._work_folder, _data_dir, folder)
                abs_outfolder_path = os.path.join(self._work_folder, outfolder_name)
                futures.append(executor.submit(self.single_video_handler, abs_input_path, abs_outfolder_path))

            # 等待所有任务完成并处理结果
            for future in futures:
                try:
                    future.result()  # 获取任务结果
                except Exception as e:
                    self.send_message(f"Error: 视频处理失败: {str(e)}")

        self.send_message(f"所有视频处理完成，输出目录：{outfolder_name}")

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

        # 获取视频名称
        info = abs_input_path + '/videoInfo.json'
        title = self._get_title(info)
        if title is None:
            # 如果 _get_title 返回 None，使用 names[1] 去掉后缀作为备用名称
            title = os.path.splitext(names[1])[0]
        else:
            print(f"视频名称解析成功: {title}")
        out_video = os.path.join(abs_outfolder_path, title + '.mp4')

        # 检查输出文件是否已经存在
        if os.path.exists(out_video):
            self.send_message(f"注意: 输出文件「{out_video}」已存在，将被覆盖")

        # 修复视频和音频文件
        self.__fix_m4s(abs_input_path, names[1])    # 改视频文件
        self.send_message(f"正在处理视频文件：{names[1]}")

        self.__fix_m4s(abs_input_path, names[0])    # 改音频文件
        self.send_message(f"正在处理音频文件：{names[0]}")

        # 名字要与__fix_m4s函数中out_file一致
        video = f"{abs_input_path}/{self.out_dir_prefix}{names[1]}"
        audio = f"{abs_input_path}/{self.out_dir_prefix}{names[0]}"

        # 合成音视频
        SUCCESS = self._transform(video, audio, out_video)
        if SUCCESS is True:
            self.send_message(f"SUCCESS: 「{title}」")
        else:
            self.send_message(f"Error: 「{title}」音视频合并失败")


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
    main()
    # pair_mode()
