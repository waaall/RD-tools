##==========================README===========================##
"""
    create date:    20241008 
    change date:    
    creator:        zhengxu
    function:       批量读取bilibili缓存,转换成可以观看的视频
    details:        _data_dir为bilibili缓存视频的文件夹,内部的文件夹为子任务


    #===========本代码子函数参考下面网址=================
    https://github.com/molihuan/BilibiliCacheVideoMergePython
    ===============================================
"""
##=========================用到的库==========================##
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import os
from ffmpy import FFmpeg
from json import load

# 获取当前文件所在目录,并加入系统环境变量(临时)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from modules.files_basic import FilesBasic

##=========================================================
##=======               DICOM导出图片              =========
##=========================================================
class BiliVideos(FilesBasic):
    def __init__(self, 
                 log_folder_name :str = 'bili_video_handle_log',
                 out_dir_suffix :str = 'videos-'):
        super().__init__()

    def __fix_m4s(path: str, name: str, bufsize: int = 256*1024*1024) -> None:
        assert bufsize > 0
        file = f"{path}/{name}"
        out_file = f"{path}/o{name}"
    
        media = open(file, 'rb')
        header = media.read(32)
        new_header = header.replace(b'000000000', b'')
        # new_header = new_header.replace(b'$', b' ')
        # new_header = new_header.replace(b'avc1', b'')
        out_media = open(out_file, 'wb')
        out_media.write(new_header)
        buf = media.read(bufsize)
        while buf:
            out_media.write(buf)
            buf = media.read(bufsize)
    
    # 解析json文件, 获取标题, 将其返回
    def __get_title(info):
        f = open(info,'r',encoding='utf8')
        info_data = load(f)
        title = info_data['title']
        print(f"该视频为：\n\t{title}\n")
        return title
    
    # 转换合并函数
    def __transform(v,a,o):
        ff = FFmpeg(inputs={v:None,a:None},outputs={o:'-vcodec copy -acodec copy'})
        print(ff.cmd)
        ff.run()
    
    def __get_file_name(path, suffix):
        files = [f for f in os.listdir(path) if f.endswith(suffix)]
    
        if files[0].endswith('-1-30280.m4s'): #audio文件后缀 '-1-30280.m4s'   
            return files
        elif files[1].endswith('-1-30280.m4s'):
            files[0], files[1] = files[1], files[0]
            return files
        elif len(files) == 0:
            return files
        else:    
            raise ValueError('获取文件失败')

    ##=====================处理(bilibili缓存文件夹)函数======================##
    def _data_dir_handler(self, _data_dir:str):
        outfolder_name = self.out_dir_suffix + _data_dir
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

    ##========处理单个文件文件夹内的视频和音频，生成一个可播放的视频========##
    def single_video_handler(self, abs_input_path:str, abs_outfolder_path:str):
        names = self.__get_file_name(abs_input_path, '.m4s')
        if len(names) == 2:
            self.__fix_m4s(abs_input_path, names[1]) #改视频文件
            self.__fix_m4s(abs_input_path, names[0]) #改音频文件

            video = f"{abs_input_path}/o{names[1]}"
            audio = f"{abs_input_path}/o{names[0]}"

            info = abs_input_path + '/videoInfo.json'
            out_video = os.path.join(abs_outfolder_path, self.__get_title(info) + '.mp4')
            
            self.__transform(video, audio, out_video) #合成音视频

##=====================main(单独执行时使用)=====================
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

##=========================调试用============================
if __name__ == '__main__':
    main()

