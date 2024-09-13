##===========================README============================##
""" 
    create date:    20240911 
    change date:    20240913
    creator:        zhengxu
    function:       批量扭转图片符合指定的四边形

    version:        beta 1.0
    updates:  

"""
##=========================用到的库==========================##
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# 获取当前脚本所在目录的父目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.files_basic import FilesBasic
##=========================================================
##=======                  扭转图片                =========
##=========================================================
class TwistImgs(FilesBasic):
    def __init__(self, 
                 vertical_angle: float = 30,    # 垂直角度, 单位: 度
                 horizontal_angle: float = 30,  # 水平角度, 单位: 度
                 viewer_pos: list = [0, 0],     # 视线相对位置 [x, y]
                 log_folder_name :str = 'twist_colors_log', 
                 frame_dpi :int = 200,
                 out_dir_suffix :str = 'twist-'):
        
        super().__init__()

        # 视角参数
        self.vertical_angle = vertical_angle
        self.horizontal_angle = horizontal_angle
        
        # 视线相对图片左下角位置 [x, y] 总长默认为[10,10]
        # 比如视线在图片中心的话, [x, y] = [5, 5]
        self.viewer_pos = viewer_pos

        # 需要处理的图片类型
        self.suffixs = ['.jpg', '.png', '.jpeg'] 

        # 设置导出图片dpi
        self.frame_dpi = (frame_dpi, frame_dpi)

        # 设置导出图片文件夹的前缀名 & log文件夹名字
        self.log_folder_name = log_folder_name
        self.out_dir_suffix = out_dir_suffix

    ##=======================扭转单张图片========================##
    def single_file_handler(self, abs_input_path:str, abs_outfolder_path:str):
        # 检查文件路径格式
        if not self.check_file_path(abs_input_path, abs_outfolder_path):
            self.send_message("Error: failed to check_file_path")
            return

        # 读取原始图像
        try:
            image = cv2.imread('path_to_your_image.jpg')
        except Exception:
            # 如果读取失败, 不抛出异常, 直接返回
            self.send_message(f"Error: failed to read the image「{abs_input_path}」")
            return
        if image is None:
            self.send_message(f"Error: Image is None for「{abs_input_path}」")
            return
        # 获取图像尺寸
        height, width = image.shape[:2]

        # 计算焦距, 使得输出图像的宽度与原图一致
        # focal_length = (image_width / 2) / tan(horizontal_angle / 2)
        horizontal_fov = self.horizontal_angle  # 水平视场角, 单位: 度
        focal_length = (width / 2) / np.tan(np.deg2rad(horizontal_fov / 2))

        # 定义图片四个角在3D空间中的坐标（假设图片位于Z=0平面）
        # 原点(0,0,0)在图片左上角, X轴向右, Y轴向下
        corners_3d = np.float32([
            [0, 0, 0],                # 左上
            [width, 0, 0],            # 右上
            [width, height, 0],       # 右下
            [0, height, 0]            # 左下
        ])

        # 视角角度转换为弧度
        theta_vertical = np.deg2rad(self.vertical_angle)
        theta_horizontal = np.deg2rad(self.horizontal_angle)

        # 绕x轴旋转的旋转矩阵（垂直角度）
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_vertical), -np.sin(theta_vertical)],
            [0, np.sin(theta_vertical),  np.cos(theta_vertical)]
        ])

        # 绕y轴旋转的旋转矩阵（水平角度）
        Ry = np.array([
            [ np.cos(theta_horizontal), 0, np.sin(theta_horizontal)],
            [0, 1, 0],
            [-np.sin(theta_horizontal), 0, np.cos(theta_horizontal)]
        ])

        # 总的旋转矩阵: 先水平旋转, 再垂直旋转
        R = Ry @ Rx

        # 相机绝对位置计算, viewer_pos = [x, y],范围 [0, 10],相对于图片左下角
        # 将 [x, y] 映射到像素坐标
        normalized_x = self.viewer_pos[0] / 10  # 范围 [0, 1]
        normalized_y = self.viewer_pos[1] / 10  # 范围 [0, 1]
        camera_x = normalized_x * width        # 映射到 [0, width]
        camera_y = height - (normalized_y * height)  # 映射到 [0, height],因为 y=0 在顶部

        camera_position = np.array([camera_x, camera_y, focal_length])

        # 转换和投影
        transformed_points = self.__transform_points(corners_3d, R, camera_position)
        projected_points = self.__project_points(transformed_points, focal_length)

        # 平移投影后的点, 使其全部为正值
        min_x = np.min(projected_points[:, 0])
        min_y = np.min(projected_points[:, 1])
        projected_points[:, 0] -= min_x
        projected_points[:, 1] -= min_y

        # 转换为浮点数坐标
        dst_pts = projected_points.astype(np.float32)

        # 定义源点（图片四个角）
        src_pts = np.float32([
            [0, 0],                # 左上
            [width - 1, 0],        # 右上
            [width - 1, height - 1], # 右下
            [0, height - 1]        # 左下
        ])

        # 计算透视变换矩阵
        try:
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        except cv2.error as e:
            self.send_message(f"Error: getPerspectiveTransform failed for「{abs_input_path}」 with error: {e}")
            return

        # 计算目标图像尺寸
        dst_width = int(np.max(dst_pts[:, 0]))
        dst_height = int(np.max(dst_pts[:, 1]))

        # 应用透视变换
        try:
            warped_image = cv2.warpPerspective(image, matrix, (dst_width, dst_height))
        except cv2.error as e:
            self.send_message(f"Error: warpPerspective failed for「{abs_input_path}」 with error: {e}")
            return

        # 生成输出文件名
        base_name = os.path.basename(abs_input_path)
        name, ext = os.path.splitext(base_name)
        output_name = f"{self.out_dir_suffix}{name}{ext}"
        output_path = os.path.join(abs_outfolder_path, output_name)

        # 保存结果
        try:
            cv2.imwrite(output_path, warped_image)
            self.send_message(f"Success: Saved warped image to「{output_path}」")
        except Exception as e:
            self.send_message(f"Error: Failed to save warped image to「{output_path}」 with error: {e}")
            return

    #====================将3D点转换到相机坐标系====================
    def __transform_points(points, R, camera_position):
        # 平移
        translated = points - camera_position
        # 旋转
        rotated = R @ translated.T
        return rotated.T

    #=========================透视投影=========================
    def __project_points(points_3d, focal_length):
        """
        透视投影, 将3D点投影到2D平面
        """
        projected = []
        for point in points_3d:
            if point[2] == 0:
                point[2] = 1e-6  # 防止除以零
            x = (focal_length * point[0]) / point[2]
            y = (focal_length * point[1]) / point[2]
            projected.append([x, y])
        return np.float32(projected)

##=====================main(单独执行时使用)=====================
def main():
    # 获取用户输入的路径
    input_path = input("请复制实验文件夹所在目录的绝对路径(若Python代码在同一目录, 请直接按Enter): \n")
    
    # 判断用户是否直接按Enter, 设置为当前工作目录
    if not input_path:
        work_folder = os.getcwd()
    elif os.path.isdir(input_path):
        work_folder = input_path
    
    img_handler = TwistImgs()
    img_handler.set_work_folder(work_folder)
    possble_dirs = img_handler.possble_dirs
    
    # 给用户显示, 请用户输入index
    number = len(possble_dirs)
    img_handler.send_message('\n')
    for i in range(number):
        print(f"{i}: {possble_dirs[i]}")
    user_input = input("\n请选择要处理的序号(用空格分隔多个序号): \n")
    
    # 解析用户输入
    try:
        indices = user_input.split()
        index_list = [int(index) for index in indices]
    except ValueError:
        img_handler.send_message("输入错误, 必须输入数字")

    RESULT = img_handler.selected_dirs_handler(index_list)
    if not RESULT:
        img_handler.send_message("输入数字不在提供范围, 请重新运行")

##=========================调试用============================
if __name__ == '__main__':
    main()
