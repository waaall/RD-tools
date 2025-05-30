"""
    ===========================README============================
    create date:    20240911
    change date:    20240913
    creator:        zhengxu
    function:       批量变换图片符合指定的四边形

    version:        beta 1.0
    updates:

    details:        opencv的y轴是向下增长, 且图片左上角置于原点, 所以要注意坐标
                    example_twisted_corner =  [
                        [0, 0],     # 原图的左上
                        [w, 0],     # 原图的右上
                        [w, h],     # 原图的右下
                        [0, h]],    # 原图的左下

"""
# =========================用到的库==========================
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# 获取当前脚本所在目录的父目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.files_basic import FilesBasic


# =========================================================
# =======                变换图片形状              =========
# =========================================================
class TwistImgs(FilesBasic):
    def __init__(self,
                 twisted_corner=None,           # 视角角度, 单位:度
                 vertical_angle: float = -30,   # 垂直(低头的角度)
                 horizontal_angle: float = 20,  # 水平(右摆头的角度)

                 # 视线相对位置 [x, y]
                 viewer_pos_x: float = 0,
                 viewer_pos_y: float = 0,

                 log_folder_name: str = 'twist_imgs_log',
                 frame_dpi: int = 200,
                 out_dir_prefix: str = 'twisted-'):
        super().__init__()

        # 视角参数, 初始化顺序不能改变 (因为涉及到方法判断, 这个是默认方法)
        self._init_viewer_pos_angles(viewer_pos_x, viewer_pos_y, vertical_angle, horizontal_angle)

        # 初始化变换后的四角坐标
        self._init_twist_corner(twisted_corner)

        # 视线相对图片左下角位置 [x, y] 总长默认为10, 即最大[9,9]
        # 比如视线在图片中心为[5, 5], 左下角为[0, 9], 右下[9,9]
        self.viewer_pos = [viewer_pos_x, viewer_pos_y]

        # 需要处理的图片类型
        self.suffixs = ['.jpg', '.png', '.jpeg']

        # 设置导出图片dpi
        self.frame_dpi = (frame_dpi, frame_dpi)

        # 设置导出图片文件夹的前缀名 & log文件夹名字
        self.log_folder_name = log_folder_name
        self.out_dir_prefix = out_dir_prefix

    def _init_viewer_pos_angles(self, viewer_pos_x: float, viewer_pos_y: float,
                                vertical_angle: float, horizontal_angle: float):
        # 检查视线位置是否在有效范围内
        if not (0 <= viewer_pos_x <= 9):
            self.send_message(f"Error: Viewer position x must be 0-9 Given: {viewer_pos_x}")
            return False
        if not (0 <= viewer_pos_y <= 9):
            self.send_message(f"Error: Viewer position y must be 0-9 Given: {viewer_pos_y}")
            return False
        # 赋值视线位置
        self.viewer_pos = [viewer_pos_x, viewer_pos_y]

        # 初始化视角参数, 并进行检查
        if not (1 <= abs(vertical_angle) <= 89):
            self.send_message(f"Error: Vertical angle must be 0-89 Given: {vertical_angle}")
            return False
        if not (1 <= abs(horizontal_angle) <= 89):
            self.send_message(f"Error: Horizontal angle must be 0-89 Given: {horizontal_angle}")
            return False
        # 角度在有效范围内, 进行赋值
        self.__vertical_angle = vertical_angle
        self.__horizontal_angle = horizontal_angle

        self.__twi_corner_or_not = False
        return True

    # ==================初始化并验证变换后的四角坐标==================
    def _init_twist_corner(self, twisted_corners):
        """
        example_twisted_corners =  [
                        [0, 0],     # 原图的左上
                        [w, 0],     # 原图的右上
                        [w, h],     # 原图的右下
                        [0, h]],    # 原图的左下
        Parameters:
            twisted_corners (list of list/tuple): 包含四个点的列表, 每个点为 [x, y]。
        Returns:
            bool: 如果验证通过返回True, 否则返回False。
        """
        if not twisted_corners:
            return False

        # 检查twisted_corners是否为列表或元组
        if not isinstance(twisted_corners, (list, tuple)):
            self.send_message("Error: twisted_corners must be a list or tuple of four points.")
            return False

        if len(twisted_corners) != 4:
            self.send_message("Error: twisted_corners must contain exactly four points.")
            return False

        for idx, point in enumerate(twisted_corners):
            if not isinstance(point, (list, tuple)):
                self.send_message(f"Error: Point {idx} in twisted_corners is not a list or tuple.")
                return False
            if len(point) != 2:
                self.send_message(f"Error: Point {idx} in twisted_corners not have 2 coordinates")
                return False

        # 检查是否为四边形
        # One simple way is to check if no three points are colinear and the polygon is convex
        try:
            pts = np.array(twisted_corners, dtype=np.float32)

            # Check if all points are unique
            if len(np.unique(pts, axis=0)) != 4:
                self.send_message("Error: All points in twisted_corners must be unique.")
                return False

            if not self.__is_convex_polygon(pts):
                self.send_message("Error: twisted_corners does not form a convex quadrilateral.")
                return False
        except Exception as e:
            self.send_message(f"Error: Exception occurred while validating twisted_corners: {e}")
            return False

        # 计算原始四边形的最小和最大坐标
        min_x = np.min(pts[:, 0])
        min_y = np.min(pts[:, 1])

        # 如果最小坐标小于零，需要平移整个四边形
        if min_x < 0 or min_y < 0:
            translation = np.array([max(-min_x, 0), max(-min_y, 0)])
            pts += translation  # 对所有点进行平移

            # 更新最小和最大坐标
            min_x = np.min(pts[:, 0])
            min_y = np.min(pts[:, 1])

        # 如果需要将四边形平移到坐标轴上（使最小坐标为零）
        if min_x > 0 or min_y > 0:
            translation = np.array([-min_x, -min_y])
            pts += translation  # 平移四边形，使其最小坐标为零

        # 更新四边形的宽度和高度
        max_x = np.max(pts[:, 0])
        max_y = np.max(pts[:, 1])
        self.quad_width = max_x - min_x
        self.quad_height = max_y - min_y

        self.__twisted_corners = pts
        self.__twi_corner_or_not = True
        return True

    def __is_convex_polygon(self, pts):
        """
        检查给定的四个点是否构成凸四边形
        Parameters:
            pts (np.ndarray): 四个点的坐标,形状为 (4, 2)。
        Returns:
            bool: 如果是凸四边形返回True,否则返回False。
        """
        # Compute the cross product of the edges
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        sign = None
        n = len(pts)
        for i in range(n):
            o = pts[i]
            a = pts[(i + 1) % n]
            b = pts[(i + 2) % n]
            cross = cross_product(o, a, b)
            current_sign = np.sign(cross)
            if current_sign == 0:
                # Points are colinear
                return False
            if sign is None:
                sign = current_sign
            elif sign != current_sign:
                return False
        return True

    # =======================扭转单张图片========================
    def single_file_handler(self, abs_input_path: str, abs_outfolder_path: str):
        # 检查文件路径格式
        if not self.check_file_path(abs_input_path, abs_outfolder_path):
            self.send_message("Error: failed to check_file_path")
            return

        # 读取原始图像
        try:
            image = cv2.imread(abs_input_path)
        except Exception:
            # 如果读取失败, 不抛出异常, 直接返回
            self.send_message(f"Error: failed to read the image「{abs_input_path}」")
            return
        if image is None:
            self.send_message(f"Error: Image is None for「{abs_input_path}」")
            return

        # 检查图像是否具有 alpha 通道
        if image.shape[2] == 3:
            # 将 BGR 图像转换为 BGRA 图像，添加 alpha 通道
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            except Exception:
                self.send_message("Warning: failed to convert to RGBA")
        # 获取图像尺寸
        height, width = image.shape[:2]

        # 定义源点（图片四个角）
        src_pts = np.float32([
            [0, 0],                     # 左上
            [width - 1, 0],             # 右上
            [width - 1, height - 1],    # 右下
            [0, height - 1]             # 左下
        ])

        # 计算变换后图片四角坐标
        if self.__twi_corner_or_not:
            dst_pts = self.__cal_corner_coordinate(height, width)
        else:
            dst_pts = self.__cal_angle_coordinate(height, width, width / 2)

        # 计算透视变换矩阵
        try:
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        except cv2.error as e:
            self.send_message(f"Error: getPerspectiveTransform failed「{abs_input_path}」error: {e}")
            return

        # 计算目标图像尺寸
        dst_width = int(np.max(dst_pts[:, 0]))
        dst_height = int(np.max(dst_pts[:, 1]))

        # 应用透视变换
        try:
            warped_image = cv2.warpPerspective(image, matrix,
                                               (dst_width, dst_height),
                                               borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=(0, 0, 0, 0))
        except Exception as e:
            try:
                self.send_message(f"Warning: {e}")
                warped_image = cv2.warpPerspective(image, matrix, (dst_width, dst_height))
            except cv2.error as e:
                self.send_message(f"Error: warpPerspective failed「{abs_input_path}」 error: {e}")
                return

        # 生成输出文件名
        base_name = os.path.basename(abs_input_path)
        name, _ = os.path.splitext(base_name)
        output_name = f"{name}.png"
        output_path = os.path.join(abs_outfolder_path, output_name)

        # 保存结果
        try:
            cv2.imwrite(output_path, warped_image)
            self.send_message(f"Success: Saved warped image to「{output_path}」")
        except Exception as e:
            self.send_message(f"Error: Failed to save warped image「{output_path}」error: {e}")

    # =============根据四角相对坐标计算变换后图片四角坐标==============
    def __cal_corner_coordinate(self, height, width):
        # twisted_corner 是相对坐标, 需要将其映射到图片的像素坐标

        # 计算缩放比例，保持宽高比不变
        scale_x = width / self.quad_width
        scale_y = height / self.quad_height
        scale = min(scale_x, scale_y)  # 选择较小的缩放比例以适应目标尺寸

        scaled_corners = self.__twisted_corners * scale
        return scaled_corners

    # =============根据视角角度计算变换后图片四角坐标==============
    def __cal_angle_coordinate(self, height, width, focal_length):
        # 定义图片四个角在3D空间中的坐标（假设图片位于Z=0平面）
        # src_pts_3d = np.hstack((src_pts, z_values))
        src_pts_3d = np.float32([
            [0, 0, 0],                      # 左上
            [width, 0, 0],                  # 右上
            [width, height, 0],             # 右下
            [0, height, 0]                  # 左下
        ])

        # 视角角度转换为弧度
        theta_vertical = np.deg2rad(self.__vertical_angle)
        theta_horizontal = np.deg2rad(self.__horizontal_angle)

        # 绕x轴旋转的旋转矩阵 (垂直角度, 抬头的角度)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_vertical), -np.sin(theta_vertical)],
            [0, np.sin(theta_vertical), np.cos(theta_vertical)]
        ])
        # 绕y轴旋转的旋转矩阵 (水平角度, 摇头的角度)
        Ry = np.array([
            [np.cos(theta_horizontal), 0, np.sin(theta_horizontal)],
            [0, 1, 0],
            [-np.sin(theta_horizontal), 0, np.cos(theta_horizontal)]
        ])
        # 总的旋转矩阵: 先水平旋转, 再垂直旋转
        R = Ry @ Rx

        # 相机/viewer的绝对位置计算(映射到像素坐标), viewer_pos = [x, y]
        normalized_x = self.viewer_pos[0] / 9
        normalized_y = self.viewer_pos[1] / 9
        camera_x = normalized_x * width
        camera_y = height - (normalized_y * height)

        # 1.调整坐标系, 将 y 轴反转, 使其与笛卡尔坐标系一致
        src_pts_3d[:, 1] = -src_pts_3d[:, 1]
        # viewer/相机的坐标(注意这里对 camera_y 也反转)
        translation_to_origin = np.array([camera_x, -camera_y, 0])

        # 2.将旋转轴平移到原点
        translated_pts = src_pts_3d - translation_to_origin

        # 3.应用旋转矩阵
        rotated_pts = (R @ translated_pts.T).T

        # 4.将旋转后的点平移回原位置
        rotated_pts += translation_to_origin

        # 5.恢复到原坐标系, 将 y 轴再反转回来
        rotated_pts[:, 1] = -rotated_pts[:, 1]
        points_3d = np.abs(rotated_pts.astype(np.float32))  # 转换为正数(主要是-0)

        # 6. 调整 z 坐标，以考虑相机的位置（焦距）
        points_3d[:, 2] += focal_length

        projected = []
        for point in points_3d:
            if point[2] == 0:
                point[2] = 1e-6  # 防止除以零
            x = (focal_length * point[0]) / point[2]
            y = (focal_length * point[1]) / point[2]
            projected.append([x, y])

        return np.float32(projected)


# =====================main(单独执行时使用)=====================
def main():
    # 获取用户输入的路径
    input_path = input("请复制实验文件夹所在目录的绝对路径(若Python代码在同一目录, 请直接按Enter): \n")

    # 判断用户是否直接按Enter, 设置为当前工作目录
    if not input_path:
        work_folder = os.getcwd()
    elif os.path.isdir(input_path):
        work_folder = input_path

    """
        example_twisted_corner =  [
                        [0, 0],     # 原图的左上
                        [w, 0],     # 原图的右上
                        [w, h],     # 原图的右下
                        [0, h]],    # 原图的左下
    """
    twisted_corner = [[0, 0], [430, 82], [432, 268], [0, 276]]
    img_handler = TwistImgs(twisted_corner=twisted_corner)

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


# =========================调试用============================
if __name__ == '__main__':
    main()
