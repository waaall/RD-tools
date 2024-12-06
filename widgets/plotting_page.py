"""
    ==========================README===========================
    create date:    20241011
    change date:    20241122
    creator:        zhengxu
    function:       串口通信数据画图

    version:        beta 3.0
    updates:

    details:
                    1. 选择端口号页面, 并且之后开始按钮绑定初始化serial_plot
"""
import random
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# from PySide6.QtGui import *
# # QPixmap, QIcon, QImage
from PySide6.QtCore import *
# QFile, QFileInfo, QPoint, QSettings, QSaveFile, Qt, QTimeLine
from PySide6.QtWidgets import *
# QAction, QApplication, QFileDialog, QMainWindow, QMessageBox, QLineEdit, QWidget
from pyqtgraph import PlotWidget, mkPen

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from modules.serial_plot import SerialPlot  # 引入 SerialPlot 类


# =========================================================
# =======                图表制作界面               =========
# =========================================================
class PlottingWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 设置主窗口
        self.setWindowTitle("波形显示系统")
        self.setGeometry(100, 100, 1000, 600)

        # 创建布局
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # 创建波形显示区域
        left_layout = QVBoxLayout()
        self.plot_widget = PlotWidget()  # 单一波形显示

        # 初始化波形曲线
        self.curve = self.plot_widget.plot(pen=mkPen('r'))

        # 添加到左侧布局
        left_layout.addWidget(self.plot_widget)

        # 创建右侧按钮区域
        right_layout = QVBoxLayout()
        button_start = QPushButton("开始")
        button_stop = QPushButton("停止")

        # 按钮信号槽
        button_start.clicked.connect(self.start_plot)
        button_stop.clicked.connect(self.stop_plot)

        # 将按钮添加到右侧布局
        right_layout.addWidget(button_start)
        right_layout.addWidget(button_stop)

        # 将左侧波形和右侧按钮布局添加到主布局
        main_layout.addLayout(left_layout, stretch=3)  # 左侧占大部分空间
        main_layout.addLayout(right_layout, stretch=1)

        # 数据缓冲区
        self.data_buffer = []
        self.max_buffer_size = 100

        # 线程池，用于非阻塞串口通信
        self.executor = ThreadPoolExecutor(max_workers=1)

    def _init_serial_plot(self) -> None:
        # 初始化 SerialPlot 对象 # 修改为实际串口和数据类型
        self.serial_plot = SerialPlot(port="COM3", baudrate=9600, data_type="uint16")
        self.serial_plot.data_packet_signal.connect(self.update_plot)

    def update_plot(self, data):
        """
        槽函数：接收串口数据并更新波形显示。
        :param data: 接收到的解析后的数据列表
        """
        print(f"Received data: {data}")

        # 更新数据缓冲区
        self.data_buffer.extend(data)
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer = self.data_buffer[-self.max_buffer_size:]

        # 更新波形
        self.curve.setData(self.data_buffer)

    def stop_plot(self):
        """
        停止波形更新。
        """
        self.executor.shutdown(wait=False)

    def start_plot(self):
        """
        开始波形更新。
        """
        self.executor.submit(self.serial_plot.start_listening)  # 在线程池中运行串口监听任务


# ===============调试用==================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    trial = PlottingWindow()
    trial.show()
    sys.exit(app.exec())
