"""
    ==========================README===========================
    create date:    20241120
    change date:    20241122
    creator:        zhengxu
    function:       串口通信数据画图/协议定义

    version:        beta 2.0
    updates:

    details:
"""
import os
import struct
import sys

from PySide6.QtCore import Signal

# 获取当前文件所在目录,并加入系统环境变量(临时)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from modules.serial_com import Com_Driver


# =========================================================
# =======                串口通信画图              =========
# =========================================================
class SerialPlot(Com_Driver):
    data_packet_signal = Signal(list)  # 自定义信号，传递数据包

    def __init__(self, port: str, baudrate: int = 9600, data_type: str = "uint8"):
        super().__init__()

        # 初始化串口通信配置
        self.port = port
        self.baudrate = baudrate

        # 数据类型格式映射表
        self.__data_format_map = {
            'uint8': {
                "format_char": 'B',
                "length": 1,
            },
            'uint16': {
                "format_char": 'H',
                "length": 2,
            },
            'float': {
                "format_char": 'f',
                "length": 4,
            },
            'char': {
                "format_char": 'c',
                "length": 1,
            },
            'char_hex': {
                "format_char": 's',
                "length": 1,
            },
        }

        # 验证传入的数据类型
        if data_type not in self.__data_format_map:
            raise ValueError(f"Unsupported data type: {data_type}")

        # 设置数据解析格式
        self.data_type = data_type
        self.format_char = self.__data_format_map[data_type]["format_char"]
        self.data_length = self.__data_format_map[data_type]["length"]

    def _received_data_handler(self, result: bool, length: int, received_data: bytes) -> None:
        """
        数据帧解析处理函数。
        """
        if result:
            try:
                # 动态解析数据包
                data = list(struct.unpack(f"<{length // self.data_length}{self.format_char}", received_data[:length]))
                self.data_packet_signal.emit(data)  # 发出信号
            except Exception as e:
                print(f"Error while parsing data: {e}")
        else:
            print("Invalid data received.")


# =========================调试用============================
def main():
    pass


if __name__ == '__main__':
    main()
