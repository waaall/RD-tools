"""
    ===========================README============================
    create date:    20241011
    change date:    20241025
    creator:        zhengxu

    function:       1. 定义基本的通信硬件接口
                    2. 定义数据帧的结构, 封装/解析

    version:        beta0.3
    updates:

    details:        此Com_Driver类为通信基类, 与通信协议无关, 分为三大部分:
                        1. 初始化函数
                        2. 监听函数
                        3. 发送函数
"""
# =========================用到的库==========================
import struct
import time
# import zlib  # 用于 CRC 校验
# from enum import Enum
from typing import Optional

import serial
from serial.tools import list_ports
from PySide6.QtCore import QObject, Signal


# =========================================================
# =======                通信基类                 =========
# =========================================================
class Com_Driver(QObject):
    data_received_signal = Signal(bool, int)    # 定义信号，传递缓冲区数据
    result_signal = Signal(bool, int)           # 定义信号，结果显示

    def __init__(self, port: str,
                 baudrate: int = 9600,
                 timeout: float = 0.1,
                 bytesize: int = serial.EIGHTBITS,
                 parity: str = serial.PARITY_NONE,
                 stopbits: float = serial.STOPBITS_TWO):
        """
        初始化 Pacemaker_Com 类, 配置串口通信参数。
        :param port: 串口端口
        :param baudrate: 串口波特率, 默认9600
        :param timeout: 串口超时时间, 默认0.1秒
        :param bytesize: 数据位长度, 默认8位
        :param parity: 校验位, 默认无校验
        :param stopbits: 停止位长度, 默认2位
        """
        super().__init__()

        self.ser: Optional[serial.Serial] = None
        self.port: str = port
        self.baudrate: int = baudrate
        self.timeout: float = timeout
        self.bytesize: int = bytesize
        self.parity: str = parity
        self.stopbits: float = stopbits

        # 初始化连接参数
        self.__init_connection_parms()

        # 初始化通信
        if self.com_connect():
            self.send_message(True, "success: connected")
        else:
            self.send_message(False, "Error: connect failed, try ctl mode")
            try:
                self.start_com_ctl()
            except Exception:
                self.send_message(False, "Error: connect failed, try ctl mode")
                return

    # =========================================================
    # =======               通信初始化函数              =========
    # =========================================================
    def com_connect(self) -> bool:
        """
        初始化通信, 打开指定的端口, 配置所有相关参数。
        如果无法打开, 会捕获异常并输出错误信息。
        """
        if self.ser and self.ser.is_open:
            print(f"串口 {self.port} 已经处于打开状态。")
            return True
        try:
            self.ser = serial.Serial(
                port=self.port,                  # 串口端口
                baudrate=self.baudrate,          # 波特率
                timeout=self.timeout,            # 超时时间
                bytesize=self.bytesize,          # 数据位长度
                parity=self.parity,              # 校验位
                stopbits=self.stopbits           # 停止位
            )

            print(f"""串口 {self.port} 已打开,
                  波特率为 {self.baudrate},
                  数据位为 {self.bytesize},
                  校验位为 {self.parity},
                  停止位为 {self.stopbits}.""")
            return True

        except serial.SerialException as e:
            self.send_message(False, f"打开串口时出错: {e}")
            self.ser = None
            return False

    def __init_connection_parms(self):
        # 初始化必要参数
        self.__FRAME_HEADER: int = 0xEF
        self._MAX_FRAME_LENGTH: int = 32
        self._MIN_FRAME_LENGTH: int = 5  # 最小帧长
        self._MAX_LISTENING_TIME: float = 10.0
        self._RX_REPEAT_INTERVAL: float = 0.1
        self._RX_IDLE_THRESHOLD: float = 0.5
        self._is_listening: bool = False

        # 接收数据缓冲区
        self.__received_buffer = bytearray()

        # 发送数据缓冲区,要指定大小,不然放不进去
        self._send_buffer = bytearray(self._MAX_FRAME_LENGTH)

        # 接收的数据（验证并去掉帧头帧尾&校验位）
        self._received_data = bytearray()

        # 连接信号到槽函数
        self.data_received_signal.connect(self._received_data_handler)

    def com_disconnect(self) -> None:
        """
        关闭通信。
        """
        if self.ser:
            self._send_buffer.clear()
            self.stop_listening()
            self.ser.close()
            print(f"串口 {self.port} 已关闭。")

    # 下面几个是CTL模式的连接
    def start_com_ctl(self) -> bool:
        """启动串口通信"""
        self.__find_comports()
        if not self.__select_comport():
            print("设备尚未指定!")
            return False
        if not self.com_connect():
            print("Error: ctl mode failed")
            return False
        return True

    def __find_comports(self) -> list:
        """查找串口设备，并保存到 `self.devices` 中

        Returns:
            `list`: 设备列表
        """
        ports = list_ports.comports()  # 查找当前连接的所有串口设备
        devices = [info for info in ports]  # 提取设备信息
        self.__devices = devices
        return devices

    def __select_comport(self, *, devices=[]) -> bool:
        """从设备列表中选择一个设备

        Args (option):
            `devices`: 设备列表

        Returns:
            `bool`: 成功保存选中的设备则返回True
        """
        if devices == []:
            _devices = self.__devices
            _num_devices = len(self.__devices)
        else:
            _devices = devices
            _num_devices = len(devices)

        if _num_devices == 0:    # 没有设备
            print("未找到设备")
            return False
        elif _num_devices == 1:  # 只有一个设备
            print("仅找到设备: %s" % _devices[0])
            self.port = _devices[0].device
            return True
        else:                    # 多个设备
            print("已连接的串口设备如下:")
            for i in range(_num_devices):
                print("%d : %s" % (i, _devices[i]))

            _inp_num = input("输入目标端口号 >> ")

            if not _inp_num.isdecimal():  # 检查输入是否为数字
                print("%s 不是一个数字!" % _inp_num)
                return False
            elif int(_inp_num) in range(_num_devices):  # 检查输入是否在设备范围内
                self.port = _devices[int(_inp_num)].device
                return True
            else:
                print("%s 超出了设备号范围!" % _inp_num)
                return False

    # =========================================================
    # =======               通信监听函数               =========
    # =========================================================
    def start_listening(self) -> bool:
        if self.ser is None:
            self.send_message(False, "Error: init failed, can not listen")
            return False

        # _is_listening 为 True，等待它变为 False
        while self._is_listening:
            self.send_message(True, "Warnning: repeat listen")
            return False
        # 当 _is_listening 为 False 时，继续执行后续代码
        self._is_listening = True

        # 不开线程因为信号传递有问题
        self._listen_one_frame()
        return True

    def stop_listening(self) -> None:
        self._is_listening = False
        self.__received_buffer.clear()
        print("received buffer clear")
        # # 取消线程任务
        # self.future.cancel()
        # self.executor.shutdown(wait=True)

    def _listen_one_frame(self) -> None:
        # 清空串口接收缓冲区
        self.ser.reset_input_buffer()
        self.__received_buffer.clear()
        print("received buffer cleared")

        last_receive_time = time.time()         # 初始化接收时间
        start_listen_time = last_receive_time   # 记录监听的开始时间
        interval = self._RX_REPEAT_INTERVAL     # 初始化监听循环间隔

        # 进入监听循环
        while self._is_listening:
            buffer_length = len(self.__received_buffer)
            max_buffer_length = 2 * self._MAX_FRAME_LENGTH
            # 检查是否超时
            if time.time() - start_listen_time > self._MAX_LISTENING_TIME:
                print("Warning: Listening time exceeded the maximum limit")
                self.stop_listening()
                self.data_received_signal.emit(False, 0)
                break

            # 缓冲区数据超过最大长度，发送错误信号（大于两倍最大帧长）
            if buffer_length > max_buffer_length:
                data_length = self._verify_received_frame()
                self.stop_listening()
                # 停止监听，发送信号调用数据处理
                if data_length > 0:
                    print(f"Warning: data oversized: {buffer_length}")
                    self.data_received_signal.emit(True, data_length)
                else:
                    print("False: data oversized and no one frame")
                    self.data_received_signal.emit(False, data_length)
                break

            # 有数据可读,就移到__received_buffer
            if self.ser.in_waiting > 0:
                data = self.ser.read(self.ser.in_waiting)   # 读取缓冲区中所有数据
                self.__received_buffer.extend(data)         # 将数据存入缓冲区
                last_receive_time = time.time()             # 更新上次接收时间

            # 检测是否空闲一段时间
            elif buffer_length > 0 and time.time() - last_receive_time > self._RX_IDLE_THRESHOLD:
                print("receive idle detected")
                if buffer_length < self._MIN_FRAME_LENGTH:
                    print("buffer length is less than one frame length, continue listening")
                    continue

                data_length = self._verify_received_frame()
                if data_length > 0:
                    # 停止监听，发送信号调用数据处理
                    self.stop_listening()
                    print("a health frame detected, to received_data_handler")
                    self.data_received_signal.emit(True, data_length)

                    break

            time.sleep(interval)

    def _verify_received_frame(self) -> int:
        """
        验证接收到的数据帧，查找帧头并验证数据长度和校验位。
        成功时返回数据帧的起始地址和数据长度。
        """
        # 帧头
        frame_header = self.__FRAME_HEADER

        # 接收缓冲区长度
        buffer_length = len(self.__received_buffer)

        if buffer_length < 3:           # 最小帧长度需要至少 3 字节（帧头+长度+校验）
            print("Error: 缓冲区数据不足以组成帧头")
            return -1                   # 表示验证失败

        # 逐字节遍历接收缓冲区
        for i in range(buffer_length - 2):
            # 查找帧头
            if self.__received_buffer[i] == frame_header:
                # 找到帧头，检查是否有足够的数据来解析
                if i + 3 > buffer_length:
                    # 剩余字节不足以解析长度和校验，跳出循环等待更多数据
                    self.send_message(False, "Error: 数据不足以解析帧头后的内容")
                    return -1           # 验证失败

                # 解析帧头后的数据长度和校验位
                data_length = self.__received_buffer[i + 1]  # 第二个字节是数据长度

                # 检查数据是否足够接收完整帧
                if i + 3 + data_length > buffer_length:
                    self.send_message(False, "Error: 数据不足，等待完整帧接收")
                    return -1           # 验证失败，等待更多数据到来

                # 验证校验位 (暂不用CRC校验)
                check_byte = self.__received_buffer[i + 2 + data_length]     # 最后一个字节是校验位

                # calculated_check_byte = zlib.crc32(self._received_data) & 0xFF # 取低8位
                calculated_check_byte = sum(
                    self.__received_buffer[i + 2: i + 2 + data_length]) & 0xFF

                if check_byte != calculated_check_byte:
                    self.send_message(False, f"""Error: 校验失败
                                      收到的校验位: {check_byte}, 计算出的校验位: {calculated_check_byte}""")
                    return -1   # 校验失败

                # 提取帧数据
                self._received_data = self.__received_buffer[i + 2: i + 2 + data_length]

                # 数据帧验证成功，返回帧的起始地址和数据长度
                print(f"Frame verified, start at {i}, length {data_length}")
                return data_length  # 返回起始地址和数据长度

        self.send_message(False, "Error: 未找到帧头")
        return -1                       # 未找到帧头

    # =========================================================
    # =======               通信发送函数               =========
    # =========================================================
    def send_data_bb(self, data_to_send: bytes) -> bool:
        """
        发送数据，
        """
        if not isinstance(data_to_send, (bytes, bytearray)):
            self.send_message(False, "Error: 数据必须为bytes类型")
            return False

        # 确保 _send_buffer 大小足够
        if len(data_to_send) >= self._MAX_FRAME_LENGTH:
            self.send_message(False, "Error: 数据长度过长")
            return False

        # 清空串口发送缓冲区
        self.ser.reset_output_buffer()

        # 打包数据帧
        frame_length = self.__pack_send_frame(data_to_send)

        # 发送数据
        self.ser.write(self._send_buffer[:frame_length])
        print(f"send: {self._send_buffer[:frame_length].hex()}")
        return True

    def __pack_send_frame(self, data_to_send: bytes) -> int:
        """
        打包发送的数据帧，帧格式如下：
        帧头(1 byte): 0xEF
        数据长度(1 byte): 数据长度
        校验位(1 byte): 数据CRC校验的低八位
        数据内容: 变长数据
        """
        # 帧头
        frame_header = self.__FRAME_HEADER

        # 数据长度为实际数据的长度
        data_length = len(data_to_send)

        # 暂不用CRC校验,改为简单校验 取数据的低8位
        # check_byte = zlib.crc32(data_to_send) & 0xFF
        check_byte = sum(data_to_send) & 0xFF  # 取低8位

        # 数据总长度 = 帧头(1) + 数据长度(1) + 数据内容(data_length) + 校验位(1)
        total_frame_length = 3 + data_length  # 3字节为帧头、长度和校验

        # 将帧头和数据长度打包到 _send_buffer 的前两字节
        struct.pack_into('BB', self._send_buffer, 0, frame_header, data_length)

        # 将数据内容拷贝到 _send_buffer 的后续位置
        self._send_buffer[2:2 + data_length] = data_to_send

        # 使用 struct.pack_into 打包校验位到缓冲区的最后一字节
        struct.pack_into('B', self._send_buffer, 2 + data_length, check_byte)

        # 返回数据帧的总长度
        return total_frame_length

    # =====================收到数据帧的处理========================
    def _received_data_handler(self, result: bool, length: int) -> None:
        # 子类重写完善
        if result:
            print("Processing data:", self._received_data)
        elif length == 0:
            print("Error: no data")
        else:
            print("Processing _data:", self._received_data)
        self._received_data.clear()  # 清空数据

    # ========================用于显示=========================
    def send_message(self, success: bool, message: str):
        """发送带有状态的消息"""
        print(f"From Com:\n\t{message}\n")
        self.result_signal.emit(success, message)


# ========================调试用===========================
def test_Com_Driver_listen():
    test_com = Com_Driver(port='/dev/cu.usbserial-1120', baudrate=9600)
    test_com.start_listening()


def test_Com_Driver_send():
    command_code = 0x11
    length = 0x00
    command = struct.pack('BB', command_code, length)
    test_com = Com_Driver(port='/dev/cu.usbserial-1120', baudrate=9600)
    test_com.send_data_bb(command)


if __name__ == '__main__':
    test_Com_Driver_listen()
