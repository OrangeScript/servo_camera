#!/usr/bin/env python3
"""
serial_node — 串口通信节点（改进版）

改进内容：
  1. [降低日志频率] 每 50 帧只打印一次 TX 日志，避免高频刷屏
  2. [重复帧过滤] 如果角度与上次发送的完全相同，跳过发送，减轻串口负担
  3. [添加详细注释] 说明串口协议格式
"""
import serial
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray


class SerialNode(Node):
    def __init__(self):
        super().__init__('serial_node')
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate',    9600)

        port = self.get_parameter('serial_port').value
        baud = self.get_parameter('baudrate').value

        self._ser = None
        try:
            self._ser = serial.Serial(port, baud, timeout=1)
            self.get_logger().info(f'✅ 串口已打开: {port} @ {baud}')
        except Exception as e:
            self.get_logger().error(
                f'❌ 串口打开失败: {e}\n'
                '   检查: ls /dev/ttyUSB* | sudo usermod -aG dialout $USER')

        self.create_subscription(Int16MultiArray, '/servo_cmd', self._cb, 10)

        # [改进] 记录上次发送的角度，避免重复发送
        self._last_pan = -1
        self._last_tilt = -1
        # [改进] 日志计数器，降低打印频率
        self._tx_count = 0

        self.get_logger().info('serial_node 就绪，订阅 /servo_cmd')

    def _cb(self, msg):
        if len(msg.data) < 2:
            return

        pan  = max(0, min(180, int(msg.data[0])))
        tilt = max(0, min(180, int(msg.data[1])))

        # [改进] 角度未变化时跳过发送，减少串口通信量
        if pan == self._last_pan and tilt == self._last_tilt:
            return
        self._last_pan = pan
        self._last_tilt = tilt

        # 串口协议帧：[帧头 0xFF] [水平角度] [垂直角度] [帧尾 0xFE]
        # STM32 端解析：检测到 0xFF 开头 + 0xFE 结尾，中间两字节为角度值
        frame = bytes([0xFF, pan, tilt, 0xFE])

        if self._ser and self._ser.is_open:
            try:
                self._ser.write(frame)
                self._tx_count += 1
                # [改进] 每 50 次才打印一次日志，避免刷屏
                if self._tx_count % 50 == 1:
                    self.get_logger().info(
                        f'TX #{self._tx_count}: FF {pan:02X} {tilt:02X} FE  ({pan}° / {tilt}°)')
            except Exception as e:
                self.get_logger().error(f'串口写失败: {e}')
        else:
            self.get_logger().warn('串口未就绪')

    def destroy_node(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SerialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()