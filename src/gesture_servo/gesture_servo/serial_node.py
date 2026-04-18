#!/usr/bin/env python3
import serial
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray


class SerialNode(Node):
    def __init__(self):
        super().__init__('serial_node')
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate',    9600)

        port  = self.get_parameter('serial_port').value
        baud  = self.get_parameter('baudrate').value

        self._ser = None
        try:
            self._ser = serial.Serial(port, baud, timeout=1)
            self.get_logger().info(f'✅ 串口已打开: {port} @ {baud}')
        except Exception as e:
            self.get_logger().error(
                f'❌ 串口打开失败: {e}\n'
                '   检查: ls /dev/ttyUSB* | sudo usermod -aG dialout $USER')

        self.create_subscription(Int16MultiArray, '/servo_cmd', self._cb, 10)
        self.get_logger().info('serial_node 就绪，订阅 /servo_cmd')

    def _cb(self, msg):
        if len(msg.data) < 2:
            return
        pan  = max(0, min(180, int(msg.data[0])))
        tilt = max(0, min(180, int(msg.data[1])))
        frame = bytes([0xFF, pan, tilt, 0xFE])
        if self._ser and self._ser.is_open:
            try:
                self._ser.write(frame)
                self.get_logger().info(
                    f'TX: FF {pan:02X} {tilt:02X} FE  ({pan}° / {tilt}°)')
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
    try:    rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()