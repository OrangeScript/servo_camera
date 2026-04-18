#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int16MultiArray
from geometry_msgs.msg import Point


class ServoCtrlNode(Node):
    def __init__(self):
        super().__init__('servo_ctrl_node')
        self.declare_parameter('kp_x',        -15.0)   # 水平增益，方向反了改正号
        self.declare_parameter('kp_y',         15.0)   # 垂直增益，方向反了改负号
        self.declare_parameter('dead_zone',     0.05)
        self.declare_parameter('angle_min',     0)
        self.declare_parameter('angle_max',     180)
        self.declare_parameter('control_rate',  15.0)

        self.kp_x      = self.get_parameter('kp_x').value
        self.kp_y      = self.get_parameter('kp_y').value
        self.dead_zone = self.get_parameter('dead_zone').value
        self.a_min     = self.get_parameter('angle_min').value
        self.a_max     = self.get_parameter('angle_max').value
        rate           = self.get_parameter('control_rate').value

        # 状态
        self.pan   = 90.0
        self.tilt  = 90.0
        self.hx    = 0.5
        self.hy    = 0.5
        self.gest  = 'none'
        self._prev = 'none'
        self.track = True
        self._last_t = 0.0

        self._cmd_pub = self.create_publisher(Int16MultiArray, '/servo_cmd', 10)
        self.create_subscription(Point,  '/hand/position', self._pos_cb, 10)
        self.create_subscription(String, '/hand/gesture',  self._ges_cb, 10)
        self.create_timer(1.0 / rate, self._loop)

        self.get_logger().info(
            f'✅ servo_ctrl_node 启动  Kp_x={self.kp_x} Kp_y={self.kp_y} '
            f'死区={self.dead_zone} 频率={rate}Hz')

    def _pos_cb(self, msg): self.hx = msg.x; self.hy = msg.y
    def _ges_cb(self, msg): self.gest = msg.data

    def _loop(self):
        self._handle_gesture()
        if self.track and self.gest not in ('none', 'fist', 'unknown'):
            self._track()
        self._prev = self.gest
        msg = Int16MultiArray()
        msg.data = [int(round(self.pan)), int(round(self.tilt))]
        self._cmd_pub.publish(msg)

    def _handle_gesture(self):
        now = time.time()
        if self.gest in ('none', 'unknown'): return
        if self.gest == self._prev:          return
        if now - self._last_t < 1.0:        return
        self._last_t = now

        if self.gest == 'fist':
            self.pan = self.tilt = 90.0
            self.get_logger().info('握拳 → 归中')
        elif self.gest == 'open':
            self.track = not self.track
            self.get_logger().info(f'张开 → 跟踪{"ON" if self.track else "OFF"}')

    def _track(self):
        ex = self.hx - 0.5
        ey = self.hy - 0.5
        if abs(ex) < self.dead_zone: ex = 0.0
        if abs(ey) < self.dead_zone: ey = 0.0
        self.pan  = max(self.a_min, min(self.a_max, self.pan  + self.kp_x * ex))
        self.tilt = max(self.a_min, min(self.a_max, self.tilt + self.kp_y * ey))


def main(args=None):
    rclpy.init(args=args)
    node = ServoCtrlNode()
    try:    rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()