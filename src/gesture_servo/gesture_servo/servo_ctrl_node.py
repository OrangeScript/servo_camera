#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int16MultiArray
from geometry_msgs.msg import Point

class ServoCtrlNode(Node):
    def __init__(self):
        super().__init__('servo_ctrl_node')
        # 参数声明保持不变
        self.declare_parameter('kp_x', 5.0)
        self.declare_parameter('kp_y', -5.0)
        self.declare_parameter('dead_zone', 0.05)
        self.declare_parameter('angle_min', 0)
        self.declare_parameter('angle_max', 180)
        self.declare_parameter('control_rate', 15.0)

        self.kp_x      = self.get_parameter('kp_x').value
        self.kp_y      = self.get_parameter('kp_y').value
        self.dead_zone = self.get_parameter('dead_zone').value
        self.a_min     = self.get_parameter('angle_min').value
        self.a_max     = self.get_parameter('angle_max').value
        rate           = self.get_parameter('control_rate').value

        # 状态变量
        self.pan  = 90.0
        self.tilt = 90.0
        self.hx   = 0.5
        self.hy   = 0.5
        self.gest = 'none'
        
        # 新增：滤波与状态机优化变量
        self.smoothed_ex = 0.0
        self.smoothed_ey = 0.0
        self.filter_alpha = 0.3  # 滤波系数(0~1)，越小越平滑但延迟越高，越大越灵敏
        self.max_step = 3.0      # 限制舵机单次最大转动角度（限速）

        self.track = True
        self._last_gest_time = time.time()
        self._stable_gest = 'none' # 用于去抖后的稳定手势

        self._cmd_pub = self.create_publisher(Int16MultiArray, '/servo_cmd', 10)
        self.create_subscription(Point, '/hand/position', self._pos_cb, 10)
        self.create_subscription(String, '/hand/gesture', self._ges_cb, 10)
        self.create_timer(1.0 / rate, self._loop)

        self.get_logger().info(f'✅ 舵机平滑控制节点启动 频率={rate}Hz')

    def _pos_cb(self, msg): 
        self.hx = msg.x
        self.hy = msg.y

    def _ges_cb(self, msg): 
        new_gest = msg.data
        # 简单的时间防抖：同一个手势必须持续识别超过 0.3 秒才认为是有效切换，防止画面一闪而过的误判
        if new_gest != self._stable_gest:
            if time.time() - self._last_gest_time > 0.3:
                self._stable_gest = new_gest
                self._last_gest_time = time.time()
        else:
            self._last_gest_time = time.time()

    def _loop(self):
        self._handle_gesture()
        
        # 只有在允许追踪，且手势不是握拳、OK或丢失时才动
        if self.track and self._stable_gest not in ('none', 'fist', 'unknown', 'ok'):
            self._track()
        else:
            # 目标丢失或停止追踪时，将平滑误差归零，防止下次开启时乱跳
            self.smoothed_ex = 0.0
            self.smoothed_ey = 0.0
            
        # 发布舵机指令
        msg = Int16MultiArray()
        msg.data = [int(round(self.pan)), int(round(self.tilt))]
        self._cmd_pub.publish(msg)

    def _handle_gesture(self):
        # 根据稳定的手势执行动作
        if self._stable_gest == 'fist':
            self.pan = 90.0
            self.tilt = 90.0
            self.track = False # 握拳归中并停止追踪
            
        elif self._stable_gest == 'open':
            if not self.track:
                self.track = True
                self.get_logger().info('手势 [张开] → 追踪 ON')
                
        elif self._stable_gest == 'ok':
            if self.track:
                self.track = False
                self.get_logger().info('手势 [OK] → 追踪 OFF (已锁定)')

    def _track(self):
        # 计算原始误差
        raw_ex = self.hx - 0.5
        raw_ey = self.hy - 0.5

        # 死区处理
        if abs(raw_ex) < self.dead_zone: raw_ex = 0.0
        if abs(raw_ey) < self.dead_zone: raw_ey = 0.0

        # 核心优化：低通滤波 (EMA) 让运动更丝滑
        self.smoothed_ex = (self.filter_alpha * raw_ex) + ((1.0 - self.filter_alpha) * self.smoothed_ex)
        self.smoothed_ey = (self.filter_alpha * raw_ey) + ((1.0 - self.filter_alpha) * self.smoothed_ey)

        # 计算增量并限制最大步长（防止“甩头”）
        delta_pan = max(-self.max_step, min(self.max_step, self.kp_x * self.smoothed_ex))
        delta_tilt = max(-self.max_step, min(self.max_step, self.kp_y * self.smoothed_ey))

        # 更新舵机角度并限制在 min/max 范围内
        self.pan  = max(self.a_min, min(self.a_max, self.pan + delta_pan))
        self.tilt = max(self.a_min, min(self.a_max, self.tilt + delta_tilt))


def main(args=None):
    rclpy.init(args=args)
    node = ServoCtrlNode()
    try:    
        rclpy.spin(node)
    except KeyboardInterrupt: 
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()