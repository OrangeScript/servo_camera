#!/usr/bin/env python3
"""
servo_ctrl_node — 舵机控制节点（完全重写版）

核心改进：
  1. [手部存在感知] 基于 pos.z 标志位判断手是否真的被检测到，
     不再依赖手势类型决定是否追踪——只要看到手就追踪
  2. [双重死区] raw_dead_zone 过滤原始噪声 + smooth_dead_zone 过滤滤波后的微小抖动
  3. [二阶EMA滤波] 两级低通滤波，彻底消除高频抖动，手不动时舵机完全静止
  4. [速度限制] max_step 限制单帧最大转动角度，防止"甩头"
  5. [手部丢失缓冲] 手消失后等待一小段时间(lost_timeout)再停止，
     避免偶尔丢帧导致舵机归中
  6. [手势去抖修复] 修正原版去抖逻辑的时间戳 BUG——新手势必须连续出现
     超过阈值时间才确认切换
  7. [PID方向统一] kp_x/kp_y 的正负号在此节点内统一定义，
     不再依赖 launch 传入的符号来纠正
"""
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int16MultiArray
from geometry_msgs.msg import Point


class ServoCtrlNode(Node):
    def __init__(self):
        super().__init__('servo_ctrl_node')

        # ============ 参数声明 ============
        # kp：比例增益（正数），方向由内部逻辑处理
        self.declare_parameter('kp_x', 15.0)
        self.declare_parameter('kp_y', 15.0)
        # [改进] 死区从 0.05 → 0.08，手静止时舵机也静止
        self.declare_parameter('dead_zone', 0.08)
        self.declare_parameter('angle_min', 0)
        self.declare_parameter('angle_max', 180)
        self.declare_parameter('control_rate', 20.0)

        self.kp_x      = self.get_parameter('kp_x').value
        self.kp_y      = self.get_parameter('kp_y').value
        self.dead_zone  = self.get_parameter('dead_zone').value
        self.a_min      = self.get_parameter('angle_min').value
        self.a_max      = self.get_parameter('angle_max').value
        rate            = self.get_parameter('control_rate').value

        # ============ 舵机状态 ============
        self.pan  = 90.0   # 水平舵机角度（0~180，90居中）
        self.tilt = 90.0   # 垂直舵机角度（0~180，90居中）

        # ============ 手部位置缓存 ============
        self.hx = 0.5           # 上次收到的手掌 x 归一化坐标
        self.hy = 0.5           # 上次收到的手掌 y 归一化坐标
        self.hand_visible = False  # [改进] 手是否被检测到（来自 pos.z）

        # ============ 手势状态 ============
        self.gest = 'none'
        self.track = True        # 是否允许追踪

        # ============ [改进] 手势去抖 — 修复原版 BUG ============
        # 原版问题：gest_candidate 在手势切换瞬间的时间戳更新逻辑错误
        # 修正后：记录"候选手势首次出现的时间"，只有连续保持超过 debounce 阈值才确认
        self._gest_candidate = 'none'       # 正在候选的手势
        self._gest_candidate_since = 0.0    # 候选手势首次出现的时间
        self._gest_debounce = 0.4           # 手势需持续 0.4 秒才确认（秒）
        self._stable_gest = 'none'          # 去抖后的稳定手势

        # ============ [改进] 二阶EMA滤波参数 ============
        # 第一级滤波（alpha1）：消除原始噪声
        # 第二级滤波（alpha2）：进一步平滑，适合慢速跟踪场景
        self._alpha1 = 0.25     # 第一级 EMA 系数，越小越平滑
        self._alpha2 = 0.35     # 第二级 EMA 系数
        self._ema1_x = 0.0      # 第一级 EMA 输出 (x)
        self._ema1_y = 0.0      # 第一级 EMA 输出 (y)
        self._ema2_x = 0.0      # 第二级 EMA 输出 (x)
        self._ema2_y = 0.0      # 第二级 EMA 输出 (y)
        # [改进] 滤波后死区：滤波输出低于此值时视为零，彻底消除微抖
        self._smooth_dead_zone = 0.03

        # [改进] 单帧最大转动角度限制：防止大幅度突变
        self._max_step = 2.5

        # ============ [改进] 手部丢失缓冲 ============
        # 手从画面消失后，等 lost_timeout 秒再认为真的丢了
        # 防止偶尔 1~2 帧丢失导致舵机突然归中
        self._last_hand_seen_time = 0.0
        self._lost_timeout = 0.5   # 手消失后 0.5 秒才判定为丢失

        # ============ 发布器 & 订阅器 ============
        self._cmd_pub = self.create_publisher(Int16MultiArray, '/servo_cmd', 10)
        self.create_subscription(Point, '/hand/position', self._pos_cb, 10)
        self.create_subscription(String, '/hand/gesture', self._ges_cb, 10)
        self.create_timer(1.0 / rate, self._loop)

        self.get_logger().info(
            f'✅ 舵机控制节点启动  rate={rate}Hz  kp=({self.kp_x},{self.kp_y})  '
            f'dead_zone={self.dead_zone}')

    # ================================================================
    #  回调：接收手掌位置
    #  pos.z = 1.0 表示手被检测到，0.0 表示未检测到
    # ================================================================
    def _pos_cb(self, msg):
        if msg.z > 0.5:
            # 手被检测到，更新坐标和时间戳
            self.hx = msg.x
            self.hy = msg.y
            self.hand_visible = True
            self._last_hand_seen_time = time.time()
        else:
            # 手未被检测到——不更新坐标，只标记状态
            self.hand_visible = False

    # ================================================================
    #  回调：接收手势 + 去抖处理（修复版）
    #  原版 BUG：时间戳在手势相同时才更新，导致首次切换时参照了错误的时间基准
    #  修正：用 candidate 模式——新手势首次出现时记录时间，持续超过阈值才确认
    # ================================================================
    def _ges_cb(self, msg):
        new_gest = msg.data
        now = time.time()

        if new_gest == self._stable_gest:
            # 当前手势就是已确认的手势，更新 candidate 以保持连续性
            self._gest_candidate = new_gest
            self._gest_candidate_since = now
        elif new_gest == self._gest_candidate:
            # 候选手势持续出现，检查是否超过去抖时间
            if now - self._gest_candidate_since >= self._gest_debounce:
                self._stable_gest = new_gest
                self.get_logger().info(f'手势确认: {new_gest}')
        else:
            # 全新的候选手势，重新计时
            self._gest_candidate = new_gest
            self._gest_candidate_since = now

    # ================================================================
    #  主控制循环 — 每帧调用一次
    # ================================================================
    def _loop(self):
        self._handle_gesture()

        now = time.time()
        # [改进] 判断手是否在画面中（带丢失缓冲）
        hand_present = self.hand_visible or (now - self._last_hand_seen_time < self._lost_timeout)

        if self.track and hand_present:
            self._track()
        else:
            # 手不在画面 或 追踪被禁止 → 清零滤波器状态，防止下次启动时残留偏差
            self._ema1_x = 0.0
            self._ema1_y = 0.0
            self._ema2_x = 0.0
            self._ema2_y = 0.0

        # 发布舵机角度指令
        msg = Int16MultiArray()
        msg.data = [int(round(self.pan)), int(round(self.tilt))]
        self._cmd_pub.publish(msg)

    # ================================================================
    #  手势处理 — 根据去抖后的稳定手势执行动作
    #  fist → 归中+停止追踪
    #  open → 开始追踪
    #  ok   → 锁定当前角度（停止追踪但不归中）
    # ================================================================
    def _handle_gesture(self):
        if self._stable_gest == 'fist':
            # 握拳：归中 + 停止追踪
            self.pan = 90.0
            self.tilt = 90.0
            self.track = False

        elif self._stable_gest == 'open':
            if not self.track:
                self.track = True
                self.get_logger().info('手势 [张开] → 追踪 ON')

        elif self._stable_gest == 'ok':
            if self.track:
                self.track = False
                self.get_logger().info('手势 [OK] → 追踪 OFF (锁定)')

    # ================================================================
    #  追踪核心算法
    #
    #  流程：
    #    1. 计算原始误差 (手掌位置 - 画面中心 0.5)
    #    2. 死区过滤：误差绝对值 < dead_zone 则视为 0
    #    3. 二阶 EMA 低通滤波：消除抖动
    #    4. 滤波后死区：极小的滤波输出也视为 0（彻底消抖）
    #    5. 比例控制：角度增量 = Kp × 滤波误差
    #    6. 速度限制：单帧增量不能超过 max_step
    #    7. 角度钳位：确保在 [angle_min, angle_max] 范围内
    #
    #  方向约定：
    #    - 画面 x 增大（手向右移）→ pan 减小（舵机向左转，使画面追上手）
    #    - 画面 y 增大（手向下移）→ tilt 减小（舵机向上仰，使画面追上手）
    #    因此 kp 取正值，但实际控制时取反
    # ================================================================
    def _track(self):
        # === 1. 原始误差 ===
        raw_ex = self.hx - 0.5   # 正 = 手在右侧
        raw_ey = self.hy - 0.5   # 正 = 手在下方

        # === 2. 原始死区 ===
        if abs(raw_ex) < self.dead_zone:
            raw_ex = 0.0
        if abs(raw_ey) < self.dead_zone:
            raw_ey = 0.0

        # === 3. 二阶 EMA 滤波 ===
        # 第一级
        self._ema1_x = self._alpha1 * raw_ex + (1.0 - self._alpha1) * self._ema1_x
        self._ema1_y = self._alpha1 * raw_ey + (1.0 - self._alpha1) * self._ema1_y
        # 第二级（对第一级的输出再滤波）
        self._ema2_x = self._alpha2 * self._ema1_x + (1.0 - self._alpha2) * self._ema2_x
        self._ema2_y = self._alpha2 * self._ema1_y + (1.0 - self._alpha2) * self._ema2_y

        # === 4. 滤波后死区 ===
        sx = self._ema2_x if abs(self._ema2_x) >= self._smooth_dead_zone else 0.0
        sy = self._ema2_y if abs(self._ema2_y) >= self._smooth_dead_zone else 0.0

        # === 5. 比例控制（方向取反，见上方注释） ===
        delta_pan  = -self.kp_x * sx
        delta_tilt = -self.kp_y * sy

        # === 6. 速度限制 ===
        delta_pan  = max(-self._max_step, min(self._max_step, delta_pan))
        delta_tilt = max(-self._max_step, min(self._max_step, delta_tilt))

        # === 7. 更新角度并钳位 ===
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
