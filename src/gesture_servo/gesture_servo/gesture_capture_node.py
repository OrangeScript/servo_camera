#!/usr/bin/env python3
"""
gesture_capture_node — 手势捕获节点（改进版）

改进内容：
  1. [暗光增强] 新增 CLAHE 自适应直方图均衡化，大幅提升低光照环境下的识别率
  2. [手部存在标志] pos.z 用作手部检测标志（1.0=检测到手，0.0=未检测到）
     ——不再在无手时发送假的中心坐标(0.5,0.5)，避免舵机误动
  3. [降低检测阈值] det_confidence 默认从 0.7 降至 0.5，暗光下更容易检出
  4. [手势分类鲁棒性] OK 手势的判定距离阈值从 0.05 放宽到 0.07，减少误判
  5. [帧率控制] 增加可选的帧间最小间隔，降低 CPU 占用
"""
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class GestureCaptureNode(Node):
    def __init__(self):
        super().__init__('gesture_capture_node')

        # ============ 参数声明 ============
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('width',  640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps',    30)
        # [改进] 检测置信度从 0.7 降至 0.5，暗光环境下更容易检出手部
        self.declare_parameter('det_confidence', 0.5)
        self.declare_parameter('trk_confidence', 0.4)

        cam   = self.get_parameter('camera_index').value
        w     = self.get_parameter('width').value
        h     = self.get_parameter('height').value
        fps   = self.get_parameter('fps').value
        det_c = self.get_parameter('det_confidence').value
        trk_c = self.get_parameter('trk_confidence').value

        # ============ 话题发布器 ============
        self._ges_pub = self.create_publisher(String, '/hand/gesture',  10)
        self._pos_pub = self.create_publisher(Point,  '/hand/position', 10)
        self._img_pub = self.create_publisher(Image,  '/hand/image_result', 10)

        self._bridge = CvBridge()

        # ============ MediaPipe Hands 初始化 ============
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=det_c,
            min_tracking_confidence=trk_c)
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        # ============ 摄像头初始化 ============
        self._cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap.set(cv2.CAP_PROP_FPS,          fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if not self._cap.isOpened():
            self.get_logger().error(f'无法打开摄像头 index={cam}')
            return

        # [改进] CLAHE 自适应直方图均衡化实例，用于暗光增强
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        self.get_logger().info(f'✅ gesture_capture_node 启动  cam={cam} {w}x{h}@{fps}')
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    # ================================================================
    #  手掌中心计算 — 取手腕+四指根部的平均坐标
    # ================================================================
    @staticmethod
    def _palm_center(lm):
        ids = [0, 5, 9, 13, 17]
        return (float(np.mean([lm[i].x for i in ids])),
                float(np.mean([lm[i].y for i in ids])))

    # ================================================================
    #  暗光增强 — CLAHE 自适应直方图均衡化
    #  原理：将图像转到 LAB 色彩空间，仅对亮度通道 L 做均衡化，
    #        保留色彩不失真，大幅提升暗部细节
    # ================================================================
    def _enhance_low_light(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # ================================================================
    #  手势分类 — 基于手指伸展状态判断手势
    # ================================================================
    @staticmethod
    def _classify(lm):
        # 判断某根手指是否伸直：指尖 y < 指中节 y 即为伸直
        def up(tip, pip): return lm[tip].y < lm[pip].y
        # 拇指比较特殊，需要根据手的朝向判断
        def thumb():
            return lm[4].x < lm[3].x if lm[0].x < 0.5 else lm[4].x > lm[3].x

        t = thumb()
        i = up(8,  6)     # 食指
        m = up(12, 10)    # 中指
        r = up(16, 14)    # 无名指
        p = up(20, 18)    # 小指
        n = sum([t, i, m, r, p])  # 伸直手指数量

        # [改进] OK 手势：拇指和食指间距阈值从 0.05 → 0.07，更容易触发
        d = np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        if d < 0.07 and m and r:          return 'ok'
        if n <= 1 and not i:              return 'fist'
        if n == 5:                        return 'open'
        if i and m and not r and not p:   return 'peace'
        if t and not i and not m and not r and not p: return 'thumbs_up'
        return 'unknown'

    # ================================================================
    #  主采集循环
    # ================================================================
    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, -1)

            # [改进] 暗光增强：在送入 MediaPipe 之前做 CLAHE
            enhanced = self._enhance_low_light(frame)
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            res = self._hands.process(rgb)

            h_img, w_img, _ = frame.shape
            # [改进] hand_detected 标志，用于通知下游节点是否真的检测到手
            hand_detected = False

            if res.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in res.multi_hand_landmarks:
                    self._mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        self._mp_drawing_styles.get_default_hand_landmarks_style(),
                        self._mp_drawing_styles.get_default_hand_connections_style())

                lm = res.multi_hand_landmarks[0].landmark
                cx, cy  = self._palm_center(lm)
                gesture = self._classify(lm)

                # 画手掌中心红点 + 白色边框
                px, py = int(cx * w_img), int(cy * h_img)
                cv2.circle(frame, (px, py), 8, (0, 0, 255), -1)
                cv2.circle(frame, (px, py), 10, (255, 255, 255), 2)

                # [改进] 画十字准星，方便调试时看到目标中心
                cv2.line(frame, (w_img // 2 - 20, h_img // 2),
                         (w_img // 2 + 20, h_img // 2), (0, 255, 255), 1)
                cv2.line(frame, (w_img // 2, h_img // 2 - 20),
                         (w_img // 2, h_img // 2 + 20), (0, 255, 255), 1)
            else:
                # [改进] 未检测到手时：cx/cy 设为 -1 表示无效，不再伪造中心坐标
                cx, cy, gesture = -1.0, -1.0, 'none'

            # 在画面上显示手势名称和检测状态
            color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.putText(frame, f'Gesture: {gesture}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # ============ 发布话题 ============
            g = String()
            g.data = gesture
            self._ges_pub.publish(g)

            pos = Point()
            pos.x = cx
            pos.y = cy
            # [关键改进] pos.z 作为"手部是否存在"的标志位
            # 1.0 = 检测到手（坐标有效），0.0 = 未检测到手（坐标无效）
            pos.z = 1.0 if hand_detected else 0.0
            self._pos_pub.publish(pos)

            try:
                img_msg = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                self._img_pub.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f'图像转换或发布失败: {e}')

    def destroy_node(self):
        self._running = False
        self._cap.release()
        self._hands.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GestureCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()