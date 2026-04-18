#!/usr/bin/env python3
import threading
import cv2
import mediapipe as mp
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image  # 新增：用于发布图像话题
from cv_bridge import CvBridge     # 新增：用于将OpenCV图像转换为ROS图像消息

class GestureCaptureNode(Node):
    def __init__(self):
        super().__init__('gesture_capture_node')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('width',  640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps',    30)
        self.declare_parameter('det_confidence', 0.7)
        self.declare_parameter('trk_confidence', 0.5)

        cam   = self.get_parameter('camera_index').value
        w     = self.get_parameter('width').value
        h     = self.get_parameter('height').value
        fps   = self.get_parameter('fps').value
        det_c = self.get_parameter('det_confidence').value
        trk_c = self.get_parameter('trk_confidence').value

        self._ges_pub = self.create_publisher(String, '/hand/gesture',  10)
        self._pos_pub = self.create_publisher(Point,  '/hand/position', 10)
        self._img_pub = self.create_publisher(Image,  '/hand/image_result', 10)

        # 新增：CvBridge 实例
        self._bridge = CvBridge()
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=det_c,
            min_tracking_confidence=trk_c)
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap.set(cv2.CAP_PROP_FPS,          fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if not self._cap.isOpened():
            self.get_logger().error(f'无法打开摄像头 index={cam}')
            return

        self.get_logger().info(f'✅ gesture_capture_node 启动  cam={cam} {w}x{h}@{fps}')
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    # ---------- 手掌中心 ----------
    @staticmethod
    def _palm_center(lm):
        ids = [0, 5, 9, 13, 17]
        return (float(np.mean([lm[i].x for i in ids])),
                float(np.mean([lm[i].y for i in ids])))

    # ---------- 手势分类 ----------
    @staticmethod
    def _classify(lm):
        def up(tip, pip): return lm[tip].y < lm[pip].y
        def thumb():
            return lm[4].x < lm[3].x if lm[0].x < 0.5 else lm[4].x > lm[3].x

        t = thumb()
        i = up(8,  6)
        m = up(12, 10)
        r = up(16, 14)
        p = up(20, 18)
        n = sum([t, i, m, r, p])
        d = np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        if d < 0.05 and m and r:          return 'ok'
        if n <= 1 and not i:              return 'fist'
        if n == 5:                        return 'open'
        if i and m and not r and not p:   return 'peace'
        if t and not i and not m and not r and not p: return 'thumbs_up'
        return 'unknown'

    # ---------- 主采集循环 ----------
# ---------- 主采集循环 ----------
    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.flip(frame, -1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self._hands.process(rgb)

            # 获取画面的实际宽高（用于将归一化坐标转为像素坐标）
            h_img, w_img, _ = frame.shape 

            if res.multi_hand_landmarks:
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

                # 【新增】：计算像素坐标并画一个醒目的红点（中心点）
                px, py = int(cx * w_img), int(cy * h_img)
                # 画一个半径为 8，颜色为红色 (BGR: 0, 0, 255)，实心 (-1) 的圆
                cv2.circle(frame, (px, py), 8, (0, 0, 255), -1) 
                # 给红点加个白边，更清晰
                cv2.circle(frame, (px, py), 10, (255, 255, 255), 2)

            else:
                cx, cy, gesture = 0.5, 0.5, 'none'

            # 在画面左上角写上当前识别到的手势名称
            cv2.putText(frame, f'Gesture: {gesture}', (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # ... 下面的发布消息逻辑保持不变 ...
            g = String(); g.data = gesture
            self._ges_pub.publish(g)
            pos = Point(); pos.x = cx; pos.y = cy; pos.z = 0.0
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
    try:    rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()