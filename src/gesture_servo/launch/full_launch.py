"""
full_launch.py — 完整启动文件（改进版）

改进内容：
  1. [参数同步] kp_x/kp_y 改为正值 15.0，方向由 servo_ctrl_node 内部处理
  2. [死区增大] dead_zone 从 0.05 → 0.08，消除手静止时的舵机微抖
  3. [控制频率] control_rate 从 15 → 20Hz，追踪更顺滑
  4. [新增参数] det_confidence / trk_confidence 可通过 launch 配置
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('camera_index',   default_value='0'),
        # [改进] 降低检测置信度，暗光下更容易检出
        DeclareLaunchArgument('det_confidence', default_value='0.5'),
        DeclareLaunchArgument('trk_confidence', default_value='0.4'),
        # [改进] kp 取正值，方向由 servo_ctrl_node 内部统一处理
        DeclareLaunchArgument('kp_x',           default_value='15.0'),
        DeclareLaunchArgument('kp_y',           default_value='15.0'),
        # [改进] 死区加大
        DeclareLaunchArgument('dead_zone',      default_value='0.08'),
        DeclareLaunchArgument('control_rate',   default_value='20.0'),
        DeclareLaunchArgument('serial_port',    default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('baudrate',       default_value='9600'),
    ]

    gesture_capture = Node(
        package='gesture_servo', executable='gesture_capture_node',
        output='screen',
        parameters=[{
            'camera_index':    LaunchConfiguration('camera_index'),
            'det_confidence':  LaunchConfiguration('det_confidence'),
            'trk_confidence':  LaunchConfiguration('trk_confidence'),
        }])

    servo_ctrl = Node(
        package='gesture_servo', executable='servo_ctrl_node',
        output='screen',
        parameters=[{
            'kp_x':         LaunchConfiguration('kp_x'),
            'kp_y':         LaunchConfiguration('kp_y'),
            'dead_zone':    LaunchConfiguration('dead_zone'),
            'control_rate': LaunchConfiguration('control_rate'),
        }])

    serial_node = Node(
        package='gesture_servo', executable='serial_node',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baudrate':    LaunchConfiguration('baudrate'),
        }])

    return LaunchDescription(args + [gesture_capture, servo_ctrl, serial_node])