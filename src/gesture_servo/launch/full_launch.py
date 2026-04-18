from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('camera_index',  default_value='0'),
        DeclareLaunchArgument('kp_x',          default_value='-15.0'),
        DeclareLaunchArgument('kp_y',          default_value='15.0'),
        DeclareLaunchArgument('dead_zone',      default_value='0.05'),
        DeclareLaunchArgument('control_rate',   default_value='15.0'),
        DeclareLaunchArgument('serial_port',    default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('baudrate',       default_value='9600'),
    ]

    gesture_capture = Node(
        package='gesture_servo', executable='gesture_capture_node',
        output='screen',
        parameters=[{'camera_index': LaunchConfiguration('camera_index')}])

    servo_ctrl = Node(
        package='gesture_servo', executable='servo_ctrl_node',
        output='screen',
        parameters=[{
            'kp_x':        LaunchConfiguration('kp_x'),
            'kp_y':        LaunchConfiguration('kp_y'),
            'dead_zone':   LaunchConfiguration('dead_zone'),
            'control_rate':LaunchConfiguration('control_rate'),
        }])

    serial_node = Node(
        package='gesture_servo', executable='serial_node',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baudrate':    LaunchConfiguration('baudrate'),
        }])

    return LaunchDescription(args + [gesture_capture, servo_ctrl, serial_node])