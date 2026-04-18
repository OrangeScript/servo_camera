import serial
import time
import random

# 修改这里：改为 USB 串口路径
PORT = '/dev/ttyUSB0' 
BAUDRATE = 9600  # 必须与 STM32 代码中的串口波特率一致

try:
    # 这里的打开动作是实时的，如果没插好会直接报错
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"✅ USB转TTL 串口 {PORT} 已打通！")

    for i in range(1, 11):
        angle1 = random.randint(0, 180)
        angle2 = random.randint(0, 180)
        
        # 协议保持不变：[0xFF, 角度1, 角度2, 0xFE]
        packet = bytes([0xFF, angle1, angle2, 0xFE])
        
        ser.write(packet)
        print(f"发送数据: {packet.hex().upper()}")
        
        time.sleep(1)

except Exception as e:
    print(f"❌ 错误: {e}")
finally:
    if 'ser' in locals():
        ser.close()