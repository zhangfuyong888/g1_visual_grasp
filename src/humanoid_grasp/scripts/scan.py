#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import re
import time
import threading

# 需要安装 sudo pip3 install pyserial
# 启动时执行 sudo chmod 777 /dev/ttyACM0

class QRCodeScannerNode(Node):
    def __init__(self):
        super().__init__('code_scanner')

        # 1. 声明参数 (可以在 launch 文件中修改)
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('scan_interval', 5.0) # 防止重复扫码的时间间隔

        # 获取参数值
        self.port = self.get_parameter('port').get_parameter_value().string_value
        self.baud = self.get_parameter('baud').get_parameter_value().integer_value
        self.scan_interval = self.get_parameter('scan_interval').get_parameter_value().double_value

        # 2. 创建发布者 (发布话题: /scan_code/result)
        self.publisher_ = self.create_publisher(String, '/scan_code/result', 10)

        # 3. 初始化串口
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            self.get_logger().info(f"成功打开串口: {self.port} @ {self.baud}")
        except serial.SerialException as e:
            self.get_logger().error(f"无法打开串口 {self.port}: {e}")
            # 这里的 return 会导致节点初始化完成但无法工作，实际应用可能需要抛出异常或重试
            return

        self.last_detect = 0.0
        self.running = True

        # 4. 开启独立线程进行串口读取，避免阻塞 ROS 主循环
        self.read_thread = threading.Thread(target=self.read_loop)
        self.read_thread.start()

    def clean_code(self, raw: bytes) -> str:
        """ 数据清洗逻辑，保持原样 """
        for mark in (b'\x13', b'\x14', b'\x15'):
            idx = raw.find(mark)
            if idx != -1:
                payload = raw[idx + 1:]
                break
        else:
            payload = raw
        if payload and payload[0] < 0x20:
            payload = payload[1:]
        end_idx = min((i for i, b in enumerate(payload) if b in (0x0d, 0x0a)), default=len(payload))
        payload = payload[:end_idx]
        text = payload.decode(errors="ignore").strip()
        text = text.lstrip("=v!W\"")
        text = text.rstrip("=v!W\"")
        if len(text) < 3:
            return ""
        if not re.match(r'^[A-Za-z0-9\-\_\./: ]+$', text):
            return ""
        return text

    def read_loop(self):
        """ 独立的串口读取循环 """
        while rclpy.ok() and self.running and self.ser.is_open:
            try:
                # 阻塞读取，直到有换行符或超时
                raw = self.ser.read_until(b'\n')
                if not raw:
                    continue

                code = self.clean_code(raw)
                if code:
                    now = time.time()
                    if now - self.last_detect < self.scan_interval:
                        self.get_logger().warn(
                            f"忽略重复扫码 ({self.scan_interval - (now - self.last_detect):.1f}s): {code}"
                        )
                        continue
                    
                    self.last_detect = now
                    self.get_logger().info(f"扫码结果: {code}")

                    # --- 发布消息 ---
                    msg = String()
                    msg.data = code
                    self.publisher_.publish(msg)

                    # --- 执行原来的业务逻辑 (Match Case) ---
                    self.handle_logic(code)

            except Exception as e:
                self.get_logger().error(f"串口读取出错: {e}")
                time.sleep(1) # 出错后休息一秒避免死循环刷屏

    def handle_logic(self, code):
        """ 原有的业务判断逻辑 """
        match code:
            case "Material1":
                self.get_logger().info(">>> 动作：检测到物料1，执行抓取A...")
                # 这里可以扩展：调用 Service 或 Action 通知机械臂
            case "Material2":
                self.get_logger().info(">>> 动作：检测到物料2，执行抓取B...")
            case "Material3":
                self.get_logger().info(">>> 动作：检测到物料3，执行抓取C...")
            case _:
                self.get_logger().info(">>> 动作：未知物料，放入废料箱。")

    def destroy_node(self):
        """ 节点销毁时的清理工作 """
        self.running = False
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
            self.get_logger().info("串口已关闭")
        if hasattr(self, 'read_thread'):
            self.read_thread.join()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = QRCodeScannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()