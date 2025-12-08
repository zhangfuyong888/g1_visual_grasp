import rclpy
from rclpy.node import Node
import sys

# 导入您的自定义服务类型
from custom_msg_srv.srv import GetBestPose

class BestPoseClient(Node):
    """
    用于请求最佳目标位姿的服务客户端节点。
    """
    def __init__(self):
        super().__init__('best_pose_client')
        self.client = self.create_client(GetBestPose, 'get_best_pose')
        
        self.request = GetBestPose.Request()

    def wait_for_service(self, timeout=3.0):
        """
        等待服务上线，带超时。
        """
        self.get_logger().info(f'Waiting for service "{self.client.srv_name}" to be available...')
        if not self.client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f'Service "{self.client.srv_name}" not available after {timeout}s.')
            return False
        return True

    def send_request(self):
        """
        发送异步请求并返回一个 'future' 对象。
        """
        self.get_logger().info('Sending request...')
        self.future = self.client.call_async(self.request)
        return self.future

def main(args=None):
    rclpy.init(args=args)

    client_node = BestPoseClient()

    if not client_node.wait_for_service():
        client_node.destroy_node()
        rclpy.shutdown()
        return

    future = client_node.send_request()

    rclpy.spin_until_future_complete(client_node, future)

    try:
        response = future.result()
    except Exception as e:
        client_node.get_logger().error(f'Service call failed with exception: {e}')
    else:
        if response.success:
            client_node.get_logger().info("--- Service Call Successful ---")
            client_node.get_logger().info(f"Message: {response.message}")
            client_node.get_logger().info(f"Frame ID: {response.pose.header.frame_id}")
            client_node.get_logger().info(
                f"Position (x, y, z): "
                f"{response.pose.pose.position.x:.4f}, "
                f"{response.pose.pose.position.y:.4f}, "
                f"{response.pose.pose.position.z:.4f}"
            )
            client_node.get_logger().info(
                f"Orientation (x, y, z, w): "
                f"{response.pose.pose.orientation.x:.4f}, "
                f"{response.pose.pose.orientation.y:.4f}, "
                f"{response.pose.pose.orientation.z:.4f}, "
                f"{response.pose.pose.orientation.w:.4f}"
            )
        else:
            client_node.get_logger().warn(f"--- Service Call Failed ---")
            client_node.get_logger().warn(f"Message: {response.message}")

    finally:
        client_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
