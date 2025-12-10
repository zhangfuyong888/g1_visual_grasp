#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <custom_msg_srv/srv/get_best_pose.hpp>
#include "humanoid_grasp/action/arm_move.hpp"

using namespace std::chrono_literals;
using GetBestPose = custom_msg_srv::srv::GetBestPose;
using ArmMove = humanoid_grasp::action::ArmMove;

class VisualPickClient : public rclcpp::Node {
public:
  // 定义类型别名
  using GoalHandleArmMove = rclcpp_action::ClientGoalHandle<ArmMove>;

  VisualPickClient() : Node("visual_pick_client"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_) {
    action_client_ = rclcpp_action::create_client<ArmMove>(this, "arm_move");
    service_client_ = create_client<GetBestPose>("/get_best_pose");
    pub_right_close_ = create_publisher<std_msgs::msg::Bool>("/gripper/close", 10);
    pub_right_open_  = create_publisher<std_msgs::msg::Bool>("/gripper/open", 10);
    
    // 简单发布静态 TF
    static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    publishStaticTF();

    RCLCPP_INFO(get_logger(), "Starting mission thread...");
    mission_thread_ = std::thread(&VisualPickClient::execute_mission_logic, this);
  }

  ~VisualPickClient() {
    if (mission_thread_.joinable()) mission_thread_.join();
  }

private:
  rclcpp_action::Client<ArmMove>::SharedPtr action_client_;
  rclcpp::Client<GetBestPose>::SharedPtr service_client_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_right_close_, pub_right_open_;
  std::thread mission_thread_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;

  void execute_mission_logic() {
    if (!action_client_->wait_for_action_server(5s)) {
      RCLCPP_ERROR(get_logger(), "Action server not available!");
      return;
    }
    
    RCLCPP_INFO(get_logger(), "Step 1: Going Home...");
    auto home_pose = make_pose(0.15, -0.15, 0.12, 0, 0, 0, 1);
    if (!move_arm_sync(true, home_pose)) return;

    while (rclcpp::ok()) {
        RCLCPP_INFO(get_logger(), "--- New Cycle Start ---");

        // 模拟等待视觉服务 (如果没有真实服务，这步会卡住或失败)
        // 为了仿真调试，这里可以注释掉 wait_for_service，直接用假数据
        // if (!wait_for_service_and_tf()) { ... }

        // 获取目标 (仿真模式可直接硬编码)
        auto target_pose = get_object_pose();
        if (!target_pose) {
             // 仅供调试：如果没有视觉服务，硬编码一个点
             target_pose = make_pose(0.30, -0.04, 0.05, 0, 0, 0, 1);
             RCLCPP_WARN(get_logger(), "Vision failed, using FAKE target for sim.");
             std::this_thread::sleep_for(1s);
        }

        geometry_msgs::msg::Pose pre, grasp, lift;
        generate_grasp_poses(*target_pose, pre, grasp, lift);

        RCLCPP_INFO(get_logger(), "Approaching...");
        if (!move_arm_sync(true, pre)) continue;

        RCLCPP_INFO(get_logger(), "Reaching...");
        if (!move_arm_sync(true, grasp)) continue;

        RCLCPP_INFO(get_logger(), "Closing Gripper...");
        control_gripper(true);
        std::this_thread::sleep_for(4s);

        RCLCPP_INFO(get_logger(), "Lifting...");
        if (!move_arm_sync(true, lift)) continue;

        RCLCPP_INFO(get_logger(), "Returning Home...");
        move_arm_sync(true, home_pose);
        
        std::this_thread::sleep_for(4s);
    }
  }

  bool move_arm_sync(bool is_right, const geometry_msgs::msg::Pose& target) {
      auto goal_msg = ArmMove::Goal();
      goal_msg.is_right_arm = is_right;
      goal_msg.target_pose = target;
      goal_msg.max_velocity_scale = 0.5;

      auto send_goal_options = rclcpp_action::Client<ArmMove>::SendGoalOptions();
      // 使用正确的类型签名
      send_goal_options.feedback_callback = [this](
        GoalHandleArmMove::SharedPtr, 
        const std::shared_ptr<const ArmMove::Feedback> feedback) 
      {
          (void)feedback;
      };

      auto future_goal_handle = action_client_->async_send_goal(goal_msg, send_goal_options);
      
      if (future_goal_handle.wait_for(2s) != std::future_status::ready) {
          RCLCPP_ERROR(get_logger(), "Send goal failed (timeout)");
          return false;
      }

      auto goal_handle = future_goal_handle.get();
      if (!goal_handle) {
          RCLCPP_ERROR(get_logger(), "Goal was rejected by server");
          return false;
      }

      auto result_future = action_client_->async_get_result(goal_handle);
      auto wrapped_result = result_future.get(); 
      
      if (wrapped_result.code == rclcpp_action::ResultCode::SUCCEEDED) {
          return true;
      } else {
          RCLCPP_ERROR(get_logger(), "Move Failed: %s", wrapped_result.result->message.c_str());
          return false;
      }
  }

  std::optional<geometry_msgs::msg::Pose> get_object_pose() {
      if (!service_client_->service_is_ready()) return std::nullopt;
      auto req = std::make_shared<GetBestPose::Request>();
      auto future = service_client_->async_send_request(req);
      if (future.wait_for(1s) != std::future_status::ready) return std::nullopt;
      auto res = future.get();
      if (!res->success) return std::nullopt;
      
      // 简单 TF 转换
      try {
         geometry_msgs::msg::PoseStamped p_in = res->pose;
         if(p_in.header.frame_id.empty()) p_in.header.frame_id = "d435_link";
         return tf_buffer_.transform(p_in, "base_link", tf2::durationFromSec(1.0)).pose;
      } catch (...) { return std::nullopt; }
  }

  void control_gripper(bool close) {
      std_msgs::msg::Bool msg; msg.data = true;
      if (close) pub_right_close_->publish(msg);
      else pub_right_open_->publish(msg);
  }

  void generate_grasp_poses(const geometry_msgs::msg::Pose& obj, 
                            geometry_msgs::msg::Pose& pre, 
                            geometry_msgs::msg::Pose& grasp, 
                            geometry_msgs::msg::Pose& lift) 
  {
      grasp = obj; 
      grasp.position.x -= 0.05; 
      pre = grasp; pre.position.z += 0.05; 
      lift = grasp; lift.position.z += 0.05;
  }

  bool wait_for_service_and_tf() {
      // 仿真时不强求
      return true;
  }

  void publishStaticTF() {
      geometry_msgs::msg::TransformStamped t;
      t.header.stamp = now();
      t.header.frame_id = "d435_link";
      t.child_frame_id = "camera_link";
      t.transform.rotation.w = 1.0;
      static_broadcaster_->sendTransform(t);
  }

  geometry_msgs::msg::Pose make_pose(double x, double y, double z, double qx, double qy, double qz, double qw) {
      geometry_msgs::msg::Pose p;
      p.position.x = x; p.position.y = y; p.position.z = z;
      p.orientation.x = qx; p.orientation.y = qy; p.orientation.z = qz; p.orientation.w = qw;
      return p;
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VisualPickClient>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}