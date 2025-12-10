#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp> // [新增] 引入 String 消息
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <random>

#include <custom_msg_srv/srv/get_best_pose.hpp>
#include "humanoid_grasp/action/arm_move.hpp"

using namespace std::chrono_literals;
using GetBestPose = custom_msg_srv::srv::GetBestPose;
using ArmMove = humanoid_grasp::action::ArmMove;

class VisualPickClient : public rclcpp::Node {
public:
  using GoalHandleArmMove = rclcpp_action::ClientGoalHandle<ArmMove>;

  VisualPickClient() : Node("visual_pick_client"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_) {
    action_client_ = rclcpp_action::create_client<ArmMove>(this, "arm_move");
    service_client_ = create_client<GetBestPose>("/get_best_pose");
    
    pub_right_close_ = create_publisher<std_msgs::msg::Bool>("/gripper/close", 10);
    pub_right_open_  = create_publisher<std_msgs::msg::Bool>("/gripper/open", 10);
    
    // [新增] 订阅扫码结果
    sub_scan_result_ = create_subscription<std_msgs::msg::String>(
        "/scan_code/result", 10,
        std::bind(&VisualPickClient::on_scan_result, this, std::placeholders::_1)
    );

    // 静态 TF
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
  // [新增] 扫码订阅者
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_scan_result_;

  std::thread mission_thread_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;

  // [新增] 扫码相关变量
  std::mutex scan_mtx_;
  std::string last_scanned_code_;
  std::atomic<bool> has_new_scan_{false};

  // [新增] 扫码回调函数
  void on_scan_result(const std_msgs::msg::String::SharedPtr msg) {
      std::lock_guard<std::mutex> lk(scan_mtx_);
      last_scanned_code_ = msg->data;
      has_new_scan_ = true; // 标记收到新码
      RCLCPP_INFO(get_logger(), ">>> 收到扫码结果: %s", last_scanned_code_.c_str());
  }

  // --------------------------------------------------------
  // 核心逻辑
  // --------------------------------------------------------
  void execute_mission_logic() {
    if (!action_client_->wait_for_action_server(5s)) {
      RCLCPP_ERROR(get_logger(), "Action server not available!");
      return;
    }
    
    RCLCPP_INFO(get_logger(), "Step 1: Going Home...");
    // 假设这是初始位置
    auto home_pose = make_pose(0.15, -0.15, 0.12, 0, 0, 0, 1);
    // 定义一个固定的扫码点 (请根据实际扫码枪位置修改坐标)
    // 假设扫码枪在右侧某个位置，夹爪需要把物体凑过去
    auto scan_pose_base = make_pose(0.35, -0.30, 0.20, 0, 0, 0, 1); 

    if (!move_arm_sync(true, home_pose)) return;

    while (rclcpp::ok()) {
        RCLCPP_INFO(get_logger(), "--- New Cycle Start ---");

        // 1. 获取视觉识别 (仿真环境使用假数据)
        auto target_pose = get_object_pose();
        if (!target_pose) {
             target_pose = make_pose(0.30, -0.04, 0.05, 0, 0, 0, 1);
             RCLCPP_WARN(get_logger(), "Vision failed, using FAKE target for sim.");
             std::this_thread::sleep_for(1s);
        }

        // 2. 生成抓取点
        geometry_msgs::msg::Pose pre, grasp, lift;
        generate_grasp_poses(*target_pose, pre, grasp, lift);

        // 3. 执行抓取
        RCLCPP_INFO(get_logger(), "Approaching...");
        if (!move_arm_sync(true, pre)) continue;

        RCLCPP_INFO(get_logger(), "Reaching...");
        if (!move_arm_sync(true, grasp)) continue;

        RCLCPP_INFO(get_logger(), "Closing Gripper...");
        control_gripper(true);
        std::this_thread::sleep_for(2s); // 等待夹紧

        RCLCPP_INFO(get_logger(), "Lifting...");
        if (!move_arm_sync(true, lift)) continue;

        // ============================================
        // [新增] 4. 移动到扫码区并执行扫码逻辑
        // ============================================
        RCLCPP_INFO(get_logger(), "Moving to Scan Pose...");
        if (perform_scanning_task(scan_pose_base)) {
            // 扫码成功
            std::lock_guard<std::mutex> lk(scan_mtx_);
            RCLCPP_INFO(get_logger(), "Processing scanned object: %s", last_scanned_code_.c_str());
            // 这里可以根据扫到的码决定放置位置，目前保持原样
        } else {
            // 扫码失败/超时
            RCLCPP_WARN(get_logger(), "Scan timeout! Proceeding anyway.");
        }

        // 5. 放置
        RCLCPP_INFO(get_logger(), "Placing...");
        auto place_pose = make_pose(0.29, -0.34, 0.15, -0.04, 0.0, -0.07, 0.99);
        if (!move_arm_sync(true, place_pose)) continue;

        RCLCPP_INFO(get_logger(), "Opening Gripper...");
        control_gripper(false);
        std::this_thread::sleep_for(2s);

        // 6. 回家
        RCLCPP_INFO(get_logger(), "Returning Home...");
        move_arm_sync(true, home_pose);
        
        std::this_thread::sleep_for(2s);
    }
  }

  // --------------------------------------------------------
  // [修改] 扫码任务逻辑：无限循环，直到扫码成功
  // 策略：初始点 -> 失败 -> 在 XYZ ±3cm 范围内随机抖动 -> 重试
  // --------------------------------------------------------
  bool perform_scanning_task(const geometry_msgs::msg::Pose& base_scan_pose) {
      // 1. 清除旧标志位
      has_new_scan_ = false; 

      // 2. 初始化随机数生成器 (范围 -0.03 到 +0.03)
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<double> dist(-0.03, 0.03);

      // 3. 先尝试移动到基准扫码点
      RCLCPP_INFO(get_logger(), ">>> Moving to Base Scan Pose...");
      if (!move_arm_sync(true, base_scan_pose)) return false;

      // 等待 3 秒看看有没有结果
      if (wait_for_scan(3.0)) {
          RCLCPP_INFO(get_logger(), ">>> Scan Success at base pose!");
          return true;
      }

      int attempt_count = 0;

      // 4. 进入随机抖动循环，直到扫码成功 (has_new_scan_ 为 true)
      // 注意：这里加入了 rclcpp::ok() 检查，确保 Ctrl+C 能退出死循环
      while (rclcpp::ok() && !has_new_scan_) {
          attempt_count++;
          RCLCPP_WARN(get_logger(), "Scan failed, attempting random wiggle #%d...", attempt_count);

          // 基于基准点生成随机位姿
          auto random_pose = base_scan_pose;
          
          // 在 XYZ 三个方向上分别增加随机偏移 (-3cm ~ +3cm)
          random_pose.position.x += dist(gen);
          random_pose.position.y += dist(gen);
          random_pose.position.z += dist(gen);

          // 保持姿态(Orientation)不变，或者你也可以给姿态加微小的随机扰动
          // random_pose.orientation = ... 

          // 移动到随机点
          move_arm_sync(true, random_pose);

          // 在新位置停留等待 3.0 秒 (根据扫码枪灵敏度调整)
          // 如果在这期间 on_scan_result 回调被触发，wait_for_scan 会立即返回 true
          if (wait_for_scan(3.0)) {
              RCLCPP_INFO(get_logger(), ">>> Scan Success after %d wiggles!", attempt_count);
              return true;
          }
      }

      // 如果程序被关闭 (Ctrl+C)，返回 false
      return false; 
  }

  // 辅助：阻塞等待扫码结果，超时返回 false
  bool wait_for_scan(double timeout_sec) {
      auto start = std::chrono::steady_clock::now();
      while (rclcpp::ok()) {
          if (has_new_scan_) return true; // 收到码了

          auto now = std::chrono::steady_clock::now();
          double elapsed = std::chrono::duration<double>(now - start).count();
          if (elapsed > timeout_sec) return false; // 超时

          std::this_thread::sleep_for(100ms);
      }
      return false;
  }

  // --------------------------------------------------------
  // 辅助函数: 同步移动机械臂 (保持不变)
  // --------------------------------------------------------
  bool move_arm_sync(bool is_right, const geometry_msgs::msg::Pose& target) {
      auto goal_msg = ArmMove::Goal();
      goal_msg.is_right_arm = is_right;
      goal_msg.target_pose = target;
      goal_msg.max_velocity_scale = 0.5;

      auto send_goal_options = rclcpp_action::Client<ArmMove>::SendGoalOptions();
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

  // (其余辅助函数 get_object_pose, control_gripper 等保持不变)
  std::optional<geometry_msgs::msg::Pose> get_object_pose() {
      if (!service_client_->service_is_ready()) return std::nullopt;
      auto req = std::make_shared<GetBestPose::Request>();
      auto future = service_client_->async_send_request(req);
      if (future.wait_for(1s) != std::future_status::ready) return std::nullopt;
      auto res = future.get();
      if (!res->success) return std::nullopt;
      
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