#include <memory>
#include <chrono>
#include "rclcpp/rclcpp.hpp"

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>

#include <moveit_visual_tools/moveit_visual_tools.h>

#include <std_msgs/msg/float32_multi_array.hpp>

#include <moveit_msgs/msg/display_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>


namespace rvt = rviz_visual_tools;


// 全局变量存储目标点（包含姿态）
geometry_msgs::msg::Pose Previous_point;
geometry_msgs::msg::Pose Target_point;
bool update_flag = false;
// 存储固定姿态
geometry_msgs::msg::Quaternion fixed_orientation;

//保存需要发布的关节角
std_msgs::msg::Float32MultiArray target_joints;
//选择左右臂，left=0，right=1
bool is_left_right_arm = true;

// 保存rviz拖动的目标关节角
std_msgs::msg::Float32MultiArray rviz_target_joints;
bool rviz_update_flag = false;

bool current_state = false; 

//读取当前的关节角
std_msgs::msg::Float32MultiArray arm_state_joints;
std::vector<double> arm_state_joints_vector;

//riviz plan后的轨迹
moveit_msgs::msg::RobotTrajectory rt;


void JointStateCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
    arm_state_joints = *msg;

    // for (size_t i = 0; i < arm_state_joints.data.size(); ++i) {
    //     RCLCPP_INFO(rclcpp::get_logger("arm_current_state"), "Joint %zu: %.2f", i, arm_state_joints.data[i]);
    // }
    current_state = true;
}

void DisplayTrajectoryCb(const moveit_msgs::msg::DisplayTrajectory::SharedPtr msg)
{
  auto LOGGER = rclcpp::get_logger("display_traj_sniffer");

  if (msg->trajectory.empty()) {
    RCLCPP_WARN(LOGGER, "DisplayTrajectory: trajectory is empty");
    return;
  }
  const auto& jt = msg->trajectory.front().joint_trajectory;
  if (jt.points.empty()) {
    RCLCPP_WARN(LOGGER, "DisplayTrajectory: joint_trajectory has no points");
    return;
  }

  // rt = msg->trajectory.front();
  // RCLCPP_INFO(LOGGER, "接收到轨迹");
  

  const auto& names = jt.joint_names;
  const auto& q_goal = jt.points.back().positions;   // 规划终点关节角（Plan 按钮生成）

  RCLCPP_INFO(LOGGER, "=== Planned goal joint positions (%zu joints) ===", q_goal.size());
  for (size_t i = 0; i < q_goal.size(); ++i) {
    // 若 joint_names 与控制器一致，打印名字更直观
    const char* n = (i < names.size() ? names[i].c_str() : "(unnamed)");
    RCLCPP_INFO(LOGGER, "  %s : %.6f", n, q_goal[i]);
  }

  // 如果你还想把这些角度发布到现有的 target_joint_topic：
  // 注意：你现有的消息第一个元素放了左右臂标记，这里也沿用
//   std_msgs::msg::Float32MultiArray rviz_target_joints;
  rviz_target_joints.data.reserve(1 + q_goal.size());
  rviz_target_joints.data.clear();
  rviz_target_joints.data.push_back(is_left_right_arm ? 1.0f : 0.0f);
  for (double v : q_goal) rviz_target_joints.data.push_back(static_cast<float>(v));


  rviz_update_flag = true;


}


int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto const node = std::make_shared<rclcpp::Node>(
        "click_and_move",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
    );

    auto const LOGGER = rclcpp::get_logger("click_and_move");


    auto display_traj_sub = node->create_subscription<moveit_msgs::msg::DisplayTrajectory>(
    "/display_planned_path", 10, &DisplayTrajectoryCb);

    // auto traj_pub = node->create_publisher<moveit_msgs::msg::RobotTrajectory>(
    // "/arm_joint_trajectory", rclcpp::QoS(1).transient_local());

    // 创建发布目标角度
    auto target_joint_publisher = node->create_publisher<std_msgs::msg::Float32MultiArray>(
        "target_joint_topic",
        10);

        // 创建订阅者-机械臂关节角
    auto current_state_subscription = node->create_subscription<std_msgs::msg::Float32MultiArray>(
        "arm_joint_state",
        10,
        JointStateCallback);
        //ros2 topic pub arm_joint_state std_msgs/msg/Float32MultiArray "{data: [0.0,0.0,0.0,0.0,0.0,0.0,0.0]}" --once
        

    // 主循环
    while (rclcpp::ok()) {
        rclcpp::spin_some(node);

        if (rviz_update_flag) { 
            rviz_update_flag = false;
            // traj_pub->publish(rt);
            target_joint_publisher->publish(rviz_target_joints);
        }

        rclcpp::sleep_for(std::chrono::milliseconds(100));  // 减少CPU占用
    }

    rclcpp::shutdown();
    return 0;
}