#include <memory>
#include <chrono>
#include "rclcpp/rclcpp.hpp"

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <std_msgs/msg/bool.hpp>

#include <moveit_visual_tools/moveit_visual_tools.h>

#include <std_msgs/msg/float32_multi_array.hpp>

namespace rvt = rviz_visual_tools;

std_msgs::msg::Bool reach_flag ;

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

bool current_state = false; 

//读取当前的关节角
std_msgs::msg::Float32MultiArray arm_state_joints;
std::vector<double> arm_state_joints_vector;

// 回调函数：只更新位置，保持固定姿态
void TargetPoseCallback(const geometry_msgs::msg::Pose::SharedPtr msg) {
    RCLCPP_INFO(rclcpp::get_logger("pose_subscriber"), 
                "Received pose - Position: (%.2f, %.2f, %.2f)",
                msg->position.x, msg->position.y, msg->position.z);
    RCLCPP_INFO(rclcpp::get_logger("pose_subscriber"), 
                "Received pose - Orientation: (%.2f, %.2f, %.2f, %.2f)",
                msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
    // 只更新位置、姿态
    Previous_point = Target_point;
    Target_point.position = msg->position;
    Target_point.orientation = msg->orientation;

    // // 只更新位置，姿态保持固定
    // Previous_point = Target_point;
    // Target_point.position = msg->position;
    // Target_point.orientation = fixed_orientation;

    update_flag = true;
}

// void JointStateCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
//     arm_state_joints = *msg;

//     // for (size_t i = 0; i < arm_state_joints.data.size(); ++i) {
//     //     RCLCPP_INFO(rclcpp::get_logger("arm_current_state"), "Joint %zu: %.2f", i, arm_state_joints.data[i]);
//     // }
//     current_state = true;
// }

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto const node = std::make_shared<rclcpp::Node>(
        "arm_pose",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
    );

    auto const LOGGER = rclcpp::get_logger("arm_pose");

        // 创建订阅者
    auto target_pose_subscription = node->create_subscription<geometry_msgs::msg::Pose>(
        "target_pose_topic",
        10,
        TargetPoseCallback);
        // left //ros2 topic pub /target_pose_topic geometry_msgs/msg/Pose "{position: {x: 0.20, y: 0.35, z: 0.10}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}" --once
        // right //ros2 topic pub /target_pose_topic geometry_msgs/msg/Pose "{position: {x: 0.20, y: -0.35, z: 0.10}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}" --once


    // 创建发布目标角度
    auto target_joint_publisher = node->create_publisher<std_msgs::msg::Float32MultiArray>(
        "target_joint_topic",
        10);

        // 创建订阅者-机械臂关节角
    // auto current_state_subscription = node->create_subscription<std_msgs::msg::Float32MultiArray>(
    //     "arm_joint_state",
    //     10,
    //     JointStateCallback);
        //ros2 topic pub arm_joint_state std_msgs/msg/Float32MultiArray "{data: [0.0,0.0,0.0,0.0,0.0,0.0,0.0]}" --once
        
    auto reach_pub = node->create_publisher<std_msgs::msg::Bool>(
        "arm/reached_target",
        1);

    // 创建可视化工具
    moveit_visual_tools::MoveItVisualTools visual_tools(node, "base_link");
    visual_tools.waitForMarkerSub(0);
    visual_tools.deleteAllMarkers();

    // 显示文本
    Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
    text_pose.translation().z() = 0.5;
    visual_tools.publishText(text_pose, "unitree_robot_arm_grasping_simulation", rvt::WHITE, rvt::XXXLARGE);
    visual_tools.trigger();

    // 加载 kinematics 参数
    // 注意：需要通过 launch 文件或手动加载 kinematics.yaml
    // 这里添加日志提示
    RCLCPP_INFO(LOGGER, "确保已通过 launch 文件加载 kinematics.yaml 到参数服务器");
    RCLCPP_INFO(LOGGER, "或使用: ros2 param load <node_name> <path_to_kinematics.yaml>");

    // 创建MoveGroup接口
    using moveit::planning_interface::MoveGroupInterface;
    // auto move_group_interface = MoveGroupInterface(node, "left_arm_group");
    auto move_group_interface = MoveGroupInterface(node, "right_arm_group");
    
    // 设置规划参数
    move_group_interface.setPlanningTime(10.0);  // 增加规划时间
    move_group_interface.setNumPlanningAttempts(10);  // 增加尝试次数
    move_group_interface.setMaxVelocityScalingFactor(0.1);  // 降低速度
    move_group_interface.setMaxAccelerationScalingFactor(0.1);  // 降低加速度

    // 设置初始姿态（固定姿态）
    auto init_pose = []{
        geometry_msgs::msg::Pose msg;

        msg.orientation.w = 1.0;
        msg.orientation.x = 0.0;
        msg.orientation.y = 0.0;
        msg.orientation.z = 0.0;

        msg.position.x = 0.3047;
        msg.position.y = -0.0966;
        msg.position.z = 0.0952;
        return msg;
    }();
    Previous_point = init_pose;
    fixed_orientation = init_pose.orientation;  // 保存固定姿态
    Target_point = init_pose;



    // 主循环
    while (rclcpp::ok()) {
        rclcpp::spin_some(node);
        
        if (update_flag) { 
            update_flag = false;
            RCLCPP_INFO(LOGGER, "开始移动到新目标点");

            // 可视化新目标和路径
            visual_tools.deleteAllMarkers();  // 清除之前的标记
            visual_tools.publishAxisLabeled(Previous_point, "Pre_point");
            visual_tools.publishAxisLabeled(Target_point, "Tar_point");
            visual_tools.publishSphere(Target_point, rvt::GREEN, rvt::SMALL);
            
            std::vector<geometry_msgs::msg::Point> path;
            path.push_back(Previous_point.position);
            path.push_back(Target_point.position);
            visual_tools.publishPath(path, rvt::RED, rvt::MEDIUM);
            visual_tools.trigger();

            // 生成笛卡尔路径（保持固定姿态）
            std::vector<geometry_msgs::msg::Pose> waypoints;
            waypoints.push_back(Target_point);

            //规划路径
            moveit_msgs::msg::RobotTrajectory trajectory;
            double fraction = move_group_interface.computeCartesianPath(
                waypoints,
                0.01,  // 步长
                0.0,   // 避障容差
                trajectory);

            RCLCPP_INFO(LOGGER, "路径规划完成度: %.2f%%", fraction * 100.0);
            

            if (fraction > 0.9) {
                moveit::planning_interface::MoveGroupInterface::Plan cartesian_plan;
                cartesian_plan.trajectory_ = trajectory;

                // 获取关节轨迹中的最后一个点
                if (!cartesian_plan.trajectory_.joint_trajectory.points.empty()) {
                    const trajectory_msgs::msg::JointTrajectoryPoint& last_point = cartesian_plan.trajectory_.joint_trajectory.points.back();
                    // 打印最后一个点对应的关节角度信息
                    RCLCPP_INFO(LOGGER, "目标点关节角度息：");
                    target_joints.data.clear();
                    if (!is_left_right_arm) {
                        target_joints.data.push_back(0);
                    } else {
                        target_joints.data.push_back(1);
                    }
                    for (size_t i = 0; i < last_point.positions.size(); ++i) {
                        RCLCPP_INFO(LOGGER, "关节 %zu 角度: %.4f", i+1, last_point.positions[i]);
                        target_joints.data.push_back(last_point.positions[i]);
                    }
                    target_joint_publisher->publish(target_joints);
                    //到达目标点
                    // reach_flag.data = true;
                    // reach_pub->publish(reach_flag);
                } else {
                    RCLCPP_WARN(LOGGER, "轨迹的关节轨迹点为空，无法获取最后一个关节序列信息");
                }
                //执行路径
                move_group_interface.execute(cartesian_plan);
                RCLCPP_INFO(LOGGER, "定姿态移动完成");
            } else {
                RCLCPP_ERROR(LOGGER, "路径规划失败，无法到达目标点");
            }
        }
        rclcpp::sleep_for(std::chrono::milliseconds(100));  // 减少CPU占用
    }

    rclcpp::shutdown();
    return 0;
}