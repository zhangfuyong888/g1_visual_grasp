#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#include <cmath>
#include <atomic>
#include <map>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <unitree_hg/msg/low_cmd.hpp>
#include <unitree_hg/msg/low_state.hpp>
#include <humanoid_grasp/g1.hpp> 

#include "humanoid_grasp/action/arm_move.hpp"

using namespace std::chrono_literals;
using LowCmd  = unitree_hg::msg::LowCmd;
using LowState= unitree_hg::msg::LowState;
using ArmMove = humanoid_grasp::action::ArmMove;
using GoalHandleArmMove = rclcpp_action::ServerGoalHandle<ArmMove>;
namespace rvt = rviz_visual_tools;

class G1DualArmActionServer : public rclcpp::Node {
public:
  // -------- 常量定义 --------
  static constexpr int NUM_ARM_JOINTS = 17; // 仅控制腰+双臂
  static constexpr auto NOT_USED_JOINT = G1Arm7JointIndex::NOT_USED_JOINT;

  // 这里的顺序对应 current_pos_ 的存储顺序，也是控制算法计算的顺序
  std::array<G1Arm7JointIndex, NUM_ARM_JOINTS> arm_joints_ = {
      G1Arm7JointIndex::LEFT_SHOULDER_PITCH, G1Arm7JointIndex::LEFT_SHOULDER_ROLL, G1Arm7JointIndex::LEFT_SHOULDER_YAW,
      G1Arm7JointIndex::LEFT_ELBOW, G1Arm7JointIndex::LEFT_WRIST_ROLL, G1Arm7JointIndex::LEFT_WRIST_PITCH, G1Arm7JointIndex::LEFT_WRIST_YAW,
      G1Arm7JointIndex::RIGHT_SHOULDER_PITCH, G1Arm7JointIndex::RIGHT_SHOULDER_ROLL, G1Arm7JointIndex::RIGHT_SHOULDER_YAW,
      G1Arm7JointIndex::RIGHT_ELBOW, G1Arm7JointIndex::RIGHT_WRIST_ROLL, G1Arm7JointIndex::RIGHT_WRIST_PITCH, G1Arm7JointIndex::RIGHT_WRIST_YAW,
      G1Arm7JointIndex::WAIST_YAW, G1Arm7JointIndex::WAIST_ROLL, G1Arm7JointIndex::WAIST_PITCH};

  std::array<float, NUM_ARM_JOINTS> home_pos_ = {
      0.2906F, 0.3078F, -0.0157F, 0.9779F, 0.1855F, -0.0855F, 0.0304F,   // left
      0.2836F,-0.3051F,  0.0182F, 0.9851F,-0.2342F, 0.0593F,-0.0431F,   // right
      0.0F, 0.0F, 0.0F                                                   // waist
  };

  G1DualArmActionServer() : Node("g1_arm_action_server") {
    // 参数声明
    this->declare_parameter<bool>("use_sim", false);
    this->get_parameter("use_sim", use_sim_);

    // 1. 通信接口
    lowcmd_pub_ = this->create_publisher<LowCmd>("/arm_sdk", 50);
    lowstate_sub_= this->create_subscription<LowState>("/lowstate", 5,
                   std::bind(&G1DualArmActionServer::onLowState, this, std::placeholders::_1));
    
    // JointState 发布器 (给 RViz/MoveIt 用)
    real_joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 50);
    debug_pos_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/debug/positions", 10);

    // 2. Action Server
    using namespace std::placeholders;
    action_server_ = rclcpp_action::create_server<ArmMove>(
      this, "arm_move",
      std::bind(&G1DualArmActionServer::handle_goal, this, _1, _2),
      std::bind(&G1DualArmActionServer::handle_cancel, this, _1),
      std::bind(&G1DualArmActionServer::handle_accepted, this, _1)
    );

    // 3. 启动控制线程
    current_pos_.fill(0.0F);
    
    if (use_sim_) {
        RCLCPP_WARN(get_logger(), ">>> SIMULATION MODE ACTIVATED <<<");
        current_pos_ = home_pos_; // 仿真模式初始化为 Home，防止全0
        state_ready_ = true;
    }

    ctrl_thread_ = std::thread([this]{ controlLoop(); });
    
    RCLCPP_INFO(get_logger(), "G1 Arm Action Server Started.");
  }

  ~G1DualArmActionServer() override {
    running_ = false;
    if (ctrl_thread_.joinable()) ctrl_thread_.join();
  }

  void initVisualTools() {
    using moveit::planning_interface::MoveGroupInterface;
    left_group_  = std::make_shared<MoveGroupInterface>(shared_from_this(),  "left_arm_group");
    right_group_ = std::make_shared<MoveGroupInterface>(shared_from_this(), "right_arm_group");
    
    // 设置最大速度比例
    left_group_->setMaxVelocityScalingFactor(1.0);
    right_group_->setMaxVelocityScalingFactor(1.0);

    visual_tools_ = std::make_unique<moveit_visual_tools::MoveItVisualTools>(
        shared_from_this(), "base_link", "rviz_visual_tools");
    visual_tools_->deleteAllMarkers();
  }

private:
  // -------- 任务上下文 --------
  struct TaskContext {
    bool is_right;
    std::vector<std::array<float, 7>> trajectory;
    std::atomic<bool> active{false};
    std::atomic<float> progress{0.0f};
    std::atomic<bool> cancel_requested{false};
  };
  std::shared_ptr<TaskContext> active_task_;
  std::mutex task_mtx_;

  // -------- ROS Handles --------
  bool use_sim_ = false;
  rclcpp::Publisher<LowCmd>::SharedPtr lowcmd_pub_;
  rclcpp::Subscription<LowState>::SharedPtr lowstate_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr real_joint_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr debug_pos_pub_;
  rclcpp_action::Server<ArmMove>::SharedPtr action_server_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> left_group_, right_group_;
  std::unique_ptr<moveit_visual_tools::MoveItVisualTools> visual_tools_;

  // -------- Control State --------
  std::array<float, NUM_ARM_JOINTS> current_pos_{};
  std::mutex state_mtx_;
  bool state_ready_{false};
  bool init_flag_{false};
  std::thread ctrl_thread_;
  std::atomic<bool> running_{true};
  
  float control_dt_{0.02F};
  float max_joint_velocity_base_{0.2F}; 
  float kp_{60.0F}, kd_{1.5F};

  // 全身关节名 (用于发布 JointState)
  // 注意：这个列表必须包含 URDF 中所有的非固定关节，顺序随意，因为我们用 name 匹配
  const std::vector<std::string> all_joint_names_ = {
      "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
      "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
      "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
      "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
      "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
      // 如果有手部关节，也加上，设为0即可
      "left_hand_index_0_joint", "left_hand_index_1_joint", "left_hand_middle_0_joint", "left_hand_middle_1_joint", "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
      "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint", "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint"
  };

  // SDK Index -> Joint Name 映射 (只映射我们控制的腰和手)
  // SDK: 0-5 LLeg, 6-11 RLeg, 12-14 Waist, 15-21 LArm, 22-28 RArm
  const std::map<int, std::string> sdk_to_name_map_ = {
      {12, "waist_yaw_joint"}, {13, "waist_roll_joint"}, {14, "waist_pitch_joint"},
      {15, "left_shoulder_pitch_joint"}, {16, "left_shoulder_roll_joint"}, {17, "left_shoulder_yaw_joint"}, {18, "left_elbow_joint"}, {19, "left_wrist_roll_joint"}, {20, "left_wrist_pitch_joint"}, {21, "left_wrist_yaw_joint"},
      {22, "right_shoulder_pitch_joint"}, {23, "right_shoulder_roll_joint"}, {24, "right_shoulder_yaw_joint"}, {25, "right_elbow_joint"}, {26, "right_wrist_roll_joint"}, {27, "right_wrist_pitch_joint"}, {28, "right_wrist_yaw_joint"}
  };

  // ================= Action Callbacks =================

  rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID &, std::shared_ptr<const ArmMove::Goal> goal) {
    RCLCPP_INFO(get_logger(), "Received goal request for %s arm", goal->is_right_arm ? "RIGHT" : "LEFT");
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandleArmMove>) {
    RCLCPP_INFO(get_logger(), "Received request to cancel goal");
    std::lock_guard<std::mutex> lk(task_mtx_);
    if (active_task_ && active_task_->active) {
        active_task_->cancel_requested = true;
    }
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandleArmMove> goal_handle) {
    std::thread{std::bind(&G1DualArmActionServer::execute, this, std::placeholders::_1), goal_handle}.detach();
  }

  // ================= Action Execution (Producer) =================

  void execute(const std::shared_ptr<GoalHandleArmMove> goal_handle) {
    auto goal = goal_handle->get_goal();
    auto result = std::make_shared<ArmMove::Result>();
    auto feedback = std::make_shared<ArmMove::Feedback>();

    feedback->state = "PLANNING";
    goal_handle->publish_feedback(feedback);

    auto &group = goal->is_right_arm ? *right_group_ : *left_group_;
    
    // 规划
    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(goal->target_pose);

    moveit_msgs::msg::RobotTrajectory traj;
    // 使用 Cartesian Path
    double fraction = group.computeCartesianPath(waypoints, 0.02, 3.0, traj);
    
    if (fraction < 0.9 || traj.joint_trajectory.points.empty()) {
        RCLCPP_ERROR(get_logger(), "Planning failed. Fraction: %.2f", fraction);
        result->success = false;
        result->message = "Planning failed";
        goal_handle->abort(result);
        return;
    }

    // 可视化
    visual_tools_->deleteAllMarkers();
    visual_tools_->publishAxisLabeled(goal->target_pose, "Target");
    visual_tools_->trigger();

    // 转换为任务
    auto new_task = std::make_shared<TaskContext>();
    new_task->is_right = goal->is_right_arm;
    
    for (const auto& point : traj.joint_trajectory.points) {
        if (point.positions.size() < 7) continue;
        std::array<float, 7> q_point;
        for (size_t i = 0; i < 7; ++i) q_point[i] = static_cast<float>(point.positions[i]);
        new_task->trajectory.push_back(q_point);
    }
    
    // 推送到消费者 (注意：绝对不要调用 group.execute)
    {
        std::lock_guard<std::mutex> lk(task_mtx_);
        new_task->active = true;
        active_task_ = new_task; 
    }
    RCLCPP_INFO(get_logger(), "Task pushed to controller. Points: %zu", new_task->trajectory.size());

    // 监控
    rclcpp::Rate r(10); 
    while (rclcpp::ok()) {
        if (goal_handle->is_canceling() || new_task->cancel_requested) {
            new_task->active = false; 
            result->success = false;
            result->message = "Canceled";
            goal_handle->canceled(result);
            return;
        }

        if (!new_task->active) {
            break; // 任务完成
        }

        feedback->state = "MOVING";
        feedback->progress = new_task->progress;
        goal_handle->publish_feedback(feedback);
        r.sleep();
    }

    result->success = true;
    result->message = "Completed";
    goal_handle->succeed(result);
    RCLCPP_INFO(get_logger(), "Action Succeeded.");
  }

  // ================= Control Loop (Consumer) =================

  void controlLoop() {
    // 仿真模式不等待 LowState
    if (!use_sim_) {
        RCLCPP_INFO(get_logger(), "Wait for LowState...");
        while(!state_ready_ && running_) std::this_thread::sleep_for(100ms);
    }
    
    RCLCPP_INFO(get_logger(), "Moving to HOME...");
    auto start_snapshot = currentSnapshotPos();
    moveToHome(home_pos_, start_snapshot, 4.0f);
    init_flag_ = true;
    RCLCPP_INFO(get_logger(), "Ready for Actions.");

    auto sleep_duration = std::chrono::milliseconds(static_cast<int>(control_dt_ * 1000));

    while (running_ && rclcpp::ok()) {
        // [SIM MODE] 如果是仿真，必须主动发布关节状态，否则 RViz 是死的
        if (use_sim_) {
            publishSimJointState(currentSnapshotPos());
        }

        std::shared_ptr<TaskContext> task;
        {
            std::lock_guard<std::mutex> lk(task_mtx_);
            task = active_task_;
        }

        if (!task || !task->active || task->trajectory.empty()) {
            std::this_thread::sleep_for(10ms); // 提高一点空闲频率
            continue;
        }

        RCLCPP_INFO(get_logger(), ">>> Controller Start Trajectory");
        std::array<float, NUM_ARM_JOINTS> base_state = currentSnapshotPos();

        for (size_t i = 0; i < task->trajectory.size(); ++i) {
            if (task->cancel_requested) break;

            task->progress = (float)i / (float)task->trajectory.size();

            std::array<float, 7> start_q, end_q;
            end_q = task->trajectory[i];
            
            if (i == 0) {
                 auto curr = currentSnapshotPos();
                 for(int k=0; k<7; ++k) start_q[k] = curr[task->is_right ? 7+k : k];
            } else {
                 start_q = task->trajectory[i-1];
            }

            float max_diff = 0.0f;
            for(int k=0; k<7; ++k) max_diff = std::max(max_diff, std::abs(end_q[k]-start_q[k]));
            if (max_diff < 1e-5) continue;
            
            float duration = max_diff / max_joint_velocity_base_;
            if (duration < control_dt_) duration = control_dt_;
            
            int steps = std::max(1, static_cast<int>(duration / control_dt_));
            std::array<float, 7> seg_vel;
            for(int k=0; k<7; ++k) seg_vel[k] = (end_q[k]-start_q[k])/duration;

            for (int s=1; s<=steps; ++s) {
                if (task->cancel_requested) break;
                float t = (float)s / steps;
                
                std::array<float, NUM_ARM_JOINTS> cmd_q = base_state;
                std::array<float, NUM_ARM_JOINTS> cmd_dq = {};

                for(int k=0; k<7; ++k) {
                    float val = start_q[k] + (end_q[k] - start_q[k]) * t;
                    int idx = task->is_right ? 7+k : k;
                    cmd_q[idx] = val;
                    cmd_dq[idx] = seg_vel[k];
                }
                
                {
                    std::lock_guard<std::mutex> lk(state_mtx_);
                    if (use_sim_) {
                        // 仿真模式：直接修改当前位置，不发 SDK
                        current_pos_ = cmd_q; 
                        publishSimJointState(current_pos_);
                    } else {
                        // 真实模式：发送 SDK
                        sendLowCmd(cmd_q, cmd_dq);
                    }
                    // Debug
                    std_msgs::msg::Float32MultiArray p_msg;
                    for(float v:cmd_q) p_msg.data.push_back(v);
                    debug_pos_pub_->publish(p_msg);
                }
                std::this_thread::sleep_for(sleep_duration);
            }
        }
        
        task->active = false; 
        
        // 发送最后一帧
        {
            std::lock_guard<std::mutex> lk(state_mtx_);
            std::array<float, NUM_ARM_JOINTS> final_q = base_state;
            std::array<float, NUM_ARM_JOINTS> final_dq = {};
            auto last_pt = task->trajectory.back();
            for(int k=0; k<7; ++k) final_q[task->is_right?7+k:k] = last_pt[k];
            
            if (use_sim_) {
                current_pos_ = final_q;
                publishSimJointState(current_pos_);
            } else {
                sendLowCmd(final_q, final_dq);
            }
        }
        RCLCPP_INFO(get_logger(), "<<< Controller Finished Trajectory");
    }
  }

  // ================= Helpers =================

  void onLowState(const LowState::SharedPtr msg) {
    if (use_sim_) return; // 仿真模式忽略真实反馈

    std::lock_guard<std::mutex> lk(state_mtx_);
    for (size_t i = 0; i < arm_joints_.size(); ++i) {
      int idx = static_cast<int>(arm_joints_[i]);
      current_pos_[i] = msg->motor_state[idx].q;
    }
    state_ready_ = true;
    
    // 真实模式下，将 LowState 转换为 JointState 发给 RViz
    publishSimJointState(current_pos_);
  }

  // 统一发布 JointState 给 RViz (29关节)
  void publishSimJointState(const std::array<float, NUM_ARM_JOINTS>& arm_pos) {
      sensor_msgs::msg::JointState js;
      js.header.stamp = this->now();
      js.header.frame_id = "base_link"; // 请根据 URDF 确认
      
      // 我们控制的部分是 arm_pos (17个)，其余部分补 0
      // 必须填充 all_joint_names_ 里的所有关节
      
      // 将 arm_pos 填入临时 map 以便查找
      std::map<std::string, float> pos_map;
      
      // 1. 映射 arm_pos (腰+双臂) 到 map
      // arm_pos 顺序: L_Arm(7), R_Arm(7), Waist(3)
      // 注意：这里需要根据你的 current_pos_ 定义手动匹配
      // SDK Index: 0-5 LLeg, 6-11 RLeg, 12-14 Waist, 15-21 LArm, 22-28 RArm
      
      // Waist (arm_pos 14,15,16) -> SDK 12,13,14
      pos_map[sdk_to_name_map_.at(12)] = arm_pos[14];
      pos_map[sdk_to_name_map_.at(13)] = arm_pos[15];
      pos_map[sdk_to_name_map_.at(14)] = arm_pos[16];

      // L Arm (arm_pos 0-6) -> SDK 15-21
      for(int i=0; i<7; ++i) pos_map[sdk_to_name_map_.at(15+i)] = arm_pos[i];

      // R Arm (arm_pos 7-13) -> SDK 22-28
      for(int i=0; i<7; ++i) pos_map[sdk_to_name_map_.at(22+i)] = arm_pos[7+i];

      // 2. 组装 JointState 消息
      for (const auto& name : all_joint_names_) {
          js.name.push_back(name);
          if (pos_map.find(name) != pos_map.end()) {
              js.position.push_back(pos_map[name]);
          } else {
              js.position.push_back(0.0); // 腿部和手部默认为0
          }
          js.velocity.push_back(0.0);
          js.effort.push_back(0.0);
      }
      real_joint_pub_->publish(js);
  }

  void sendLowCmd(const std::array<float, NUM_ARM_JOINTS>& q, const std::array<float, NUM_ARM_JOINTS>& dq) {
    LowCmd cmd;
    for (size_t i = 0; i < arm_joints_.size(); ++i) {
      int idx = static_cast<int>(arm_joints_[i]);
      cmd.motor_cmd[idx].q = q[i];
      cmd.motor_cmd[idx].dq = dq[i];
      cmd.motor_cmd[idx].kp = (i >= NUM_ARM_JOINTS-3) ? kp_*5.0f : kp_*3.0f;
      cmd.motor_cmd[idx].kd = kd_ * 4.0f;
    }
    cmd.motor_cmd[static_cast<int>(NOT_USED_JOINT)].q = 1.0F;
    lowcmd_pub_->publish(cmd);
  }

  std::array<float, NUM_ARM_JOINTS> currentSnapshotPos() {
    std::lock_guard<std::mutex> lk(state_mtx_);
    return current_pos_;
  }

  void moveToHome(const std::array<float, NUM_ARM_JOINTS>& target, std::array<float, NUM_ARM_JOINTS>& current, float duration) {
     int steps = duration / control_dt_;
     std::array<float, NUM_ARM_JOINTS> diff;
     for(int i=0;i<NUM_ARM_JOINTS;++i) diff[i] = (target[i] - current[i]) / steps;
     auto cmd = current;
     for(int s=0; s<steps; ++s) {
         std::array<float, NUM_ARM_JOINTS> dq{};
         for(int i=0;i<NUM_ARM_JOINTS;++i) { cmd[i] += diff[i]; }
         { 
             std::lock_guard<std::mutex> lk(state_mtx_); 
             if (use_sim_) {
                 current_pos_ = cmd;
                 publishSimJointState(cmd);
             } else {
                 sendLowCmd(cmd, dq); 
             }
         }
         std::this_thread::sleep_for(std::chrono::milliseconds((int)(control_dt_*1000)));
     }
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<G1DualArmActionServer>();
  node->initVisualTools();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}