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

// ROS 2 Core
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

// Messages
#include "std_msgs/msg/float32_multi_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

// MoveIt
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>

// Unitree SDK Messages
#include <unitree_hg/msg/low_cmd.hpp>
#include <unitree_hg/msg/low_state.hpp>
#include <humanoid_grasp/g1.hpp> 

// Custom Action Interface
#include "humanoid_grasp/action/arm_move.hpp"

using namespace std::chrono_literals;

// 类型别名
using LowCmd  = unitree_hg::msg::LowCmd;
using LowState= unitree_hg::msg::LowState;
using ArmMove = humanoid_grasp::action::ArmMove;
using GoalHandleArmMove = rclcpp_action::ServerGoalHandle<ArmMove>;
namespace rvt = rviz_visual_tools;

// 用命令行发送 Action 目标示例:
// ros2 action send_goal /arm_move humanoid_grasp/action/ArmMove "{
//   is_right_arm: true, 
//   target_pose: {
//     position: {x: 0.3, y: -0.04, z: 0.05}, 
//     orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
//   }, 
//   max_velocity_scale: 0.5
// }" --feedback

//rqt_plot 使用示例:
// 查看速度曲线 /debug/velocity/data[0] -> data[6]
// 查看关节曲线 /debug/joints/data[0] -> data[6]

/**
 * @class G1DualArmActionServer
 * @brief G1 双臂控制服务端
 * 核心功能：
 * 1. 接收 Action 目标 (Pose)
 * 2. 调用 MoveIt 规划笛卡尔路径
 * 3. 将路径转换为密集点列
 * 4. [优化] 使用五次多项式插值进行平滑控制
 */
class G1DualArmActionServer : public rclcpp::Node {
public:
  // -------- 常量定义 --------
  static constexpr int NUM_ARM_JOINTS = 17; // 控制关节数: 腰(3) + 左臂(7) + 右臂(7)
  static constexpr auto NOT_USED_JOINT = G1Arm7JointIndex::NOT_USED_JOINT;

  // 关节索引映射 (对应 current_pos_ 的顺序)
  std::array<G1Arm7JointIndex, NUM_ARM_JOINTS> arm_joints_ = {
      // 左臂 (7)
      G1Arm7JointIndex::LEFT_SHOULDER_PITCH, G1Arm7JointIndex::LEFT_SHOULDER_ROLL, G1Arm7JointIndex::LEFT_SHOULDER_YAW,
      G1Arm7JointIndex::LEFT_ELBOW, G1Arm7JointIndex::LEFT_WRIST_ROLL, G1Arm7JointIndex::LEFT_WRIST_PITCH, G1Arm7JointIndex::LEFT_WRIST_YAW,
      // 右臂 (7)
      G1Arm7JointIndex::RIGHT_SHOULDER_PITCH, G1Arm7JointIndex::RIGHT_SHOULDER_ROLL, G1Arm7JointIndex::RIGHT_SHOULDER_YAW,
      G1Arm7JointIndex::RIGHT_ELBOW, G1Arm7JointIndex::RIGHT_WRIST_ROLL, G1Arm7JointIndex::RIGHT_WRIST_PITCH, G1Arm7JointIndex::RIGHT_WRIST_YAW,
      // 腰部 (3)
      G1Arm7JointIndex::WAIST_YAW, G1Arm7JointIndex::WAIST_ROLL, G1Arm7JointIndex::WAIST_PITCH
  };

  // 安全的 Home 姿态 (T-Pose 变种，手臂微抬，避免穿模)
  std::array<float, NUM_ARM_JOINTS> home_pos_ = {
      // 左臂: 抬肩，微屈肘
      0.2906F, 0.3078F, -0.0157F, 0.9779F, 0.1855F, -0.0855F, 0.0304F,   
      // 右臂: 镜像
      0.2836F,-0.3051F,  0.0182F, 0.9851F,-0.2342F, 0.0593F,-0.0431F,   
      // 腰部: 居中
      0.0F, 0.0F, 0.0F                                                   
  };

  G1DualArmActionServer() : Node("g1_arm_action_server") {
    RCLCPP_INFO(get_logger(), ">>> Initializing G1DualArmActionServer...");

    // 1. 参数声明
    this->declare_parameter<bool>("use_sim", false);
    this->get_parameter("use_sim", use_sim_);
    
    if (use_sim_) {
        RCLCPP_WARN(get_logger(), "!!! 仿真模式已激活 !!! (不需要连接真实机器人)");
        RCLCPP_WARN(get_logger(), "!!! 将直接发布关节状态驱动 RViz");
    }

    // 2. 通信接口
    lowcmd_pub_ = this->create_publisher<LowCmd>("/arm_sdk", 50);
    lowstate_sub_= this->create_subscription<LowState>("/lowstate", 5,
                   std::bind(&G1DualArmActionServer::onLowState, this, std::placeholders::_1));
    
    // JointState 发布器 (RViz/MoveIt 需要)
    real_joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 50);
    
    // 调试数据发布
    debug_joints_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/debug/joints", 10);
    debug_velocity_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/debug/velocity", 10);

    // 3. Action Server
    using namespace std::placeholders;
    action_server_ = rclcpp_action::create_server<ArmMove>(
      this, 
      "arm_move",
      std::bind(&G1DualArmActionServer::handle_goal, this, _1, _2),
      std::bind(&G1DualArmActionServer::handle_cancel, this, _1),
      std::bind(&G1DualArmActionServer::handle_accepted, this, _1)
    );
    RCLCPP_INFO(get_logger(), "Action Server 'arm_move' created.");

    // 4. 状态初始化
    current_pos_.fill(0.0F);
    
    if (use_sim_) {
        // 仿真模式直接初始化为 Home，防止全0导致 TF 报错
        current_pos_ = home_pos_; 
        state_ready_ = true; 
    }

    // 5. 启动控制线程 (与 ROS 回调分离)
    ctrl_thread_ = std::thread([this]{ controlLoop(); });
    
    RCLCPP_INFO(get_logger(), ">>> Initialization Complete.");
  }

  ~G1DualArmActionServer() override {
    running_ = false;
    if (ctrl_thread_.joinable()) ctrl_thread_.join();
  }

  // 初始化 MoveIt (必须在 spin 之后调用)
  void initVisualTools() {
    RCLCPP_INFO(get_logger(), "Initializing MoveIt interfaces...");
    using moveit::planning_interface::MoveGroupInterface;
    
    left_group_  = std::make_shared<MoveGroupInterface>(shared_from_this(),  "left_arm_group");
    right_group_ = std::make_shared<MoveGroupInterface>(shared_from_this(), "right_arm_group");
    
    // 设置最大速度比例
    left_group_->setMaxVelocityScalingFactor(1.0);
    right_group_->setMaxVelocityScalingFactor(1.0);

    visual_tools_ = std::make_unique<moveit_visual_tools::MoveItVisualTools>(
        shared_from_this(), "base_link", "rviz_visual_tools");
    visual_tools_->deleteAllMarkers();
    RCLCPP_INFO(get_logger(), "MoveIt interfaces initialized.");
  }

private:
  // -------- 任务上下文 --------
  struct TaskContext {
    bool is_right;                                  // 目标手臂
    std::vector<std::array<float, 7>> trajectory;   // 密集轨迹点
    std::atomic<bool> active{false};                // 任务激活标志
    std::atomic<float> progress{0.0f};              // 进度 0.0~1.0
    std::atomic<bool> cancel_requested{false};      // 取消请求
  };
  std::shared_ptr<TaskContext> active_task_;        
  std::mutex task_mtx_;                             

  // -------- ROS Handles --------
  bool use_sim_ = false;
  rclcpp::Publisher<LowCmd>::SharedPtr lowcmd_pub_;
  rclcpp::Subscription<LowState>::SharedPtr lowstate_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr real_joint_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr debug_joints_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr debug_velocity_pub_;
  rclcpp_action::Server<ArmMove>::SharedPtr action_server_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> left_group_, right_group_;
  std::unique_ptr<moveit_visual_tools::MoveItVisualTools> visual_tools_;

  // -------- 控制状态 --------
  std::array<float, NUM_ARM_JOINTS> current_pos_{}; 
  std::mutex state_mtx_;                            
  bool state_ready_{false};                         
  bool init_flag_{false};                           
  std::thread ctrl_thread_;
  std::atomic<bool> running_{true};
  
  // -------- 控制参数 --------
  float control_dt_{0.02F};             // 50Hz 控制频率
  float max_joint_velocity_base_{0.2F}; // 基准速度 Rad/s
  float kp_{60.0F}, kd_{1.5F};          // PD 增益

  // -------- 关节映射 (URDF) --------
  const std::vector<std::string> all_joint_names_ = {
      "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
      "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
      "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
      "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
      "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
      "left_hand_index_0_joint", "left_hand_index_1_joint", "left_hand_middle_0_joint", "left_hand_middle_1_joint", "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
      "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint", "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint"
  };

  // SDK Index -> Name 映射
  const std::map<int, std::string> sdk_to_name_map_ = {
      {12, "waist_yaw_joint"}, {13, "waist_roll_joint"}, {14, "waist_pitch_joint"},
      {15, "left_shoulder_pitch_joint"}, {16, "left_shoulder_roll_joint"}, {17, "left_shoulder_yaw_joint"}, {18, "left_elbow_joint"}, {19, "left_wrist_roll_joint"}, {20, "left_wrist_pitch_joint"}, {21, "left_wrist_yaw_joint"},
      {22, "right_shoulder_pitch_joint"}, {23, "right_shoulder_roll_joint"}, {24, "right_shoulder_yaw_joint"}, {25, "right_elbow_joint"}, {26, "right_wrist_roll_joint"}, {27, "right_wrist_pitch_joint"}, {28, "right_wrist_yaw_joint"}
  };

  // ================= 辅助函数：五次多项式插值 =================
  
  // 计算五次多项式的位置系数 s(t)
  // t_norm: 归一化时间 [0, 1]
  // 返回: 位置比例 [0, 1]
  float getQuinticScaling(float t_norm) {
      // 限制范围防止越界
      t_norm = std::clamp(t_norm, 0.0f, 1.0f);
      // s(t) = 10t^3 - 15t^4 + 6t^5
      float t3 = t_norm * t_norm * t_norm;
      float t4 = t3 * t_norm;
      float t5 = t4 * t_norm;
      return 10.0f * t3 - 15.0f * t4 + 6.0f * t5;
  }

  // 计算五次多项式的速度系数 ds/dt
  // t_norm: 归一化时间 [0, 1]
  // 返回: 速度比例 (无量纲，需要除以总时间 T 才能得到真实速度)
  float getQuinticVelocityScaling(float t_norm) {
      t_norm = std::clamp(t_norm, 0.0f, 1.0f);
      // v_scale(t) = 30t^2 - 60t^3 + 30t^4
      float t2 = t_norm * t_norm;
      float t3 = t2 * t_norm;
      float t4 = t3 * t_norm;
      return 30.0f * t2 - 60.0f * t3 + 30.0f * t4;
  }

  // ================= Action 回调函数 =================

  rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID &, std::shared_ptr<const ArmMove::Goal> goal) {
    RCLCPP_INFO(get_logger(), "[Action] 收到目标: 移动 %s 臂", goal->is_right_arm ? "右" : "左");
    
    // 安全检查：如果机器人还未完成回零初始化，拒绝请求
    if (!init_flag_) {
        RCLCPP_WARN(get_logger(), "[Action] 拒绝: 机器人尚未初始化 (Waiting for Homing).");
        return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandleArmMove>) {
    RCLCPP_INFO(get_logger(), "[Action] 收到取消请求.");
    std::lock_guard<std::mutex> lk(task_mtx_);
    if (active_task_ && active_task_->active) {
        active_task_->cancel_requested = true;
    }
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandleArmMove> goal_handle) {
    // 启动新线程执行规划，避免阻塞 ROS 线程
    std::thread{std::bind(&G1DualArmActionServer::execute, this, std::placeholders::_1), goal_handle}.detach();
  }

  // ================= Action 执行线程 (生产者) =================

  void execute(const std::shared_ptr<GoalHandleArmMove> goal_handle) {
    auto goal = goal_handle->get_goal();
    auto result = std::make_shared<ArmMove::Result>();
    auto feedback = std::make_shared<ArmMove::Feedback>();

    RCLCPP_INFO(get_logger(), "[Action] 开始规划...");
    feedback->state = "PLANNING";
    goal_handle->publish_feedback(feedback);

    auto &group = goal->is_right_arm ? *right_group_ : *left_group_;
    group.setGoalPositionTolerance(0.03); // 3 cm 位置容差
    group.setGoalOrientationTolerance(0.3); // 约 17 度 方向容差

    // 1. 规划笛卡尔路径
    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(goal->target_pose);

    moveit_msgs::msg::RobotTrajectory traj;
    double fraction = group.computeCartesianPath(waypoints, 0.03, 0.0, traj);
    
    RCLCPP_INFO(get_logger(), "[Action] 规划覆盖率: %.2f%%", fraction * 100.0);

    if (fraction < 0.9 || traj.joint_trajectory.points.empty()) {
        RCLCPP_ERROR(get_logger(), "[Action] 规划失败 (覆盖率过低或路径为空).");
        result->success = false;
        result->message = "Planning failed";
        goal_handle->abort(result);
        return;
    }
    feedback->state = "PLAN RATE: " + std::to_string(fraction);
    goal_handle->publish_feedback(feedback);

    // 2. 可视化目标
    visual_tools_->deleteAllMarkers();
    visual_tools_->publishAxisLabeled(goal->target_pose, "Target");
    visual_tools_->trigger();

    // 3. 准备任务数据
    auto new_task = std::make_shared<TaskContext>();
    new_task->is_right = goal->is_right_arm;
    
    // 提取路径点
    for (const auto& point : traj.joint_trajectory.points) {
        if (point.positions.size() < 7) continue;
        std::array<float, 7> q_point;
        for (size_t i = 0; i < 7; ++i) q_point[i] = static_cast<float>(point.positions[i]);
        new_task->trajectory.push_back(q_point);
    }
    
    // 4. 推送到控制循环
    // 注意: 这里不调用 group.execute()，防止与 Unitree SDK 冲突
    {
        std::lock_guard<std::mutex> lk(task_mtx_);
        new_task->active = true;
        active_task_ = new_task; 
    }
    RCLCPP_INFO(get_logger(), "[Action] 任务已推送到控制器. 路径点数: %zu", new_task->trajectory.size());

    // 5. 监控执行进度
    rclcpp::Rate r(10); // 10Hz 反馈
    while (rclcpp::ok()) {
        if (goal_handle->is_canceling() || new_task->cancel_requested) {
            RCLCPP_WARN(get_logger(), "[Action] 任务被取消.");
            new_task->active = false; // 停止控制器
            result->success = false;
            result->message = "Canceled";
            goal_handle->canceled(result);
            return;
        }

        if (!new_task->active) {
            // 控制器完成任务后会将 active 设为 false
            break; 
        }

        feedback->state = "MOVING";
        feedback->progress = new_task->progress;
        goal_handle->publish_feedback(feedback);
        r.sleep();
    }

    result->success = true;
    result->message = "Completed";
    goal_handle->succeed(result);
    RCLCPP_INFO(get_logger(), "[Action] 动作执行成功.");
  }

  // ================= 底层控制线程 (消费者) =================

  void controlLoop() {
    RCLCPP_INFO(get_logger(), "[Ctrl] 控制线程启动.");

    // 1. 等待硬件连接 (仿真模式跳过)
    if (!use_sim_) {
        RCLCPP_INFO(get_logger(), "[Ctrl] 等待接收 LowState...");
        while(!state_ready_ && running_) std::this_thread::sleep_for(100ms);
        RCLCPP_INFO(get_logger(), "[Ctrl] 硬件已连接.");
    }
    
    // 2. 回零初始化 (使用五次多项式插值)
    RCLCPP_INFO(get_logger(), "[Ctrl] 正在移动到安全 HOME 姿态 (五次多项式插值)...");
    auto start_snapshot = currentSnapshotPos();
    // 使用新的 Quintic 版本回零，耗时 4 秒
    moveToHomeQuintic(home_pos_, start_snapshot, 4.0f); 
    
    init_flag_ = true;
    RCLCPP_INFO(get_logger(), "[Ctrl] 机器人初始化完成，等待指令.");

    auto sleep_duration = std::chrono::milliseconds(static_cast<int>(control_dt_ * 1000));

    // 3. 主循环
    while (running_ && rclcpp::ok()) {
        
        // [仿真模式] 主动发布 JointState 驱动 RViz
        if (use_sim_) {
            publishSimJointState(currentSnapshotPos());
        }

        // 获取任务
        std::shared_ptr<TaskContext> task;
        {
            std::lock_guard<std::mutex> lk(task_mtx_);
            task = active_task_;
        }

        // 如果没有任务，空闲等待
        if (!task || !task->active || task->trajectory.empty()) {
            std::this_thread::sleep_for(10ms); 
            continue;
        }

        RCLCPP_INFO(get_logger(), "[Ctrl] >>> 开始执行轨迹 (Arm: %s, Points: %zu)", 
                    task->is_right ? "Right" : "Left", task->trajectory.size());
        
        std::array<float, NUM_ARM_JOINTS> base_state = currentSnapshotPos();

        // 遍历路径点
        for (size_t i = 0; i < task->trajectory.size(); ++i) {
            if (task->cancel_requested) {
                RCLCPP_WARN(get_logger(), "[Ctrl] 执行被中断.");
                break;
            }

            task->progress = (float)i / (float)task->trajectory.size();

            std::array<float, 7> start_q, end_q;
            end_q = task->trajectory[i];
            
            // 确定当前段的起点
            if (i == 0) {
                 // 第一段：从当前实际位置开始
                 auto curr = currentSnapshotPos();
                 for(int k=0; k<7; ++k) start_q[k] = curr[task->is_right ? 7+k : k];
            } else {
                 start_q = task->trajectory[i-1];
            }

            // 计算该段耗时
            float max_diff = 0.0f;
            for(int k=0; k<7; ++k) max_diff = std::max(max_diff, std::abs(end_q[k]-start_q[k]));
            if (max_diff < 1e-5) continue; // 跳过微小移动
            
            float duration = max_diff / max_joint_velocity_base_;
            if (duration < control_dt_) duration = control_dt_;
            
            int steps = std::max(1, static_cast<int>(duration / control_dt_));
            std::array<float, 7> seg_vel;
            // 线性插值的速度是恒定的
            for(int k=0; k<7; ++k) seg_vel[k] = (end_q[k]-start_q[k])/duration;

            // 插值子循环
            for (int s=1; s<=steps; ++s) {
                if (task->cancel_requested) break;
                float t = (float)s / steps;
                
                std::array<float, NUM_ARM_JOINTS> cmd_q = base_state;
                std::array<float, NUM_ARM_JOINTS> cmd_dq = {};
                
                // 发布cmd_q,cmd_dp用于调试，在rqt_plot中查看,不分左右臂
                std_msgs::msg::Float32MultiArray joints_cmd_msg;
                std_msgs::msg::Float32MultiArray velocity_cmd_msg;

                // 更新活动关节
                for(int k=0; k<7; ++k) {
                    // 对于 MoveIt 生成的密集点列，我们在微小段内保持线性插值
                    // 这样做是为了保持整个轨迹的速度连续性，因为 MoveIt 已经做过速度规划
                    float val = start_q[k] + (end_q[k] - start_q[k]) * t;
                    int idx = task->is_right ? 7+k : k; 
                    cmd_q[idx] = val;
                    cmd_dq[idx] = seg_vel[k]; // 前馈速度

                    joints_cmd_msg.data.push_back(val); // 记录关节角发布
                    velocity_cmd_msg.data.push_back(seg_vel[k]); // 记录关节速度发布
                }

                // 发布cmd_q,cmd_dp用于调试信息
                debug_joints_pub_->publish(joints_cmd_msg);
                debug_velocity_pub_->publish(velocity_cmd_msg);

                // 发送指令
                {
                    std::lock_guard<std::mutex> lk(state_mtx_);
                    if (use_sim_) {
                        current_pos_ = cmd_q; 
                        publishSimJointState(current_pos_);
                    } else {
                        sendLowCmd(cmd_q, cmd_dq);
                    }
                }
                std::this_thread::sleep_for(sleep_duration);
            }
        }
        
        task->active = false; 
        
        // 发送最后一帧保持位置 (速度为0)
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
        RCLCPP_INFO(get_logger(), "[Ctrl] <<< 轨迹执行完毕.");
    }
  }

  // ================= 辅助工具函数 =================

  void onLowState(const LowState::SharedPtr msg) {
    if (use_sim_) return; // 仿真模式忽略真实反馈

    std::lock_guard<std::mutex> lk(state_mtx_);
    for (size_t i = 0; i < arm_joints_.size(); ++i) {
      int idx = static_cast<int>(arm_joints_[i]);
      current_pos_[i] = msg->motor_state[idx].q;
    }
    state_ready_ = true;
    
    // 将真实状态转发给 RViz
    publishSimJointState(current_pos_);
  }

  /**
   * @brief 发布 sensor_msgs::JointState 以驱动 RViz
   * 处理全机 29 个关节，补全未控制关节为 0
   */
  void publishSimJointState(const std::array<float, NUM_ARM_JOINTS>& arm_pos) {
      sensor_msgs::msg::JointState js;
      js.header.stamp = this->now();
      js.header.frame_id = "base_link"; // 请根据 URDF 确认根节点
      
      std::map<std::string, float> pos_map;
      
      // 填充腰部
      pos_map[sdk_to_name_map_.at(12)] = arm_pos[14];
      pos_map[sdk_to_name_map_.at(13)] = arm_pos[15];
      pos_map[sdk_to_name_map_.at(14)] = arm_pos[16];

      // 填充左臂
      for(int i=0; i<7; ++i) pos_map[sdk_to_name_map_.at(15+i)] = arm_pos[i];

      // 填充右臂
      for(int i=0; i<7; ++i) pos_map[sdk_to_name_map_.at(22+i)] = arm_pos[7+i];

      // 组装消息
      for (const auto& name : all_joint_names_) {
          js.name.push_back(name);
          if (pos_map.find(name) != pos_map.end()) {
              js.position.push_back(pos_map[name]);
          } else {
              js.position.push_back(0.0); // 腿部和手部默认为 0
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
      // 腰部刚度稍大
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

  /**
   * @brief [核心优化] 使用五次多项式插值进行平滑回零
   * 该函数确保机器人在启动时动作连贯，没有速度和加速度突变。
   */
  void moveToHomeQuintic(const std::array<float, NUM_ARM_JOINTS>& target, std::array<float, NUM_ARM_JOINTS>& start_pos, float duration) {
     if (duration < 1e-3) duration = 1.0;
     
     // 计算总步数
     int steps = static_cast<int>(duration / control_dt_);
     
     // 预计算差值
     std::array<float, NUM_ARM_JOINTS> diff;
     for(int i=0; i<NUM_ARM_JOINTS; ++i) diff[i] = target[i] - start_pos[i];
     
     for(int s=0; s<=steps; ++s) {
         // 当前归一化时间 t (0.0 -> 1.0)
         float t_norm = static_cast<float>(s) / steps;
         
         // 获取五次多项式系数
         float scale_pos = getQuinticScaling(t_norm);       // 位置系数 s(t)
         float scale_vel = getQuinticVelocityScaling(t_norm); // 速度系数 ds/dt
         
         std::array<float, NUM_ARM_JOINTS> cmd_q = start_pos;
         std::array<float, NUM_ARM_JOINTS> cmd_dq = {}; // 速度前馈
         
         for(int i=0; i<NUM_ARM_JOINTS; ++i) {
             // 位置 = 起点 + 总位移 * s(t)
             cmd_q[i] = start_pos[i] + diff[i] * scale_pos;
             
             // 速度 = 总位移 * (ds/dt) / 总时间
             // 注意: scale_vel 是对归一化时间的导数，需要除以 duration 变回真实时间导数
             cmd_dq[i] = diff[i] * scale_vel / duration;
         }

         { 
             std::lock_guard<std::mutex> lk(state_mtx_); 
             if (use_sim_) {
                 current_pos_ = cmd_q;
                 publishSimJointState(cmd_q);
             } else {
                 sendLowCmd(cmd_q, cmd_dq); 
             }
         }
         std::this_thread::sleep_for(std::chrono::milliseconds((int)(control_dt_*1000)));
     }
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<G1DualArmActionServer>();
  // 必须在 spin 之前初始化 MoveIt 相关工具
  node->initVisualTools();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}