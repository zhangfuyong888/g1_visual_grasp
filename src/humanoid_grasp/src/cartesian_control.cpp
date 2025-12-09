#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#include <cmath> // 确保包含 abs, max

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "geometry_msgs/msg/pose.hpp"

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include <unitree_hg/msg/low_cmd.hpp>
#include <unitree_hg/msg/low_state.hpp>

#include <humanoid_grasp/g1.hpp>   // 提供 G1Arm7JointIndex 枚举

using namespace std::chrono_literals;
using LowCmd  = unitree_hg::msg::LowCmd;
using LowState= unitree_hg::msg::LowState;
namespace rvt = rviz_visual_tools;

// ros2 topic pub left/target_pose geometry_msgs/msg/Pose "{position: {x: 0.15, y: 0.15, z: 0.12}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}" --once
// ros2 topic pub right/target_pose geometry_msgs/msg/Pose "{position: {x: 0.15, y: -0.15, z: 0.15}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}" --once


class G1DualArmMotion : public rclcpp::Node {
public:
  // ---------------- 基本常量 ----------------
  static constexpr int NUM_ARM_JOINTS = 17;
  static constexpr auto NOT_USED_JOINT = G1Arm7JointIndex::NOT_USED_JOINT;

  // 关节索引映射（左7 + 右7 + 腰3）
  std::array<G1Arm7JointIndex, NUM_ARM_JOINTS> arm_joints_ = {
      G1Arm7JointIndex::LEFT_SHOULDER_PITCH,
      G1Arm7JointIndex::LEFT_SHOULDER_ROLL,
      G1Arm7JointIndex::LEFT_SHOULDER_YAW,
      G1Arm7JointIndex::LEFT_ELBOW,
      G1Arm7JointIndex::LEFT_WRIST_ROLL,
      G1Arm7JointIndex::LEFT_WRIST_PITCH,
      G1Arm7JointIndex::LEFT_WRIST_YAW,
      G1Arm7JointIndex::RIGHT_SHOULDER_PITCH,
      G1Arm7JointIndex::RIGHT_SHOULDER_ROLL,
      G1Arm7JointIndex::RIGHT_SHOULDER_YAW,
      G1Arm7JointIndex::RIGHT_ELBOW,
      G1Arm7JointIndex::RIGHT_WRIST_ROLL,
      G1Arm7JointIndex::RIGHT_WRIST_PITCH,
      G1Arm7JointIndex::RIGHT_WRIST_YAW,
      G1Arm7JointIndex::WAIST_YAW,
      G1Arm7JointIndex::WAIST_ROLL,
      G1Arm7JointIndex::WAIST_PITCH};

  // 站立初始（可按需要调整）
  std::array<float, NUM_ARM_JOINTS> home_pos_ = {
      0.2906F, 0.1478F, -0.0157F, 0.9779F, 0.1855F, -0.0855F, 0.0304F,   // left 7
      0.2836F,-0.2251F,  0.0182F, 0.9851F,-0.2342F, 0.0593F,-0.0431F,   // right 7
      0.0F, 0.0F, 0.0F                                                   // waist 3
  };

  G1DualArmMotion() : Node("g1_dual_arm_motion") {
    // ---- 通信接口 ----
    lowcmd_pub_ = this->create_publisher<LowCmd>("/arm_sdk", 50);
    lowstate_sub_= this->create_subscription<LowState>("/lowstate", 5,
                   std::bind(&G1DualArmMotion::onLowState, this, std::placeholders::_1));

    reach_pub_right = this->create_publisher<std_msgs::msg::Bool>("/arm_right/reached_target", 1);
    reach_pub_left = this->create_publisher<std_msgs::msg::Bool>("/arm_left/reached_target", 1);

    jointstate_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/arm_joint_state", 10);

    // 左/右臂目标位姿两个独立接口
    left_pose_sub_  = this->create_subscription<geometry_msgs::msg::Pose>(
        "/left/target_pose", 10,
        std::bind(&G1DualArmMotion::onLeftPose, this, std::placeholders::_1));

    right_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
        "/right/target_pose", 10,
        std::bind(&G1DualArmMotion::onRightPose, this, std::placeholders::_1));

    // ---- 控制参数 ----
    sleep_time_ = std::chrono::milliseconds(static_cast<int>(control_dt_ * 1000));

    // 初值
    current_pos_.fill(0.0F);

    // 控制线程
    ctrl_thread_ = std::thread([this]{ controlLoop(); });
  }

  ~G1DualArmMotion() override {
    running_ = false;
    if (ctrl_thread_.joinable()) ctrl_thread_.join();
  }

  void initVisualTools() {
    // ---- MoveIt & 可视化 ----
    using moveit::planning_interface::MoveGroupInterface;
    left_group_  = std::make_shared<MoveGroupInterface>(shared_from_this(),  "left_arm_group");
    right_group_ = std::make_shared<MoveGroupInterface>(shared_from_this(), "right_arm_group");

    visual_tools_ = std::make_unique<moveit_visual_tools::MoveItVisualTools>(
        shared_from_this(), "base_link", "rviz_visual_tools");
    visual_tools_->deleteAllMarkers();

    // 显示文本
    Eigen::Isometry3d text_pose = Eigen::Isometry3d::Identity();
    text_pose.translation().z() = 0.5;
    visual_tools_->publishText(text_pose, "g1_arms_grab_demo", rvt::WHITE, rvt::XXXLARGE);
    visual_tools_->trigger();
  }


private:
  // -------- ROS 句柄 --------
  rclcpp::Publisher<LowCmd>::SharedPtr lowcmd_pub_;
  rclcpp::Subscription<LowState>::SharedPtr lowstate_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr reach_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr jointstate_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr left_pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr right_pose_sub_;
  std::unique_ptr<moveit_visual_tools::MoveItVisualTools> visual_tools_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> left_group_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> right_group_;

  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr reach_pub_right;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr reach_pub_left;

  // -------- 状态/控制缓存 --------
  std::array<float, NUM_ARM_JOINTS> current_pos_{};
  std::mutex state_mtx_;
  bool state_ready_{false};

  bool init_flag_{false};

  // ----------------------------------------------------------------
  // [修改] ArmGoal 现在存储整个轨迹，而不仅仅是终点
  // ----------------------------------------------------------------
  struct ArmGoal {
    bool is_right;                 // true: 右臂，false: 左臂
    std::vector<std::array<float, 7>> trajectory;   // 存储轨迹中所有的点
  };

  std::deque<ArmGoal> goal_queue_;
  std::mutex goal_mtx_;

  // 控制参数
  float control_dt_{0.02F};
  float kp_{60.0F}, kd_{1.5F};
  float max_joint_velocity_{0.2F}; // 轨迹执行时的最大速度限制 rad/s
  std::chrono::milliseconds sleep_time_{};

  std::thread ctrl_thread_;
  std::atomic<bool> running_{true};


  // ----------------- 回调 -----------------
  void onLowState(const LowState::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(state_mtx_);
    for (size_t i = 0; i < arm_joints_.size(); ++i) {
      current_pos_[i] = msg->motor_state[static_cast<int>(arm_joints_[i])].q;
    }
    state_ready_ = true;

    // 把当前17关节发布出去
    std_msgs::msg::Float32MultiArray out;
    out.data.reserve(NUM_ARM_JOINTS);
    for (float q : current_pos_) out.data.push_back(q);
    jointstate_pub_->publish(out);
  }

  void onLeftPose(const geometry_msgs::msg::Pose::SharedPtr pose) {
    planAndEnqueue(false /*left*/, *pose);
  }

  void onRightPose(const geometry_msgs::msg::Pose::SharedPtr pose) {
    planAndEnqueue(true /*right*/, *pose);
  }

  // ----------------- 规划 & 入队 -----------------
  void planAndEnqueue(bool is_right, const geometry_msgs::msg::Pose& target_pose) {
    RCLCPP_INFO(this->get_logger(), "Got pose.");
    auto &group = is_right ? *right_group_ : *left_group_;

    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(target_pose);

    moveit_msgs::msg::RobotTrajectory traj;
    // computeCartesianPath 生成密集路径点
    double fraction = group.computeCartesianPath(waypoints, 0.02 /*eef_step*/, 3.0 /*jump*/, traj);
    RCLCPP_INFO(this->get_logger(), "Visualizing plan (Cartesian path) (%.2f%% achieved)", fraction * 100.0);
    group.setGoalOrientationTolerance(0.2);
    group.setGoalPositionTolerance(0.005);
    RCLCPP_INFO(this->get_logger(), "[%s] Cartesian fraction: %.1f%%",
                is_right ? "RIGHT":"LEFT", 100.0 * fraction);

    if (fraction <= 0.9 || traj.joint_trajectory.points.empty()) {
      RCLCPP_ERROR(this->get_logger(), "[%s] planning failed or empty trajectory",
                   is_right ? "RIGHT":"LEFT");
      return;
    }

    group.asyncExecute(traj);
    
    // 可视化
    visual_tools_->deleteAllMarkers();
    visual_tools_->publishAxisLabeled(target_pose, is_right ? "R_Target_pose":"L_Target_pose");
    visual_tools_->publishSphere(target_pose, rvt::GREEN, rvt::LARGE);
    visual_tools_->trigger();

    RCLCPP_INFO(this->get_logger(), "moveit update done... Trajectory points: %zu", traj.joint_trajectory.points.size());

    // ----------------------------------------------------------------
    // [修改] 提取轨迹中所有的点，而不仅仅是最后一个
    // ----------------------------------------------------------------
    ArmGoal g;
    g.is_right = is_right;
    
    for (const auto& point : traj.joint_trajectory.points) {
        if (point.positions.size() < 7) continue;
        std::array<float, 7> q_point;
        for (size_t i = 0; i < 7; ++i) {
            q_point[i] = static_cast<float>(point.positions[i]);
        }
        g.trajectory.push_back(q_point);
    }

    {
      std::lock_guard<std::mutex> lk(goal_mtx_);
      goal_queue_.push_back(g);
    }
    RCLCPP_INFO(this->get_logger(), "[%s] trajectory goal enqueued with %zu points", 
                is_right ? "RIGHT":"LEFT", g.trajectory.size());
  }

  // ----------------- 控制主循环 -----------------
  void controlLoop() {
    // 先回 home
    RCLCPP_INFO(this->get_logger(), "Move to home...");
    auto home_last_state = currentSnapshot();
    // 初始回零仍然使用平滑的五次多项式插值
    moveTo(home_pos_, home_last_state, 4.0F, true);
    
    while (!init_flag_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting moving home...");
        std::this_thread::sleep_for(100ms);
    }
    RCLCPP_INFO(this->get_logger(), "Init done. Ready for goals...");

    while (running_ && rclcpp::ok()) {

      ArmGoal g;
      {
        std::lock_guard<std::mutex> lk(goal_mtx_);
        if (goal_queue_.empty()) {
          std::this_thread::sleep_for(50ms);
          continue;
        }
        g = goal_queue_.front();
        goal_queue_.pop_front();
      }

      RCLCPP_INFO(this->get_logger(), "Executing %s arm trajectory (%zu points)...",
                  g.is_right ? "RIGHT":"LEFT", g.trajectory.size());

      // [新增] 专门的轨迹执行函数
      executeTrajectory(g);

      // 验证最终是否到达
      if (g.trajectory.empty()) continue;
      
      // 构造最后的目标点用于校验
      std::array<float, NUM_ARM_JOINTS> final_target = currentSnapshot();
      const auto& final_q7 = g.trajectory.back();
      if (g.is_right) {
         for(int i=0; i<7; ++i) final_target[7+i] = final_q7[i];
      } else {
         for(int i=0; i<7; ++i) final_target[i] = final_q7[i];
      }

      float max_err = 0.0F;
      auto reach_last_state = currentSnapshot();
      for (size_t i = 0; i < final_target.size(); ++i){
        max_err = std::max(max_err, std::abs(final_target[i] - reach_last_state[i]));
      }
      
      if (max_err < 0.05F) {
        std_msgs::msg::Bool flag; flag.data = true;   
        if(g.is_right) reach_pub_right->publish(flag);
        else reach_pub_left->publish(flag);
        RCLCPP_INFO(this->get_logger(), "%s arm reached.", g.is_right ? "RIGHT":"LEFT");
      } else {
        RCLCPP_INFO(this->get_logger(), "%s arms joint error high : %f", g.is_right ? "RIGHT":"LEFT", max_err);
      }
    }
  }

  // ----------------------------------------------------------------
  // [新增] 轨迹执行函数
  // 遍历所有路径点，在相邻点之间进行线性插值（保证速度连续性）
  // ----------------------------------------------------------------
  void executeTrajectory(const ArmGoal& g) {
      if (g.trajectory.empty()) return;

      // 锁定非活动关节：获取当前的完整状态，作为整个轨迹执行期间的基础
      // 这样在移动一只手臂时，腰部和另一只手臂会保持在轨迹开始时的位置
      std::array<float, NUM_ARM_JOINTS> base_state = currentSnapshot();
      
      // 我们需要从机器人的当前位置开始平滑过渡到轨迹的第一个点
      // 为了简单起见，且由于computeCartesianPath通常从当前位置开始，
      // 我们假设轨迹的起点就是当前位置（或者非常接近）。
      
      // 遍历轨迹中的每一段 (Point A -> Point B)
      for (size_t i = 0; i < g.trajectory.size(); ++i) {
          
          // 确定当前段的 起点 和 终点
          std::array<float, 7> start_q, end_q;
          
          if (i == 0) {
              // 第一段：从当前电机反馈位置 -> 轨迹第0个点
              // 注意：为了更平滑，这里取 currentSnapshot 对应手臂的部分
              auto current_full = currentSnapshot();
              if (g.is_right) {
                  for(int k=0;k<7;++k) start_q[k] = current_full[7+k];
              } else {
                  for(int k=0;k<7;++k) start_q[k] = current_full[k];
              }
          } else {
              // 后续段：从上一个轨迹点 -> 当前轨迹点
              start_q = g.trajectory[i-1];
          }
          end_q = g.trajectory[i];

          // 1. 计算这一微小段需要的执行时间
          // 这里的策略是：距离 / 设定速度
          float max_diff = 0.0f;
          for (int k = 0; k < 7; ++k) {
              max_diff = std::max(max_diff, std::abs(end_q[k] - start_q[k]));
          }
          
          // 如果两点几乎重合，跳过
          if (max_diff < 1e-5f) continue;

          // 计算时间，限制最小时间为一个控制周期，防止除零
          float duration = max_diff / max_joint_velocity_; 
          if (duration < control_dt_) duration = control_dt_;

          // 2. 在这一段内进行插值发送
          // 注意：因为MoveIt生成的笛卡尔路径点非常密集（eef_step=0.02），
          // 点与点之间近似直线。如果我们在这里使用"五次多项式"（起点终点速度为0），
          // 机器人会在每个点停顿，造成剧烈抖动。
          // 因此，对于密集轨迹段，我们使用【线性插值】（Linear Interpolation），
          // 这等效于在该段保持【恒定速度】，从而实现流畅的连续运动。
          
          int steps = std::max(1, static_cast<int>(duration / control_dt_));
          
          // 预先计算该段的速度（前馈）
          std::array<float, 7> segment_vel;
          for(int k=0; k<7; ++k) {
              segment_vel[k] = (end_q[k] - start_q[k]) / duration;
          }

          for (int s = 1; s <= steps; ++s) {
              float progress = static_cast<float>(s) / static_cast<float>(steps); // 0.0 -> 1.0
              
              // 构造完整的 17 维命令
              std::array<float, NUM_ARM_JOINTS> cmd_q = base_state;
              std::array<float, NUM_ARM_JOINTS> cmd_dq = {}; // 默认为0

              // 填充活动臂的插值数据
              for (int k = 0; k < 7; ++k) {
                  // 线性插值位置: q = start + (end-start)*t
                  float val = start_q[k] + (end_q[k] - start_q[k]) * progress;
                  
                  // 填充到对应的全身索引中
                  int joint_idx = g.is_right ? (7 + k) : k;
                  
                  cmd_q[joint_idx] = val;
                  cmd_dq[joint_idx] = segment_vel[k]; // 速度前馈
                  RCLCPP_INFO(this->get_logger(), "Step %d/%d, Joint %d: q=%.4f, dq=%.4f",
                      s, steps, joint_idx, cmd_q[joint_idx], cmd_dq[joint_idx]);
              }

              // 发送命令
              {
                std::lock_guard<std::mutex> lk(state_mtx_);
                sendLowCmd(cmd_q, cmd_dq);
              }
              
              // 保持频率
              std::this_thread::sleep_for(sleep_time_);
          }
          RCLCPP_INFO(this->get_logger(), "[%s] Executed segment %zu/%zu",
                      g.is_right ? "RIGHT":"LEFT", i+1, g.trajectory.size());
      }
      
      // 轨迹结束，发送最后一个点，速度归零
      {
          std::lock_guard<std::mutex> lk(state_mtx_);
          std::array<float, NUM_ARM_JOINTS> final_q = base_state;
          std::array<float, NUM_ARM_JOINTS> final_dq = {}; // 零速
          const auto& last_pt = g.trajectory.back();
          for(int k=0; k<7; ++k) {
              int joint_idx = g.is_right ? (7 + k) : k;
              final_q[joint_idx] = last_pt[k];
          }
          sendLowCmd(final_q, final_dq);
      }
  }


  // 当前位置快照（线程安全取值）
  std::array<float, NUM_ARM_JOINTS> currentSnapshot() {
    std::lock_guard<std::mutex> lk(state_mtx_);
    return current_pos_;
  }

  // -----------------------------------------------------------
  // [保留] 原有的 moveTo (五次多项式)，仅用于初始化回零 (Home)
  // -----------------------------------------------------------
  void moveTo(const std::array<float, NUM_ARM_JOINTS>& target,
              std::array<float, NUM_ARM_JOINTS>& current,
              float duration_sec, bool /*smooth*/) {

    const int steps = std::max(1, static_cast<int>(duration_sec / control_dt_));
    std::array<float, NUM_ARM_JOINTS> q_start = current;
    std::array<float, NUM_ARM_JOINTS> q_des  = q_start;
    std::array<float, NUM_ARM_JOINTS> dq_des = {};

    for (int i = 0; i <= steps; ++i) {
      float t = static_cast<float>(i) * control_dt_;
      if (t > duration_sec) t = duration_sec;

      float s = (duration_sec > 1e-6f) ? (t / duration_sec) : 1.0f;
      s = std::clamp(s, 0.0f, 1.0f);

      // 五次多项式位置系数
      float s2 = s * s;
      float s3 = s2 * s;
      float s4 = s3 * s;
      float s5 = s4 * s;
      float blend_pos = 10.0f * s3 - 15.0f * s4 + 6.0f * s5;

      // 五次多项式速度系数 (d/dt)
      float blend_vel = 0.0f;
      if (duration_sec > 1e-6f) {
        blend_vel = (30.0f * s2 - 60.0f * s3 + 30.0f * s4) / duration_sec;
      }

      {
        std::lock_guard<std::mutex> lk(state_mtx_);
        for (size_t j = 0; j < arm_joints_.size(); ++j) {
          float dq_total = target[j] - q_start[j];
          q_des[j]  = q_start[j] + blend_pos * dq_total;
          dq_des[j] = blend_vel * dq_total;
        }
        sendLowCmd(q_des, dq_des);
      }
      std::this_thread::sleep_for(sleep_time_);
    }
    // 结束定点
    {
      std::lock_guard<std::mutex> lk(state_mtx_);
      std::array<float, NUM_ARM_JOINTS> dq_zero{};
      sendLowCmd(target, dq_zero);
    }
    init_flag_ = true;
  }

  // 发送位置 + 速度期望
  void sendLowCmd(const std::array<float, NUM_ARM_JOINTS>& q,
                  const std::array<float, NUM_ARM_JOINTS>& dq)
  {
    LowCmd cmd;
    for (size_t i = 0; i < arm_joints_.size(); ++i) {
      int idx = static_cast<int>(arm_joints_[i]);

      cmd.motor_cmd[idx].q   = q[i];   // 位置参考
      cmd.motor_cmd[idx].dq  = dq[i];  // 速度参考
      cmd.motor_cmd[idx].tau = 0.0F;   

      // 增益参数
      if (i >= arm_joints_.size() - 3) {   // 腰部 3 个
        cmd.motor_cmd[idx].kp = kp_ * 5.0F;   
        cmd.motor_cmd[idx].kd = kd_ * 4.0F;
      } else {                             // 双臂 14 个
        cmd.motor_cmd[idx].kp = kp_ * 3.0F;   
        cmd.motor_cmd[idx].kd = kd_ * 4.0F;   
      }
    }
    cmd.motor_cmd[static_cast<int>(NOT_USED_JOINT)].q = 1.0F;
    lowcmd_pub_->publish(cmd);
  }

};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<G1DualArmMotion>();
  node->initVisualTools();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}