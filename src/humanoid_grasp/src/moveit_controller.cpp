#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#include <sstream>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "sensor_msgs/msg/joint_state.hpp"  //   给 MoveIt / robot_state_publisher 发布 joint_states

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

//   旧版：MoveIt 时间参数化（IPP）；这版我们不用它做实机控制，只保留头文件以备后面需要
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>

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

  //   对应 URDF/SRDF 中的关节名（务必和你的 URDF 一致）
  std::array<std::string, NUM_ARM_JOINTS> joint_names_ = {
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
      "left_wrist_pitch_joint",
      "left_wrist_yaw_joint",
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
      "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
      "waist_yaw_joint",
      "waist_roll_joint",
      "waist_pitch_joint"
  };

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
    reach_pub_left  = this->create_publisher<std_msgs::msg::Bool>("/arm_left/reached_target", 1);

    jointstate_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/arm_joint_state", 10);

    //   给 MoveIt / robot_state_publisher 用
    js_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 50);

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

    RCLCPP_INFO(this->get_logger(), "Visual tools init...");
  }

private:
  // -------- ROS 句柄 --------
  rclcpp::Publisher<LowCmd>::SharedPtr lowcmd_pub_;
  rclcpp::Subscription<LowState>::SharedPtr lowstate_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr reach_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr jointstate_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr js_pub_;   //  

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

  //   保存一整条轨迹（每个点的 7 关节位置 + 速度）
  struct ArmTraj {
    bool is_right;                         // true: 右臂，false: 左臂
    std::vector<std::array<float, 7>> q;   // 每个点 7 关节位置
    std::vector<std::array<float, 7>> dq;  // 每个点 7 关节速度（我们自己用 Δq/Δt 算）
    double planned_duration{0.0};          //  MoveIt 规划总时长
  };

  std::deque<ArmTraj> traj_queue_;
  std::mutex goal_mtx_;

  // 控制参数
  float control_dt_{0.02F};
  float kp_{60.0F}, kd_{1.5F};
  float max_joint_velocity_{0.2F}; // 用来限制我们重建出来的 dq
  std::chrono::milliseconds sleep_time_{};

  std::thread ctrl_thread_;
  std::atomic<bool> running_{true};

  // ----------------- 回调 -----------------
  void onLowState(const LowState::SharedPtr msg) {
    {
      std::lock_guard<std::mutex> lk(state_mtx_);
      for (size_t i = 0; i < arm_joints_.size(); ++i) {
        current_pos_[i] = msg->motor_state[static_cast<int>(arm_joints_[i])].q;
      }
      state_ready_ = true;
    }

    // 1）原来的 Float32MultiArray（保留）
    std_msgs::msg::Float32MultiArray out;
    out.data.reserve(NUM_ARM_JOINTS);
    for (float q : current_pos_) out.data.push_back(q);
    jointstate_pub_->publish(out);

    // 2）  发布标准 JointState，供 MoveIt / robot_state_publisher 使用
    sensor_msgs::msg::JointState js;
    js.header.stamp = this->now();
    js.name.reserve(NUM_ARM_JOINTS);
    js.position.reserve(NUM_ARM_JOINTS);

    {
      std::lock_guard<std::mutex> lk(state_mtx_);
      for (size_t i = 0; i < NUM_ARM_JOINTS; ++i) {
        js.name.push_back(joint_names_[i]);     // 名字要和 URDF 对上
        js.position.push_back(current_pos_[i]); // 关节角
      }
    }

    // js_pub_->publish(js);
    // 如果连接真机器人，取消注释，要实时读取机器人真实状态
  }

  void onLeftPose(const geometry_msgs::msg::Pose::SharedPtr pose) {
    planAndEnqueue(false /*left*/, *pose);
  }

  void onRightPose(const geometry_msgs::msg::Pose::SharedPtr pose) {
    planAndEnqueue(true /*right*/, *pose);
  }

  // ----------------- 规划 & 入队（整条轨迹） -----------------
  void planAndEnqueue(bool is_right, const geometry_msgs::msg::Pose& target_pose) {
    auto &group = is_right ? *right_group_ : *left_group_;
    group.setStartStateToCurrentState();

    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(target_pose);

    moveit_msgs::msg::RobotTrajectory traj;
    group.setGoalOrientationTolerance(0.1);// 设置目标姿态容差
    group.setGoalPositionTolerance(0.005);
    double fraction = group.computeCartesianPath(waypoints, 0.005 /*eef_step*/, 3.0 /*jump*/, traj);

    RCLCPP_INFO(this->get_logger(), "[%s] Cartesian fraction: %.1f%%",
                is_right ? "RIGHT":"LEFT", 100.0 * fraction);

    if (fraction <= 0.9 || traj.joint_trajectory.points.empty()) {
      RCLCPP_ERROR(this->get_logger(), "[%s] planning failed or empty trajectory",
                   is_right ? "RIGHT":"LEFT");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "[%s] moveit trajectory computed, points: %zu",
                is_right ? "RIGHT":"LEFT",
                traj.joint_trajectory.points.size());

    //   将 MoveIt 轨迹转换成我们自己的 ArmTraj
    //   1）位置 q7[i] 直接用 MoveIt 的 positions
    //   2）速度 dq7[i] 不再用 MoveIt 的 velocities，而是用 Δq/Δt 自己重建并限速
    ArmTraj arm_traj;
    arm_traj.is_right = is_right;

    const auto &points = traj.joint_trajectory.points;
    if (points.empty()) {
      RCLCPP_ERROR(this->get_logger(), "[%s] joint_trajectory is empty", is_right ? "RIGHT":"LEFT");
      return;
    }

    // 预先取出每个点的 time_from_start（秒）
    std::vector<double> t_list(points.size(), 0.0);
    for (size_t idx = 0; idx < points.size(); ++idx) {
      const auto &pt = points[idx];
      double t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9;
      t_list[idx] = t;
    }

    // 轨迹规划的总时间（MoveIt 理论时间）
    double planned_duration = t_list.back();
    RCLCPP_INFO(this->get_logger(), "[%s] MoveIt planned duration: %.3f s",
                is_right ? "RIGHT" : "LEFT", planned_duration);

    //记到 ArmTraj 里, 如果只有一个点，则设为 1s
    if (planned_duration < 0.1){
      arm_traj.planned_duration = 1;
    }  else {
      arm_traj.planned_duration = planned_duration;
    }
   


    for (size_t idx = 0; idx < points.size(); ++idx) {
      const auto &pt = points[idx];

      if (pt.positions.size() < 7) {
        RCLCPP_ERROR(this->get_logger(), "[%s] point %zu has less than 7 joints (%zu)",
                     is_right ? "RIGHT" : "LEFT", idx, pt.positions.size());
        return;
      }

      std::array<float, 7> q7{};
      std::array<float, 7> dq7{};

      // ---- 关节位置 ----
      for (size_t i = 0; i < 7; ++i) {
        q7[i] = static_cast<float>(pt.positions[i]);
      }

      // ---- 我们用 Δq / Δt 重建速度 dq7 ----
      if (idx == 0) {
        // 第一个点：速度设为 0
        dq7.fill(0.0F);
      } else {
        const auto &pt_prev = points[idx - 1];
        double t  = t_list[idx];
        double t0 = t_list[idx - 1];
        double dt = t - t0;

        // 防止时间间隔太小或为 0
        if (dt < 1e-4) {
          dt = static_cast<double>(control_dt_);
        }

        for (size_t i = 0; i < 7; ++i) {
          double q_cur  = pt.positions[i];
          double q_prev = pt_prev.positions[i];
          double v = (q_cur - q_prev) / dt;  // Δq / Δt

          // 对速度做限幅，避免过大
          double vmax = static_cast<double>(max_joint_velocity_);
          if (v >  vmax) v =  vmax;
          if (v < -vmax) v = -vmax;

          dq7[i] = static_cast<float>(v);
        }
      }

      arm_traj.q.push_back(q7);
      arm_traj.dq.push_back(dq7);
    }

    if (arm_traj.q.empty()) {
      RCLCPP_ERROR(this->get_logger(), "[%s] trajectory conversion empty", is_right ? "RIGHT":"LEFT");
      return;
    }

    //   调试打印：把这次规划出来的所有关节角和速度都打一遍
    {
      std::ostringstream oss;
      oss << (is_right ? "[RIGHT]" : "[LEFT]")
          << " planned trajectory, " << arm_traj.q.size() << " points:\n";

      for (size_t k = 0; k < arm_traj.q.size(); ++k) {
        oss << "  pt " << k << " q=[";
        for (int j = 0; j < 7; ++j) {
          oss << arm_traj.q[k][j];
          if (j < 6) oss << ", ";
        }
        oss << "] dq=[";
        for (int j = 0; j < 7; ++j) {
          oss << arm_traj.dq[k][j];
          if (j < 6) oss << ", ";
        }
        oss << "]\n";
      }

      RCLCPP_INFO_STREAM(this->get_logger(), oss.str());
    }

    // 可视化 target
    visual_tools_->deleteAllMarkers();
    visual_tools_->publishAxisLabeled(target_pose, is_right ? "R_Target_pose":"L_Target_pose");
    visual_tools_->publishSphere(target_pose, rvt::GREEN, rvt::LARGE);
    visual_tools_->trigger();

    //   同时让 MoveIt 执行（供 RViz 显示），这里是同步 blocking 的
    // 真机时如果由本节点直接发 LowCmd 控制，可以考虑注释掉这一句，避免双控制源。
    moveit::core::MoveItErrorCode exe_ret = group.execute(traj);
    if (!exe_ret) {
      RCLCPP_WARN(this->get_logger(), "[%s] MoveIt execute failed with code %d",
                  is_right ? "RIGHT" : "LEFT", exe_ret.val);
    }

    {
      std::lock_guard<std::mutex> lk(goal_mtx_);
      traj_queue_.push_back(std::move(arm_traj));   // FIFO，必须执行完一条才会执行下一条
    }
    RCLCPP_INFO(this->get_logger(), "[%s] trajectory enqueued", is_right ? "RIGHT":"LEFT");
  }

  // ----------------- 控制主循环 -----------------
  void controlLoop() {
    // 可以等待第一帧状态（若需要）
    while (running_ && !state_ready_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for /lowstate...");
      std::this_thread::sleep_for(100ms);
    }

    // // 先回 home
    RCLCPP_INFO(this->get_logger(), "Move to home...");
    auto home_last_state = currentSnapshot();
    moveTo(home_pos_, home_last_state, 8.0F, true);

    // // 等待移动到home
    while (!init_flag_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting moving home...");
      std::this_thread::sleep_for(100ms);
    }

    RCLCPP_INFO(this->get_logger(), "Ready for goals...");
    while (running_ && rclcpp::ok()) {
      ArmTraj traj;
      {
        std::lock_guard<std::mutex> lk(goal_mtx_);
        if (traj_queue_.empty()) {
          std::this_thread::sleep_for(50ms);
          continue;
        }
        traj = std::move(traj_queue_.front());
        traj_queue_.pop_front();
      }

      executeArmTrajectory(traj);
    }
  }

  // ----------------- 执行一整条 MoveIt 轨迹 -----------------
  void executeArmTrajectory(const ArmTraj &arm_traj) {
    if (arm_traj.q.empty()) return;

    RCLCPP_INFO(this->get_logger(), "Executing %s arm trajectory with %zu points",
                arm_traj.is_right ? "RIGHT" : "LEFT",
                arm_traj.q.size());

    //  如果 MoveIt 没给时间（=0），就用10s
    double total_T = arm_traj.planned_duration;
    if (total_T <= 1e-4 || arm_traj.q.size() <= 1) {
      // total_T = arm_traj.q.size() * static_cast<double>(control_dt_);
      total_T = 10.0;
      RCLCPP_WARN(this->get_logger(),
                  "%s arm traj has no valid planned duration, fallback total_T = %.3f s",
                  arm_traj.is_right ? "RIGHT" : "LEFT",
                  total_T);
    }

    // 每两个点之间的播放间隔
    double dt_point = total_T / std::max<size_t>(1, arm_traj.q.size() - 1);

    
    // 记录开始时间
    auto exec_start = std::chrono::steady_clock::now();

    // 逐点发送（控制频率由 control_dt_ 决定）
    for (size_t k = 0; k < arm_traj.q.size() && running_ && rclcpp::ok(); ++k) {
      auto q_all  = currentSnapshot();          // 当前 17 关节位置（另一侧 + 腰保持当前）
      std::array<float, NUM_ARM_JOINTS> dq_all{};
      dq_all.fill(0.0F);

      if (arm_traj.is_right) {
        // 右臂关节在 [7..13]
        for (int i = 0; i < 7; ++i) {
          q_all[7 + i]  = arm_traj.q[k][i];
          dq_all[7 + i] = arm_traj.dq[k][i];
        }
      } else {
        // 左臂关节在 [0..6]
        for (int i = 0; i < 7; ++i) {
          q_all[i]   = arm_traj.q[k][i];
          dq_all[i]  = arm_traj.dq[k][i];
        }
      }

      sendLowCmd(q_all, dq_all);
      //  按规划时间来 sleep
      std::this_thread::sleep_for(std::chrono::duration<double>(dt_point));
    }

    // 结束时再发一帧精准到位，速度清零 + 做误差检查
    {
      auto q_all  = currentSnapshot();
      std::array<float, NUM_ARM_JOINTS> dq_zero{};
      dq_zero.fill(0.0F);

      const auto &last_q = arm_traj.q.back();
      if (arm_traj.is_right) {
        for (int i = 0; i < 7; ++i) {
          q_all[7 + i] = last_q[i];
        }
      } else {
        for (int i = 0; i < 7; ++i) {
          q_all[i] = last_q[i];
        }
      }

      sendLowCmd(q_all, dq_zero);

      // 简单的到达验证
      float max_err = 0.0F;
      auto reach_last_state = currentSnapshot();
      for (size_t i = 0; i < q_all.size(); ++i) {
        max_err = std::max(max_err, std::abs(q_all[i] - reach_last_state[i]));
      }
      

      //测试
      std_msgs::msg::Bool test_flag; 
      test_flag.data = true; 
      if(arm_traj.is_right) {
        reach_pub_right->publish(test_flag);
      } else {
        reach_pub_left->publish(test_flag);
      } 

      if (max_err < 0.05F) {
        std_msgs::msg::Bool flag; 
        flag.data = true;   
        if(arm_traj.is_right){
          reach_pub_right->publish(flag);
        } else{
          reach_pub_left->publish(flag);
        } 
        RCLCPP_INFO(this->get_logger(), "%s arm reached.", arm_traj.is_right ? "RIGHT":"LEFT");
      } else {
        // 离线输出时，误差很大
        RCLCPP_INFO(this->get_logger(), "%s arm error high : %f",
                    arm_traj.is_right ? "RIGHT":"LEFT", max_err);
      }
    }

    //  记录结束时间并打印总耗时
    auto exec_end = std::chrono::steady_clock::now();
    double exec_sec = std::chrono::duration<double>(exec_end - exec_start).count();
    RCLCPP_INFO(this->get_logger(),
                "%s arm trajectory EXECUTED in %.3f s (points: %zu)",
                arm_traj.is_right ? "RIGHT" : "LEFT",
                exec_sec,
                arm_traj.q.size());
  }

  // 当前位置快照（线程安全取值）
  std::array<float, NUM_ARM_JOINTS> currentSnapshot() {
    std::lock_guard<std::mutex> lk(state_mtx_);
    return current_pos_;
  }

  float estimateMoveDuration(const std::array<float, NUM_ARM_JOINTS>& current,
                             const std::array<float, NUM_ARM_JOINTS>& target)
  {
    float max_delta = 0.0F;
    float max_velocity = 0.2F;
    for (size_t i = 0; i < NUM_ARM_JOINTS; ++i)
      max_delta = std::max(max_delta, std::abs(target[i] - current[i]));

    // 防止除 0
    float duration = max_delta / max_velocity;
    duration = std::clamp(duration, 4.0F, 20.0F); // 限制最短2秒，最长20秒
    return duration;
  }

  //// 原版插值发送（保留注释）
  // void moveTo(const std::array<float, NUM_ARM_JOINTS>& target,
  //             std::array<float, NUM_ARM_JOINTS>& current,
  //             float duration_sec, bool smooth) {
  //   const int steps = std::max(1, static_cast<int>(duration_sec / control_dt_));
  //   const float max_delta = max_joint_velocity_ * control_dt_;
  //   std::array<float, NUM_ARM_JOINTS> command_pos_ = current; 
  //   for (int i=0;i<steps;++i) {
  //     float phase = static_cast<float>(i) / static_cast<float>(steps);
  //     {
  //       std::lock_guard<std::mutex> lk(state_mtx_);
  //       for (size_t j=0;j<arm_joints_.size();++j) {
  //         if (smooth) {
  //           command_pos_[j] = current[j] * (1.0F - phase) + target[j] * phase;
  //         } else {
  //           float diff = target[j] - command_pos_[j];
  //           float step = std::clamp(diff, -max_delta, max_delta);
  //           command_pos_[j] += step;
  //         }
  //       }
  //       sendLowCmd(command_pos_);
  //     }
  //     std::this_thread::sleep_for(sleep_time_);
  //   }
  //   // 结束时发一帧精准到位
  //   {
  //     std::lock_guard<std::mutex> lk(state_mtx_);
  //     sendLowCmd(target);
  //   }
  //   init_flag_ = true;
  // }

  // 多项式插值（五次多项式）版本，带速度前馈 —— 用于回 home
  void moveTo(const std::array<float, NUM_ARM_JOINTS>& target,
              std::array<float, NUM_ARM_JOINTS>& current,
              float duration_sec, bool /*smooth*/) {

    // 总步数
    const int steps = std::max(1, static_cast<int>(duration_sec / control_dt_));

    // 起始关节位置快照
    std::array<float, NUM_ARM_JOINTS> q_start = current;

    // 这两个数组用来存每一拍要发出去的位置和速度
    std::array<float, NUM_ARM_JOINTS> q_des  = q_start;
    std::array<float, NUM_ARM_JOINTS> dq_des = {};

    for (int i = 0; i <= steps; ++i) {
      float t = static_cast<float>(i) * control_dt_;
      if (t > duration_sec) t = duration_sec;

      // 归一化时间 s ∈ [0,1]
      float s = (duration_sec > 1e-6f) ? (t / duration_sec) : 1.0f;
      s = std::clamp(s, 0.0f, 1.0f);

      // 五次多项式 0-1 blend: 位置
      float s2 = s * s;
      float s3 = s2 * s;
      float s4 = s3 * s;
      float s5 = s4 * s;

      float blend_pos = 10.0f * s3 - 15.0f * s4 + 6.0f * s5;

      // 五次多项式的一阶导（对 s 求导）：速度部分
      // d/ds (10 s^3 - 15 s^4 + 6 s^5) = 30 s^2 - 60 s^3 + 30 s^4
      // dq/dt = blend_vel * (target - start) ，其中 ds/dt = 1/T
      float blend_vel = 0.0f;
      if (duration_sec > 1e-6f) {
        blend_vel = (30.0f * s2 - 60.0f * s3 + 30.0f * s4) / duration_sec;
      }

      {
        std::lock_guard<std::mutex> lk(state_mtx_);

        for (size_t j = 0; j < arm_joints_.size(); ++j) {
          float dq_total = target[j] - q_start[j];

          // 位置参考
          q_des[j]  = q_start[j] + blend_pos * dq_total;
          // 速度参考
          dq_des[j] = blend_vel * dq_total;
        }

        // 同时下发位置 + 速度
        sendLowCmd(q_des, dq_des);
      }

      std::this_thread::sleep_for(sleep_time_);
    }

    // 最后一帧强制发目标位置，速度清零
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
      cmd.motor_cmd[idx].dq  = dq[i];  // 速度参考（我们重建并限速后的值）
      cmd.motor_cmd[idx].tau = 0.0F;   // 先不加前馈力矩

      // 增益可以稍微柔一点（你后面可以根据手感再调）
      if (i >= arm_joints_.size() - 3) {   // 腰部 3 个
        cmd.motor_cmd[idx].kp = kp_ * 5.0F;   // 原来是 *7
        cmd.motor_cmd[idx].kd = kd_ * 4.0F;
      } else {                             // 双臂 14 个
        cmd.motor_cmd[idx].kp = kp_ * 3.0F;   // 原来是 *4
        cmd.motor_cmd[idx].kd = kd_ * 4.0F;   // 稍微大一点
      }
    }

    // NOT_USED_JOINT 按原来逻辑保持
    cmd.motor_cmd[static_cast<int>(NOT_USED_JOINT)].q = 1.0F;

    lowcmd_pub_->publish(cmd);
  }

  // 原版只发位置的版本（保留注释）
  // void sendLowCmd(const std::array<float, NUM_ARM_JOINTS>& q) {
  //   LowCmd cmd;
  //   for (size_t i=0;i<arm_joints_.size();++i) {
  //     int idx = static_cast<int>(arm_joints_[i]);
  //     cmd.motor_cmd[idx].q   = q[i];
  //     cmd.motor_cmd[idx].dq  = 0.0F;
  //     cmd.motor_cmd[idx].tau = 0.0F;
  //     // 腰部更高刚度
  //     if (i >= arm_joints_.size()-3) {
  //       cmd.motor_cmd[idx].kp = kp_ * 7.0F;
  //       cmd.motor_cmd[idx].kd = kd_  * 4.0F;
  //     } else {
  //       cmd.motor_cmd[idx].kp = kp_ * 4.0F;
  //       cmd.motor_cmd[idx].kd = kd_  * 3.0F;
  //     }
  //   }
  //   cmd.motor_cmd[static_cast<int>(NOT_USED_JOINT)].q = 1.0F;
  //   lowcmd_pub_->publish(cmd);
  // }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<G1DualArmMotion>();
  node->initVisualTools();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
