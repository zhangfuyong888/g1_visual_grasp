#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

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

  struct ArmGoal {
    bool is_right;                 // true: 右臂，false: 左臂
    std::array<float, 7> q_goal;   // 对应臂的7个关节目标
  };

  std::deque<ArmGoal> goal_queue_;
  std::mutex goal_mtx_;

  // 控制参数
  float control_dt_{0.02F};
  float kp_{60.0F}, kd_{1.5F};
  float max_joint_velocity_{0.2F};
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

    // 把当前17关节发布出去（如需给 MoveIt 同步）
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
    double fraction = group.computeCartesianPath(waypoints, 0.02 /*eef_step*/, 3.0 /*jump*/, traj);
    RCLCPP_INFO(this->get_logger(), "Visualizing plan 4 (Cartesian path) (%.2f%% acheived)", fraction * 100.0);
    group.setGoalOrientationTolerance(0.2);// 设置目标姿态容差
    group.setGoalPositionTolerance(0.005);
    RCLCPP_INFO(this->get_logger(), "[%s] Cartesian fraction: %.1f%%",
                is_right ? "RIGHT":"LEFT", 100.0 * fraction);

    if (fraction <= 0.9 || traj.joint_trajectory.points.empty()) {
      RCLCPP_ERROR(this->get_logger(), "[%s] planning failed or empty trajectory",
                   is_right ? "RIGHT":"LEFT");
      return;
    }

    group.asyncExecute(traj);
    
    // 可视化（可选）
    visual_tools_->deleteAllMarkers();
    visual_tools_->publishAxisLabeled(target_pose, is_right ? "R_Target_pose":"L_Target_pose");
    visual_tools_->publishSphere(target_pose, rvt::GREEN, rvt::LARGE);
    visual_tools_->trigger();

    RCLCPP_INFO(this->get_logger(), "moveit update done...");

    const auto& last = traj.joint_trajectory.points.back();
    if (last.positions.size() < 7) {
      RCLCPP_ERROR(this->get_logger(), "[%s] expected 7 joints, got %zu",
                   is_right ? "RIGHT":"LEFT", last.positions.size());
      return;
    }

    ArmGoal g{is_right, {}};
    for (size_t i=0; i<7; ++i) g.q_goal[i] = static_cast<float>(last.positions[i]);

    {
      std::lock_guard<std::mutex> lk(goal_mtx_);
      goal_queue_.push_back(g);
    }
    RCLCPP_INFO(this->get_logger(), "[%s] goal enqueued", is_right ? "RIGHT":"LEFT");


  }

  // ----------------- 控制主循环 -----------------
  void controlLoop() {
    //测试
    // 等待第一帧状态
    // while (running_ && !state_ready_) {
    //   RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for /lowstate...");
    //   std::this_thread::sleep_for(100ms);
    // }
    

    // 先回 home
    RCLCPP_INFO(this->get_logger(), "Move to home...");
    auto home_last_state = currentSnapshot();
    moveTo(home_pos_, home_last_state, 4.0F, true);
    // 等待移动到home
    while (!init_flag_)
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting moving home...");
        std::this_thread::sleep_for(100ms);
    }
    RCLCPP_INFO(this->get_logger(), "Init done.");
    RCLCPP_INFO(this->get_logger(), "Ready for goals...");
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

      // 生成一个 17 关节的目标数组：另一侧和腰保持当前值
      std::array<float, NUM_ARM_JOINTS> target = currentSnapshot();
      if (g.is_right) {
        for (int i=0;i<7;++i) target[7+i] = g.q_goal[i];      // 右臂 7 个
      } else {
        for (int i=0;i<7;++i) target[i]     = g.q_goal[i];    // 左臂 7 个
      }

      RCLCPP_INFO(this->get_logger(), "Moving %s arm...",
                  g.is_right ? "RIGHT":"LEFT");
      auto target_last_state = currentSnapshot();
      auto move_time = estimateMoveDuration(target_last_state,target);
      RCLCPP_INFO(this->get_logger(), "move time: %.1f s", move_time);
      moveTo(target, target_last_state, move_time + 2, true);

      // 简单的到达验证
      float max_err = 0.0F;
      auto reach_last_state = currentSnapshot();
      for (size_t i = 0; i < target.size(); ++i){
        max_err = std::max(max_err, std::abs(target[i] - reach_last_state[i]));
      }
          
      // std_msgs::msg::Bool test_flag; test_flag.data = true;//测试
      // if(g.is_right){
      //   reach_pub_right->publish(test_flag);
      // } else{
      //   reach_pub_left->publish(test_flag);
      // } 
      
      if (max_err < 0.05F) {
        std_msgs::msg::Bool flag; flag.data = true;   
        if(g.is_right){
          reach_pub_right->publish(flag);
        } else{
          reach_pub_left->publish(flag);
        } 
        RCLCPP_INFO(this->get_logger(), "%s arm reached.", g.is_right ? "RIGHT":"LEFT");
      } else {
          // 离线输出时，误差很大
          RCLCPP_INFO(this->get_logger(), "%s arm erro high : %f", g.is_right ? "RIGHT":"LEFT",max_err);
      }
    }
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
        duration = std::clamp(duration, 2.0F, 20.0F); // 限制最短2秒，最长12秒
        return duration;
    }

    ////原版
  // 插值发送
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

  //     //原版
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


// 多项式插值（五次多项式）版本，带速度前馈
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
    cmd.motor_cmd[idx].dq  = dq[i];  // 速度参考
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


};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<G1DualArmMotion>();
  node->initVisualTools();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
