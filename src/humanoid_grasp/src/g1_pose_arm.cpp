#include <algorithm>
#include <array>
#include <chrono>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <unitree_hg/msg/low_cmd.hpp>
#include <unitree_hg/msg/low_state.hpp>

#include "std_msgs/msg/float32_multi_array.hpp"
#include <std_msgs/msg/bool.hpp>

#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <deque>

#include <humanoid_grasp/g1.hpp>

// G1ARM5 or G1ARM7
#define G1ARM7 7
#define ARM_TYPE G1ARM7

using namespace std::chrono_literals;
using LowCmd = unitree_hg::msg::LowCmd;
using LowState = unitree_hg::msg::LowState;

class ArmLowLevelController : public rclcpp::Node {

  static constexpr int NUM_ARM_JOINTS = 17;
  static constexpr auto NOT_USED_JOINT = G1Arm7JointIndex::NOT_USED_JOINT;
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
  //站立初始状态
  std::array<float, NUM_ARM_JOINTS> target_pos_ = {
      0.2906F, 0.1478F, -0.0157F, 0.9779F, 0.1855F, -0.0855F, 0.0304F,   // left 7
      0.2836F,-0.2251F,  0.0182F, 0.9851F,-0.2342F, 0.0593F,-0.0431F,   // right 7
      0.0F, 0.0F, 0.0F                                                   // waist 3
      };
  // std::array<float, NUM_ARM_JOINTS> target_pos_ = {
  //     0.0F, PI_2,  0.0F, PI_2, 0.0F, 0.0F, 0.0F,  // left
  //     0.0F, -PI_2, 0.0F, PI_2, 0.0F, 0.0F, 0.0F,  // right
  //     0.0F, 0.0F,  0.0F};
    // std::array<float, NUM_ARM_JOINTS> target_pos_ = {
    //   0.2906F, 0.1278F, -0.0157F, 0.9779F, 0.1855F, -0.0855F, 0.0304F,  // left
    //   0.2936F, -0.1251F, 0.0182F, 0.9851F, -0.2342F, 0.0593F, -0.0431F,  // right
    //   0.0F, 0.0F, 0.0F};

 public:
  ArmLowLevelController() : Node("g1_arm_pose") {
    // ROS2接口初始化
    pub_ = this->create_publisher<LowCmd>("/arm_sdk", 50);//针对抖动，从10调整到50
    sub_ = this->create_subscription<LowState>(
        "/lowstate", 5,
        [this](const LowState::SharedPtr msg) { StateCallback(msg); });

    reach_pub = this->create_publisher<std_msgs::msg::Bool>("/arm/reached_target", 1);


    
    // 订阅目标角度
    sub_target_joint = this->create_subscription<std_msgs::msg::Float32MultiArray>(
    "/target_joint_topic", 10,
    [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        TargetCallback(msg);
    });

    pub_arm_joint_state = this->create_publisher<std_msgs::msg::Float32MultiArray>("/arm_joint_state", 10);

    sleep_time_ =
        std::chrono::milliseconds(static_cast<int>(control_dt_ * 1000));

    // 位置初始化
    init_pos_.fill(0.0F);

    thread_ = std::thread([this]() { ControlLoop(); });
  }

 private:
 
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr reach_pub;
  rclcpp::Publisher<LowCmd>::SharedPtr pub_;
  rclcpp::Subscription<LowState>::SharedPtr sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_target_joint;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_arm_joint_state;
  std::thread thread_;

  // 订阅规划后的整条轨迹
  rclcpp::Subscription<moveit_msgs::msg::RobotTrajectory>::SharedPtr traj_sub_;

  // 缓存轨迹点（只用 positions）
  std::deque<trajectory_msgs::msg::JointTrajectoryPoint> traj_queue_;
  std::vector<std::string> traj_joint_names_;
  std::mutex traj_mutex_;
  bool new_trajectory_received_ = false;  // 收到新轨迹标志

  // 目标角度
  std_msgs::msg::Float32MultiArray target_joint;
  std::mutex targetJ_mutex_;

  std::array<float, NUM_ARM_JOINTS> target_J{};

  bool new_target_angle_received = false; 

  std_msgs::msg::Bool reach_flag ;


  LowState last_state_;
  std::mutex state_mutex_;
  bool state_received_ = false;

  bool init_done = false;

  float control_dt_{0.02F};
  float kp_{60.0F}, kd_{1.5F};
  float max_joint_velocity_{0.4F};
  std::chrono::milliseconds sleep_time_{};

  std::array<float, NUM_ARM_JOINTS> init_pos_{};

  std::array<float, NUM_ARM_JOINTS> current_jpos_{};

    //订阅目标角度回调函数
  void TargetCallback(const std_msgs::msg::Float32MultiArray::SharedPtr Target_joint) {
    if (init_done == false) //判断初始化完成之后，才接收目标角度
      return;

    std::lock_guard<std::mutex> lock(targetJ_mutex_);
    target_joint = *Target_joint;
    RCLCPP_INFO(this->get_logger(), "Received target joint angles:");
    target_J = current_jpos_;

    //左臂控制
    if (target_joint.data[0]==0)
    {
      RCLCPP_INFO(this->get_logger(), "控制臂为左臂");
      for (size_t i = 1; i < target_joint.data.size(); ++i) {
        target_J[i-1] = target_joint.data[i];
        RCLCPP_INFO(this->get_logger(), "第 %zu 关节: %f", i, target_J[i-1]);
      }
    } else { //右臂控制
      RCLCPP_INFO(this->get_logger(), "控制臂为右臂");
      for (size_t i = 1; i < target_joint.data.size(); ++i) {
        target_J[7 + i-1] = target_joint.data[i];
        RCLCPP_INFO(this->get_logger(), "第 %zu 关节: %f", i, target_J[7 + i-1]);
      }
    }
    // 设置新目标角度已接收的标志位
    new_target_angle_received = true;
}
//命令行发布 ros2 topic pub /target_joint_topic std_msgs/msg/Float32MultiArray "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"

  void StateCallback(const LowState::SharedPtr msg) {//接收机器人实际状态，并发布到moveit中同步
    std::lock_guard<std::mutex> lock(state_mutex_);
    last_state_ = *msg;

    for (size_t i = 0; i < arm_joints_.size(); ++i) {
      current_jpos_[i] = last_state_.motor_state[static_cast<int>(arm_joints_[i])].q;
      // RCLCPP_INFO(this->get_logger(), "第 %zu 关节: %f", i, current_jpos_[i]);
    }
    state_received_ = true;

    //发布当前关节角度给moveit
    std_msgs::msg::Float32MultiArray current_joint_state;
    current_joint_state.data.clear();
    for (size_t i = 0; i < current_jpos_.size(); ++i)
    {
      current_joint_state.data.push_back(current_jpos_[i]);
    }
    pub_arm_joint_state->publish(current_joint_state);

  }

  void ControlLoop() {
    while (!state_received_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                           "Waiting for LowState...");
      // state_received_ = false;
      std::this_thread::sleep_for(100ms);
    }
    RCLCPP_INFO(this->get_logger(), "LowState received. Starting control...");
    RCLCPP_INFO(this->get_logger(), "Moving to initial position...");
    moveTo(target_pos_, current_jpos_, 10.0F, true);
    init_done = true;//初始化完成
    RCLCPP_INFO(this->get_logger(), "Waiting for target joint angles...");
    StartControlSequence();
  }



void StartControlSequence() {

  while (rclcpp::ok()){
    if (new_target_angle_received)
    {
      RCLCPP_INFO(this->get_logger(), "正在移动到目标位置...");
      moveTo(target_J, current_jpos_, 10.0F, true);
      // final stage: stop control
      // StopControl();
      new_target_angle_received = false;
      RCLCPP_INFO(this->get_logger(), "已到达目标位置。");
      //到达目标点
      reach_flag.data = true;
      reach_pub->publish(reach_flag);
    }

    std::this_thread::sleep_for(100ms);
  }

}

  // // move to target position from current position
  // void MoveTo(const std::array<float, NUM_ARM_JOINTS>& target,
  //             std::array<float, NUM_ARM_JOINTS>& current, float duration,
  //             bool smooth) {
  //   const int steps = static_cast<int>(duration / control_dt_);
  //   const float max_delta = max_joint_velocity_ * control_dt_;

  //   for (int i = 0; i < steps; ++i) {
  //     float phase = static_cast<float>(i) / static_cast<float>(steps);

  //     for (size_t j = 0; j < arm_joints_.size(); ++j) {
  //       if (smooth) {
  //         // smooth mode: linear interpolation
  //         current[j] = current[j] * (1 - phase) + target[j] * phase;
  //       } else {
  //         // non-smooth mode: move with max velocity
  //         float diff = target[j] - current[j];
  //         current[j] += std::clamp(diff, -max_delta, max_delta);
  //       }
  //     }

  //     SendPositionCommand(current);
  //     std::this_thread::sleep_for(sleep_time_);
  //   }
  // }

  // void SendPositionCommand(const std::array<float, NUM_ARM_JOINTS>& positions) {
  //   LowCmd cmd;

  //   for (size_t i = 0; i < arm_joints_.size(); ++i) {
  //     int idx = static_cast<int>(arm_joints_[i]);
  //     cmd.motor_cmd[idx].q = positions[i];
  //     cmd.motor_cmd[idx].dq = 0.0F;
  //     cmd.motor_cmd[idx].tau = 0.0F;
  //     if (i >= arm_joints_.size() - 3) {
  //       cmd.motor_cmd[idx].kp = kp_ * 7.0F;
  //       cmd.motor_cmd[idx].kd = kd_ * 4.0F;
  //     } else {
  //       cmd.motor_cmd[idx].kp = kp_ * 4.0F;
  //       cmd.motor_cmd[idx].kd = kd_ * 3.0F;
  //     }
  //     // if (i >= arm_joints_.size() - 3) {
  //     //   cmd.motor_cmd[idx].kp = kp_ * 4.0F;
  //     //   cmd.motor_cmd[idx].kd = kd_ * 4.0F;
  //     // } else {
  //     //   cmd.motor_cmd[idx].kp = kp_;
  //     //   cmd.motor_cmd[idx].kd = kd_;
  //     // }
  //   }

  //   cmd.motor_cmd[static_cast<int>(NOT_USED_JOINT)].q = 1.0F;

  //   pub_->publish(cmd);
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
      // std::lock_guard<std::mutex> lk(state_mtx_);

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
    // std::lock_guard<std::mutex> lk(state_mtx_);
    std::array<float, NUM_ARM_JOINTS> dq_zero{};
    sendLowCmd(target, dq_zero);
  }
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

  pub_->publish(cmd);
}
  void StopControl() {
    RCLCPP_INFO(this->get_logger(), "Stopping control...");

    const int steps = static_cast<int>(2.0F / control_dt_);
    const float delta_w = 0.2F * control_dt_;
    float weight = 1.0F;

    for (int i = 0; i < steps; ++i) {
      weight -= delta_w;
      weight = std::clamp(weight, 0.0F, 1.0F);

      LowCmd cmd;

      for (size_t j = 0; j < arm_joints_.size(); ++j) {
        int idx = static_cast<int>(arm_joints_[j]);
        cmd.motor_cmd[idx].q = current_jpos_[j];
        cmd.motor_cmd[idx].dq = 0.0F;
        cmd.motor_cmd[idx].kp = kp_;
        cmd.motor_cmd[idx].kd = kd_;
        cmd.motor_cmd[idx].tau = 0.0F;
      }
      cmd.motor_cmd[static_cast<int>(NOT_USED_JOINT)].q = weight;
      pub_->publish(cmd);

      rclcpp::sleep_for(sleep_time_);
    }

    LowCmd cmd;
    cmd.motor_cmd[static_cast<int>(NOT_USED_JOINT)].q = 0.0F;
    pub_->publish(cmd);
    RCLCPP_INFO(this->get_logger(), "Control stopped.");
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArmLowLevelController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}



