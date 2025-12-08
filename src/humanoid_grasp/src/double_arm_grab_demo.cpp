// file: duble_arm_grab_demo.cpp
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/bool.hpp>

#include "custom_msg_srv/srv/get_best_pose.hpp"
using GetBestPose = custom_msg_srv::srv::GetBestPose;

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include <chrono>
#include <cmath>
#include <optional>
#include <future>
#include <string>

// ros2 topic pub /arm/reached_target std_msgs/msg/Bool "data: true" --once

using namespace std::chrono_literals;

// ---------------- 工具函数 ----------------
static void normalize_quat(double &x, double &y, double &z, double &w) {
  const double n = std::sqrt(x*x + y*y + z*z + w*w);
  if (n > 1e-12) { x/=n; y/=n; z/=n; w/=n; } else { x=0; y=0; z=0; w=1; }
}

static geometry_msgs::msg::Pose make_pose_xyz_q(
  double x, double y, double z,
  double qx, double qy, double qz, double qw)
{
  normalize_quat(qx,qy,qz,qw);
  geometry_msgs::msg::Pose p;
  p.position.x = x;  p.position.y = y;  p.position.z = z;
  p.orientation.x = qx; p.orientation.y = qy; p.orientation.z = qz; p.orientation.w = qw;
  return p;
}

// =====================================================
//                     双臂抓取 FSM
// =====================================================
class VisualPickFSMDual : public rclcpp::Node {
public:
  VisualPickFSMDual()
  : Node("double_arm_grab_demo"),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    // ========== 通用参数 ==========
    declare_parameter<std::string>("frame_id", "base_link");
    declare_parameter<std::string>("camera_frame", "d435_link");
    declare_parameter<std::string>("service_name", "/get_best_pose");

    // 抓取姿态：是否使用目标物体朝向，否则使用 approach_* 四元数
    declare_parameter<bool>("use_object_orientation", false);

    // 三点位高度参数（相对物体的 z），pre = 物体上方，grasp = 物体处，lift = 抬起高度
    declare_parameter<double>("pre_dz",   0.15);  // 预抓取：在物体正上方
    declare_parameter<double>("lift_dz",  0.11);  // 抓取后抬起的高度

    // 夹爪动作等待（秒）
    declare_parameter<double>("grip_close_wait", 3.0);
    declare_parameter<double>("grip_open_wait",  3.0);

    // 静态TF（可选，用于 camera_link 等）
    declare_parameter<std::string>("static_parent_frame", "d435_link");
    declare_parameter<std::string>("static_child_frame",  "camera_link");
    declare_parameter<double>("static_x", 0.0);
    declare_parameter<double>("static_y", 0.0);
    declare_parameter<double>("static_z", 0.0);
    declare_parameter<double>("static_roll",  0.0);
    declare_parameter<double>("static_pitch", 0.0);
    declare_parameter<double>("static_yaw",   0.0);

    // ========== 左臂（L）参数 ==========
    // 目标话题
    declare_parameter<std::string>("left_target_pose_topic", "/left/target_pose");
    declare_parameter<std::string>("left_reach_topic",       "/arm_left/reached_target");
    declare_parameter<std::string>("left_grip_close_topic",  "/left_gripper/close");
    declare_parameter<std::string>("left_grip_open_topic",   "/left_gripper/open");

    // HOME 位姿
    declare_parameter<double>("left_home_x",  0.15);
    declare_parameter<double>("left_home_y",  0.15);
    declare_parameter<double>("left_home_z",  0.12);
    declare_parameter<double>("left_home_qx", 0.0);
    declare_parameter<double>("left_home_qy", 0.0);
    declare_parameter<double>("left_home_qz", 0.0);
    declare_parameter<double>("left_home_qw", 1.0);

    // 预抓取/抓取/抬起姿态的“姿态四元数”（如果不用目标物体朝向）
    declare_parameter<double>("left_approach_qx", 0.0);
    declare_parameter<double>("left_approach_qy", 0.0);
    declare_parameter<double>("left_approach_qz", 0.0);
    declare_parameter<double>("left_approach_qw", 1.0);

    // 左臂“放置在机器人中间”的位姿
    declare_parameter<double>("left_place_mid_x",  0.2967);
    declare_parameter<double>("left_place_mid_y",  -0.0258);
    declare_parameter<double>("left_place_mid_z",  0.1175);
    declare_parameter<double>("left_place_mid_qx", 0.0824);
    declare_parameter<double>("left_place_mid_qy", -0.0579);
    declare_parameter<double>("left_place_mid_qz", -0.2423);
    declare_parameter<double>("left_place_mid_qw", 0.9649);

    // ========== 右臂（R）参数 ==========
    declare_parameter<std::string>("right_target_pose_topic", "/right/target_pose");
    declare_parameter<std::string>("right_reach_topic",       "/arm_right/reached_target");
    declare_parameter<std::string>("right_grip_close_topic",  "/right_gripper/close");
    declare_parameter<std::string>("right_grip_open_topic",   "/right_gripper/open");

    declare_parameter<double>("right_home_x",  0.15);
    declare_parameter<double>("right_home_y",  -0.15);
    declare_parameter<double>("right_home_z",  0.12);
    declare_parameter<double>("right_home_qx", 0.0);
    declare_parameter<double>("right_home_qy",0.0);
    declare_parameter<double>("right_home_qz", 0.0);
    declare_parameter<double>("right_home_qw", 1.0);

    declare_parameter<double>("right_approach_qx", 0.0);
    declare_parameter<double>("right_approach_qy", 0.0);
    declare_parameter<double>("right_approach_qz", 0.0);
    declare_parameter<double>("right_approach_qw", 1.0);

    // 右臂“放到右侧固定位置”
    declare_parameter<double>("right_place_fix_x",  0.2921);
    declare_parameter<double>("right_place_fix_y", -0.3415);
    declare_parameter<double>("right_place_fix_z",  0.1486);
    declare_parameter<double>("right_place_fix_qx", -0.0426);
    declare_parameter<double>("right_place_fix_qy", -0.0031);
    declare_parameter<double>("right_place_fix_qz", -0.0750);
    declare_parameter<double>("right_place_fix_qw", 0.9962);

    // ========== 取参 ==========
    frame_id_     = get_parameter("frame_id").as_string();
    camera_frame_ = get_parameter("camera_frame").as_string();
    service_name_ = get_parameter("service_name").as_string();

    use_obj_ori_  = get_parameter("use_object_orientation").as_bool();
    pre_dz_       = get_parameter("pre_dz").as_double();
    lift_dz_      = get_parameter("lift_dz").as_double();

    grip_close_wait_ = get_parameter("grip_close_wait").as_double();
    grip_open_wait_  = get_parameter("grip_open_wait").as_double();

    static_parent_ = get_parameter("static_parent_frame").as_string();
    static_child_  = get_parameter("static_child_frame").as_string();
    static_x_      = get_parameter("static_x").as_double();
    static_y_      = get_parameter("static_y").as_double();
    static_z_      = get_parameter("static_z").as_double();
    static_roll_   = get_parameter("static_roll").as_double();
    static_pitch_  = get_parameter("static_pitch").as_double();
    static_yaw_    = get_parameter("static_yaw").as_double();

    left_target_topic_ = get_parameter("left_target_pose_topic").as_string();
    left_reach_topic_  = get_parameter("left_reach_topic").as_string();
    left_close_topic_  = get_parameter("left_grip_close_topic").as_string();
    left_open_topic_   = get_parameter("left_grip_open_topic").as_string();

    right_target_topic_ = get_parameter("right_target_pose_topic").as_string();
    right_reach_topic_  = get_parameter("right_reach_topic").as_string();
    right_close_topic_  = get_parameter("right_grip_close_topic").as_string();
    right_open_topic_   = get_parameter("right_grip_open_topic").as_string();

    // 左臂 home
    L_home_x_  = get_parameter("left_home_x").as_double();
    L_home_y_  = get_parameter("left_home_y").as_double();
    L_home_z_  = get_parameter("left_home_z").as_double();
    L_home_qx_ = get_parameter("left_home_qx").as_double();
    L_home_qy_ = get_parameter("left_home_qy").as_double();
    L_home_qz_ = get_parameter("left_home_qz").as_double();
    L_home_qw_ = get_parameter("left_home_qw").as_double();

    // 左臂 approach 姿态
    L_app_qx_ = get_parameter("left_approach_qx").as_double();
    L_app_qy_ = get_parameter("left_approach_qy").as_double();
    L_app_qz_ = get_parameter("left_approach_qz").as_double();
    L_app_qw_ = get_parameter("left_approach_qw").as_double();

    // 左臂中间放置
    L_place_mid_x_  = get_parameter("left_place_mid_x").as_double();
    L_place_mid_y_  = get_parameter("left_place_mid_y").as_double();
    L_place_mid_z_  = get_parameter("left_place_mid_z").as_double();
    L_place_mid_qx_ = get_parameter("left_place_mid_qx").as_double();
    L_place_mid_qy_ = get_parameter("left_place_mid_qy").as_double();
    L_place_mid_qz_ = get_parameter("left_place_mid_qz").as_double();
    L_place_mid_qw_ = get_parameter("left_place_mid_qw").as_double();

    // 右臂 home
    R_home_x_  = get_parameter("right_home_x").as_double();
    R_home_y_  = get_parameter("right_home_y").as_double();
    R_home_z_  = get_parameter("right_home_z").as_double();
    R_home_qx_ = get_parameter("right_home_qx").as_double();
    R_home_qy_ = get_parameter("right_home_qy").as_double();
    R_home_qz_ = get_parameter("right_home_qz").as_double();
    R_home_qw_ = get_parameter("right_home_qw").as_double();

    // 右臂 approach 姿态
    R_app_qx_ = get_parameter("right_approach_qx").as_double();
    R_app_qy_ = get_parameter("right_approach_qy").as_double();
    R_app_qz_ = get_parameter("right_approach_qz").as_double();
    R_app_qw_ = get_parameter("right_approach_qw").as_double();

    // 右臂固定放置
    R_place_fix_x_  = get_parameter("right_place_fix_x").as_double();
    R_place_fix_y_  = get_parameter("right_place_fix_y").as_double();
    R_place_fix_z_  = get_parameter("right_place_fix_z").as_double();
    R_place_fix_qx_ = get_parameter("right_place_fix_qx").as_double();
    R_place_fix_qy_ = get_parameter("right_place_fix_qy").as_double();
    R_place_fix_qz_ = get_parameter("right_place_fix_qz").as_double();
    R_place_fix_qw_ = get_parameter("right_place_fix_qw").as_double();

    // ---------- 静态 TF 发布 ----------
    static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    publishStaticTF_();

    // ---------- ROS 通信 ----------
    // 左臂
    L_target_pub_ = create_publisher<geometry_msgs::msg::Pose>(left_target_topic_, 10);
    L_grip_close_pub_ = create_publisher<std_msgs::msg::Bool>(left_close_topic_, 10);
    L_grip_open_pub_  = create_publisher<std_msgs::msg::Bool>(left_open_topic_, 10);
    L_reach_sub_ = create_subscription<std_msgs::msg::Bool>(
      left_reach_topic_, 10, [this](const std_msgs::msg::Bool::SharedPtr b){ left_reached_ = b->data; });

    // 右臂
    R_target_pub_ = create_publisher<geometry_msgs::msg::Pose>(right_target_topic_, 10);
    R_grip_close_pub_ = create_publisher<std_msgs::msg::Bool>(right_close_topic_, 10);
    R_grip_open_pub_  = create_publisher<std_msgs::msg::Bool>(right_open_topic_, 10);
    R_reach_sub_ = create_subscription<std_msgs::msg::Bool>(
      right_reach_topic_, 10, [this](const std_msgs::msg::Bool::SharedPtr b){ right_reached_ = b->data; });

    // 服务客户端
    client_ = create_client<GetBestPose>(service_name_);

    // 定时器（10Hz）
    timer_ = create_wall_timer(100ms, std::bind(&VisualPickFSMDual::onTimer, this));

    RCLCPP_INFO(get_logger(),
      "Dual-arm pick FSM started. base='%s', camera='%s', service='%s'",
      frame_id_.c_str(), camera_frame_.c_str(), service_name_.c_str());
  }

private:
  // ----------------- FSM 枚举 -----------------
  enum class State {
    // 初始化：双臂各回 HOME
    INIT_PUBLISH_HOME_BOTH,
    INIT_WAIT_HOME_BOTH,

    // 先左臂流程
    WAIT_TF,
    WAIT_SERVICE,
    QUERY_OBJECT_LEFT,
    L_PUBLISH_PRE,
    L_WAIT_PRE,
    L_PUBLISH_GRASP,
    L_WAIT_GRASP,
    L_CLOSE_GRIPPER,
    L_WAIT_GRIPPER_CLOSE,
    L_PUBLISH_LIFT,
    L_WAIT_LIFT,
    L_PUBLISH_PLACE_MID,
    L_WAIT_PLACE_MID,

    // 新增：左臂放置完成后先回 HOME，再让右臂进场
    L_PUBLISH_HOME_AFTER_PLACE,
    L_WAIT_HOME_AFTER_PLACE,

    // 再右臂流程
    QUERY_OBJECT_RIGHT,
    R_PUBLISH_PRE,
    R_WAIT_PRE,
    R_PUBLISH_GRASP,
    R_WAIT_GRASP,
    R_CLOSE_GRIPPER,
    R_WAIT_GRIPPER_CLOSE,
    R_PUBLISH_LIFT,
    R_WAIT_LIFT,
    R_PUBLISH_PLACE_FIX,
    R_WAIT_PLACE_FIX,

    // 本轮结束：回 HOME
    PUBLISH_HOME_BOTH,
    WAIT_HOME_BOTH
  } state_ { State::INIT_PUBLISH_HOME_BOTH };

  // ----------------- 工具方法 -----------------
  bool check_tf_ready(double timeout_sec = 0.5) {
    return tf_buffer_.canTransform(frame_id_, camera_frame_, tf2::TimePointZero,
                                   tf2::durationFromSec(timeout_sec));
  }

  std::optional<geometry_msgs::msg::PoseStamped>
  to_frame_id(const geometry_msgs::msg::PoseStamped &in) {
    geometry_msgs::msg::PoseStamped tmp = in;
    tmp.header.stamp.sec = 0; tmp.header.stamp.nanosec = 0;
    if (tmp.header.frame_id.empty()) tmp.header.frame_id = frame_id_;
    if (tmp.header.frame_id == frame_id_) return tmp;

    try {
      auto out = tf_buffer_.transform(tmp, frame_id_, tf2::durationFromSec(0.3));
      return out;
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(get_logger(), "TF transform %s -> %s failed: %s",
                  tmp.header.frame_id.c_str(), frame_id_.c_str(), ex.what());
      return std::nullopt;
    }
  }

  void publishStaticTF_() {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = now();
    t.header.frame_id = static_parent_;
    t.child_frame_id  = static_child_;
    t.transform.translation.x = static_x_;
    t.transform.translation.y = static_y_;
    t.transform.translation.z = static_z_;
    tf2::Quaternion q; q.setRPY(static_roll_, static_pitch_, static_yaw_); q.normalize();
    t.transform.rotation.x = q.x();
    t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z();
    t.transform.rotation.w = q.w();
    static_broadcaster_->sendTransform(t);
    RCLCPP_INFO(get_logger(),
      "Published STATIC TF: %s -> %s | xyz=[%.3f %.3f %.3f], rpy=[%.3f %.3f %.3f]",
      static_parent_.c_str(), static_child_.c_str(),
      static_x_, static_y_, static_z_, static_roll_, static_pitch_, static_yaw_);
  }

  // 给定物体位姿与（是否使用物体姿态），生成 “预抓取/抓取/抬起” 三个位姿
  void make_triplet_poses(const geometry_msgs::msg::PoseStamped &obj,
                          bool use_obj_orientation,
                          double app_qx, double app_qy, double app_qz, double app_qw,
                          double pre_dz, double lift_dz,
                          geometry_msgs::msg::Pose &pre_pose,
                          geometry_msgs::msg::Pose &grasp_pose,
                          geometry_msgs::msg::Pose &lift_pose)
  {
    double qx, qy, qz, qw;
    if (use_obj_orientation) {
      qx = obj.pose.orientation.x; qy = obj.pose.orientation.y;
      qz = obj.pose.orientation.z; qw = obj.pose.orientation.w;
    } else {
      qx = app_qx; qy = app_qy; qz = app_qz; qw = app_qw;
    }
    normalize_quat(qx,qy,qz,qw);
    const auto &p = obj.pose.position;

    pre_pose   = make_pose_xyz_q(p.x - 0.05, p.y, p.z + pre_dz, qx, qy, qz, qw);
    grasp_pose = make_pose_xyz_q(p.x - 0.05, p.y, p.z,            qx, qy, qz, qw);
    lift_pose  = make_pose_xyz_q(p.x - 0.05, p.y, p.z + lift_dz,  qx, qy, qz, qw);
  }

  // 发送到某一只手
  enum class Arm { LEFT, RIGHT };

  void send_target(Arm arm, const geometry_msgs::msg::Pose &pose) {
    if (arm == Arm::LEFT) {
      left_reached_ = false;
      L_target_pub_->publish(pose);
    } else {
      right_reached_ = false;
      R_target_pub_->publish(pose);
    }
  }

  bool reached(Arm arm) const {
    return (arm == Arm::LEFT) ? left_reached_ : right_reached_;
  }

  void gripper_cmd(Arm arm, bool close) {
    std_msgs::msg::Bool msg; msg.data = true;
    if (arm == Arm::LEFT) {
      (close ? L_grip_close_pub_ : L_grip_open_pub_)->publish(msg);
    } else {
      (close ? R_grip_close_pub_ : R_grip_open_pub_)->publish(msg);
    }
  }

  // ----------------- 主定时器 -----------------
  void onTimer() {
    switch (state_) {
      // ======= 双臂同时回 HOME =======
      case State::INIT_PUBLISH_HOME_BOTH: {
        auto L_home = make_pose_xyz_q(L_home_x_, L_home_y_, L_home_z_, L_home_qx_, L_home_qy_, L_home_qz_, L_home_qw_);
        auto R_home = make_pose_xyz_q(R_home_x_, R_home_y_, R_home_z_, R_home_qx_, R_home_qy_, R_home_qz_, R_home_qw_);
        send_target(Arm::LEFT,  L_home);
        send_target(Arm::RIGHT, R_home);
        RCLCPP_INFO(get_logger(), "[INIT] Published HOME for both arms.");
        state_ = State::INIT_WAIT_HOME_BOTH;
        break;
      }
      case State::INIT_WAIT_HOME_BOTH: {
        if (!reached(Arm::LEFT) || !reached(Arm::RIGHT)) return;
        RCLCPP_INFO(get_logger(), "[INIT] Both arms at HOME.");
        state_ = State::WAIT_TF;
        break;
      }

      // ======= 检查 TF 与 服务 =======
      case State::WAIT_TF: {
        if (!check_tf_ready(0.5)) {
          if (!warned_tf_) {
            RCLCPP_WARN(get_logger(),
              "Waiting TF: '%s' -> '%s' not available yet. (Ensure robot_state_publisher/static TF running)",
              camera_frame_.c_str(), frame_id_.c_str());
            warned_tf_ = true;
          }
          return;
        }
        RCLCPP_INFO(get_logger(), "TF ready: %s <-> %s", frame_id_.c_str(), camera_frame_.c_str());
        state_ = State::WAIT_SERVICE;
        break;
      }
      case State::WAIT_SERVICE: {
        if (!client_->wait_for_service(1s)) {
          if (!warned_srv_) {
            RCLCPP_WARN(get_logger(), "Waiting for service '%s'...", service_name_.c_str());
            warned_srv_ = true;
          }
          return;
        }
        RCLCPP_INFO(get_logger(), "Service available.");
        state_ = State::QUERY_OBJECT_LEFT;
        break;
      }

      // ======= 左臂：请求物体位姿 =======
      case State::QUERY_OBJECT_LEFT: {
        if (!pending_future_valid_) {
          auto req = std::make_shared<GetBestPose::Request>();
          req->is_left_hand = true;  // 左臂：true
          auto future_req = client_->async_send_request(req);
          pending_future_ = future_req.future.share();
          pending_future_valid_ = true;
          RCLCPP_INFO(get_logger(), "[LEFT] Sending GetBestPose request...");
          return;
        }
        if (pending_future_.wait_for(0s) != std::future_status::ready) return;

        {
          auto resp = pending_future_.get();
          pending_future_valid_ = false;
          if (!resp->success) {
            RCLCPP_WARN(get_logger(), "[LEFT] Service failure: %s", resp->message.c_str());
            return;
          }
          auto maybe_pose = to_frame_id(resp->pose);
          if (!maybe_pose) {
            RCLCPP_WARN(get_logger(), "[LEFT] Pose cannot transform to '%s'.", frame_id_.c_str());
            return;
          }
          L_obj_pose_ = *maybe_pose;
          RCLCPP_INFO(get_logger(), "[LEFT] Target @ [%.3f, %.3f, %.3f]",
                      L_obj_pose_.pose.position.x, L_obj_pose_.pose.position.y, L_obj_pose_.pose.position.z);
        }

        // 生成左臂三点
        make_triplet_poses(L_obj_pose_, use_obj_ori_, L_app_qx_, L_app_qy_, L_app_qz_, L_app_qw_,
                           pre_dz_, lift_dz_, L_pre_, L_grasp_, L_lift_);
        state_ = State::L_PUBLISH_PRE;
        break;
      }

      // ======= 左臂：预抓取 → 抓取 → 抬起 → 放中间 =======
      case State::L_PUBLISH_PRE: {
        send_target(Arm::LEFT, L_pre_);
        RCLCPP_INFO(get_logger(), "[LEFT] Published PRE.");
        state_ = State::L_WAIT_PRE;
        break;
      }
      case State::L_WAIT_PRE: {
        if (!reached(Arm::LEFT)) return;
        RCLCPP_INFO(get_logger(), "[LEFT] PRE reached.");
        state_ = State::L_PUBLISH_GRASP;
        break;
      }
      case State::L_PUBLISH_GRASP: {
        send_target(Arm::LEFT, L_grasp_);
        RCLCPP_INFO(get_logger(), "[LEFT] Published GRASP.");
        state_ = State::L_WAIT_GRASP;
        break;
      }
      case State::L_WAIT_GRASP: {
        if (!reached(Arm::LEFT)) return;
        RCLCPP_INFO(get_logger(), "[LEFT] GRASP reached.");
        state_ = State::L_CLOSE_GRIPPER;
        break;
      }
      case State::L_CLOSE_GRIPPER: {
        gripper_cmd(Arm::LEFT, true);
        RCLCPP_INFO(get_logger(), "[LEFT] Gripper CLOSE. Wait %.2fs", grip_close_wait_);
        wait_until_ = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(grip_close_wait_));
        state_ = State::L_WAIT_GRIPPER_CLOSE;
        break;
      }
      case State::L_WAIT_GRIPPER_CLOSE: {
        if (std::chrono::steady_clock::now() < wait_until_) return;
        RCLCPP_INFO(get_logger(), "[LEFT] Gripper close done.");
        state_ = State::L_PUBLISH_LIFT;
        break;
      }
      case State::L_PUBLISH_LIFT: {
        send_target(Arm::LEFT, L_lift_);
        RCLCPP_INFO(get_logger(), "[LEFT] Published LIFT.");
        state_ = State::L_WAIT_LIFT;
        break;
      }
      case State::L_WAIT_LIFT: {
        if (!reached(Arm::LEFT)) return;
        RCLCPP_INFO(get_logger(), "[LEFT] LIFT reached.");
        state_ = State::L_PUBLISH_PLACE_MID;
        break;
      }
      case State::L_PUBLISH_PLACE_MID: {
        auto place_mid = make_pose_xyz_q(L_place_mid_x_, L_place_mid_y_, L_place_mid_z_,
                                         L_place_mid_qx_, L_place_mid_qy_, L_place_mid_qz_, L_place_mid_qw_);
        send_target(Arm::LEFT, place_mid);
        RCLCPP_INFO(get_logger(), "[LEFT] Published PLACE (middle).");
        state_ = State::L_WAIT_PLACE_MID;
        break;
      }
      case State::L_WAIT_PLACE_MID: {
        if (!reached(Arm::LEFT)) return;
        RCLCPP_INFO(get_logger(), "[LEFT] PLACE middle reached.");
        // 张开左夹爪（释放物体）
        gripper_cmd(Arm::LEFT, false);
        RCLCPP_INFO(get_logger(), "[LEFT] Gripper OPEN. Wait %.2fs", grip_open_wait_);
        wait_until_ = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(grip_open_wait_));
        // 改进点：不再直接右臂查询；先让左臂回 HOME
        state_ = State::L_PUBLISH_HOME_AFTER_PLACE;
        break;
      }

      // ======= 新增：左臂放置完成后回 HOME，再让右臂抓取 =======
      case State::L_PUBLISH_HOME_AFTER_PLACE: {
        // 确保左爪完全打开后再移动
        if (std::chrono::steady_clock::now() < wait_until_) return;
        auto L_home = make_pose_xyz_q(L_home_x_, L_home_y_, L_home_z_,
                                      L_home_qx_, L_home_qy_, L_home_qz_, L_home_qw_);
        send_target(Arm::LEFT, L_home);
        RCLCPP_INFO(get_logger(), "[LEFT] After placing, go HOME.");
        state_ = State::L_WAIT_HOME_AFTER_PLACE;
        break;
      }
      case State::L_WAIT_HOME_AFTER_PLACE: {
        if (!reached(Arm::LEFT)) return;
        RCLCPP_INFO(get_logger(), "[LEFT] HOME reached after placing. Safe for RIGHT to approach.");
        state_ = State::QUERY_OBJECT_RIGHT;
        break;
      }

      // ======= 右臂：请求物体位姿 =======
      case State::QUERY_OBJECT_RIGHT: {
        if (!pending_future_valid_) {
          auto req = std::make_shared<GetBestPose::Request>();
          req->is_left_hand = false;  // 右臂：false
          auto future_req = client_->async_send_request(req);
          pending_future_ = future_req.future.share();
          pending_future_valid_ = true;
          RCLCPP_INFO(get_logger(), "[RIGHT] Sending GetBestPose request...");
          return;
        }
        if (pending_future_.wait_for(0s) != std::future_status::ready) return;

        {
          auto resp = pending_future_.get();
          pending_future_valid_ = false;
          if (!resp->success) {
            RCLCPP_WARN(get_logger(), "[RIGHT] Service failure: %s", resp->message.c_str());
            return;
          }
          auto maybe_pose = to_frame_id(resp->pose);
          if (!maybe_pose) {
            RCLCPP_WARN(get_logger(), "[RIGHT] Pose cannot transform to '%s'.", frame_id_.c_str());
            return;
          }
          R_obj_pose_ = *maybe_pose;
          RCLCPP_INFO(get_logger(), "[RIGHT] Target @ [%.3f, %.3f, %.3f]",
                      R_obj_pose_.pose.position.x, R_obj_pose_.pose.position.y, R_obj_pose_.pose.position.z);
        }

        // 生成右臂三点
        make_triplet_poses(R_obj_pose_, use_obj_ori_, R_app_qx_, R_app_qy_, R_app_qz_, R_app_qw_,
                           pre_dz_, lift_dz_, R_pre_, R_grasp_, R_lift_);
        state_ = State::R_PUBLISH_PRE;
        break;
      }

      // ======= 右臂：预抓取 → 抓取 → 抬起 → 放固定位 =======
      case State::R_PUBLISH_PRE: {
        send_target(Arm::RIGHT, R_pre_);
        RCLCPP_INFO(get_logger(), "[RIGHT] Published PRE.");
        state_ = State::R_WAIT_PRE;
        break;
      }
      case State::R_WAIT_PRE: {
        if (!reached(Arm::RIGHT)) return;
        RCLCPP_INFO(get_logger(), "[RIGHT] PRE reached.");
        state_ = State::R_PUBLISH_GRASP;
        break;
      }
      case State::R_PUBLISH_GRASP: {
        send_target(Arm::RIGHT, R_grasp_);
        RCLCPP_INFO(get_logger(), "[RIGHT] Published GRASP.");
        state_ = State::R_WAIT_GRASP;
        break;
      }
      case State::R_WAIT_GRASP: {
        if (!reached(Arm::RIGHT)) return;
        RCLCPP_INFO(get_logger(), "[RIGHT] GRASP reached.");
        state_ = State::R_CLOSE_GRIPPER;
        break;
      }
      case State::R_CLOSE_GRIPPER: {
        gripper_cmd(Arm::RIGHT, true);
        RCLCPP_INFO(get_logger(), "[RIGHT] Gripper CLOSE. Wait %.2fs", grip_close_wait_);
        wait_until_ = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(grip_close_wait_));
        state_ = State::R_WAIT_GRIPPER_CLOSE;
        break;
      }
      case State::R_WAIT_GRIPPER_CLOSE: {
        if (std::chrono::steady_clock::now() < wait_until_) return;
        RCLCPP_INFO(get_logger(), "[RIGHT] Gripper close done.");
        state_ = State::R_PUBLISH_LIFT;
        break;
      }
      case State::R_PUBLISH_LIFT: {
        send_target(Arm::RIGHT, R_lift_);
        RCLCPP_INFO(get_logger(), "[RIGHT] Published LIFT.");
        state_ = State::R_WAIT_LIFT;
        break;
      }
      case State::R_WAIT_LIFT: {
        if (!reached(Arm::RIGHT)) return;
        RCLCPP_INFO(get_logger(), "[RIGHT] LIFT reached.");
        state_ = State::R_PUBLISH_PLACE_FIX;
        break;
      }
      case State::R_PUBLISH_PLACE_FIX: {
        auto place_fix = make_pose_xyz_q(R_place_fix_x_, R_place_fix_y_, R_place_fix_z_,
                                         R_place_fix_qx_, R_place_fix_qy_, R_place_fix_qz_, R_place_fix_qw_);
        send_target(Arm::RIGHT, place_fix);
        RCLCPP_INFO(get_logger(), "[RIGHT] Published PLACE (fixed).");
        state_ = State::R_WAIT_PLACE_FIX;
        break;
      }
      case State::R_WAIT_PLACE_FIX: {
        if (!reached(Arm::RIGHT)) return;
        RCLCPP_INFO(get_logger(), "[RIGHT] PLACE fixed reached.");
        // 右爪张开（如果需要放置后释放）
        gripper_cmd(Arm::RIGHT, false);
        RCLCPP_INFO(get_logger(), "[RIGHT] Gripper OPEN. Wait %.2fs", grip_open_wait_);
        wait_until_ = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(grip_open_wait_));
        state_ = State::PUBLISH_HOME_BOTH;
        break;
      }

      // ======= 收尾：双臂回 HOME，进入下一轮 =======
      case State::PUBLISH_HOME_BOTH: {
        if (std::chrono::steady_clock::now() < wait_until_) return;
        auto L_home = make_pose_xyz_q(L_home_x_, L_home_y_, L_home_z_, L_home_qx_, L_home_qy_, L_home_qz_, L_home_qw_);
        auto R_home = make_pose_xyz_q(R_home_x_, R_home_y_, R_home_z_, R_home_qx_, R_home_qy_, R_home_qz_, R_home_qw_);
        send_target(Arm::LEFT,  L_home);
        send_target(Arm::RIGHT, R_home);
        RCLCPP_INFO(get_logger(), "Published HOME for both arms (end of cycle).");
        state_ = State::WAIT_HOME_BOTH;
        break;
      }
      case State::WAIT_HOME_BOTH: {
        if (!reached(Arm::LEFT) || !reached(Arm::RIGHT)) return;
        RCLCPP_INFO(get_logger(), "Both arms HOME. Cycle done -> loop.");
        state_ = State::QUERY_OBJECT_LEFT;  // 下一轮从左臂开始
        break;
      }
    }
  }

private:
  // ========== 通用配置 ==========
  std::string frame_id_, camera_frame_, service_name_;
  bool   use_obj_ori_{false};
  double pre_dz_{0.15};
  double lift_dz_{0.15};

  // 夹爪等待
  double grip_close_wait_{5.0}, grip_open_wait_{5.0};
  std::chrono::steady_clock::time_point wait_until_;

  // 静态 TF
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
  std::string static_parent_, static_child_;
  double static_x_{0.0}, static_y_{0.0}, static_z_{0.0};
  double static_roll_{0.0}, static_pitch_{0.0}, static_yaw_{0.0};

  // ========== 左臂配置 ==========
  std::string left_target_topic_, left_reach_topic_, left_close_topic_, left_open_topic_;

  double L_home_x_{}, L_home_y_{}, L_home_z_{};
  double L_home_qx_{}, L_home_qy_{}, L_home_qz_{}, L_home_qw_{};
  double L_app_qx_{}, L_app_qy_{}, L_app_qz_{}, L_app_qw_{};

  double L_place_mid_x_{}, L_place_mid_y_{}, L_place_mid_z_{};
  double L_place_mid_qx_{}, L_place_mid_qy_{}, L_place_mid_qz_{}, L_place_mid_qw_{};

  // ========== 右臂配置 ==========
  std::string right_target_topic_, right_reach_topic_, right_close_topic_, right_open_topic_;

  double R_home_x_{}, R_home_y_{}, R_home_z_{};
  double R_home_qx_{}, R_home_qy_{}, R_home_qz_{}, R_home_qw_{};
  double R_app_qx_{}, R_app_qy_{}, R_app_qz_{}, R_app_qw_{};

  double R_place_fix_x_{}, R_place_fix_y_{}, R_place_fix_z_{};
  double R_place_fix_qx_{}, R_place_fix_qy_{}, R_place_fix_qz_{}, R_place_fix_qw_{};

  // ========== 运行时状态 ==========
  geometry_msgs::msg::PoseStamped L_obj_pose_, R_obj_pose_;
  geometry_msgs::msg::Pose L_pre_, L_grasp_, L_lift_;
  geometry_msgs::msg::Pose R_pre_, R_grasp_, R_lift_;

  // ROS 通信对象
  rclcpp::Client<GetBestPose>::SharedPtr client_;

  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr L_target_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr L_grip_close_pub_, L_grip_open_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr L_reach_sub_;

  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr R_target_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr R_grip_close_pub_, R_grip_open_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr R_reach_sub_;

  rclcpp::TimerBase::SharedPtr timer_;

  // TF
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Service future
  rclcpp::Client<GetBestPose>::SharedFuture pending_future_;
  bool pending_future_valid_{false};

  // 到位标志
  bool left_reached_{false};
  bool right_reached_{false};
  bool warned_tf_{false};
  bool warned_srv_{false};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VisualPickFSMDual>());
  rclcpp::shutdown();
  return 0;
}
