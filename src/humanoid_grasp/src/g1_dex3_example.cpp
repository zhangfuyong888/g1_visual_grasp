// file: dex3_hand_node.cpp

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <array>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "unitree_hg/msg/hand_cmd.hpp"
#include "unitree_hg/msg/hand_state.hpp"
using namespace std::chrono_literals;

enum State { INIT, ROTATE, GRIP, STOP, PRINT, TORQUE, OPEN,CLOSE };

// set URDF Limits
const std::array<float, 7> max_limits_left  = { 1.05f,  1.05f,  1.75f,  0.00f, 0.00f, 0.00f, 0.00f };
const std::array<float, 7> min_limits_left  = {-1.05f, -0.724f, 0.00f, -1.57f,-1.75f,-1.57f,-1.75f};
const std::array<float, 7> max_limits_right = { 1.05f,  0.742f, 0.00f,  1.57f, 1.75f, 1.57f, 1.75f};
const std::array<float, 7> min_limits_right = {-1.05f, -1.05f,-1.75f,  0.00f, 0.00f, 0.00f, 0.00f};

#define MOTOR_MAX 7
#define SENSOR_MAX 9

struct RisMode {
  uint8_t id : 4;
  uint8_t status : 3;
  uint8_t timeout : 1;
};

class Dex3HandNode : public rclcpp::Node {  // NOLINT
 public:
  Dex3HandNode(std::string hand_side, std::string network_interface)
      : Node("dex3_hand_node"),
        hand_side_(std::move(hand_side)),
        network_interface_(std::move(network_interface)) {
    // Set up DDS topics based on hand side
    if (hand_side_ == "L") {
      dds_namespace_ = "/dex3/left/cmd";
      sub_namespace_ = "/lf/dex3/left/state";
    } else {
      dds_namespace_ = "/dex3/right/cmd";
      sub_namespace_ = "/lf/dex3/right/state";
    }

    // 可通过参数覆盖的外部控制话题名
    declare_parameter<std::string>("grip_close_topic", "/gripper/close");
    declare_parameter<std::string>("grip_open_topic",  "/gripper/open");
    close_topic_ = get_parameter("grip_close_topic").as_string();
    open_topic_  = get_parameter("grip_open_topic").as_string();

    // Initialize publishers and subscribers (底层总线)
    handcmd_publisher_ =
        this->create_publisher<unitree_hg::msg::HandCmd>(dds_namespace_, 10);
    handstate_subscriber_ =
        this->create_subscription<unitree_hg::msg::HandState>(
            sub_namespace_, rclcpp::QoS(10),
            [this](const std::shared_ptr<const unitree_hg::msg::HandState> message) {
              stateHandler(message);
            });

    // 订阅外部 open/close 控制
    close_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      close_topic_, rclcpp::QoS(10),
      [this](const std_msgs::msg::Bool::SharedPtr b){
        if (b->data) {
          current_state_ = GRIP;   // close -> StopMotors
          RCLCPP_INFO(this->get_logger(), "Received %s: CLOSE -> ", close_topic_.c_str());
        }
      });

    open_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      open_topic_, rclcpp::QoS(10),
      [this](const std_msgs::msg::Bool::SharedPtr b){
        if (b->data) {
          current_state_ = TORQUE; // open -> SendOpenLoopTorque
          RCLCPP_INFO(this->get_logger(), "Received %s: OPEN -> ", open_topic_.c_str());
        }
      });

    state_.motor_state.resize(MOTOR_MAX);
    state_.press_sensor_state.resize(SENSOR_MAX);
    msg_.motor_cmd.resize(MOTOR_MAX);

    // Print initialization message
    RCLCPP_INFO(this->get_logger(), "Dex3 %s Hand Node initialized", hand_side_.c_str());
    RCLCPP_INFO(this->get_logger(), "Subscribe close='%s', open='%s'", close_topic_.c_str(), open_topic_.c_str());
    PrintHelp();

    // Create timer for state machine
    thread_control_ = std::thread([this] { Loop(); });
    thread_input_ = std::thread([this] {
      while (rclcpp::ok()) {
        HandleUserInput();
        std::this_thread::sleep_for(100ms);
      }
    });
  }

  ~Dex3HandNode() override {
    StopMotors();
    if (thread_control_.joinable()) thread_control_.join();
    if (thread_input_.joinable()) thread_input_.join();
  }

 private:
  void Loop() {
    while (rclcpp::ok()) {
      usleep(1000); // 1kHz loop
      State state = current_state_.load();

      if (state != last_state_) {
        std::cout << "\n--- Current State: " << stateToString(state) << " ---\n";
        std::cout << "Commands:\n";
        std::cout << "  r - Rotate\n";
        std::cout << "  g - Grip (position)\n";
        std::cout << "  t - Torque (open-loop)\n";
        std::cout << "  s - Stop\n";
        std::cout << "  p - Print state\n";
        std::cout << "  q - Quit\n";
        last_state_ = state;
      }

      switch (state) {
        case INIT:
          RCLCPP_INFO_ONCE(this->get_logger(), "Initializing...");
          current_state_ = ROTATE;
          break;
        case ROTATE:
          RotateMotors();
          break;
        case GRIP:
          GripHand();      // 位置闭环 -> “夹紧”示意
          break;
        case STOP:
          StopMotors();    // 对应 /gripper/close
          break;
        case PRINT:
          PrintState();
          break;
        case TORQUE:
          SendOpenLoopTorque(); // 对应 /gripper/open
          break;
        case OPEN:
          OpenHand(); // 对应 /gripper/open
          break;
        case CLOSE:
          CloseHand(); // 对应 /gripper/open
          break;
      }
    }
  }

  void SendOpenLoopTorque() {
    for (int i = 0; i < MOTOR_MAX; ++i) {
      // 1) 平滑追踪到目标 τ
      float diff = tau_target_[i] - tau_now_[i];
      if (diff >  tau_step_)      tau_now_[i] += tau_step_;
      else if (diff < -tau_step_) tau_now_[i] -= tau_step_;
      else                        tau_now_[i]  = tau_target_[i];

      // 2) 限幅
      float tau_cmd = std::clamp(tau_now_[i], -tau_max_[i], tau_max_[i]);

      // 3) 组包并下发
      RisMode m{}; m.id=i; m.status=0x01; m.timeout=0x00;
      uint8_t mode = 0;
      mode |= (m.id & 0x0F);
      mode |= (m.status & 0x07) << 4;
      mode |= (m.timeout & 0x01) << 7;

      msg_.motor_cmd[i].mode = mode;
      msg_.motor_cmd[i].kp   = 0.0f;
      msg_.motor_cmd[i].kd   = 0.05f; // 少量阻尼
      msg_.motor_cmd[i].q    = 0.0f;
      msg_.motor_cmd[i].dq   = 0.0f;
      msg_.motor_cmd[i].tau  = tau_cmd;
    }
    handcmd_publisher_->publish(msg_);
  }

  void HandleUserInput() {
    char ch = getNonBlockingInput();
    if (ch != 0) {
      switch (ch) {
        case 'q':
          RCLCPP_INFO(this->get_logger(), "Exiting...");
          rclcpp::shutdown();
          break;
        case 'r':
          current_state_ = ROTATE; break;
        case 'g':
          current_state_ = GRIP;   break;
        case 't':
          current_state_ = TORQUE; break;
        case 's':
          current_state_ = STOP;   break;
        case 'p':
          current_state_ = PRINT;  break;
        case 'h':
          PrintHelp();             break;
      }
    }
  }

  static char getNonBlockingInput() {
    struct termios oldt {}, newt {};
    char ch = 0;
    int oldf = 0;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);  // NOLINT

    ch = getchar();  // NOLINT

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);  // NOLINT
    return ch;
  }

  void stateHandler(const std::shared_ptr<const unitree_hg::msg::HandState> message) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_ = *message;
  }

  void RotateMotors() {
    static int count = 1;
    static int dir = 1;

    const auto& max_limits = (hand_side_ == "L") ? max_limits_left : max_limits_right;
    const auto& min_limits = (hand_side_ == "L") ? min_limits_left : min_limits_right;

    for (int i = 0; i < MOTOR_MAX; i++) {
      RisMode ris_mode{};
      ris_mode.id = i;
      ris_mode.status = 0x01;
      ris_mode.timeout = 0x00;
      uint8_t mode = 0;
      mode |= (ris_mode.id & 0x0F);
      mode |= (ris_mode.status & 0x07) << 4;
      mode |= (ris_mode.timeout & 0x01) << 7;

      msg_.motor_cmd[i].mode = (mode);
      msg_.motor_cmd[i].tau  = 0.0f;
      msg_.motor_cmd[i].kp   = 0.5f;
      msg_.motor_cmd[i].kd   = 0.1f;

      float range = max_limits[i] - min_limits[i];
      float mid   = (max_limits[i] + min_limits[i]) / 2.0F;
      float amplitude = range / 2.0F;
      auto q = static_cast<float>(mid + amplitude * std::sin(static_cast<float>(count) / 20000.0F * M_PI));

      msg_.motor_cmd[i].q  = q;
      msg_.motor_cmd[i].dq = 0.0f;
    }

    handcmd_publisher_->publish(msg_);
    count += dir;

    if (count >= 10000) dir = -1;
    if (count <= -10000) dir = 1;
  }
  void CloseHand() {
    // const auto& max_limits = (hand_side_ == "L") ? max_limits_left : max_limits_right;
    // const auto& min_limits = (hand_side_ == "L") ? min_limits_left : min_limits_right;

    for (int i = 0; i < MOTOR_MAX; i++) {
      RisMode ris_mode{};
      ris_mode.id = i;
      ris_mode.status = 0x01;
      ris_mode.timeout = 0x00;
      uint8_t mode = 0;
      mode |= (ris_mode.id & 0x0F);
      mode |= (ris_mode.status & 0x07) << 4;
      mode |= (ris_mode.timeout & 0x01) << 7;

      msg_.motor_cmd[i].mode = (mode);
      msg_.motor_cmd[i].tau  = 0.0f;
      msg_.motor_cmd[i].kp   = 1.5f;
      msg_.motor_cmd[i].kd   = 0.1f;

      // float mid = (max_limits[i] + min_limits[i]) / 2.0F;
      // msg_.motor_cmd[i].q  = mid;
      msg_.motor_cmd[i].q  = close_q[i];
      msg_.motor_cmd[i].dq = 0.0f;
    }

    handcmd_publisher_->publish(msg_);
  }
  void OpenHand() {
    // const auto& max_limits = (hand_side_ == "L") ? max_limits_left : max_limits_right;
    // const auto& min_limits = (hand_side_ == "L") ? min_limits_left : min_limits_right;

    for (int i = 0; i < MOTOR_MAX; i++) {
      RisMode ris_mode{};
      ris_mode.id = i;
      ris_mode.status = 0x01;
      ris_mode.timeout = 0x00;
      uint8_t mode = 0;
      mode |= (ris_mode.id & 0x0F);
      mode |= (ris_mode.status & 0x07) << 4;
      mode |= (ris_mode.timeout & 0x01) << 7;

      msg_.motor_cmd[i].mode = (mode);
      msg_.motor_cmd[i].tau  = 0.0f;
      msg_.motor_cmd[i].kp   = 1.5f;
      msg_.motor_cmd[i].kd   = 0.1f;

      // float mid = (max_limits[i] + min_limits[i]) / 2.0F;
      msg_.motor_cmd[i].q  = open_q[i];
      msg_.motor_cmd[i].dq = 0.0f;
    }

    handcmd_publisher_->publish(msg_);
  }
  void GripHand() {
    const auto& max_limits = (hand_side_ == "L") ? max_limits_left : max_limits_right;
    const auto& min_limits = (hand_side_ == "L") ? min_limits_left : min_limits_right;

    for (int i = 0; i < MOTOR_MAX; i++) {
      RisMode ris_mode{};
      ris_mode.id = i;
      ris_mode.status = 0x01;
      ris_mode.timeout = 0x00;
      uint8_t mode = 0;
      mode |= (ris_mode.id & 0x0F);
      mode |= (ris_mode.status & 0x07) << 4;
      mode |= (ris_mode.timeout & 0x01) << 7;

      msg_.motor_cmd[i].mode = (mode);
      msg_.motor_cmd[i].tau  = 0.0f;
      msg_.motor_cmd[i].kp   = 1.5f;
      msg_.motor_cmd[i].kd   = 0.1f;

      float mid = (max_limits[i] + min_limits[i]) / 2.0F;
      msg_.motor_cmd[i].q  = mid;

      if(i == 0){
        msg_.motor_cmd[i].q  = 0.70;
      }

      msg_.motor_cmd[i].dq = 0.0f;
    }

    handcmd_publisher_->publish(msg_);
  }

  void StopMotors() {
    for (int i = 0; i < MOTOR_MAX; i++) {
      RisMode ris_mode{};
      ris_mode.id = i;
      ris_mode.status = 0x01;
      ris_mode.timeout = 0x01;

      uint8_t mode = 0;
      mode |= (ris_mode.id & 0x0F);
      mode |= (ris_mode.status & 0x07) << 4;
      mode |= (ris_mode.timeout & 0x01) << 7;

      msg_.motor_cmd[i].mode = (mode);
      msg_.motor_cmd[i].tau  = 0.0f;
      msg_.motor_cmd[i].dq   = 0.0f;
      msg_.motor_cmd[i].kp   = 0.0f;
      msg_.motor_cmd[i].kd   = 0.0f;
      msg_.motor_cmd[i].q    = 0.0f;
    }
    handcmd_publisher_->publish(msg_);
  }

  void PrintState() {
    std::lock_guard<std::mutex> lock(state_mutex_);

    Eigen::Matrix<float, 7, 1> q;
    const auto& max_limits = (hand_side_ == "L") ? max_limits_left : max_limits_right;
    const auto& min_limits = (hand_side_ == "L") ? min_limits_left : min_limits_right;

    for (int i = 0; i < 7; i++) {
      q(i) = state_.motor_state[i].q;
      q(i) = (q(i) - min_limits[i]) / (max_limits[i] - min_limits[i]);
      q(i) = std::clamp(q(i), 0.0F, 1.0F);
    }

    std::cout << "\033[2J\033[H";  // Clear screen
    std::cout << "-- " << hand_side_ << " Hand State --\n";
    std::cout << "Current State: " << stateToString(current_state_) << "\n";
    std::cout << "Position(norm): " << q.transpose() << "\n";
    if (!state_.press_sensor_state.empty())
      std::cout << "PressSensor[0][0]: "
                << state_.press_sensor_state[0].pressure[0] << std::endl;
    PrintHelp();
  }

  static void PrintHelp() {
    std::cout << "Commands:\n";
    std::cout << "  r - Rotate\n";
    std::cout << "  g - Grip (position control)\n";
    std::cout << "  t - Torque (open-loop)\n";
    std::cout << "  s - Stop\n";
    std::cout << "  p - Print state\n";
    std::cout << "  h - Help\n";
    std::cout << "  q - Quit\n";
    std::cout << "External control:\n";
    std::cout << "  ros2 topic pub /gripper/open  std_msgs/Bool \"{data: true}\" --once  # -> TORQUE (open)\n";
    std::cout << "  ros2 topic pub /gripper/close std_msgs/Bool \"{data: true}\" --once  # -> STOP   (close)\n";
  }

  static const char* stateToString(State state) {
    switch (state) {
      case INIT:   return "INIT";
      case ROTATE: return "ROTATE";
      case GRIP:   return "GRIP";
      case STOP:   return "STOP";
      case PRINT:  return "PRINT";
      case TORQUE: return "TORQUE";
      case OPEN:   return "OPEN";    // 添加缺失的
      case CLOSE:  return "CLOSE";   // 添加缺失的
      default:     return "UNKNOWN";
    }
  }

  // Member variables
  std::string hand_side_;
  std::string network_interface_;
  std::string dds_namespace_;
  std::string sub_namespace_;
  std::string close_topic_;
  std::string open_topic_;

  rclcpp::Publisher<unitree_hg::msg::HandCmd>::SharedPtr handcmd_publisher_;
  rclcpp::Subscription<unitree_hg::msg::HandState>::SharedPtr handstate_subscriber_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr close_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr open_sub_;

  unitree_hg::msg::HandCmd msg_;
  unitree_hg::msg::HandState state_;

  std::atomic<State> current_state_{INIT};
  std::atomic<State> last_state_{INIT};
  std::mutex state_mutex_;
  std::thread thread_control_;
  std::thread thread_input_;

  // 开环力矩控制配置
  std::array<float, MOTOR_MAX> tau_target_{0.0f, 0.12f, 0.0f, -0.12f, 0.0f, -0.12f, 0.0f};
  std::array<float, MOTOR_MAX> tau_now_{};
  std::array<float, MOTOR_MAX> tau_max_{0.20f, 0.18f, 0.18f, 0.18f, 0.16f, 0.10f, 0.10f};
  float tau_step_ = 0.0015f; // 每循环步最多变化 0.0015 N·m

  // 演示抓取位姿（未在本需求中使用，可保留）
  const std::array<float, 7> grip_q {0.745682f, 0.485597f, 0.293422f, 0.574162f, 0.745597f, 0.670184f, 0.552903f};

  const std::array<float, 7> open_q {0.690526, 0.640838, 0.197425, 0.753401, 0.993808,  0.80413,  0.73762};

  const std::array<float, 7> close_q {0.754167, 0.608672,  0.55256 ,0.349742, 0.641842 ,0.372445 ,0.695733};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <L/R> <network_interface>" << std::endl;
    return 1;
  }

  std::string hand_side = argv[1];
  if (hand_side != "L" && hand_side != "R") {
    std::cerr << "Invalid hand side. Please specify 'L' or 'R'." << std::endl;
    return 1;
  }

  std::string network_interface = argv[2];

  auto node = std::make_shared<Dex3HandNode>(hand_side, network_interface);
  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
