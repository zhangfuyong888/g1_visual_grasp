#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "humanoid_grasp/action/visual_grasp.hpp"

class VisualGraspActionClient : public rclcpp::Node
{
public:
  using VisualGrasp = humanoid_grasp::action::VisualGrasp;
  using GoalHandleVisualGrasp = rclcpp_action::ClientGoalHandle<VisualGrasp>;

  VisualGraspActionClient()
  : Node("visual_grasp_action_client")
  {
    using namespace std::chrono_literals;

    // ✅ 使用各个 interface，而不是 shared_from_this()
    client_ = rclcpp_action::create_client<VisualGrasp>(
      this->get_node_base_interface(),
      this->get_node_graph_interface(),
      this->get_node_logging_interface(),
      this->get_node_waitables_interface(),
      "visual_grasp"   // 必须和 server 一致
    );

    // 等待服务器上线
    if (!client_->wait_for_action_server(10s)) {
      RCLCPP_ERROR(this->get_logger(), "Action server not available after waiting");
      return;
    }

    // 构造 goal
    VisualGrasp::Goal goal_msg;
    goal_msg.target_position = 5.0f;

    RCLCPP_INFO(this->get_logger(),
                "Sending goal: target_position = %.2f",
                goal_msg.target_position);

    rclcpp_action::Client<VisualGrasp>::SendGoalOptions options;
    options.goal_response_callback =
      std::bind(&VisualGraspActionClient::goal_response_callback, this, std::placeholders::_1);
    options.feedback_callback =
      std::bind(&VisualGraspActionClient::feedback_callback, this,
                std::placeholders::_1, std::placeholders::_2);
    options.result_callback =
      std::bind(&VisualGraspActionClient::result_callback, this, std::placeholders::_1);

    client_->async_send_goal(goal_msg, options);
  }


private:
  rclcpp_action::Client<VisualGrasp>::SharedPtr client_;

  void goal_response_callback(GoalHandleVisualGrasp::SharedPtr goal_handle)
  {
    if (!goal_handle) {
      RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
    } else {
      RCLCPP_INFO(this->get_logger(), "Goal accepted by server, waiting for result");
    }
  }

  void feedback_callback(
    GoalHandleVisualGrasp::SharedPtr,
    const std::shared_ptr<const VisualGrasp::Feedback> feedback)
  {
    RCLCPP_INFO(this->get_logger(),
                "Feedback: current_position = %.2f",
                feedback->current_position);
  }

  void result_callback(const GoalHandleVisualGrasp::WrappedResult & result)
  {
    switch (result.code) {
      case rclcpp_action::ResultCode::SUCCEEDED:
        RCLCPP_INFO(this->get_logger(), "Result: success = %d",
                    result.result->success);
        break;
      case rclcpp_action::ResultCode::ABORTED:
        RCLCPP_ERROR(this->get_logger(), "Result: ABORTED");
        break;
      case rclcpp_action::ResultCode::CANCELED:
        RCLCPP_ERROR(this->get_logger(), "Result: CANCELED");
        break;
      default:
        RCLCPP_ERROR(this->get_logger(), "Result: UNKNOWN");
        break;
    }
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VisualGraspActionClient>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
