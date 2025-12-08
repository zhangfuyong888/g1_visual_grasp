#include <chrono>
#include <memory>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "humanoid_grasp/action/visual_grasp.hpp"

class VisualGraspActionServer : public rclcpp::Node
{
public:
  using VisualGrasp = humanoid_grasp::action::VisualGrasp;
  using GoalHandleVisualGrasp = rclcpp_action::ServerGoalHandle<VisualGrasp>;

  VisualGraspActionServer()
  : Node("visual_grasp_action_server")
  {}

  void init()
  {
    using namespace std::placeholders;

    server_ = rclcpp_action::create_server<VisualGrasp>(
      shared_from_this(),            // 现在可以安全用 shared_from_this 了
      "visual_grasp",                // action 名字
      std::bind(&VisualGraspActionServer::handle_goal, this, _1, _2),
      std::bind(&VisualGraspActionServer::handle_cancel, this, _1),
      std::bind(&VisualGraspActionServer::handle_accepted, this, _1)
    );
  }

private:
  rclcpp_action::Server<VisualGrasp>::SharedPtr server_;

  // 收到 goal
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const VisualGrasp::Goal> goal)
  {
    (void)uuid;
    RCLCPP_INFO(this->get_logger(), "Received goal: target_position = %.2f",
                goal->target_position);
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  // 收到 cancel
  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleVisualGrasp> goal_handle)
  {
    (void)goal_handle;
    RCLCPP_INFO(this->get_logger(), "Received cancel request");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  // goal 被接受后
  void handle_accepted(
    const std::shared_ptr<GoalHandleVisualGrasp> goal_handle)
  {
    // 开线程执行长任务
    std::thread{std::bind(&VisualGraspActionServer::execute, this, std::placeholders::_1),
                goal_handle}
      .detach();
  }

  void execute(const std::shared_ptr<GoalHandleVisualGrasp> goal_handle)
  {
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<VisualGrasp::Feedback>();
    auto result = std::make_shared<VisualGrasp::Result>();

    float current = 0.0f;
    const float target = goal->target_position;

    RCLCPP_INFO(this->get_logger(), "Executing goal: target_position = %.2f", target);

    rclcpp::Rate rate(2.0);  // 2Hz

    while (current < target) {
      if (goal_handle->is_canceling()) {
        RCLCPP_WARN(this->get_logger(), "Goal canceled");
        result->success = false;
        goal_handle->canceled(result);
        return;
      }

      current += 0.5f;
      feedback->current_position = current;
      goal_handle->publish_feedback(feedback);
      RCLCPP_INFO(this->get_logger(), "Feedback: current_position = %.2f", current);

      rate.sleep();
    }

    // 完成
    result->success = true;
    goal_handle->succeed(result);
    RCLCPP_INFO(this->get_logger(), "Goal succeeded");
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<VisualGraspActionServer>();
  node->init();                         
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
