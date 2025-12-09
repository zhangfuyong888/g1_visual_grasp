from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder
import os

def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("g1_29dof_with_hand", package_name="g1_moveit_config")
        .to_moveit_configs()
    )

    demo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                FindPackageShare("g1_moveit_config").find("g1_moveit_config"),
                "launch",
                "demo.launch.py"
            )
        )
    )

    action_server_node = Node(
        package="humanoid_grasp",
        executable="action_server",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {"use_sim": True}  # 开启仿真模式
        ],
    )

    delayed_action_server = TimerAction(
        period=5.0,
        actions=[action_server_node]
    )

    return LaunchDescription([
        demo_launch,
        delayed_action_server
    ])