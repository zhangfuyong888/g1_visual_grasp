from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder
import os

def generate_launch_description():

    # ① 加载 MoveIt 的完整配置（URDF / SRDF / kinematics.yaml / controllers）
    moveit_config = (
        MoveItConfigsBuilder("g1_29dof_with_hand",
                             package_name="g1_moveit_config")
        .to_moveit_configs()
    )

    # ② 包含 g1_moveit_config 的 demo.launch.py
    demo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                FindPackageShare("g1_moveit_config").find("g1_moveit_config"),
                "launch",
                "demo.launch.py"
            )
        )
    )

    # ③ 启动你的双臂节点 (延迟 10 秒启动)
    double_arms_control_node = Node(
        package="humanoid_grasp",
        executable="double_arms_control",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    # 延迟启动节点 (等待 10 秒后启动)
    delayed_node = TimerAction(
        period=10.0,  # 延迟 10 秒，可根据需要调整
        actions=[double_arms_control_node]
    )

    return LaunchDescription([
        demo_launch,
        delayed_node
    ])