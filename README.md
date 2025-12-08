# 宇树人形机器人抓取开发包

## 环境：
1. ubuntu 22.04  
2. ros2 humble
3. 安装官方 unitree_ros包
4. 安装moveit相关开发包：
sudo apt install ros-humble-moveit-*
5. 安装realsense ros相关包  在/opt/ros/humble/share/realsense2_camera/launch/rs_launch.py 中修改
{'name': 'pointcloud.enable',   'default': 'true', 'description': ''},
6. 安装视觉检测相关包：
sudo apt update
sudo apt install python3-pip
pip3 install ultralytics open3d==0.18.0


## 架构：
g1_ws
├── README.md
├── build
├── install
├── log
└── src
    ├── custom_msg_srv     // 获取抓取点的自定义服务
    ├── g1_moveit_config   // moveit配置文件，机器人可视化
    ├── giim_seg           // 视觉检测
    ├── humanoid_grasp     // 抓取主程序
    └── unitree_ros2       // 官方unitree_ros2包



## 运行：
 ├── custom_msg_srv     // 获取抓取点的自定义服务
    ├── g1_moveit_config   // moveit配置文件，机器人可视化
    ├── giim_seg           // 视觉检测
    ├── humanoid_grasp     // 抓取主程序
    └── unitree_ros2       // 官方unitree_ros2包
### 双臂抓取demo

1. 启动相机
    ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=1280x720x30 rgb_camera.color_profile:=1280x720x30 align_depth.enable:=true
2. 启动识别服务
    ros2 run giim_seg pcl_best_target_Plane_Pose
    测试 ： ros2 service call /get_best_pose custom_msg_srv/srv/GetBestPose "{is_left_hand: 0}"     //请求物料位置
3. 可视化
    ros2 launch g1_moveit_config demo.launch.py
4. 启动机器人
    ros2 run humanoid_grasp double_arms_control         //底层控制
    ros2 run humanoid_grasp double_arm_grab_demo        //抓取流程

    ros2 run humanoid_grasp moveit_controller           //moveit底层控制

    ros2 topic pub /arm_left/reached_target std_msgs/msg/Bool "{data: true}" --once


### 单臂运动控制测试-12.2
先测试传统插值：
    ros2 run humanoid_grasp double_arms_control
单臂测试流程：
     ros2 run humanoid_grasp single_arm_demo

#### 运动控制测试之后，手臂如果不往下掉，就用action封装起来

用moveit规划：
    ros2 run humanoid_grasp moveit_controller
    代码中发布joint_states --->>  js_pub_->publish(js);取消注释； ros2_control中关闭joint_state_broadcaster。


### 通讯架构：

    ①左臂移动[action]：（子任务让功能actino调用）
    ②右臂移动[action]：（子任务让功能actino调用）


    左臂抓取[action]：  
                获取位置点（左）[server]
                ①左臂移动[action]

    扫码（到达位置点后）[action]：
                二维码识别[action]
                ①左臂移动（放置）[action]  

    右臂抓取[action]：  
                获取位置点（右）[server]
                ②右臂移动[action]



## demo更新优化（完成抓取、扫码、放置demo）（计划12.8-12.6）
#TODO
1.手臂运动接口转为action接口，拆解抓取流程转化为action动作。

2.关节运动转为笛卡尔末端运动，增加运动范围可解性。考虑末端步进运动

3.手部抓取姿态