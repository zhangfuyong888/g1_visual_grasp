cmake_minimum_required(VERSION 3.8)

project(giim_seg)

find_package(ament_cmake REQUIRED)
find_package(ament_python REQUIRED)

find_package(rclpy REQUIRED)

find_package(rosidl_default_generators REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(giim_seg REQUIRED)                      # CHANGE

find_package(custom_msg_srv REQUIRED) 

#rosidl_generate_interfaces(${PROJECT_NAME}
#  "srv/GetBestPose.srv"
#  DEPENDENCIES geometry_msgs
#)

ament_package()
