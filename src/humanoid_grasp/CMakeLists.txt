cmake_minimum_required(VERSION 3.8)
project(humanoid_grasp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)

# 自动查找 src 目录下的所有 C++ 文件
file(GLOB_RECURSE CPP_FILES "src/*.cpp")

# 查找其他必要的依赖包
#find_package(g1_moveit_config REQUIRED)
find_package(moveit_visual_tools REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(unitree_ros2_example REQUIRED)
find_package(unitree_hg REQUIRED)
#find_package(realsense2_camera REQUIRED)
find_package(custom_msg_srv REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)




# 生成接口文件
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/VisualGrasp.action"  # 添加你的 action 文件
  DEPENDENCIES std_msgs  # 如果有依赖的其他消息
)

ament_export_dependencies(rosidl_default_runtime)



# 遍历所有 C++ 文件，为每个文件生成一个可执行文件
foreach(CPP_FILE ${CPP_FILES})
    # 获取源文件的基本名称（不含路径和扩展名）
    get_filename_component(EXE_NAME ${CPP_FILE} NAME_WE)

    # 创建一个可执行文件，其名称与源文件相同
    add_executable(${EXE_NAME} ${CPP_FILE})

    # 设置目标的头文件搜索路径
    target_include_directories(${EXE_NAME}
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include  # 添加 include 目录

    )

    # 链接库：添加 ROS 2 和其他必要的依赖
    ament_target_dependencies(${EXE_NAME}
      rclcpp
      moveit_visual_tools
      moveit_ros_planning_interface
      moveit_msgs
      unitree_ros2_example
      unitree_hg
      std_msgs
      custom_msg_srv
      kdl_parser
      Eigen3
      rosidl_default_generators
      rclcpp_action
    )

    # 拿到 C++ typesupport 的 target
    rosidl_get_typesupport_target(cpp_typesupport_target
      ${PROJECT_NAME} "rosidl_typesupport_cpp"
    )

    # 把这个 typesupport target 链接到你的可执行文件上
    target_link_libraries(${EXE_NAME}
      ${cpp_typesupport_target}
    )
    # 安装可执行文件到指定位置
    install(TARGETS ${EXE_NAME}
      DESTINATION lib/${PROJECT_NAME}
    )
endforeach()

# 安装 launch 文件
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
