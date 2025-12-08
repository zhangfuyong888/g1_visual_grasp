cmake_minimum_required(VERSION 3.8)
project(custom_msg_srv)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)


find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)



rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetBestPose.srv"
  DEPENDENCIES geometry_msgs # Add packages that above messages depend on, in this case geometry_msgs for Sphere.msg
)



#add_executable(talker_custom src/pub.cpp)
#ament_target_dependencies(talker_custom rclcpp custom_msg_srv)    # CHANGE

#add_executable(listener_custom src/sub.cpp)
#ament_target_dependencies(listener_custom rclcpp custom_msg_srv)  # CHANGE

## ================ SERVICE
#add_executable(srv_Server_custom src/srv_Server_custom.cpp)
#ament_target_dependencies(srv_Server_custom rclcpp custom_msg_srv)                      # CHANGE

#add_executable(srv_Client_custom src/srv_Client_custom.cpp)
#ament_target_dependencies(srv_Client_custom rclcpp custom_msg_srv)                      # CHANGE

##rosidl_get_typesupport_target(cpp_typesupport_target
##  ${PROJECT_NAME} rosidl_typesupport_cpp)

##target_link_libraries(publish_address_book "${cpp_typesupport_target}")

#install(TARGETS
#  srv_Server_custom
#  srv_Client_custom
#  DESTINATION lib/${PROJECT_NAME})

#install(TARGETS
#  talker_custom
#  listener_custom
#  DESTINATION lib/${PROJECT_NAME})

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
