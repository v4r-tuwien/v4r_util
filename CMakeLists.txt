cmake_minimum_required(VERSION 3.0.2)
project(v4r_util)

find_package(catkin REQUIRED
    rospy
    std_msgs
    vision_msgs
    tf2_sensor_msgs
    visualization_msgs
    )


catkin_package()

include_directories()

catkin_python_setup()
