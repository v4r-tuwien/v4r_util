cmake_minimum_required(VERSION 3.0.2)
project(v4r_util)

find_package(catkin REQUIRED
    rospy
    std_msgs
    vision_msgs
    tf2_sensor_msgs
    visualization_msgs
    geometry_msgs
    sensor_msgs
    ros_numpy
    #TODO fix circular dependency properly:
    # v4r_util <=> table_plane_extractor <=> grasping_pipeline_msgs
    # grasping_pipeline_msgs
    )


catkin_package()

include_directories()

catkin_python_setup()
