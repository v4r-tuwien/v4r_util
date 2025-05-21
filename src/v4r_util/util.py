"""Module for general utility functions that can't be classified into a specific category."""

# Generic python imports
import numpy as np

# ROS imports
import rospy
from geometry_msgs.msg import Pose

# V4R imports
import v4r_util.conversions
import v4r_util.alignment
import v4r_util.bb

def transformPoseFormat(pose, format_str):
    if format_str == "tuple":
        new = Pose()
        new.position.x = pose.pos.x
        new.position.y = pose.pos.y
        new.position.z = pose.pos.z
        new.orientation.x = pose.ori.x
        new.orientation.y = pose.ori.y
        new.orientation.z = pose.ori.z
        new.orientation.w = pose.ori.w
        return new
    elif format_str == "pose":
        return ((pose.position.x, pose.position.y, pose.position.z), (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))
    else:
        return None

def rotmat_around_axis(axis, angle):
    axis = axis / np.linalg.norm(axis)

    rot_mat = np.eye(3)
    rot_mat[0, 0] = np.cos(angle) + axis[0]**2 * (1 - np.cos(angle))
    rot_mat[0, 1] = axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle)
    rot_mat[0, 2] = axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)
    rot_mat[1, 0] = axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle)
    rot_mat[1, 1] = np.cos(angle) + axis[1]**2 * (1 - np.cos(angle))
    rot_mat[1, 2] = axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle)
    rot_mat[2, 0] = axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle)
    rot_mat[2, 1] = axis[2] * axis[1] * (1 - np.cos(angle)) + axis[0] * np.sin(angle)
    rot_mat[2, 2] = np.cos(angle) + axis[2]**2 * (1 - np.cos(angle))
    
    return rot_mat  



# ------------------------------------------------------------------------------
# Deprecated/Moved functions. To be removed - please don't use them.
# ------------------------------------------------------------------------------
def create_ros_bb_stamped(center, extent, rot_mat, frame_id, stamp):
    rospy.logwarn("This function has been moved to v4r_util.bb. Please use v4r_util.bb.create_ros_bb_stamped instead.")
    return v4r_util.bb.create_ros_bb_stamped(center, extent, rot_mat, frame_id, stamp)

def o3d_bb_to_ros_bb(o3d_bb): # Copied into v4r_conversions
    '''
    Converts open3d OrientedBoundingBox to ros BoundingBox3D.
    Input: open3d.geometry.OrientedBoundingBox o3d_bb
    Output: vision_msgs/BoundingBox3D ros_bb
    '''
    rospy.logwarn("This function has been moved to v4r_util.conversions. Please use v4r_util.conversions.o3d_bb_to_ros_bb instead.")
    return v4r_util.conversions.o3d_bb_to_ros_bb(o3d_bb)

def o3d_bb_to_ros_bb_stamped(o3d_bb, frame_id, stamp): # Copied into v4r_conversions
    '''
    Converts open3d OrientedBoundingBox to ros BoundingBox3DStamped.
    Input: open3d.geometry.OrientedBoundingBox o3d_bb
    Output: v4r_msgs/BoundingBox3DStamped ros_bb
    '''
    rospy.logwarn("This function has been moved to v4r_util.conversions. Please use v4r_util.conversions.o3d_bb_to_ros_bb_stamped instead.")
    return v4r_util.conversions.o3d_bb_to_ros_bb_stamped(o3d_bb, frame_id, stamp)

def ros_bb_to_o3d_bb(ros_bb): # Copied into v4r_conversions
    '''
    Converts ros BoundingBox3D to open3d OrientedBoundingBox.
    Input: vision_msgs/BoundingBox3D ros_bb
    Output: open3d.geometry.OrientedBoundingBox o3d_bb
    '''
    rospy.logwarn("This function has been moved to v4r_util.conversions. Please use v4r_util.conversions.ros_bb_to_o3d_bb instead.")
    return v4r_util.conversions.ros_bb_to_o3d_bb(ros_bb)

def ros_bb_arr_to_o3d_bb_list(ros_bb_arr): # Copied into v4r_conversions
    '''
    Converts ros BoundingBox3DArray to a list of open3d OrientedBoundingBoxes.
    Input: vision_msgs/BoundingBox3DArray ros_bb_arr
    Output: list[Open3d.geometry.OrientedBoundingBox] o3d_bb_list
    '''
    rospy.logwarn("This function has been moved to v4r_util.conversions. Please use v4r_util.conversions.ros_bb_arr_to_o3d_bb_list instead.")
    return v4r_util.conversions.ros_bb_arr_to_o3d_bb_list(ros_bb_arr)

def transformPointCloud(cloud, target_frame, source_frame, tf_buffer): # Copied into v4r.tf2
    ''' 
    Transform pointcloud from source_frame to target_frame
    Input: sensor_msgs/PointCloud2 cloud, string target_frame, string source_frame
            tf2_ros.Buffer tf_buffer
    Output: sensor_msgs/PointCloud2 transformedCloud
    '''
    rospy.logerr("This function is deprecated. Use transform_cloud form the TFWrapper (tf2.py) instead.")
    raise NotImplementedError("This function is deprecated. Use transform_cloud form the TFWrapper (tf2.py) instead.")

def ros_bb_to_rviz_marker(ros_bb, ns="sasha"): # Copied into v4r_conversions
    '''
    Converts BoundingBox3D into rviz Marker.
    Input: grasping_pipeline_msgs/BoundingBox3DStamped ros_bb
    Output: visualization_msgs/Marker marker
    '''
    rospy.logwarn("This function has been moved to v4r_util.conversions. Please use v4r_util.conversions.ros_bb_to_rviz_marker instead.")
    return v4r_util.conversions.ros_bb_to_rviz_marker(ros_bb, ns)
    

def ros_bb_arr_to_rviz_marker_arr(ros_bb_arr, clear_old_markers=True): # Copied into v4r_conversions
    '''
    Converts BoundingBox3DArray into rviz MarkerArray. If clear_old_markers is set, a delete_all marker
    is added as the first marker so that old rviz markers get cleared.
    Input: vision_msgs/BoundingBox3DArray ros_bb_arr
           bool clear_old_markers
    Output: visualization_msgs/MarkerArray marker_arr
    '''
    rospy.logwarn("This function has been moved to v4r_util.conversions. Please use v4r_util.conversions.ros_bb_arr_to_rviz_marker_arr instead.")
    return v4r_util.conversions.ros_bb_arr_to_rviz_marker_arr(ros_bb_arr, clear_old_markers)


def o3d_bb_list_to_ros_bb_arr(o3d_bb_list, frame_id, stamp): # Copied into v4r_conversions
    '''
    Converts a list of opend3d OrientedBoundingBoxes to a ros BoundingBox3DArray.
    Input: list[Open3d.geometry.OrientedBoundingBox] o3d_bb_list
           string frame_id
           float stamp
    Output: vision_msgs/BoundingBox3DArray ros_bb_arr
    '''
    rospy.logwarn("This function has been moved to v4r_util.conversions. Please use v4r_util.conversions.o3d_bb_list_to_ros_bb_arr instead.")
    return v4r_util.conversions.o3d_bb_list_to_ros_bb_arr(o3d_bb_list, frame_id, stamp)


def get_minimum_oriented_bounding_box(o3d_pc): # copied into bb.py
    '''
    Computes the oriented minimum bounding box of a set of points in 3D space.
    Input: open3d.geometry.PointCloud o3d_pc
    Output: open3d.geometry.OrientedBoundingBox o3d_bb
    '''
    rospy.logwarn("This function has been moved to v4r_util.bb. Please use v4r_util.bb.get_minimum_oriented_bounding_box instead.")
    return v4r_util.bb.get_minimum_oriented_bounding_box(o3d_pc)

 
def transform_pose(target_frame, source_frame, pose, listener=None): # Copied into v4r.tf2
    '''
    Transforms pose from source_frame to target_frame using tf.
    Input: string target_frame
           string source_frame
           geometry_msgs/Pose pose
    Output: geometry_msgs/PoseStamped transformed_pose
    '''
    rospy.logerr("This function is deprecated. Use transform_pose form the TFWrapper (tf2.py) instead.")
    raise NotImplementedError("This function is deprecated. Use transform_pose form the TFWrapper (tf2.py) instead.")


def transform_bounding_box(ros_bb, source_frame, target_frame, listener=None): # copied into v4r.tf2
    '''
    Transforms bounding from source_frame to target_frame using tf.
    Input: vision_msgs/BoundingBox3D ros_bb
           string source_frame
           string target_frame
           tf/Transformlistener listener
    Output: vision_msgs/BoundingBox3D ros_bb
    '''
    rospy.logerr("This function is deprecated. Use transform_bounding_box form the TFWrapper (tf2.py) instead.")
    raise NotImplementedError("This function is deprecated. Use transform_bounding_box form the TFWrapper (tf2.py) instead.")
 

def get_best_aligning_axis(coordinate_axis_vec, reference_frame_axes): # copied into alignment.py
    rospy.logwarn("This function has been moved to v4r_util.alignment. Please use v4r_util.alignment.get_best_aligning_axis instead.")
    return v4r_util.alignment.get_best_aligning_axis(coordinate_axis_vec, reference_frame_axes)

def get_best_aligning_axis_to_world_frame(coordinate_axis_vec): # copied into alignment.py
    rospy.logwarn("This function has been moved to v4r_util.alignment. Please use v4r_util.alignment.get_best_aligning_axis_to_world_frame instead.")
    return v4r_util.alignment.get_best_aligning_axis_to_world_frame(coordinate_axis_vec)

    
def align_transformation(transform): # copied into alignment.py
    '''
    Aligns transformation to coordinate axes. This means that the x-axis will be the one that aligns the best
    with the x-axis of the coordinate frame, and so on.
    Input: np.array homogenous transformation (4x4)
    Output: np.array aligned homogenous transformation (4x4)
    '''
    rospy.logwarn("This function has been moved to v4r_util.alignment. Please use v4r_util.alignment.align_transformation instead.")
    return v4r_util.alignment.align_transformation(transform)

def align_pose_rotation(pose):# copied into alignment.py
    # ros pose, not pose stamped
    rospy.logwarn("This function has been moved to v4r_util.alignment. Please use v4r_util.alignment.align_pose_rotation instead.")
    return v4r_util.alignment.align_pose_rotation(pose)

def align_bounding_box_rotation(bb): # copied into alignment.py
    '''
    Aligns bounding box rotation to coordinate axes. This means that the BB x-axis will be the one that aligns the best 
    with the x-axis of the coordinate frame, and so on.
    Input: o3d/OrientedBoundingBox bb
    Output: o3d/OrientedBoundingBox bb
    '''
    rospy.logwarn("This function has been moved to v4r_util.alignment. Please use v4r_util.alignment.align_bounding_box_rotation instead.")
    return v4r_util.alignment.align_bounding_box_rotation(bb)



def transform_bounding_box_w_transform(ros_bb, transform): # copied into bb.py
    rospy.logwarn("This function has been moved to v4r_util.bb. Please use v4r_util.bb.transform_bounding_box_w_transform instead.")
    return v4r_util.bb.transform_bounding_box_w_transform(ros_bb, transform)