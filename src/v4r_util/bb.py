"""
Module for different functions related to bounding boxes.
"""

# Generic python imports
import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy

# ROS imports
from vision_msgs.msg import BoundingBox3D
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point, Quaternion, Vector3
from grasping_pipeline_msgs.msg import BoundingBox3DStamped
from std_msgs.msg import Header

# V4R imports
import v4r_util.conversions
import v4r_util.alignment

def create_ros_bb_stamped(center, extent, rot_mat, frame_id, stamp):
    """Creates a ROS BoundingBox3DStamped from center, extent and rotation matrix.

    Args:
        center (array like): Center of the bounding box. Format: [x, y, z]
        extent (array like): Extent of the bounding box. Format: [x, y, z]
        rot_mat (array like): Rotation matrix of the bounding box. Used to create the quaternion.
        frame_id (string): Name of the frame of the bounding box.
        stamp (float): Time stamp of the bounding box.

    Returns:
        grasping_pipeline_msgs/BoundingBox3DStamped: ROS bounding box stamped.
    """
    pos = Point(x = center[0], y = center[1], z = center[2])
    rot_mat_np = np.array(rot_mat)
    rot = R.from_matrix(rot_mat_np)
    quat = rot.as_quat()
    quat_ros = Quaternion(x = quat[0], y = quat[1], z = quat[2], w = quat[3])
    pose = Pose(position = pos, orientation = quat_ros)

    size = Vector3(x = extent[0], y = extent[1], z = extent[2])

    header = Header(frame_id = frame_id, stamp = stamp)

    return BoundingBox3DStamped(center = pose, size = size, header = header)


def get_minimum_oriented_bounding_box(o3d_pc):
    '''
    Computes the oriented minimum bounding box of a set of points in 3D space.
    Input: open3d.geometry.PointCloud o3d_pc
    Output: open3d.geometry.OrientedBoundingBox o3d_bb
    '''
    return o3d_pc.get_minimal_oriented_bounding_box(robust=True)


def transform_bounding_box_w_transform(ros_bb, transform):
    """Transforms a bounding box using a transformation matrix.

    Args:
        ros_bb (vision_msgs/BoundingBox3D): ROS bounding box to be transformed.
        transform (geometry_msgs/Transform): Transformation matrix to be applied.
    Returns:
        vision_msgs/BoundingBox3D: Transformed bounding box.
    """
    trans = np.array(
        [transform.translation.x, transform.translation.y, transform.translation.z])
    rot = np.array([transform.rotation.x, transform.rotation.y,
                transform.rotation.z, transform.rotation.w])
    rot = rot/np.linalg.norm(rot)
    rot_mat = R.from_quat(rot).as_matrix()

    transform = np.eye(4)
    transform[:3, :3] = rot_mat
    transform[:3, 3] = trans

    bb_pose = np.eye(4)
    bb_center = np.array(
        [ros_bb.center.position.x, ros_bb.center.position.y, ros_bb.center.position.z])
    bb_pose[:3, 3] = bb_center
    bb_quat = ros_bb.center.orientation
    bb_rot = R.from_quat([bb_quat.x, bb_quat.y, bb_quat.z, bb_quat.w])
    bb_rot_mtx = bb_rot.as_matrix()
    bb_pose[:3, :3] = bb_rot_mtx

    transformed_pose = transform@bb_pose
    rot = R.from_matrix(transformed_pose[:3, :3])
    quat = rot.as_quat()

    transformed_bb = BoundingBox3D()
    transformed_bb.center.position = Point(
        transformed_pose[0, 3], transformed_pose[1, 3], transformed_pose[2, 3])
    transformed_bb.center.orientation = Quaternion(
        quat[0], quat[1], quat[2], quat[3])
    transformed_bb.size = ros_bb.size

    return transformed_bb



# ------------------------------------------------------------------------------
# Conversion functions (from v4r_util.conversions)
# ------------------------------------------------------------------------------
def o3d_bb_to_ros_bb(o3d_bb):
    """Converts open3d OrientedBoundingBox to ros BoundingBox3D. 
    Same as v4r_util.conversions.o3d_bb_to_ros_bb.

    Args:
        o3d_bb (open3d.geometry.OrientedBoundingBox): open3d bounding box.

    Returns:
        vision_msgs/BoundingBox3D: ROS bounding box.
    """
    return v4r_util.conversions.o3d_bb_to_ros_bb(o3d_bb)

def o3d_bb_to_ros_bb_stamped(o3d_bb, frame_id, stamp): 
    """Converts open3d OrientedBoundingBox to ros BoundingBox3DStamped. 
    Same as v4r_util.conversions.o3d_bb_to_ros_bb_stamped.

    Args:
        o3d_bb (o3d.geometry.OrientedBoundingBox): open3d bounding box.
        frame_id (string): Name of the frame of the bounding box.
        stamp (float): Time stamp of the bounding box. Used for the header.

    Returns:
        grasping_pipeline_msgs/BoundingBox3DStamped: ROS bounding box stamped.
    """
    return v4r_util.conversions.o3d_bb_to_ros_bb_stamped(o3d_bb, frame_id, stamp)

def ros_bb_to_o3d_bb(ros_bb): 
    """Converts ros BoundingBox3D to open3d OrientedBoundingBox.
    Same as v4r_util.conversions.ros_bb_to_o3d_bb.

    Args:
        ros_bb (vision_msgs/BoundingBox3D): ROS bounding box.

    Returns:
        open3d.geometry.OrientedBoundingBox: open3d bounding box.
    """
    return v4r_util.conversions.ros_bb_to_o3d_bb(ros_bb)

def ros_bb_arr_to_o3d_bb_list(ros_bb_arr): 
    """Converts ros BoundingBox3DArray to a list of open3d OrientedBoundingBoxes.
    Same as v4r_util.conversions.ros_bb_arr_to_o3d_bb_list.

    Args:
        ros_bb_arr (vision_msgs/BoundingBox3DArray): ROS bounding box array.

    Returns:
        list[Open3d.geometry.OrientedBoundingBox]: List of open3d bounding boxes.
    """
    return v4r_util.conversions.ros_bb_arr_to_o3d_bb_list(ros_bb_arr)

def ros_bb_to_rviz_marker(ros_bb, ns="sasha"): 
    """Converts BoundingBox3D into rviz Marker.
    Same as v4r_util.conversions.ros_bb_to_rviz_marker.

    Args:
        ros_bb (grasping_pipeline_msgs/BoundingBox3DStamped): ROS bounding box.
        ns (str, optional): Namespace of the marker. Defaults to "sasha".

    Returns:
        visualization_msgs/Marker: RViz marker.
    """
    return v4r_util.conversions.ros_bb_to_rviz_marker(ros_bb, ns)

def ros_bb_arr_to_rviz_marker_arr(ros_bb_arr, clear_old_markers=True):
    """Converts BoundingBox3DArray into rviz MarkerArray. If clear_old_markers is set, 
    a delete_all marker is added as the first marker so that old rviz markers get cleared.
    Same as v4r_util.conversions.ros_bb_arr_to_rviz_marker_arr.

    Args:
        ros_bb_arr (vision_msgs/BoundingBox3DArray): ROS bounding box array.
        clear_old_markers (bool, optional): If set, a delete_all marker is added as the first marker. Defaults to True.
    Returns:
        visualization_msgs/MarkerArray: RViz marker array.
    """
    return v4r_util.conversions.ros_bb_arr_to_rviz_marker_arr(ros_bb_arr, clear_old_markers)


def o3d_bb_list_to_ros_bb_arr(o3d_bb_list, frame_id, stamp):
    """Converts a list of opend3d OrientedBoundingBoxes to a ros BoundingBox3DArray.
    Same as v4r_util.conversions.o3d_bb_list_to_ros_bb_arr.

    Args:
        o3d_bb_list (list[Open3d.geometry.OrientedBoundingBox]): List of open3d bounding boxes.
        frame_id (string): Name of the frame of the bounding box.
        stamp (float): Time stamp of the bounding box. Used for the header.

    Returns:
        vision_msgs/BoundingBox3DArray: ROS bounding box array.
    """
    return v4r_util.conversions.o3d_bb_list_to_ros_bb_arr(o3d_bb_list, frame_id, stamp)


# ------------------------------------------------------------------------------
# Alignment functions (from v4r_util.alignment)
# ------------------------------------------------------------------------------
def align_bounding_box_rotation(bb):
    """Aligns bounding box rotation to coordinate axes. This means that the BB x-axis will be the one that aligns the best 
    with the x-axis of the coordinate frame, and so on.
    Same as v4r_util.alignment.align_bounding_box_rotation.

    Args:
        bb (o3d/OrientedBoundingBox): Bounding box to be aligned.

    Raises:
        ValueError: Rotation matrix has negative determinant
        ValueError: Could not find consistent alignment

    Returns:
        o3d/OrientedBoundingBox: Aligned bounding box.
    """
    return v4r_util.alignment.align_bounding_box_rotation(bb)


# ------------------------------------------------------------------------------
# Transform functions. To be removed - please don't use them.
# ------------------------------------------------------------------------------
def transform_bounding_box(ros_bb, source_frame, target_frame, listener=None):
    """Transforms bounding box to target_frame.

    Args:
        ros_bb (grasping_pipeline_msgs/BoundingBox3DStamped): ROS bounding box to be transformed.
        target_frame (string): Name of the target frame.

    Returns:
        grasping_pipeline_msgs/BoundingBox3DStamped: Transformed bounding box.
    """
    rospy.logerr("This function is deprecated. Use transform_bounding_box from the TFWrapper (tf2.py) instead.")
    raise NotImplementedError("This function is deprecated. Use transform_bounding_box from the TFWrapper (tf2.py) instead.")
    