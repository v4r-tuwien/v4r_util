"""
Module for converting between ROS messages and/or other formats.
Conversions which don't include ROS messages are in their own modules.
"""

# Generic python imports
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import transforms3d

# ROS imports
import rospy
import std_msgs
import geometry_msgs
import sensor_msgs
import vision_msgs
import visualization_msgs
import ros_numpy
import open3d_ros_helper.open3d_ros_helper as orh

# V4R imports
import grasping_pipeline_msgs.msg
from v4r_util.cmap import get_cmap

# ------------------------------------------------------------------------------
# Basic geometry_msgs conversions
# ------------------------------------------------------------------------------
def point_to_vector3(point):
    """geometry_msgs/Point to geometry_msgs/Vector3"""
    return deepcopy(geometry_msgs.msg.Vector3(x = point.x, y = point.y, z = point.z))

def vector3_to_point(vec3):
    """geometry_msgs/Vector3 to geometry_msgs/Point"""
    return deepcopy(geometry_msgs.msg.Point(x = vec3.x, y = vec3.y, z = vec3.z))

def vector3_to_list(vec3):
    """geometry_msgs/Vector3 to Pythonlist of len() = 3"""
    return deepcopy([vec3.x, vec3.y, vec3.z])

def list_to_vector3(list):
    """Pythonlist of len() = 3 to geometry_msgs/Vector3"""
    assert len(list) == 3
    return deepcopy(geometry_msgs.msg.Vector3(x = list[0], y = list[1], z = list[2]))



# ------------------------------------------------------------------------------
# Pose and Transform conversions (ROS ↔ ROS)
# ------------------------------------------------------------------------------
def pose_to_transform_stamped(pose, origin_frame, child_frame, stamp):
    """geometry_msgs/Pose to geometry_msgs/TransformStamped"""
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = origin_frame
    t.child_frame_id = child_frame
    t.transform.translation =  point_to_vector3(pose.position)
    t.transform.rotation = pose.orientation
    
    return deepcopy(t)

def pose_stamped_to_transform_stamped(pose_stamped, child_frame):
    """geometry_msgs/PoseStamped to geometry_msgs/TransformStamped"""
    return pose_to_transform_stamped(
        pose_stamped.pose, 
        pose_stamped.header.frame_id, 
        child_frame, 
        pose_stamped.header.stamp)

def transform_stamped_to_pose_stamped(transform):
    """geometry_msgs/TransformStamped to geometry_msgs/PoseStamped"""
    p = geometry_msgs.msg.PoseStamped()
    p.header = transform.header
    p.pose.orientation = transform.transform.rotation
    p.pose.position = vector3_to_point(transform.transform.translation)
    
    return deepcopy(p)


# ------------------------------------------------------------------------------
# ROS ↔ NumPy conversions
# ------------------------------------------------------------------------------
def rot_mat_to_quat(rot_mat):
    """np.array of shape (3,3) to geometry_msgs/Quaternion"""
    quat = R.from_matrix(rot_mat).as_quat()
    return geometry_msgs.msg.Quaternion(
        x = quat[0], y = quat[1], z = quat[2], w = quat[3])
    
def quat_to_rot_mat(quat):
    """geometry_msgs/Quaternion to np.array of shape (3,3)"""
    quat = [quat.x, quat.y, quat.z, quat.w]
    return R.from_quat(quat).as_matrix()

def np_transform_to_ros_transform(transform):
    """np.array of shape (4,4) to geometry_msgs/Transform"""
    t = geometry_msgs.msg.Transform()
    t.translation = list_to_vector3(transform[:3,3])
    t.rotation = rot_mat_to_quat(transform[:3,:3])
    
    return deepcopy(t)

    
def np_transform_to_ros_pose(transform):
    """np.array of shape (4,4) to geometry_msgs/Pose"""
    p = geometry_msgs.msg.Pose()
    p.position = list_to_vector3(transform[:3,3])
    p.orientation = rot_mat_to_quat(transform[:3,:3])
    
    return deepcopy(p)

def ros_pose_to_np_transform(pose):
    """geometry_msgs/Pose to np.array of shape (4,4)"""
    transform = np.eye(4)
    transform[:3,:3] = quat_to_rot_mat(pose.orientation)
    transform[:3,3] = vector3_to_list(pose.position)
    
    return deepcopy(transform)

def ros_poses_to_np_transforms(poses):
    """list of geometry_msgs/Pose to list of np.array of shape (4,4)"""
    transforms = []
    for pose in poses:
        transforms.append(ros_pose_to_np_transform(pose))
    
    return deepcopy(transforms)



# ------------------------------------------------------------------------------
# ROS BoundingBox ↔ V4R BoundingBox conversions
# ------------------------------------------------------------------------------
def bounding_box_to_bounding_box_stamped(bounding_box, frame_id, stamp):
    """vision_msgs/BoundingBox3D to grasping_pipeline_msgs/BoundingBox3DStamped"""
    header = std_msgs.msg.Header(frame_id = frame_id, stamp = stamp)
    return grasping_pipeline_msgs.msg.BoundingBox3DStamped(
        header = header,
        center = bounding_box.center,
        size = bounding_box.size
        )

# ------------------------------------------------------------------------------
# ROS BoundingBox ↔ open3d conversions
# ------------------------------------------------------------------------------
def o3d_bb_to_ros_bb(o3d_bb):
    """Converts open3d OrientedBoundingBox to ros BoundingBox3D.

    Args:
        o3d_bb (open3d.geometry.OrientedBoundingBox): open3d bounding box.

    Returns:
        vision_msgs/BoundingBox3D: ROS bounding box.
    """
    rot = R.from_matrix(deepcopy(o3d_bb.R))
    quat = rot.as_quat()

    ros_bb = vision_msgs.msg.BoundingBox3D()
    ros_bb.center.position.x = o3d_bb.center[0]
    ros_bb.center.position.y = o3d_bb.center[1]
    ros_bb.center.position.z = o3d_bb.center[2]
    ros_bb.center.orientation.x = quat[0]
    ros_bb.center.orientation.y = quat[1]
    ros_bb.center.orientation.z = quat[2]
    ros_bb.center.orientation.w = quat[3]
    ros_bb.size.x = o3d_bb.extent[0]
    ros_bb.size.y = o3d_bb.extent[1]
    ros_bb.size.z = o3d_bb.extent[2]

    return ros_bb

def o3d_bb_to_ros_bb_stamped(o3d_bb, frame_id, stamp):
    """Converts open3d OrientedBoundingBox to ros BoundingBox3DStamped.

    Args:
        o3d_bb (o3d.geometry.OrientedBoundingBox): open3d bounding box.
        frame_id (string): Name of the frame of the bounding box.
        stamp (float): Time stamp of the bounding box. Used for the header.

    Returns:
        grasping_pipeline_msgs/BoundingBox3DStamped: ROS bounding box stamped.
    """
    ros_bb = o3d_bb_to_ros_bb(o3d_bb)
    header = std_msgs.msg.Header(frame_id = frame_id, stamp = stamp)
    ros_bb_stamped = grasping_pipeline_msgs.msg.BoundingBox3DStamped(
        center = ros_bb.center,
        size = ros_bb.size,
        header = header
        )
    return ros_bb_stamped

def ros_bb_to_o3d_bb(ros_bb):
    """Converts ros BoundingBox3D to open3d OrientedBoundingBox.

    Args:
        ros_bb (vision_msgs/BoundingBox3D): ROS bounding box.

    Returns:
        open3d.geometry.OrientedBoundingBox: open3d bounding box.
    """
    center = np.array(
        [ros_bb.center.position.x, ros_bb.center.position.y, ros_bb.center.position.z])
    quat = ros_bb.center.orientation
    rot = R.from_quat([quat.x, quat.y, quat.z, quat.w])
    rot_mtx = rot.as_matrix()
    extent = np.array([ros_bb.size.x, ros_bb.size.y, ros_bb.size.z])

    o3d_bb = o3d.geometry.OrientedBoundingBox(center, rot_mtx, extent)

    return o3d_bb

def ros_bb_arr_to_o3d_bb_list(ros_bb_arr):
    """Converts ros BoundingBox3DArray to a list of open3d OrientedBoundingBoxes.

    Args:
        ros_bb_arr (vision_msgs/BoundingBox3DArray): ROS bounding box array.

    Returns:
        list[Open3d.geometry.OrientedBoundingBox]: List of open3d bounding boxes.
    """
    o3d_bb_list = []
    for ros_bb in ros_bb_arr.boxes:
        o3d_bb_list.append(ros_bb_to_o3d_bb(ros_bb))
    return o3d_bb_list

def o3d_bb_list_to_ros_bb_arr(o3d_bb_list, frame_id, stamp):
    """Converts a list of opend3d OrientedBoundingBoxes to a ros BoundingBox3DArray.

    Args:
        o3d_bb_list (list[Open3d.geometry.OrientedBoundingBox]): List of open3d bounding boxes.
        frame_id (string): Name of the frame of the bounding box.
        stamp (float): Time stamp of the bounding box. Used for the header.

    Returns:
        vision_msgs/BoundingBox3DArray: ROS bounding box array.
    """
    bb_arr_ros = vision_msgs.msg.BoundingBox3DArray()
    bb_arr_ros.header.frame_id = frame_id
    bb_arr_ros.header.stamp = stamp
    bb_arr_ros.boxes = []
    for bb_o3d in o3d_bb_list:
        bb_arr_ros.boxes.append(o3d_bb_to_ros_bb(bb_o3d))
    return bb_arr_ros

# ------------------------------------------------------------------------------
# ROS BoundingBox ↔ RViz Marker conversions
# ------------------------------------------------------------------------------
def ros_bb_to_rviz_marker(ros_bb, ns="sasha"):
    """Converts BoundingBox3D into rviz Marker.

    Args:
        ros_bb (grasping_pipeline_msgs/BoundingBox3DStamped): ROS bounding box.
        ns (str, optional): Namespace of the marker. Defaults to "sasha".

    Returns:
        visualization_msgs/Marker: RViz marker.
    """
    marker = visualization_msgs.msg.Marker()
    marker.header.frame_id = ros_bb.header.frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = 0
    marker.type = marker.CUBE
    marker.action = marker.ADD
    marker.pose = ros_bb.center
    marker.scale = ros_bb.size
    marker.color.g = 1.0
    marker.color.a = 0.6
    return marker

def ros_bb_arr_to_rviz_marker_arr(ros_bb_arr, clear_old_markers=True):
    """Converts BoundingBox3DArray into rviz MarkerArray. If clear_old_markers is set, 
    a delete_all marker is added as the first marker so that old rviz markers get cleared.

    Args:
        ros_bb_arr (vision_msgs/BoundingBox3DArray): ROS bounding box array.
        clear_old_markers (bool, optional): If set, a delete_all marker is added as the first marker. Defaults to True.
    Returns:
        visualization_msgs/MarkerArray: RViz marker array.
    """
    # add delete_all as the first marker so that old markers are cleared
    marker_arr = visualization_msgs.msg.MarkerArray()
    marker_arr.markers = []
    if clear_old_markers:
        marker_delete_all = visualization_msgs.msg.Marker()
        marker_delete_all.action = marker_delete_all.DELETEALL
        marker_arr.markers.append(marker_delete_all)
    id = 0
    # add marker for each detected object
    for obj in ros_bb_arr.boxes:
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = ros_bb_arr.header.frame_id
        marker.header.stamp = rospy.get_rostime()
        marker.ns = "ObjectsOnTable"
        marker.id = id
        id = id + 1
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.pose = obj.center
        marker.scale = obj.size
        marker.color.g = 1.0
        marker.color.a = 0.6
        marker_arr.markers.append(marker)
    return marker_arr



# ------------------------------------------------------------------------------
# Angles/Transformation/Quaternions conversions
# ------------------------------------------------------------------------------
def quaternion_to_euler(quaternion):
    """Convert quaternion to euler RPY angles.

    Args:
        quaternion (geometry_msgs/Quaternion): Quaternions

    Returns:
        float: roll, pitch, yaw [rad].
    """
    angles = transforms3d.euler.quat2euler([quaternion.w, quaternion.x, quaternion.y, quaternion.z])

    return angles[0], angles[1], angles[2]


def euler_to_quaternion(euler):
    """Convert euler RPY angles to quaternion.

    Args:
        euler (geometry_msgs/Point, list): Euler angles

    Returns:
        geometry_msgs/Quaternion: Quaternion.
    """
    if isinstance(euler, geometry_msgs.msg.Point):
        q = transforms3d.euler.euler2quat(euler.x, euler.y, euler.z)
    elif isinstance(euler, list) or isinstance(euler, tuple):
        q = transforms3d.euler.euler2quat(euler[0], euler[1], euler[2])

    return geometry_msgs.msg.Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])

def rotmat_to_quaternion(rotmat):
    """Convert rotation matrix to quaternion.

    Args:
        rotmat (numpy.ndarray): Rotation matrix.

    Returns:
        geometry_msgs/Quaternion: Quaternion.
    """
    q = R.from_matrix(rotmat).as_quat()
    return geometry_msgs.msg.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

def trans_to_transmat(trans):
    """
    Transform geometry_msgs/Transform to numpy 4x4 transformation matrix.

    Args:
        geometry_msgs/Transform: Transformation to be converted.

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    rot = transforms3d.quaternions.quat2mat(
        [trans.transform.rotation.w, trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z])
    transmat = np.eye(4)
    transmat[:3, :3] = rot
    transmat[:3, 3] = [trans.transform.translation.x,
                    trans.transform.translation.y, trans.transform.translation.z]
    return transmat

def transmat_to_trans(transmat):
    """
    Transform numpy 4x4 transformation matrix to geometry_msgs/Transform.

    Args:
        transmat (np.ndarray): Transformation matrix to be converted.

    Returns:
        geometry_msgs/Transform: Transformed geometry_msgs/Transform.   
    """
    quat = transforms3d.quaternions.mat2quat(transmat[:3, :3])

    transform = geometry_msgs.msg.Transform()
    transform.translation = geometry_msgs.msg.Vector3(transmat[0, 3], transmat[1, 3], transmat[2, 3])
    transform.rotation = geometry_msgs.msg.Quaternion(quat[1], quat[2], quat[3], quat[0])

    return transform


# ------------------------------------------------------------------------------
# ROS Depth images / open3d pcd / numpy conversions
# ------------------------------------------------------------------------------
def convert_np_depth_img_to_o3d_pcd(depth_img, ros_cam_info, depth_scale=1000, project_valid_depth_only=True):
    """
    Transform a numpy depth image to an Open3D point cloud.
    
    Args:
        depth_img (np.ndarray): Depth image
        ros_cam_info (sensor_msgs/CameraInfo): Camera info message containing intrinsic parameters.
        depth_scale (float): Scale factor for depth values (e.g. 1000 for mm to m).
        project_valid_depth_only (bool): Set to False if you want to use pixel indices from a 2D image. Otherwise some pixels might be removed, which means the indices between image and o3d_pcd won't align.
 
    Returns:
        o3d.geometry.PointCloud: Open3D point cloud.
    """
    width = ros_cam_info.width
    height = ros_cam_info.height
    intrinsics = np.array(ros_cam_info.K).reshape(3, 3)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    cam_intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    depth_img_o3d = o3d.geometry.Image(depth_img.astype(np.uint16))

    o3d_pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_img_o3d, 
            cam_intr, 
            depth_scale=depth_scale, 
            project_valid_depth_only=project_valid_depth_only)
    return o3d_pcd

def convert_ros_depth_img_to_pcd(ros_depth_img, ros_cam_info, depth_scale=1000, project_valid_depth_only=True):
    """
    Converts a ROS depth image to a point cloud.

    Args:
        ros_depth_img (sensor_msgs/Image): ROS depth image message.
        ros_cam_info (sensor_msgs/CameraInfo): ROS camera info message.
        depth_scale (float): Scale factor for depth values (e.g. 1000 for mm to m).
        project_valid_depth_only (bool): Set to False if you want to use pixel indices from a 2D image. Otherwise some pixels might be removed, which means the indices between image and o3d_pcd won't align.
    
    Returns:
        ros_pcd (sensor_msgs/PointCloud2): ROS point cloud message.
        o3d_pcd (open3d.geometry.PointCloud): Open3D point cloud object.
    """
    depth_img = ros_numpy.numpify(ros_depth_img)
    o3d_pcd = convert_np_depth_img_to_o3d_pcd(depth_img, ros_cam_info, depth_scale, project_valid_depth_only)
    ros_pcd = orh.o3dpc_to_rospc(o3d_pcd, frame_id = ros_depth_img.header.frame_id, stamp=ros_depth_img.header.stamp)
    return ros_pcd, o3d_pcd

def convert_np_label_img_to_np_color_img(np_label_img, np_rgb_img):
    assert(np_rgb_img.dtype == np.uint8)
    colors = get_cmap(np_label_img / (np_label_img.max() if np_label_img.max() > 0 else 1))
    # RGBA float64 to RGB uint8
    colors = (colors[:, :, :3] * 255).astype(np.uint8)
    colors[np_label_img < 0] = np_rgb_img[np_label_img < 0]
    return colors

def convert_np_label_img_to_ros_color_img(np_label_img, np_rgb_img):
    colors = convert_np_label_img_to_np_color_img(np_label_img, np_rgb_img)
    return ros_numpy.msgify(sensor_msgs.msg.Image, colors, encoding='rgb8')