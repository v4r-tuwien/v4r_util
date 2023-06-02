import copy
import numpy as np
import open3d as o3d
import tf
from scipy.spatial.transform import Rotation as R
import rospy
from vision_msgs.msg import BoundingBox3D, BoundingBox3DArray
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Pose
from geometry_msgs.msg import Point, Quaternion, Vector3
from grasping_pipeline_msgs.msg import BoundingBox3DStamped
from std_msgs.msg import Header

#TODO should probably split into transform.py and bbstuff.py

def create_ros_bb_stamped(center, extent, rot_mat, frame_id, stamp):
    pos = Point(x = center[0], y = center[1], z = center[2])
    rot_mat_np = np.array(rot_mat)
    rot = R.from_matrix(rot_mat_np)
    quat = rot.as_quat()
    quat_ros = Quaternion(x = quat[0], y = quat[1], z = quat[2], w = quat[3])
    pose = Pose(position = pos, orientation = quat_ros)

    size = Vector3(x = extent[0], y = extent[1], z = extent[2])

    header = Header(frame_id = frame_id, stamp = stamp)

    return BoundingBox3DStamped(center = pose, size = size, header = header)

def o3d_bb_to_ros_bb(o3d_bb):
    '''
    Converts open3d OrientedBoundingBox to ros BoundingBox3D.
    Input: open3d.geometry.OrientedBoundingBox o3d_bb
    Output: vision_msgs/BoundingBox3D ros_bb
    '''
    rot = R.from_matrix(copy.deepcopy(o3d_bb.R))
    quat = rot.as_quat()

    ros_bb = BoundingBox3D()
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
    '''
    Converts open3d OrientedBoundingBox to ros BoundingBox3DStamped.
    Input: open3d.geometry.OrientedBoundingBox o3d_bb
    Output: v4r_msgs/BoundingBox3DStamped ros_bb
    '''
    ros_bb = o3d_bb_to_ros_bb(o3d_bb)
    header = Header(frame_id = frame_id, stamp = stamp)
    ros_bb_stamped = BoundingBox3DStamped(
        position = ros_bb.position, 
        orientation = ros_bb.orientation,
        header = header
        )
    return ros_bb_stamped

def ros_bb_to_o3d_bb(ros_bb):
    '''
    Converts ros BoundingBox3D to open3d OrientedBoundingBox.
    Input: vision_msgs/BoundingBox3D ros_bb
    Output: open3d.geometry.OrientedBoundingBox o3d_bb
    '''
    center = np.array(
        [ros_bb.center.position.x, ros_bb.center.position.y, ros_bb.center.position.z])
    quat = ros_bb.center.orientation
    rot = R.from_quat([quat.x, quat.y, quat.z, quat.w])
    rot_mtx = rot.as_matrix()
    extent = np.array([ros_bb.size.x, ros_bb.size.y, ros_bb.size.z])

    o3d_bb = o3d.geometry.OrientedBoundingBox(center, rot_mtx, extent)

    return o3d_bb


def transformPointCloud(cloud, target_frame, source_frame, tf_buffer):
    ''' 
    Transform pointcloud from source_frame to target_frame
    Input: sensor_msgs/PointCloud2 cloud, string target_frame, string source_frame
            tf2_ros.Buffer tf_buffer
    Output: sensor_msgs/PointCloud2 transformedCloud
    '''

    while not rospy.is_shutdown():
        try:
            transform = tf_buffer.lookup_transform(
                target_frame, source_frame, rospy.Time())
        except:
            continue
        transformedCloud = do_transform_cloud(cloud, transform)
        return transformedCloud


def ros_bb_arr_to_rviz_marker_arr(ros_bb_arr, clear_old_markers=True):
    '''
    Converts BoundingBox3DArray into rviz MarkerArray. If clear_old_markers is set, a delete_all marker
    is added as the first marker so that old rviz markers get cleared.
    Input: vision_msgs/BoundingBox3DArray ros_bb_arr
           bool clear_old_markers
    Output: visualization_msgs/MarkerArray marker_arr
    '''
    # add delete_all as the first marker so that old markers are cleared
    marker_arr = MarkerArray()
    marker_arr.markers = []
    if clear_old_markers:
        marker_delete_all = Marker()
        marker_delete_all.action = marker_delete_all.DELETEALL
        marker_arr.markers.append(marker_delete_all)
    id = 0
    # add marker for each detected object
    for obj in ros_bb_arr.boxes:
        marker = Marker()
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


def o3d_bb_list_to_ros_bb_arr(o3d_bb_list, frame_id, stamp):
    '''
    Converts a list of opend3d OrientedBoundingBoxes to a ros BoundingBox3DArray.
    Input: list[Open3d.geometry.OrientedBoundingBox] o3d_bb_list
           string frame_id
           float stamp
    Output: vision_msgs/BoundingBox3DArray ros_bb_arr
    '''
    bb_arr_ros = BoundingBox3DArray()
    bb_arr_ros.header.frame_id = frame_id
    bb_arr_ros.header.stamp = stamp
    bb_arr_ros.boxes = []
    for bb_o3d in o3d_bb_list:
        bb_arr_ros.boxes.append(o3d_bb_to_ros_bb(bb_o3d))
    return bb_arr_ros


def get_minimum_oriented_bounding_box(o3d_pc):
    '''
    Computes the oriented minimum bounding box of a set of points in 3D space.
    Input: open3d.geometry.PointCloud o3d_pc
    Output: open3d.geometry.OrientedBoundingBox o3d_bb
    '''
    return o3d_pc.get_minimal_oriented_bounding_box(robust=True)


def transform_pose(target_frame, source_frame, pose, listener=None):
    '''
    Transforms pose from source_frame to target_frame using tf.
    Input: string target_frame
           string source_frame
           geometry_msgs/Pose pose
    Output: geometry_msgs/PoseStamped transformed_pose
    '''
    # transform pose from source to target frame
    source_pose = PoseStamped()
    source_pose.header.frame_id = source_frame
    source_pose.header.stamp = rospy.Time(0)
    source_pose.pose = pose
    if listener is None:
        listener = tf.TransformListener()
    listener.waitForTransform(
        source_frame, target_frame, rospy.Time(0), rospy.Duration(4.0))

    try:
        target_pose = listener.transformPose(target_frame, source_pose)
    except tf.Exception:
        rospy.logerr("Transform failure")

    return target_pose


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


def transform_bounding_box(ros_bb, source_frame, target_frame, listener=None):
    '''
    Transforms bounding from source_frame to target_frame using tf.
    Input: vision_msgs/BoundingBox3D ros_bb
           string source_frame
           string target_frame
           tf/Transformlistener listener
    Output: vision_msgs/BoundingBox3D ros_bb
    '''
    if listener is None:
        listener = tf.TransformListener()
    listener.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(4.0))

    try:
        trans, rot = listener.lookupTransform(
            target_frame, source_frame, rospy.Time(0))
    except tf.Exception as e:
        rospy.logerr(f"Transform failure: {e}")
        return ros_bb
    rot_mat = R.from_quat(rot).as_matrix()

    transform = np.eye(4)
    transform[:3, :3] = rot_mat
    transform[:3, 3] = trans

    bb_pose = np.eye(4)
    bb_center = np.array([ros_bb.center.position.x, ros_bb.center.position.y, ros_bb.center.position.z])
    bb_pose[:3, 3] = bb_center
    bb_quat = ros_bb.center.orientation
    bb_rot = R.from_quat([bb_quat.x, bb_quat.y, bb_quat.z, bb_quat.w])
    bb_rot_mtx = bb_rot.as_matrix()
    bb_pose[:3, :3] = bb_rot_mtx

    transformed_pose = transform@bb_pose
    rot = R.from_matrix(transformed_pose[:3, :3])
    quat = rot.as_quat()

    transformed_bb = BoundingBox3D()
    transformed_bb.center.position = Point(transformed_pose[0, 3], transformed_pose[1, 3], transformed_pose[2, 3])
    transformed_bb.center.orientation = Quaternion(quat[0], quat[1], quat[2], quat[3])
    transformed_bb.size = ros_bb.size

    return transformed_bb
