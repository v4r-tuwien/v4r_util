from copy import deepcopy
import numpy as np
import std_msgs
import geometry_msgs
from grasping_pipeline_msgs.msg import BoundingBox3DStamped
from scipy.spatial.transform import Rotation as R

def point_to_vector3(point):
    '''
    geometry_msgs/Point to geometry_msgs/Vector3
    '''
    return deepcopy(geometry_msgs.msg.Vector3(x = point.x, y = point.y, z = point.z))

def vector3_to_point(vec3):
    '''
    geometry_msgs/Vector3 to geometry_msgs/Point
    '''
    return deepcopy(geometry_msgs.msg.Point(x = vec3.x, y = vec3.y, z = vec3.z))

def vector3_to_list(vec3):
    '''
    geometry_msgs/Vector3 to Pythonlist of len() = 3
    '''
    return deepcopy([vec3.x, vec3.y, vec3.z])

def list_to_vector3(list):
    '''
    Pythonlist of len() = 3 to geometry_msgs/Vector3
    '''
    assert len(list) == 3
    return deepcopy(geometry_msgs.msg.Vector3(x = list[0], y = list[1], z = list[2]))

def pose_to_transform_stamped(pose, origin_frame, child_frame, stamp):
    '''
    geometry_msgs/Pose to geometry_msgs/TransformStamped
    '''
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = origin_frame
    t.child_frame_id = child_frame
    t.transform.translation =  point_to_vector3(pose.position)
    t.transform.rotation = pose.orientation
    
    return deepcopy(t)

def pose_stamped_to_transform_stamped(pose_stamped, child_frame):
    '''
    geometry_msgs/PoseStamped to geometry_msgs/TransformStamped
    '''
    return pose_to_transform_stamped(
        pose_stamped.pose, 
        pose_stamped.header.frame_id, 
        child_frame, 
        pose_stamped.header.stamp)

def transform_stamped_to_pose_stamped(transform):
    '''
    geometry_msgs/TransformStamped to geometry_msgs/PoseStamped
    '''
    p = geometry_msgs.msg.PoseStamped()
    p.header = transform.header
    p.pose.orientation = transform.transform.rotation
    p.pose.position = vector3_to_point(transform.transform.translation)
    
    return deepcopy(p)

def bounding_box_to_bounding_box_stamped(bounding_box, frame_id, stamp):
    '''
    vision_msgs/BoundingBox3D to v4r_msgs/BoundingBox3DStamped
    '''
    header = std_msgs.msg.Header(frame_id = frame_id, stamp = stamp)
    return BoundingBox3DStamped(
        header = header,
        center = bounding_box.center,
        size = bounding_box.size
        )
    
def rot_mat_to_quat(rot_mat):
    '''
    np.array of shape (3,3) to geometry_msgs/Quaternion
    '''
    quat = R.from_matrix(rot_mat).as_quat()
    return geometry_msgs.msg.Quaternion(
        x = quat[0], y = quat[1], z = quat[2], w = quat[3])
    
def quat_to_rot_mat(quat):
    '''
    geometry_msgs/Quaternion to np.array of shape (3,3)
    '''
    quat = [quat.x, quat.y, quat.z, quat.w]
    return R.from_quat(quat).as_matrix()

def np_transform_to_ros_transform(transform):
    '''
    np.array of shape (4,4) to geometry_msgs/Transform
    '''
    t = geometry_msgs.msg.Transform()
    t.translation = list_to_vector3(transform[:3,3])
    t.rotation = rot_mat_to_quat(transform[:3,:3])
    
    return deepcopy(t)

    
def np_transform_to_ros_pose(transform):
    '''
    np.array of shape (4,4) to geometry_msgs/Pose
    '''
    p = geometry_msgs.msg.Pose()
    p.position = list_to_vector3(transform[:3,3])
    p.orientation = rot_mat_to_quat(transform[:3,:3])
    
    return deepcopy(p)

def ros_pose_to_np_transform(pose):
    '''
    geometry_msgs/Pose to np.array of shape (4,4)
    '''
    transform = np.eye(4)
    transform[:3,:3] = quat_to_rot_mat(pose.orientation)
    transform[:3,3] = vector3_to_list(pose.position)
    
    return deepcopy(transform)

def ros_poses_to_np_transforms(poses):
    '''
    list of geometry_msgs/Pose to list of np.array of shape (4,4)
    '''
    transforms = []
    for pose in poses:
        transforms.append(ros_pose_to_np_transform(pose))
    
    return deepcopy(transforms)