from copy import deepcopy
import geometry_msgs

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

def pose_to_transform_stamped(pose, origin_frame, child_frame, stamp):
    '''
    geometry_msgs/Pose to geometry_msgs/TransformStamped
    '''
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = origin_frame
    t.child_frame_id = child_frame
    t.transform.translation =  point_to_vector3(pose.position)
    t.transform.orientation = pose.orientation
    
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