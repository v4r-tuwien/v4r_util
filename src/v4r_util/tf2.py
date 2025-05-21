import sys
import numpy as np
import rospy
import tf2_geometry_msgs
import tf2_ros
import transforms3d
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion, Vector3Stamped, Vector3, Transform, TransformStamped
from nav_msgs.msg import Path
from vision_msgs.msg import BoundingBox3D
from tf2_sensor_msgs import tf2_sensor_msgs

import v4r_util.conversions
from scipy.spatial.transform import Rotation as R
from grasping_pipeline_msgs.msg import BoundingBox3DStamped

class TF2Wrapper:
    """Coordinate transformation library using TF."""
    
    def __init__(self):

        self.init_ros_time = rospy.Time.now()
        self.update_ros_time = {}
        self.prev_ros_time = self.init_ros_time

        self.tf2buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf2buffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()

    def send_transform(self, origin_frame, child_frame, pose):
        """Broadcast transform from origin_frame to child_frame in tf2

        Args:
            origin_frame (str): Name of the origin frame.
            child_frame (str): Name of the child frame.
            pose (geometry_msgs/Pose): Translation and Rotation from origin_frame to child_frame
        """
        t = v4r_util.conversions.pose_to_transform_stamped(pose, origin_frame, child_frame, rospy.Time.now())
        self.broadcaster.sendTransform(t)

    def get_transform_between_frames(self, source_frame, target_frame, stamp = None,timeout = 3.0):
        """Get transformation from source_frame to target_frame

        Args:
            source_frame (str): Name of the origin frame.
            target_frame (str): Name of the child frame.

        Returns:
            geometry_msgs/TransformStamped: Transformation from source_frame to target_frame.
        """
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            if rospy.Time.now() - start > rospy.Duration(timeout):
                raise TimeoutError("[" + rospy.get_name() + f"]: Transform FAILURE: '{source_frame}' to '{target_frame}': Timeout")
            try:
                if stamp is None:
                    # Used to be rospy.Time.now(), but this often caused transform failures
                    # rospy.Time(0) uses the latest available transform
                    # https://answers.ros.org/question/243179/whats-the-diff-between-rospytime0-and-rospytimenow/
                    stamp = rospy.Time(0)
                trans = self.tf2buffer.lookup_transform(
                    target_frame, 
                    source_frame, 
                    stamp,
                    timeout = rospy.Duration(1.0))
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn("[" + rospy.get_name() + f"]: Transform FAILURE: '{source_frame}' to '{target_frame}': {e}")

        return trans

    def transform_pose(self, target_frame, pose):
        """Transform pose from source_frame to target_frame

        Args:
            target_frame (str): Name of the target frame.
            pose (geometry_msgs/PoseStamped): pose that should be transformed from source to target frame

        Returns:
            geometry_msgs/PoseStamped: Transformed pose.
        """
        while not rospy.is_shutdown():
            
            t = self.get_transform_between_frames(pose.header.frame_id, target_frame, pose.header.stamp)
            p = tf2_geometry_msgs.do_transform_pose(pose, t)
            return p
    
    def transform_point(self, target_frame, point):
        """Transform point from source_frame to target_frame
        
        Args:
            target_frame (str): Name of the target frame.
            point (geometry_msgs/PointStamped): point that should be transformed from source to target frame
            
        Returns:
            geometry_msgs/PointStamped: Transformed point.
        """
        while not rospy.is_shutdown():
            t = self.get_transform_between_frames(point.header.frame_id, target_frame, point.header.stamp)
            p = tf2_geometry_msgs.do_transform_point(point, t)
            return p
    
    def transform_vector3(self, target_frame, vector3):
        """Transform vector3 from target_frame to source_frame
        
        Args:
            target_frame (str): Name of the target frame.
            vector3 (geometry_msgs/Vector3Stamped): vector3 that should be transformed from source to target frame
            
        Returns:
            geometry_msgs/Vector3Stamped: Transformed vector3.
        """
        while not rospy.is_shutdown():
            t = self.get_transform_between_frames(vector3.header.frame_id, target_frame, vector3.header.stamp)
            p = tf2_geometry_msgs.do_transform_vector3(vector3, t)
            return p
    
    def transform_3d_array(self, source_frame, target_frame, array):
        """Transform 3d array from source_frame to target_frame
        
        Args:
            source_frame (str): Name of the source frame.
            target_frame (str): Name of the target frame.
            array (np.ndarray): 3d array representing a vector 
                that should be transformed from source to target frame
            
        Returns:
            np.ndarray: Transformed 3d array.
        """
        header = Header(frame_id = source_frame, stamp = rospy.Time.now())
        vec = Vector3Stamped(header=header, vector=Vector3(x=array[0], y=array[1], z=array[2]))
        transformed_vec = self.transform_vector3(target_frame, vec).vector
        transformed_array = [transformed_vec.x, transformed_vec.y, transformed_vec.z]
        return np.array(transformed_array)
    
    def transform_path(self, target_frame, path):
        """Transform a nav_msgs/path message to target_frame.
        
        Args:
            target_frame (str): Name of the target frame.
            path (nav_msgs/Path): Path that should be transformed to target frame
            
        Returns:
            nav_msgs/Path: Transformed path.
        """
        transformed_path = Path()
        transformed_path.header.frame_id = target_frame
        transformed_path.header.stamp = path.header.stamp

        for pose in path.poses:
            transformed_pose = self.transform_pose(target_frame, pose)
            transformed_path.poses.append(transformed_pose)
        return transformed_path

    def transform_bounding_box(self, ros_bb, target_frame):
        """Transforms bounding box to target_frame.

        Args:
            ros_bb (grasping_pipeline_msgs/BoundingBox3DStamped): ROS bounding box to be transformed.
            target_frame (string): Name of the target frame.

        Returns:
            grasping_pipeline_msgs/BoundingBox3DStamped: Transformed bounding box.
        """
        trans = self.get_transform_between_frames(ros_bb.header.frame_id, target_frame, ros_bb.header.stamp).transform
        translation = trans.translation
        rot = trans.rotation
        rot_mat = R.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()

        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = [translation.x, translation.y, translation.z]

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

        transformed_bb = BoundingBox3DStamped()
        transformed_bb.header.frame_id = target_frame
        transformed_bb.center.position = Point(transformed_pose[0, 3], transformed_pose[1, 3], transformed_pose[2, 3])
        transformed_bb.center.orientation = Quaternion(quat[0], quat[1], quat[2], quat[3])
        transformed_bb.size = ros_bb.size

        return transformed_bb
    
    def transformPointCloud(self, cloud, target_frame, source_frame):
        """Transform pointcloud from source_frame to target_frame.

        Args:
            cloud (sensor_msgs/PointCloud2): PointCloud2 message to be transformed.
            target_frame (string): Name of the target frame.
            source_frame (string): Name of the source frame.

        Returns:
            sensor_msgs/PointCloud2: Transformed PointCloud2 message.
        """
        transform = self.get_transform_between_frames(source_frame, target_frame, cloud.header.stamp)
        return tf2_sensor_msgs.do_transform_cloud(cloud, transform)
    

    
    # ------------------------------------------------------------------------------
    # Moved functions. To be removed - please don't use them.
    # ------------------------------------------------------------------------------
    def quaternion_to_euler(self, quaternion):
        """Convert quaternion to euler RPY angles.

        Args:
            quaternion (geometry_msgs/Quaternion): Quaternions

        Returns:
            float: roll, pitch, yaw [rad].
        """
        rospy.logwarn("This function has been moved to v4r_util.conversions. Please use v4r_util.conversions.quaternion_to_euler instead.")
        return v4r_util.conversions.quaternion_to_euler(quaternion)
    

    def euler_to_quaternion(self, euler):
        """Convert euler RPY angles to quaternion.

        Args:
            euler (geometry_msgs/Point, list): Euler angles

        Returns:
            geometry_msgs/Quaternion: Quaternion.
        """
        rospy.logwarn("This function has been moved to v4r_util.conversions. Please use v4r_util.conversions.euler_to_quaternion instead.")
        return v4r_util.conversions.euler_to_quaternion(euler)
    
    
    def rotmat_to_quaternion(self, rotmat):
        """Convert rotation matrix to quaternion.

        Args:
            rotmat (numpy.ndarray): Rotation matrix.

        Returns:
            geometry_msgs/Quaternion: Quaternion.
        """
        rospy.logwarn("This function has been moved to v4r_util.conversions. Please use v4r_util.conversions.rotmat_to_quaternion instead.")
        return v4r_util.conversions.rotmat_to_quaternion(rotmat)

    def trans_to_transmat(self, trans):
        """
        Transform geometry_msgs/Transform to numpy 4x4 transformation matrix.

        Args:
            geometry_msgs/Transform: Transformation to be converted.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        rospy.logwarn("This function has been moved to v4r_util.conversions. Please use v4r_util.conversions.trans_to_transmat instead.")
        return v4r_util.conversions.trans_to_transmat(trans)
    
    def transmat_to_trans(self, transmat):
        """
        Transform numpy 4x4 transformation matrix to geometry_msgs/Transform.

        Args:
            transmat (np.ndarray): Transformation matrix to be converted.

        Returns:
            geometry_msgs/Transform: Transformed geometry_msgs/Transform.   
        """
        rospy.logwarn("This function has been moved to v4r_util.conversions. Please use v4r_util.conversions.transmat_to_trans instead.")
        return v4r_util.conversions.transmat_to_trans(transmat)
