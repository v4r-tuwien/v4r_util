import sys
import rospy
import tf2_geometry_msgs
import tf2_ros
import transforms3d
from geometry_msgs.msg import Point, Quaternion
from v4r_util.conversions import pose_to_transform_stamped
import tf

class TF2Wrapper:
    """Coordinate transformation library using TF."""
    
    def __init__(self):

        self.init_ros_time = rospy.Time.now()
        self.update_ros_time = {}
        self.prev_ros_time = self.init_ros_time

        self.tf2buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf2buffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()

    def send_transform(self,
                       origin_frame,
                       child_frame,
                       pose):
        """Broadcast transform from origin_frame to child_frame in tf2

        Args:
            origin_frame (str): Name of the origin frame.
            child_frame (str): Name of the child frame.
            pose (geometry_msgs/Pose): Translation and Rotation from origin_frame to child_frame
        """
        t = pose_to_transform_stamped(pose, rospy.Time.now(), origin_frame, child_frame)
        self.broadcaster.sendTransform(t)


    def get_transform_between_frames(self,
                             source_frame,
                             target_frame,
                             stamp = None):
        """Get transformation from source_frame to target_frame

        Args:
            source_frame (str): Name of the origin frame.
            target_frame (str): Name of the child frame.

        Returns:
            geometry_msgs/Transform: Transformation from source_frame to target_frame.
        """
        while not rospy.is_shutdown():
            try:
                if stamp is None:
                    stamp = rospy.Time.now()
                trans = self.tf2buffer.lookup_transform(
                    target_frame, 
                    source_frame, 
                    stamp,
                    timeout = rospy.Duration(1.0))
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn("[" + rospy.get_name() + f"]: Transform FAILURE: '{source_frame}' to '{target_frame}': {e}")
                continue

        return trans

    def transform_pose(
        self,
        target_frame,
        pose):
        """Transform pose from target_frame to source_frame

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


    def quaternion2euler(self, quaternion):
        """Convert quaternion to euler RPY angles.

        Args:
            quaternion (geometry_msgs/Quaternion): Quaternions

        Returns:
            float: roll, pitch, yaw [rad].
        """
        angles = transforms3d.euler.quat2euler([quaternion.w, quaternion.x, quaternion.y, quaternion.z])
    
        return angles[0], angles[1], angles[2]
    

    def euler2quaternion(self, euler):
        """Convert euler RPY angles to quaternion.

        Args:
            euler (geometry_msgs/Point, list): Euler angles

        Returns:
            geometry_msgs/Quaternion: Quaternion.
        """
        if isinstance(euler, Point):
            q = transforms3d.euler.euler2quat(euler.x, euler.y, euler.z)
        elif isinstance(euler, list) or isinstance(euler, tuple):
            q = transforms3d.euler.euler2quat(euler[0], euler[1], euler[2])

        return Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])
