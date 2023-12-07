
import rospy
import copy
from visualization_msgs.msg import MarkerArray, Marker
from vision_msgs.msg import BoundingBox3DArray, BoundingBox3D
from scipy.spatial.transform import Rotation as R


class RvizVisualizer:
    '''
    Allows for easy visualization of BoundingBox3DArray in rviz.
    Initialize with a topic name, and then order the visualization via
    namespace. Cam publish ros and o3d bounding boxes, 
    aswell as clear markers from certain namespaces.
    '''

    def __init__(self, topic="TablePlaneExtractorVisualizer"):
        self.pub = rospy.Publisher(
            topic, 
            MarkerArray, 
            queue_size=10)
        self.markers = {}
        self.ids = {}

    def publish_ros_bb(self, ros_bb, header, namespace="", clear_old_markers=True):
        '''
        Publishes a single BoundingBox3D to rviz in the given namespace.
        '''
        if ros_bb is None:
            rospy.logerr("No bounding box found")
            return
        ros_bb_arr = BoundingBox3DArray()
        ros_bb_arr.header = header
        ros_bb_arr.boxes = [ros_bb]
        marker_arr = self.ros_bb_arr_to_rviz_marker_arr(ros_bb_arr, namespace, clear_old_markers)
        self.pub.publish(marker_arr)

    def publish_ros_bb_arr(self, ros_bb_arr, namespace="", clear_old_markers=True):
        '''
        Publishes a BoundingBox3DArray to rviz in the given namespace.
        '''
        if ros_bb_arr is None:
            rospy.logerr("No bounding boxes found")
            return

        marker_arr = self.ros_bb_arr_to_rviz_marker_arr(ros_bb_arr, namespace, clear_old_markers)
        self.pub.publish(marker_arr)


    def publish_o3d_bb_arr(self, o3d_bb_arr, header, namespace="", clear_old_markers=True):
        '''
        Publishes a o3d boundig box arr to rviz in the given namespace.
        '''

        if o3d_bb_arr is None:
            rospy.logerr("No bounding boxes found")
            return
        ros_bb_arr = BoundingBox3DArray()
        ros_bb_arr.header = header
        ros_bb_arr.boxes = [self.o3d_bb_to_ros_bb(bb) for bb in o3d_bb_arr]
        marker_arr = self.ros_bb_arr_to_rviz_marker_arr(ros_bb_arr, namespace, clear_old_markers)
        self.pub.publish(marker_arr)

    def ros_bb_to_rviz_marker(self, ros_bb, header, namespace="", id=0):
        '''
        Converts BoundingBox3D to rviz Marker.
        Input: vision_msgs/BoundingBox3D ros_bb
        Output: visualization_msgs/Marker marker
        '''
        if ros_bb is None:
            rospy.logerr("No bounding box found")
            return

        if header is None:
            rospy.logerr("No header")
            return

        marker = Marker()
        marker.header.frame_id = header.frame_id
        marker.header.stamp = rospy.get_rostime()
        marker.ns = namespace
        marker.id = id
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.pose = ros_bb.center
        marker.scale = ros_bb.size
        marker.color.g = 1.0
        marker.color.a = 0.6
        return marker
    
    def clear_markers(self, namespace = ""):
        '''
        Removes all markers from the given namespace. 
        If none is given, all markers are removed.
        We have to do this because DELETEALL deletes all markers 
        regardless of the namespace
        '''
        if namespace not in self.markers.keys() and namespace != "":
            rospy.logerr("No markers with namespace {} found".format(namespace))
            return
        marker_arr = MarkerArray()
        marker = Marker()
        marker.action = marker.DELETEALL
        marker_arr.markers.append(marker)

        if namespace != "":
            del self.markers[namespace]
            self.ids[namespace] = 0
            l = [marker for marker_list in self.markers.values() for marker in marker_list.markers]
            marker_arr.markers += l
        self.pub.publish(marker_arr)

    def ros_bb_arr_to_rviz_marker_arr(self, ros_bb_arr, namespace, clear_old_markers=True):
        '''
        Converts BoundingBox3DArray into rviz MarkerArray. If clear_old_markers is set, 
        all markers with the given namespace are removed before publishing.
        If no namespace is given, all markers are removed.
        Input: vision_msgs/BoundingBox3DArray ros_bb_arr bool clear_old_markers
        Output: visualization_msgs/MarkerArray marker_arr
        '''

        marker_arr = MarkerArray()
        marker_arr.markers = []
        if clear_old_markers:
            if namespace in self.markers:
                self.clear_markers(namespace)

        if namespace in self.ids.keys():    
            id = self.ids[namespace]
        else:
            id = 0
        for obj in ros_bb_arr.boxes:
            id += 1
            marker = self.ros_bb_to_rviz_marker(obj, ros_bb_arr.header, namespace, id)
            marker_arr.markers.append(marker)
        self.ids[namespace] = id
        self.markers[namespace] = marker_arr

        return marker_arr
    
    def o3d_bb_to_ros_bb(self, o3d_bb):
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