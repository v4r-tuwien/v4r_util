
import rospy
from visualization_msgs.msg import MarkerArray, Marker
from v4r_util.util import  o3d_bb_to_ros_bb
from vision_msgs.msg import BoundingBox3DArray

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

    def publish_ros_bb(self, ros_bb, namespace="", clear_old_markers=True):
        '''
        Publishes a single BoundingBox3D to rviz in the given namespace.
        '''
        marker_arr = self.ros_bb_arr_to_rviz_marker_arr([ros_bb], namespace, clear_old_markers)
        self.pub.publish(marker_arr)

    def publish_ros_bb_arr(self, ros_bb_arr, namespace="", clear_old_markers=True):
        '''
        Publishes a BoundingBox3DArray to rviz in the given namespace.
        '''
        marker_arr = self.ros_bb_arr_to_rviz_marker_arr(ros_bb_arr, namespace, clear_old_markers)
        self.pub.publish(marker_arr)


    def publish_o3d_bb_arr(self, o3d_bb_arr, header, namespace="", clear_old_markers=True):
        '''
        Publishes a o3d boundig box arr to rviz in the given namespace.
        '''

        if o3d_bb_arr is None:
            print("No bounding boxes found")
            return
        ros_bb_arr = BoundingBox3DArray()
        ros_bb_arr.header = header
        ros_bb_arr.boxes = [o3d_bb_to_ros_bb(bb) for bb in o3d_bb_arr]
        marker_arr = self.ros_bb_arr_to_rviz_marker_arr(ros_bb_arr, namespace, clear_old_markers)
        self.pub.publish(marker_arr)

    def ros_bb_to_rviz_marker(self, ros_bb, namespace="", id=0, header=None):
        '''
        Converts BoundingBox3D to rviz Marker.
        Input: vision_msgs/BoundingBox3D ros_bb
        Output: visualization_msgs/Marker marker
        '''
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
        '''
        print(self.markers.keys())
        if namespace not in self.markers.keys() and namespace != "":
            print("No markers with namespace {} found".format(namespace))
            return
        marker_arr = MarkerArray()
        marker = Marker()
        marker.action = marker.DELETEALL
        marker_arr.markers.append(marker)

        if namespace != "":
            del self.markers[namespace]
            l = [marker for marker_list in self.markers.values() for marker in marker_list.markers]
            marker_arr.markers += l
        self.pub.publish(marker_arr)

    def ros_bb_arr_to_rviz_marker_arr(self, ros_bb_arr, namespace, clear_old_markers=True):
        '''
        Converts BoundingBox3DArray into rviz MarkerArray. If clear_old_markers is set, a delete_all marker
        is added as the first marker so that old rviz markers get cleared.
        Input: vision_msgs/BoundingBox3DArray ros_bb_arr bool clear_old_markers
        Output: visualization_msgs/MarkerArray marker_arr
        '''

        marker_arr = MarkerArray()
        marker_arr.markers = []
        if clear_old_markers:
            if namespace in self.markers:
                self.clear_markers(namespace)

        for i, obj in enumerate(ros_bb_arr.boxes):
            marker = self.ros_bb_to_rviz_marker(obj, namespace, i, ros_bb_arr.header)
            marker_arr.markers.append(marker)

        self.markers[namespace] = marker_arr

        return marker_arr