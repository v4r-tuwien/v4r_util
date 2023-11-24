import numpy as np
import rospy
from sensor_msgs.msg import Image
import cv2
import open3d as o3d
import ros_numpy
from v4r_util.conversions import ros_poses_to_np_transforms
import open3d.visualization.rendering as rendering

class PoseEstimationVisualizer():
    
    def __init__(self, topic, image_width, image_height, intrinsics_matrix):
        '''
        Renders contours of models and modelnames into an image
        topic: str, topic to publish visualization to
        image_width: int, width of image
        image_height: int, height of image
        intrinsics_matrix: flattened camera matrix
        '''
        self.image_pub = rospy.Publisher(topic, Image, queue_size=10)
        self.image_width = image_width
        self.image_height = image_height
        self.renderer = rendering.OffscreenRenderer(image_width, image_height)
        fx = intrinsics_matrix[0]
        fy = intrinsics_matrix[5]
        cx = intrinsics_matrix[2]
        cy = intrinsics_matrix[6]
        pinhole = o3d.camera.PinholeCameraIntrinsic(
            image_width, 
            image_height, 
            fx, 
            fy, 
            cx, 
            cy)
        self.renderer.setup_camera(pinhole, np.eye(4))
    
    def publish_pose_estimation_result(self, ros_image, ros_model_poses, model_meshes, model_names):
        '''
        Renders contours of models and modelnames into an image and publishes the result
        ros_image: sensor_msgs.msg.Image
        ros_model_poses: list of geometry_msgs.msg.Pose
        model_meshes: list of open3d.geometry.TriangleMesh, scaled to meters
        model_name: list of str, names of models
        '''
        model_poses = ros_poses_to_np_transforms(ros_model_poses)
        np_img = ros_numpy.numpify(ros_image)
        vis_img = self.create_visualization(np_img, model_poses, model_meshes, model_names)
        vis_img_ros = ros_numpy.msgify(Image, vis_img, encoding='rgb8')
        self.image_pub.publish(vis_img_ros)

    def create_visualization(self, image_scene, model_poses, model_meshes, model_names):
        '''
        Renders contours of models and modelnames into an image
        image_scene: np image, only tested with UINT8 rgb encoding
        model_poses: list of np.arrays of shape (4,4), poses of each model in camera frame in meters
        model_meshes: list of open3d.geometry.TriangleMesh, scaled to meters
        model_names: list of str, names of models
        '''
        output_image = image_scene.copy()
        for pose, mesh, name in zip(model_poses, model_meshes, model_names):
            self.renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA
        
            mesh.transform(pose)
            mesh.paint_uniform_color([1.0, 1.0, 1.0]) 
            
            mtl = o3d.visualization.rendering.MaterialRecord()
            mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA, does not replace the mesh color
            mtl.shader = "defaultUnlit"
            self.renderer.scene.add_geometry(name, mesh, mtl)

            image_model = self.renderer.render_to_image()
            self.renderer.scene.remove_geometry(name)
            image_model = np.asarray(image_model, dtype=np.uint8)
        
            valid_pts = image_model >= 200 # Remove some artifacts from o3d rendering
            image_model[~valid_pts] = 0
        
            contours, hierarchy = cv2.findContours(image_model[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image_scene_with_model = cv2.drawContours(output_image, contours, -1, (0,255,0), 2)
            text_starting_point = (np.min(contours[0][:,:,0]), np.max(contours[0][:,:,1]) + 13)
            output_image = cv2.putText(
                image_scene_with_model, 
                name, 
                text_starting_point, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.45, 
                (0, 0, 0), 
                1, 
                cv2.LINE_AA)
        return output_image