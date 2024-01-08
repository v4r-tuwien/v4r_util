import numpy as np
import cv2
import open3d as o3d
import open3d.visualization.rendering as rendering

class PoseEstimationVisualizer():

    def __init__(self, image_width, image_height, intrinsics_matrix):
        '''
        Renders contours of models and modelnames into an image
        topic: str, topic to publish visualization to
        image_width: int, width of image
        image_height: int, height of image
        intrinsics_matrix: flattened camera matrix
        '''
        self.image_width = image_width
        self.image_height = image_height
        self.renderer = rendering.OffscreenRenderer(image_width, image_height)
        fx = intrinsics_matrix[0]
        fy = intrinsics_matrix[4]
        cx = intrinsics_matrix[2]
        cy = intrinsics_matrix[5]
        self.pinhole = o3d.camera.PinholeCameraIntrinsic(
            image_width, 
            image_height, 
            fx, 
            fy, 
            cx, 
            cy)
        self.renderer.setup_camera(self.pinhole, np.eye(4))

    def create_visualization(self, image_scene, model_poses, model_meshes, model_names):
        '''
        Renders contours of models and modelnames into an image
        image_scene: np image, only tested with UINT8 rgb encoding
        model_poses: list of np.arrays of shape (4,4), poses of each model in camera frame in meters
        model_meshes: list of open3d.geometry.TriangleMesh, scaled to meters
        model_names: list of str, names of models
        '''
        # Maybe can move the pinhole construction and setup_camera here to always do it dynamically
        # instead of checking wheteher the shape of the image is correct
        assert image_scene.shape[0] == self.image_height
        assert image_scene.shape[1] == self.image_width
        
        output_image = image_scene.copy()
        for pose, mesh, name in zip(model_poses, model_meshes, model_names):
            self.renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA
        
            mesh.transform(pose)
            mesh.paint_uniform_color([1.0, 1.0, 1.0]) 
            
            mtl = o3d.visualization.rendering.MaterialRecord()
            mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA, does not replace the mesh color
            mtl.shader = "defaultUnlit"
        
            self.renderer.scene.add_geometry(name, mesh, mtl)
            # need to setup camera EVERYTIME AFTER adding the geometry because setup_camera()
            # uses the models/geometry in the scene to determine the near and far plane, 
            # otherwise objects are clipped if they are further away than 1 meter (default far plane)
            # Unfortunately there is no way to change the near and far plane manually :))
            self.renderer.setup_camera(self.pinhole, np.eye(4))

            image_model = self.renderer.render_to_image()
            self.renderer.scene.remove_geometry(name)
            image_model = np.asarray(image_model, dtype=np.uint8)
        
            valid_pts = image_model >= 200 # Remove some artifacts from o3d rendering
            image_model[~valid_pts] = 0

            if np.sum(image_model) == 0:
                print(f"PoseEstimVis: No pixels were rendered for model {name}")
                continue
            
        
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
    

