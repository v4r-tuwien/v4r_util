import numpy as np
import ros_numpy
import open3d as o3d
import open3d_ros_helper.open3d_ros_helper as orh
from .cmap import get_cmap
from sensor_msgs.msg import Image

def convert_np_depth_img_to_o3d_pcd(depth_img, ros_cam_info, depth_scale=1000, project_valid_depth_only=True):
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
        '''
        Set project_valid_depth_only to False if you want to use pixel indices from a 2D image. Otherwise some pixels
        might be removed, which means the indices between image and o3d_pcd won't align.
        '''
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
    return ros_numpy.msgify(Image, colors, encoding='rgb8')
