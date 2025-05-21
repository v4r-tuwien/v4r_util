"""Module for functions related to depth images and point clouds."""

# Generic python imports
import numpy as np
import open3d as o3d
import copy

# ROS imports
import rospy
import ros_numpy
import open3d_ros_helper.open3d_ros_helper as orh
from sensor_msgs.msg import Image

# V4R imports
from .cmap import get_cmap
import v4r_util.conversions

# ------------------------------------------------------------------------------
# Conversion functions (from v4r_util.conversions)
# ------------------------------------------------------------------------------
def convert_np_depth_img_to_o3d_pcd(depth_img, ros_cam_info, depth_scale=1000, project_valid_depth_only=True):
    """
    Transform a numpy depth image to an Open3D point cloud.
    Same as v4r_util.conversions.convert_np_depth_img_to_o3d_pcd.
    
    Args:
        depth_img (np.ndarray): Depth image
        ros_cam_info (sensor_msgs/CameraInfo): Camera info message containing intrinsic parameters.
        depth_scale (float): Scale factor for depth values (e.g. 1000 for mm to m).
        project_valid_depth_only (bool): Set to False if you want to use pixel indices from a 2D image. Otherwise some pixels might be removed, which means the indices between image and o3d_pcd won't align.
 
    Returns:
        o3d.geometry.PointCloud: Open3D point cloud.
    """
    return v4r_util.conversions.convert_np_depth_img_to_o3d_pcd(depth_img, ros_cam_info, depth_scale, project_valid_depth_only)

def convert_ros_depth_img_to_pcd(ros_depth_img, ros_cam_info, depth_scale=1000, project_valid_depth_only=True):
    """
    Converts a ROS depth image to a point cloud.
    Same as v4r_util.conversions.convert_ros_depth_img_to_pcd.

    Args:
        ros_depth_img (sensor_msgs/Image): ROS depth image message.
        ros_cam_info (sensor_msgs/CameraInfo): ROS camera info message.
        depth_scale (float): Scale factor for depth values (e.g. 1000 for mm to m).
        project_valid_depth_only (bool): Set to False if you want to use pixel indices from a 2D image. Otherwise some pixels might be removed, which means the indices between image and o3d_pcd won't align.
    
    Returns:
        ros_pcd (sensor_msgs/PointCloud2): ROS point cloud message.
        o3d_pcd (open3d.geometry.PointCloud): Open3D point cloud object.
    """
    return v4r_util.conversions.convert_ros_depth_img_to_pcd(ros_depth_img, ros_cam_info, depth_scale, project_valid_depth_only)

def convert_np_label_img_to_np_color_img(np_label_img, np_rgb_img):
    """Same as v4r_util.conversions.convert_np_label_img_to_np_color_img"""
    return  v4r_util.conversions.convert_np_label_img_to_np_color_img(np_label_img, np_rgb_img)

def convert_np_label_img_to_ros_color_img(np_label_img, np_rgb_img):
    """Same as v4r_util.conversions.convert_np_label_img_to_ros_color_img"""
    return  v4r_util.conversions.convert_np_label_img_to_ros_color_img(np_label_img, np_rgb_img)



# ------------------------------------------------------------------------------
# Open3d Point cloud Plane removal/segmentation
# ------------------------------------------------------------------------------
def o3d_segment_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    Segments a plane from a point cloud using Open3Ds RANSAC implementation and returns the plane model and inliers.
    For more information, see: https://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html
    
    Args:
        pcd (open3d.geometry.PointCloud): Input point cloud.
        distance_threshold (float): Distance threshold for RANSAC.
        ransac_n (int): Number of points to sample for the plane model.
        num_iterations (int): Number of iterations for RANSAC.

    Returns:
        plane_model (list): Coefficients of the plane model [a, b, c, d] for the equation ax + by + cz + d = 0.
        inliers (list): Indices of the inliers in the point cloud.
    """
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    return plane_model, inliers

def o3d_remove_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000, visualize=False, invert=True):
    """
    Segments a plane from a point cloud using Open3Ds RANSAC implementation and removes the inliers our outliers from the point cloud.

    Args:
        pcd (open3d.geometry.PointCloud): Input point cloud.
        distance_threshold (float): Distance threshold for RANSAC.
        ransac_n (int): Number of points to sample for the plane model.
        num_iterations (int): Number of iterations for RANSAC.
        visualize (bool): If True, visualize the inliers and outliers using Open3D.
        invert (bool): If True, remove the inliers from the point cloud, if False remove the outliers.

    Returns:
        pcd (open3d.geometry.PointCloud): Point cloud with the inliers (or outliers) removed.
        plane_model (list): Coefficients of the plane model [a, b, c, d] for the equation ax + by + cz + d = 0.
    """
    plane_model, inliers = o3d_segment_plane(pcd, distance_threshold, ransac_n, num_iterations)
    wanted_pcd = pcd.select_by_index(inliers, invert=invert)
    
    if visualize:
        removed_pcd = pcd.select_by_index(inliers, invert=not invert)
        draw_pcds([wanted_pcd, removed_pcd], [np.eye(4), np.eye(4)], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
         
    return wanted_pcd, plane_model

def o3d_get_points_above_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000, nb_neighbours=20, std_ratio=2.0, visualize=False, invert=False):
    """
    Segments a plane from a point cloud using Open3Ds RANSAC implementation and returns the points above or below the plane.
    
    Args:
        pcd (open3d.geometry.PointCloud): Input point cloud.
        distance_threshold (float): Distance threshold for RANSAC.
        ransac_n (int): Number of points to sample for the plane model.
        num_iterations (int): Number of iterations for RANSAC.
        nb_neighbours (int): Number of neighbors which are taken into account in order to calculate the average distance for a given point for the statistical outlier removal.
        std_ratio (float): Standard deviation ratio to use for the statistical outlier removal. The lower this value the more aggressive the filter will be.
        visualize (bool): If True, visualize the inliers and outliers using Open3D.
        invert (bool): If True, return points above the plane, if False return points below the plane.
    
    Returns:
        pcd (open3d.geometry.PointCloud): Point cloud with the points above or below the plane.
    """
    pcd, plane_model = o3d_remove_plane(pcd, distance_threshold, ransac_n, num_iterations, visualize)
    
    pcd_points = np.asarray(pcd.points)
    [a, b, c, d] = plane_model
    signed_distance = a * pcd_points[:, 0] + b * pcd_points[:, 1] + c * pcd_points[:, 2] + d
    above_plane = np.where(signed_distance > 0)[0]

    wanted_pcd =  pcd.select_by_index(above_plane, invert=invert)

    if visualize:
        removed_pcd = pcd.select_by_index(above_plane, invert=not invert)
        draw_pcds([wanted_pcd, removed_pcd], [np.eye(4), np.eye(4)], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    pcd = o3d_remove_outliers(wanted_pcd, nb_neighbors=nb_neighbours, std_ratio=std_ratio)
    return pcd



# ------------------------------------------------------------------------------
# Open3d Point cloud Global Registration
# ------------------------------------------------------------------------------
def preprocess_point_cloud(pcd, voxel_size):
    """
    Preprocess the point cloud by downsampling, estimating normals and extracting FPFH (Fast Point Feature Histogram) features.
    
    Args:
        pcd (open3d.geometry.PointCloud): Input point cloud.
        voxel_size (float): Voxel size for downsampling and feature extraction.
    Returns:
        pcd_down (open3d.geometry.PointCloud): Downsampled point cloud.
        pcd_fpfh (open3d.pipelines.registration.Feature): FPFH features of the downsampled point cloud.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    pcd_down.orient_normals_consistent_tangent_plane(k=100)

    return pcd_down, pcd_fpfh

def global_registration(source, target, voxel_size, max_iterations_icp=30):
    """
    Perform global registration of two point clouds using RANSAC and ICP.
    For more information, see: https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html

    Args:
        source (open3d.geometry.PointCloud): Source point cloud.
        target (open3d.geometry.PointCloud): Target point cloud.
        voxel_size (float): Voxel size for downsampling and feature extraction.
        max_iterations_icp (int): Maximum number of iterations for ICP.
    Returns:
        result (open3d.pipelines.registration.RegistrationResult): Result of the global registration.
        refined_result (open3d.pipelines.registration.RegistrationResult): Result of the refined (loca) registration using ICP.
    """
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.999))

    refined_result = local_registration_icp(source, target, distance_threshold, result.transformation, max_iterations=max_iterations_icp)
    
    return result, refined_result

def local_registration_icp(source, target, distance_threshold, initial_transformation, max_iterations=30):
    """
    Applies ICP (Iterative Closest Point) algorithm to refine the alignment of two point clouds (local registration).

    Args:
        source (open3d.geometry.PointCloud): Source point cloud.
        target (open3d.geometry.PointCloud): Target point cloud.
        distance_threshold (float): Maximum correspondence points-pair distance.
        initial_transformation (np.ndarray): Initial transformation matrix.
        max_iterations (int): Maximum number of iterations for ICP.

    Returns:
        open3d.pipelines.registration.RegistrationResult: Result of the ICP registration.
    """
    return o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations, relative_fitness=1e-7, relative_rmse=1e-7))



# ------------------------------------------------------------------------------
# Open3d Point cloud Filtering and outlier removal
# ------------------------------------------------------------------------------
def filter_pcd(center_point, o3d_pcd, filter_distance):
    """
    Filters the point cloud around a center point with a given threshold.

    Args:
        center_point (np.array): x, y, z coordinates of the current grasping points
        o3d_pcd (open3d.geometry.PointCloud): Pointcloud of the current scene
        filter_distance (float): Filter distance threshold in meter
    
    Returns:
        open3d.geometry.PointCloud: Filtered point cloud
    """
    distances = np.linalg.norm(np.asarray(o3d_pcd.points) - center_point, axis=1)
    within_threshold_mask = distances < filter_distance
    filtered_point_cloud = o3d_pcd.select_by_index(np.where(within_threshold_mask)[0])
    return filtered_point_cloud

def filter_pcd_around_path(path_points, pcd, filter_coeff=3.0):
    """Filters the point cloud around the path points. The point cloud is filtered based on the distance to the path points.

    Args:
        path_points (np.array): The path points. Shape: (n, 3)
        pcd (o3d.geometry.PointCloud): The point cloud
        filter_coeff (float, optional): The filter coefficient. The point cloud is filtered around the middle point of 
            the path points with a distance of filter_coeff * max(distance to path points). Defaults to 3.0.

    Returns:
        o3d.geometry.PointCloud: The filtered pointcloud
    """
    if path_points.ndim != 2:
        if len(path_points) == 3:
            rospy.logwarn("Only as single path point has been passed. Pointcloud will be filtered around it as a center with a distance of 0.5m.")
            return filter_pcd(path_points, pcd, 1)
        elif len(path_points) == 0:
            rospy.logwarn("Empty path passed. Will return an unfiltered pointcloud.")
            return pcd

    # Get middle point of the path and filter pcd around it
    min_x = np.min(path_points[:, 0])
    max_x = np.max(path_points[:, 0])
    min_y = np.min(path_points[:, 1])
    max_y = np.max(path_points[:, 1])
    min_z = np.min(path_points[:, 2])
    max_z = np.max(path_points[:, 2])
    center_point_head_camera = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2])
    filter_distance = filter_coeff * np.max([(max_x - min_x) / 2, (max_y - min_y) / 2, (max_z - min_z) / 2])

    distances = np.linalg.norm(np.asarray(pcd.points) - center_point_head_camera, axis=1)
    within_threshold_mask = distances < filter_distance
    pcd = pcd.select_by_index(np.where(within_threshold_mask)[0])

    # Downsample and estimate normals
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=10)

    return pcd

def o3d_remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    Removes outliers from a point cloud using Open3D's statistical outlier removal method. 
    For more information, see: https://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html
    
    Args:
        pcd (open3d.geometry.PointCloud): Input point cloud.
        nb_neighbors (int): Number of neighbors which are taken into account in order to calculate the average distance for a given point.
        std_ratio (float): Standard deviation ratio to use for the statistical outlier removal. The lower this value the more aggressive the filter will be.
    
    Returns:
        pcd (open3d.geometry.PointCloud): Point cloud with outliers removed.
    """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd = pcd.select_by_index(ind)
    return pcd



# ------------------------------------------------------------------------------
# Open3d Point cloud visualization
# ------------------------------------------------------------------------------

def draw_pcds(pcds, transformations, colors, window_name="Open3D"):
    """
    Visualizes multiple point clouds with transformations applied to each point cloud.

    Args:
        pcds (open3d.geometry.PointCloud[]): List of open3d.geometry.PointCloud objects.
        transformations (np.ndarray[]): List of transformation matrices to apply to each point cloud.
        colors (list): List of colors for each point cloud (RGB values).
    """
    pcds_temp = []
    for i, pcd in enumerate(pcds):
        pcd_temp = copy.deepcopy(pcd)
        pcd_temp.paint_uniform_color(colors[i])
        pcd_temp.transform(transformations[i])
        pcds_temp.append(pcd_temp)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    pcds_temp.append(coord_frame)

    o3d.visualization.draw_geometries(pcds_temp, window_name=window_name)


def draw_registration_result(source, target, transformation=np.eye(4), color_source=[1, 0.706, 0], color_target=[0, 0.651, 0.929], window_name="Open3D"):
    """
    Visualizes two point clouds with a transformation applied to the source point cloud. This is useful to visualize the result of a registration.

    Args:
        source (open3d.geometry.PointCloud): Source point cloud.
        target (open3d.geometry.PointCloud): Target point cloud.
        transformation (np.ndarray): Transformation matrix to apply to the source point cloud.
        color_source (list): Color for the source point cloud (RGB values).
        color_target (list): Color for the target point cloud (RGB values).
    """
    draw_pcds([source, target], [transformation, np.eye(4)], [color_source, color_target], window_name=window_name)