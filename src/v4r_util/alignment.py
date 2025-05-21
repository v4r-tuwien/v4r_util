"""Module for aligning transformations and bounding boxes to coordinate axes."""

# Generic python imports
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
from enum import IntEnum

# ROS imports
import rospy
from geometry_msgs.msg import Quaternion


class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2    

def get_best_aligning_axis(coordinate_axis_vec, reference_frame_axes):
    """Given a vector in 3D space, this function finds the axis of the 
    reference frame that it aligns with the best.

    Args:
        coordinate_axis_vec (array like): The vector in 3D space to be aligned.
        reference_frame_axes (array like): The reference frame axes to align with.

    Returns:
        tuple: The axis of the reference frame that aligns best with the input vector,
               and a boolean indicating whether the vector should be inverted.
    """
    x_vec = reference_frame_axes[Axis.X]
    y_vec = reference_frame_axes[Axis.Y]
    z_vec = reference_frame_axes[Axis.Z]
    
    x_align = np.abs(np.dot(coordinate_axis_vec, x_vec))
    y_align = np.abs(np.dot(coordinate_axis_vec, y_vec))
    z_align = np.abs(np.dot(coordinate_axis_vec, z_vec))

    if x_align >= y_align and x_align >= z_align:
        return Axis.X, np.dot(coordinate_axis_vec, x_vec) < 0
    elif y_align >= x_align and y_align >= z_align:
        return Axis.Y, np.dot(coordinate_axis_vec, y_vec) < 0
    else:
        return Axis.Z, np.dot(coordinate_axis_vec, z_vec) < 0

def get_best_aligning_axis_to_world_frame(coordinate_axis_vec):
    """Given a vector in 3D space, this function finds the axis of the
    world frame that it aligns with the best.

    Args:
        coordinate_axis_vec (array like): The vector in 3D space to be aligned.

    Returns:
        tuple: The axis of the world frame that aligns best with the input vector,
               and a boolean indicating whether the vector should be inverted.
    """
    reference_vecs = [[], [], []]
    reference_vecs[Axis.X] = np.array([1, 0, 0])
    reference_vecs[Axis.Y] = np.array([0, 1, 0])
    reference_vecs[Axis.Z] = np.array([0, 0, 1])

    return get_best_aligning_axis(coordinate_axis_vec, reference_vecs)

    
def align_transformation(transform):
    """Aligns transformation to coordinate axes. This means that the x-axis will 
    be the one that aligns the best with the x-axis of the coordinate frame, and so on.

    Args:
        transform (np.array): Homogenous transformation matrix (4x4) to be aligned.

    Raises:
        ValueError: Rotation matrix has negative determinant
        ValueError: Could not find consistent alignment

    Returns:
        np.array: Aligned homogenous transformation matrix (4x4).
    """
    center = transform[:3, 3] 
    pose_x_vec = center - (transform @ np.array([1, 0, 0, 1]))[:3]
    pose_y_vec = center - (transform @ np.array([0, 1, 0, 1]))[:3]
    pose_z_vec = center - (transform @ np.array([0, 0, 1, 1]))[:3]
    new_rot_mat = np.eye(3)
    axes_aligned = [False, False, False]
    
    for coordinate_axis_vec in [pose_x_vec, pose_y_vec, pose_z_vec]:
        axis, should_invert = get_best_aligning_axis_to_world_frame(coordinate_axis_vec)
        if should_invert:
            coordinate_axis_vec *= -1
        new_rot_mat[:, axis] = coordinate_axis_vec
        axes_aligned[axis] = True

    # check handedness, should theoretically never happen becase reference frame should be right-handed
    if np.linalg.det(new_rot_mat) < 0:
        raise ValueError("Rotation matrix has negative determinant")
        
    if not axes_aligned[Axis.X] and not axes_aligned[Axis.Y] and not axes_aligned[Axis.Z]:
        raise ValueError("Could not find consistent alignment")
    
    aligned_transform = np.eye(4)
    aligned_transform[:3, :3] = new_rot_mat
    aligned_transform[:3, 3] = center

    return aligned_transform

def align_pose_rotation(pose):
    """Aligns the rotation of a pose to coordinate axes. This means that the x-axis will
    be the one that aligns the best with the x-axis of the coordinate frame, and so on.

    Args:
        pose (geometry_msgs/Pose): Pose to be aligned.

    Returns:
        geometry_msgs/Pose: Aligned pose.
    """
    # ros pose, not pose stamped
    center = [pose.position.x, pose.position.y, pose.position.z]
    transform = np.eye(4)
    transform[:3, :3] = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]).as_matrix()
    transform[:3, 3] = np.array(center)

    aligned_transform = align_transformation(transform)
    new_rot_mat = aligned_transform[:3, :3]

    quat = R.from_matrix(new_rot_mat).as_quat()
    ret_pose = copy.deepcopy(pose)
    ret_pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
    return ret_pose

def align_bounding_box_rotation(bb):
    """Aligns bounding box rotation to coordinate axes. This means that the BB x-axis will be the one that aligns the best 
    with the x-axis of the coordinate frame, and so on.

    Args:
        bb (o3d/OrientedBoundingBox): Bounding box to be aligned.

    Raises:
        ValueError: Rotation matrix has negative determinant
        ValueError: Could not find consistent alignment

    Returns:
        o3d/OrientedBoundingBox: Aligned bounding box.
    """
    transform = np.eye(4)
    transform[:3, :3] = bb.R
    transform[:3, 3] = bb.center
    
    bb_x_vec = bb.center - (transform @ np.array([1, 0, 0, 1]))[:3]
    bb_y_vec = bb.center - (transform @ np.array([0, 1, 0, 1]))[:3]
    bb_z_vec = bb.center - (transform @ np.array([0, 0, 1, 1]))[:3]
    new_rot_mat = np.eye(3)
    new_extent = np.zeros(3)
    axes_aligned = [False, False, False]
    
    for coordinate_axis_vec, extent in zip([bb_x_vec, bb_y_vec, bb_z_vec], [bb.extent[0], bb.extent[1], bb.extent[2]]):
        axis, should_invert = get_best_aligning_axis_to_world_frame(coordinate_axis_vec)
        if should_invert:
            coordinate_axis_vec *= -1
        new_rot_mat[:, axis] = coordinate_axis_vec
        axes_aligned[axis] = True
        new_extent[axis] = extent 

        
    # check handedness, should theoretically never happen becase reference frame should be right-handed
    if np.linalg.det(new_rot_mat) < 0:
        raise ValueError("Rotation matrix has negative determinant")
        
    if not axes_aligned[Axis.X] and not axes_aligned[Axis.Y] and not axes_aligned[Axis.Z]:
        raise ValueError("Could not find consistent alignment")
    
    rotated_bb = copy.deepcopy(bb)
    rotated_bb.R = new_rot_mat
    rotated_bb.extent = new_extent
    return rotated_bb