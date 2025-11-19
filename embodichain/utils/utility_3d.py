# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import numpy as np
import cv2
from typing import List, Dict, Union
import open3d as o3d
import hashlib
import os
import ffmpeg
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import copy


def dist_points_to_plane(point, plane_params):
    plane_normal = np.array(plane_params[:3]).reshape(-1)
    plane_center = np.array([0, 0, -plane_params[-1] / plane_params[2]]).reshape(-1)
    distance = np.dot(plane_normal, (point - plane_center))
    return np.abs(distance)


def pose_shift(pose_in_cam: np.ndarray, axis: int, shift: float) -> np.ndarray:
    shift_pose = np.copy(pose_in_cam)
    shift_pose = np.linalg.inv(shift_pose)
    shift_pose[axis, -1] += shift
    shift_pose = np.linalg.inv(shift_pose)
    return shift_pose


def align_pose_with_z_down(pose: np.ndarray) -> np.ndarray:
    """align pose with z down

    Args:
        pose (np.ndarray): pose in camera coordinate

    Returns:
        np.ndarray: pose in camera
    """
    negative_z = np.array([0, 0, -1])
    align_pose = np.eye(4, dtype=float)
    pose_z = pose[:3, 2]
    rota_axis = np.cross(pose_z, negative_z)
    temp_norm = np.linalg.norm(rota_axis)
    if temp_norm < 1e-5:
        return pose

    rota_axis = rota_axis / temp_norm
    rota_angle = np.arccos(pose_z.dot(negative_z))
    align_r = R.from_rotvec(rota_axis * rota_angle).as_matrix()
    align_pose[:3, :3] = align_r @ pose[:3, :3]
    align_pose[:3, 3] = pose[:3, 3]
    return align_pose


class Sequence3DVisSaver:
    def __init__(self, cam_dict: Dict, save_dir: str):
        self.save_dir = save_dir
        self.cam_dict = cam_dict
        os.makedirs(save_dir, exist_ok=True)
        # vis_pose = np.eye(4)
        # vis_pose[:3, :3] = (
        #     R.from_rotvec(np.array([0, np.pi / 4 * 3, 0])).as_matrix()
        #     @ R.from_rotvec(np.array([np.pi / 4, 0, 0])).as_matrix()
        # )

    def visualize(
        self,
        pcds: List[np.ndarray],
        background_pcd: o3d.geometry.PointCloud = None,
        **kwargs,
    ):
        # TODO: set view control
        # if len(kwargs.keys()) > 0:
        #     for key, value in kwargs.items():
        #         assert len(pcds) == len(value)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        ctr = vis.get_view_control()

        if background_pcd is not None:
            vis.add_geometry(background_pcd)

        # vis.add_geometry(pcd)
        for i, pcd in tqdm(enumerate(pcds)):
            if isinstance(pcd, o3d.geometry.PointCloud):
                pass
            elif isinstance(pcd, np.ndarray):
                pcd_ = o3d.geometry.PointCloud()
                pcd_.points = o3d.utility.Vector3dVector(pcd[:, :3])
                if pcd.shape[-1] == 6:
                    pcd_.colors = o3d.utility.Vector3dVector(
                        pcd[:, 3:6].astype(np.float32)
                    )
                pcd = pcd_
            # pcd.transform(vis_pose)
            vis.add_geometry(pcd)
            if len(kwargs.keys()) > 0:
                pcd_n = {}
                for key, value in kwargs.items():
                    if isinstance(value[i], np.ndarray):
                        pcd_n[key] = o3d.geometry.PointCloud()
                        pcd_n[key].points = o3d.utility.Vector3dVector(value[i][:, :3])
                        if value[i].shape[-1] == 6:
                            pcd_n[key].colors = o3d.utility.Vector3dVector(
                                value[i][:, 3:6].astype(np.float32)
                            )
                        vis.add_geometry(pcd_n[key])
                    elif isinstance(value[i], o3d.geometry.PointCloud):
                        pcd_n[key] = value[i]
                        vis.add_geometry(value[i])
                    else:
                        raise TypeError(f"Unsupported geometry type: {type(value[i])}")

            extrinsic = np.eye(4)

            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            fx, fy = (
                self.cam_dict["rect_cam_k"][0, 0],
                self.cam_dict["rect_cam_k"][1, 1],
            )
            cx, cy = (
                self.cam_dict["rect_cam_k"][0, -1],
                self.cam_dict["rect_cam_k"][1, -1],
            )
            intrinsic.set_intrinsics(int(cx * 2), int(cy * 2), fx, fy, cx, cy)

            # Set the camera parameters
            camera_params = o3d.camera.PinholeCameraParameters()
            camera_params.intrinsic = intrinsic
            camera_params.extrinsic = extrinsic
            ctr.convert_from_pinhole_camera_parameters(camera_params, True)

            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(os.path.join(self.save_dir, "temp_%04d.jpg" % i))
            vis.remove_geometry(pcd)
            if len(kwargs.keys()) > 0:
                for key, value in kwargs.items():
                    vis.remove_geometry(pcd_n[key])

        if background_pcd is not None:
            vis.remove_geometry(background_pcd)

        (
            ffmpeg.input(
                os.path.join(self.save_dir, "*.jpg"), pattern_type="glob", framerate=10
            )
            .output(os.path.join(self.save_dir, "video.mp4"))
            .run()
        )


class TableExtractor:
    def __init__(
        self,
        points: o3d.geometry.PointCloud,
    ):
        self.points = points
        self.plane = None

    def generate_table_frame(self, voxel_size: float = 1e-3) -> np.ndarray:
        pcd = self.points.voxel_down_sample(voxel_size)
        plane_params, inliers = pcd.segment_plane(5 * voxel_size, 10, 100)
        self.plane = pcd.select_by_index(inliers)
        z = dist_points_to_plane(np.array([0, 0, 0]), plane_params)
        z_direction = plane_params[:3]
        if z_direction[2] > 0:
            z_direction = -z_direction
        pose = np.eye(4)
        z_direction = z_direction / np.linalg.norm(z_direction)
        y_direction = np.cross(z_direction, np.array([1, 0, 0]))
        y_direction = y_direction / np.linalg.norm(y_direction)
        x_direction = np.cross(y_direction, z_direction)
        x_direction = x_direction / np.linalg.norm(x_direction)
        R = np.c_[x_direction, y_direction, z_direction]
        pose[:3, :3] = R
        pose[2, 3] = z
        return pose


def rotate_pose_by_mirror(
    pose: np.ndarray, mirror_axis: int = 1, target_axis: int = 2
) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    from copy import deepcopy

    res_axis = list(set(list([0, 1, 2])).difference(set([mirror_axis])))
    pose_rz_0 = pose[res_axis[0], target_axis]
    pose_rz_2 = pose[res_axis[1], target_axis]
    rota_y = np.arctan2(-pose_rz_0, pose_rz_2)

    # construct global rotation
    r_axis = np.array([0, 0, 0])  # rotate around y axis
    r_axis[mirror_axis] = 1
    global_r = R.from_rotvec(2 * rota_y * r_axis).as_matrix()

    pose_flip = deepcopy(pose)
    pose_flip[:3, :3] = global_r @ pose[:3, :3]
    return pose_flip


def get_mesh_md5(mesh: o3d.t.geometry.TriangleMesh) -> str:
    """get mesh md5 unique key

    Args:
        mesh (o3d.geometry.TriangleMesh): mesh

    Returns:
        str: mesh md5值
    """
    vert = np.array(mesh.vertex.positions.numpy(), dtype=float)
    face = np.array(mesh.triangle.indices.numpy(), dtype=float)
    mix = np.vstack([vert, face])
    return hashlib.md5(np.array2string(mix).encode()).hexdigest()


def get_inv_intrinsic(intrinsic):
    return np.array(
        [
            [1 / intrinsic[0, 0], 0, -intrinsic[0, 2] / intrinsic[0, 0]],
            [0, 1 / intrinsic[1, 1], -intrinsic[1, 2] / intrinsic[1, 1]],
            [0, 0, 1],
        ]
    )


def depth_rgb_to_o3d(depth_map, rgb, intrinsic):
    resolution_x, resolution_y = depth_map.shape
    real_depth = depth_map.reshape(resolution_x * resolution_y)
    if len(rgb.shape) == 3:  # color
        rgb_list = rgb.reshape(resolution_x * resolution_y, 3) / 255.0
    elif len(rgb.shape) == 2:  # gray, repeat to rgb image
        rgb_copy = np.repeat(rgb[:, :, None], 3, axis=2)
        rgb_list = rgb_copy.reshape(resolution_x * resolution_y, 3) / 255.0
    pixel_index = np.arange(resolution_x * resolution_y)
    pixel_index_x = pixel_index % resolution_y
    pixel_index_y = pixel_index // resolution_y
    inv_intrinsic = get_inv_intrinsic(intrinsic)
    pixel_homo_t = np.stack(
        [pixel_index_x, pixel_index_y, np.ones(resolution_x * resolution_y)]
    )
    camera_homo_t = inv_intrinsic @ pixel_homo_t
    camera_pc_t = camera_homo_t * real_depth
    camera_pc_h_t = np.ones(
        shape=(camera_pc_t.shape[0] + 1, camera_pc_t.shape[1]), dtype=np.float32
    )
    camera_pc_h_t[:3, :] = camera_pc_t

    camera_pc_o3d = o3d.geometry.PointCloud()
    camera_pc_o3d.points = o3d.utility.Vector3dVector(camera_pc_t.T)
    camera_pc_o3d.colors = o3d.utility.Vector3dVector(rgb_list)
    return camera_pc_o3d


def depth_to_pc(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    color: np.ndarray = None,
    mask: np.ndarray = None,
    organized=True,
    scale: float = 1.0,
) -> np.ndarray:
    height, width = depth.shape
    depth = cv2.resize(
        depth, (int(width * scale), int(height * scale)), cv2.INTER_NEAREST
    )
    if color is not None:
        color = cv2.resize(
            color, (int(width * scale), int(height * scale)), cv2.INTER_NEAREST
        )
    if mask is not None:

        mask = cv2.resize(
            mask.astype(np.uint8),
            (int(width * scale), int(height * scale)),
            cv2.INTER_NEAREST,
        )

    height, width = depth.shape
    xmap = np.arange(width)
    ymap = np.arange(height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    fx = intrinsic[0, 0] * scale
    fy = intrinsic[1, 1] * scale
    cx = intrinsic[0, 2] * scale
    cy = intrinsic[1, 2] * scale
    points_z = depth
    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)

    if not organized:
        cloud = cloud.reshape([-1, 3])

    if color is None:
        return cloud
    else:
        if color.shape[-1] == 3:
            pass
        else:
            color = cv2.cvtColor(color, cv2.COLOR_GRAY2RGB)

        if not organized:
            color = color.reshape([-1, 3])

        cloud = np.concatenate([cloud, color], -1)

    if mask is not None:
        assert len(mask.shape), "{}".format(mask.shape)

        if not organized:
            cloud = cloud[mask.reshape(-1).astype(np.bool_), :]
        else:
            cloud = np.multiply(np.expand_dims(mask, -1), cloud)

    return cloud


def remove_plane(
    pc: np.ndarray, pc_num: int = 2048, remove_far: float = 10.0
) -> np.ndarray:
    assert pc.shape[-1] == 3

    import open3d as o3d

    pc = pc[pc[:, -1] <= remove_far, :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd = pcd.voxel_down_sample(1e-3)
    _, indices = pcd.segment_plane(5e-3, 10, 100)
    indices_preserve = np.array(
        list(
            set(np.arange(np.array(pcd.points).shape[0])).difference(set(list(indices)))
        )
    )
    replace = indices_preserve.shape[0] < pc_num
    indices_preserve = indices_preserve[
        np.random.choice(indices_preserve.shape[0], pc_num, replace=replace)
    ]
    pc = np.array(pcd.select_by_index(list(indices_preserve)).points).astype(np.float32)
    return pc


def keep_n_points(pc: np.ndarray, pc_num: int = 2048):
    n = pc.shape[0]
    replace = n < pc_num
    idx = np.arange(n)
    indices_preserve = idx[np.random.choice(n, pc_num, replace=replace)]
    pc = pc[indices_preserve]
    return pc


def generate_depth_from_disp(
    disp: np.ndarray, fx: float, baseline: float
) -> np.ndarray:
    depth_img = np.zeros_like(disp)
    depth_img[disp > 0] = fx * baseline / disp[disp > 0]
    return depth_img


def load_pose_from_txt(file_path: str) -> np.ndarray:
    with open(file_path, "r") as f:
        lines = f.readlines()
    pose = np.array([list(map(float, line.split())) for line in lines])
    return pose


def generate_pcd_from_disp(
    left_image,
    disp,
    fx: float = 0.0,
    fy: float = 0.0,
    cx: float = 0.0,
    cy: float = 0.0,
    baseline: float = 0.0,
    cam_params: dict = None,
) -> o3d.geometry.PointCloud:
    if cam_params is not None:
        fx = cam_params["rect_cam_k"][0, 0]
        fy = cam_params["rect_cam_k"][1, 1]
        cx = cam_params["rect_cam_k"][0, -1]
        cy = cam_params["rect_cam_k"][1, -1]
        baseline = cam_params["baseline"]

    assert fx != 0.0
    assert fy != 0.0
    assert cx != 0.0
    assert cy != 0.0
    assert baseline != 0.0

    img_h, img_w, _ = left_image.shape
    depth_img = generate_depth_from_disp(disp, fx, baseline)
    rgb_img_o3d = o3d.geometry.Image(left_image.astype(np.uint8))
    depth_img_o3d = o3d.geometry.Image(depth_img.astype(np.float32))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(img_w, img_h, fx, fy, cx, cy)
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_img_o3d,
        depth_img_o3d,
        depth_scale=1.0,
        depth_trunc=2.0,  # 1.6m humanoid
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic)
    return pcd


def gen_disp_colormap(inputs, normalize=True, torch_transpose=True):
    import matplotlib.pyplot as plt
    import torch

    _DEPTH_COLORMAP = plt.get_cmap("plasma", 256)  # for plotting
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis


def triangulation(xL, xR, R, T, left_intrinsic, d_left, right_intrinsic, d_right):
    xt = cv2.undistortPoints(xL.astype(np.float32), left_intrinsic, d_left).reshape(
        [-1, 2]
    )
    xtt = cv2.undistortPoints(xR.astype(np.float32), right_intrinsic, d_right).reshape(
        [-1, 2]
    )
    # xt = xL
    # xtt = xR
    xt = np.hstack((xt, np.ones_like(xt)[:, 0:1])).T
    xtt = np.hstack((xtt, np.ones_like(xtt)[:, 0:1])).T
    u = np.matrix(R) * np.matrix(xt)
    n_xt2 = np.sum(np.power(xt, 2), 0)
    n_xtt2 = np.sum(np.power(xtt, 2), 0)
    DD = np.multiply(n_xt2, n_xtt2) - np.power(np.sum(np.multiply(u, xtt), 0), 2)
    dot_uT = np.sum(np.multiply(u, T), 0)
    dot_xttT = np.sum(np.multiply(xtt, T), 0)
    dot_xttu = np.sum(np.multiply(u, xtt), 0)
    NN1 = np.multiply(dot_xttu, dot_xttT) - np.multiply(n_xtt2, dot_uT)
    NN2 = np.multiply(n_xt2, dot_xttT) - np.multiply(dot_uT, dot_xttu)
    Zt = np.divide(NN1, DD)
    Ztt = np.divide(NN2, DD)
    X1 = np.multiply(xt, Zt)
    X2 = R.T * (np.multiply(xtt, Ztt) - T)
    XL = X1
    XR = R * XL + T
    error = np.mean(np.sqrt(np.sum(np.power(X1 - X2, 2), 0)))
    return XL.T, XR.T, error


def visualize_colored_objects(
    obj_pcds_for_vis: List[List[o3d.geometry.PointCloud]],
    background_pcd_path: str,
    save_dir: str,
    cam_dict: Dict,
    object_indices: List[int],
):
    """Visualize sequences of object point clouds.
    Args:
        obj_pcds_for_vis (List[List[o3d.geometry.PointCloud]]): Sequence of point clouds for each object.
        background_pcd_path (str): Path to the background point cloud file.
        save_dir (str): Directory to save visualization results.
        cam_dict (Dict): Camera parameter dictionary.
        object_indices (List[int]): List of object indices to visualize.
    """
    pcds_for_vis_o3d = np.asarray(obj_pcds_for_vis)

    max_index = pcds_for_vis_o3d.shape[1] - 1
    for idx in object_indices:
        if idx < 0 or idx > max_index:
            raise ValueError(
                f"Invalid object index {idx}. Valid range: 0 to {max_index}"
            )

    background_pcd = o3d.io.read_point_cloud(background_pcd_path)
    if len(background_pcd.points) == 0:
        raise ValueError(
            f"Failed to load background point cloud from {background_pcd_path}"
        )

    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ]
    if len(object_indices) > len(colors):
        import random

        colors.extend(
            [
                [random.random(), random.random(), random.random()]
                for _ in range(len(object_indices) - len(colors))
            ]
        )

    object_points = {}
    for idx, obj_idx in enumerate(object_indices):
        obj_points = []
        for pcd in pcds_for_vis_o3d[:, obj_idx]:
            points = np.asarray(pcd.points)
            colors_array = np.tile(colors[idx], (points.shape[0], 1))
            combined = np.concatenate([points, colors_array], axis=-1)
            obj_points.append(combined)
        object_points[f"obj{obj_idx}_points"] = obj_points

    vis = Sequence3DVisSaver(cam_dict, save_dir=save_dir)

    vis.visualize(
        object_points[f"obj{object_indices[0]}_points"],  # 第一个物体点云
        background_pcd=background_pcd,
        **{
            f"obj{idx}_points": object_points[f"obj{idx}_points"]  #  其他物体点云
            for idx in object_indices[1:]
        },
    )


def _clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    # Remove non-finite vertices (NaN/Inf)
    V = np.asarray(mesh.vertices)
    good = np.isfinite(V).all(axis=1)
    if not np.all(good):
        mesh = mesh.select_by_index(np.where(good)[0])

    # Topology cleanup
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    return mesh


def _robust_bbox_from_mesh(
    mesh: o3d.geometry.TriangleMesh,
    n_pts: int = 15000,
    nb_neighbors: int = 40,
    std_ratio: float = 2.0,
) -> o3d.geometry.OrientedBoundingBox:
    """
    Sample points on the surface, remove outliers, then compute OBB.
    This avoids a few stray vertices from blowing up the bbox.
    """
    # Sample
    if len(mesh.triangles) == 0 or len(mesh.vertices) < 10:
        # Fallback to vertex cloud if mesh is trivial
        pcd = o3d.geometry.PointCloud(mesh.vertices)
    else:
        pcd = mesh.sample_points_uniformly(number_of_points=n_pts)

    # Remove statistical outliers
    pcd, idx = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )

    # Open3D >= 0.17 has robust OBB for meshes: mesh.get_oriented_bounding_box(robust=True, outlier_ratio=...)
    # But building from the cleaned point cloud works across versions:
    obb = pcd.get_oriented_bounding_box()
    return obb


def compute_scale_and_rotation_to_match(
    ref_path: str, target_path: str, visualize: bool = False
):
    """Compute per-axis scale and rotation matrix to make target mesh similar
    in size and orientation to the reference mesh, and visualize reference,
    unaligned target, and aligned target meshes with bounding boxes.
    """

    # ---- Load meshes ----
    ref_mesh = o3d.io.read_triangle_mesh(ref_path)
    tgt_mesh = o3d.io.read_triangle_mesh(target_path)

    # ---- Clean meshes ----
    ref_mesh = _clean_mesh(ref_mesh)
    tgt_mesh = _clean_mesh(tgt_mesh)

    # ---- ICP alignment (point-to-point) ----
    pcd_ref = ref_mesh.sample_points_uniformly(5000)
    pcd_tgt = tgt_mesh.sample_points_uniformly(5000)
    reg = o3d.pipelines.registration.registration_icp(
        pcd_tgt,
        pcd_ref,
        max_correspondence_distance=0.05,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    T_icp = reg.transformation

    # ---- Apply alignment on a copy ----
    tgt_mesh_aligned = copy.deepcopy(tgt_mesh)
    tgt_mesh_aligned.transform(T_icp)

    # ---- Robust OBBs ----
    ref_obb = _robust_bbox_from_mesh(ref_mesh)
    tgt_obb_unaligned = _robust_bbox_from_mesh(tgt_mesh)
    tgt_obb_aligned = _robust_bbox_from_mesh(tgt_mesh_aligned)

    ref_extent = np.array(ref_obb.extent)
    tgt_extent = np.array(tgt_obb_aligned.extent)
    tgt_extent[tgt_extent == 0] = 1e-9
    scale_vec = ref_extent / tgt_extent
    scale_mean = float(np.mean(scale_vec))

    # ---- Visualization ----
    if visualize:
        # Paint meshes
        ref_mesh.paint_uniform_color([1, 0, 0])  # red → reference
        tgt_mesh.paint_uniform_color([0, 0, 1])  # blue → unaligned target
        tgt_mesh_aligned.paint_uniform_color([0, 1, 0])  # green → aligned target

        # Paint bounding boxes
        ref_obb.color = (1, 0, 0)  # red
        tgt_obb_unaligned.color = (0, 0, 1)  # blue
        tgt_obb_aligned.color = (0, 1, 0)  # green

        # Apply small offsets for side-by-side viewing
        offset_unaligned = np.eye(4)
        offset_unaligned[0, 3] = -0.1
        offset_aligned = np.eye(4)
        offset_aligned[0, 3] = 0.1

        tgt_mesh_unaligned_vis = copy.deepcopy(tgt_mesh).transform(offset_unaligned)
        tgt_obb_unaligned_vis = copy.deepcopy(tgt_obb_unaligned).translate(
            offset_unaligned[:3, 3]
        )

        tgt_mesh_aligned_vis = copy.deepcopy(tgt_mesh_aligned).transform(offset_aligned)
        tgt_obb_aligned_vis = copy.deepcopy(tgt_obb_aligned).translate(
            offset_aligned[:3, 3]
        )

        # Visualize all together
        o3d.visualization.draw_geometries(
            [
                ref_mesh,
                ref_obb,
                tgt_mesh_unaligned_vis,
                tgt_obb_unaligned_vis,
                tgt_mesh_aligned_vis,
                tgt_obb_aligned_vis,
            ]
        )

    return scale_vec, scale_mean, T_icp
