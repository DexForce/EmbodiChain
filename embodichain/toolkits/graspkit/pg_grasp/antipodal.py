# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import time
import pathlib
import pickle
import os

from enum import Enum
from copy import deepcopy
from typing import List

from .cone_sampler import ConeSampler
from embodichain.utils.utility import get_mesh_md5
from embodichain.utils import logger


class GraspSelectMethod(Enum):
    NORMAL_SCORE = 0
    NEAR_APPROACH = 1
    CENTER = 2


class AntipodalGrasp:
    def __init__(self, pose: np.ndarray, open_len: float, score: float) -> None:
        self.pose = pose  # [4, 4] of float grasp pose
        self.open_len = open_len  # gripper open length
        self.score = score  # grasp pose score

    def grasp_pose_visual_mesh(self, gripper_open_length: float = None):
        if gripper_open_length is None:
            gripper_open_length = self.open_len
            open_ratio = 1.0
        else:
            open_ratio = self.open_len / gripper_open_length
        open_ratio = max(1e-4, open_ratio)
        gripper_finger = o3d.geometry.TriangleMesh.create_box(
            gripper_open_length * 0.04,
            gripper_open_length * 0.04,
            gripper_open_length * 0.5,
        )
        gripper_finger.translate(
            np.array(
                [
                    -gripper_open_length * 0.02,
                    -gripper_open_length * 0.02,
                    -gripper_open_length * 0.25,
                ]
            )
        )
        gripper_left = deepcopy(gripper_finger)
        gripper_left = gripper_left.translate(
            np.array(
                [
                    -gripper_open_length * open_ratio * 0.5,
                    0,
                    -gripper_open_length * 0.25,
                ]
            )
        )

        gripper_right = deepcopy(gripper_finger)
        gripper_right = gripper_right.translate(
            np.array(
                [gripper_open_length * open_ratio * 0.5, 0, -gripper_open_length * 0.25]
            )
        )

        gripper_root1 = o3d.geometry.TriangleMesh.create_box(
            gripper_open_length * open_ratio,
            gripper_open_length * 0.04,
            gripper_open_length * 0.04,
        )
        gripper_root1.translate(
            np.array(
                [
                    gripper_open_length * open_ratio * -0.5,
                    gripper_open_length * -0.02,
                    gripper_open_length * -0.02,
                ]
            )
        )
        gripper_root1.translate(
            np.array(
                [
                    0,
                    0,
                    gripper_open_length * -0.5,
                ]
            )
        )

        gripper_root2 = o3d.geometry.TriangleMesh.create_box(
            gripper_open_length * 0.04,
            gripper_open_length * 0.04,
            gripper_open_length * 0.5,
        )
        gripper_root2.translate(
            np.array(
                [
                    gripper_open_length * -0.02,
                    gripper_open_length * -0.02,
                    gripper_open_length * -0.25,
                ]
            )
        )
        gripper_root2.translate(
            np.array(
                [
                    0,
                    0,
                    gripper_open_length * -0.75,
                ]
            )
        )

        gripper_visual = gripper_left + gripper_right + gripper_root1 + gripper_root2
        gripper_visual.compute_vertex_normals()
        return gripper_visual


class Antipodal:
    def __init__(
        self,
        point_a: np.ndarray,
        point_b: np.ndarray,
        normal_a: np.ndarray,
        normal_b: np.ndarray,
    ) -> None:
        """antipodal contact pair

        Args:
            point_a (np.ndarray): position in point a
            point_b (np.ndarray): position in point b
            normal_a (np.ndarray): normal in point a
            normal_b (np.ndarray): normal in point b
        """
        self.point_a = point_a
        self.point_b = point_b
        self.normal_a = normal_a
        self.normal_b = normal_b
        self.dis = np.linalg.norm(point_a - point_b)
        self.angle_cos = self.normal_a.dot(-self.normal_b)
        self.score = self._get_score()
        self._canonical_pose = self._get_canonical_pose()

    def _get_score(self):
        # TODO:  only normal angle is taken into account
        return self.angle_cos

    def get_dis(self, another) -> float:
        """get distance acoording to another antipodal

        Args:
            other (Antipodal): another antipodal

        Returns:
            float: distance
        """
        aa_dis = np.linalg.norm(self.point_a - another.point_a)
        bb_dis = np.linalg.norm(self.point_b - another.point_b)
        ab_dis = np.linalg.norm(self.point_a - another.point_b)
        ba_dis = np.linalg.norm(self.point_b - another.point_a)
        return min(aa_dis, bb_dis, ab_dis, ba_dis)

    def get_dis_arr(self, others) -> np.ndarray:
        """get distance acoording to others antipodals

        Args:
            others (List[Antipodal]): other antipodals

        Returns:
            np.ndarray: distance array
        """
        other_num = len(others)
        other_a = np.empty(shape=(other_num, 3), dtype=float)
        other_b = np.empty(shape=(other_num, 3), dtype=float)
        for i in range(other_num):
            other_a[i] = others[i].point_a
            other_b[i] = others[i].point_b
        aa_dis = np.linalg.norm(other_a - self.point_a, axis=1)
        ab_dis = np.linalg.norm(other_a - self.point_b, axis=1)
        ba_dis = np.linalg.norm(other_b - self.point_a, axis=1)
        bb_dis = np.linalg.norm(other_b - self.point_b, axis=1)
        dis_arr = np.vstack([aa_dis, ab_dis, ba_dis, bb_dis]).min(axis=0)
        return dis_arr

    def _get_canonical_pose(self) -> np.ndarray:
        """get canonical pose of antipodal contact pair

        Returns:
            np.ndarray: canonical pose
        """
        # assume gripper closing along x_axis
        x_d = self.point_a - self.point_b
        x_d = x_d / np.linalg.norm(x_d)
        y_d = np.cross(np.array([0, 0, 1.0], dtype=float), x_d)
        if np.linalg.norm(y_d) < 1e-4:
            y_d = np.cross(np.array([1, 0, 0.0], dtype=float), x_d)
        y_d = y_d / np.linalg.norm(y_d)
        z_d = np.cross(x_d, y_d)
        pose = np.eye(4, dtype=float)
        pose[:3, 0] = x_d  # rotation x
        pose[:3, 1] = y_d  # rotation y
        pose[:3, 2] = z_d  # rotation z
        pose[:3, 3] = (self.point_a + self.point_b) / 2  # position
        return pose

    def sample_pose(self, sample_num: int = 36) -> np.ndarray:
        """sample parallel gripper grasp poses given antipodal contact pairs

        Args:
            sample_num (int, optional): sample number. Defaults to 36.

        Returns:
            np.ndarray: [sample_num, 4, 4] of float. Sample poses.
        """
        # assume gripper closing along x_axis
        x_d = self._canonical_pose[:3, 0]
        y_d = self._canonical_pose[:3, 1]
        z_d = self._canonical_pose[:3, 2]
        position = self._canonical_pose[:3, 3]
        beta_list = np.linspace(2 * np.pi / sample_num, 2 * np.pi, sample_num)
        grasp_poses = np.empty(shape=(sample_num, 4, 4), dtype=float)
        for i in range(sample_num):
            sample_z = np.sin(beta_list[i]) * y_d + np.cos(beta_list[i]) * z_d
            sample_z = sample_z / np.linalg.norm(sample_z)
            sample_y = np.cross(sample_z, x_d)
            pose = np.eye(4, dtype=float)
            pose[:3, 0] = x_d  # rotation x
            pose[:3, 1] = sample_y  # rotation y
            pose[:3, 2] = sample_z  # rotation z
            pose[:3, 3] = position  # position
            grasp_poses[i] = pose
        return grasp_poses


class AntipodalGenerator:
    def __init__(
        self,
        mesh_o3dt: o3d.t.geometry.TriangleMesh,
        open_length: float,
        min_open_length: float = 0.002,
        max_angle: float = np.pi / 10,
        surface_sample_num: int = 5000,
        layer_num: int = 4,
        sample_each_layer: int = 6,
        nms_ratio: float = 0.02,
        antipodal_sample_num: int = 16,
        unique_id: str = None,
        cache_dir: str = None,
    ):
        """antipodal grasp pose generator

        Args:
            mesh_o3dt (o3d.t.geometry.TriangleMesh): input mesh
            open_length (float): gripper maximum open length
            max_angle (float, optional): maximum grasp direction with surface normal. Defaults to np.pi/10.
            surface_sample_num (int, optional): contact sample number in mesh surface. Defaults to 5000.
            layer_num (int, optional): cone sample layer number . Defaults to 4.
            sample_each_layer (int, optional): cone sample number in each layer. Defaults to 6.
            nms_ratio (float, optional): nms distance ratio. Defaults to 0.02.
            antipodal_sample_num (int, optional): grasp poses sample on each antipodal contact pair. Defaults to 16.
            cache_dir (str, optional): file cache directory. Defaults to None.
        """
        self._antipodal_max_angle = max_angle
        self._open_length = open_length
        self._min_open_length = min_open_length
        self._mesh_o3dt = mesh_o3dt
        verts = mesh_o3dt.vertex.positions.numpy()
        self._center_of_mass = verts.mean(axis=0)
        if unique_id is None:
            unique_file_name = self._get_unique_id(
                mesh_o3dt, open_length, max_angle, surface_sample_num
            )
        else:
            unique_file_name = f"{unique_id}_{str(open_length)}_{str(max_angle)}_{str(surface_sample_num)}"
        if cache_dir is None:
            cache_dir = os.path.join(pathlib.Path.home(), "grasp_pose")
            logger.log_debug(f"Set cache directory to {cache_dir}.")
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        cache_file = os.path.join(cache_dir, f"{unique_file_name}.pickle")
        if not os.path.isfile(cache_file):
            # generate cache
            grasp_list = self._generate_cache(
                cache_file,
                mesh_o3dt=mesh_o3dt,
                max_angle=max_angle,
                surface_sample_num=surface_sample_num,
                layer_num=layer_num,
                sample_each_layer=sample_each_layer,
                nms_ratio=nms_ratio,
                antipodal_sample_num=antipodal_sample_num,
            )
        else:
            # load cache
            grasp_list = self._load_cache(cache_file)
        self._grasp_list = grasp_list

    def _get_unique_id(
        self,
        mesh_o3dt: o3d.t.geometry.TriangleMesh,
        open_length: float,
        max_angle: float,
        surface_sample_num: int,
    ) -> str:
        mesh_md5 = get_mesh_md5(mesh_o3dt)
        return (
            f"{mesh_md5}_{str(open_length)}_{str(max_angle)}_{str(surface_sample_num)}"
        )

    def _generate_cache(
        self,
        cache_file: str,
        mesh_o3dt: o3d.t.geometry.TriangleMesh,
        max_angle: float = np.pi / 10,
        surface_sample_num: int = 5000,
        layer_num: int = 4,
        sample_each_layer: int = 6,
        nms_ratio: float = 0.02,
        antipodal_sample_num: int = 16,
    ):
        self._mesh_o3dt = mesh_o3dt
        self._cone_sampler = ConeSampler(
            max_angle=max_angle,
            layer_num=layer_num,
            sample_each_layer=sample_each_layer,
        )
        mesh_o3dt = mesh_o3dt.compute_vertex_normals()
        assert 1e-4 < max_angle < np.pi / 2
        self._mesh_len = self._get_pc_size(mesh_o3dt.vertex.positions.numpy()).max()
        start_time = time.time()
        antipodal_list = self._antipodal_generate(mesh_o3dt, surface_sample_num)
        logger.log_debug(
            f"Antipodal sampling cost {(time.time() - start_time) * 1000} ms."
        )
        logger.log_debug(f"Find {len(antipodal_list)} initial antipodal pairs.")

        valid_antipodal_list = self._antipodal_clean(antipodal_list)

        start_time = time.time()
        nms_antipodal_list = self._antipodal_nms(
            valid_antipodal_list, nms_ratio=nms_ratio
        )
        logger.log_debug(f"NMS cost {(time.time() - start_time) * 1000} ms.")
        logger.log_debug(
            f"There are {len(nms_antipodal_list)} antipodal pair after nms."
        )
        # all poses
        start_time = time.time()
        grasp_poses, score, open_length = self._sample_grasp_pose(
            nms_antipodal_list, antipodal_sample_num
        )
        logger.log_debug(f"Pose sampling cost {(time.time() - start_time) * 1000} ms.")
        logger.log_debug(
            f"There are {grasp_poses.shape[0]} poses after grasp pose sampling."
        )
        # write data
        data_dict = {
            "grasp_poses": grasp_poses,
            "score": score,
            "open_length": open_length,
        }
        pickle.dump(data_dict, open(cache_file, "wb"))
        # TODO: contact pair visualization
        # self.antipodal_visual(nms_antipodal_list)
        grasp_num = grasp_poses.shape[0]
        logger.log_debug(f"Write {grasp_num} poses to pickle file {cache_file}.")
        grasp_list = [None for i in range(grasp_num)]
        for i in range(grasp_num):
            grasp_list[i] = AntipodalGrasp(grasp_poses[i], open_length[i], score[i])
        return grasp_list

    def _load_cache(self, cache_file: str):
        data_dict = pickle.load(open(cache_file, "rb"))
        grasp_num = data_dict["grasp_poses"].shape[0]
        logger.log_debug(f"Load {grasp_num} poses from pickle file {cache_file}.")
        grasp_list = [None for i in range(grasp_num)]
        for i in range(grasp_num):
            grasp_list[i] = AntipodalGrasp(
                data_dict["grasp_poses"][i],
                data_dict["open_length"][i],
                data_dict["score"][i],
            )
        return grasp_list

    def _get_pc_size(self, vertices):
        return np.array(
            [
                vertices[:, 0].max() - vertices[:, 0].min(),
                vertices[:, 1].max() - vertices[:, 1].min(),
                vertices[:, 2].max() - vertices[:, 2].min(),
            ]
        )

    def _antipodal_generate(
        self, mesh_o3dt: o3d.t.geometry.TriangleMesh, surface_sample_num: int = 5000
    ):
        surface_pcd = mesh_o3dt.to_legacy().sample_points_uniformly(surface_sample_num)
        points = np.array(surface_pcd.points)
        normals = np.array(surface_pcd.normals)
        point_num = points.shape[0]
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_o3dt)
        # raycast
        ray_n = self._cone_sampler._ray_num
        ray_num = point_num * ray_n
        ray_begins = np.empty(shape=(ray_num, 3), dtype=float)
        ray_direcs = np.empty(shape=(ray_num, 3), dtype=float)
        origin_normals = np.empty(shape=(ray_num, 3), dtype=float)
        origin_points = np.empty(shape=(ray_num, 3), dtype=float)
        start_time = time.time()
        for i in range(point_num):
            ray_direc = self._cone_sampler.cone_sample_direc(
                normals[i], is_visual=False
            )
            # raycast from outside of object
            ray_begin = points[i] - 2 * self._mesh_len * ray_direc
            ray_direcs[i * ray_n : (i + 1) * ray_n, :] = ray_direc
            ray_begins[i * ray_n : (i + 1) * ray_n, :] = ray_begin
            origin_normals[i * ray_n : (i + 1) * ray_n, :] = normals[i]
            origin_points[i * ray_n : (i + 1) * ray_n, :] = points[i]
        logger.log_debug(f"Cone sampling cost {(time.time() - start_time) * 1000} ms.")

        start_time = time.time()
        rays = o3d.core.Tensor(
            np.hstack([ray_begins, ray_direcs]), dtype=o3d.core.Dtype.Float32
        )
        logger.log_debug(f"Open3d raycast {(time.time() - start_time) * 1000} ms.")

        raycast_rtn = scene.cast_rays(rays)
        hit_dis_all = raycast_rtn["t_hit"].numpy()
        hit_normal_all = raycast_rtn["primitive_normals"].numpy()

        # max_angle_cos = np.cos(self._antipodal_max_angle)
        antipodal_list = []
        # get antipodal points
        start_time = time.time()
        for i in range(ray_num):
            hit_dis = hit_dis_all[i]
            hit_normal = hit_normal_all[i]
            hit_point = ray_begins[i] + ray_direcs[i] * hit_dis
            antipodal_dis = np.linalg.norm(origin_points[i] - hit_point)
            if (
                antipodal_dis > self._min_open_length
                and antipodal_dis < self._open_length
            ):
                # reject thin close object
                antipodal = Antipodal(
                    origin_points[i], hit_point, origin_normals[i], hit_normal
                )
                antipodal_list.append(antipodal)
        logger.log_debug(
            f"Antipodal initialize cost {(time.time() - start_time) * 1000} ms."
        )
        return antipodal_list

    def _sample_grasp_pose(
        self, antipodal_list: List[Antipodal], antipodal_sample_num: int = 36
    ):
        antipodal_num = len(antipodal_list)
        grasp_poses = np.empty(
            shape=(antipodal_sample_num * antipodal_num, 4, 4), dtype=float
        )
        scores = np.empty(shape=(antipodal_sample_num * antipodal_num,), dtype=float)
        open_length = np.empty(
            shape=(antipodal_sample_num * antipodal_num,), dtype=float
        )
        for i in range(antipodal_num):
            grasp_poses[i * antipodal_sample_num : (i + 1) * antipodal_sample_num] = (
                antipodal_list[i].sample_pose(antipodal_sample_num)
            )
            scores[i * antipodal_sample_num : (i + 1) * antipodal_sample_num] = (
                antipodal_list[i].score
            )
            open_length[i * antipodal_sample_num : (i + 1) * antipodal_sample_num] = (
                antipodal_list[i].dis
            )
        return grasp_poses, scores, open_length

    def get_all_grasp(self) -> List[AntipodalGrasp]:
        """get all grasp

        Returns:
            List[AntipodalGrasp]: list of grasp
        """
        return self._grasp_list

    def select_grasp(
        self,
        approach_direction: np.ndarray,
        select_num: int = 20,
        max_angle: float = np.pi / 10,
        select_method: GraspSelectMethod = GraspSelectMethod.NORMAL_SCORE,
    ) -> List[AntipodalGrasp]:
        """Select grasps. Masked by max_angle and sort by grasp score

        Args:
            approach_direction (np.ndarray): gripper approach direction
            select_num (int, optional): select grasp number. Defaults to 10.
            max_angle (float, optional): max angle threshold (angle with surface normal). Defaults to np.pi/10.
            select_method (select_method, optional)
        Return:
            List[AntipodalGrasp]: list of grasp
        """
        grasp_num = len(self._grasp_list)
        all_idx = np.arange(grasp_num)
        grasp_poses = np.empty(shape=(grasp_num, 4, 4), dtype=float)
        scores = np.empty(shape=(grasp_num,), dtype=float)
        position = grasp_poses[:, :3, 3]

        for i in range(grasp_num):
            grasp_poses[i] = self._grasp_list[i].pose
            scores[i] = self._grasp_list[i].score

        # mask acoording to table up direction
        grasp_z = grasp_poses[:, :3, 2]
        direc_dot = (grasp_z * approach_direction).sum(axis=1)
        valid_mask = direc_dot > np.cos(max_angle)
        valid_id = all_idx[valid_mask]

        # sort acoording to different grasp score
        if select_method == GraspSelectMethod.NORMAL_SCORE:
            valid_score = scores[valid_id]
            sort_valid_idx = np.argsort(valid_score)[::-1]  # large is better
        elif select_method == GraspSelectMethod.NEAR_APPROACH:
            position_dot = (position * approach_direction).sum(axis=1)
            valid_height = position_dot[valid_id]
            sort_valid_idx = np.argsort(valid_height)
        elif select_method == GraspSelectMethod.CENTER:
            center_dis = np.linalg.norm(position - self._center_of_mass, axis=1)
            valid_center_dis = center_dis[valid_id]
            sort_valid_idx = np.argsort(valid_center_dis)
        else:
            logger.log_warning(f"select_method {select_method.name} not implemented.")
            # return all grasp
            return self._grasp_list

        # get best score sample index
        result_num = min(len(sort_valid_idx), select_num)
        best_valid_idx = sort_valid_idx[:result_num]
        best_idx = valid_id[best_valid_idx]

        result_grasp_list = []
        for idx in best_idx:
            result_grasp_list.append(self._grasp_list[idx])
        return result_grasp_list

    def _antipodal_nms(
        self, antipodal_list: List[Antipodal], nms_ratio: float = 0.02
    ) -> List[Antipodal]:
        antipodal_num = len(antipodal_list)
        nms_mask = np.empty(shape=(antipodal_num,), dtype=bool)
        nms_mask.fill(True)
        score_list = np.empty(shape=(antipodal_num,), dtype=float)

        for i in range(antipodal_num):
            score_list[i] = antipodal_list[i].score

        sort_idx = np.argsort(score_list)[::-1]

        dis_th = self._mesh_len * nms_ratio
        for i in range(antipodal_num):
            if not nms_mask[sort_idx[i]]:
                continue
            antipodal_max = antipodal_list[sort_idx[i]]
            other_antipodal = []
            other_idx = []
            for j in range(i + 1, antipodal_num):
                if nms_mask[sort_idx[j]]:
                    other_antipodal.append(antipodal_list[sort_idx[j]])
                    other_idx.append(sort_idx[j])
            dis_arr = antipodal_max.get_dis_arr(other_antipodal)
            invalid_mask = dis_arr < dis_th
            for j, flag in enumerate(invalid_mask):
                if flag:
                    nms_mask[other_idx[j]] = False
        nms_antipodal_list = []
        for i in range(antipodal_num):
            if nms_mask[sort_idx[i]]:
                nms_antipodal_list.append(antipodal_list[sort_idx[i]])

        # TODO: nms validation check. remove in future
        # antipodal_num = len(nms_antipodal_list)
        # for i in range(antipodal_num):
        #     for j in range(i + 1, antipodal_num):
        #         antipodal_dis = nms_antipodal_list[i].get_dis(nms_antipodal_list[j])
        #         if antipodal_dis < dis_th:
        #             logger.log_warning(f"find near antipodal {i} and {j} with dis {antipodal_dis}")
        return nms_antipodal_list

    def _antipodal_clean(self, antipodal_list: List[Antipodal]):
        # TODO: need collision checker

        valid_antipodal = []
        max_angle_cos = np.cos(self._antipodal_max_angle)
        for antipodal in antipodal_list:
            if (
                1e-5 < antipodal.dis < self._open_length
                and antipodal.angle_cos > max_angle_cos
            ):
                valid_antipodal.append(antipodal)
        return valid_antipodal

    def antipodal_visual(self, antipodal_list):
        mesh_visual = self._mesh_o3dt.to_legacy()
        antipodal_num = len(antipodal_list)
        draw_points = np.empty(shape=(antipodal_num * 2, 3), dtype=float)
        draw_lines = np.empty(shape=(antipodal_num, 2), dtype=int)
        for i in range(antipodal_num):
            direc = antipodal_list[i].point_b - antipodal_list[i].point_a
            direc = direc / np.linalg.norm(direc)
            anti_begin = antipodal_list[i].point_a - direc * 0.005
            anti_end = antipodal_list[i].point_b + direc * 0.005
            draw_points[i * 2] = anti_begin
            draw_points[i * 2 + 1] = anti_end
            draw_lines[i] = np.array([i * 2, i * 2 + 1], dtype=int)

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(draw_points),
            lines=o3d.utility.Vector2iVector(draw_lines),
        )
        draw_colors = np.empty(shape=(antipodal_num, 3), dtype=float)
        draw_colors[:] = np.array([0.0, 1.0, 1.0])
        line_set.colors = o3d.utility.Vector3dVector(draw_colors)
        o3d.visualization.draw_geometries([line_set, mesh_visual])

    def grasp_pose_visual(
        self, grasp_list: List[AntipodalGrasp]
    ) -> List[o3d.t.geometry.TriangleMesh]:
        """visualize grasp pose

        Args:
            grasp_list (List[AntipodalGrasp]): list of grasp

        Returns:
            List[o3d.t.geometry.TriangleMesh]: list of visualization mesh
        """
        pose_num = len(grasp_list)
        visual_mesh_list = [self._mesh_o3dt.compute_vertex_normals()]

        max_angle_cos = np.cos(self._antipodal_max_angle)

        for i in range(pose_num):
            grasp_mesh = grasp_list[i].grasp_pose_visual_mesh(
                gripper_open_length=self._open_length
            )
            grasp_mesh.transform(grasp_list[i].pose)
            # low score: red | hight score: blue
            score_ratio = (grasp_list[i].score - max_angle_cos) / (1 - max_angle_cos)
            score_ratio = min(1.0, score_ratio)
            score_ratio = max(0.0, score_ratio)
            grasp_mesh.paint_uniform_color(np.array([1 - score_ratio, 0, score_ratio]))
            visual_mesh_list.append(o3d.t.geometry.TriangleMesh.from_legacy(grasp_mesh))
        return visual_mesh_list
