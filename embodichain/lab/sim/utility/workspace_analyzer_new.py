# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import gc
import os
import time
import numpy as np
import open3d as o3d
import torch
import dexsim

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Sequence
from itertools import product, islice
from tqdm import tqdm

from embodichain.utils import logger
from embodichain.lab.sim.objects import Robot
from scipy.spatial.transform import Rotation as R


@dataclass
class JointConfig:
    """Joint configuration parameters"""

    range: Tuple[float, float]  # Joint motion range
    samples: int  # Number of samples


@dataclass
class JointSamplingConfig:
    """Joint space sampling configuration"""

    joints: List[JointConfig]  # List of joint configurations


def batched(iterable, n):
    """Yield successive n-sized batches from iterable."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


class WorkspaceAnalyzer:
    def __init__(
        self,
        robot: Robot,
        name: str,
        joint_ranges: np.ndarray,
        resolution: float = np.radians(35),
    ):
        self.robot = robot
        self.solver = self.robot.get_solver(name)
        self.control_part = name
        self.resolution = resolution
        self.joint_ranges = np.array(joint_ranges)
        self.device = "cpu"

        self._sampling_configs = self._init_sampling_configs()

        self.control_part_base_xpos = self.robot.get_control_part_base_pose(
            name=name, to_matrix=True
        )

    def _get_fk_result(self, qpos: np.ndarray) -> Tuple[bool, np.ndarray]:
        r"""Calculate forward kinematics

        Computes the end-effector pose given joint angles.

        Args:
            qpos: Joint angles array

        Returns:
            tuple: (success, pose)
                - success (bool): True if calculation succeeded
                - pose (np.ndarray): 4x4 homogeneous transformation matrix
        """
        try:
            result = self.robot.compute_fk(name=self.control_part, qpos=qpos)

            # Default values
            success = False
            xpos = np.eye(4)

            # Handle different return types
            if isinstance(result, tuple):
                if len(result) >= 2:
                    success, xpos = result[:2]
            else:
                if result is None:
                    success = False
                else:
                    success = True
                    xpos = result

            return success, xpos

        except Exception as e:
            logger.log_warning(f"FK calculation failed: {str(e)}")
            return False, np.eye(4)

    def _get_ik_result(
        self, xpos: np.ndarray, qpos_seed: Optional[np.ndarray] = np.array([])
    ) -> Tuple[bool, np.ndarray]:
        """Calculate inverse kinematics

        Computes joint angles that achieve the desired end-effector pose.

        Args:
            xpos: Target 4x4 homogeneous transformation matrix
            qpos_seed: Initial joint angles for IK solver (optional)

        Returns:
            tuple: (success, joint_angles)
                - success (bool): True if solution found
                - joint_angles (np.ndarray): Solution joint angles
        """
        # try:
        # Call robot's IK solver
        result = self.robot.get_ik(
            uid=self.control_part, xpos=xpos, qpos_seed=qpos_seed
        )

        # Default values
        success = False
        q_sol = np.zeros(self.robot.get_dof(self.control_part))

        # Process IK result
        if isinstance(result, tuple):
            if len(result) >= 2:
                success, q_sol = result[:2]
        else:
            if result is None:
                success = False
            else:
                success = True
                q_sol = result

        return success, q_sol

        # except Exception as e:
        #     logger.log_warning(f"IK calculation failed: {str(e)}")
        #     return False, None

    def _init_sampling_configs(self) -> Dict[str, JointSamplingConfig]:
        r"""Initialize joint space sampling configurations

        Returns:
            Dictionary mapping config names to sampling configurations
        """
        original_ranges = self.joint_ranges.copy()

        self.joint_ranges = np.clip(self.joint_ranges, -np.pi, np.pi)

        clipped_joints = []
        for i, (orig, clipped) in enumerate(zip(original_ranges, self.joint_ranges)):
            if not np.allclose(orig, clipped):
                clipped_joints.append(i)

        if clipped_joints:
            logger.log_info("Some joint ranges were clipped to [-π, π]:")
            for joint_idx in clipped_joints:
                orig_range = original_ranges[joint_idx]
                new_range = self.joint_ranges[joint_idx]
                logger.log_info(
                    f"Joint {joint_idx}: [{orig_range[0]:.3f}, {orig_range[1]:.3f}] -> "
                    f"[{new_range[0]:.3f}, {new_range[1]:.3f}] rad"
                )

        # Calculate joint range sizes
        joint_ranges_size = np.abs(self.joint_ranges[:, 1] - self.joint_ranges[:, 0])

        # Calculate number of samples per joint
        samples = [
            max(3, int(np.ceil(range_size / self.resolution)))
            for range_size in joint_ranges_size
        ]

        # Create default sampling configuration
        sampling_config = JointSamplingConfig(
            joints=[
                JointConfig(range=joint_range, samples=sample_num)
                for joint_range, sample_num in zip(self.joint_ranges, samples)
            ],
        )

        # Log sampling configuration info
        logger.log_info(f"Analyze control part: [{self.control_part}]")
        logger.log_info(
            f"Angular Resolution: {self.resolution:.3f} rad ({np.degrees(self.resolution):.1f}°)"
        )
        for i, (joint_range, num_samples) in enumerate(zip(self.joint_ranges, samples)):
            range_size = abs(joint_range[1] - joint_range[0])
            actual_resolution = range_size / (num_samples - 1) if num_samples > 1 else 0
            logger.log_info(
                f"- Joint {i+1}: Range={range_size:.2f}rad, Samples={num_samples}, "
                f"Actual Resolution={actual_resolution:.3f}rad ({np.degrees(actual_resolution):.1f}°)"
            )

        return sampling_config

    def _generate_combinations(self, joint_values):
        r"""Generator function to produce joint angle combinations one at a time

        This avoids generating all combinations at once to save memory
        """
        if not joint_values:
            yield []
        else:
            for first in joint_values[0]:
                for rest in self._generate_combinations(joint_values[1:]):
                    yield [first] + rest

    def _process_batch(
        self, batch: List[np.ndarray], timeout: float = 10.0
    ) -> List[np.ndarray]:
        r"""Process a batch of joint configurations

        Args:
            batch: List of joint configurations to process
            timeout: Batch processing timeout in seconds

        Returns:
            List of end effector XYZ positions
        """
        positions = []
        start_time = time.time()

        for qpos in batch:
            if time.time() - start_time > timeout:
                logger.log_warning(f"Batch processing timeout ({timeout}s)")
                break

            try:
                qpos = np.array(qpos)
                res, xpos = self._get_fk_result(qpos=qpos)
                if res:
                    # Only save XYZ position
                    positions.append(xpos[:3, 3])
            except Exception as e:
                logger.log_warning(f"Error processing joint configuration: {str(e)}")
                continue

        return positions

    def _validate_params(self, cache_mode: str, save_dir: str):
        r"""Validate input parameters"""
        if cache_mode not in ["memory", "disk"]:
            raise ValueError("cache_mode must be 'memory' or 'disk'")

        if cache_mode == "disk" and save_dir is None:
            raise ValueError("save_dir must be provided when cache_mode is 'disk'")

    def _init_joint_values(self, config: JointSamplingConfig) -> List[np.ndarray]:
        r"""Initialize joint sampling values"""
        return [
            np.linspace(joint.range[0], joint.range[1], joint.samples)
            for joint in config.joints
        ]

    def _save_batch_results(
        self, positions: List[np.ndarray], save_dir: str, batch_id: int
    ):
        r"""Save results for a single batch

        Args:
            positions: List of XYZ positions
            save_dir: Directory to save results
            batch_id: Batch identifier
        """

        batch_dir = os.path.join(save_dir, "batches")
        # Ensure directory exists
        os.makedirs(batch_dir, exist_ok=True)
        # Save numpy array
        batch_path = os.path.join(batch_dir, f"batch_{batch_id:04d}.npy")
        np.save(batch_path, np.array(positions))
        logger.log_info(
            f"Saved batch {batch_id}: {len(positions)} points -> {batch_path}"
        )

    def _process_point_cloud(
        self,
        positions: List[np.ndarray],
        voxel_size: float = 0.05,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
        is_voxel_down: bool = True,
    ) -> o3d.geometry.PointCloud:
        r"""Process sampled point cloud data

        Args:
            positions: List of XYZ positions
            voxel_size: Voxel size (m)
            nb_neighbors: Number of neighbors for statistical filter
            std_ratio: Standard deviation ratio for statistical filter

        Returns:
            o3d.geometry.PointCloud: Processed point cloud
        """
        # Create point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(positions))

        logger.log_info(f"Point cloud processing:")

        if is_voxel_down:
            # 1. Voxel downsampling
            logger.log_info(
                f"- Performing voxel downsampling (voxel_size={voxel_size}m)"
            )
            pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

            # 2. Statistical outlier removal
            logger.log_info(
                f"- Removing outliers (neighbors={nb_neighbors}, std_ratio={std_ratio})"
            )
            cl, ind = pcd_down.remove_statistical_outlier(
                nb_neighbors=nb_neighbors, std_ratio=std_ratio
            )
            pcd_clean = pcd_down.select_by_index(ind)
        else:
            pcd_clean = pcd

        # 3. Estimate normals
        logger.log_info("- Estimating point cloud normals")
        pcd_clean.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2, max_nn=30
            )
        )

        # 4. Orient normals consistently
        logger.log_info("- Orienting normals consistently")

        pcd_clean.orient_normals_to_align_with_direction()

        # 5. Add color based on distance to origin
        points = np.asarray(pcd_clean.points)

        # Calculate distances to origin
        distances = np.linalg.norm(points, axis=1)

        # Find the centroid
        center = np.mean(points, axis=0)

        # Calculate distances to the centroid
        distances_to_center = np.linalg.norm(points - center, axis=1)

        # Normalize distances
        max_dist = np.max(distances_to_center)
        normalized_distances = distances_to_center / max_dist

        # Create HSV color space (green to red gradient)
        hsv_colors = np.zeros((len(points), 3))
        hsv_colors[:, 0] = 0.3333 * (
            1 - normalized_distances
        )  # Hue: green (0.3333) to red (0)
        hsv_colors[:, 1] = 1.0  # Saturation: max saturation
        hsv_colors[:, 2] = 0.8  # Value: medium brightness

        # Convert HSV to RGB
        colors = np.zeros_like(points)
        for i in range(len(points)):
            h, s, v = hsv_colors[i]

            # HSV to RGB conversion
            c = v * s
            x = c * (1 - abs((h * 6) % 2 - 1))
            m = v - c

            if h < 1 / 6:
                rgb = [c, x, 0]
            elif h < 2 / 6:
                rgb = [x, c, 0]
            elif h < 3 / 6:
                rgb = [0, c, x]
            elif h < 4 / 6:
                rgb = [0, x, c]
            elif h < 5 / 6:
                rgb = [x, 0, c]
            else:
                rgb = [c, 0, x]

            colors[i] = [r + m for r in rgb]

        pcd_clean.colors = o3d.utility.Vector3dVector(colors)

        logger.log_info(f"- Original points: {len(positions)}")
        logger.log_info(f"- Processed points: {len(pcd_clean.points)}")
        logger.log_info(
            f"- Distance range: {np.min(distances):.3f}m ~ {np.max(distances):.3f}m"
        )

        return pcd_clean

    def _merge_batch_files(self, save_dir: str, total_batches: int) -> List[np.ndarray]:
        r"""Merge all sampled points from batch files

        Args:
            save_dir: Directory to save data
            total_batches: Total number of batches

        Returns:
            List[np.ndarray]: List of all sampled positions
        """
        # Get current date for subdirectory name
        # current_date = time.strftime("%Y%m%d")
        batch_dir = os.path.join(save_dir, "batches")

        logger.log_info("Starting to merge batch files...")
        all_xpos = []

        # Load and process batches
        for batch_id in tqdm(range(total_batches), desc="Merging progress"):
            batch_path = os.path.join(batch_dir, f"batch_{batch_id:04d}.npy")

            try:
                # Load batch data
                batch_data = np.load(batch_path)
                all_xpos.extend(batch_data)
                # Delete processed batch file
                # os.remove(batch_path)
            except Exception as e:
                logger.log_warning(f"Error processing batch {batch_id}: {str(e)}")

        # Remove empty batch directory
        if os.path.exists(batch_dir) and not os.listdir(batch_dir):
            os.rmdir(batch_dir)

        logger.log_info(f"Merging complete: {len(all_xpos)} sampled points")
        return all_xpos

    def sample_qpos_workspace(
        self,
        resolution: float = None,
        cache_mode: str = "memory",  # Cache mode "memory" or "disk"
        save_dir: str = None,  # Save directory
        batch_size: int = 100000,  # Batch processing size
        save_threshold: int = 10000000,  # Save threshold
        use_cached: bool = True,  # Use cached results if available
    ) -> List[np.ndarray]:
        r"""Sample joint space and calculate corresponding workspace poses

        Args:
            resolution: Sampling resolution
            cache_mode: Cache mode ("memory" - in-memory list, "disk" - disk storage)
            save_dir: Save directory path (must be provided when cache_mode="disk")
            batch_size: Number of samples per batch
            save_threshold: Number of samples to accumulate before saving in disk mode
            use_cached: Whether to use cached results if available (only in disk mode)

        Returns:
            List[np.ndarray]: List of valid end effector poses of poses (in memory mode) or empty list (in disk mode)
        """
        if resolution is not None:
            self.resolution = resolution
            self._sampling_configs = self._init_sampling_configs()

        # Validate parameters
        self._validate_params(cache_mode, save_dir)

        # Initialize sampling configuration
        joint_values = self._init_joint_values(self._sampling_configs)
        total_samples = np.prod([len(values) for values in joint_values])

        logger.log_info(
            f"Sampling joint space with resolution {np.degrees(self.resolution):.1f}°..."
        )
        logger.log_info(f"Total sample points: {total_samples}")
        logger.log_info(f"Cache mode: {cache_mode}")
        logger.log_info(f"Save directory: {save_dir if save_dir else 'N/A'}")
        logger.log_info(f"Sampling using: {self.device}")

        if cache_mode == "memory":
            return self._sample_memory_mode(joint_values, total_samples, batch_size)
        else:
            return self._sample_disk_mode(
                joint_values,
                total_samples,
                save_dir,
                batch_size,
                save_threshold,
                use_cached,
            )

    def _sample_memory_mode(
        self, joint_values: List[np.ndarray], total_samples: int, batch_size: int
    ) -> List[np.ndarray]:
        r"""Memory mode sampling"""
        if not self.robot.pk_serial_chain:
            all_xpos = []
            for qpos in tqdm(
                product(*joint_values),
                total=total_samples,
                desc="Memory mode serial sampling",
            ):
                q = np.array(qpos, dtype=np.float32)
                res, xpos = self._get_fk_result(qpos=q)
                if res:
                    all_xpos.append(xpos)
                if len(all_xpos) % 1000 == 0:
                    gc.collect()
            return all_xpos
        self.chain = self.robot.pk_serial_chain[self.control_part].to(
            dtype=torch.float32, device=self.device
        )
        sampled_xpos = []
        joint_combinations = product(*joint_values)

        T_tcp = torch.as_tensor(self.solver.get_tcp(), dtype=torch.float32).to(
            self.device
        )

        with tqdm(
            total=total_samples,
            desc=f"Sampling {total_samples} points (batch={batch_size})",
        ) as pbar:
            for qpos_batch in batched(joint_combinations, batch_size):
                # compute and collect
                batch_mats = self._compute_batch_xpos(qpos_batch, T_tcp)
                sampled_xpos.extend(batch_mats)

                # advance progress bar and cleanup
                pbar.update(len(batch_mats))
                gc.collect()

        return sampled_xpos

    def _sample_disk_mode(
        self,
        joint_values: List[np.ndarray],
        total_samples: int,
        save_dir: str,
        batch_size: int,
        save_threshold: int,
        use_cached: bool = True,
    ) -> List[np.ndarray]:
        r"""Disk mode sampling, with serial fallback if no pk_serial_chain."""
        # 1) If batches already exist, just merge & return
        batches_dir = os.path.join(save_dir, "batches")
        if os.path.exists(batches_dir) and use_cached:
            npy_files = [f for f in os.listdir(batches_dir) if f.endswith(".npy")]
            if npy_files:
                return self._merge_batch_files(save_dir, len(npy_files))

        sampled_xpos = []
        current_batch = []
        total_processed = 0
        batch_count = 0

        # 2) Choose serial vs. GPU path
        if not self.robot.pk_serial_chain:
            # serial, one qpos at a time
            current_batch = []
            with tqdm(total=total_samples, desc="Disk mode serial sampling") as pbar:
                for qpos in product(*joint_values):
                    q = np.array(qpos, dtype=np.float32)
                    res, xpos = self._get_fk_result(qpos=q)
                    if res:
                        current_batch.append(xpos)
                        # flush by batch_size
                        if len(current_batch) >= batch_size:
                            sampled_xpos.extend(current_batch)
                            total_processed += len(current_batch)
                            current_batch = []
                            # flush to disk by save_threshold
                            if len(sampled_xpos) >= save_threshold:
                                self._save_batch_results(
                                    sampled_xpos, save_dir, batch_count
                                )
                                batch_count += 1
                                sampled_xpos = []
                                gc.collect()
                    pbar.update(1)

        else:
            self.chain = self.robot.pk_serial_chain[self.control_part].to(
                dtype=torch.float32, device=self.device
            )
            # GPU‐batched path
            T_tcp = torch.as_tensor(
                self.robot.get_tcp(self.control_part),
                dtype=torch.float32,
                device=self.device,
            )
            with tqdm(
                total=total_samples, desc=f"Sampling in {batch_size}-sized batches"
            ) as pbar:
                for qpos_batch in batched(product(*joint_values), batch_size):
                    batch_mats = self._compute_batch_xpos(qpos_batch, T_tcp)
                    sampled_xpos.extend(batch_mats)
                    total_processed += len(batch_mats)
                    # flush to disk by save_threshold
                    if len(sampled_xpos) >= save_threshold:
                        self._save_batch_results(sampled_xpos, save_dir, batch_count)
                        batch_count += 1
                        sampled_xpos = []
                        gc.collect()
                    pbar.update(len(batch_mats))

        # Process remaining samples
        if sampled_xpos:
            self._save_batch_results(sampled_xpos, save_dir, batch_count)
            batch_count += 1

        logger.log_info(
            f"Sampling complete: {total_processed} samples, {batch_count} batches"
        )

        # If there are saved batches, read and merge them to process point cloud
        if batch_count > 0:
            all_xpos = self._merge_batch_files(save_dir, batch_count)
            return all_xpos

        return None

    def sample_xpos_workspace(
        self,
        ref_xpos: np.ndarray,
        xpos_resolution: float = 0.2,
        qpos_resolution: float = np.radians(60),
        cache_mode: str = "memory",
        save_dir: str = None,
        batch_size: int = 5000,
        save_threshold: int = 10000000,
        pos_eps: float = 5e-4,
        rot_eps: float = 5e-4,
        max_iterations: int = 1500,
        num_samples: int = 5,
        use_cached: bool = True,
    ) -> List[np.ndarray]:
        r"""Sample Cartesian space and calculate corresponding joints

        Args:
            ref_xpos (np.ndarray): Reference end-effector pose matrix (4x4) defining the
                                orientation for IK solutions. Translation components will
                                be overridden during sampling.
            xpos_resolution (float, optional): Cartesian space sampling resolution in meters.
                                            Smaller values provide finer sampling but increase
                                            computation time. Defaults to 0.2 meters.
            qpos_resolution (float, optional): Angular resolution for initial joint space
                                            sampling in radians. Used to determine workspace
                                            bounds. Defaults to 60 degrees.
            cache_mode (str, optional): Caching strategy, either:
                                    - "memory": Store samples in memory (faster but memory-intensive)
                                    - "disk": Save samples to disk (slower but memory-efficient)
                                    Defaults to "memory".
            save_dir (str, optional): Directory path for saving results when using disk cache.
                                    Must be provided if cache_mode is "disk". Defaults to None.
            batch_size (int, optional): Number of samples to process in each batch.
                                    Larger values may improve performance but increase
                                    memory usage. Defaults to 5000.
            save_threshold (int, optional): Number of samples to accumulate before saving
                                        to disk in disk mode. Defaults to 10,000,000.
            pos_eps (float, optional): Position tolerance for IK solutions in meters.
                                            Defaults to 5e-4.
            rot_eps (float, optional): Rotation tolerance for IK solutions in radians.
                                            Defaults to 5e-4.
            max_iterations (int, optional): Maximum iterations for IK solver.
                                            Defaults to 1500.
            num_samples (int, optional): Number of IK samples to generate for each position.
                                            Defaults to 5.
            use_cached (bool, optional): Whether to use cached results if available (only in disk mode)

        Returns:
            List[np.ndarray]: List of valid end effector poses
        """
        # logger.set_log_level(level="error")

        start_time = time.time()
        try:
            qpos_sampled_xpos = self.sample_qpos_workspace(
                resolution=qpos_resolution,
                cache_mode="memory",
                batch_size=5000,
                save_threshold=save_threshold,
            )

            qpos_all_positions = [xpos[:3, 3] for xpos in qpos_sampled_xpos]
            qpos_pcd = self._process_point_cloud(positions=qpos_all_positions)
            aabb = qpos_pcd.get_axis_aligned_bounding_box()

            sample_points = self._sample_in_aabb(
                aabb.min_bound, aabb.max_bound, xpos_resolution
            )

            # Validate parameters
            self._validate_params(cache_mode, save_dir)

            if cache_mode == "memory":
                return self._sample_xpos_memory_mode(
                    positions=sample_points,
                    ref_xpos=ref_xpos,
                    batch_size=batch_size,
                    pos_eps=pos_eps,
                    rot_eps=rot_eps,
                    max_iterations=max_iterations,
                    num_samples=num_samples,
                )
            else:
                return self._sample_xpos_disk_mode(
                    positions=sample_points,
                    ref_xpos=ref_xpos,
                    save_dir=save_dir,
                    batch_size=batch_size,
                    pos_eps=pos_eps,
                    rot_eps=rot_eps,
                    max_iterations=max_iterations,
                    num_samples=num_samples,
                    save_threshold=save_threshold,
                    use_cached=use_cached,
                )
        finally:
            logger.set_log_level(level="info")
            # Record the end time
            end_time = time.time()
            # Calculate the time cost
            time_cost = end_time - start_time
            logger.log_info(f"Time cost: {time_cost:.2f} seconds")

    def _compute_batch_xpos(
        self, qpos_batch: Sequence[np.ndarray], T_tcp: torch.Tensor
    ) -> List[np.ndarray]:
        """Given a batch of q-poses, compute TCP-transformed FK matrices
        and return them as numpy float16 arrays."""
        # 1) to NumPy (float32) → to torch.Tensor on correct device
        np_qpos = np.array(qpos_batch, dtype=np.float32)
        tensor_qpos = torch.as_tensor(np_qpos, dtype=torch.float32, device=self.device)

        # 2) batched forward kinematics → 4×4 matrices
        ret_batch = self.chain.forward_kinematics(
            tensor_qpos, end_only=True
        ).get_matrix()

        # 3) apply TCP offset
        T_final = torch.matmul(ret_batch, T_tcp)

        T_final = torch.bmm(
            self.control_part_base_xpos.to(dtype=torch.float32).expand(
                T_final.shape[0], -1, -1
            ),
            T_final,
        )

        # 4) move to CPU, cast to float16
        T_cpu16 = T_final.cpu().to(dtype=torch.float16)

        # 5) return list of numpy arrays
        return [mat.numpy() for mat in T_cpu16]

    def _sample_in_aabb(
        self, min_bound: np.ndarray, max_bound: np.ndarray, resolution: float
    ) -> np.ndarray:
        r"""Uniformly sample within an axis-aligned bounding box (AABB)

        Args:
            min_bound: AABB minimum bound [x_min, y_min, z_min]
            max_bound: AABB maximum bound [x_max, y_max, z_max]
            resolution: Sampling resolution (m)

        Returns:
            np.ndarray: Array of sampled points with shape (N, 3)
        """
        # Calculate number of samples per axis
        num_samples = np.ceil((max_bound - min_bound) / resolution).astype(int)

        # Ensure at least 2 samples per dimension
        num_samples = np.maximum(num_samples, 2)

        # Generate sample points for each axis
        x = np.linspace(min_bound[0], max_bound[0], num_samples[0])
        y = np.linspace(min_bound[1], max_bound[1], num_samples[1])
        z = np.linspace(min_bound[2], max_bound[2], num_samples[2])

        # Create a grid of points
        X, Y, Z = np.meshgrid(x, y, z)

        # Convert grid to N×3 array
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        logger.log_info(f"Sampling space range:")
        logger.log_info(f"- X: [{min_bound[0]:.3f}, {max_bound[0]:.3f}] m")
        logger.log_info(f"- Y: [{min_bound[1]:.3f}, {max_bound[1]:.3f}] m")
        logger.log_info(f"- Z: [{min_bound[2]:.3f}, {max_bound[2]:.3f}] m")
        logger.log_info(f"Sampling resolution: {resolution:.3f} m")
        logger.log_info(f"Number of samples: {len(points)}")

        return points

    def _sample_xpos_memory_mode(
        self,
        positions: List[np.ndarray],
        ref_xpos: np.ndarray,
        batch_size: int,
        pos_eps: float,
        rot_eps: float,
        max_iterations: int,
        num_samples: int,
    ) -> List[np.ndarray]:
        r"""Memory mode sampling with batch processing and progress bar

        Args:
            positions: List of positions to validate.
            ref_xpos: Reference end effector pose.
            batch_size (int): Number of positions to process in each batch.

        Returns:
            List[np.ndarray]: List of valid end effector poses.
        """
        valid_xpos = []

        # Get the degree of freedom (DOF) of the robot to create joint seed
        dof_number = self.robot.get_dof(self.control_part)

        # Total number of positions to process
        total_positions = len(positions)

        # TODO: Optimize efficiency by using batch IK if available.
        #       If self.robot implements get_batch_ik_solution, prefer batch processing for IK to significantly accelerate sampling.
        #       Otherwise, fall back to single-point IK calls (slower).
        #       This check ensures the most efficient computation path is used automatically.
        #       (Batch IK can greatly improve performance for large-scale workspace sampling.)
        # Example:
        # if hasattr(self.robot, "get_batch_ik_solution"):
        if False:
            # If the robot has get_batch_ik_solution, use it for batch processing
            num_batches = (total_positions // batch_size) + (
                1 if total_positions % batch_size != 0 else 0
            )

            # Create progress bar with total samples and batch size
            with tqdm(
                total=total_positions, desc=f"Sampling in {batch_size}-sized batches"
            ) as pbar:
                # Iterate through positions in batches
                for batch_idx in range(num_batches):
                    # Select the current batch of positions
                    batch_positions = positions[
                        batch_idx * batch_size : (batch_idx + 1) * batch_size
                    ]

                    # Create a batch of target poses (batch_size, 4, 4)
                    target_xpos_batch = []
                    for point in batch_positions:
                        target_xpos = ref_xpos.copy()
                        target_xpos[:3, 3] = point
                        target_xpos_batch.append(target_xpos)

                    # Convert to numpy array (batch_size, 4, 4)
                    target_xpos_batch = np.array(target_xpos_batch)
                    # Create joint seed batch of zeros (batch_size, dof)
                    joint_seed_batch = np.zeros((len(batch_positions), dof_number))
                    # Use get_batch_ik_solution for batch processing
                    res, _ = self.robot.get_batch_ik_solution(
                        target_xpos_list=target_xpos_batch,  # Batch of target poses
                        joint_seed_list=joint_seed_batch,  # Batch of joint seeds (zeros)
                        uid=self.control_part,
                        is_world_coordinates=False,  # Set based on your use case
                        pos_eps=pos_eps,
                        rot_eps=rot_eps,
                        max_iterations=max_iterations,
                        num_samples=num_samples,
                    )

                    # Append valid target poses to valid_xpos
                    for j, is_valid in enumerate(res):
                        if is_valid:
                            valid_xpos.append(target_xpos_batch[j])

                    # Update the progress bar after processing the batch
                    pbar.update(
                        len(batch_positions)
                    )  # Update progress bar with batch size

                    # Perform garbage collection after every batch
                    if len(valid_xpos) % 1000 == 0:
                        gc.collect()

        else:
            # Fallback to the previous method if get_batch_ik_solution is not available
            with tqdm(
                total=total_positions, desc="Sampling in single IK calls"
            ) as pbar:
                for point in positions:
                    # Construct target pose
                    target_xpos = ref_xpos.copy()
                    target_xpos[:3, 3] = point

                    # Calculate IK using the old method (get_ik)
                    res, _ = self.robot.get_ik(uid=self.control_part, xpos=target_xpos)
                    if res:
                        valid_xpos.append(target_xpos)

                    # Update the progress bar after each point is processed
                    pbar.update(1)  # Update progress bar with 1 point

                    # Perform garbage collection after every 1000 valid points
                    if len(valid_xpos) % 1000 == 0:
                        gc.collect()

        return valid_xpos if valid_xpos else None

    def _sample_xpos_disk_mode(
        self,
        positions: List[np.ndarray],
        ref_xpos: np.ndarray,
        save_dir: str,
        batch_size: int,
        pos_eps: float,
        rot_eps: float,
        max_iterations: int,
        num_samples: int,
        save_threshold: int,
        use_cached: bool = True,
    ) -> List[np.ndarray]:
        r"""Disk mode sampling with batch processing

        Args:
            positions: List of positions to validate.
            ref_xpos: Reference end effector pose.
            save_dir: Directory to save results.
            batch_size: Number of samples per batch.
            save_threshold: Number of samples to accumulate before saving.

        Returns:
            List[np.ndarray]: List of valid end effector poses.
        """
        valid_positions = []
        current_batch = []
        total_processed = 0
        batch_count = 0
        # Record the start time
        logger.log_info(f"Starting disk mode sampling...")
        logger.log_info(f"Save directory: {save_dir}")

        # If there are saved batches, read and return without calculation
        batches_dir = os.path.join(save_dir, "batches")
        if os.path.exists(batches_dir) and use_cached:
            npy_files = [f for f in os.listdir(batches_dir) if f.endswith(".npy")]
            batch_count = len(npy_files)

            if batch_count > 0:
                all_xpos = self._merge_batch_files(save_dir, batch_count)
                return all_xpos

        # Check if self.robot has the method get_batch_ik_solution
        if hasattr(self.robot, "get_batch_ik_solution"):
            # If get_batch_ik_solution is available, use batch processing
            with tqdm(total=len(positions), desc="Disk mode sampling") as pbar:
                for i in range(0, len(positions), batch_size):
                    # Select the current batch of positions
                    batch_positions = positions[i : i + batch_size]

                    # Create a batch of target poses (batch_size, 4, 4)
                    target_xpos_batch = []
                    for point in batch_positions:
                        target_xpos = ref_xpos.copy()
                        target_xpos[:3, 3] = point
                        target_xpos_batch.append(target_xpos)

                    # Convert to numpy array (batch_size, 4, 4)
                    target_xpos_batch = np.array(target_xpos_batch)

                    # Create the joint seed batch (batch_size, dof)
                    dof_number = self.robot.get_dof(self.control_part)
                    joint_seed_batch = np.zeros((len(batch_positions), dof_number))

                    # Use get_batch_ik_solution for batch processing
                    res, _ = self.robot.get_batch_ik_solution(
                        target_xpos_list=target_xpos_batch,  # Batch of target poses
                        joint_seed_list=joint_seed_batch,  # Batch of joint seeds (zeros)
                        uid=self.control_part,
                        is_world_coordinates=False,  # Set based on your use case
                        pos_eps=pos_eps,
                        rot_eps=rot_eps,
                        max_iterations=max_iterations,
                        num_samples=num_samples,
                    )

                    # Append valid target poses to valid_positions
                    for j, is_valid in enumerate(res):
                        if is_valid:
                            current_batch.append(target_xpos_batch[j])

                    # Process batch when it reaches batch_size
                    if len(current_batch) >= batch_size:
                        valid_positions.extend(current_batch)
                        total_processed += len(current_batch)

                        current_batch = []

                        # Save when reaching the threshold
                        if len(valid_positions) >= save_threshold:
                            self._save_batch_results(
                                valid_positions, save_dir, batch_count
                            )
                            batch_count += 1
                            valid_positions = []
                            gc.collect()

                    # Update the progress bar
                    pbar.update(len(batch_positions))  # Update with batch size

        else:
            # Fallback to the previous method if get_batch_ik_solution is not available
            with tqdm(total=len(positions), desc="Disk mode sampling") as pbar:
                for point in positions:
                    # Construct target pose
                    target_xpos = ref_xpos.copy()
                    target_xpos[:3, 3] = point

                    # Calculate IK using the old method (get_ik)
                    res, _ = self.robot.compute_ik(
                        name=self.control_part, pose=target_xpos
                    )
                    if res:
                        current_batch.append(target_xpos)

                    # Process batch when it reaches batch_size
                    if len(current_batch) >= batch_size:
                        valid_positions.extend(current_batch)
                        total_processed += len(current_batch)

                        current_batch = []

                        # Save when reaching the threshold
                        if len(valid_positions) >= save_threshold:
                            self._save_batch_results(
                                valid_positions, save_dir, batch_count
                            )
                            batch_count += 1
                            valid_positions = []
                            gc.collect()

                    # Update the progress bar
                    pbar.update(1)  # Update with 1 point per iteration

        # Process remaining data
        if current_batch:
            valid_positions.extend(current_batch)
            total_processed += len(current_batch)

        if valid_positions:
            self._save_batch_results(valid_positions, save_dir, batch_count)
            batch_count += 1

        logger.log_info(
            f"Sampling complete: {total_processed} samples, {batch_count} batches"
        )

        # If there are saved batches, read and merge them to process point cloud
        if batch_count > 0:
            all_xpos = self._merge_batch_files(save_dir, batch_count)
            return all_xpos

        return None

    def sample_voxel_workspace(
        self,
        voxel_size: float = 0.04,
        num_directions: int = 50,
        num_yaws: int = 6,
        pos_eps: float = 2e-4,
        rot_eps: float = 2e-4,
        max_iterations: int = 1500,
        num_samples: int = 5,
        cache_mode: str = "memory",
        save_dir: str = None,
        batch_size: int = 5000,
        save_threshold: int = 10000000,
        use_cached: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        r"""Sample Cartesian space using voxel‐based IK reachability.

        Divides the workspace into a grid of voxels around the arm base, then for
        each voxel center sweeps through a set of directions and yaw rotations,
        calling the IK solver to test reachability.

        Args:
            voxel_size (float, optional):
                Edge length of each cubic voxel in meters.
                Smaller voxels give finer resolution but increase computation.
                Defaults to 0.04.
            num_directions (int, optional):
                Number of unit‐vector directions to sample on the sphere for each
                voxel. More directions improve angular coverage at the cost of
                additional IK calls. Defaults to 50.
            num_yaws (int, optional):
                Number of discrete yaw rotations **around the local Z‐axis** to
                attempt for each direction when solving IK. Higher values increase
                rotational sampling but incur more IK calls. Defaults to 6.
            pos_eps (float, optional):
                Position tolerance for IK solutions in meters.
                Defaults to 5e-4.
            rot_eps (float, optional):
                Rotation tolerance for IK solutions in radians.
                Defaults to 5e-4.
            max_iterations (int, optional):
                Maximum iterations for IK solver.
                Defaults to 1500.
            num_samples (int, optional):
                Number of IK samples to generate for each position.
                Defaults to 5.
            cache_mode (str, optional):
                Caching strategy for IK results:
                - `"memory"`: keep all samples in RAM (fast, memory‐intensive)
                - `"disk"`: stream to disk in batches (slower, memory‐efficient)
                Defaults to `"memory"`.
            save_dir (str, optional):
                Directory path for saving/loading cached batches when using
                `cache_mode="disk"`. Required in disk mode. Defaults to None.
            batch_size (int, optional):
                Number of successful IK poses to accumulate before adding them to
                the in‐memory pool. Larger values may improve throughput but
                increase temporary memory usage. Defaults to 5000.
            save_threshold (int, optional):
                Number of poses in the in‐memory pool at which point they are
                written out to disk as a batch file. Helps limit peak RAM use.
                Defaults to 10,000,000.
            use_cached: Whether to use cached results if available (only in disk mode)

        Returns:
            Tuple[
                np.ndarray,          # (M,3) array of voxel‐center coordinates
                np.ndarray,          # (M,) array of success counts per center
                List[np.ndarray]     # flat list of all valid 4×4 IK pose matrices
            ]
        """
        logger.set_log_level(level="error")

        try:
            self._validate_params(cache_mode, save_dir)

            logger.log_info(f"Sampling robot workspace with voxel size {voxel_size}...")
            logger.log_info(f"Cache mode: {cache_mode}")
            logger.log_info(f"Sampling using: {self.device}")

            arm_base_pos = self.robot.get_base_xpos(name=self.control_part)[:3, 3]
            arm_ee_pos = self.robot.get_current_xpos(name=self.control_part)[:3, 3]
            arm_length = float(np.linalg.norm(arm_ee_pos - arm_base_pos))

            if cache_mode == "memory":
                return self._sample_voxels_memory_mode(
                    voxel_size, num_directions, num_yaws, arm_base_pos, arm_length
                )
            else:
                return self._sample_voxels_disk_mode(
                    voxel_size,
                    num_directions,
                    num_yaws,
                    arm_base_pos,
                    arm_length,
                    save_dir=save_dir,
                    save_threshold=save_threshold,
                    batch_size=batch_size,
                    use_cached=use_cached,
                )
        finally:
            logger.set_log_level(level="info")

    def _voxel_centers_in_sphere(self, arm_base, arm_length, voxel_size):
        """
        Compute centers of all voxels of size `voxel_size` whose centers lie
        within a sphere of radius `arm_length` around `arm_base`, using the
        exact range definitions you provided for x, y, and z.

        Args:
            arm_base (sequence of 3 floats): (x, y, z) origin.
            arm_length (float): radius of the sphere.
            voxel_size (float): edge length of each cubic voxel.

        Returns:
            numpy.ndarray of shape (M, 3): each row is a valid (x, y, z) center.
        """
        x, y, z = arm_base
        r = float(arm_length)
        half = voxel_size / 2.0

        # follow your exact ranges
        x_range = np.arange(x - half, x + r + half, voxel_size)
        y_range = np.arange(y - half, y + r + half, voxel_size)
        z_range = np.arange(z - r / 2 - half, z + r / 2 + half, voxel_size)

        # build full grid of candidate centers
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        pts = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)

        # keep only those inside the sphere of radius r
        d2 = np.sum((pts - np.array(arm_base)) ** 2, axis=1)
        return pts[d2 <= r**2]

    def _generate_uniform_directions(self, num_directions: int = 50):
        """
        Generate vectors in evenly distributed n directions
        """
        phi = np.pi * (3.0 - np.sqrt(5.0))
        directions = []
        for i in range(num_directions):
            z = 1 - 2 * i / float(num_directions - 1)
            theta = phi * i
            x = np.sqrt(1 - z * z) * np.cos(theta)
            y = np.sqrt(1 - z * z) * np.sin(theta)
            directions.append(np.array([x, y, z]))

        return directions

    # Helper function
    def normalize(self, v: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v  # Avoid division by zero
        return v / norm

    def _compute_ik_solutions(
        self,
        centers: List[np.ndarray],
        directions: List[np.ndarray],
        voxel_size: float,
        num_yaws: int,
        pos_eps: float = 2e-4,
        rot_eps: float = 2e-4,
        max_iterations: int = 1500,
        num_samples: int = 5,
    ) -> List[np.ndarray]:
        """
        Compute IK solutions for a set of centers and directions.
        This function will process the centers and directions in batches if `get_batch_ik_solution` is available.

        Args:
            centers: List of center positions to compute IK for.
            directions: List of direction vectors to compute IK for.
            voxel_size: Size of the voxel to offset the centers.
            num_yaws: Number of yaw sweeps to attempt.
            robot_base: Transformation matrix of the robot base.
            yaw_rot: Rotation matrix for yaw rotation.

        Returns:
            List[np.ndarray]: List of valid IK poses.
        """
        valid_poses = []
        success_counts = [0] * len(centers)

        # Create progress bar
        pbar = tqdm(total=len(centers), ncols=100, desc="Computing IK (per-center)")

        yaw_angle = 360.0 / num_yaws
        yaw_rot = R.from_euler("z", yaw_angle, degrees=True).as_matrix()
        robot_base = self.robot.get_base_xpos(name=self.control_part)

        # Check if self.robot has the method get_batch_ik_solution
        if hasattr(self.robot, "get_batch_ik_solution"):
            # If get_batch_ik_solution is available, we process in batches
            for i, center in enumerate(centers):
                batch_positions = []
                batch_xpos = []

                for d in directions:
                    # Build local frame so that its Z-axis = -d
                    z_axis = -d
                    up = (
                        np.array([0, 1, 0])
                        if abs(z_axis[1]) < 0.9
                        else np.array([1, 0, 0])
                    )
                    x_axis = self.normalize(np.cross(up, z_axis))
                    y_axis = np.cross(z_axis, x_axis)
                    frame = np.stack([x_axis, y_axis, z_axis], axis=1)

                    # Shift out to the surface of the voxel
                    pos = center + d * (voxel_size * 0.5)

                    # Try yaw sweeps
                    for _ in range(num_yaws):
                        frame = frame @ yaw_rot
                        xpos = np.eye(4)
                        xpos[:3, :3] = frame
                        xpos[:3, 3] = pos
                        xpos_robot = np.linalg.inv(robot_base) @ xpos

                        # Prepare batch for IK computation
                        batch_positions.append(pos)
                        batch_xpos.append(xpos_robot)

                # Convert lists to numpy arrays (batch_size, 4, 4)
                batch_xpos_array = np.array(batch_xpos)

                # Create the joint seed batch (batch_size, dof)
                dof_number = self.robot.get_dof(self.control_part)
                joint_seed_batch = np.zeros((len(batch_xpos), dof_number))

                # Use get_batch_ik_solution for batch processing
                res, _ = self.robot.get_batch_ik_solution(
                    target_xpos_list=batch_xpos_array,  # Batch of target poses
                    joint_seed_list=joint_seed_batch,  # Batch of joint seeds (zeros)
                    uid=self.control_part,
                    is_world_coordinates=False,  # Set based on your use case
                    pos_eps=pos_eps,
                    rot_eps=rot_eps,
                    max_iterations=max_iterations,
                    num_samples=num_samples,
                )

                # Append valid target poses to valid_poses
                for j, is_valid in enumerate(res):
                    if is_valid:
                        success_counts[i] += 1
                        valid_poses.append(batch_xpos_array[j])

                # Update the progress bar after processing the batch
                pbar.update(1)

        else:
            # Fallback to the previous method (get_ik) if get_batch_ik_solution is not available
            for i, center in enumerate(centers):
                for d in directions:
                    # Build local frame so that its Z-axis = -d
                    z_axis = -d
                    up = (
                        np.array([0, 1, 0])
                        if abs(z_axis[1]) < 0.9
                        else np.array([1, 0, 0])
                    )
                    x_axis = self.normalize(np.cross(up, z_axis))
                    y_axis = np.cross(z_axis, x_axis)
                    frame = np.stack([x_axis, y_axis, z_axis], axis=1)

                    # Shift out to the surface of the voxel
                    pos = center + d * (voxel_size * 0.5)

                    # Try yaw sweeps
                    for _ in range(num_yaws):
                        frame = frame @ yaw_rot
                        xpos = np.eye(4)
                        xpos[:3, :3] = frame
                        xpos[:3, 3] = pos
                        xpos_robot = np.linalg.inv(robot_base) @ xpos

                        # Calculate IK using the old method (get_ik)
                        is_success, _ = self.robot.get_ik(
                            xpos=xpos_robot, uid=self.control_part
                        )
                        if is_success:
                            success_counts[i] += 1
                            valid_poses.append(xpos_robot.copy())
                            break  # stop yaw for this direction

                pbar.update(1)

        logger.log_info(f"Sampling complete: {sum(success_counts)} valid positions.")

        return success_counts, valid_poses

    def _sample_voxels_memory_mode(
        self,
        voxel_size: float,
        num_directions: int,
        num_yaws: int,
        arm_base: np.ndarray,
        arm_length: float,
        pos_eps: float,
        rot_eps: float,
        max_iterations: int,
        num_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:

        dirs = self._generate_uniform_directions(num_directions)
        centers = self._voxel_centers_in_sphere(arm_base, arm_length, voxel_size)

        success_counts, ik_matrices = self._compute_ik_solutions(
            centers,
            dirs,
            voxel_size,
            num_yaws,
            pos_eps,
            rot_eps,
            max_iterations,
            num_samples,
        )

        return centers, success_counts, ik_matrices

    def _sample_voxels_disk_mode(
        self,
        voxel_size: float,
        num_directions: int,
        num_yaws: int,
        arm_base: np.ndarray,
        arm_length: float,
        pos_eps: float,
        rot_eps: float,
        max_iterations: int,
        num_samples: int,
        save_dir: str,
        batch_size: int,
        save_threshold: int,
        use_cached: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """
        Returns:
            centers:        (M,3) np.ndarray of voxel centers
            success_counts: (M,) np.ndarray of ints
            valid_poses:    list of 4x4 np.ndarrays
        """
        counts_file = os.path.join(save_dir, "success_counts.npy")
        batches_dir = os.path.join(save_dir, "batches")

        # 1) generate dirs & centers
        dirs = self._generate_uniform_directions(num_directions)
        centers = self._voxel_centers_in_sphere(arm_base, arm_length, voxel_size)

        # 2) if already computed, load & return
        if os.path.isdir(batches_dir) and os.path.exists(counts_file) and use_cached:
            npy_files = [f for f in os.listdir(batches_dir) if f.endswith(".npy")]
            if npy_files:
                success_counts = np.load(counts_file)
                valid_poses = self._merge_batch_files(save_dir, len(npy_files))
                return centers, success_counts, valid_poses

        os.makedirs(batches_dir, exist_ok=True)

        # 3) run IK sweep
        success_counts, valid_poses = self._compute_ik_solutions(
            centers,
            dirs,
            voxel_size,
            num_yaws,
            pos_eps,
            rot_eps,
            max_iterations,
            num_samples,
        )
        if success_counts.sum() == 0:
            return centers, success_counts, []

        # 4) save counts
        np.save(counts_file, success_counts)

        # 5) batch & save using a local temp buffer
        temp_valid = []
        valid_block = []
        batch_count = 0

        for pose in valid_poses:
            # collect into small blocks of batch_size
            valid_block.append(pose)
            if len(valid_block) >= batch_size:
                # move into temp_valid
                temp_valid.extend(valid_block)
                valid_block = []

                # once buffer reaches save_threshold, flush to disk
                if len(temp_valid) >= save_threshold:
                    self._save_batch_results(temp_valid, save_dir, batch_count)
                    batch_count += 1
                    temp_valid = []
                    gc.collect()

        # move any remaining block into temp_valid
        if valid_block:
            temp_valid.extend(valid_block)

        # final flush of anything left in temp_valid
        if temp_valid:
            self._save_batch_results(temp_valid, save_dir, batch_count)
            batch_count += 1

        # 6) merge all batch files and return
        all_poses = self._merge_batch_files(save_dir, batch_count)
        return centers, success_counts, all_poses


def compute_xpos_reachability(
    robot: Robot,
    name: str,
    ref_xpos: np.ndarray,
    xpos_resolution: float = 0.2,
    qpos_resolution: float = np.radians(60),
    pos_eps: float = 5e-4,
    rot_eps: float = 5e-4,
    max_iterations: int = 1500,
    num_samples: int = 5,
    batch_size: int = 100000,
    save_threshold: int = 10000000,
    qpos_limits: np.ndarray = None,
    cache_mode: str = "disk",
    visualize: bool = True,
    use_cached: bool = True,
    **kwargs,
) -> Tuple[
    Optional[list[np.ndarray]],  # First return: list of sampled 4x4 poses
    Optional[
        dexsim.models.PointCloud
    ],  # Second return: point cloud handle if visualization is enabled
]:
    """Compute the robot's reachable workspace by Cartesian space sampling.

        Samples points in Cartesian space and checks reachability using inverse kinematics.
        If `visualize` is True, visualizes reachable positions as a colored point cloud;
        Otherwise, only performs the sampling result as open3d PointCloud.


    Args:
        name (str): Identifier of the robot drive controller to analyze
        ref_xpos (np.ndarray): Reference end-effector pose matrix (4x4) defining the
                            orientation for IK solutions
        xpos_resolution (float, optional): Cartesian space sampling resolution in meters.
                                        Smaller values provide finer sampling but increase
                                        computation time. Defaults to 0.2 meters.
        qpos_resolution (float, optional): Angular resolution for initial joint space
                                        sampling in radians. Used to determine workspace
                                        bounds. Defaults to 60 degrees.
        pos_eps (float, optional): Position tolerance for IK solutions in meters.
                                        Defaults to 2e-4 meters.
        rot_eps (float, optional): Rotation tolerance for IK solutions in radians.
                                        Defaults to 2e-4 radians.
        max_iterations (int, optional): Maximum number of IK iterations per sample.
                                        Defaults to 2000.
        num_samples (int, optional): Number of samples to generate in Cartesian space.
                                        Defaults to 10.
        qpos_limits (np.ndarray, optional): Custom joint limits array of shape (n_joints, 2).
                                        If None, uses limits from drive controller or
                                        articulation. Defaults to None
        cache_mode (str, optional): Cache mode for workspace analysis. Options include "memory" and "disk".
                                    Defaults to "memory".
        visualize (bool, optional): If set to True, returns an extra Dexsim PointCloud handle for visualization.
                                    Defaults to True.
        use_cached (bool, optional): If True and `cache_mode` is "disk", attempts to load precomputed results.
                                    Ignored for "memory" mode. Defaults to True.

    Returns:
        Tuple[Optional[list[np.ndarray]], Optional[dexsim.models.PointCloud]]:
            The first element is a list of sampled end-effector poses (4×4 transformation matrices) if sampling succeeds, otherwise None.
            The second element is a point cloud handle if visualization is enabled and successful, otherwise None.
    """
    from embodichain.lab.sim import REACHABLE_XPOS_DIR
    from dexsim.utility.env_utils import create_point_cloud_from_o3d_pcd
    from dexsim.utility import inv_transform

    if name not in robot.control_parts:
        logger.log_warning(f"Drive controller '{name}' not found")
        return None, None

    # try:
    # Get robot configuration
    # base_xpos = robot.get_control_part_base_pose(name=name, to_matrix=True).squeeze(0).cpu().numpy()
    # ref_xpos_robot = inv_transform(base_xpos) @ ref_xpos
    ref_xpos_robot = ref_xpos

    if qpos_limits is None:
        joint_ranges = (
            robot.body_data.qpos_limits[0].cpu().numpy()[robot.get_joint_ids(name=name)]
        )
    else:
        joint_ranges = qpos_limits

    urdf_path = robot.cfg.fpath
    robot_name = os.path.splitext(os.path.basename(urdf_path))[0]

    qpos_resolution_str = f"{qpos_resolution:.2f}".replace(".", "_")
    xpos_resolution_str = f"{xpos_resolution:.2f}".replace(".", "_")
    # Join into one directory name
    save_dir = (
        REACHABLE_XPOS_DIR
        / f"{robot_name}_{name}_{qpos_resolution_str}_{xpos_resolution_str}"
    )

    # Initialize workspace analyzer
    analyzer = WorkspaceAnalyzer(
        robot=robot,
        name=name,
        resolution=qpos_resolution,
        joint_ranges=joint_ranges,
    )
    # Sample workspace points
    sampled_xpos = analyzer.sample_xpos_workspace(
        ref_xpos=ref_xpos_robot,
        xpos_resolution=xpos_resolution,
        qpos_resolution=qpos_resolution,
        cache_mode=cache_mode,
        batch_size=batch_size,
        save_dir=save_dir,
        save_threshold=save_threshold,
        pos_eps=pos_eps,
        rot_eps=rot_eps,
        max_iterations=max_iterations,
        num_samples=num_samples,
        use_cached=use_cached,
    )

    if visualize:
        if sampled_xpos is None:
            logger.log_warning("No reachable positions found.")
            return None, None
        all_positions = [xpos[:3, 3] for xpos in sampled_xpos]
        pcd = analyzer._process_point_cloud(
            positions=all_positions, is_voxel_down=False
        )
        # Transfer to World Coordinate
        # pcd.transform(base_xpos)
        # Create and configure point cloud visualization
        from embodichain.lab.sim.utility.sim_utils import get_dexsim_arenas

        pcd_handle = create_point_cloud_from_o3d_pcd(
            pcd=pcd, env=get_dexsim_arenas()[0]
        )
    else:
        return sampled_xpos, None

    return sampled_xpos, pcd_handle
