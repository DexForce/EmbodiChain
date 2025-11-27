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

import gc
import os
import time
import numpy as np
import torch

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
from contextlib import contextmanager
from pathlib import Path
from enum import Enum

from embodichain.utils import logger
from embodichain.lab.sim.objects.robot import Robot

# Import configuration classes
from embodichain.lab.sim.utility.workspace_analyzer.configs import (
    CacheConfig,
    DimensionConstraint,
    SamplingConfig,
    VisualizationConfig,
    MetricConfig,
)

# Import modules
from embodichain.lab.sim.utility.workspace_analyzer.samplers import (
    SamplerFactory,
    BaseSampler,
)
from embodichain.lab.sim.utility.workspace_analyzer.caches import CacheManager
from embodichain.lab.sim.utility.workspace_analyzer.constraints import (
    WorkspaceConstraintChecker,
)
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import (
    create_visualizer,
    VisualizationType,
)

__all__ = [
    "WorkspaceAnalyzer",
    "WorkspaceAnalyzerConfig",
    "AnalysisMode",
]


class AnalysisMode(Enum):
    """Workspace analysis mode."""

    JOINT_SPACE = "joint_space"
    """Sample in joint space, compute FK to get workspace points."""

    CARTESIAN_SPACE = "cartesian_space"
    """Sample in Cartesian space, compute IK to verify reachability."""


@dataclass
class WorkspaceAnalyzerConfig:
    """Complete configuration for workspace analyzer."""

    mode: AnalysisMode = AnalysisMode.JOINT_SPACE
    """Analysis mode: joint space or Cartesian space sampling."""

    sampling: SamplingConfig = None
    cache: CacheConfig = None
    constraint: DimensionConstraint = None
    visualization: VisualizationConfig = None
    metric: MetricConfig = None

    ik_success_threshold: float = 0.9
    """For Cartesian mode: minimum IK success rate to consider a point reachable."""

    ik_samples_per_point: int = 1
    """For Cartesian mode: number of random joint seeds to try for each Cartesian point."""

    def __post_init__(self):
        """Initialize sub-configs with defaults if not provided."""
        if self.sampling is None:
            self.sampling = SamplingConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.constraint is None:
            self.constraint = DimensionConstraint()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
        if self.metric is None:
            self.metric = MetricConfig()


class WorkspaceAnalyzer:
    """Main workspace analyzer class for robotic manipulation.

    Analyzes the reachable workspace of a robot by sampling joint configurations,
    computing forward kinematics, and generating metrics and visualizations.
    """

    def __init__(
        self,
        robot: Robot,
        config: Optional[WorkspaceAnalyzerConfig] = None,
        control_part_name: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize the workspace analyzer.

        Args:
            robot: Robot instance to analyze.
            config: Configuration object. If None, uses defaults.
            control_part_name: Name of the control part (e.g., "left_arm", "right_arm").
                              If None, uses the default solver or first available control part.
            device: PyTorch device for computations.
        """
        self.robot = robot
        self.config = config or WorkspaceAnalyzerConfig()
        self.device = device

        # Determine control part name
        self.control_part_name = self._determine_control_part(control_part_name)

        # Extract joint limits from robot
        self._setup_joint_limits()

        # Initialize components
        self.sampler = self._create_sampler()
        self.cache = self._create_cache()
        self.constraint_checker = self._create_constraint_checker()

        # Storage for analysis results
        self.workspace_points: Optional[torch.Tensor] = None
        self.joint_configurations: Optional[torch.Tensor] = None
        self.metrics_results: Dict[str, Any] = {}

        logger.log_info("WorkspaceAnalyzer initialized successfully")

    def _determine_control_part(
        self, control_part_name: Optional[str]
    ) -> Optional[str]:
        """Determine the control part name to use.

        Args:
            control_part_name: User-specified control part name, or None.

        Returns:
            The control part name to use, or None for default solver.
        """
        if control_part_name is not None:
            # User explicitly specified a control part
            logger.log_info(f"Using user-specified control part: {control_part_name}")
            return control_part_name

        # Try to find a suitable default control part
        if hasattr(self.robot, "cfg") and hasattr(self.robot.cfg, "control_parts"):
            control_parts = self.robot.cfg.control_parts
            if control_parts:
                # Priority order for default control parts
                priority_parts = ["left_arm", "right_arm"]

                # Try priority parts first
                for part in priority_parts:
                    if part in control_parts:
                        logger.log_info(f"Auto-selected control part: {part}")
                        return part

                # If no priority part found, use the first available
                first_part = next(iter(control_parts.keys()))
                logger.log_info(
                    f"Auto-selected first available control part: {first_part}"
                )
                return first_part

        # Fall back to None (will use default solver)
        logger.log_info("No specific control part specified, using default solver")
        return None

    def _setup_joint_limits(self) -> None:
        """Extract and setup joint limits from the robot."""
        # Get all joint limits from robot
        all_joint_limits = self.robot._entities[0].get_joint_limits()

        # If control_part_name is specified, get only the joints for that part
        if self.control_part_name is not None:
            joint_ids = self.robot.get_joint_ids(self.control_part_name)
            self.qpos_limits = all_joint_limits[joint_ids]
            logger.log_info(
                f"Using {len(joint_ids)} joints from control part '{self.control_part_name}'"
            )
        else:
            # Use all joints
            self.qpos_limits = all_joint_limits
            logger.log_info("Using all robot joints (no control part specified)")

        # Apply scaling factor if specified
        if self.config.constraint.joint_limits_scale != 1.0:
            scale = self.config.constraint.joint_limits_scale
            center = (self.qpos_limits[:, 0] + self.qpos_limits[:, 1]) / 2
            range_half = (self.qpos_limits[:, 1] - self.qpos_limits[:, 0]) / 2 * scale

            self.qpos_limits[:, 0] = center - range_half
            self.qpos_limits[:, 1] = center + range_half

            logger.log_info(f"Joint limits scaled by factor: {scale}")

        self.num_joints = len(self.qpos_limits)
        logger.log_debug(f"Number of joints: {self.num_joints}")

    def _create_sampler(self) -> BaseSampler:
        """Create sampler based on configuration."""
        factory = SamplerFactory()
        return factory.create_sampler(
            strategy=self.config.sampling.strategy,
            seed=self.config.sampling.seed,
        )

    def _create_cache(self):
        """Create cache manager based on configuration."""
        if not self.config.cache.enabled:
            return None
        return CacheManager.create_cache_from_config(self.config.cache)

    def _create_constraint_checker(self) -> WorkspaceConstraintChecker:
        """Create constraint checker based on configuration."""
        return WorkspaceConstraintChecker.from_config(
            self.config.constraint, device=self.device
        )

    def sample_joint_space(self, num_samples: Optional[int] = None) -> torch.Tensor:
        """Sample joint configurations within joint limits.

        Args:
            num_samples: Number of samples to generate. If None, uses config value.

        Returns:
            Tensor of shape (num_samples, num_joints) containing joint configurations.
        """
        num_samples = num_samples or self.config.sampling.num_samples

        # Sample from joint space
        joint_samples = self.sampler.sample(
            bounds=self.qpos_limits, num_samples=num_samples
        )

        logger.log_info(f"Generated {num_samples} joint space samples")
        return joint_samples

    def sample_cartesian_space(self, num_samples: Optional[int] = None) -> torch.Tensor:
        """Sample Cartesian positions within workspace bounds.

        Args:
            num_samples: Number of samples to generate. If None, uses config value.

        Returns:
            Tensor of shape (num_samples, 3) containing Cartesian positions.
        """
        num_samples = num_samples or self.config.sampling.num_samples

        # Determine Cartesian bounds
        if (
            self.config.constraint.min_bounds is not None
            and self.config.constraint.max_bounds is not None
        ):
            cartesian_bounds = torch.stack(
                [
                    torch.tensor(self.config.constraint.min_bounds, device=self.device),
                    torch.tensor(self.config.constraint.max_bounds, device=self.device),
                ],
                dim=1,
            )
        else:
            # Use a default reasonable workspace
            logger.log_warning(
                "No Cartesian bounds specified, using default [-1, 1] for all axes"
            )
            cartesian_bounds = torch.tensor(
                [[-1.0, 1.0], [-1.0, 1.0], [0.0, 2.0]],
                device=self.device,
            )

        # Sample from Cartesian space
        cartesian_samples = self.sampler.sample(
            bounds=cartesian_bounds, num_samples=num_samples
        )

        logger.log_info(f"Generated {num_samples} Cartesian space samples")
        return cartesian_samples

    def compute_workspace_points(
        self, joint_configs: torch.Tensor, batch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute end-effector positions for given joint configurations.

        Args:
            joint_configs: Joint configurations, shape (num_samples, num_joints).
            batch_size: Batch size for FK computation. If None, uses config value.

        Returns:
            Tuple of:
                - workspace_points: End-effector positions, shape (num_valid, 3)
                - valid_configs: Valid joint configurations, shape (num_valid, num_joints)
        """
        num_samples = len(joint_configs)

        workspace_points_list = []
        valid_configs_list = []

        logger.log_info(f"Computing FK for {num_samples} samples...")

        # Track valid points for progress bar
        total_valid = 0

        # Robot expects one configuration at a time (batch_size from robot environments, not samples)
        # Process each configuration individually
        pbar = tqdm(
            range(num_samples),
            desc="Computing FK",
            smoothing=0.05,  # 快速响应速度变化
            mininterval=0.5,  # 每0.5秒更新一次
            unit="cfg",  # 单位：配置
            unit_scale=False,
        )
        for i in pbar:
            qpos = joint_configs[i : i + 1]  # Keep batch dimension

            try:
                # Compute forward kinematics
                pose = self.robot.compute_fk(
                    qpos=qpos,
                    name=self.control_part_name,
                    to_matrix=True,
                )

                # Extract position (x, y, z)
                position = pose[:, :3, 3]  # Shape: (1, 3)

                # Filter by constraints
                valid_mask = self.constraint_checker.check_bounds(position)

                # Store valid results
                if valid_mask.any():
                    workspace_points_list.append(position[valid_mask])
                    valid_configs_list.append(qpos[valid_mask])
                    total_valid += 1

                # Update progress bar with validity statistics
                validity_rate = total_valid / (i + 1) * 100
                pbar.set_postfix(
                    {"valid": f"{total_valid}/{i+1}", "rate": f"{validity_rate:.1f}%"}
                )

            except Exception as e:
                logger.log_warning(f"FK computation failed for sample {i}: {e}")
                continue

        # Concatenate all results
        if workspace_points_list:
            workspace_points = torch.cat(workspace_points_list, dim=0)
            valid_configs = torch.cat(valid_configs_list, dim=0)
        else:
            workspace_points = torch.empty((0, 3), device=self.device)
            valid_configs = torch.empty((0, self.num_joints), device=self.device)

        logger.log_info(
            f"Computed {len(workspace_points)} valid workspace points "
            f"from {num_samples} samples "
            f"({len(workspace_points) / num_samples * 100:.1f}% success rate)"
        )

        return workspace_points, valid_configs

    def compute_reachability(
        self, cartesian_points: torch.Tensor, batch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute reachability for Cartesian points using IK.

        Args:
            cartesian_points: Cartesian positions, shape (num_samples, 3).
            batch_size: Batch size for IK computation. If None, uses config value.

        Returns:
            Tuple of:
                - reachable_points: Reachable positions, shape (num_reachable, 3)
                - success_rates: IK success rate for each point, shape (num_samples,)
                - best_configs: Best joint configurations, shape (num_reachable, num_joints)
        """
        batch_size = batch_size or self.config.sampling.batch_size
        num_samples = len(cartesian_points)
        ik_samples_per_point = self.config.ik_samples_per_point

        reachable_points_list = []
        success_rates_list = []
        best_configs_list = []

        logger.log_info(
            f"Computing IK for {num_samples} Cartesian samples "
            f"({ik_samples_per_point} seeds per point)..."
        )

        # Create a random sampler for generating IK seeds (avoid UniformSampler issues)
        from embodichain.lab.sim.utility.workspace_analyzer.samplers import (
            RandomSampler,
        )

        random_sampler = RandomSampler(seed=self.config.sampling.seed)

        # Track statistics for progress bar
        total_reachable = 0

        # Process each point individually (robot expects batch_size from environments, not samples)
        pbar = tqdm(
            range(num_samples),
            desc="Computing IK",
            smoothing=0.05,
            mininterval=0.5,
            unit="pt",
            unit_scale=False,
        )
        # Get current end-effector orientation from FK
        # Use current joint configuration to determine a realistic target orientation
        current_qpos = self.robot.get_qpos()[0][
            self.robot.get_joint_ids(self.control_part_name)
        ]
        current_ee_pose = self.robot.compute_fk(
            name=self.control_part_name, qpos=current_qpos.unsqueeze(0), to_matrix=True
        )  # Shape: (1, 4, 4)
        for i in pbar:
            position = cartesian_points[i]  # Shape: (3,)

            # Create target pose: use current orientation, replace position with sampled position
            pose = current_ee_pose.clone()
            pose[0, :3, 3] = position

            # Try multiple random seeds for this point
            success_count = 0
            best_qpos = None

            logger.set_log_level("ERROR")  # Suppress warnings during IK attempts
            for seed_idx in range(ik_samples_per_point):
                # Generate random joint seed using RandomSampler
                random_seed = random_sampler.sample(
                    bounds=self.qpos_limits, num_samples=1
                )  # Shape: (1, num_joints)

                try:
                    # Compute IK
                    ret, qpos = self.robot.compute_ik(
                        pose=pose,
                        joint_seed=random_seed,
                        name=self.control_part_name,
                    )

                    # Count successes
                    if ret is not None and ret[0]:
                        success_count += 1
                        # Store first successful configuration
                        if best_qpos is None:
                            best_qpos = qpos[0]  # Extract from batch dimension

                except Exception as e:
                    logger.log_warning(
                        f"IK computation failed for sample {i}, seed {seed_idx}: {e}"
                    )
                    continue
            logger.set_log_level("INFO")  # Restore log level

            # Calculate success rate for this point
            success_rate = success_count / ik_samples_per_point

            # Filter by success threshold
            if (
                success_rate >= self.config.ik_success_threshold
                and best_qpos is not None
            ):
                reachable_points_list.append(position.unsqueeze(0))  # Add batch dim
                success_rates_list.append(
                    torch.tensor([success_rate], device=self.device)
                )
                best_configs_list.append(best_qpos.unsqueeze(0))  # Add batch dim
                total_reachable += 1

            # Update progress bar with reachability statistics
            reachability_rate = total_reachable / (i + 1) * 100
            pbar.set_postfix(
                {
                    "reachable": f"{total_reachable}/{i+1}",
                    "rate": f"{reachability_rate:.1f}%",
                }
            )

        # Concatenate results
        if reachable_points_list:
            reachable_points = torch.cat(reachable_points_list, dim=0)
            success_rates = torch.cat(success_rates_list, dim=0)
            best_configs = torch.cat(best_configs_list, dim=0)
        else:
            reachable_points = torch.empty((0, 3), device=self.device)
            success_rates = torch.empty((0,), device=self.device)
            best_configs = torch.empty((0, self.num_joints), device=self.device)

        logger.log_info(
            f"Found {len(reachable_points)} reachable points "
            f"from {num_samples} samples "
            f"({len(reachable_points) / num_samples * 100:.1f}% reachability)"
        )

        return reachable_points, success_rates, best_configs

    def analyze(
        self, num_samples: Optional[int] = None, force_recompute: bool = False
    ) -> Dict[str, Any]:
        """Perform complete workspace analysis.

        Args:
            num_samples: Number of samples to generate. If None, uses config value.
            force_recompute: If True, recomputes even if cached results exist.

        Returns:
            Dictionary containing analysis results.
        """
        logger.log_info("=" * 60)
        logger.log_info("Starting Workspace Analysis")
        logger.log_info("=" * 60)

        start_time = time.time()

        # Check cache
        if not force_recompute and self.cache is not None:
            cached_results = self._load_from_cache()
            if cached_results is not None:
                logger.log_info("Loaded results from cache")
                return cached_results

        # Choose analysis mode
        if self.config.mode == AnalysisMode.JOINT_SPACE:
            # Joint space mode: Sample joints → FK → Workspace points
            logger.log_info(f"\nMode: {AnalysisMode.JOINT_SPACE.value}")

            # Step 1: Sample joint space
            logger.log_info("\n[1/3] Sampling joint space...")
            joint_configs = self.sample_joint_space(num_samples)

            # Step 2: Compute workspace points
            logger.log_info("\n[2/3] Computing workspace points via FK...")
            workspace_points, valid_configs = self.compute_workspace_points(
                joint_configs
            )

            # Store results
            self.workspace_points = workspace_points
            self.joint_configurations = valid_configs

            results = {
                "mode": AnalysisMode.JOINT_SPACE.value,
                "workspace_points": workspace_points,
                "joint_configurations": valid_configs,
                "num_samples": num_samples or self.config.sampling.num_samples,
                "num_valid": len(workspace_points),
            }

        else:  # CARTESIAN_SPACE mode
            # Cartesian space mode: Sample Cartesian → IK → Verify reachability
            logger.log_info(f"\nMode: {AnalysisMode.CARTESIAN_SPACE.value}")

            # Step 1: Sample Cartesian space
            logger.log_info("\n[1/3] Sampling Cartesian space...")
            cartesian_samples = self.sample_cartesian_space(num_samples)

            # Step 2: Compute reachability via IK
            logger.log_info("\n[2/3] Computing reachability via IK...")
            reachable_points, success_rates, best_configs = self.compute_reachability(
                cartesian_samples
            )

            # Store results
            self.workspace_points = reachable_points
            self.joint_configurations = best_configs

            results = {
                "mode": AnalysisMode.CARTESIAN_SPACE.value,
                "workspace_points": reachable_points,
                "joint_configurations": best_configs,
                "success_rates": success_rates,
                "num_samples": num_samples or self.config.sampling.num_samples,
                "num_reachable": len(reachable_points),
            }

        # Step 3: Compute metrics (common for both modes)
        logger.log_info("\n[3/3] Computing metrics...")
        metrics = self._compute_metrics()
        results["metrics"] = metrics
        results["config"] = self.config
        results["analysis_time"] = time.time() - start_time

        # Cache results
        if self.cache is not None:
            self._save_to_cache(results)

        logger.log_info("\n" + "=" * 60)
        logger.log_info(f"Analysis completed in {results['analysis_time']:.2f}s")
        logger.log_info("=" * 60)

        return results

    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute workspace metrics based on configuration."""
        if self.workspace_points is None or len(self.workspace_points) == 0:
            logger.log_warning("No workspace points available for metrics computation")
            return {}

        metrics = {}

        # TODO: Implement metric computation using metrics module
        # For now, compute basic statistics
        points_np = self.workspace_points.cpu().numpy()

        metrics["bounding_box"] = {
            "min": points_np.min(axis=0).tolist(),
            "max": points_np.max(axis=0).tolist(),
        }

        metrics["centroid"] = points_np.mean(axis=0).tolist()

        dimensions = points_np.max(axis=0) - points_np.min(axis=0)
        metrics["dimensions"] = dimensions.tolist()

        # Approximate volume (bounding box)
        metrics["bounding_box_volume"] = float(np.prod(dimensions))

        logger.log_info(f"Computed {len(metrics)} metrics")

        return metrics

    def visualize(
        self,
        vis_type: VisualizationType = VisualizationType.POINT_CLOUD,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> Any:
        """Visualize the workspace.

        Args:
            vis_type: Type of visualization to create.
            show: Whether to display the visualization.
            save_path: Optional path to save the visualization.

        Returns:
            Visualization object.
        """
        if self.workspace_points is None or len(self.workspace_points) == 0:
            logger.log_error("No workspace points available for visualization")
            return None

        if not self.config.visualization.enabled:
            logger.log_warning("Visualization is disabled in configuration")
            return None

        logger.log_info(f"Creating {vis_type.value} visualization...")

        # Create visualizer
        visualizer = create_visualizer(
            vis_type=vis_type, config=self.config.visualization
        )

        # Convert points to numpy
        points_np = self.workspace_points.cpu().numpy()

        # Create visualization
        vis_obj = visualizer.visualize(points_np)

        # Save if requested
        if save_path:
            visualizer.save(save_path)
            logger.log_info(f"Saved visualization to {save_path}")

        # Show if requested
        if show:
            try:
                import open3d as o3d

                if hasattr(vis_obj, "paint_uniform_color"):
                    o3d.visualization.draw_geometries([vis_obj])
            except ImportError:
                logger.log_warning("Open3D not available for interactive visualization")

        return vis_obj

    def _load_from_cache(self) -> Optional[Dict[str, Any]]:
        """Load analysis results from cache."""
        if self.cache is None:
            return None

        try:
            # TODO: Implement cache loading logic
            return None
        except Exception as e:
            logger.log_warning(f"Failed to load from cache: {e}")
            return None

    def _save_to_cache(self, results: Dict[str, Any]) -> None:
        """Save analysis results to cache."""
        if self.cache is None:
            return

        try:
            # TODO: Implement cache saving logic
            pass
        except Exception as e:
            logger.log_warning(f"Failed to save to cache: {e}")

    def get_workspace_bounds(self) -> Dict[str, np.ndarray]:
        """Get the bounding box of the analyzed workspace.

        Returns:
            Dictionary with 'min' and 'max' bounds.
        """
        if self.workspace_points is None or len(self.workspace_points) == 0:
            logger.log_warning("No workspace points available")
            return {"min": None, "max": None}

        points_np = self.workspace_points.cpu().numpy()
        return {"min": points_np.min(axis=0), "max": points_np.max(axis=0)}

    def export_results(self, output_path: str, format: str = "npz") -> None:
        """Export analysis results to file.

        Args:
            output_path: Path to save the results.
            format: Output format ('npz', 'pkl', 'json').
        """
        if self.workspace_points is None:
            logger.log_error("No analysis results to export")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "npz":
            np.savez(
                output_path,
                workspace_points=self.workspace_points.cpu().numpy(),
                joint_configurations=self.joint_configurations.cpu().numpy(),
                metrics=self.metrics_results,
            )
        elif format == "pkl":
            import pickle

            with open(output_path, "wb") as f:
                pickle.dump(
                    {
                        "workspace_points": self.workspace_points.cpu().numpy(),
                        "joint_configurations": self.joint_configurations.cpu().numpy(),
                        "metrics": self.metrics_results,
                    },
                    f,
                )
        elif format == "json":
            import json

            # Convert tensors to lists for JSON serialization
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "workspace_points": self.workspace_points.cpu()
                        .numpy()
                        .tolist(),
                        "metrics": self.metrics_results,
                    },
                    f,
                    indent=2,
                )
        else:
            logger.log_error(f"Unsupported format: {format}")
            return

        logger.log_info(f"Exported results to {output_path}")

    @contextmanager
    def profiling(self):
        """Context manager for profiling workspace analysis."""
        logger.log_info("Starting profiled analysis...")
        start_time = time.time()
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        yield

        end_time = time.time()
        end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        logger.log_info(f"Analysis time: {end_time - start_time:.2f}s")
        if torch.cuda.is_available():
            mem_used = (end_mem - start_mem) / 1024**2
            logger.log_info(f"GPU memory used: {mem_used:.2f} MB")


def draw_workspace_points(
    sim,
    workspace_points,
    marker_name="workspace_points",
    axis_size=0.01,
    axis_len=0.03,
    arena_index=0,
):
    from embodichain.lab.sim.cfg import MarkerCfg

    if isinstance(workspace_points, torch.Tensor):
        points = workspace_points.cpu().numpy()
    else:
        points = np.array(workspace_points)

    transforms = []
    for point in points:
        T = np.eye(4)
        T[:3, 3] = point
        transforms.append(T)

    cfg = MarkerCfg(
        name=marker_name,
        marker_type="axis",
        axis_xpos=transforms,
        axis_size=axis_size,
        axis_len=axis_len,
        arena_index=arena_index,
    )

    # 绘制标记
    markers = sim.draw_marker(cfg)

    print(f"✓ 绘制了 {len(points)} 个工作空间点标记")
    print(f"  标记名称: {marker_name}")
    print(f"  坐标轴长度: {axis_len}m")
    print(f"  坐标轴粗细: {axis_size}m")

    return markers


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    # Example usage
    from IPython import embed
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.robots.dexforce_w1.types import (
        DexforceW1HandBrand,
        DexforceW1ArmSide,
        DexforceW1ArmKind,
        DexforceW1Version,
    )
    from embodichain.lab.sim.robots.dexforce_w1.utils import build_dexforce_w1_cfg

    config = SimulationManagerCfg(headless=False, sim_device="cpu")
    sim = SimulationManager(config)
    sim.build_multiple_arenas(1)
    sim.set_manual_update(False)

    from embodichain.lab.sim.robots import DexforceW1Cfg

    cfg = DexforceW1Cfg.from_dict(
        {"uid": "dexforce_w1", "version": "v021", "arm_kind": "anthropomorphic"}
    )
    robot = sim.add_robot(cfg=cfg)
    print("DexforceW1 robot added to the simulation.")

    # Set left arm joint positions (mirrored)
    robot.set_qpos(
        qpos=[0, -np.pi / 4, 0.0, -np.pi / 2, -np.pi / 4, 0.0, 0.0],
        joint_ids=robot.get_joint_ids("left_arm"),
    )
    # Set right arm joint positions (mirrored)
    robot.set_qpos(
        qpos=[0, np.pi / 4, 0.0, np.pi / 2, np.pi / 4, 0.0, 0.0],
        joint_ids=robot.get_joint_ids("right_arm"),
    )

    # Example 1: Joint space analysis (default)
    print("\n" + "=" * 60)
    print("Example 1: Joint Space Analysis")
    print("=" * 60)
    wa_joint = WorkspaceAnalyzer(robot=robot)
    results_joint = wa_joint.analyze(num_samples=1000)
    print(f"\nJoint Space Results:")
    print(
        f"  Valid points: {results_joint['num_valid']} / {results_joint['num_samples']}"
    )
    print(f"  Analysis time: {results_joint['analysis_time']:.2f}s")
    print(f"  Metrics: {results_joint['metrics']}")

    # Example 2: Cartesian space analysis
    print("\n" + "=" * 60)
    print("Example 2: Cartesian Space Analysis")
    print("=" * 60)
    from embodichain.lab.sim.utility.workspace_analyzer import (
        WorkspaceAnalyzerConfig,
        AnalysisMode,
    )
    from embodichain.lab.sim.utility.workspace_analyzer.configs import (
        DimensionConstraint,
    )
    import numpy as np

    cartesian_config = WorkspaceAnalyzerConfig(
        mode=AnalysisMode.CARTESIAN_SPACE,
        constraint=DimensionConstraint(
            min_bounds=np.array([-0.4, -0.2, 0.7]),
            max_bounds=np.array([0.4, 0.8, 2.0]),
        ),
        ik_samples_per_point=1,
        ik_success_threshold=0.4,
    )
    wa_cartesian = WorkspaceAnalyzer(robot=robot, config=cartesian_config)
    results_cartesian = wa_cartesian.analyze(num_samples=500)
    print(f"\nCartesian Space Results:")
    print(
        f"  Reachable points: {results_cartesian['num_reachable']} / {results_cartesian['num_samples']}"
    )
    print(f"  Analysis time: {results_cartesian['analysis_time']:.2f}s")
    print(f"  Metrics: {results_cartesian['metrics']}")

    markers = draw_workspace_points(
        sim,
        results_cartesian["workspace_points"],
        marker_name="cartesian_workspace",
        axis_size=0.002,
        axis_len=0.005,
    )

    # Visualize (optional)
    # wa_joint.visualize(show=True)
    # wa_cartesian.visualize(show=True)

    embed(header="Workspace Analyzer Test Environment")
