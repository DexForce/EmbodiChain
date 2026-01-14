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

import time
import torch
import numpy as np

from tqdm import tqdm
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
from typing import List, Tuple, Dict, Any
import os
import sys

try:
    import psutil
except ImportError:
    psutil = None

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.objects.robot import Robot

from embodichain.lab.sim.utility.workspace_analyzer.configs import (
    CacheConfig,
    DimensionConstraint,
    SamplingConfig,
    VisualizationType,
    VisualizationConfig,
    MetricConfig,
)
from embodichain.lab.sim.utility.workspace_analyzer.samplers import (
    SamplerFactory,
    BaseSampler,
)
from embodichain.lab.sim.utility.workspace_analyzer.caches import CacheManager
from embodichain.lab.sim.utility.workspace_analyzer.constraints import (
    WorkspaceConstraintChecker,
)

from embodichain.utils import logger

__all__ = [
    "AnalysisMode",
    "WorkspaceAnalyzerConfig",
    "WorkspaceAnalyzer",
]


class AnalysisMode(Enum):
    """Workspace analysis mode."""

    JOINT_SPACE = "joint_space"
    """Sample in joint space, compute FK to get workspace points."""

    CARTESIAN_SPACE = "cartesian_space"
    """Sample in Cartesian space, compute IK to verify reachability."""

    PLANE_SAMPLING = "plane_sampling"
    """Sample on a specific plane within Cartesian space."""


@dataclass
class WorkspaceAnalyzerConfig:
    """Complete configuration for workspace analyzer."""

    mode: AnalysisMode = AnalysisMode.JOINT_SPACE
    """Analysis mode: joint space or Cartesian space sampling."""

    sampling: SamplingConfig = None
    """Sampling configuration."""
    cache: CacheConfig = None
    """Cache configuration."""
    constraint: DimensionConstraint = None
    """Dimension constraint configuration."""
    visualization: VisualizationConfig = None
    """Visualization configuration."""
    metric: MetricConfig = None
    """Metric configuration."""

    ik_samples_per_point: int = 1
    """For Cartesian mode: number of random joint seeds to try for each Cartesian point."""
    reference_pose: Any | None = None
    """Optional reference pose (4x4 matrix) for IK target orientation. If None, uses current robot pose."""
    control_part_name: str | None = None
    """Name of the control part (e.g., 'left_arm', 'right_arm'). If None, uses the default solver or first available control part."""

    # Plane sampling parameters
    enable_plane_sampling: bool = False
    """Whether to enable plane sampling functionality (uses existing samplers directly)"""

    plane_normal: torch.Tensor | None = None
    """Normal vector of the plane for plane sampling [nx, ny, nz]"""

    plane_point: torch.Tensor | None = None
    """A point on the plane for plane sampling [x, y, z]"""

    plane_bounds: torch.Tensor | None = None
    """Bounds for 2D plane coordinates [[u_min, u_max], [v_min, v_max]]"""

    # Geometric constraint parameters for sampling
    constraint_type: str | None = None
    """Type of geometric constraint: 'box', 'sphere', None. If None, no constraint applied."""

    constraint_bounds: torch.Tensor | None = None
    """Bounds for constraint: For box: [[x_min, x_max], [y_min, y_max], ...]. For sphere: used to auto-calculate radius if sphere_radius is None."""

    sphere_center: torch.Tensor | None = None
    """Center point for sphere constraint [x, y, z, ...]. If None and constraint_type='sphere', calculated from constraint_bounds."""

    sphere_radius: float | None = None
    """Radius for sphere constraint. If None and constraint_type='sphere', auto-calculated from constraint_bounds."""

    sphere_radius_mode: str = "inscribed"
    """Mode for auto-calculating sphere radius from bounds: 'inscribed' or 'circumscribed'. Only used if sphere_radius is None."""

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

    Note:
        Currently designed for single environment operation (num_envs=1). When multiple
        environments are present, the analyzer will use the first environment (index 0)
        and log appropriate warnings. Multi-environment support will be added in future versions.
    """

    # Default priority order for control parts selection
    DEFAULT_CONTROL_PART_PRIORITY = ["left_arm", "right_arm"]

    def __init__(
        self,
        robot: Robot,
        config: WorkspaceAnalyzerConfig | None = None,
        sim_manager: SimulationManager | None = None,
    ):
        """Initialize the workspace analyzer.

        Args:
            robot: Robot instance to analyze.
            config: Configuration object. If None, uses defaults.
            sim_manager: SimulationManager instance. Defaults None.
        """
        self.robot = robot
        self.config = config or WorkspaceAnalyzerConfig()
        self.sim_manager = sim_manager

        # Check multi-environment compatibility and add protection
        self._check_num_envs_compatibility()

        # Use sim_manager's device if available, otherwise default to CPU
        self.device = (
            sim_manager.device if sim_manager is not None else torch.device("cpu")
        )

        # Determine control part name from config
        self.control_part_name = self._determine_control_part(
            self.config.control_part_name
        )

        # Extract joint limits from robot
        self._setup_joint_limits()

        # Initialize components
        self.sampler = self._create_sampler()
        self.cache = self._create_cache()
        self.constraint_checker = self._create_constraint_checker()

        # Storage for analysis results
        self.workspace_points: torch.Tensor | None = None
        self.joint_configurations: torch.Tensor | None = None
        self.metrics_results: Dict[str, Any] = {}
        self.current_mode: AnalysisMode | None = None
        self.success_rates: torch.Tensor | None = None

    def _determine_control_part(self, control_part_name: str | None) -> str | None:
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
                # Try priority parts first
                for part in self.DEFAULT_CONTROL_PART_PRIORITY:
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

    def _check_num_envs_compatibility(self) -> None:
        """Check multi-environment compatibility and provide appropriate warnings.

        WorkspaceAnalyzer is currently designed for num_envs=1. If multiple environments
        are present, it will use the first environment and log appropriate warnings.
        """
        if self.sim_manager is None:
            # No sim_manager provided, cannot check num_envs
            logger.log_debug(
                "No SimulationManager provided, assuming single environment setup"
            )
            return

        num_envs = self.sim_manager.num_envs

        if num_envs == 1:
            logger.log_debug(
                "WorkspaceAnalyzer initialized with single environment (num_envs=1)"
            )
        else:
            logger.log_warning(
                f"WorkspaceAnalyzer is currently designed for single environment operation (num_envs=1), "
                f"but {num_envs} environments detected. Will use the first environment (index 0) for analysis. "
                f"Multi-environment support will be added in future versions."
            )
            logger.log_info("Using environment index 0 for workspace analysis")

    def _setup_joint_limits(self) -> None:
        """Extract and setup joint limits from the robot."""
        # Get all joint limits from robot (using first entity/environment)
        all_joint_limits = self.robot.body_data.qpos_limits[0]

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

    # Note: Geometric constraint creation methods temporarily removed

    # Note: Explicit constraint creation temporarily removed

    def _compute_dynamic_workspace_bounds(self) -> torch.Tensor:
        """Compute workspace bounds dynamically from joint space FK.

        Returns:
            Tensor of shape (3, 2) representing [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        """
        logger.log_info("Computing workspace bounds dynamically from joint space FK...")

        # Create a temporary sampler without constraints for initial FK computation
        from embodichain.lab.sim.utility.workspace_analyzer.samplers import (
            RandomSampler,
        )

        temp_sampler = RandomSampler(seed=self.config.sampling.seed)

        # Sample joint space to compute FK bounds
        joint_samples = temp_sampler.sample(num_samples=1000, bounds=self.qpos_limits)

        # Compute FK for all samples with progress tracking
        workspace_pts_list = []

        pbar = self._create_optimized_tqdm(
            range(len(joint_samples)),
            desc="Computing Workspace Bounds (FK)",
            unit="cfg",
            color="cyan",
            emoji="📏",
        )

        successful_fk = 0
        for i in pbar:
            qpos = joint_samples[i : i + 1]  # Keep batch dimension
            try:
                pose = self.robot.compute_fk(
                    qpos=qpos,
                    name=self.control_part_name,
                    to_matrix=True,
                )
                position = pose[:, :3, 3]  # Extract position
                workspace_pts_list.append(position)
                successful_fk += 1
            except Exception:
                continue

            # Update progress bar with success rate
            self._update_progress_with_stats(
                pbar, i, successful_fk, metric_name="FK success", show_rate=True
            )

        if workspace_pts_list:
            workspace_pts = torch.cat(workspace_pts_list, dim=0)
            # Compute min/max bounds for each dimension
            min_bounds = workspace_pts.min(dim=0).values
            max_bounds = workspace_pts.max(dim=0).values

            # Add margin (10%)
            margin = (max_bounds - min_bounds) * 0.1
            min_bounds = min_bounds - margin
            max_bounds = max_bounds + margin

            # Create bounds tensor: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            bounds = torch.stack([min_bounds, max_bounds], dim=1)

            logger.log_info(
                f"Computed workspace bounds from {len(workspace_pts)} FK samples:\n"
                f"\t X: [{min_bounds[0]:.3f}, {max_bounds[0]:.3f}] m\n"
                f"\t Y: [{min_bounds[1]:.3f}, {max_bounds[1]:.3f}] m\n"
                f"\t Z: [{min_bounds[2]:.3f}, {max_bounds[2]:.3f}] m"
            )
            return bounds
        else:
            # Fallback to default bounds if FK computation fails
            logger.log_warning("FK computation failed, using fallback bounds")
            return torch.tensor(
                [[-1.0, 1.0], [-1.0, 1.0], [0.0, 2.0]], device=self.device
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

    def _create_optimized_tqdm(
        self, iterable, desc: str, unit: str, color: str = "blue", emoji: str = "⚡"
    ):
        """Create an optimized tqdm progress bar with adaptive updates and smart formatting.

        Args:
            iterable: The iterable to track progress for
            desc: Description text
            unit: Unit name (e.g., 'cfg', 'pt')
            color: Progress bar color
            emoji: Emoji for the description

        Returns:
            Configured tqdm instance
        """
        total = len(iterable) if hasattr(iterable, "__len__") else None

        # Adaptive parameters based on total count
        if total:
            if total < 100:
                mininterval, maxinterval = 0.1, 1.0
                smoothing = 0.1
            elif total < 1000:
                mininterval, maxinterval = 0.2, 2.0
                smoothing = 0.05
            else:
                mininterval, maxinterval = 0.5, 5.0
                smoothing = 0.02
        else:
            mininterval, maxinterval = 0.5, 5.0
            smoothing = 0.05

        # Terminal width detection
        try:
            terminal_width = os.get_terminal_size().columns
            ncols = min(120, max(80, terminal_width - 10))
        except OSError:
            # Terminal size unavailable (e.g., non-terminal environment)
            ncols = 100

        # Color codes for different states
        color_codes = {
            "blue": "\033[34m",
            "cyan": "\033[36m",
            "magenta": "\033[35m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "red": "\033[31m",
        }

        # Enhanced bar format with better spacing
        bar_format = (
            f"{color_codes.get(color, '')}{{desc}}\033[0m: "
            f"{{percentage:3.0f}}%|{{bar}}| {{n_fmt}}/{{total_fmt}} "
            f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
        )

        # Performance-aware tqdm configuration
        pbar = tqdm(
            iterable,
            desc=f"{emoji} {desc}",
            unit=unit,
            unit_scale=True,
            smoothing=smoothing,
            mininterval=mininterval,
            maxinterval=maxinterval,
            bar_format=bar_format,
            ncols=ncols,
            dynamic_ncols=True,
            colour=color,
            leave=True,
            ascii=False if sys.stdout.encoding == "utf-8" else True,
            # Advanced features
            position=0,  # Top position for multiple bars
            file=sys.stdout,
            disable=False,
        )

        # Add performance tracking attributes
        pbar._start_time = time.time()
        pbar._last_update = 0
        pbar._update_count = 0

        return pbar

    def _update_progress_with_stats(
        self,
        pbar,
        current_idx: int,
        success_count: int,
        metric_name: str = "success",
        show_rate: bool = True,
    ):
        """Update progress bar with intelligent statistics and color coding.

        Args:
            pbar: tqdm progress bar instance
            current_idx: Current iteration index
            success_count: Number of successful operations
            metric_name: Name of the metric being tracked
            show_rate: Whether to show the success rate
        """
        if not show_rate:
            return

        total_processed = current_idx + 1
        rate = (success_count / total_processed) * 100 if total_processed > 0 else 0

        # Intelligent color coding with thresholds
        if rate >= 85:
            color, icon = "\033[92m", "🟢"  # Bright green, excellent
        elif rate >= 70:
            color, icon = "\033[32m", "✅"  # Green, good
        elif rate >= 50:
            color, icon = "\033[93m", "🟡"  # Bright yellow, moderate
        elif rate >= 30:
            color, icon = "\033[33m", "🟠"  # Yellow, low
        else:
            color, icon = "\033[91m", "🔴"  # Bright red, poor

        # Adaptive display with performance metrics
        current_time = time.time()

        # Smart update throttling based on performance
        if hasattr(pbar, "_last_update") and hasattr(pbar, "_update_count"):
            time_since_last = current_time - pbar._last_update
            pbar._update_count += 1

            # Adaptive update frequency based on processing speed
            if pbar._update_count > 100:  # After first 100 updates
                avg_time_per_update = time_since_last / max(
                    1,
                    (
                        pbar._update_count - pbar._last_update_count
                        if hasattr(pbar, "_last_update_count")
                        else 1
                    ),
                )
                if avg_time_per_update < 0.01:  # Very fast processing
                    update_threshold = 0.5  # Update every 0.5s
                elif avg_time_per_update < 0.1:  # Medium speed
                    update_threshold = 0.3  # Update every 0.3s
                else:  # Slow processing
                    update_threshold = 0.1  # Update every 0.1s

                if time_since_last < update_threshold:
                    return  # Skip update to reduce overhead

            pbar._last_update = current_time
            pbar._last_update_count = pbar._update_count

        # Enhanced display with ETA and throughput
        if total_processed < 10:
            # Show individual counts for small numbers
            stats = f" {icon} {success_count}/{total_processed}"
        else:
            # Show percentage and throughput for larger numbers
            if hasattr(pbar, "_start_time"):
                elapsed = current_time - pbar._start_time
                throughput = total_processed / elapsed if elapsed > 0 else 0
                if throughput > 10:
                    stats = f" {icon} {color}{rate:.1f}%\033[0m {metric_name} ({throughput:.0f}/s)"
                else:
                    stats = f" {icon} {color}{rate:.1f}%\033[0m {metric_name} ({throughput:.1f}/s)"
            else:
                stats = f" {icon} {color}{rate:.1f}%\033[0m {metric_name}"

        pbar.set_postfix_str(stats, refresh=False)

    def sample_joint_space(self, num_samples: int | None = None) -> torch.Tensor:
        """Sample joint configurations within joint limits.

        Args:
            num_samples: Number of samples to generate. If None, uses config value.

        Returns:
            Tensor of shape (num_samples, num_joints) containing joint configurations.
        """
        num_samples = num_samples or self.config.sampling.num_samples

        # Performance-aware sampling with progress indication
        start_time = time.time()

        # Sample from joint space
        joint_samples = self.sampler.sample(
            bounds=self.qpos_limits, num_samples=num_samples
        )

        sampling_time = time.time() - start_time
        samples_per_sec = (
            num_samples / sampling_time if sampling_time > 0 else float("inf")
        )

        logger.log_info(
            f"Generated {num_samples} joint space samples "
            f"({samples_per_sec:.0f} samples/s)"
        )
        return joint_samples

    def sample_cartesian_space(self, num_samples: int | None = None) -> torch.Tensor:
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
            # Compute bounds from joint space FK using dedicated method
            logger.log_info(
                "No Cartesian bounds specified, computing from joint space..."
            )
            cartesian_bounds = self._compute_dynamic_workspace_bounds()

        # Sample from Cartesian space using bounds
        cartesian_samples = self.sampler.sample(
            bounds=cartesian_bounds, num_samples=num_samples
        )

        # Check how many samples pass workspace constraints
        valid_bounds = self.constraint_checker.check_bounds(cartesian_samples)
        valid_collision = self.constraint_checker.check_collision(cartesian_samples)
        valid_constraints = valid_bounds & valid_collision

        constraint_pass_rate = valid_constraints.sum().item() / num_samples * 100
        exclude_zones_count = self.constraint_checker.get_num_exclude_zones()

        logger.log_info(
            f"Generated {num_samples} Cartesian space samples. "
            f"Constraint check: {valid_constraints.sum()}/{num_samples} "
            f"({constraint_pass_rate:.1f}%) pass bounds+collision constraints "
            f"({exclude_zones_count} exclude zones configured)"
        )
        return cartesian_samples

    def sample_plane(
        self,
        num_samples: int | None = None,
        plane_normal: torch.Tensor | None = None,
        plane_point: torch.Tensor | None = None,
        plane_bounds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample points on a specified plane using existing samplers (ultra-simplified version).

        Args:
            num_samples: Number of samples to generate. If None, uses config value.
            plane_normal: Plane normal vector [nx, ny, nz]. Defaults to [0,0,1] (XY plane).
            plane_point: A point on the plane [x, y, z]. Defaults to [0,0,0].
            plane_bounds: 2D bounds [[u_min, u_max], [v_min, v_max]]. Defaults to [[-1,1], [-1,1]].

        Returns:
            Tensor of shape (num_samples, 3) containing 3D points on the plane.
        """
        num_samples = num_samples or self.config.sampling.num_samples

        # Set default values
        if plane_normal is None:
            plane_normal = torch.tensor([0.0, 0.0, 1.0], device=self.device)  # XY plane
        else:
            plane_normal = plane_normal.to(self.device) / torch.norm(
                plane_normal.to(self.device)
            )

        if plane_point is None:
            plane_point = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        else:
            plane_point = plane_point.to(self.device)

        if plane_bounds is None:
            # Compute dynamic workspace bounds from joint space FK
            dynamic_bounds_3d = self._compute_dynamic_workspace_bounds()

            # Project 3D bounds to 2D plane coordinate system
            plane_bounds = self._compute_plane_bounds_from_3d(
                dynamic_bounds_3d, plane_normal, plane_point
            )

            logger.log_info(
                f"Using dynamic plane bounds computed from FK: "
                f"U: [{plane_bounds[0, 0]:.3f}, {plane_bounds[0, 1]:.3f}], "
                f"V: [{plane_bounds[1, 0]:.3f}, {plane_bounds[1, 1]:.3f}] "
                f"(projected to plane with normal {plane_normal.cpu().numpy()})"
            )
        else:
            plane_bounds = plane_bounds.to(self.device)

        # Generate 2D samples and convert to 3D
        plane_samples_2d = self.sampler.sample(num_samples, bounds=plane_bounds)
        plane_samples_3d = self._plane_to_world_optimized(
            plane_samples_2d, plane_normal, plane_point
        )

        logger.log_info(
            f"Generated {num_samples} plane samples using {self.sampler.get_strategy_name()}"
        )

        return plane_samples_3d

    def _project_to_plane(
        self,
        points_3d: torch.Tensor,
        plane_normal: torch.Tensor,
        plane_point: torch.Tensor,
    ) -> torch.Tensor:
        """Project 3D points onto a specified plane.

        Args:
            points_3d: 3D points to project, shape (num_samples, 3)
            plane_normal: Normal vector of the plane [nx, ny, nz]
            plane_point: A point on the plane [x, y, z]

        Returns:
            Projected 3D points on the plane, shape (num_samples, 3)
        """
        # Normalize the plane normal
        plane_normal = plane_normal / torch.norm(plane_normal)

        # Vector from plane_point to each 3D point
        vectors_to_points = points_3d - plane_point.unsqueeze(0)

        # Project vectors onto plane normal (signed distance from plane)
        distances = torch.sum(vectors_to_points * plane_normal.unsqueeze(0), dim=1)

        # Project points onto plane by subtracting the normal component
        projected_points = points_3d - distances.unsqueeze(1) * plane_normal.unsqueeze(
            0
        )

        return projected_points

    def _plane_to_world_optimized(
        self,
        plane_coords: torch.Tensor,
        plane_normal: torch.Tensor,
        plane_point: torch.Tensor,
    ) -> torch.Tensor:
        """Convert 2D plane coordinates to 3D world coordinates with optimized basis generation.

        This method uses a more numerically stable approach to generate orthogonal basis vectors
        and supports orientation optimization for better workspace coverage.

        Args:
            plane_coords: 2D coordinates on the plane, shape (num_samples, 2)
            plane_normal: Normal vector of the plane [nx, ny, nz]
            plane_point: A point on the plane [x, y, z]

        Returns:
            3D world coordinates, shape (num_samples, 3)
        """
        num_samples = plane_coords.shape[0]

        # Generate orthogonal basis vectors using improved method
        u, v = self._generate_orthogonal_basis(plane_normal)

        # Convert 2D plane coordinates to 3D with vectorized operations
        world_coords = (
            plane_point.unsqueeze(0)
            + plane_coords[:, 0:1]  # Base point broadcast to all samples
            * u.unsqueeze(0)
            + plane_coords[:, 1:2]  # First plane direction
            * v.unsqueeze(0)  # Second plane direction
        )

        return world_coords

    def _generate_orthogonal_basis(
        self, plane_normal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate orthogonal basis vectors for a plane with improved numerical stability.

        This method uses the most stable approach based on the plane normal direction
        and optionally optimizes orientation for workspace coverage.

        Args:
            plane_normal: Normal vector of the plane [nx, ny, nz]

        Returns:
            Tuple of two orthogonal unit vectors (u, v) that span the plane
        """
        # Find the coordinate with smallest absolute value for numerical stability
        abs_normal = torch.abs(plane_normal)
        min_idx = torch.argmin(abs_normal)

        # Create an arbitrary vector with 1 in the most stable coordinate
        arbitrary = torch.zeros_like(plane_normal)
        arbitrary[min_idx] = 1.0

        # Generate first tangent vector using Gram-Schmidt
        u = arbitrary - torch.dot(arbitrary, plane_normal) * plane_normal
        u = u / torch.norm(u)

        # Generate second tangent vector via cross product
        v = torch.linalg.cross(plane_normal, u)
        v = v / torch.norm(v)

        return u, v

    def _compute_plane_bounds_from_3d(
        self,
        bounds_3d: torch.Tensor,
        plane_normal: torch.Tensor,
        plane_point: torch.Tensor,
    ) -> torch.Tensor:
        """Project 3D workspace bounds to 2D plane coordinate system.

        Args:
            bounds_3d: 3D workspace bounds, shape (3, 2) representing
                      [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            plane_normal: Normal vector of the plane [nx, ny, nz]
            plane_point: A point on the plane [x, y, z]

        Returns:
            2D plane bounds, shape (2, 2) representing [[u_min, u_max], [v_min, v_max]]
        """
        # Generate orthogonal basis vectors for the plane
        u, v = self._generate_orthogonal_basis(plane_normal)

        # Create all 8 corners of the 3D bounding box
        corners_3d = []
        for x in [bounds_3d[0, 0], bounds_3d[0, 1]]:  # x_min, x_max
            for y in [bounds_3d[1, 0], bounds_3d[1, 1]]:  # y_min, y_max
                for z in [bounds_3d[2, 0], bounds_3d[2, 1]]:  # z_min, z_max
                    corners_3d.append(torch.tensor([x, y, z], device=self.device))

        corners_3d = torch.stack(corners_3d)  # Shape: (8, 3)

        # Project all corners to the plane coordinate system
        # Transform from world coordinates to plane coordinates
        corners_relative = corners_3d - plane_point.unsqueeze(
            0
        )  # Relative to plane point

        # Project onto plane basis vectors
        u_coords = torch.sum(corners_relative * u.unsqueeze(0), dim=1)  # Shape: (8,)
        v_coords = torch.sum(corners_relative * v.unsqueeze(0), dim=1)  # Shape: (8,)

        # Find min/max in both plane directions
        u_min, u_max = u_coords.min(), u_coords.max()
        v_min, v_max = v_coords.min(), v_coords.max()

        # Create 2D plane bounds
        plane_bounds = torch.tensor(
            [[u_min, u_max], [v_min, v_max]], device=self.device
        )

        return plane_bounds

    def _get_robot_base_position(self) -> torch.Tensor:
        """Get the robot base position as default plane point.

        Returns:
            Robot base position as a 3D tensor
        """
        try:
            # Try to get current robot pose (using first environment)
            current_pose = self.robot.compute_fk(
                qpos=self.robot.get_qpos()[None, :],  # Add batch dimension
                name=self.control_part_name,
                to_matrix=True,
            )
            # Use current end-effector position projected to a reasonable height
            base_pos = current_pose[0, :3, 3].clone()
            base_pos[2] = 0.0  # Project to ground plane
            return base_pos
        except Exception:
            # Fallback to origin
            return torch.tensor([0.0, 0.0, 0.0], device=self.device)

    def _generate_plane_samples(
        self,
        plane_bounds: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Generate 2D plane samples using existing base sampler directly."""
        return self.sampler.sample(num_samples, bounds=plane_bounds)

    def compute_workspace_points(
        self, joint_configs: torch.Tensor, batch_size: int | None = None
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
        pbar = self._create_optimized_tqdm(
            range(num_samples),
            desc="Forward Kinematics",
            unit="cfg",
            color="cyan",
            emoji="🤖",
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

                # Filter by constraints (bounds + collision check)
                valid_bounds = self.constraint_checker.check_bounds(position)
                valid_collision = self.constraint_checker.check_collision(position)
                valid_mask = valid_bounds & valid_collision

                # Store valid results
                if valid_mask.any():
                    workspace_points_list.append(position[valid_mask])
                    valid_configs_list.append(qpos[valid_mask])
                    total_valid += 1

                # Update progress bar with intelligent statistics
                self._update_progress_with_stats(
                    pbar, i, total_valid, metric_name="valid", show_rate=True
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

        # Performance summary for FK computation
        pbar.close()  # Ensure progress bar is closed
        success_rate = len(workspace_points) / num_samples * 100

        # Performance indicator based on success rate
        if success_rate >= 90:
            perf_icon = "🏆"  # Trophy for excellent performance
        elif success_rate >= 75:
            perf_icon = "✅"  # Check mark for good performance
        elif success_rate >= 50:
            perf_icon = "🟡"  # Yellow circle for moderate performance
        else:
            perf_icon = "⚠️"  # Warning for low performance

        logger.log_info(
            f"{perf_icon} FK Results: {len(workspace_points)}/{num_samples} valid points "
            f"({success_rate:.1f}% success rate)"
        )

        return workspace_points, valid_configs

    def compute_reachability(
        self, cartesian_points: torch.Tensor, batch_size: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute reachability for Cartesian points using IK.

        Args:
            cartesian_points: Cartesian positions, shape (num_samples, 3).
            batch_size: Batch size for IK computation. If None, uses config value.

        Returns:
            Tuple of:
                - all_points: All Cartesian positions, shape (num_samples, 3)
                - reachable_points: Reachable positions, shape (num_reachable, 3)
                - success_rates: IK success rate for each point, shape (num_samples,)
                - reachability_mask: Boolean mask indicating reachable points, shape (num_samples,)
                - best_configs: Best joint configurations, shape (num_reachable, num_joints)
        """
        num_samples = len(cartesian_points)
        ik_samples_per_point = self.config.ik_samples_per_point

        # Pre-filter Cartesian points by workspace constraints
        # This eliminates points that are outside bounds or in collision zones
        valid_cartesian_mask = self.constraint_checker.check_bounds(
            cartesian_points
        ) & self.constraint_checker.check_collision(cartesian_points)

        logger.log_info(
            f"Pre-filtered Cartesian points: {valid_cartesian_mask.sum()}/{num_samples} "
            f"points pass workspace constraints ({(valid_cartesian_mask.sum()/num_samples*100):.1f}%)"
        )

        # Store results for all points (including invalid ones for consistent indexing)
        all_success_rates = torch.zeros(num_samples, device=self.device)
        reachable_points_list = []
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

        # Get reference end-effector pose for IK target orientation
        # Priority: use reference_pose if provided, otherwise compute from current joint configuration
        if (
            hasattr(self.config, "reference_pose")
            and self.config.reference_pose is not None
        ):
            # Use provided reference pose (should be 4x4 transformation matrix)
            reference_pose = self.config.reference_pose
            if isinstance(reference_pose, np.ndarray):
                reference_pose = torch.from_numpy(reference_pose).to(self.device)
            if reference_pose.dim() == 2:  # Shape: (4, 4) -> (1, 4, 4)
                reference_pose = reference_pose.unsqueeze(0)
            current_ee_pose = reference_pose  # Shape: (1, 4, 4)
            logger.log_info("Using provided reference pose for IK target orientation")
        else:
            # Fallback: compute current end-effector pose from joint configuration
            try:
                # Using first environment (index 0) for qpos retrieval
                current_qpos = self.robot.get_qpos()[0][
                    self.robot.get_joint_ids(self.control_part_name)
                ]
                current_ee_pose = self.robot.compute_fk(
                    name=self.control_part_name,
                    qpos=current_qpos.unsqueeze(0),
                    to_matrix=True,
                )  # Shape: (1, 4, 4)
                logger.log_info(
                    "Computing reference pose from current robot configuration"
                )
            except Exception as e:
                logger.log_warning(f"Failed to compute current robot pose: {e}")
                # Create identity pose as fallback
                current_ee_pose = torch.eye(4, device=self.device).unsqueeze(0)
                current_ee_pose[0, :3, 3] = torch.tensor(
                    [0.5, 0.0, 1.0], device=self.device
                )  # Default position
                logger.log_info("Using default identity pose as fallback")

            # Print current joint configuration and computed pose
            pose_np = current_ee_pose[0].cpu().numpy()
            position = pose_np[:3, 3]
            rotation_matrix = pose_np[:3, :3]

            # Convert rotation matrix to Euler angles
            import scipy.spatial.transform as spt

            euler_angles = spt.Rotation.from_matrix(rotation_matrix).as_euler(
                "xyz", degrees=True
            )

        # Print detailed reference pose information
        pose_np = current_ee_pose[0].cpu().numpy()
        position = pose_np[:3, 3]
        rotation_matrix = pose_np[:3, :3]

        # Convert rotation matrix to Euler angles (ZYX convention)
        import scipy.spatial.transform as spt

        euler_angles = spt.Rotation.from_matrix(rotation_matrix).as_euler(
            "xyz", degrees=True
        )

        # Format matrix with proper indentation
        matrix_lines = np.array2string(pose_np, precision=4, suppress_small=True).split(
            "\n"
        )
        matrix_str = "\n".join(f"\t   {line}" for line in matrix_lines)
        logger.log_info(
            f"🎯 Using provided reference pose for IK target orientation:\n"
            f"\t Position: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}] m\n"
            f"\t Rotation (XYZ Euler): [{euler_angles[0]:.2f}°, {euler_angles[1]:.2f}°, {euler_angles[2]:.2f}°]\n"
            f"\t Matrix:\n{matrix_str}"
        )

        # Track statistics for progress bar
        total_reachable = 0

        # Process each point individually (robot expects batch_size from environments, not samples)
        pbar = self._create_optimized_tqdm(
            range(num_samples),
            desc="Inverse Kinematics",
            unit="pt",
            color="magenta",
            emoji="🎯",
        )

        for i in pbar:
            position = cartesian_points[i]  # Shape: (3,)

            # Skip points that don't satisfy workspace constraints
            if not valid_cartesian_mask[i]:
                # Mark as unreachable due to constraint violation
                all_success_rates[i] = 0.0
                # Update progress bar
                reachability_rate = total_reachable / (i + 1) * 100
                if reachability_rate >= 70:
                    reach_color = "\033[32m"  # Green for high reachability
                elif reachability_rate >= 40:
                    reach_color = "\033[33m"  # Yellow for medium reachability
                else:
                    reach_color = "\033[31m"  # Red for low reachability
                pbar.set_postfix_str(
                    f"🎯 Reachable: {total_reachable}/{i+1} | {reach_color}{reachability_rate:.1f}%\033[0m rate (❌ constraint)"
                )
                continue

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
            all_success_rates[i] = success_rate

            # Filter by success threshold for reachable points
            if success_rate and best_qpos is not None:
                reachable_points_list.append(position.unsqueeze(0))  # Add batch dim
                best_configs_list.append(best_qpos.unsqueeze(0))  # Add batch dim
                total_reachable += 1

            # Update progress bar with reachability statistics
            reachability_rate = total_reachable / (i + 1) * 100
            # Use color coding for the reachability rate
            if reachability_rate >= 70:
                reach_color = "\033[32m"  # Green for high reachability
            elif reachability_rate >= 40:
                reach_color = "\033[33m"  # Yellow for medium reachability
            else:
                reach_color = "\033[31m"  # Red for low reachability

            # Add success rate indicator for this specific point
            if success_rate:
                point_status = "✅ IK"
            elif success_rate > 0:
                point_status = f"🟡 IK({success_rate:.1f})"
            else:
                point_status = "❌ IK"

            pbar.set_postfix_str(
                f"🎯 Reachable: {total_reachable}/{i+1} | {reach_color}{reachability_rate:.1f}%\033[0m rate | {point_status}"
            )

        # Concatenate reachable results
        if reachable_points_list:
            reachable_points = torch.cat(reachable_points_list, dim=0)
            best_configs = torch.cat(best_configs_list, dim=0)
        else:
            reachable_points = torch.empty((0, 3), device=self.device)
            best_configs = torch.empty((0, self.num_joints), device=self.device)

        # Create reachability mask
        reachability_mask = all_success_rates > 0

        # Performance summary for IK computation
        pbar.close()  # Ensure progress bar is closed
        reachability = len(reachable_points) / num_samples * 100

        # Reachability performance indicator
        if reachability >= 80:
            reach_icon = "🏆"  # Trophy for high reachability
        elif reachability >= 60:
            reach_icon = "🚀"  # Rocket for good reachability
        elif reachability >= 40:
            reach_icon = "🟡"  # Yellow for moderate reachability
        elif reachability >= 20:
            reach_icon = "🟠"  # Orange for low reachability
        else:
            reach_icon = "⚠️"  # Warning for very low reachability

        logger.log_info(
            f"{reach_icon} IK Results: {len(reachable_points)}/{num_samples} reachable points "
            f"({reachability:.1f}% reachability)"
        )

        return (
            cartesian_points,
            reachable_points,
            all_success_rates,
            reachability_mask,
            best_configs,
        )

    def analyze(
        self,
        num_samples: int | None = None,
        force_recompute: bool = False,
        visualize: bool = False,
    ) -> Dict[str, Any]:
        """Perform complete workspace analysis.

        Args:
            num_samples: Number of samples to generate. If None, uses config value.
            force_recompute: If True, recomputes even if cached results exist.
            visualize: If True, visualizes the workspace points. Prefers sim_manager visualization
                      if available, otherwise falls back to visualizers module.

        Returns:
            Dictionary containing analysis results.
        """
        logger.log_info("Starting Workspace Analysis...")

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
            logger.log_info(f"Mode: {AnalysisMode.JOINT_SPACE.value}")

            # Step 1: Sample joint space
            logger.log_info("[1/3] Sampling joint space...")
            joint_configs = self.sample_joint_space(num_samples)

            # Step 2: Compute workspace points
            logger.log_info("[2/3] Computing workspace points via computing FK...")
            workspace_points, valid_configs = self.compute_workspace_points(
                joint_configs
            )

            # Store results
            self.workspace_points = workspace_points
            self.joint_configurations = valid_configs
            self.current_mode = AnalysisMode.JOINT_SPACE
            self.success_rates = None  # All points are reachable in joint space mode

            # Add constraint check statistics
            constraint_stats = self._compute_constraint_statistics(workspace_points)

            results = {
                "mode": AnalysisMode.JOINT_SPACE.value,
                "workspace_points": workspace_points,
                "joint_configurations": valid_configs,
                "num_samples": num_samples or self.config.sampling.num_samples,
                "num_valid": len(workspace_points),
                "constraint_statistics": constraint_stats,
            }

        elif self.config.mode == AnalysisMode.CARTESIAN_SPACE:
            # Cartesian space mode: Sample Cartesian → IK → Verify reachability
            logger.log_info(f"Mode: {AnalysisMode.CARTESIAN_SPACE.value}")

            # Step 1: Sample Cartesian space
            logger.log_info("[1/3] Sampling Cartesian space...")
            cartesian_samples = self.sample_cartesian_space(num_samples)

            # Step 2: Compute reachability via IK
            logger.log_info("[2/3] Computing reachability via computing IK...")
            (
                all_points,
                reachable_points,
                success_rates,
                reachability_mask,
                best_configs,
            ) = self.compute_reachability(cartesian_samples)

            # Store results - now storing all points for visualization
            self.workspace_points = all_points  # Store all sampled points
            self.reachable_points = reachable_points  # Store only reachable points
            self.joint_configurations = best_configs
            self.current_mode = AnalysisMode.CARTESIAN_SPACE
            self.success_rates = success_rates  # Store success rates for all points
            self.reachability_mask = reachability_mask  # Store reachability mask

            # Add constraint check statistics for both all_points and reachable_points
            constraint_stats_all = self._compute_constraint_statistics(all_points)
            constraint_stats_reachable = (
                self._compute_constraint_statistics(reachable_points)
                if len(reachable_points) > 0
                else {}
            )

            results = {
                "mode": AnalysisMode.CARTESIAN_SPACE.value,
                "all_points": all_points,  # All sampled Cartesian points
                "workspace_points": all_points,  # For compatibility
                "reachable_points": reachable_points,  # Only reachable points
                "joint_configurations": best_configs,
                "success_rates": success_rates,
                "reachability_mask": reachability_mask,
                "num_samples": num_samples or self.config.sampling.num_samples,
                "num_reachable": len(reachable_points),
                "constraint_statistics": {
                    "all_points": constraint_stats_all,
                    "reachable_points": constraint_stats_reachable,
                },
            }

        elif self.config.mode == AnalysisMode.PLANE_SAMPLING:
            # Plane sampling mode: Sample on plane → IK → Verify reachability
            logger.log_info(f"Mode: {AnalysisMode.PLANE_SAMPLING.value}")

            # Step 1: Sample on specified plane
            logger.log_info("[1/3] Sampling on specified plane...")
            cartesian_samples = self.sample_plane(
                num_samples=num_samples,
                plane_normal=self.config.plane_normal,
                plane_point=self.config.plane_point,
                plane_bounds=self.config.plane_bounds,
            )

            # Step 2: Compute reachability via IK
            logger.log_info("[2/3] Computing reachability via computing IK...")
            (
                all_points,
                reachable_points,
                success_rates,
                reachability_mask,
                best_configs,
            ) = self.compute_reachability(cartesian_samples)

            # Store results
            self.workspace_points = all_points
            self.reachable_points = reachable_points
            self.joint_configurations = best_configs
            self.current_mode = AnalysisMode.PLANE_SAMPLING
            self.success_rates = success_rates
            self.reachability_mask = reachability_mask

            # Add constraint check statistics
            constraint_stats_all = self._compute_constraint_statistics(all_points)
            constraint_stats_reachable = (
                self._compute_constraint_statistics(reachable_points)
                if len(reachable_points) > 0
                else {}
            )

            results = {
                "mode": AnalysisMode.PLANE_SAMPLING.value,
                "all_points": all_points,  # All sampled plane points
                "workspace_points": all_points,  # For compatibility
                "reachable_points": reachable_points,  # Only reachable points
                "joint_configurations": best_configs,
                "success_rates": success_rates,
                "reachability_mask": reachability_mask,
                "num_samples": len(cartesian_samples),
                "num_reachable": len(reachable_points),
                "constraint_statistics": {
                    "all_points": constraint_stats_all,
                    "reachable_points": constraint_stats_reachable,
                },
                "plane_sampling_config": {
                    "plane_normal": self.config.plane_normal,
                    "plane_point": self.config.plane_point,
                    "plane_bounds": self.config.plane_bounds,
                },
            }

        else:
            raise ValueError(f"Unknown analysis mode: {self.config.mode}")

        # Step 3: Compute metrics (common for both modes)
        logger.log_info("[3/3] Computing metrics...")
        metrics = self._compute_metrics()
        results["metrics"] = metrics
        results["config"] = self.config
        results["analysis_time"] = time.time() - start_time

        # Cache results
        if self.cache is not None:
            self._save_to_cache(results)

        # Enhanced completion summary with performance insights
        self._log_analysis_summary(results)

        # Visualize if requested
        if visualize:
            self._visualize_workspace()

        return results

    def _log_analysis_summary(self, results: Dict[str, Any]) -> None:
        """Log enhanced analysis summary with performance insights."""
        analysis_time = results["analysis_time"]
        mode = results["mode"]

        # Time-based performance indicators
        if analysis_time < 30:
            time_icon, time_color = "⚡", "\033[92m"  # Lightning, bright green
        elif analysis_time < 120:
            time_icon, time_color = "🚀", "\033[32m"  # Rocket, green
        elif analysis_time < 300:
            time_icon, time_color = "⏱️", "\033[33m"  # Clock, yellow
        else:
            time_icon, time_color = "🐌", "\033[31m"  # Snail, red

        logger.log_info(
            f"{time_icon} Analysis completed in {time_color}{analysis_time:.2f}s\033[0m"
        )

        if mode == "joint_space":
            success_rate = results["num_valid"] / results["num_samples"] * 100
            logger.log_info(
                f"📊 Joint Space Results: {results['num_valid']}/{results['num_samples']} "
                f"valid points ({success_rate:.1f}% success)"
            )

            # Show constraint statistics
            if "constraint_statistics" in results:
                stats = results["constraint_statistics"]
                logger.log_info(
                    f"🔒 Constraint Check: Bounds: {stats['bounds_pass_rate']:.1f}% | "
                    f"Collision: {stats['collision_pass_rate']:.1f}% | "
                    f"Overall: {stats['overall_pass_rate']:.1f}% "
                    f"({stats['exclude_zones_count']} exclude zones)"
                )

        elif mode in ["cartesian_space", "plane_sampling"]:
            reachability = results["num_reachable"] / results["num_samples"] * 100
            mode_name = (
                "Plane Sampling" if mode == "plane_sampling" else "Cartesian Space"
            )
            logger.log_info(
                f"📊 {mode_name} Results: {results['num_reachable']}/{results['num_samples']} "
                f"reachable points ({reachability:.1f}% reachability)"
            )

            # Show plane sampling specific info
            if mode == "plane_sampling" and "plane_sampling_config" in results:
                plane_config = results["plane_sampling_config"]
                if plane_config:
                    logger.log_info(
                        f"🎯 Plane Configuration: Normal: {plane_config['plane_normal']}, "
                        f"Point: {plane_config['plane_point']}"
                    )

            # Show constraint statistics for all points and reachable points
            if "constraint_statistics" in results:
                all_stats = results["constraint_statistics"]["all_points"]
                logger.log_info(
                    f"🔒 All Points Constraint Check: Bounds: {all_stats['bounds_pass_rate']:.1f}% | "
                    f"Collision: {all_stats['collision_pass_rate']:.1f}% | "
                    f"Overall: {all_stats['overall_pass_rate']:.1f}% "
                    f"({all_stats['exclude_zones_count']} exclude zones)"
                )

                if (
                    "reachable_points" in results["constraint_statistics"]
                    and results["constraint_statistics"]["reachable_points"]
                ):
                    reach_stats = results["constraint_statistics"]["reachable_points"]
                    logger.log_info(
                        f"✅ Reachable Points Constraint Check: Bounds: {reach_stats['bounds_pass_rate']:.1f}% | "
                        f"Collision: {reach_stats['collision_pass_rate']:.1f}% | "
                        f"Overall: {reach_stats['overall_pass_rate']:.1f}%"
                    )

    def _visualize_workspace(self) -> None:
        """Visualize the workspace using configured visualization type and backend.

        Uses the vis_type specified in configuration (default: POINT_CLOUD).
        Tries multiple backends in order: sim_manager → open3d → matplotlib.
        """
        # Early return checks
        if self.workspace_points is None or len(self.workspace_points) == 0:
            logger.log_warning("No workspace points available for visualization")
            return

        if not self.config.visualization.enabled:
            logger.log_warning("Visualization is disabled in configuration")
            return

        # Define backend priority order
        backends = self._get_backend_priority_list()

        # Try each backend in order until one succeeds
        for i, backend in enumerate(backends):
            try:
                logger.log_info(f"Attempting visualization with '{backend}' backend")
                self.visualize(
                    vis_type=self.config.visualization.vis_type,
                    show=True,
                    backend=backend,
                )
                logger.log_info(f"Successfully visualized with '{backend}' backend")
                return

            except Exception as e:
                logger.log_warning(f"Failed to visualize with '{backend}' backend: {e}")

                # If this is not the last backend, try the next one
                if i < len(backends) - 1:
                    continue
                else:
                    logger.log_error(
                        f"All visualization backends failed. "
                        f"Tried: {', '.join(backends)}"
                    )
                    break

    def _compute_constraint_statistics(self, points: torch.Tensor) -> Dict[str, Any]:
        """Compute constraint check statistics for a set of points.

        Args:
            points: Tensor of shape (N, 3) containing workspace points.

        Returns:
            Dictionary containing constraint statistics.
        """
        if len(points) == 0:
            return {
                "num_points": 0,
                "bounds_pass_count": 0,
                "bounds_pass_rate": 0.0,
                "collision_pass_count": 0,
                "collision_pass_rate": 0.0,
                "overall_pass_count": 0,
                "overall_pass_rate": 0.0,
                "exclude_zones_count": self.constraint_checker.get_num_exclude_zones(),
            }

        num_points = len(points)

        # Check bounds constraints
        bounds_pass = self.constraint_checker.check_bounds(points)
        bounds_pass_count = bounds_pass.sum().item()
        bounds_pass_rate = bounds_pass_count / num_points * 100

        # Check collision constraints (exclude zones)
        collision_pass = self.constraint_checker.check_collision(points)
        collision_pass_count = collision_pass.sum().item()
        collision_pass_rate = collision_pass_count / num_points * 100

        # Overall constraint pass (both bounds and collision)
        overall_pass = bounds_pass & collision_pass
        overall_pass_count = overall_pass.sum().item()
        overall_pass_rate = overall_pass_count / num_points * 100

        return {
            "num_points": num_points,
            "bounds_pass_count": bounds_pass_count,
            "bounds_pass_rate": bounds_pass_rate,
            "collision_pass_count": collision_pass_count,
            "collision_pass_rate": collision_pass_rate,
            "overall_pass_count": overall_pass_count,
            "overall_pass_rate": overall_pass_rate,
            "exclude_zones_count": self.constraint_checker.get_num_exclude_zones(),
        }

    def _get_backend_priority_list(self) -> List[str]:
        """Get the priority-ordered list of visualization backends to try.

        Returns:
            List of backend names in order of preference.
        """
        backends = []

        # Prefer sim_manager if available
        if self.sim_manager is not None and hasattr(self.sim_manager, "get_env"):
            backends.append("sim_manager")

        # Always include open3d and matplotlib as fallbacks
        backends.extend(["open3d", "matplotlib"])

        return backends

    def _create_visualizer_with_config(self, factory, vis_type, backend):
        """Create a visualizer with appropriate configuration parameters.

        Args:
            factory: VisualizerFactory instance.
            vis_type: VisualizationType enum.
            backend: Backend string.

        Returns:
            Configured visualizer instance.
        """
        # Prepare common arguments for all visualizers
        common_kwargs = {
            "backend": backend,
            "sim_manager": self.sim_manager,
            "control_part_name": self.control_part_name,
        }

        # Add visualization-type specific arguments
        if vis_type == VisualizationType.POINT_CLOUD:
            common_kwargs["point_size"] = getattr(
                self.config.visualization, "point_size", 8.0
            )
        elif vis_type == VisualizationType.VOXEL:
            common_kwargs["voxel_size"] = getattr(
                self.config.visualization, "voxel_size", 0.05
            )
        elif vis_type == VisualizationType.SPHERE:
            common_kwargs["sphere_radius"] = getattr(
                self.config.visualization, "sphere_radius", 0.005
            )
            common_kwargs["sphere_resolution"] = getattr(
                self.config.visualization, "sphere_resolution", 10
            )
        elif vis_type == VisualizationType.AXIS:
            common_kwargs["axis_length"] = getattr(
                self.config.visualization, "axis_length", 0.05
            )
            common_kwargs["axis_size"] = getattr(
                self.config.visualization, "axis_size", 0.003
            )
            # Pass reference pose if available
            if (
                hasattr(self.config, "reference_pose")
                and self.config.reference_pose is not None
            ):
                common_kwargs["reference_pose"] = self.config.reference_pose
        # For other visualization types (MESH, HEATMAP), use only common arguments

        return factory.create_visualizer(viz_type=vis_type, **common_kwargs)

    def _generate_point_colors_and_sizes(
        self, points: np.ndarray, filtered_to_reachable: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate colors and sizes for workspace points based on reachability.

        Args:
            points: Workspace points, shape (N, 3).
            filtered_to_reachable: Whether points have been pre-filtered to only include reachable ones.

        Returns:
            Tuple of:
                - Colors array, shape (N, 3) with RGB values in [0, 1].
                - Sizes array, shape (N,) with point sizes.
        """
        num_points = len(points)
        colors = np.zeros((num_points, 3))
        sizes = (
            np.ones(num_points) * self.config.visualization.point_size
        )  # Default size

        # Check if we have current_mode attribute (set during analyze)
        if not hasattr(self, "current_mode"):
            # Fallback: color based on available reachability information
            if (
                hasattr(self, "reachability_mask")
                and self.reachability_mask is not None
            ):
                reachability_mask_np = self.reachability_mask.cpu().numpy()
                if len(reachability_mask_np) == num_points:
                    colors[reachability_mask_np, 1] = 1.0  # Green for reachable
                    colors[~reachability_mask_np, 0] = 1.0  # Red for unreachable
                    logger.log_debug("Using available reachability mask for coloring")
                else:
                    colors[:, 1] = 1.0  # Green fallback
                    logger.log_debug(
                        "Reachability mask size mismatch, using green fallback"
                    )
            else:
                colors[:, 1] = 1.0  # Green fallback
                logger.log_debug(
                    "No reachability information available, using green fallback"
                )
            return colors, sizes

        if self.current_mode == AnalysisMode.JOINT_SPACE:
            # Joint space mode: all points are reachable (green, same size)
            colors[:, 1] = 1.0  # Green channel = 1.0
            logger.log_debug(f"Coloring {num_points} points as reachable (green)")

        elif self.current_mode in [
            AnalysisMode.CARTESIAN_SPACE,
            AnalysisMode.PLANE_SAMPLING,
        ]:
            # Cartesian/Plane space mode: different colors and sizes based on reachability
            mode_name = (
                "Cartesian"
                if self.current_mode == AnalysisMode.CARTESIAN_SPACE
                else "Plane sampling"
            )
            if self.success_rates is not None and hasattr(self, "reachability_mask"):
                if filtered_to_reachable:
                    # Points have been pre-filtered, but we still need to check IK reachability
                    # Only color as green if we have verified IK solutions
                    if (
                        hasattr(self, "reachability_mask")
                        and self.reachability_mask is not None
                    ):
                        # Use the actual IK reachability results
                        reachability_mask_np = self.reachability_mask.cpu().numpy()
                        if len(reachability_mask_np) == num_points:
                            # Apply the actual reachability coloring
                            reachable_indices = reachability_mask_np
                            colors[reachable_indices, 1] = 1.0  # Green for IK-reachable
                            sizes[reachable_indices] = (
                                self.config.visualization.point_size * 1.5
                            )

                            unreachable_indices = ~reachability_mask_np
                            colors[unreachable_indices, 0] = (
                                1.0  # Red for IK-unreachable
                            )
                            sizes[unreachable_indices] = (
                                self.config.visualization.point_size * 0.7
                            )

                            num_reachable = np.sum(reachable_indices)
                            num_unreachable = np.sum(unreachable_indices)
                            logger.log_debug(
                                f"Coloring {num_reachable} IK-reachable points (green) and "
                                f"{num_unreachable} IK-unreachable points (red) in {mode_name} mode"
                            )
                        else:
                            # Fallback: color based on geometric constraint only
                            colors[:, 1] = 1.0  # All green (geometrically valid)
                            sizes[:] = self.config.visualization.point_size * 1.5
                            logger.log_warning(
                                f"IK reachability mask length mismatch. "
                                f"Coloring {num_points} geometrically valid points (green) in {mode_name} mode"
                            )
                    else:
                        # No IK verification available, assume geometric validity only
                        colors[:, 1] = 1.0  # All green (geometrically valid)
                        sizes[:] = self.config.visualization.point_size * 1.5
                        logger.log_debug(
                            f"No IK verification available. "
                            f"Coloring {num_points} geometrically valid points (green) in {mode_name} mode"
                        )
                else:
                    # Original logic for showing both reachable and unreachable points
                    reachability_mask_np = self.reachability_mask.cpu().numpy()

                    # Check if mask length matches points length
                    if len(reachability_mask_np) != num_points:
                        logger.log_warning(
                            f"Reachability mask length ({len(reachability_mask_np)}) doesn't match "
                            f"points length ({num_points}). Defaulting to all green."
                        )
                        colors[:, 1] = 1.0  # All green as fallback
                        sizes[:] = self.config.visualization.point_size * 1.5
                    else:
                        # Reachable points: green color, larger size
                        reachable_indices = reachability_mask_np
                        colors[reachable_indices, 1] = 1.0  # Pure green
                        sizes[reachable_indices] = (
                            self.config.visualization.point_size * 1.5
                        )  # Larger size

                        # Unreachable points: red color, smaller size
                        unreachable_indices = ~reachability_mask_np
                        colors[unreachable_indices, 0] = 1.0  # Pure red
                        sizes[unreachable_indices] = (
                            self.config.visualization.point_size * 0.7
                        )  # Smaller size

                    num_reachable = np.sum(reachable_indices)
                    num_unreachable = np.sum(unreachable_indices)
                    logger.log_debug(
                        f"Coloring {num_reachable} reachable points (green, large) and "
                        f"{num_unreachable} unreachable points (red, small) in {mode_name} mode"
                    )
            else:
                # No success rates available, assume all reachable
                colors[:, 1] = 1.0  # Green
                logger.log_warning(
                    f"No success rates available in {mode_name} mode, "
                    "defaulting to green (reachable)"
                )

        return colors, sizes

    def _generate_point_colors(self, points: np.ndarray) -> np.ndarray:
        """Generate colors for workspace points based on reachability (backward compatibility).

        Args:
            points: Workspace points, shape (N, 3).

        Returns:
            Colors array, shape (N, 3) with RGB values in [0, 1].
        """
        colors, _ = self._generate_point_colors_and_sizes(
            points, filtered_to_reachable=False
        )
        return colors

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
        vis_type: VisualizationType | str | None = None,
        show: bool = True,
        save_path: str | None = None,
        backend: str | None = None,
    ) -> Any:
        """Visualize the workspace.

        Args:
            vis_type: Type of visualization to create. Can be VisualizationType enum or string.
                     If None, uses the vis_type from configuration (default: POINT_CLOUD).
                     Supported types: 'point_cloud', 'voxel', 'sphere'.
            show: Whether to display the visualization.
            save_path: Optional path to save the visualization.
            backend: Backend to use ('sim_manager', 'open3d', 'matplotlib', 'data').
                    If None, automatically selects based on availability.

        Returns:
            Visualization object.
        """
        if self.workspace_points is None or len(self.workspace_points) == 0:
            logger.log_error("No workspace points available for visualization")
            return None

        if not self.config.visualization.enabled:
            logger.log_warning("Visualization is disabled in configuration")
            return None

        # Use configured vis_type if not specified
        if vis_type is None:
            vis_type = self.config.visualization.vis_type

        # Handle string vis_type by converting to enum
        if isinstance(vis_type, str):
            try:
                vis_type = VisualizationType(vis_type)
            except ValueError:
                logger.log_warning(
                    f"Unknown visualization type '{vis_type}', falling back to POINT_CLOUD"
                )
                vis_type = VisualizationType.POINT_CLOUD

        # Convert points to numpy first
        points_np = self.workspace_points.cpu().numpy()
        filtered_points = False  # Track if points were filtered

        # Enhanced visualization logging with point count info
        vis_start_time = time.time()
        logger.log_info(
            f"Creating {vis_type.value} visualization for {len(points_np)} points..."
        )

        # Auto-select backend if not specified
        if backend is None:
            if self.sim_manager is not None and hasattr(self.sim_manager, "sim"):
                backend = "sim_manager"
            else:
                backend = "open3d"

        # Filter points if configured to hide unreachable ones in Cartesian/Plane space mode
        if (
            self.current_mode
            in [AnalysisMode.CARTESIAN_SPACE, AnalysisMode.PLANE_SAMPLING]
            and not self.config.visualization.show_unreachable_points
            and hasattr(self, "reachability_mask")
        ):
            # Only show reachable points
            reachable_mask = self.reachability_mask.cpu().numpy()

            # Check if mask length matches points length before filtering
            if len(reachable_mask) != len(points_np):
                logger.log_warning(
                    f"Cannot filter points: reachability mask length ({len(reachable_mask)}) "
                    f"doesn't match points length ({len(points_np)}). Showing all points."
                )
            else:
                points_np = points_np[reachable_mask]
                filtered_points = True
                logger.log_info(
                    f"Filtering to show only {len(points_np)} reachable points"
                )

        # Generate colors and sizes based on reachability
        colors, sizes = self._generate_point_colors_and_sizes(
            points_np, filtered_points
        )

        # Create visualizer using factory pattern
        from embodichain.lab.sim.utility.workspace_analyzer.visualizers import (
            VisualizerFactory,
        )

        factory = VisualizerFactory()
        visualizer = self._create_visualizer_with_config(factory, vis_type, backend)

        # Create visualization with sizes if supported
        try:
            # Try to pass sizes to visualizer (some backends may support it)
            vis_obj = visualizer.visualize(points_np, colors=colors, sizes=sizes)
        except TypeError:
            # Fallback to colors-only visualization if sizes not supported
            vis_obj = visualizer.visualize(points_np, colors=colors)

        # Performance tracking for visualization
        vis_time = time.time() - vis_start_time
        logger.log_info(f"✨ Visualization created in {vis_time:.2f}s")

        # Save if requested
        if save_path:
            save_start = time.time()
            visualizer.save(save_path)
            save_time = time.time() - save_start
            logger.log_info(f"💾 Saved visualization to {save_path} ({save_time:.2f}s)")

        # Show if requested (only for non-sim_manager backends)
        if show and backend != "sim_manager":
            try:
                visualizer.show()
            except Exception as e:
                logger.log_warning(f"Failed to show visualization: {e}")

        return vis_obj

    def _load_from_cache(self) -> dict[str, Any] | None:
        """Load analysis results from cache."""
        if self.cache is None:
            return None

        # TODO: Implement cache loading logic
        return None

    def _save_to_cache(self, results: Dict[str, Any]) -> None:
        """Save analysis results to cache."""
        if self.cache is None:
            return

        # TODO: Implement cache saving logic
        pass

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

        # File size information for export
        try:
            file_size = output_path.stat().st_size
            if file_size > 1024 * 1024:  # > 1MB
                size_str = f"{file_size / (1024*1024):.1f} MB"
            elif file_size > 1024:  # > 1KB
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size} bytes"
            logger.log_info(f"💾 Exported results to {output_path} ({size_str})")
        except OSError:
            # File size unavailable
            logger.log_info(f"💾 Exported results to {output_path}")

    @contextmanager
    def profiling(self):
        """Enhanced context manager for profiling workspace analysis with detailed metrics."""
        logger.log_info("🔍 Starting profiled analysis...")
        start_time = time.time()
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # CPU memory tracking (if psutil is available)
        start_cpu_mem = 0
        process = None
        if psutil is not None:
            try:
                process = psutil.Process()
                start_cpu_mem = process.memory_info().rss
            except (psutil.Error, OSError):
                # Process info unavailable
                process = None

        yield

        end_time = time.time()
        end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        end_cpu_mem = 0
        if process is not None:
            try:
                end_cpu_mem = process.memory_info().rss
            except (psutil.Error, OSError):
                # Process info unavailable, use start value
                end_cpu_mem = start_cpu_mem

        # Detailed performance summary
        analysis_time = end_time - start_time
        if analysis_time < 60:
            time_str = f"{analysis_time:.2f}s"
        else:
            minutes = int(analysis_time // 60)
            seconds = analysis_time % 60
            time_str = f"{minutes}m {seconds:.1f}s"

        logger.log_info(f"⏱️ Analysis time: {time_str}")

        # Memory usage summary
        if torch.cuda.is_available():
            gpu_mem_used = (end_mem - start_mem) / 1024**2
            logger.log_info(f"💾 GPU memory used: {gpu_mem_used:.2f} MB")

        # CPU memory tracking (if available)
        if process is not None and end_cpu_mem > start_cpu_mem:
            cpu_mem_used = (end_cpu_mem - start_cpu_mem) / 1024**2
            logger.log_info(f"💻 CPU memory used: {cpu_mem_used:.2f} MB")

        # Performance rating
        if analysis_time < 30 and (not torch.cuda.is_available() or gpu_mem_used < 100):
            logger.log_info("🚀 Performance: Excellent!")
        elif analysis_time < 120:
            logger.log_info("✅ Performance: Good")
        elif analysis_time < 300:
            logger.log_info("🟡 Performance: Moderate")
        else:
            logger.log_info("🐌 Performance: Needs optimization")
