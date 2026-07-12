# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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

"""Simulation manager for gravity-based asset placement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh

from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.sim_manager import (
    SimulationManager as _EmbodiSimManager,
    SimulationManagerCfg,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.simulation_manager.schemas import (
    GravityDropRequest,
    GravityDropResult,
)

__all__ = ["SimulationManager"]


class SimulationManager:
    """Manager for gravity-based asset placement.

    Wraps an EmbodiChain simulation instance with typed request/response
    methods, following the same pattern as service clients.
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        physics_dt: float = 0.01,
        sim_device: str = "cpu",
    ) -> None:
        """Initialize the simulation manager.

        Args:
            headless: Whether to run without a GUI.
            physics_dt: Physics timestep in seconds.
            sim_device: Device to run the simulation on.
        """
        self._headless = headless
        self._physics_dt = physics_dt
        self._sim_device = sim_device

    def run_gravity_simulation(self, request: GravityDropRequest) -> GravityDropResult:
        """Drop one GLB under gravity and return its final pose."""
        glb_path = request.glb_path.expanduser().resolve()
        if not glb_path.is_file():
            raise FileNotFoundError(f"GLB file not found: {glb_path}")
        settle_mode = str(request.gravity_settle_mode or "geometry").strip().lower()
        if settle_mode == "geometry":
            return GravityDropResult(final_pose=self._aabb_bottom_center_pose(glb_path))
        if settle_mode != "physics":
            raise ValueError(
                "gravity_settle_mode must be 'geometry' or 'physics', "
                f"got {request.gravity_settle_mode!r}."
            )

        initial_height = (
            float(request.initial_height)
            if request.initial_height is not None
            else self._compute_adaptive_drop_height(glb_path)
        )
        sim = _EmbodiSimManager(
            SimulationManagerCfg(
                headless=self._headless,
                physics_dt=self._physics_dt,
                sim_device=self._sim_device,
            )
        )
        obj = sim.add_rigid_object(
            RigidObjectCfg(
                uid="dropped_asset",
                shape=MeshCfg(fpath=str(glb_path)),
                init_pos=(0.0, 0.0, initial_height),
                init_rot=(0.0, 0.0, 0.0),
                body_type="dynamic",
                max_convex_hull_num=request.max_convex_hull_num,
            )
        )
        sim.update(step=300)

        final_pose = obj.get_local_pose(to_matrix=True)[0].detach().cpu()
        sim._deferred_destroy()
        return GravityDropResult(
            final_pose=np.asarray(final_pose.numpy(), dtype=float),
        )

    @staticmethod
    def _aabb_bottom_center_pose(glb_path: Path) -> np.ndarray:
        """Return a pose that moves the sim-frame AABB bottom center to origin."""
        mesh = trimesh.load(str(glb_path), force="mesh")
        # Gravity inputs are GLB Y-up, but callers apply the returned pose in
        # the internal/simulation Z-up frame. Convert vertices before measuring.
        glb_y_up_to_sim_z_up = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        mesh = mesh.copy()
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        mesh.vertices = vertices @ glb_y_up_to_sim_z_up.T
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
            raise ValueError(f"Cannot compute AABB for GLB: {glb_path}")
        bottom_center = np.array(
            [
                0.5 * (bounds[0, 0] + bounds[1, 0]),
                0.5 * (bounds[0, 1] + bounds[1, 1]),
                bounds[0, 2],
            ],
            dtype=np.float64,
        )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = -bottom_center
        return pose

    def _compute_adaptive_drop_height(
        self,
        glb_path: Path,
        *,
        min_clearance: float = 0.2,
        height_scale: float = 1.25,
    ) -> float:
        """Compute an initial drop height from a GLB bounding box."""
        if min_clearance < 0.0:
            raise ValueError("min_clearance must be non-negative.")
        if height_scale <= 0.0:
            raise ValueError("height_scale must be positive.")

        glb_path = glb_path.expanduser().resolve()
        loaded = trimesh.load(glb_path, force=None)
        if isinstance(loaded, trimesh.Scene):
            bounds = loaded.bounds
        else:
            bounds = loaded.bounds
        height = float(bounds[1][2] - bounds[0][2])
        return max(height * height_scale, height + min_clearance)
