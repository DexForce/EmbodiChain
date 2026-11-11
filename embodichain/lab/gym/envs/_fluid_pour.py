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

"""Measured qualification for configured online fluid pouring collections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import time
from typing import Any

import numpy as np
import torch

from embodichain.lab.gym.envs.managers.observations import (
    get_fluid_summary,
    get_fluid_volume_in_region,
)

__all__: list[str] = []


def configure_fluid_spacing(cfg: Any, spacing: float) -> None:
    """Resample one procedural fluid while keeping its render scale consistent."""
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("particle spacing must be finite and positive")
    if len(cfg.fluid_object) != 1:
        raise ValueError("particle spacing override requires one fluid object")
    fluid = cfg.fluid_object[0]
    if fluid.spawn.kind == "points":
        raise ValueError("particle spacing requires a procedural volume source")
    solver = cfg.sim_cfg.physics_cfg.solver_cfg
    if solver.get("solver_type") != "dexuni" or not isinstance(
        solver.get("sph_cfg"), dict
    ):
        raise ValueError("fluid spacing override requires DexUni with sph_cfg")
    scale = spacing / fluid.particle_spacing
    if fluid.render.render_radius is not None:
        fluid.render.render_radius *= scale
    for key in ("cell_size", "kernel_radius"):
        if key in fluid.render.mesh_surface:
            fluid.render.mesh_surface[key] *= scale
    fluid.particle_spacing = spacing
    fluid.particle_radius = 0.3 * spacing
    solver["sph_cfg"]["particle_support_radius"] = 2.25 * spacing


def cup_motion_qualification(
    poses: Sequence[np.ndarray],
    *,
    min_lift_m: float,
    min_tilt_deg: float,
    max_return_error_m: float,
) -> dict[str, Any]:
    """Measure lift, tilt and return using arena-local XYZW cup poses."""
    measured = np.stack(poses, axis=1)
    lift = measured[..., 2].max(axis=1) - measured[:, 0, 2]
    vertical = 1 - 2 * (measured[..., 3] ** 2 + measured[..., 4] ** 2)
    tilt = np.rad2deg(np.arccos(np.clip(vertical, -1, 1))).max(axis=1)
    return_error = np.linalg.norm(measured[:, -1, :3] - measured[:, 0, :3], axis=1)
    return {
        "lift_m": lift.tolist(),
        "max_tilt_deg": tilt.tolist(),
        "return_error_m": return_error.tolist(),
        "accepted": (
            (lift >= min_lift_m)
            & (tilt >= min_tilt_deg)
            & (return_error <= max_return_error_m)
        ).tolist(),
    }


class FluidPourQualification:
    """Collect low-rate liquid samples and every control-boundary cup pose."""

    def __init__(self, env: Any, declaration: Mapping[str, Any]) -> None:
        if declaration.get("kind") != "fluid_pour":
            raise ValueError("collection.qualification.kind must be fluid_pour")
        allowed = {
            "kind",
            "cup_uid",
            "summary_observation",
            "volume_observation",
            "min_lift_m",
            "min_tilt_deg",
            "max_return_error_m",
            "min_received_ml",
        }
        if set(declaration) - allowed:
            raise ValueError("Unknown fluid pour qualification fields")
        self._env = env
        self.cup_uid = declaration.get("cup_uid", "cup")
        if not isinstance(self.cup_uid, str) or not self.cup_uid:
            raise ValueError(
                "collection.qualification.cup_uid must be a nonempty string"
            )
        self._summary = self._observation(
            declaration.get("summary_observation", "water_summary"),
            get_fluid_summary,
        )
        self._volume = self._observation(
            declaration.get("volume_observation", "received_volume"),
            get_fluid_volume_in_region,
        )
        self._fluid_uid = self._summary.params["entity_cfg"].uid
        self._fluid_cfg = next(
            (fluid for fluid in env.cfg.fluid_object if fluid.uid == self._fluid_uid),
            None,
        )
        if self._fluid_cfg is None:
            raise ValueError("Fluid qualification observation has no configured fluid")
        self.min_lift_m = self._threshold(declaration, "min_lift_m", 0.05)
        self.min_tilt_deg = self._threshold(declaration, "min_tilt_deg", 45.0)
        self.max_return_error_m = self._threshold(
            declaration, "max_return_error_m", 0.03
        )
        self.min_received_ml = self._threshold(declaration, "min_received_ml", 50.0)
        self._poses: list[np.ndarray] = []
        self._samples: list[dict[str, Any]] = []
        self._stride = max(1, round(1 / env.step_dt))
        self._started = time.perf_counter()

    @staticmethod
    def _threshold(declaration: Mapping[str, Any], name: str, default: float) -> float:
        value = declaration.get(name, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(
                f"collection.qualification.{name} must be finite and non-negative"
            )
        return float(value)

    def _observation(self, name: Any, expected: Any) -> Any:
        if not isinstance(name, str) or not name:
            raise ValueError("fluid qualification observation names must be strings")
        cfg = getattr(self._env.cfg.observations, name, None)
        if cfg is None or cfg.func is not expected:
            raise ValueError(
                f"fluid qualification requires {expected.__name__} observation"
            )
        return cfg

    def _cup_pose(self) -> np.ndarray:
        return (
            self._env.sim.get_rigid_object(self.cup_uid)
            .get_local_pose()
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

    def _sample(self) -> dict[str, Any]:
        summary = self._summary.func(self._env, {}, **self._summary.params)
        if not torch.isfinite(summary).all() or not torch.all(summary[:, -1] == 1):
            raise RuntimeError("Fluid state became non-finite")
        received = self._volume.func(self._env, {}, **self._volume.params)
        if not torch.isfinite(received).all():
            raise RuntimeError("Received fluid volume became non-finite")
        return {
            "time_s": (self._env.elapsed_steps * self._env.step_dt).cpu().tolist(),
            "received_ml": (received[:, 0] * 1e6).cpu().tolist(),
            "summary": summary.cpu().tolist(),
            "cup_pose": self._cup_pose().tolist(),
        }

    def start(self) -> None:
        """Seed the initial pose and liquid sample after the episode reset."""
        self._poses = [self._cup_pose()]
        self._samples = [self._sample()]
        self._started = time.perf_counter()

    def after_step(self) -> None:
        """Capture the post-control cup pose and one liquid sample per second."""
        self._poses.append(self._cup_pose())
        if int(self._env.elapsed_steps.max()) % self._stride == 0:
            self._samples.append(self._sample())

    def finish(self, result: Any, selected_rows: Sequence[int]) -> dict[str, Any]:
        """Return measured qualification for the selected source rows."""
        self._poses.append(self._cup_pose())
        final = self._sample()
        self._samples.append(final)
        cup = cup_motion_qualification(
            self._poses,
            min_lift_m=self.min_lift_m,
            min_tilt_deg=self.min_tilt_deg,
            max_return_error_m=self.max_return_error_m,
        )
        accepted = bool(
            result.completed
            and result.all_success
            and all(
                cup["accepted"][row]
                and final["received_ml"][row] >= self.min_received_ml
                for row in selected_rows
            )
        )
        return {
            "accepted": accepted,
            "coupling": "online one-way rigid-to-fluid",
            "num_envs": self._env.num_envs,
            "renderer": self._env.cfg.sim_cfg.render_cfg.renderer,
            "particle_count_per_env": self._env.sim.get_fluid_object(
                self._fluid_uid
            ).num_particles,
            "fluid_source": self._fluid_cfg.spawn.to_dict(),
            "particle_spacing_m": self._fluid_cfg.particle_spacing,
            "physics_dt": self._env.cfg.sim_cfg.physics_dt,
            "control_dt": self._env.step_dt,
            "task_program": result.to_metadata(),
            "cup": cup,
            "samples": self._samples,
            "wall_seconds": time.perf_counter() - self._started,
            "selected_rows": list(selected_rows),
        }
