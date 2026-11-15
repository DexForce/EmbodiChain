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

"""Native eight-scale Sharpa task with explicit reset and contact-law scope.

Physics is owned by SimulationManager and its object, sensor and physics
adapters. This experimental task reuses the reference observation/reward
contract; it does not instantiate the upstream environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _native_task import _NativeTask
from _simulation import _JOINT_NAMES
from unilab.base.torch_env import TorchEnvState

from embodichain.lab.sim.physics._contact_material_view import _ContactMaterialScaleView
from embodichain.lab.sim.physics._cylinder_variants import _shrink_cylinder_variants
from embodichain.lab.sim.physics._joint_drive_view import _JointDriveGainView
from embodichain.lab.sim.physics._source_joint_properties import (
    _apply_source_joint_damping,
)

__all__: list[str] = []


class _RandomizedNativeTask(_NativeTask):
    """Reset native state and batch source-distribution physical parameters."""

    def __init__(
        self,
        cfg: Any,
        upstream: Path,
        output: Path,
        *,
        num_envs: int,
        seed: int,
        scale: float,
    ) -> None:
        self._randomization_ready = False
        self._materials = self._drives = None
        self._normal_physical_reset = None
        super().__init__(
            cfg,
            upstream,
            output,
            num_envs=num_envs,
            seed=seed,
            scale=scale,
            source_task_profile=True,
        )
        try:
            self._default_kp = self.xi[:, :22].clone()
            self._default_kd = self.xi[:, 22:44].clone()
            authored = self.physical.reference_profile["joints"]
            self._materials = _ContactMaterialScaleView(self.sim, ("hand", "cylinder"))
            self._materials.set_impedance((0.95, 0.99, 0.001, 0.5, 2.0))
            _apply_source_joint_damping(
                self.sim,
                {name: authored[name]["passive_damping"] for name in _JOINT_NAMES},
            )
            self._drives = _JointDriveGainView(self.sim, tuple(_JOINT_NAMES))
            self._gravity = torch.zeros(num_envs, 3, device=self.device)
            self._disturbance = torch.zeros_like(self._gravity)
            self._normal_physical_reset = self.physical.reset
            self.physical.reset = self._reset_physical_state
            self._randomization_ready = True
            self.identity.update(
                parameter_randomization=True,
                tactile="native elastomer/object geometry-pair netforce",
                solver_algorithm="newton",
                solver_integrator="euler",
                parameter_ranges={
                    "pd_gain_scale": [0.5, 2.0],
                    "friction_scale": [0.75, 1.25],
                    "mass_kg": [0.01, 0.25],
                    "com_m": [-0.01, 0.01],
                    "gravity_m_s2": 9.81,
                },
                raw_cache_replay=True,
                limitation="Native hand self-contacts use condim6, metal torsional/rolling friction .1/.02 and impedance .95/.99; upstream hand geoms use condim3, metal friction .005/.0001 and impedance .9/.95. Backend versions and learner resource sizes also differ. Strict baseline parity is not qualified.",
            )
            self.reset(seed=seed)
        except BaseException:
            self.close()
            raise

    def _uniform(self, shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
        return (
            torch.rand(shape, device=self.device, generator=self.physical.generator)
            * (high - low)
            + low
        )

    def _gain_scale(self, shape: tuple[int, ...]) -> torch.Tensor:
        low = self._uniform(shape, 0.5, 1.0)
        high = self._uniform(shape, 1.0, 2.0)
        return torch.where(self._uniform(shape, 0.0, 1.0) > 0.5, low, high)

    @torch.no_grad()
    def _reset_physical_state(
        self, env_ids: torch.Tensor | None = None, *, seed: int | None = None
    ) -> torch.Tensor:
        physical = self.physical
        if seed is not None:
            physical.generator.manual_seed(seed)
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else env_ids
        )
        if not len(ids):
            return physical.read_state()
        grasp = self._sample_grasps(ids)
        q = grasp[:, :22]
        self.robot.set_state(
            qpos=q,
            target_qpos=q,
            qvel=torch.zeros_like(q),
            joint_ids=physical.joint_ids,
            env_ids=ids,
            clear_dynamics=True,
            clamp_qpos=False,
            clamp_target_qpos=False,
        )
        pose = torch.cat((grasp[:, 22:25], grasp[:, 26:29], grasp[:, 25:26]), -1)
        self.object.clear_dynamics(env_ids=ids)
        self.object.set_local_pose(pose, env_ids=ids)
        physical.context[ids] = torch.cat((q, grasp[:, 22:25], q), -1)
        physical.steps[ids] = 0
        physical.episode_ids[ids] += 1
        return physical.read_state()

    def _sample_grasps(self, ids: torch.Tensor) -> torch.Tensor:
        choices = torch.randint(
            len(self.physical.grasps),
            (len(ids),),
            generator=self.physical.generator,
            device=self.device,
        )
        return self.physical.grasps[choices]

    @torch.no_grad()
    def reset(
        self, env_indices: torch.Tensor | None = None, *, seed: int | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        obs, info = super().reset(env_indices, seed=seed)
        if not self._randomization_ready:
            return obs, info
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_indices is None
            else env_indices
        )
        n = len(ids)
        if not n:
            return obs, info
        kp = self._default_kp[ids] * self._gain_scale((n, 22))
        kd = self._default_kd[ids] * self._gain_scale((n, 22))
        self._drives.apply(kp, kd, ids)
        self.xi[ids, :22], self.xi[ids, 22:44] = kp, kd
        friction = self._uniform((n,), 0.75, 1.25)
        self._materials.apply(friction, ids)
        self.xi[ids, 44] = friction
        mass = self._uniform((n,), 0.01, 0.25)
        self.object.set_mass(mass, env_ids=ids)
        self.xi[ids, 45] = mass
        com = self._uniform((n, 3), -0.01, 0.01)
        pose = torch.zeros(n, 7, device=self.device)
        pose[:, :3], pose[:, 6] = com, 1
        self.object.set_com_pose(pose, env_ids=ids)
        self.xi[ids, 46:49] = com
        direction = torch.randn(
            n, 3, device=self.device, generator=self.physical.generator
        )
        self._gravity[ids] = direction / direction.norm(dim=-1, keepdim=True) * 9.81
        self._disturbance[ids] = 0
        self.physical.gravity_force[ids] = mass[:, None] * self._gravity[ids]
        obs = self._observe(ids, reset=True)
        self.state = TorchEnvState(
            obs, self.state.reward, self.state.terminated, self.state.truncated, info
        )
        return obs, info

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> TorchEnvState:
        self._disturbance.mul_(0.9 ** (0.05 / 0.08))
        ids = (self._uniform((self.num_envs,), 0.0, 1.0) < 0.25).nonzero().flatten()
        if len(ids):
            self._disturbance[ids] = (
                torch.randn(
                    len(ids), 3, device=self.device, generator=self.physical.generator
                )
                * self.xi[ids, 45:46]
                * 2.0
            )
        self.physical.gravity_force.copy_(
            self.xi[:, 45:46] * self._gravity + self._disturbance
        )
        return super().step(action)

    def close(self) -> None:
        if self._drives is not None:
            self._drives.close()
            self._drives = None
        if self._materials is not None:
            self._materials.close()
            self._materials = None
        if self._normal_physical_reset is not None:
            self.physical.reset = self._normal_physical_reset
            self._normal_physical_reset = None
        if self.sim is not None:
            super().close()


class _FullNativeTask(_RandomizedNativeTask):
    """Own eight fixed collider sizes in one native manager, with equal row counts."""

    def __init__(
        self, cfg: Any, upstream: Path, output: Path, *, num_envs: int, seed: int
    ) -> None:
        self._variants_ready = False
        if num_envs < 8 or num_envs % 8:
            raise ValueError("Use a positive multiple of eight native environments")
        super().__init__(cfg, upstream, output, num_envs=num_envs, seed=seed, scale=1.5)
        try:
            self.width = num_envs // 8
            scales = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
            self._scale_ids = torch.arange(8, device=self.device).repeat_interleave(
                self.width
            )
            self._scale_values = torch.tensor(scales, device=self.device)[
                self._scale_ids
            ]
            assets = upstream / "src/sharpa_rl_unilab/assets/caches"
            rows = [
                np.load(assets / f"sharpa_grasp_linspace_{scale:g}.npy")
                for scale in scales
            ]
            if len({row.shape for row in rows}) != 1:
                raise ValueError("Expected equally sized source grasp caches")
            self._grasp_variants = torch.as_tensor(
                np.stack(rows), device=self.device, dtype=torch.float32
            )
            _shrink_cylinder_variants(self.sim, "cylinder", self._scale_values / 1.5)
            self.xi[:, 49] = self._scale_values
            self._variants_ready = True
            self.identity.update(
                scales=scales,
                object_radius_m=[0.016, 0.03],
                object_height_m=[0.0256, 0.048],
                environment_layout="one manager with eight fixed cylinder variants",
                geometry_bounds="largest enclosing template; actual narrow-phase cylinder sizes and constant-mass inertias scaled per Arena",
                render_geometry="template only; external video uses corresponding scale models and measured states",
            )
            self.reset(seed=seed)
            (output / "native_identity.json").write_text(
                json.dumps(self.identity, indent=2) + "\n"
            )
        except BaseException:
            self.close()
            raise

    def _sample_grasps(self, ids: torch.Tensor) -> torch.Tensor:
        if not self._variants_ready:
            return super()._sample_grasps(ids)
        choices = torch.randint(
            self._grasp_variants.shape[1],
            (len(ids),),
            generator=self.physical.generator,
            device=self.device,
        )
        return self._grasp_variants[self._scale_ids[ids], choices]
