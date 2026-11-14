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

"""Bounded native deployment and matched teacher evaluation through SDK Scene."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from importlib import metadata
from pathlib import Path
import platform
from typing import Any

import torch
import warp as wp

from embodichain.lab.sim.cfg import NewtonPhysicsCfg
from embodichain_solvers.physics.nerd._binding import NeRDBinding

from ._scene_assets import reference_profile
from ._scene_setup import declare_reference_scene, reference_friction

__all__: list[str] = []


def _runtime_identity(device: torch.device) -> dict[str, Any]:
    """Record actual execution identity separately from historical teacher metadata."""
    from embodichain_models.dynamics.nerd import geometry, transition
    from embodichain_solvers.physics.nerd import provider
    from embodichain.lab.sim.diff import kinematics

    sources = {
        "geometry.py": Path(geometry.__file__),
        "transition.py": Path(transition.__file__),
        "provider.py": Path(provider.__file__),
        "_binding.py": Path(provider.__file__).with_name("_binding.py"),
        "kinematics.py": Path(kinematics.__file__),
    }
    for name in (
        "_scene.py",
        "_scene_setup.py",
        "_scene_assets.py",
        "_reference_model.py",
        "_reference_assets.py",
    ):
        sources[name] = Path(__file__).with_name(name)
    versions = {}
    for name in ("embodichain", "dexsim-engine", "newton", "warp-lang", "torch"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "source-checkout"
    return {
        "device": str(device),
        "engine": "current_newton",
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.machine()
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "state_dtype": "float32",
        "cuda_runtime": torch.version.cuda,
        "versions": versions,
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sources.items()
        },
    }


class DynamicsSimulation:
    """Own one SDK reference Scene for native inference or analytical evaluation.

    This consumer never participates in learning-env rollout. It uses public
    batched Scene reads/writes and leaves stepping, double buffers and reset to
    the SDK. The reference scene has one articulation per zero-offset arena.
    """

    def __init__(
        self,
        robot: str,
        *,
        device: str = "cpu",
        num_envs: int = 1,
        solver: Mapping[str, Any] | None = None,
        num_substeps: int = 10,
        requires_grad: bool = False,
        physics_profile: str = "native",
        use_cuda_graph: bool = True,
    ) -> None:
        import dexsim
        from dexsim.engine.newton_physics import ProviderNewtonSolverCfg
        from dexsim.spawn import SceneBuilder
        from dexsim.spawn.descs import ReplicatePlan

        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be positive")
        if type(num_substeps) is not int or num_substeps < 1:
            raise ValueError("num_substeps must be positive")
        self.profile = reference_profile(robot)
        self.robot, self.device, self.num_envs = robot, torch.device(device), num_envs
        self.dt = self.profile["dt"]
        configured = dict(
            solver
            or {
                "solver_type": "featherstone",
                "update_mass_matrix_interval": num_substeps,
            }
        )
        friction = reference_friction(robot, physics_profile, configured)
        neural = configured.get("solver_type") == "nerd"
        if neural:
            if num_substeps != 1:
                raise ValueError(
                    "NeRD native deployment requires one prediction per step"
                )
            from embodichain_solvers.physics.nerd.provider import NeRDProvider

            solver_cfg = ProviderNewtonSolverCfg(
                provider_factory=NeRDProvider, options=configured.get("options", {})
            )
        else:
            solver_cfg = configured
        physics = NewtonPhysicsCfg(
            device=str(self.device),
            physics_dt=self.dt,
            num_substeps=num_substeps,
            requires_grad=requires_grad,
            use_cuda_graph=use_cuda_graph,
            collision_cfg=(
                None
                if neural
                else (
                    {"update_interval": None}
                    if physics_profile == "warp_reference"
                    else {}
                )
            ),
            solver_cfg=solver_cfg,
        )
        config = dexsim.WorldConfig()
        config.open_windows = False
        config.cpu_num = 2
        config.log_startup_info = False
        config.scene_node_capacity = max(1024, num_envs * 32 + 256)
        config.init_ground = False
        config.use_default_physics = False
        config.backend = dexsim.types.Backend.VULKAN
        config.renderer = dexsim.types.Renderer.NORENDER
        config.newton_cfg = physics.to_dexsim_cfg(gpu_id=0)
        self.world = dexsim.World(config)
        self.scene = None
        self.batch = None
        try:
            builder = SceneBuilder(self.world)
            builder.replicate_plan = ReplicatePlan(num_envs, spacing=(0.0, 0.0, 0.0))
            declare_reference_scene(builder, robot, physics_profile, friction)
            self.scene = builder.finalize()
            metadata = dict(self.scene.dynamics_metadata())
            self.binding = NeRDBinding.from_metadata(metadata, robot)
            paths = [f"{name}/{robot}" for name in builder.replicate_plan.env_names()]
            self.batch = self.scene.create_articulation_batch(
                [self.scene.get_articulation(path) for path in paths]
            )
            public = self.batch.joint_names_per_articulation
            self._joint_ids = [
                public[0].index(name) for name in self.binding.joint_names
            ]
            if any(tuple(names) != tuple(public[0]) for names in public):
                raise ValueError(
                    "Reference batch has inconsistent public joint ordering"
                )
            self._q = torch.empty(
                self.num_envs, self.batch.dof_width, device=self.device
            )
            self._qd = torch.empty_like(self._q)
            self._root = torch.empty(self.num_envs, 7, device=self.device)
            self._linear = torch.empty(self.num_envs, 3, device=self.device)
            self._angular = torch.empty_like(self._linear)
            self._links = torch.empty(
                self.num_envs, self.batch.link_width, 7, device=self.device
            )
            labels = metadata["body_label"]
            worlds = metadata["body_world"]
            worlds = (
                wp.to_torch(worlds).cpu().tolist()
                if isinstance(worlds, wp.array)
                else torch.as_tensor(worlds).tolist()
            )
            body_ids = {
                (world, name.rsplit("/", 1)[-1]): index
                for index, (world, name) in enumerate(zip(worlds, labels))
            }
            self._body_order = torch.tensor(
                [
                    body_ids[(row, name)]
                    for row, names in enumerate(self.batch.link_names_per_articulation)
                    for name in names
                ],
                dtype=torch.long,
                device=self.device,
            )
            self._body_count = metadata["body_count"]
            self.gravity_dir = self.binding.geometry.gravity_dir
            self.manifest = {
                **self.profile,
                **self.binding.metadata(self.dt),
                "physics_profile": physics_profile,
                "teacher": {
                    "solver": configured,
                    "num_substeps": num_substeps,
                    "physics_profile": physics_profile,
                },
            }
            if friction is not None:
                self.manifest["reference_friction_mapping"] = friction
            self.reset()
        except Exception:
            self.close()
            raise

    def snapshot(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return owned canonical state/contact snapshots from the live Scene."""
        self.batch.fetch_state(
            self._q, self._qd, self._root, self._linear, self._angular
        )
        self.batch.fetch_link_pose(self._links)
        root = torch.cat((self._root[:, 4:], self._root[:, :4]), -1)
        q, qd = self._q[:, self._joint_ids], self._qd[:, self._joint_ids]
        if self.robot == "ant":
            q = torch.cat((root, q), -1)
            qd = torch.cat((self._linear, self._angular, qd), -1)
        poses = torch.cat((self._links[..., 4:], self._links[..., :4]), -1).reshape(
            -1, 7
        )
        body_q = poses.new_zeros((self._body_count, 7)).index_copy(
            0, self._body_order, poses
        )
        return {
            "q": q.clone(),
            "qd": qd.clone(),
            "root_body_q": root.clone(),
        }, self.binding.geometry.contacts_from_poses(body_q)

    def reset(
        self, q: torch.Tensor | None = None, qd: torch.Tensor | None = None
    ) -> None:
        """Restore native rows, including solver history, then install canonical state."""
        q = (
            torch.tensor(self.profile["default_q"], device=self.device).expand(
                self.num_envs, -1
            )
            if q is None
            else q
        )
        qd = (
            torch.tensor(self.profile["default_qd"], device=self.device).expand(
                self.num_envs, -1
            )
            if qd is None
            else qd
        )
        if q.shape != (self.num_envs, self.profile["q_dim"]) or qd.shape != (
            self.num_envs,
            self.profile["qd_dim"],
        ):
            raise ValueError("Reset must supply the complete canonical batch")
        kwargs = dict(dof_ids=self._joint_ids, clear_dynamics=True)
        if self.robot == "ant":
            kwargs.update(
                joint_position=q[:, 7:],
                joint_velocity=qd[:, 6:],
                root_pose=torch.cat((q[:, 3:7], q[:, :3]), -1),
                root_linear_velocity=qd[:, :3],
                root_angular_velocity=qd[:, 3:6],
            )
        else:
            kwargs.update(joint_position=q, joint_velocity=qd)
        self.batch.apply_state(**kwargs)

    def step(self, effort: torch.Tensor) -> None:
        """Hold generalized reference efforts for one native macro interval."""
        if effort.shape != (self.num_envs, self.profile["qd_dim"]):
            raise ValueError("Effort must use the complete generalized layout")
        if self.robot == "ant":
            if bool(torch.any(effort[:, :6] != 0)):
                raise ValueError("Ant cannot apply free-base forces")
            effort = effort[:, 6:]
        self.batch.apply_joint_force(effort, dof_ids=self._joint_ids)
        self.world.update()

    def close(self) -> None:
        """Release public child views before the native scene owner."""
        self.batch = None
        if self.scene is not None:
            scene, self.scene = self.scene, None
            scene.close()
        self.world = None

    def __enter__(self) -> DynamicsSimulation:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
