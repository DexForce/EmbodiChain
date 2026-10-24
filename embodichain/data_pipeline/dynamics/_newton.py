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

"""Batched analytical Newton teacher for offline trajectory collection."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from importlib import metadata
from pathlib import Path
import json
from typing import Any

import numpy as np
import torch
import warp as wp

from embodichain.compute.dynamics import plane_contact_features, quat_rotate
from ._reference_assets import (
    _ant_reference_friction,
    _import_reference,
    reference_profile,
)

__all__: list[str] = []


class _NewtonTeacher:
    """Own an independent Newton model without a DexSim registry or Scene."""

    def __init__(
        self,
        robot: str,
        *,
        device: str = "cpu",
        num_envs: int = 1,
        solver: Mapping[str, Any] | None = None,
        num_substeps: int = 10,
        physics_profile: str = "native",
        use_cuda_graph: bool = True,
    ) -> None:
        import newton

        self.profile = reference_profile(robot)
        self.robot, self.device, self.num_envs = robot, torch.device(device), num_envs
        self.dt, self.num_substeps = self.profile["dt"], num_substeps
        solver_cfg = dict(
            solver
            or (
                {
                    "solver_type": "featherstone",
                    "update_mass_matrix_interval": num_substeps,
                }
                if physics_profile == "warp_reference"
                else {
                    "solver_type": "mujoco_warp",
                    "use_mujoco_cpu": self.device.type == "cpu",
                    "separate_worlds": True,
                }
            )
        )
        kind = solver_cfg.pop("solver_type", "mujoco_warp")
        if kind not in {"featherstone", "mujoco_warp"}:
            raise ValueError("Offline teachers support featherstone or mujoco_warp")
        if kind == "featherstone":
            interval = solver_cfg.setdefault("update_mass_matrix_interval", 1)
            if type(interval) is not int or interval < 1 or num_substeps % interval:
                raise ValueError(
                    "Featherstone mass-matrix interval must divide num_substeps"
                )
        friction = None
        if (
            robot == "ant"
            and physics_profile == "warp_reference"
            and kind == "featherstone"
        ):
            friction = _ant_reference_friction()
            delta = solver_cfg.setdefault(
                "friction_smoothing", friction["friction_smoothing"]
            )
            if isinstance(delta, bool) or delta != friction["friction_smoothing"]:
                raise ValueError(
                    "Ant warp_reference requires the coupled contact mapping"
                )
        # CUDA graphs cannot capture MuJoCo's CPU executor.
        capture = (
            use_cuda_graph
            and self.device.type == "cuda"
            and not solver_cfg.get("use_mujoco_cpu", False)
        )
        template = _import_reference(robot, physics_profile)
        if friction is not None:
            template.shape_material_mu[:] = [
                mu * friction["material_mu_scale"] for mu in template.shape_material_mu
            ]
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        ground = None
        if robot == "ant":
            ground = dict(
                ke=1e5,
                kd=1e3,
                kf=1e3,
                mu=friction["ground_mu"] if friction else 0.5,
                margin=1e-5,
                gap=0.05,
                mu_torsional=0.0,
                mu_rolling=0.0,
            )
            builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(**ground))
        for _ in range(num_envs):
            builder.add_world(template)
        self.model = builder.finalize(device=str(self.device))
        self.state, self.next_state = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        solver_type = (
            newton.solvers.SolverFeatherstone
            if kind == "featherstone"
            else newton.solvers.SolverMuJoCo
        )
        self.solver = solver_type(self.model, **solver_cfg)
        self._mujoco = kind == "mujoco_warp"
        qdim, vdim = self.profile["q_dim"], self.profile["qd_dim"]
        if (
            self.model.joint_coord_count != num_envs * qdim
            or self.model.joint_dof_count != num_envs * vdim
        ):
            raise ValueError(
                "Imported reference state dimensions do not match the codec"
            )
        names = [name.rsplit("/", 1)[-1] for name in template.joint_label]
        scalar = [
            name
            for name, start, end in zip(
                names,
                template.joint_qd_start,
                template.joint_qd_start[1:] + [template.joint_dof_count],
            )
            if end - start == 1
        ]
        if scalar != self.profile["joint_names"]:
            raise ValueError("Imported scalar joint order differs from the reference")
        self._root_bodies = torch.tensor(
            np.asarray(self.model.articulation_start.numpy())[:-1],
            device=self.device,
            dtype=torch.long,
        )
        child = wp.to_torch(self.model.joint_child).long()
        self._root_bodies = child[self._root_bodies]
        self.gravity_dir = wp.to_torch(self.model.gravity)[:num_envs].clone()
        self.gravity_dir /= self.gravity_dir.norm(dim=-1, keepdim=True)
        self._prepare_contacts(newton)
        arrays = {}
        for name in (
            "body_mass",
            "body_inertia",
            "body_com",
            "body_flags",
            "joint_type",
            "joint_parent",
            "joint_child",
            "joint_q_start",
            "joint_qd_start",
            "joint_X_p",
            "joint_X_c",
            "joint_axis",
            "joint_armature",
            "joint_damping",
            "joint_limit_lower",
            "joint_limit_upper",
            "joint_limit_ke",
            "joint_limit_kd",
            "joint_target_ke",
            "joint_target_kd",
            "shape_body",
            "shape_type",
            "shape_scale",
            "shape_transform",
            "shape_margin",
            "shape_gap",
            "shape_material_mu",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_kf",
        ):
            values = getattr(template, name)
            arrays[name] = hashlib.sha256(np.asarray(values).tobytes()).hexdigest()
        gravity = wp.to_torch(self.model.gravity)[0].tolist()
        signature = hashlib.sha256(
            json.dumps(
                {
                    "arrays": arrays,
                    "ground": ground,
                    "gravity": gravity,
                    "profile": physics_profile,
                    "robot": robot,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        origin = {
            "kind": "current_newton",
            "collector": "direct_newton",
            "device": str(self.device),
            "solver": {"solver_type": kind, **solver_cfg},
            "num_substeps": num_substeps,
            "physics_profile": physics_profile,
            "use_cuda_graph": capture,
            "versions": {
                name: metadata.version(name)
                for name in ("newton", "warp-lang", "torch")
            },
        }
        self.manifest = {
            **self.profile,
            "physics_profile": physics_profile,
            "teacher_profile": "direct_newton_reference_import",
            "physics_signature": signature,
            "physics_signature_schema": "nerd-direct-newton-builder-v1",
            "source_physical_array_sha256": arrays,
            "ground_material": ground,
            "collector_source_sha256": {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in ("_newton.py", "_reference_assets.py", "generation.py")
            },
            "gravity": gravity,
            "codec": {
                **self.profile["codec"],
                "base_com": wp.to_torch(self.model.body_com)[
                    self._root_bodies[0]
                ].tolist(),
            },
            "contact_schema": "nerd-static-plane-v1",
            "teacher": origin,
            "teacher_origin": {
                **origin,
                "dt": self.dt,
                "physics_signature": signature,
                "physics_signature_schema": "nerd-direct-newton-builder-v1",
            },
        }
        if friction is not None:
            self.manifest["reference_friction_mapping"] = friction
        self.graph = None
        self.reset()
        if capture:
            # Warm up outside capture, then restore inputs. A graph must return
            # to the same buffer addresses even for an odd substep count.
            self._advance()
            self.reset()
            with wp.ScopedCapture(device=str(self.device)) as captured:
                self._advance()
            self.graph = captured.graph
            self.reset()

    def _prepare_contacts(self, newton: Any) -> None:
        bodies, points, thicknesses = [], [], []
        if self.robot == "ant":
            shape_body = self.model.shape_body.numpy()
            flags = self.model.shape_flags.numpy()
            transforms = self.model.shape_transform.numpy()
            scales, types = (
                self.model.shape_scale.numpy(),
                self.model.shape_type.numpy(),
            )
            margins = self.model.shape_margin.numpy()
            body_world = self.model.body_world.numpy()
            for world in range(self.num_envs):
                wb, wpnt, wt = [], [], []
                for index, body in enumerate(shape_body):
                    if (
                        body < 0
                        or body_world[body] != world
                        or not flags[index] & int(newton.ShapeFlags.COLLIDE_SHAPES)
                    ):
                        continue
                    if types[index] == int(newton.GeoType.SPHERE):
                        centers = torch.zeros(1, 3)
                    elif types[index] == int(newton.GeoType.CAPSULE):
                        centers = torch.tensor(
                            [
                                [0.0, 0.0, scales[index, 1]],
                                [0.0, 0.0, -scales[index, 1]],
                            ]
                        )
                    else:
                        raise ValueError(
                            "Ant contact features require sphere/capsule shapes"
                        )
                    transform = torch.tensor(transforms[index])
                    local = transform[:3] + quat_rotate(transform[3:], centers)
                    wb.extend([int(body)] * len(centers))
                    wpnt.extend(local.tolist())
                    wt.extend([float(scales[index, 0] + margins[index])] * len(centers))
                if len(wb) != 25:
                    raise ValueError("Ant requires 25 ordered contact slots")
                bodies.append(wb)
                points.append(wpnt)
                thicknesses.append(wt)
        self._contact_bodies = torch.tensor(
            bodies, device=self.device, dtype=torch.long
        )
        self._contact_points = torch.tensor(
            points, device=self.device, dtype=torch.float32
        )
        self._contact_thickness = torch.tensor(
            thicknesses, device=self.device, dtype=torch.float32
        )

    def snapshot(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        q = wp.to_torch(self.state.joint_q).view(self.num_envs, -1).clone()
        qd = wp.to_torch(self.state.joint_qd).view(self.num_envs, -1).clone()
        body_q = wp.to_torch(self.state.body_q)
        state = {"q": q, "qd": qd, "root_body_q": body_q[self._root_bodies].clone()}
        if self.robot == "cartpole":
            return state, {
                name: q.new_empty((self.num_envs, 0))
                for name in (
                    "contact_points_0",
                    "contact_points_1",
                    "contact_normals",
                    "contact_depths",
                    "contact_thicknesses_0",
                    "contact_thicknesses_1",
                )
            }
        poses = body_q[self._contact_bodies]
        points = poses[..., :3] + quat_rotate(poses[..., 3:], self._contact_points)
        contacts = plane_contact_features(
            points,
            self._contact_thickness,
            q.new_tensor([0.0, 0.0, 1.0]),
            q.new_zeros(()),
            1e-5,
        )
        contacts["contact_points_0"] = self._contact_points.flatten(1).clone()
        contacts.pop("contact_masks", None)
        return state, contacts

    def reset(
        self, q: torch.Tensor | None = None, qd: torch.Tensor | None = None
    ) -> None:
        import newton

        if q is None:
            q = torch.tensor(self.profile["default_q"], device=self.device).expand(
                self.num_envs, -1
            )
        if qd is None:
            qd = torch.tensor(self.profile["default_qd"], device=self.device).expand(
                self.num_envs, -1
            )
        if q.shape != (self.num_envs, self.profile["q_dim"]) or qd.shape != (
            self.num_envs,
            self.profile["qd_dim"],
        ):
            raise ValueError("Reset tensors must match the complete reference batch")
        if self._mujoco:
            self.solver.reset(self.state)
        for state in (self.state, self.next_state):
            wp.to_torch(state.joint_q).copy_(q.reshape(-1))
            wp.to_torch(state.joint_qd).copy_(qd.reshape(-1))
            state.clear_forces()
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        self.control.joint_f.zero_()

    def _advance(self) -> None:
        for _ in range(self.num_substeps):
            self.state.clear_forces()
            if not self._mujoco:
                self.pipeline.collide(self.state, self.contacts)
            self.solver.step(
                self.state,
                self.next_state,
                self.control,
                self.contacts,
                self.dt / self.num_substeps,
            )
            self.state, self.next_state = self.next_state, self.state
        if self.num_substeps % 2:
            # Publish into the stable input buffer used by captured kernels.
            self.next_state.assign(self.state)
            self.state, self.next_state = self.next_state, self.state

    def step(self, effort: torch.Tensor) -> None:
        if effort.shape != (self.num_envs, self.profile["qd_dim"]):
            raise ValueError("Effort must use the full reference generalized layout")
        if self.robot == "ant" and bool(torch.any(effort[:, :6] != 0)):
            raise ValueError("Ant sampling cannot apply free-base forces")
        self.control.joint_f.assign(wp.from_torch(effort.contiguous().reshape(-1)))
        if self.graph is None:
            self._advance()
        else:
            wp.capture_launch(self.graph)

    def __enter__(self) -> _NewtonTeacher:
        return self

    def __exit__(self, *_args: Any) -> None:
        # All arrays and solver buffers belong to this instance, with no World
        # singleton or native renderer requiring asynchronous destruction.
        self.graph = None
