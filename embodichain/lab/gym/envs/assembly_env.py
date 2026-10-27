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
"""Shared native-simulator lifecycle for robot-held precision assembly tasks."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, ClassVar

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import torch

from embodichain.gen_sim.assembly import AssemblyAsset
from embodichain.gen_sim.assembly.task_assets import build_task_assembly
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
from embodichain.lab.sim.sensors import ContactSensorCfg, ArticulationContactFilterCfg
from embodichain.utils.math import (
    matrix_from_quat,
    trans_matrix_to_xyz_quat,
    euler_xyz_from_quat,
    quat_from_matrix,
)

__all__ = ["BlindAssemblyEnv"]


def _assembly_kinematic_chain(robot_cfg: Any, device: torch.device) -> Any:
    """Include the application's solver TCP in encoder-only FK and Jacobians."""
    chain = robot_cfg.build_pk_serial_chain(device=torch.device("cpu"))["arm"]
    tcp = np.asarray(robot_cfg.solver_cfg["arm"].tcp)
    if not np.array_equal(tcp, np.eye(4)):
        import pytorch_kinematics as pk

        end_frame = chain.find_frame(chain.get_frame_names(False)[-1])
        name = end_frame.name + "_assembly_tcp"
        end_frame.add_child(
            pk.Frame(
                name=name,
                link=pk.Link(name=name),
                joint=pk.Joint(
                    offset=pk.Transform3d(
                        matrix=torch.as_tensor(tcp, dtype=chain.dtype)
                    ),
                    joint_type="fixed",
                ),
            )
        )
        chain = pk.SerialChain(chain, name, dtype=chain.dtype)
    return chain.to(device=device)


def _grasp_joint_reset(
    robot: Any, value: float, rows: int, device: torch.device
) -> tuple[list[int], torch.Tensor]:
    """Expand the EEF leaders through source mimic relations without name rules."""
    values = {int(i): value for i in robot.get_joint_ids(name="eef")}
    relations = list(
        zip(
            robot.mimic_ids,
            robot.mimic_parents,
            robot.mimic_multipliers,
            robot.mimic_offsets,
        )
    )
    for _ in range(len(relations) + 1):
        for child, parent, multiplier, offset in relations:
            if parent in values:
                values[int(child)] = values[parent] * multiplier + offset
    indices = sorted(values)
    return indices, torch.tensor(
        [values[i] for i in indices], dtype=torch.float32, device=device
    ).expand(rows, -1)


def _body_vertices(body: Any) -> np.ndarray:
    clouds = []
    for geometry in body.geometries:
        if not geometry.collision:
            continue
        rotation = Rotation.from_quat(geometry.quaternion).as_matrix()
        if geometry.kind == "mesh":
            vertices = np.asarray(geometry.vertices)
        elif geometry.kind == "cylinder":
            radius, half_height = geometry.size[:2]
            angle = np.arange(64) * 2 * np.pi / 64
            vertices = np.concatenate(
                [
                    np.column_stack(
                        [radius * np.cos(angle), radius * np.sin(angle), np.full(64, z)]
                    )
                    for z in (-half_height, half_height)
                ]
            )
        elif geometry.kind == "box":
            from itertools import product

            vertices = np.array(list(product((-1, 1), repeat=3))) * geometry.size[:3]
        else:
            raise ValueError(f"Unsupported assembly mating geometry {geometry.kind!r}.")
        clouds.append(vertices @ rotation.T + geometry.position)
    return np.concatenate(clouds)


def _perimeter_samples(polygon: np.ndarray, count: int) -> np.ndarray:
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    position = np.arange(count) * cumulative[-1] / count
    index = np.minimum(
        np.searchsorted(cumulative, position, side="right") - 1, len(edges) - 1
    )
    return (
        polygon[index]
        + edges[index] * ((position - cumulative[index]) / lengths[index])[:, None]
    )


class BlindAssemblyEnv(EmbodiedEnv):
    """Native Gym task base with real contact gripping and privileged labels.

    A development run holds one immutable geometry per run. Formal profiles
    require the heterogeneous-asset capability and qualified robot dynamics;
    they fail before World creation when those prerequisites are absent.
    Nominal deployments explicitly opt into public, uncalibrated robot assets.
    """

    assembly_kind: ClassVar[str]

    def _setup_scene(self, **kwargs: Any) -> None:
        if getattr(self, "require_exact_robot_dynamics", False):
            import json

            marker = Path(self.cfg.robot._pk_urdf_path).with_name("fidelity.json")
            fidelity = json.loads(marker.read_text()) if marker.is_file() else {}
            if not fidelity.get("exact_source_dynamics", False):
                raise ValueError(
                    "Exact assembly training requires qualified robot dynamics; "
                    "public URDFs and approximate gripper closed loops are not "
                    "an identified model. Select a development environment or "
                    "provide a physically qualified fidelity.json."
                )
        if getattr(self, "require_heterogeneous_assets", False):
            raise RuntimeError(
                "Formal assembly training requires per-arena geometry/COM/inertia "
                "variants. This task currently supports one assembly per run. "
                "Use task.*.nominal.yaml for explicit development or implement "
                "the missing heterogeneous Spawn capability before qualification."
            )
        super()._setup_scene(**kwargs)

    def _prepare_scene(self, **kwargs: Any) -> None:
        contact_capacities = {}
        for name, default in (
            ("assembly_contact_capacity", 2048),
            ("grasp_contact_capacity", 128),
        ):
            capacity = getattr(self, name, default)
            if (
                isinstance(capacity, bool)
                or not isinstance(capacity, int)
                or capacity < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer, got {capacity!r}."
                )
            contact_capacities[name] = capacity
        super()._prepare_scene(**kwargs)
        manifest = getattr(self, "assembly_manifest", None)
        selected_seed = getattr(self, "assembly_seed", None)
        if selected_seed is None:
            selected_seed = self.cfg.seed
        seed = 0 if selected_seed is None else int(selected_seed)
        self.assembly_asset = (
            AssemblyAsset.load(manifest)
            if manifest
            else build_task_assembly(
                self.assembly_kind, getattr(self, "assembly_config", {}), seed=seed
            )
        )
        if manifest is None:
            from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT

            cache = Path(EMBODICHAIN_DEFAULT_DATA_ROOT) / "generated" / "assembly"
            manifest = self.assembly_asset.save(
                cache / f"{self.assembly_asset.content_hash}.json"
            )
        self.assembly_manifest = str(Path(manifest).resolve())
        if self.cfg.extensions is not None:
            self.cfg.extensions["assembly_manifest"] = self.assembly_manifest
        self._assembly_directory = tempfile.TemporaryDirectory(
            prefix="embodichain-assembly-"
        )
        self._moving_uid, self._fixture_uid = {
            "peg": ("peg", "plate"),
            "gear": ("gear_b", "shaft"),
            "nut": ("nut", "bolt"),
        }[self.assembly_kind]
        source_contact = bool(getattr(self, "preserve_source_contacts", True))
        backend = "newton" if self.sim.is_newton_backend else "default"
        for body in self.assembly_asset.bodies:
            if body.joint.kind != "hinge":
                config = body.to_rigid_config(
                    source_contact=source_contact, backend=backend
                )
                if body.name != self._moving_uid:
                    config.body_type = "kinematic"
                self.sim.add_rigid_object(cfg=config)
        for config in self.assembly_asset.to_articulation_configs(
            self._assembly_directory.name,
            source_contact=source_contact,
            backend=backend,
        ).values():
            self.sim.add_articulation(cfg=config)
        bodies = {body.name: body for body in self.assembly_asset.bodies}
        self._moving_definition = bodies[self._moving_uid]
        self._fixture_definition = bodies[self._fixture_uid]
        vertices = _body_vertices(self._moving_definition)
        self._moving_zmin, self._moving_zmax = float(vertices[:, 2].min()), float(
            vertices[:, 2].max()
        )
        grip_vertices = vertices[
            vertices[:, 2]
            > self._moving_zmax - 0.2 * (self._moving_zmax - self._moving_zmin)
        ]
        width = float(np.ptp(grip_vertices[:, 1]))
        # These coefficients are the pinned public Robotiq four-bar geometry,
        # including the two inner pad faces. Bisection is preparation-only.
        a, b, c = getattr(self, "grasp_aperture_coefficients", (0.0057, 0.037, -0.044))
        aperture = lambda angle: 2 * (a + b * np.cos(angle) + c * np.sin(angle))
        upper = float(getattr(self, "grasp_joint_upper", 0.8))
        self._assembly_grasp_links = list(
            getattr(
                self,
                "grasp_link_names",
                ("gripper_right_pad", "gripper_left_pad"),
            )
        )
        if (
            len(self._assembly_grasp_links) != 2
            or len(set(self._assembly_grasp_links)) != 2
        ):
            raise ValueError("Assembly grasp requires two distinct contact links.")
        if width >= aperture(0) or width <= aperture(upper):
            raise ValueError(
                f"Grip width {width:g} m is outside the public 2F85 aperture."
            )
        low, high = 0.0, upper
        for _ in range(40):
            middle = (low + high) / 2
            if aperture(middle) > width + 0.0005:
                low = middle
            else:
                high = middle
        self._parallel_jaw = (
            getattr(self, "gripper_model", "four_bar_mimic") == "parallel_jaw"
        )
        self._grasp_reset_joint = (
            (aperture(0) - width - 0.0005) / 2
            if self._parallel_jaw
            else (low + high) / 2
        )
        self._grasp_target_joint = (
            min(0.0425, self._grasp_reset_joint + 0.0035)
            if self._parallel_jaw
            else min(upper, self._grasp_reset_joint + 0.08)
        )
        fixture_vertices = _body_vertices(self._fixture_definition)
        self._fixture_zmax = float(fixture_vertices[:, 2].max())
        bottom = vertices[np.isclose(vertices[:, 2], self._moving_zmin, atol=1e-7), :2]
        if len(bottom) >= 3:
            polygon = np.unique(bottom, axis=0)
            polygon = polygon[ConvexHull(polygon).vertices]
        else:
            angle = np.arange(64) * 2 * np.pi / 64
            polygon = np.column_stack([np.cos(angle), np.sin(angle)]) * 0.01
        self._edge_points = np.column_stack(
            [_perimeter_samples(polygon, 64), np.full(64, self._moving_zmin)]
        )
        self._bottom_vertices = np.column_stack(
            [polygon, np.full(len(polygon), self._moving_zmin)]
        )
        filters = (
            [
                ArticulationContactFilterCfg(
                    articulation_uid="gear_a", link_name_list=["gear_a"]
                )
            ]
            if self.assembly_kind == "gear"
            else []
        )
        self.cfg.sensor.extend(
            [
                ContactSensorCfg(
                    uid="assembly_contacts",
                    rigid_uid_list=[self._moving_uid, self._fixture_uid],
                    articulation_cfg_list=filters,
                    filter_need_both_actor=True,
                    max_contacts_per_env=contact_capacities[
                        "assembly_contact_capacity"
                    ],
                ),
                ContactSensorCfg(
                    uid="grasp_contacts",
                    rigid_uid_list=[self._moving_uid],
                    articulation_cfg_list=[
                        ArticulationContactFilterCfg(
                            articulation_uid=self.robot.uid,
                            link_name_list=self._assembly_grasp_links,
                        )
                    ],
                    filter_need_both_actor=True,
                    max_contacts_per_env=contact_capacities["grasp_contact_capacity"],
                ),
            ]
        )
        self.assembly_fidelity = {
            "profile": getattr(self, "assembly_profile", "nominal"),
            "geometry_per_run": 1,
            "asset_sha256": self.assembly_asset.content_hash,
            "robot_source_urdf": self.cfg.robot._pk_urdf_path,
            "source_contacts_preserved": source_contact,
            "contact_capacities": contact_capacities,
            "paper_robot_dynamics_identified": False,
            "gripper_model": getattr(self, "gripper_model", "custom"),
            "robot_asset_source": getattr(self, "robot_asset_source", "default"),
            "grasp_contact_links": self._assembly_grasp_links,
            "source_gripper_dynamics_preserved": False,
            "reset_distribution": "local contact, fixed ready pose; workspace-wide IK reset not qualified",
            "contact_labels": (
                "public assembly_task.js peg signed-distance labels and seated interval; "
                "additional bilateral grasp, penetration and 3-control-step hold guards"
                if self.assembly_kind == "peg"
                else "analytic mating-feature labels; recorded engineering thresholds"
            ),
            "proprioception_convention": getattr(
                self, "proprioception_convention", "legacy_v1"
            ),
        }

    def _init_sim_state(self, **kwargs: Any) -> None:
        self._assembly_revision = 0
        self._assembly_physics_revision = 0
        self._assembly_state_revision = -1
        self._assembly_state_physics_revision = -1
        self._assembly_reset_rows = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._assembly_chain = _assembly_kinematic_chain(self.cfg.robot, self.device)
        self._assembly_observation_convention = getattr(
            self, "proprioception_convention", "legacy_v1"
        )
        if self._assembly_observation_convention not in {"legacy_v1", "official_v1"}:
            raise ValueError("Unknown assembly proprioception_convention.")
        if self._assembly_observation_convention == "official_v1":
            limits = torch.as_tensor(
                self.proprioceptive_joint_limits,
                dtype=torch.float32,
                device=self.device,
            )
            velocity = torch.as_tensor(
                self.proprioceptive_velocity_scale,
                dtype=torch.float32,
                device=self.device,
            )
            if (
                limits.shape != (7, 2)
                or velocity.shape != (7,)
                or not torch.isfinite(limits).all()
                or not torch.isfinite(velocity).all()
                or not (limits[:, 0] < limits[:, 1]).all()
                or not (velocity > 0).all()
            ):
                raise ValueError(
                    "Assembly proprioception requires 7 finite joint scales."
                )
            self._assembly_q_center = limits.mean(-1)
            self._assembly_q_half_range = (limits[:, 1] - limits[:, 0]) * 0.5
            self._assembly_velocity_scale = velocity
        self._assembly_success_hold = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._assembly_milestones = torch.zeros(
            (self.num_envs, 4), dtype=torch.bool, device=self.device
        )
        self._assembly_new_milestones = torch.zeros(
            (self.num_envs, 4), dtype=torch.bool, device=self.device
        )
        self._assembly_angle = torch.zeros(self.num_envs, device=self.device)
        self._assembly_previous_yaw = torch.zeros(self.num_envs, device=self.device)
        self._assembly_capture_angle = torch.full(
            (self.num_envs,), float("nan"), device=self.device
        )
        self._assembly_capture_depth = torch.full(
            (self.num_envs,), float("nan"), device=self.device
        )
        self._assembly_labels = torch.zeros(
            (self.num_envs, 128 if self.assembly_kind == "gear" else 64),
            dtype=torch.long,
            device=self.device,
        )
        self._assembly_metrics: dict[str, torch.Tensor] = {}
        self._assembly_edge_points = torch.as_tensor(
            self._edge_points, dtype=torch.float32, device=self.device
        )
        self._assembly_bottom_vertices = torch.as_tensor(
            self._bottom_vertices, dtype=torch.float32, device=self.device
        )
        self._prepare_gravity_compensation()
        grasp_sensor = self.get_sensor("grasp_contacts")
        self._assembly_pad_actor_ids = grasp_sensor.get_actor_ids(
            self.robot.uid, self._assembly_grasp_links
        )
        self._assembly_part_actor_ids = grasp_sensor.get_actor_ids(self._moving_uid)[
            :, 0
        ]
        self._grasp_history = grasp_sensor.create_history(
            "bilateral_grasp",
            self._assembly_pad_actor_ids,
            counterpart_ids=self._assembly_part_actor_ids[:, None],
            force_threshold=0.05,
        )
        self.assembly_fidelity["grasp_guard"] = (
            "bilateral part-pad contact in the control interval, at most 5 ms current air time, force threshold 0.05 N"
        )
        super()._init_sim_state(**kwargs)
        arm_ids = self.robot.get_joint_ids(name="arm", remove_mimic=True)
        self.robot.set_joint_drive(
            stiffness=torch.full(
                (self.num_envs, 7),
                float(getattr(self, "joint_stiffness", 40.0)),
                device=self.device,
            ),
            damping=torch.full(
                (self.num_envs, 7),
                float(getattr(self, "joint_damping", 20.0)),
                device=self.device,
            ),
            joint_ids=arm_ids,
        )

    def proprioceptive_frame(self) -> torch.Tensor:
        """Encoder-only 33/36-value frame using FK and Jacobian times qvel."""
        cached = getattr(self, "_assembly_frame_cache", None)
        if cached is not None and cached[0] == self._assembly_revision:
            return cached[1]
        q = self.robot.get_qpos(name="arm")
        qvel = self.robot.get_qvel(name="arm")
        jacobian, transform = self._assembly_chain.jacobian(q, ret_eef_pose=True)
        official = self._assembly_observation_convention == "official_v1"
        if official:
            transform = self._assembly_robot_base @ transform
        position = transform[:, :3, 3]
        rotation = transform[:, :3, :3]
        orientation = (
            rotation[:, :, :2].transpose(1, 2).reshape(self.num_envs, 6)
            if self.assembly_kind == "nut"
            else torch.stack(euler_xyz_from_quat(quat_from_matrix(rotation)), dim=-1)
        )
        twist = (jacobian @ qvel.unsqueeze(-1)).squeeze(-1)
        if official:
            base_rotation = self._assembly_robot_base[:3, :3]
            linear = twist[:, :3] @ base_rotation.T
            angular_world = twist[:, 3:] @ base_rotation.T
            angular_local = (
                rotation.transpose(1, 2) @ angular_world.unsqueeze(-1)
            ).squeeze(-1)
            twist = torch.cat([linear, angular_local], dim=-1)
            q = (q - self._assembly_q_center) / self._assembly_q_half_range
            qvel = qvel / self._assembly_velocity_scale
        previous = (
            self.action_manager.get_term("arm_delta").raw_actions
            if self.action_manager is not None
            else torch.zeros_like(q)
        )
        frame = torch.cat([q, qvel, position, orientation, twist, previous], dim=-1)
        self._assembly_frame_cache = (self._assembly_revision, frame)
        return frame

    def reset_assembly_rows(self, env_ids: torch.Tensor) -> None:
        """Initialize a physical in-finger grasp without advancing other rows."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        n = len(ids)
        q = self.robot.get_qpos(name="arm")[ids]
        tcp = (
            self._assembly_robot_base[None]
            @ self._assembly_chain.forward_kinematics(q).get_matrix()
        )
        turn = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=self.device))
        rotation = tcp[:, :3, :3] @ turn
        # Keep the upper grip section between the physical pads. Gripper
        # closure remains a force-limited drive; no fixed grasp constraint.
        grip_z = (
            (float(self.assembly_asset.config["face_width"]) / 2 + 0.0085 + 0.01875)
            if self.assembly_kind == "gear"
            else self._moving_zmax - 0.012
        )
        part = torch.eye(4, device=self.device).repeat(n, 1, 1)
        part[:, :3, :3] = rotation
        part[:, :3, 3] = tcp[:, :3, 3] - (
            rotation
            @ torch.tensor([0.0, 0.0, grip_z], device=self.device)
            .expand(n, -1)
            .unsqueeze(-1)
        ).squeeze(-1)
        grasp_jitter = float(getattr(self, "grasp_position_range", 0.0025))
        # Along-pad displacement is along z; lateral x does not change the
        # closing aperture. A y displacement would create an invalid initial
        # one-sided clamp rather than the paper's along-pad randomization.
        part[:, :3, 3] += (
            rotation
            @ (
                (torch.rand((n, 3), device=self.device) - 0.5)
                * torch.tensor([0.0, 0.0, 2 * grasp_jitter], device=self.device)
            ).unsqueeze(-1)
        ).squeeze(-1)
        fixture = part.clone()
        initial_offset = float(
            getattr(
                self,
                "initial_position_range",
                {"peg": 0.0125, "gear": 0.0075, "nut": 0.015}[self.assembly_kind],
            )
        )
        offset = (torch.rand((n, 3), device=self.device) - 0.5) * torch.tensor(
            [2 * initial_offset, 2 * initial_offset, 0.0], device=self.device
        )
        offset[:, 2] = self._moving_zmin - self._fixture_zmax
        fixture[:, :3, 3] += (rotation @ offset.unsqueeze(-1)).squeeze(-1)
        self.sim.get_rigid_object(self._moving_uid).set_local_pose(part, env_ids=ids)
        self.sim.get_rigid_object(self._moving_uid).set_velocity(
            lin_vel=torch.zeros((n, 3), device=self.device),
            ang_vel=torch.zeros((n, 3), device=self.device),
            env_ids=ids,
        )
        self.sim.get_rigid_object(self._fixture_uid).set_local_pose(
            fixture, env_ids=ids
        )
        if self.assembly_kind == "gear":
            # Every fixture entity is transformed rigidly with the shaft.
            anchor = self.sim.get_articulation("gear_a")
            source_shaft = np.asarray(self._fixture_definition.position)
            source_anchor = next(
                b for b in self.assembly_asset.bodies if b.name == "gear_a"
            )
            local = torch.as_tensor(
                np.asarray(source_anchor.position) - source_shaft,
                dtype=torch.float32,
                device=self.device,
            )
            anchor_pose = fixture.clone()
            anchor_pose[:, :3, 3] += (
                rotation @ local.expand(n, -1).unsqueeze(-1)
            ).squeeze(-1)
            anchor.set_local_pose(anchor_pose, env_ids=ids)
            anchor.set_qpos(
                torch.rand((n, 1), device=self.device) * 2 * torch.pi,
                env_ids=ids,
                target=False,
            )
        grip_ids, grip_values = _grasp_joint_reset(
            self.robot, self._grasp_reset_joint, n, self.device
        )
        self.robot.set_qpos(
            grip_values,
            joint_ids=grip_ids,
            env_ids=ids,
            target=False,
        )
        self.robot.set_qpos(
            torch.full(
                (n, len(self.robot.get_joint_ids(name="eef"))),
                self._grasp_target_joint,
                device=self.device,
            ),
            name="eef",
            env_ids=ids,
        )
        self._assembly_reset_rows[ids] = True
        self._assembly_success_hold[ids] = 0
        self._assembly_milestones[ids] = False
        self._assembly_new_milestones[ids] = False
        self._assembly_angle[ids] = 0
        self._assembly_previous_yaw[ids] = 0
        self._assembly_capture_angle[ids] = float("nan")
        self._assembly_capture_depth[ids] = float("nan")
        self._assembly_revision += 1

    def _prepare_gravity_compensation(self) -> None:
        import xml.etree.ElementTree as ET
        import pytorch_kinematics as pk

        xml = Path(self.cfg.robot._pk_urdf_path).read_bytes()
        self._gravity_chain = pk.build_chain_from_urdf(xml).to(device=self.device)
        names = self._gravity_chain.get_joint_parameter_names()
        self._gravity_joint_ids = torch.tensor(
            [self.robot.joint_names.index(name) for name in names], device=self.device
        )
        self._gravity_arm_ids = torch.tensor(
            [names.index(f"joint{i}") for i in range(1, 8)], device=self.device
        )
        self._gravity_links = []
        for link in ET.fromstring(xml).findall("link"):
            inertia = link.find("inertial")
            if inertia is not None:
                origin = inertia.find("origin")
                com = (
                    np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
                    if origin is not None
                    else np.zeros(3)
                )
                self._gravity_links.append(
                    (
                        link.attrib["name"],
                        float(inertia.find("mass").attrib["value"]),
                        torch.tensor(com, dtype=torch.float32, device=self.device),
                    )
                )
        gravity = np.asarray(self.sim_cfg.physics_cfg.gravity, dtype=float)
        if gravity.ndim == 0:
            gravity = np.array([0, 0, float(gravity)])
        local_pose = getattr(self.cfg.robot, "init_local_pose", None)
        root_rotation = (
            np.asarray(local_pose, dtype=float)[:3, :3]
            if local_pose is not None
            else Rotation.from_euler(
                "xyz", self.cfg.robot.init_rot, degrees=True
            ).as_matrix()
        )
        root_transform = (
            np.asarray(local_pose, dtype=float).copy()
            if local_pose is not None
            else np.eye(4)
        )
        if local_pose is None:
            root_transform[:3, :3] = root_rotation
            root_transform[:3, 3] = getattr(self.cfg.robot, "init_pos", [0, 0, 0])
        self._assembly_robot_base = torch.tensor(
            root_transform, dtype=torch.float32, device=self.device
        )
        self._gravity_acceleration = torch.tensor(
            root_rotation.T @ gravity, dtype=torch.float32, device=self.device
        )
        self.assembly_fidelity["gravity_compensation"] = (
            "URDF COM potential gradient from encoders, held at 50 Hz"
        )

    def _gravity_torques(self) -> torch.Tensor:
        # One batched differentiation per control step. The chain and COMs
        # are prepared once; no host transfers or simulator truth enter it.
        with torch.enable_grad():
            q = (
                self.robot.get_qpos()[:, self._gravity_joint_ids]
                .detach()
                .requires_grad_(True)
            )
            transforms = self._gravity_chain.forward_kinematics(q)
            potential = torch.zeros(self.num_envs, device=self.device)
            for name, mass, com in self._gravity_links:
                transform = transforms[name].get_matrix()
                position = transform[:, :3, :3] @ com + transform[:, :3, 3]
                potential = potential - mass * (
                    position * self._gravity_acceleration
                ).sum(dim=-1)
            torque = torch.autograd.grad(potential.sum(), q)[0]
        return torque[:, self._gravity_arm_ids].detach()

    def _preprocess_action(self, action: Any) -> Any:
        if isinstance(action, torch.Tensor):
            self.robot.set_qf(self._gravity_torques(), name="arm")
            action = action.clone()
            action[
                self._elapsed_steps < int(getattr(self, "grasp_warmup_steps", 10))
            ] = 0
            self.robot.set_qpos(
                torch.full(
                    (self.num_envs, len(self.robot.get_joint_ids(name="eef"))),
                    self._grasp_target_joint,
                    device=self.device,
                ),
                name="eef",
            )
        return super()._preprocess_action(action)

    def _update_sim_state(self, **kwargs: Any) -> None:
        self._assembly_revision += 1
        self._assembly_physics_revision += 1
        super()._update_sim_state(**kwargs)

    def _contact_grasp(self) -> torch.Tensor:
        sensor = self.get_sensor("grasp_contacts")
        if sensor.dropped_contacts:
            raise RuntimeError(
                "Grasp contact buffer overflow; increase the configured capacity."
            )
        history = getattr(self, "_grasp_history", None)
        if history is not None:
            return history.found.all(dim=1) & (history.current_air_time <= 0.005).all(
                dim=1
            )
        data = sensor.get_data()
        if not hasattr(self, "_assembly_pad_actor_ids"):
            self._assembly_pad_actor_ids = sensor.get_actor_ids(
                self.robot.uid,
                getattr(
                    self,
                    "_assembly_grasp_links",
                    ["gripper_right_pad", "gripper_left_pad"],
                ),
            )
            self._assembly_part_actor_ids = sensor.get_actor_ids(self._moving_uid)[:, 0]
        robot_ids = self._assembly_pad_actor_ids
        part_ids = self._assembly_part_actor_ids
        pairs = data["user_ids"]
        valid = data["is_valid"] & (data["distance"] <= 0.0005)
        touches = []
        for side in range(2):
            pad = robot_ids[:, side]
            match = (
                (pairs[:, :, 0] == pad[:, None]) & (pairs[:, :, 1] == part_ids[:, None])
            ) | (
                (pairs[:, :, 1] == pad[:, None]) & (pairs[:, :, 0] == part_ids[:, None])
            )
            touches.append((valid & match).any(dim=1))
        return touches[0] & touches[1]

    def refresh_assembly_state(self) -> None:
        """Compute supervision and milestone transitions once per physical step."""
        if self._assembly_state_revision == self._assembly_revision:
            return
        self._assembly_state_revision = self._assembly_revision
        update_rows = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if self._assembly_state_physics_revision != self._assembly_physics_revision
            else self._assembly_reset_rows.clone()
        )
        self._assembly_state_physics_revision = self._assembly_physics_revision
        part = self.sim.get_rigid_object(self._moving_uid).get_local_pose(
            to_matrix=True
        )
        fixture = self.sim.get_rigid_object(self._fixture_uid).get_local_pose(
            to_matrix=True
        )
        # Rigid inverse avoids a general LU solve and its CUDA error-check
        # synchronization on every control observation.
        inverse_rotation = fixture[:, :3, :3].transpose(1, 2)
        relative = torch.zeros_like(part)
        relative[:, :3, :3] = inverse_rotation @ part[:, :3, :3]
        relative[:, :3, 3] = (
            inverse_rotation @ (part[:, :3, 3] - fixture[:, :3, 3]).unsqueeze(-1)
        ).squeeze(-1)
        relative[:, 3, 3] = 1
        points = self._assembly_edge_points.expand(self.num_envs, -1, -1)
        points = points @ relative[:, :3, :3].transpose(1, 2) + relative[:, None, :3, 3]
        radial = torch.linalg.vector_norm(relative[:, :2, 3], dim=-1)
        tilt = torch.acos(relative[:, 2, 2].clamp(-1, 1))
        sensor = self.get_sensor("assembly_contacts")
        if sensor.dropped_contacts:
            raise RuntimeError(
                "Assembly contact buffer overflow; increase the configured capacity."
            )
        contacts = sensor.get_data()
        penetration = torch.where(
            contacts["is_valid"], (-contacts["distance"]).clamp_min(0), 0
        ).amax(dim=1)
        grasped = self._contact_grasp()
        clearance = float(self.assembly_asset.config.get("clearance", 0.0005))
        penetration_ok = penetration <= max(0.0001, clearance * 0.5)
        warm = self._elapsed_steps >= int(getattr(self, "grasp_warmup_steps", 10))
        yaw = torch.atan2(relative[:, 1, 0], relative[:, 0, 0])
        delta = torch.atan2(
            torch.sin(yaw - self._assembly_previous_yaw),
            torch.cos(yaw - self._assembly_previous_yaw),
        )
        delta[self._assembly_reset_rows] = 0
        self._assembly_angle += delta * update_rows
        self._assembly_previous_yaw.copy_(
            torch.where(update_rows, yaw, self._assembly_previous_yaw)
        )
        self._assembly_new_milestones.zero_()
        depth = self._fixture_zmax - points[:, :, 2].amax(dim=1)
        capture = torch.zeros_like(grasped)
        success = torch.zeros_like(grasped)
        if self.assembly_kind == "peg":
            if not hasattr(self, "_hole_polygon"):
                outline = self.assembly_asset.config.get("mating_outline")
                if outline is None:
                    raise ValueError(
                        "Peg task manifests require the generated mating_outline for dependency-free labeling."
                    )
                self._hole_polygon = torch.tensor(
                    outline, dtype=torch.float32, device=self.device
                )
            # Supervision samples 64 points; success checks every actual
            # convex bottom vertex, so corners between samples cannot escape.
            bottom = getattr(
                self, "_assembly_bottom_vertices", self._assembly_edge_points
            )
            bottom_points = (
                bottom[None] @ relative[:, :3, :3].transpose(1, 2)
                + relative[:, None, :3, 3]
            )
            # Match signedDistance/labelEdge in the authors' public
            # static/js/assembly_task.js: segment distance, even-odd interior,
            # and a 1 mm band around both the opening rim and top plane.
            # Bound point-edge temporaries independently of arena count.
            polygon = self._hole_polygon
            endpoint = polygon.roll(-1, 0)
            edges = endpoint - polygon
            length_squared = edges.square().sum(dim=-1).clamp_min(1e-12)
            samples = torch.cat([points[:, :, :2], bottom_points[:, :, :2]], dim=1)
            distances = []
            for chunk in samples.reshape(-1, 2).split(16384):
                rx = chunk[:, 0, None] - polygon[None, :, 0]
                ry = chunk[:, 1, None] - polygon[None, :, 1]
                projection = (
                    (rx * edges[None, :, 0] + ry * edges[None, :, 1])
                    / length_squared[None]
                ).clamp(0, 1)
                distance = (
                    (
                        (rx - projection * edges[None, :, 0]).square()
                        + (ry - projection * edges[None, :, 1]).square()
                    )
                    .amin(dim=-1)
                    .sqrt()
                )
                crossings = (
                    (
                        (polygon[None, :, 1] > chunk[:, 1, None])
                        != (endpoint[None, :, 1] > chunk[:, 1, None])
                    )
                    & (
                        chunk[:, 0, None]
                        < edges[None, :, 0]
                        * (chunk[:, 1, None] - polygon[None, :, 1])
                        / (edges[None, :, 1] + 1e-12)
                        + polygon[None, :, 0]
                    )
                ).sum(dim=-1)
                distances.append(torch.where(crossings % 2 == 1, -distance, distance))
            signed_distance = torch.cat(distances).reshape(samples.shape[:2])
            edge_distance = signed_distance[:, : points.shape[1]]
            all_inside = (signed_distance[:, points.shape[1] :] <= 0).all(dim=-1)
            depth = self._fixture_zmax - bottom_points[:, :, 2].amax(dim=1)
            band = 0.001
            height = points[:, :, 2] - self._fixture_zmax
            near_surface = height.abs() <= band
            near_edge = edge_distance.abs() <= band
            inside = ((height <= band) & (edge_distance < -band)) | (
                (height <= 0) & near_edge
            )
            contact = near_surface & ((edge_distance > band) | near_edge)
            self._assembly_labels = torch.where(
                contact, 1, torch.where(inside, 2, 0)
            ).long()
            capture = all_inside & (depth > 0)
            target_bottom_z = self._fixture_zmax - float(
                self.assembly_asset.config["plate_thickness"]
            )
            # The public task accepts every bottom point within 1 mm of the
            # seated plane. Grasp, penetration and the three-step hold below
            # remain additional project guards, separate from that geometry.
            success = all_inside & (
                (bottom_points[:, :, 2] - target_bottom_z).abs() <= band
            ).all(dim=-1)
        elif self.assembly_kind == "gear":
            cfg = self.assembly_asset.config
            bore = float(cfg["shaft_radius"]) + float(cfg["bore_clearance"])
            axis = torch.tensor(
                [0.0, 0.0, self._fixture_zmax], device=self.device
            ).expand(self.num_envs, -1)
            local_tip = (axis - relative[:, :3, 3]).unsqueeze(1) @ relative[:, :3, :3]
            aligned = torch.linalg.vector_norm(local_tip[:, 0, :2], dim=-1) < float(
                cfg["bore_clearance"]
            )
            capture = aligned & (depth > 0)
            bottom = points[:, :, 2].mean(dim=1)
            success = capture & (bottom.abs() < 0.001) & (tilt < 0.05)
            rim = torch.arange(64, device=self.device) * 2 * torch.pi / 64
            shaft_pts = torch.stack(
                [
                    torch.cos(rim) * float(cfg["shaft_radius"]),
                    torch.sin(rim) * float(cfg["shaft_radius"]),
                    torch.full_like(rim, self._fixture_zmax),
                ],
                dim=-1,
            )
            in_part = (shaft_pts[None] - relative[:, None, :3, 3]) @ relative[:, :3, :3]
            in_bore = torch.linalg.vector_norm(in_part[:, :, :2], dim=-1) < bore
            rim_labels = torch.where(
                in_part[:, :, 2] < self._moving_zmin - 0.001,
                0,
                torch.where(in_bore, 2, 1),
            )
            contact = contacts["is_valid"].any(dim=1)
            tooth_labels = torch.where(
                (depth > 0)[:, None],
                torch.where(
                    contact[:, None],
                    torch.ones_like(points[:, :, 0], dtype=torch.long),
                    torch.full_like(points[:, :, 0], 2, dtype=torch.long),
                ),
                0,
            )
            self._assembly_labels = torch.cat([rim_labels, tooth_labels], dim=1).long()
        else:
            cfg = self.assembly_asset.config
            pitch = float(cfg["pitch"])
            near_axis = radial < 0.001
            advance = depth.clamp_min(0)
            capture = near_axis & (advance > 0.29 * pitch) & (tilt < 0.15)
            first_capture = (
                capture & ~torch.isfinite(self._assembly_capture_angle) & update_rows
            )
            self._assembly_capture_angle[first_capture] = self._assembly_angle[
                first_capture
            ]
            self._assembly_capture_depth[first_capture] = advance[first_capture]
            turned = (self._assembly_angle - self._assembly_capture_angle).abs()
            axial_progress = advance - self._assembly_capture_depth
            expected_progress = pitch * turned / (2 * torch.pi)
            # Require new axial advance *after* first capture. Rotating at a
            # fixed captured depth is not threading. The 0.1-pitch band is an
            # explicit nominal tolerance, not an author calibration.
            success = (
                capture
                & (turned >= torch.pi / 2)
                & (axial_progress >= pitch * 0.2)
                & ((axial_progress - expected_progress).abs() <= pitch * 0.1)
            )
            free = points[:, :, 2] > self._fixture_zmax + 0.29 * pitch
            around = near_axis[:, None] & (points[:, :, 2] < self._fixture_zmax)
            self._assembly_labels = torch.where(
                free, 0, torch.where(capture[:, None], 3, torch.where(around, 2, 1))
            ).long()
        safe = warm & grasped & penetration_ok
        next_hold = torch.where(success & safe, self._assembly_success_hold + 1, 0)
        self._assembly_success_hold = torch.where(
            update_rows, next_hold, self._assembly_success_hold
        )
        stable_success = self._assembly_success_hold >= 3
        milestones = (
            torch.stack(
                [
                    capture,
                    capture & (depth > 0.001),
                    stable_success,
                    torch.zeros_like(success),
                ],
                dim=-1,
            )
            & safe[:, None]
        )
        self._assembly_new_milestones.copy_(
            milestones & ~self._assembly_milestones & update_rows[:, None]
        )
        self._assembly_milestones |= milestones & update_rows[:, None]
        failure = (torch.linalg.vector_norm(relative[:, :3, 3], dim=-1) > 0.25) | (
            ~grasped & (self._elapsed_steps > 25)
        )
        self._assembly_metrics = {
            "radial_error": radial,
            "tilt": tilt,
            "depth": depth,
            "penetration": penetration,
            "grasped": grasped,
            "captured": capture,
            "success": self._assembly_success_hold >= 3,
            "fail": failure,
            "relative_pose": trans_matrix_to_xyz_quat(relative),
        }

    def compute_task_state(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Require seated mating geometry, retained grasp and penetration bounds."""
        self.refresh_assembly_state()
        return (
            self._assembly_metrics["success"],
            self._assembly_metrics["fail"],
            {
                key: value
                for key, value in self._assembly_metrics.items()
                if key not in {"success", "fail", "relative_pose"}
            },
        )

    def export_checkpoint_gym_config(
        self, config: dict[str, Any], config_dir: Path
    ) -> dict[str, Any]:
        """Freeze generated geometry and the prepared robot into the RUN.

        Relative resource references are resolved by the ordinary Gym loader.
        Replay therefore uses this exact model preparation and needs neither
        mjfit nor the source-model cache. The bundle records its engineering
        fidelity and never upgrades the nominal model's qualification.
        """
        from copy import deepcopy
        import json
        import shutil

        frozen = deepcopy(config)
        resources = Path(config_dir) / "resources"
        assembly_directory = resources / "assembly"
        manifest_name = f"{self.assembly_asset.content_hash}.json"
        self.assembly_asset.save(assembly_directory / manifest_name)
        (assembly_directory / "fidelity.json").write_text(
            json.dumps(self.assembly_fidelity, indent=2) + "\n"
        )
        source = Path(self.cfg.robot._pk_urdf_path).resolve()
        robot_directory = resources / "robot"
        if source.parent.resolve() != robot_directory.resolve():
            shutil.copytree(source.parent, robot_directory, dirs_exist_ok=True)
        # Preserve upstream license texts and immutable resource identities,
        # even though generated mesh files live in a separate cache directory.
        licenses = robot_directory / "licenses"
        licenses.mkdir(parents=True, exist_ok=True)
        for model in ("flexiv_rizon4s", "robotiq_2f85"):
            license_path = source.parent.parent / model / "LICENSE"
            if license_path.is_file():
                shutil.copy2(license_path, licenses / f"{model}.LICENSE")
        source_manifest = source.parent.parent / "source_manifest.json"
        if source_manifest.is_file():
            shutil.copy2(source_manifest, robot_directory / "source_manifest.json")
            marker = robot_directory / "fidelity.json"
            if marker.is_file():
                fidelity = json.loads(marker.read_text())
                fidelity["source_manifest"] = "./source_manifest.json"
                marker.write_text(json.dumps(fidelity, indent=2) + "\n")
        # Preserve declared robot physics as well as geometry. The task's
        # explicit joint_stiffness/joint_damping fields still own its native
        # impedance override at the normal initialization boundary.
        for name in ("joint_drive_props", "root_props", "attrs"):
            value = getattr(self.cfg.robot, name, None)
            if value is not None:
                frozen["robot"][name] = value.to_dict()
        frozen["robot"]["link_attrs"] = {
            name: value.to_dict()
            for name, value in (
                getattr(self.cfg.robot, "link_attrs", None) or {}
            ).items()
        }
        prepared = f"./resources/robot/{source.name}"
        frozen["robot"]["fpath"] = prepared
        frozen["robot"].pop("urdf_cfg", None)
        for solver in frozen["robot"].get("solver_cfg", {}).values():
            solver["urdf_path"] = prepared
        frozen.setdefault("env", {}).setdefault("extensions", {})[
            "assembly_manifest"
        ] = f"./resources/assembly/{manifest_name}"
        return frozen

    def close(self, *, exit_process: bool | None = None) -> None:
        """Destroy the simulation before removing temporary hinge assets."""
        try:
            super().close(exit_process=exit_process)
        finally:
            directory = getattr(self, "_assembly_directory", None)
            if directory is not None:
                directory.cleanup()
