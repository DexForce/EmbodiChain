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

"""Prepare assembly manifests as ordinary scene configuration."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from embodichain.toolkits.assembly import AssemblyAsset
from embodichain.lab.sim.assembly import to_rigid_config, to_articulation_configs
from embodichain.lab.sim.cfg import NewtonPhysicsCfg
from embodichain.lab.sim.sensors import ContactSensorCfg, ArticulationContactFilterCfg
from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.gym.envs.managers.cfg import EventCfg, ConditionCfg, RLTaskCfg
from .assets import build_task_assembly

__all__ = ["prepare_assembly_scene"]


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


class AssemblyScene:
    """Prepared geometry and grasp metadata, without runtime world ownership."""

    def __init__(
        self, cfg: EmbodiedEnvCfg, kind: str, *, materialize: bool = False
    ) -> None:
        self.cfg = cfg
        self.options = cfg.extensions or {}
        self.assembly_kind = kind
        self.materialize = materialize
        self.sensors = []
        self.prepare()

    def prepare(self) -> None:
        contact_capacities = {}
        for name, default in (
            ("assembly_contact_capacity", 2048),
            ("grasp_contact_capacity", 128),
        ):
            capacity = self.options.get(name, default)
            if (
                isinstance(capacity, bool)
                or not isinstance(capacity, int)
                or capacity < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer, got {capacity!r}."
                )
            contact_capacities[name] = capacity
        manifest = self.options.get("assembly_manifest", None)
        selected_seed = self.options.get("assembly_seed", None)
        if selected_seed is None:
            selected_seed = self.cfg.seed
        seed = 0 if selected_seed is None else int(selected_seed)
        if manifest:
            self.assembly_asset = AssemblyAsset.load(manifest)
        else:
            self.assembly_asset = build_task_assembly(
                self.assembly_kind, self.options.get("assembly_config", {}), seed=seed
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
        from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT

        self.directory = (
            Path(EMBODICHAIN_DEFAULT_DATA_ROOT)
            / "generated"
            / "assembly"
            / self.assembly_asset.content_hash
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self._moving_uid, self._fixture_uid = {
            "peg": ("peg", "plate"),
            "gear": ("gear_b", "shaft"),
            "nut": ("nut", "bolt"),
        }[self.assembly_kind]
        source_contact = bool(self.options.get("preserve_source_contacts", True))
        backend = (
            "newton"
            if isinstance(self.cfg.sim_cfg.physics_cfg, NewtonPhysicsCfg)
            else "default"
        )
        if self.materialize:
            for body in self.assembly_asset.bodies:
                if body.joint.kind != "hinge":
                    config = to_rigid_config(
                        body, source_contact=source_contact, backend=backend
                    )
                    if body.name != self._moving_uid:
                        config.body_type = "kinematic"
                    target = (
                        self.cfg.rigid_object
                        if config.body_type == "dynamic"
                        else self.cfg.background
                    )
                    target.append(config)
        if self.materialize:
            for config in to_articulation_configs(
                self.assembly_asset,
                self.directory,
                source_contact=source_contact,
                backend=backend,
            ).values():
                self.cfg.articulation.append(config)
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
        a, b, c = self.options.get(
            "grasp_aperture_coefficients", (0.0057, 0.037, -0.044)
        )
        aperture = lambda angle: 2 * (a + b * np.cos(angle) + c * np.sin(angle))
        upper = float(self.options.get("grasp_joint_upper", 0.8))
        self._assembly_grasp_links = list(
            self.options.get(
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
            self.options.get("gripper_model", "four_bar_mimic") == "parallel_jaw"
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
        self.sensors.extend(
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
                            articulation_uid=self.cfg.robot.uid,
                            link_name_list=self._assembly_grasp_links,
                        )
                    ],
                    filter_need_both_actor=True,
                    max_contacts_per_env=contact_capacities["grasp_contact_capacity"],
                ),
            ]
        )
        self.assembly_fidelity = {
            "profile": self.options.get("assembly_profile", "nominal"),
            "geometry_per_run": 1,
            "asset_sha256": self.assembly_asset.content_hash,
            "robot_source_urdf": getattr(
                self.cfg.robot, "_pk_urdf_path", self.cfg.robot.fpath
            ),
            "source_contacts_preserved": source_contact,
            "contact_capacities": contact_capacities,
            "paper_robot_dynamics_identified": False,
            "gripper_model": self.options.get("gripper_model", "custom"),
            "robot_asset_source": self.options.get("robot_asset_source", "default"),
            "grasp_contact_links": self._assembly_grasp_links,
            "source_gripper_dynamics_preserved": False,
            "reset_distribution": "local contact, fixed ready pose; workspace-wide IK reset not qualified",
            "contact_labels": (
                "public assembly_task.js peg signed-distance labels and seated interval; "
                "additional bilateral grasp, penetration and 3-control-step hold guards"
                if self.assembly_kind == "peg"
                else "analytic mating-feature labels; recorded engineering thresholds"
            ),
            "proprioception_convention": self.options.get(
                "proprioception_convention", "legacy_v1"
            ),
        }


def prepare_assembly_scene(cfg: EmbodiedEnvCfg, kind: str) -> EmbodiedEnvCfg:
    """Materialize assets and bind native managers before World creation.

    Args:
        cfg: Owned environment configuration.
        kind: Peg, gear or nut family.

    Returns:
        Configuration of ordinary entities, sensors and functors.
    """
    from ._state import initialize_assembly
    from ._control import AssemblyJointPositionAction
    from embodichain.lab.gym.envs.managers.assembly import (
        assembly_success,
        assembly_failure,
    )

    options = cfg.extensions or {}
    if options.get("require_heterogeneous_assets", False):
        raise RuntimeError(
            "Formal assembly training requires per-arena geometry/COM/inertia variants."
        )
    if options.get("require_exact_robot_dynamics", False):
        import json

        path = getattr(cfg.robot, "_pk_urdf_path", None) or getattr(
            cfg.robot, "fpath", None
        )
        marker = Path(path).with_name("fidelity.json") if path else None
        fidelity = json.loads(marker.read_text()) if marker and marker.is_file() else {}
        if not fidelity.get("exact_source_dynamics", False):
            raise ValueError(
                "Exact assembly training requires qualified robot dynamics."
            )
    cfg.extensions = dict(options, assembly_kind=kind)
    if cfg.rl_task is not None:
        raise ValueError("Assembly preparation owns its success/failure conditions.")
    if not cfg.enable_sensor:
        raise ValueError("Assembly tasks require mating and grasp contact sensors.")
    scene = AssemblyScene(cfg, kind, materialize=True)
    cfg.sensor.extend(scene.sensors)
    events = (
        cfg.events
        if isinstance(cfg.events, dict)
        else vars(cfg.events) if cfg.events is not None else {}
    )
    if "initialize_assembly" in events:
        raise ValueError("initialize_assembly is owned by assembly preparation.")
    cfg.events = {
        "initialize_assembly": EventCfg(func=initialize_assembly, mode="startup"),
        **events,
    }
    actions = cfg.actions if isinstance(cfg.actions, dict) else vars(cfg.actions)
    if set(actions) != {"arm_delta"}:
        raise ValueError(
            "Assembly control owns the arm and gripper; configure only arm_delta."
        )
    from embodichain.lab.gym.envs.managers.actions import RelativeJointPositionAction

    if actions["arm_delta"].func not in {
        RelativeJointPositionAction,
        AssemblyJointPositionAction,
    }:
        raise ValueError("Assembly requires measured joint-relative arm actions.")
    actions["arm_delta"].func = AssemblyJointPositionAction
    cfg.rl_task = RLTaskCfg(
        success={"assembly": ConditionCfg(func=assembly_success)},
        failure={"assembly": ConditionCfg(func=assembly_failure)},
    )
    return cfg
