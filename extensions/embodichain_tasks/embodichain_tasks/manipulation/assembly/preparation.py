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
from embodichain.toolkits.assembly.native import (
    to_rigid_config,
    to_articulation_configs,
)
from embodichain.lab.sim.cfg import NewtonPhysicsCfg
from embodichain.lab.sim.sensors import ContactSensorCfg, ArticulationContactFilterCfg
from embodichain.lab.gym.envs import EmbodiedEnvCfg
from ._config import assembly_cfg
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

    def __init__(self, cfg: EmbodiedEnvCfg, *, materialize: bool = False) -> None:
        self.cfg = cfg
        self.task_cfg = assembly_cfg(cfg.extensions)
        self.assembly_kind = self.task_cfg.kind
        self.materialize = materialize
        self.sensors = []
        self.prepare()

    def prepare(self) -> None:
        contact_capacities = {
            "assembly_contact_capacity": self.task_cfg.assembly_contact_capacity,
            "grasp_contact_capacity": self.task_cfg.grasp_contact_capacity,
        }
        manifest = self.task_cfg.assembly_manifest
        selected_seed = self.task_cfg.assembly_seed
        if selected_seed is None:
            selected_seed = self.cfg.seed
        seed = 0 if selected_seed is None else int(selected_seed)
        if manifest:
            self.assembly_asset = AssemblyAsset.load(manifest)
        else:
            self.assembly_asset = build_task_assembly(
                self.assembly_kind, self.task_cfg.assembly_config, seed=seed
            )
        if manifest is None:
            from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT

            cache = Path(EMBODICHAIN_DEFAULT_DATA_ROOT) / "generated" / "assembly"
            manifest = self.assembly_asset.save(
                cache / f"{self.assembly_asset.content_hash}.json"
            )
        self.assembly_manifest = str(Path(manifest).resolve())
        if self.cfg.extensions is not None:
            self.cfg.extensions["assembly"][
                "assembly_manifest"
            ] = self.assembly_manifest
        from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT

        self.directory = (
            Path(EMBODICHAIN_DEFAULT_DATA_ROOT)
            / "generated"
            / "assembly"
            / self.assembly_asset.content_hash
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self._moving_uid = self.task_cfg.moving_entity_cfg.uid
        self._fixture_uid = self.task_cfg.fixture_entity_cfg.uid
        self._hinge_uid = (
            None
            if self.task_cfg.hinge_entity_cfg is None
            else self.task_cfg.hinge_entity_cfg.uid
        )
        if self.task_cfg.robot_entity_cfg.uid != self.cfg.robot.uid:
            raise ValueError(
                "Assembly robot selector must match the configured embodiment."
            )
        source_contact = self.task_cfg.preserve_source_contacts
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
        a, b, c = self.task_cfg.grasp_aperture_coefficients
        aperture = lambda angle: 2 * (a + b * np.cos(angle) + c * np.sin(angle))
        upper = self.task_cfg.grasp_joint_upper
        self._assembly_grasp_links = self.task_cfg.grasp_link_names
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
        self._parallel_jaw = self.task_cfg.gripper_model == "parallel_jaw"
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
            raise ValueError(
                "Assembly moving body requires a nondegenerate bottom outline."
            )
        self._edge_points = np.column_stack(
            [
                _perimeter_samples(polygon, self.task_cfg.contact_points_per_ring),
                np.full(self.task_cfg.contact_points_per_ring, self._moving_zmin),
            ]
        )
        self._bottom_vertices = np.column_stack(
            [polygon, np.full(len(polygon), self._moving_zmin)]
        )
        filters = (
            [
                ArticulationContactFilterCfg(
                    articulation_uid=self._hinge_uid, link_name_list=[self._hinge_uid]
                )
            ]
            if self.assembly_kind == "gear"
            else []
        )
        self.sensors.extend(
            [
                ContactSensorCfg(
                    uid=self.task_cfg.mating_sensor,
                    rigid_uid_list=[self._moving_uid, self._fixture_uid],
                    articulation_cfg_list=filters,
                    filter_need_both_actor=True,
                    max_contacts_per_env=contact_capacities[
                        "assembly_contact_capacity"
                    ],
                ),
                ContactSensorCfg(
                    uid=self.task_cfg.grasp_sensor,
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
            "profile": self.task_cfg.assembly_profile,
            "geometry_per_run": 1,
            "asset_sha256": self.assembly_asset.content_hash,
            "robot_source_urdf": getattr(
                self.cfg.robot, "_pk_urdf_path", self.cfg.robot.fpath
            ),
            "source_contacts_preserved": source_contact,
            "contact_capacities": contact_capacities,
            "paper_robot_dynamics_identified": False,
            "gripper_model": self.task_cfg.gripper_model,
            "robot_asset_source": self.task_cfg.robot_asset_source,
            "grasp_contact_links": self._assembly_grasp_links,
            "source_gripper_dynamics_preserved": False,
            "reset_distribution": "local contact, fixed ready pose; workspace-wide IK reset not qualified",
            "contact_labels": (
                "public assembly_task.js peg signed-distance labels and seated interval; "
                "additional bilateral grasp, penetration and 3-control-step hold guards"
                if self.assembly_kind == "peg"
                else "analytic mating-feature labels; recorded engineering thresholds"
            ),
            "proprioception_convention": "official_v1",
        }


def prepare_assembly_scene(cfg: EmbodiedEnvCfg) -> EmbodiedEnvCfg:
    """Expand configured assets and sensors before World creation.

    Args:
        cfg: Owned configuration with explicit assembly, manager and RL declarations.

    Returns:
        Configuration with generated native entities and contact sensors.
    """
    task = assembly_cfg(cfg.extensions)
    if task.require_heterogeneous_assets:
        raise RuntimeError(
            "Formal assembly training requires per-arena geometry/COM/inertia variants."
        )
    if task.require_exact_robot_dynamics:
        import json

        path = getattr(cfg.robot, "_pk_urdf_path", None) or cfg.robot.fpath
        marker = Path(path).with_name("fidelity.json") if path else None
        fidelity = json.loads(marker.read_text()) if marker and marker.is_file() else {}
        if not fidelity.get("exact_source_dynamics", False):
            raise ValueError(
                "Exact assembly training requires qualified robot dynamics."
            )
    if not cfg.enable_sensor:
        raise ValueError("Assembly tasks require mating and grasp contact sensors.")
    if cfg.rl_task is None:
        raise ValueError("Assembly requires explicit configured rl_task conditions.")
    for part in (task.arm_part, task.gripper_part):
        if part not in cfg.robot.control_parts:
            raise ValueError(
                f"Assembly control part {part!r} is absent from the robot."
            )
    actions = cfg.actions if isinstance(cfg.actions, dict) else vars(cfg.actions)
    if set(actions) != {task.action_term}:
        raise ValueError(
            "Assembly requires one explicitly named arm/gripper action owner."
        )
    if actions[task.action_term].params.get("part_name") != task.arm_part:
        raise ValueError(
            "Assembly action selection must match the configured arm_part."
        )
    scene = AssemblyScene(cfg, materialize=True)
    cfg.sensor.extend(scene.sensors)
    return cfg
