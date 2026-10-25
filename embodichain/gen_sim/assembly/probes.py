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
"""Constrained precision probes with analytic circle/square ground truth.

S0 prescribes static poses only. S2-S4 integrate real slide/hinge constraints
under constant force or torque; no pose is rewritten during integration.
The plate uses the imported mjfit convex pieces and the probe peg is an exact
primitive, so errors are charged against analytic outlines, not mesh samples.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np

from embodichain.utils import configclass
from .model import AssemblyAsset

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import ArticulationCfg, RigidObjectCfg

__all__ = [
    "AnalyticPegGeometry",
    "ConstrainedProbeCfg",
    "ConstrainedProbeDefinition",
    "prepare_probe",
    "run_reference_probe",
    "run_native_probe",
    "run_warp_probe",
]


def _settled_error_statistics(errors: np.ndarray, tolerance: float) -> dict[str, Any]:
    """Score the whole settled window, including worst error and secular drift.

    Both the largest absolute wall error and the difference between the first
    and last quarter means must fit the same physical accuracy budget. A zero
    median cannot certify a large oscillation or a continuing drift.
    """
    errors = np.asarray(errors, dtype=float).reshape(-1)
    finite = bool(errors.size and np.isfinite(errors).all())
    if not finite:
        return {
            "wall_error_m": None,
            "wall_error_min_m": None,
            "wall_error_max_m": None,
            "wall_worst_error_m": None,
            "wall_jitter_m": None,
            "wall_drift_m": None,
            "finite": False,
            "passed": False,
        }
    quarter = max(1, len(errors) // 4)
    drift = float(errors[-quarter:].mean() - errors[:quarter].mean())
    worst = float(np.abs(errors).max())
    return {
        "wall_error_m": float(np.median(errors)),
        "wall_error_min_m": float(errors.min()),
        "wall_error_max_m": float(errors.max()),
        "wall_worst_error_m": worst,
        "wall_jitter_m": float(errors.std()),
        "wall_drift_m": drift,
        "finite": True,
        "passed": worst <= tolerance and abs(drift) <= tolerance,
    }


def _contact_overlap_worlds(
    geom_body_ids: np.ndarray,
    geom_pairs: np.ndarray,
    world_ids: np.ndarray,
    distances: np.ndarray,
    num_worlds: int,
    plate_id: int,
    peg_id: int,
) -> np.ndarray:
    """Identify initial plate/peg overlaps without counting unrelated contacts."""
    overlapping = np.zeros(num_worlds, dtype=bool)
    if not len(geom_pairs):
        return overlapping
    bodies = geom_body_ids[geom_pairs]
    relevant = ((bodies[:, 0] == plate_id) & (bodies[:, 1] == peg_id)) | (
        (bodies[:, 1] == plate_id) & (bodies[:, 0] == peg_id)
    )
    overlapping[world_ids[relevant & (distances < -1e-9)]] = True
    return overlapping


@configclass
class AnalyticPegGeometry:
    """Independent ideal through-hole geometry, in metres and radians."""

    shape: str = "circle"
    hole_half_size: float = 0.008
    clearance: float = 0.0001
    thickness: float = 0.01
    peg_length: float = 0.03

    def __post_init__(self) -> None:
        if self.shape not in {"circle", "square"}:
            raise ValueError("Analytic probes support circle and square only.")
        if min(self.hole_half_size, self.thickness, self.peg_length) <= 0:
            raise ValueError("Probe geometric dimensions must be positive.")
        if not 0 < self.clearance < self.hole_half_size / 4:
            raise ValueError("Constrained probes require a positive narrow clearance.")
        if self.peg_length < 2 * self.thickness:
            raise ValueError(
                "Probe peg must span both plate faces throughout small tilts."
            )

    @property
    def peg_half_size(self) -> float:
        """Return the exact peg radius or square half-width."""
        return self.hole_half_size - self.clearance

    def lateral_support(self, direction: np.ndarray) -> float:
        """Maximum projected displacement under a force in this unit direction."""
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (2,) or not np.isclose(np.linalg.norm(direction), 1):
            raise ValueError("Lateral direction must be a unit XY vector.")
        return self.clearance * (
            1.0 if self.shape == "circle" else float(np.abs(direction).sum())
        )

    def translation_for_gap(self, direction: np.ndarray, gap: float) -> np.ndarray:
        """Locate a static pose with the requested minimum normal wall gap.

        At a square corner this is a per-wall gap, not Euclidean distance to
        the feasible translation set. Negative values describe real overlap.
        """
        direction = np.asarray(direction, dtype=float)
        self.lateral_support(direction)
        denominator = 1.0 if self.shape == "circle" else float(np.abs(direction).max())
        if gap > self.clearance:
            raise ValueError("A centered peg cannot have a gap larger than clearance.")
        return direction * (self.clearance - gap) / denominator

    def tilt_limit(self, phi: float = 0.0) -> float:
        """Largest tilt with one common translation fitting both plate faces."""
        radius, half_t, hole = (
            self.peg_half_size,
            self.thickness / 2,
            self.hole_half_size,
        )
        if self.shape == "circle":
            return math.acos(radius / math.hypot(hole, half_t)) - math.atan2(
                half_t, hole
            )
        axis = np.array([math.cos(phi), math.sin(phi), 0.0])
        cross = np.array(
            [[0.0, 0.0, axis[1]], [0.0, 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
        )
        corners = np.array(
            [[x * radius, y * radius, 0.0] for x in (-1, 1) for y in (-1, 1)]
        )

        def fits(angle: float) -> bool:
            rotation = (
                np.eye(3)
                + math.sin(angle) * cross
                + (1 - math.cos(angle)) * (cross @ cross)
            )
            prism_axis = rotation[:, 2]
            base = corners @ rotation.T
            sections = np.concatenate(
                [
                    base + ((z - base[:, 2]) / prism_axis[2])[:, None] * prism_axis
                    for z in (-half_t, half_t)
                ]
            )
            return bool(np.all(np.ptp(sections[:, :2], axis=0) <= 2 * hole))

        lower, upper = 0.0, math.pi / 4
        for _ in range(70):
            midpoint = (lower + upper) / 2
            if fits(midpoint):
                lower = midpoint
            else:
                upper = midpoint
        return lower

    def yaw_limit(self) -> float:
        """First square contact angle; a circle has no yaw constraint."""
        if self.shape != "square":
            raise ValueError("A circular peg has no finite yaw limit.")
        return (
            math.asin(self.hole_half_size / (math.sqrt(2) * self.peg_half_size))
            - math.pi / 4
        )


@configclass
class ConstrainedProbeCfg:
    """Reproducible S0/S2/S3/S4 workload and contact settings."""

    probe: str = "S2"
    num_directions: int = 24
    duration: float = 1.0
    timestep: float = 0.001
    settle_time: float = 0.02
    force: float = 0.1
    tilt_torque: float = 0.0015
    yaw_torque: float = 0.0008
    tilt_phi: float = 0.0
    solver_iterations: int = 100
    contact_offset: float = 0.0
    rest_offset: float = 0.0
    source_contact: bool = True
    contact_capacity: int = 4096
    gaps_um: tuple[float, ...] = (
        1000.0,
        200.0,
        100.0,
        50.0,
        20.0,
        10.0,
        5.0,
        2.0,
        1.0,
        0.0,
        -1.0,
        -2.0,
        -5.0,
        -10.0,
        -20.0,
        -50.0,
        -100.0,
        -200.0,
    )

    def __post_init__(self) -> None:
        if self.probe not in {"S0", "S2", "S3", "S4"}:
            raise ValueError("Constrained probe must be S0, S2, S3 or S4.")
        if (
            self.num_directions <= 0
            or min(self.duration, self.timestep, self.settle_time) <= 0
        ):
            raise ValueError(
                "Probe directions, duration, timestep and settling time must be positive."
            )
        if round(self.duration / self.timestep) < 2:
            raise ValueError("A settled probe needs at least two integration steps.")
        if self.solver_iterations <= 0 or self.contact_capacity <= 0:
            raise ValueError("Solver iterations and contact capacity must be positive.")
        if self.probe == "S0" and not self.gaps_um:
            raise ValueError("Static contact scans need nonempty gap samples.")


@configclass
class ConstrainedProbeDefinition:
    """Prepared reference scene and the equivalent constrained native peg."""

    asset: AssemblyAsset = MISSING
    cfg: ConstrainedProbeCfg = MISSING
    geometry: AnalyticPegGeometry = MISSING
    source_mjcf: str = MISSING
    mass: float = MISSING
    inertia: list[float] = MISSING

    @property
    def axes(self) -> list[list[float]]:
        """Return serial slide/hinge axes in generalized-coordinate order."""
        axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        if self.cfg.probe == "S3":
            axes.append([math.cos(self.cfg.tilt_phi), math.sin(self.cfg.tilt_phi), 0.0])
        elif self.cfg.probe == "S4":
            axes.append([0.0, 0.0, 1.0])
        return axes

    @property
    def damping(self) -> list[float]:
        """Passive damping matching the reference inertial settling time."""
        damping = [self.mass / self.cfg.settle_time] * 2
        if len(self.axes) == 3:
            axis = np.asarray(self.axes[-1])
            damping.append(
                float(np.dot(np.square(axis), self.inertia)) / self.cfg.settle_time
            )
        return damping

    def write_urdf(self, directory: str | Path) -> Path:
        """Export actual XY slides and optional hinge with one colliding peg.

        Intermediate links have no mass or geometry. The sole moving peg owns
        all mass and inertia; drive gains are zero and damping is passive.
        This export does not encode the independent fixed plate.
        """
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        root = ET.Element("robot", name="assembly_probe")
        ET.SubElement(root, "link", name="anchor")
        parent = "anchor"
        for index, axis in enumerate(self.axes):
            name = "peg" if index == len(self.axes) - 1 else f"slide_link_{index}"
            link = ET.SubElement(root, "link", name=name)
            if name == "peg":
                inertial = ET.SubElement(link, "inertial")
                ET.SubElement(inertial, "mass", value=str(self.mass))
                ET.SubElement(
                    inertial,
                    "inertia",
                    ixx=str(self.inertia[0]),
                    iyy=str(self.inertia[1]),
                    izz=str(self.inertia[2]),
                    ixy="0",
                    ixz="0",
                    iyz="0",
                )
                for tag in ("collision", "visual"):
                    element = ET.SubElement(link, tag)
                    geometry = ET.SubElement(element, "geometry")
                    if self.geometry.shape == "circle":
                        ET.SubElement(
                            geometry,
                            "cylinder",
                            radius=str(self.geometry.peg_half_size),
                            length=str(self.geometry.peg_length),
                        )
                    else:
                        ET.SubElement(
                            geometry,
                            "box",
                            size=f"{2*self.geometry.peg_half_size} {2*self.geometry.peg_half_size} {self.geometry.peg_length}",
                        )
            kind = "prismatic" if index < 2 else "revolute"
            joint = ET.SubElement(root, "joint", name=f"probe_joint_{index}", type=kind)
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=name)
            ET.SubElement(joint, "axis", xyz=" ".join(map(str, axis)))
            # Wide limits cannot clip the micron/radian contact measurements.
            limit = 1.0 if index < 2 else math.pi
            ET.SubElement(
                joint,
                "limit",
                lower=str(-limit),
                upper=str(limit),
                effort="1000",
                velocity="1000",
            )
            ET.SubElement(
                joint, "dynamics", damping=str(self.damping[index]), friction="0"
            )
            parent = name
        path = directory / f"probe_{self.cfg.probe}.urdf"
        ET.indent(root)
        ET.ElementTree(root).write(path, encoding="unicode")
        return path

    def native_configs(
        self,
        directory: str | Path,
        *,
        backend: str = "newton",
        source_contact: bool | None = None,
    ) -> tuple[RigidObjectCfg, ArticulationCfg]:
        """Build the fixed imported plate and the physically constrained peg."""
        from embodichain.lab.sim.cfg import (
            ArticulationCfg,
            ArticulationRootPropertiesCfg,
            JointDrivePropertiesCfg,
        )

        plate = deepcopy(
            next(body for body in self.asset.bodies if body.name == "plate")
        )
        for geom in plate.geometries:
            geom.friction = [0.0, 0.0, 0.0]
            geom.condim = 1
        plate_cfg = plate.to_rigid_config(
            source_contact=(
                self.cfg.source_contact if source_contact is None else source_contact
            ),
            backend=backend,
        )
        plate_cfg.attrs.collision_props.contact_offset = self.cfg.contact_offset
        plate_cfg.attrs.collision_props.rest_offset = self.cfg.rest_offset
        attrs = deepcopy(plate_cfg.attrs)
        attrs.mass_props.mass = None
        attrs.mass_props.inertia = None
        attrs.mass_props.com_position = None
        transform = np.eye(4)
        transform[2, 3] = self.geometry.thickness / 2
        peg_cfg = ArticulationCfg(
            uid="peg",
            fpath=str(self.write_urdf(directory)),
            init_local_pose=transform,
            attrs=attrs,
            asset_physics_mode="overlay",
            root_props=ArticulationRootPropertiesCfg(
                fixed_base=True, self_collision_enabled=False
            ),
            joint_drive_props=JointDrivePropertiesCfg(
                target_mode="effort",
                stiffness=0.0,
                armature=0.0,
                damping={
                    f"probe_joint_{index}": value
                    for index, value in enumerate(self.damping)
                },
            ),
        )
        return plate_cfg, peg_cfg


def prepare_probe(
    asset: AssemblyAsset, cfg: ConstrainedProbeCfg
) -> ConstrainedProbeDefinition:
    """Preserve imported plate blocks and construct the analytic primitive peg."""
    import mujoco

    if asset.config.get("kind") != "peg" or asset.config.get("islands"):
        raise ValueError(
            "Analytic probes need a simple circle/square peg without islands."
        )
    hole = asset.config["hole"]
    if hole["type"] == "circle":
        shape, size = "circle", float(hole["radius"])
    elif hole["type"] in {"rect", "rectangle", "square"}:
        width = float(hole.get("width", hole.get("size", 0.0)))
        height = float(hole.get("height", width))
        if not math.isclose(width, height):
            raise ValueError(
                "Analytic square probes require equal hole width and height."
            )
        shape, size = "square", width / 2
    else:
        raise ValueError("Analytic probes only support circle/square hole outlines.")
    geometry = AnalyticPegGeometry(
        shape=shape,
        hole_half_size=size,
        clearance=float(asset.config["clearance"]),
        thickness=float(asset.config["plate_thickness"]),
        peg_length=float(asset.config["peg_length"]),
    )
    if cfg.probe == "S0" and not any(
        gap * 1e-6 <= geometry.clearance for gap in cfg.gaps_um
    ):
        raise ValueError("Static gap samples all exceed the centered clearance.")
    if cfg.probe == "S4":
        geometry.yaw_limit()
    plate = next(body for body in asset.bodies if body.name == "plate")
    if not np.allclose(
        plate.position, [0.0, 0.0, geometry.thickness / 2]
    ) or not np.allclose(plate.quaternion, [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(
            "Reference probes require the source plate in its canonical horizontal frame."
        )
    mass = next(body.mass for body in asset.bodies if body.name == "peg")
    # mjfit assigns a nested default class to the plate. The replacement peg
    # is outside that class and would otherwise inherit MuJoCo's global 20 ms
    # contact time constant instead of the source plate's 2 ms profile. Resolve
    # inheritance through the public compiled model, including mixing rules.
    source_model = mujoco.MjModel.from_xml_string(asset.source_mjcf)
    plate_id = source_model.body("plate").id
    collision_ids = np.flatnonzero(
        (source_model.geom_bodyid == plate_id)
        & ((source_model.geom_contype != 0) | (source_model.geom_conaffinity != 0))
    )
    if not len(collision_ids):
        raise ValueError("The reference plate needs collision geometry.")
    reference_geom = int(collision_ids[0])
    contact_profile = {
        "solref": " ".join(map(str, source_model.geom_solref[reference_geom])),
        "solimp": " ".join(map(str, source_model.geom_solimp[reference_geom])),
        "solmix": str(float(source_model.geom_solmix[reference_geom])),
        "priority": str(int(source_model.geom_priority[reference_geom])),
    }
    radius, length = geometry.peg_half_size, geometry.peg_length
    inertia = (
        [mass * (3 * radius**2 + length**2) / 12] * 2 + [mass * radius**2 / 2]
        if shape == "circle"
        else [mass * ((2 * radius) ** 2 + length**2) / 12] * 2
        + [mass * (2 * radius) ** 2 / 6]
    )
    definition = ConstrainedProbeDefinition(
        asset=asset,
        cfg=cfg,
        geometry=geometry,
        source_mjcf="",
        mass=mass,
        inertia=inertia,
    )
    root = ET.fromstring(asset.source_mjcf)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("gravity", "0 0 0")
    option.set("timestep", str(cfg.timestep))
    option.set("iterations", str(cfg.solver_iterations))
    option.set("integrator", "implicitfast")
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Source MJCF must have worldbody.")
    for element in list(world):
        if element.tag != "body" or element.get("name") != "plate":
            world.remove(element)
    for tag in ("actuator", "sensor", "keyframe"):
        for element in list(root.findall(tag)):
            root.remove(element)
    for geom in world.iter("geom"):
        geom.set("friction", "0 0 0")
        geom.set("condim", "1")
        geom.set("margin", str(cfg.contact_offset))
        geom.set("gap", str(cfg.contact_offset - cfg.rest_offset))
    peg = ET.SubElement(world, "body", name="peg", pos=f"0 0 {geometry.thickness/2}")
    ET.SubElement(
        peg,
        "inertial",
        pos="0 0 0",
        mass=str(mass),
        diaginertia=" ".join(map(str, inertia)),
    )
    for index, axis in enumerate(definition.axes):
        ET.SubElement(
            peg,
            "joint",
            name=f"probe_joint_{index}",
            type="slide" if index < 2 else "hinge",
            axis=" ".join(map(str, axis)),
            damping=str(definition.damping[index]),
        )
    size_string = (
        f"{radius} {length/2}" if shape == "circle" else f"{radius} {radius} {length/2}"
    )
    ET.SubElement(
        peg,
        "geom",
        name="analytic_peg",
        type="cylinder" if shape == "circle" else "box",
        size=size_string,
        friction="0 0 0",
        condim="1",
        margin=str(cfg.contact_offset),
        gap=str(cfg.contact_offset - cfg.rest_offset),
        **contact_profile,
    )
    definition.source_mjcf = ET.tostring(root, encoding="unicode")
    definition.validate()
    return definition


def run_reference_probe(
    asset: AssemblyAsset, cfg: ConstrainedProbeCfg
) -> dict[str, Any]:
    """Execute CPU MuJoCo S0/S2/S3/S4 and score independent ideal geometry."""
    import mujoco

    definition = prepare_probe(asset, cfg)
    geometry = definition.geometry
    model = mujoco.MjModel.from_xml_string(definition.source_mjcf)
    plate_id = model.body("plate").id
    peg_id = model.body("peg").id
    directions = np.column_stack(
        (
            np.cos(np.arange(cfg.num_directions) * 2 * math.pi / cfg.num_directions),
            np.sin(np.arange(cfg.num_directions) * 2 * math.pi / cfg.num_directions),
        )
    )
    tolerance = max(5e-6, 0.1 * geometry.clearance)
    rows = []
    if cfg.probe == "S0":
        data = mujoco.MjData(model)
        for direction in directions:
            for gap_um in cfg.gaps_um:
                gap = gap_um * 1e-6
                if gap > geometry.clearance + 1e-12:
                    continue
                data.qpos[:2] = geometry.translation_for_gap(direction, gap)
                mujoco.mj_forward(model, data)
                distances = []
                for contact in data.contact[: data.ncon]:
                    if {
                        int(model.geom_bodyid[contact.geom1]),
                        int(model.geom_bodyid[contact.geom2]),
                    } == {plate_id, peg_id}:
                        distances.append(float(contact.dist))
                rows.append(
                    {
                        "direction_rad": math.atan2(direction[1], direction[0]),
                        "true_wall_gap_m": gap,
                        "contact": bool(distances),
                        "minimum_contact_distance_m": (
                            min(distances) if distances else None
                        ),
                        "passed": not (
                            (gap > tolerance and distances)
                            or (gap < -tolerance and not distances)
                        ),
                    }
                )
    else:
        forces = directions if cfg.probe == "S2" else directions[:1]
        steps = round(cfg.duration / cfg.timestep)
        for direction in forces:
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            initial_overlap = any(
                contact.dist < -1e-9 for contact in data.contact[: data.ncon]
            )
            if cfg.probe == "S2":
                data.qfrc_applied[:2] = cfg.force * direction
                truth = geometry.lateral_support(direction)
                scale = 1.0
            elif cfg.probe == "S3":
                data.qfrc_applied[2] = cfg.tilt_torque
                truth = geometry.tilt_limit(cfg.tilt_phi)
                scale = geometry.thickness / 2
            else:
                data.qfrc_applied[2] = cfg.yaw_torque
                truth = geometry.yaw_limit()
                scale = geometry.peg_half_size * math.sqrt(2)
            values = []
            contact_count = 0
            for step in range(steps):
                mujoco.mj_step(model, data)
                if step >= steps // 2:
                    values.append(
                        float(data.qpos[:2] @ direction)
                        if cfg.probe == "S2"
                        else float(data.qpos[2])
                    )
                    contact_count += int(data.ncon)
            errors = (np.asarray(values) - truth) * scale
            statistics = _settled_error_statistics(errors, tolerance)
            rows.append(
                {
                    "direction_rad": math.atan2(direction[1], direction[0]),
                    "truth": truth,
                    "settled_value": float(np.median(values)),
                    **statistics,
                    "initial_overlap": initial_overlap,
                    "contact_samples": contact_count,
                    "warning_count": sum(int(w.number) for w in data.warning),
                    "passed": statistics["passed"]
                    and not initial_overlap
                    and not any(w.number for w in data.warning),
                }
            )
    return {
        "probe": cfg.probe,
        "backend": "mujoco_cpu",
        "status": "passed" if all(row["passed"] for row in rows) else "failed",
        "asset_sha256": asset.content_hash,
        "config": cfg.to_dict(),
        "analytic_geometry": geometry.to_dict(),
        "mujoco_version": mujoco.__version__,
        "wall_error_tolerance_m": tolerance,
        "rows": rows,
        "constraint_semantics": (
            "static pose scan"
            if cfg.probe == "S0"
            else "real XY slides and optional hinge under constant generalized effort"
        ),
        "qualified_backends": ["mujoco_cpu"],
    }


def run_native_probe(
    asset: AssemblyAsset,
    cfg: ConstrainedProbeCfg,
    backend: str = "newton",
    *,
    work_dir: str | Path = "outputs/assembly_precision/constrained",
) -> dict[str, Any]:
    """Integrate native S2-S4 with real constraints and constant joint effort.

    S2 uses one isolated environment per force direction. S3/S4 use one
    environment. Contact-capacity health and raw profile support must be
    independently qualified before these behavior measurements certify a
    backend; missing SDK functionality is reported explicitly.
    """
    if backend not in {"newton", "default"}:
        raise ValueError("Native constrained probe backend must be newton or default.")
    definition = prepare_probe(asset, cfg)
    result: dict[str, Any] = {
        "probe": cfg.probe,
        "backend": backend,
        "asset_sha256": asset.content_hash,
        "config": cfg.to_dict(),
        "analytic_geometry": definition.geometry.to_dict(),
        "rows": [],
        "status": "unverified",
    }
    if cfg.probe == "S0":
        return {
            **result,
            "status": "skipped",
            "reason": "Native read-only collision forward/query is not exposed; no physical steps substituted for a static contact-onset scan.",
        }
    try:
        import torch
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import (
            DefaultPhysicsCfg,
            NewtonPhysicsCfg,
            RenderCfg,
        )
    except ImportError as error:
        return {**result, "status": "skipped", "reason": str(error)}
    if backend == "newton" and not torch.cuda.is_available():
        return {
            **result,
            "status": "skipped",
            "reason": "Native MuJoCo Warp probes require CUDA.",
        }
    directions = np.column_stack(
        (
            np.cos(np.arange(cfg.num_directions) * 2 * math.pi / cfg.num_directions),
            np.sin(np.arange(cfg.num_directions) * 2 * math.pi / cfg.num_directions),
        )
    )
    directions = directions if cfg.probe == "S2" else directions[:1]
    count = len(directions)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path(work_dir) / asset.content_hash / cfg.probe
    output_dir.mkdir(parents=True, exist_ok=True)
    native_model_path = output_dir / "native_model.xml"
    physics = (
        NewtonPhysicsCfg(
            device=str(device),
            gravity=[0.0, 0.0, 0.0],
            num_substeps=1,
            use_cuda_graph=False,
            solver_cfg={
                "solver_type": "mujoco_warp",
                "iterations": cfg.solver_iterations,
                "integrator": "implicitfast",
                "nconmax": cfg.contact_capacity,
                "njmax": 6 * cfg.contact_capacity,
                "save_to_mjcf": str(native_model_path.resolve()),
            },
        )
        if backend == "newton"
        else DefaultPhysicsCfg(device=str(device), gravity=[0.0, 0.0, 0.0])
    )
    sim = None
    try:
        sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                startup_summary="off",
                enable_entity_gizmo=False,
                enable_ground_plane=False,
                num_envs=count,
                physics_dt=cfg.timestep,
                sim_device=device,
                physics_cfg=physics,
                render_cfg=RenderCfg(renderer="no-render"),
            )
        )
        plate_cfg, peg_cfg = definition.native_configs(output_dir, backend=backend)
        sim.add_rigid_object(plate_cfg)
        peg = sim.add_articulation(peg_cfg)
        sim.prepare()
        names = list(peg.joint_names)
        indices = [
            names.index(f"probe_joint_{index}") for index in range(len(definition.axes))
        ]
        if len(names) != len(indices):
            raise RuntimeError("Native probe lost or added generalized coordinates.")
        efforts = torch.zeros(count, len(names), dtype=torch.float32, device=sim.device)
        if cfg.probe == "S2":
            efforts[:, indices[:2]] = torch.as_tensor(
                cfg.force * directions, device=sim.device, dtype=torch.float32
            )
        else:
            efforts[:, indices[2]] = (
                cfg.tilt_torque if cfg.probe == "S3" else cfg.yaw_torque
            )
        steps = round(cfg.duration / cfg.timestep)
        trajectory = torch.empty(
            count,
            steps - steps // 2,
            len(indices),
            dtype=torch.float32,
            device=sim.device,
        )
        for step in range(steps):
            peg.set_qf(efforts)
            sim.update(step=1)
            if step >= steps // 2:
                trajectory[:, step - steps // 2].copy_(peg.get_qpos()[:, indices])
        samples = trajectory.detach().cpu().numpy()
        geometry = definition.geometry
        tolerance = max(5e-6, 0.1 * geometry.clearance)
        rows = []
        for index, direction in enumerate(directions):
            if cfg.probe == "S2":
                values = samples[index, :, :2] @ direction
                truth, scale = geometry.lateral_support(direction), 1.0
            elif cfg.probe == "S3":
                values = samples[index, :, 2]
                truth, scale = geometry.tilt_limit(cfg.tilt_phi), geometry.thickness / 2
            else:
                values = samples[index, :, 2]
                truth, scale = geometry.yaw_limit(), geometry.peg_half_size * math.sqrt(
                    2
                )
            errors = (values - truth) * scale
            statistics = _settled_error_statistics(errors, tolerance)
            finite = bool(np.isfinite(samples[index]).all()) and statistics["finite"]
            rows.append(
                {
                    "direction_rad": math.atan2(direction[1], direction[0]),
                    "truth": truth,
                    "settled_value": float(np.median(values)),
                    **statistics,
                    "initial_overlap": None,
                    "finite": finite,
                    "passed": finite and statistics["passed"],
                }
            )
        result.update(
            rows=rows,
            behavior_status=(
                "passed" if all(row["passed"] for row in rows) else "failed"
            ),
            wall_error_tolerance_m=tolerance,
            overflow_count=None,
            status="unverified" if all(row["passed"] for row in rows) else "failed",
            num_envs=count,
            native_model_path=(
                str(native_model_path.resolve()) if backend == "newton" else None
            ),
            reason="Behavior measured; native contact overflow and profile fidelity need qualified SDK diagnostics.",
            constraint_semantics="Real imported XY slides and optional hinge, constant generalized efforts; no pose writes during integration.",
        )
    except (
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        NotImplementedError,
    ) as error:
        result.update(
            status="failed", reason=f"Native constrained probe unavailable: {error}"
        )
    finally:
        if sim is not None:
            sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()
    return result


def run_warp_probe(asset: AssemblyAsset, cfg: ConstrainedProbeCfg) -> dict[str, Any]:
    """Execute raw MuJoCo Warp S0/S2-S4 with the CPU reference's exact MJCF.

    Static poses are processed in bounded batches, with no integration. Dynamic
    trajectories and cumulative overflow flags remain on device until the end
    of each experiment. Every nonzero overflow flag invalidates the run.
    """
    definition = prepare_probe(asset, cfg)
    result: dict[str, Any] = {
        "probe": cfg.probe,
        "backend": "mujoco_warp",
        "asset_sha256": asset.content_hash,
        "config": cfg.to_dict(),
        "analytic_geometry": definition.geometry.to_dict(),
        "rows": [],
        "status": "unverified",
    }
    try:
        import mujoco
        import mujoco_warp as mjw
        import warp as wp
    except ImportError as error:
        return {**result, "status": "skipped", "reason": str(error)}
    if not wp.is_cuda_available():
        return {
            **result,
            "status": "skipped",
            "reason": "Raw MuJoCo Warp probes require CUDA.",
        }
    from ._probe_warp import record_probe_state

    geometry = definition.geometry
    model = mujoco.MjModel.from_xml_string(definition.source_mjcf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    plate_id, peg_id = model.body("plate").id, model.body("peg").id
    directions = np.column_stack(
        (
            np.cos(np.arange(cfg.num_directions) * 2 * math.pi / cfg.num_directions),
            np.sin(np.arange(cfg.num_directions) * 2 * math.pi / cfg.num_directions),
        )
    )
    tolerance = max(5e-6, 0.1 * geometry.clearance)
    rows: list[dict[str, Any]] = []
    overflow_count = 0
    try:
        with wp.ScopedDevice("cuda:0"):
            model_warp = mjw.put_model(model)
            if cfg.probe == "S0":
                cases = [
                    (direction, gap * 1e-6)
                    for direction in directions
                    for gap in cfg.gaps_um
                    if gap * 1e-6 <= geometry.clearance + 1e-12
                ]
                # Contacts are globally allocated; a bounded chunk prevents a
                # direction/gap grid from multiplying solver capacity unboundedly.
                for start in range(0, len(cases), 64):
                    chunk = cases[start : start + 64]
                    count = len(chunk)
                    dw = mjw.put_data(
                        model,
                        data,
                        nworld=count,
                        naconmax=cfg.contact_capacity * count,
                        njmax=6 * cfg.contact_capacity,
                    )
                    qpos = np.zeros((count, model.nq), dtype=np.float32)
                    for index, (direction, gap) in enumerate(chunk):
                        qpos[index, :2] = geometry.translation_for_gap(direction, gap)
                    wp.copy(dw.qpos, wp.array(qpos, dtype=wp.float32, device="cuda:0"))
                    mjw.forward(model_warp, dw)
                    wp.synchronize()
                    flags = dw.overflow.numpy()
                    overflow_count += int(np.count_nonzero(flags))
                    ncontact = int(dw.nacon.numpy()[0])
                    contact_world = dw.contact.worldid.numpy()[:ncontact]
                    contact_geoms = dw.contact.geom.numpy()[:ncontact]
                    contact_dist = dw.contact.dist.numpy()[:ncontact]
                    for index, (direction, gap) in enumerate(chunk):
                        selected = contact_world == index
                        pairs = contact_geoms[selected]
                        distances = contact_dist[selected]
                        if len(pairs):
                            bodies = model.geom_bodyid[pairs]
                            relevant = (
                                (bodies[:, 0] == plate_id) & (bodies[:, 1] == peg_id)
                            ) | ((bodies[:, 1] == plate_id) & (bodies[:, 0] == peg_id))
                            distances = distances[relevant]
                        reported = bool(len(distances))
                        rows.append(
                            {
                                "direction_rad": math.atan2(direction[1], direction[0]),
                                "true_wall_gap_m": gap,
                                "contact": reported,
                                "minimum_contact_distance_m": (
                                    float(distances.min()) if reported else None
                                ),
                                "passed": not flags[index]
                                and not (
                                    (gap > tolerance and reported)
                                    or (gap < -tolerance and not reported)
                                ),
                            }
                        )
                    del dw
            else:
                forces = directions if cfg.probe == "S2" else directions[:1]
                count = len(forces)
                dw = mjw.put_data(
                    model,
                    data,
                    nworld=count,
                    naconmax=cfg.contact_capacity * count,
                    njmax=6 * cfg.contact_capacity,
                )
                effort = np.zeros((count, model.nv), dtype=np.float32)
                if cfg.probe == "S2":
                    effort[:, :2] = cfg.force * forces
                else:
                    effort[:, 2] = (
                        cfg.tilt_torque if cfg.probe == "S3" else cfg.yaw_torque
                    )
                wp.copy(
                    dw.qfrc_applied, wp.array(effort, dtype=wp.float32, device="cuda:0")
                )
                steps = round(cfg.duration / cfg.timestep)
                history = wp.empty(
                    (count, steps, model.nq), dtype=wp.float32, device="cuda:0"
                )
                mjw.forward(model_warp, dw)
                wp.synchronize()
                initial_flags = dw.overflow.numpy()
                ncontact = int(dw.nacon.numpy()[0])
                if ncontact < 0 or ncontact > dw.contact.dist.shape[0]:
                    raise RuntimeError(
                        "Initial Warp contact count exceeds allocated capacity."
                    )
                initial_overlap = _contact_overlap_worlds(
                    model.geom_bodyid,
                    dw.contact.geom.numpy()[:ncontact],
                    dw.contact.worldid.numpy()[:ncontact],
                    dw.contact.dist.numpy()[:ncontact],
                    count,
                    plate_id,
                    peg_id,
                )
                # Forward can overflow before the first integration step; retain
                # those flags even if a later collision pass clears its buffer.
                health = wp.array(initial_flags, dtype=wp.int32, device="cuda:0")
                for step in range(steps):
                    mjw.step(model_warp, dw)
                    wp.launch(
                        record_probe_state,
                        dim=(count, model.nq),
                        inputs=[dw.qpos, dw.overflow, history, health, step],
                        device="cuda:0",
                    )
                wp.synchronize()
                samples = history.numpy()[:, steps // 2 :]
                flags = health.numpy()
                overflow_count = int(np.count_nonzero(flags))
                for index, direction in enumerate(forces):
                    if cfg.probe == "S2":
                        values = samples[index, :, :2] @ direction
                        truth, scale = geometry.lateral_support(direction), 1.0
                    elif cfg.probe == "S3":
                        values = samples[index, :, 2]
                        truth, scale = (
                            geometry.tilt_limit(cfg.tilt_phi),
                            geometry.thickness / 2,
                        )
                    else:
                        values = samples[index, :, 2]
                        truth, scale = (
                            geometry.yaw_limit(),
                            geometry.peg_half_size * math.sqrt(2),
                        )
                    errors = (values - truth) * scale
                    statistics = _settled_error_statistics(errors, tolerance)
                    finite = (
                        bool(np.isfinite(samples[index]).all()) and statistics["finite"]
                    )
                    rows.append(
                        {
                            "direction_rad": math.atan2(direction[1], direction[0]),
                            "truth": truth,
                            "settled_value": float(np.median(values)),
                            **statistics,
                            "finite": finite,
                            "initial_overlap": bool(initial_overlap[index]),
                            "overflow_flags": int(flags[index]),
                            "passed": finite
                            and not flags[index]
                            and not initial_overlap[index]
                            and statistics["passed"],
                        }
                    )
        result.update(
            rows=rows,
            overflow_count=overflow_count,
            wall_error_tolerance_m=tolerance,
            mujoco_version=mujoco.__version__,
            status=(
                "passed"
                if rows and not overflow_count and all(row["passed"] for row in rows)
                else "failed"
            ),
            qualified_backends=["mujoco_warp"],
            num_envs=(
                min(64, len(rows))
                if cfg.probe == "S0"
                else cfg.num_directions if cfg.probe == "S2" else 1
            ),
        )
    except (RuntimeError, ValueError, TypeError, AttributeError) as error:
        result.update(
            status="failed",
            reason=f"Raw MuJoCo Warp constrained probe unavailable: {error}",
        )
    return result
