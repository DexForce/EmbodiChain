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

"""Reconstructed object poses and calibrated, geometry-qualified grasp anchors."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path
from typing import Any, TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np
import torch

from embodichain.utils import configclass

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_skills.affordance_sampling import (
        AffordancePoseCandidates,
    )
    from embodichain.lab.sim.atomic_skills.fitting import SkillFitCfg
    from embodichain.lab.sim.atomic_skills.primitives import GraspGoal
    from embodichain.toolkits.graspkit.pg_grasp.fixed_geometry import (
        FixedGripperCollisionGeometry,
    )
    from ._robot import RobotPort, RobotProfile

__all__ = []


@configclass
class _PickUpDefaults:
    """Scene-owned pickup compensation and fixed-cadence execution settings."""

    grasp_world_z_offset_m: float = 0.0
    replay_interpolation_factor: int = 1

    def __post_init__(self) -> None:
        if type(self.grasp_world_z_offset_m) not in (float, int) or not np.isfinite(
            self.grasp_world_z_offset_m
        ):
            raise ValueError("grasp_world_z_offset_m must be finite metres.")
        if (
            type(self.replay_interpolation_factor) is not int
            or not 1 <= self.replay_interpolation_factor <= 16
        ):
            raise ValueError(
                "replay_interpolation_factor must be an integer in [1, 16]."
            )


def _load_pickup_defaults(path: Path) -> tuple[_PickUpDefaults, dict[str, str]]:
    """Read optional defaults and bind them to the exact scene declaration bytes."""
    import yaml

    source = (path / "scene_assets.yaml" if path.is_dir() else path).resolve(
        strict=True
    )
    content = source.read_bytes()
    raw = yaml.safe_load(content)
    if not isinstance(raw, dict):
        raise ValueError("Scene declaration must be a mapping.")
    settings = raw.get("skill_fitting", {})
    if not isinstance(settings, dict) or settings.keys() - {"pick_up"}:
        raise ValueError(
            "skill_fitting must contain only the optional pick_up mapping."
        )
    pickup = settings.get("pick_up", {})
    if (
        not isinstance(pickup, dict)
        or pickup.keys() - _PickUpDefaults.__dataclass_fields__.keys()
    ):
        raise ValueError("Unknown scene pickup fitting defaults.")
    return _PickUpDefaults(**pickup), {str(source): hashlib.sha256(content).hexdigest()}


@dataclass(frozen=True)
class _Object:
    identifier: str
    vertices: np.ndarray
    triangles: np.ndarray
    pose: np.ndarray
    mesh_path: Path
    max_hulls: int


def _pose(value: Any, name: str) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
        or not np.allclose(
            matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5, rtol=0
        )
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError(
            f"{name} must be a rigid world-from-object 4x4 pose in metres."
        )
    # Blender/URDF float32 exports can be rigid only to their stored precision.
    # Validate first, then remove that rounding error before strict geometry APIs.
    matrix = matrix.copy()
    matrix[:3, :3] = Rotation.from_matrix(matrix[:3, :3]).as_matrix()
    return matrix


def load_scene(
    path: Path, *, include_ungraspable: bool = False
) -> tuple[list[_Object], dict[str, str]]:
    """Read explicit mesh paths and world poses without constructing a simulator.

    A physical-export directory uses scene_assets.yaml and export_report.json's
    per-object anchor matrices. A standalone YAML can instead declare ``pose``
    on every object. Mesh vertices remain object-local; poses are applied once.
    """
    import trimesh
    import yaml
    from embodichain.data_pipeline.skill_fitting import file_digest

    source = (path / "scene_assets.yaml" if path.is_dir() else path).resolve(
        strict=True
    )
    files = {str(source): file_digest(source)}
    raw = yaml.safe_load(source.read_text())
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("objects"), dict)
        or not raw["objects"]
    ):
        raise ValueError("Scene must declare a non-empty objects mapping.")
    report_path = source.with_name("export_report.json")
    report = {}
    if report_path.is_file():
        files[str(report_path)] = file_digest(report_path)
        report = json.loads(report_path.read_text())
    objects = []
    for identifier, declaration in raw["objects"].items():
        if not isinstance(declaration, dict):
            raise ValueError(f"Scene object {identifier!r} must be a mapping.")
        if not include_ungraspable and declaration.get("graspable", True) is False:
            continue
        shape = declaration.get("shape", {})
        if shape.get("shape_type") != "Mesh":
            raise ValueError(
                f"Scene object {identifier!r} requires an explicit triangle mesh."
            )
        pose = _pose(
            declaration.get("pose", report.get(identifier, {}).get("anchor")),
            identifier,
        )
        mesh_path = (source.parent / shape["fpath"]).resolve(strict=True)
        files[str(mesh_path)] = file_digest(mesh_path)
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        scale = np.asarray(shape.get("scale", [1, 1, 1]), dtype=np.float64)
        if (
            scale.shape not in ((), (3,))
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError("Mesh scale must be a positive scalar or three-vector.")
        vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
        triangles = np.asarray(mesh.faces, dtype=np.int64)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or len(vertices) < 3
            or not np.isfinite(vertices).all()
            or triangles.ndim != 2
            or triangles.shape[1] != 3
            or not len(triangles)
        ):
            raise ValueError(f"Empty or invalid triangle mesh for {identifier!r}.")
        hulls = shape.get("collision", {}).get("max_hulls", 16)
        if type(hulls) is not int or not 1 <= hulls <= 128:
            raise ValueError("Scene collision max_hulls must be in 1..128.")
        objects.append(_Object(identifier, vertices, triangles, pose, mesh_path, hulls))
    if not objects:
        raise ValueError("Scene contains no graspable objects.")
    return objects, files


def _hand_geometry(
    profile: RobotProfile, pair: dict[str, str], qpos: np.ndarray, contact_inset: float
):
    """Use configured TCP/open commands and URDF topology for parallel fingers."""
    import pytorch_kinematics as pk
    from embodichain.toolkits.graspkit.pg_grasp.fixed_geometry import (
        FixedGripperCollisionGeometry,
    )

    arm, hand = pair["motion"], pair["grasp"]
    solver = profile.solvers[arm]
    path = Path(solver["urdf_path"])
    source = path.read_text()
    document = ET.fromstring(source)
    joints = {j.attrib["name"]: j for j in document.findall("joint")}
    parents = {
        j.find("child").attrib["link"]: j.find("parent").attrib["link"]
        for j in joints.values()
    }

    def ancestors(link: str) -> list[str]:
        result = [link]
        while link in parents:
            link = parents[link]
            result.append(link)
        return result

    names = profile.control_parts[hand]
    if any(n not in joints or joints[n].attrib["type"] != "prismatic" for n in names):
        raise ValueError(
            "Automatic grasp geometry needs configured parallel prismatic fingers and their collision meshes."
        )
    branches = [ancestors(joints[n].find("parent").attrib["link"]) for n in names]
    root = next(link for link in branches[0] if all(link in b for b in branches))
    chain = pk.build_chain_from_urdf(source).to(dtype=torch.float64, device="cpu")
    parameters = chain.get_joint_parameter_names()
    values = dict(zip(profile.joint_names, qpos, strict=True))
    if set(parameters) - set(values):
        raise ValueError(
            "Robot profile must declare every active URDF joint for grasp calibration."
        )

    def frames(command: str) -> dict[str, np.ndarray]:
        state = {
            **values,
            **dict(zip(names, profile.hand_commands[hand][command], strict=True)),
        }
        fk = chain.forward_kinematics(
            torch.tensor([[state[n] for n in parameters]], dtype=torch.float64)
        )
        return {name: value.get_matrix()[0].numpy() for name, value in fk.items()}

    opened, closed = frames("open"), frames("grasp")
    root_from_world = np.linalg.inv(opened[root])
    translations = np.array(
        [
            root_from_world[:3, :3]
            @ (
                closed[joints[n].find("child").attrib["link"]][:3, 3]
                - opened[joints[n].find("child").attrib["link"]][:3, 3]
            )
            for n in names
        ]
    )
    travel = np.linalg.norm(translations, axis=1)
    if (
        len(names) != 2
        or np.any(travel < 1e-6)
        or np.dot(translations[0] / travel[0], translations[1] / travel[1]) > -0.99
    ):
        raise ValueError(
            "Automatic grasp calibration requires two opposing parallel finger joints."
        )
    tcp_offset = _pose(solver.get("tcp", np.eye(4)), "solver TCP")
    root_from_tcp = _pose(
        root_from_world @ opened[solver["end_link_name"]] @ tcp_offset, "calibrated TCP"
    )
    x = translations[0] / travel[0]
    if np.dot(x, root_from_tcp[:3, 0]) < 0:
        x = -x
    z = root_from_tcp[:3, 2] - x * np.dot(x, root_from_tcp[:3, 2])
    if np.linalg.norm(z) < 1e-4:
        raise ValueError(
            "Configured TCP approach axis is parallel to finger closing; provide a grasp-centred TCP."
        )
    z /= np.linalg.norm(z)
    canonical = root_from_tcp.copy()
    canonical[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    links = tuple(
        link.attrib["name"]
        for link in document.findall("link")
        if link.findall("collision") and root in ancestors(link.attrib["name"])
    )
    geometry = FixedGripperCollisionGeometry.from_urdf(
        path,
        root_link=root,
        tcp_link=solver["end_link_name"],
        collision_links=links,
        joint_positions=dict(
            zip(names, profile.hand_commands[hand]["open"], strict=True)
        ),
        link_from_tcp=tcp_offset,
        grasp_from_tcp=np.linalg.inv(canonical) @ root_from_tcp,
        point_spacing=0.006,
    )
    # The runtime TCP need not be the antipodal contact centre. Ground contact
    # pairs near the overlapping physical fingertip region, then map candidates
    # back to the unchanged user TCP. This is a declared geometric convention;
    # it is not a measured force-contact or hardware calibration.
    fingers = {joints[n].find("child").attrib["link"] for n in names}
    parts = geometry.provenance["parts"]
    tip_groups = [
        [p["bounds_m"][1][2] for p in parts if p["link"] == finger]
        for finger in sorted(fingers)
    ]
    if any(not group for group in tip_groups):
        raise ValueError(
            "Each parallel finger needs collision geometry for automatic contact grounding."
        )
    tips = [max(group) for group in tip_groups]
    shift = min(tips) - contact_inset
    transform = np.eye(4)
    transform[2, 3] = -shift
    provenance = geometry.provenance.copy()
    provenance["grasp_from_tcp"] = (
        transform @ np.asarray(provenance["grasp_from_tcp"])
    ).tolist()
    for part in parts:
        part["grasp_from_mesh"] = (
            transform @ np.asarray(part["grasp_from_mesh"])
        ).tolist()
        part["bounds_m"] = (np.asarray(part["bounds_m"]) - [0, 0, shift]).tolist()
    provenance["contact_origin"] = "overlapping_finger_collision_tip_bounds"
    provenance["contact_inset_m"] = contact_inset
    provenance["runtime_tcp_to_contact_shift_m"] = shift
    points = geometry.points - geometry.points.new_tensor([0, 0, shift])
    provenance["points_sha256"] = hashlib.sha256(points.numpy().tobytes()).hexdigest()
    geometry = FixedGripperCollisionGeometry(points, provenance)
    gripper_inputs = dict(
        model_id=hand, min_opening_width=0.001, max_opening_width=float(travel.sum())
    )
    return geometry, gripper_inputs


def rank_grasps(
    candidates: AffordancePoseCandidates, reference: np.ndarray, rotation_weight: float
) -> list[int]:
    """Rank valid poses by translation plus a weighted SO(3) geodesic angle."""
    from scipy.spatial.transform import Rotation

    poses = candidates.poses[0].detach().cpu().numpy()
    valid = candidates.valid[0].detach().cpu().numpy()
    ids = np.flatnonzero(valid)
    translation = np.linalg.norm(poses[ids, :3, 3] - reference[:3, 3], axis=1)
    rotation = (
        Rotation.from_matrix(poses[ids, :3, :3] @ reference[:3, :3].T).magnitude()
        if len(ids)
        else np.array([])
    )
    return ids[
        np.argsort(translation + rotation_weight * rotation, kind="stable")
    ].tolist()


def _tcp_z_half_turn(
    geometry: FixedGripperCollisionGeometry,
) -> FixedGripperCollisionGeometry:
    """Build the exact local-TCP half turn in the sampler's contact frame.

    For a canonical pose G and contact-from-TCP calibration C, the alternative
    TCP is G C S, where S rotates pi around its own Z and leaves its origin
    fixed. Its physical surfaces in the contact frame must therefore be
    transformed by C S C^-1, including the calibration's translation. Rotating
    about canonical Z instead is not equivalent for a tilted/offset TCP.
    """
    from embodichain.toolkits.graspkit.pg_grasp.fixed_geometry import (
        FixedGripperCollisionGeometry,
    )

    provenance = deepcopy(geometry.provenance)
    calibration = np.asarray(provenance["grasp_from_tcp"])
    half_turn = np.diag([-1.0, -1.0, 1.0, 1.0])
    contact_transform = calibration @ half_turn @ np.linalg.inv(calibration)
    points = geometry.points @ geometry.points.new_tensor(
        contact_transform[:3, :3].T
    ) + geometry.points.new_tensor(contact_transform[:3, 3])
    provenance["grasp_from_tcp"] = (calibration @ half_turn).tolist()
    provenance["tcp_local_z_rotation_rad"] = float(np.pi)
    provenance["points_sha256"] = hashlib.sha256(points.numpy().tobytes()).hexdigest()
    for part in provenance.get("parts", ()):
        part["grasp_from_mesh"] = (
            contact_transform @ np.asarray(part["grasp_from_mesh"])
        ).tolist()
        corners = np.array(np.meshgrid(*np.asarray(part["bounds_m"]).T)).T.reshape(
            -1, 3
        )
        moved = corners @ contact_transform[:3, :3].T + contact_transform[:3, 3]
        part["bounds_m"] = np.stack((moved.min(0), moved.max(0))).tolist()
    return FixedGripperCollisionGeometry(points, provenance)


def prepare_scene_grasps(
    scene_path: Path,
    object_id: str,
    robot: RobotPort,
    profile: RobotProfile,
    pair: dict[str, str],
    qpos: np.ndarray,
    cfg: SkillFitCfg,
    phase_hints: list[dict[str, Any]] | None,
    *,
    grasp_world_z_offset_m: float = 0.0,
) -> tuple[list[GraspGoal], dict[str, Any], dict[str, str]]:
    """Sample one bounded affordance pool, then rank against the observed grasp."""
    from embodichain.lab.sim.atomic_skills import ObjectSemantics
    from embodichain.lab.sim.atomic_skills.affordance import AntipodalAffordance
    from embodichain.lab.sim.atomic_skills.affordance_sampling import (
        AffordancePoseCandidates,
    )
    from embodichain.lab.sim.atomic_skills.fitting import _approach, _transitions
    from embodichain.lab.sim.atomic_skills.primitives import GraspGoal
    from embodichain.toolkits.graspkit.pg_grasp.pose_generator import (
        AntipodalGraspPoseGenerator,
        AntipodalGraspPoseGeneratorCfg,
        ParallelJawGraspCollisionCfg,
    )

    arm, hand = pair["motion"], pair["grasp"]
    closes, _ = _transitions(
        qpos[:, robot.get_joint_ids(hand)],
        np.asarray(profile.hand_commands[hand]["open"]),
        np.asarray(profile.hand_commands[hand]["grasp"]),
    )
    hinted = next((p for p in phase_hints or () if p["name"] == "close"), None)
    if hinted is not None:
        closes = [(hinted["start"], hinted["stop"])]
    if len(closes) != 1:
        raise ValueError(
            "Select one complete skill with an observable closure or explicit phase hints."
        )
    with torch.no_grad():
        observed = (
            robot.compute_fk(
                torch.as_tensor(
                    qpos[:, robot.get_joint_ids(arm)],
                    dtype=torch.float32,
                    device=robot.device,
                ),
                arm,
            )
            .cpu()
            .numpy()
        )
    reference = observed[closes[0][0]]
    direction, _, _ = _approach(observed, closes[0][0], relaxed=True)
    objects, files = load_scene(scene_path)
    if object_id:
        matches = [obj for obj in objects if obj.identifier == object_id]
        if not matches:
            raise ValueError(
                f"Unknown grasp target {object_id!r}; available: {[o.identifier for o in objects]}."
            )
        target = matches[0]
    else:

        def distance(obj: _Object) -> float:
            center = (obj.vertices.min(0) + obj.vertices.max(0)) * 0.5
            return float(
                np.linalg.norm(
                    obj.pose[:3, :3] @ center + obj.pose[:3, 3] - reference[:3, 3]
                )
            )

        target = min(objects, key=distance)
    geometry, gripper_inputs = _hand_geometry(
        profile, pair, qpos[0], cfg.grasp_contact_inset_m
    )
    files.update(geometry.provenance["files_sha256"])
    affordance = AntipodalAffordance(
        mesh_vertices=torch.as_tensor(
            target.vertices, dtype=torch.float32, device=robot.device
        ),
        mesh_triangles=torch.as_tensor(
            target.triangles, dtype=torch.long, device=robot.device
        ),
    )
    rows, costs, roll_variants = [], [], []
    for half_turn in (False, True):
        # Keep the actual TCP position/approach axis fixed, even for a virtual
        # TCP offset from the antipodal contact frame. Qualify each physical
        # hand realization independently before ranking its full TCP pose.
        variant = _tcp_z_half_turn(geometry) if half_turn else geometry
        generator = AntipodalGraspPoseGenerator(
            **gripper_inputs,
            algorithm_cfg=AntipodalGraspPoseGeneratorCfg(
                sample_count=cfg.grasp_sample_count,
                max_candidates=cfg.grasp_max_candidates,
                approach_deviation_angle=cfg.grasp_approach_deviation_rad,
                # Fit the observed direction explicitly, without random cone
                # proposals changing the nearest grasp on every button press.
                approach_direction_samples=1,
            ),
            collision_cfg=ParallelJawGraspCollisionCfg(
                max_decomposition_hulls=target.max_hulls,
                filter_ground_collision=cfg.grasp_filter_ground_collision,
            ),
            collision_geometry=variant,
        )
        try:
            pool = affordance.get_grasp_candidates(
                generator,
                torch.as_tensor(target.pose, dtype=torch.float32, device=robot.device)[
                    None
                ],
                torch.as_tensor(direction, dtype=torch.float32, device=robot.device),
            )
            keep = pool.valid[0]
            rows.append(pool.poses[0, keep])
            costs.append(pool.costs[0, keep])
            roll_variants.extend(
                ["tcp_z_pi" if half_turn else "original"] * int(keep.sum())
            )
        finally:
            generator.clear_cache()
    candidates = AffordancePoseCandidates.from_rows(
        [(torch.cat(rows), torch.cat(costs))], device=robot.device
    )
    order = rank_grasps(candidates, reference, cfg.grasp_rotation_weight_m)
    if not order:
        raise ValueError(
            f"No valid affordance grasps for {target.identifier!r} with the configured hand and approach direction."
        )
    positions = candidates.poses[0, :, :3, 3].detach().cpu().numpy()
    distances = np.linalg.norm(positions - reference[:3, 3], axis=1)
    order = [i for i in order if distances[i] <= cfg.max_grasp_distance_m]
    if not order:
        raise ValueError(
            "Nearest valid grasp is too far from the recording; check target, world poses and robot base calibration."
        )
    poses = candidates.poses[0, order[: cfg.max_candidates]].detach().clone()
    nominal_poses = poses.clone()
    poses[:, 2, 3] += grasp_world_z_offset_m
    semantics = ObjectSemantics(
        affordance=affordance, geometry={}, entity_id=target.identifier
    )
    goals = [
        GraspGoal(
            semantics=semantics,
            grasp_xpos=pose,
            object_pose=pose.new_tensor(target.pose),
        )
        for pose in poses
    ]
    metadata = {
        "object_id": target.identifier,
        "grasp_world_z_offset_m": grasp_world_z_offset_m,
        "nominal_grasp_xpos": nominal_poses.cpu().tolist(),
        "offset_qualification": (
            "nominal_surface_checks_before_explicit_world_z_compensation; "
            "compensated_goals_require_kinematic_fit_and_separate_physics_validation"
            if grasp_world_z_offset_m
            else "nominal_surface_checks"
        ),
        "world_from_object": target.pose.tolist(),
        "scene_path": str(scene_path.resolve()),
        "mesh_path": str(target.mesh_path),
        "observed_grasp_xpos": reference.tolist(),
        "approach_direction": direction.tolist(),
        "approach_deviation_rad": cfg.grasp_approach_deviation_rad,
        "ground_collision_filter": cfg.grasp_filter_ground_collision,
        "candidate_count": len(order),
        "candidate_indices": order[: len(goals)],
        "candidate_hand_rolls": [roll_variants[i] for i in order[: len(goals)]],
        "grasp_symmetry": {
            "axis": "tcp_local_z",
            "angles_rad": [0.0, float(np.pi)],
            "qualification": "independent_fixed_hand_surface_checks",
        },
        "direction_sampling": "fitted_direction_with_independently_qualified_tcp_z_half_turn",
        "ranking": "translation_m + rotation_weight_m * SO3_angle_rad",
        "rotation_weight_m": cfg.grasp_rotation_weight_m,
        "hand_geometry": geometry.provenance,
        "grasp_validity_scope": (
            "antipodal_width_and_fixed_open_hand_target_and_inferred_support_surface_samples"
            if cfg.grasp_filter_ground_collision
            else "antipodal_width_and_fixed_open_hand_target_surface_samples"
        ),
    }
    return goals, metadata, files
