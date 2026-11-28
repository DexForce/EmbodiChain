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

"""Generate object-local TCP poses for a mesh and a parallel-jaw gripper URDF.

Drop the object into stable table placements, sample antipodal candidates from
upper-hemisphere approach directions and test them in parallel with GraspValidator.
Pose NMS retains distinct successful grasps. The output PyTorch file pairs each
4×4 grasp_pose with its open_length in metres and records the source placement
and world-frame approach direction. Each original trial keeps its measured
hold and disturbance evidence and its assignment to an NMS representative.
"""

from __future__ import annotations

from collections.abc import Callable
import fcntl
import hashlib
from itertools import islice
import json
import math
import time
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    link_attrs_from_dict,
)
from embodichain.lab.sim.cfg.gripper import ParallelJawGripperCfg
from embodichain.lab.sim.affordance.grasp.grasp_validation import GraspValidator
from embodichain.lab.sim.utility.gripper_geometry import (
    build_gripper_collision_geometry,
    grasp_generator_parameters,
)
from embodichain.robot_registry import (
    _validate_gripper_description,
    resolve_gripper_defaults,
)
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGenerator,
    AntipodalGraspPoseGeneratorCfg,
    GraspAnnotationCfg,
    ParallelJawGraspCollisionCfg,
)
from embodichain.utils.nms import pose_nms_indices
from .config import VALIDATION_REVISION, GraspSamplingCfg

__all__ = ["label_grasps"]


def _resolve_hand(settings: dict, urdf: Path):
    options = settings.copy()
    model = options.pop("model", None)
    description = options.pop("description", None)
    unknown = options.keys() - {
        "variant",
        "root_from_tcp",
        "open_qpos",
        "close_qpos",
        "stiffness",
        "max_effort",
    }
    if unknown:
        raise ValueError(f"Unknown gripper settings: {sorted(unknown)}")
    if (model is None) == (description is None):
        raise ValueError("Set exactly one of gripper.model and gripper.description")
    if description is not None:
        opened, closed = options["open_qpos"], options["close_qpos"]
        data = _validate_gripper_description(
            dict(
                **description,
                fpath=str(urdf),
                default_root_from_tcp=options["root_from_tcp"],
                control_parts={"eef": list(opened)},
                joint_commands={"open": opened, "grasp": closed},
                grasp_generators=["antipodal_parallel_jaw"],
            )
        )
        if not data.get("pad_links"):
            raise ValueError("Custom grippers require two pad_links")
        drives = data.pop("joint_drive_props", None)
        data["attrs"] = RigidBodyPhysicsCfg.from_dict(data.get("attrs", {}))
        data["link_attrs"] = link_attrs_from_dict(data.get("link_attrs", {}))
        model = ParallelJawGripperCfg(
            uid=data["model_id"],
            **data,
            solver_cfg={},
        )
        if drives is not None:
            model.joint_drive_props = JointDrivePropertiesCfg.from_dict(drives)
        model.root_props.fixed_base = True
    return model, options


def _generator(v, urdf, sampling, clearance, *, generator_settings=None):
    settings = generator_settings or {}
    hand = v.cfg
    description = {
        name: getattr(hand, name)
        for name in (
            "model_id",
            "root_link",
            "collision_links",
            "default_root_from_tcp",
            "root_from_contact_frame",
            "joint_commands",
            "control_parts",
            "qpos_limits",
            "min_opening_width",
            "max_opening_width",
        )
    }
    geometry = build_gripper_collision_geometry(
        description | {"fpath": str(urdf)},
        point_spacing=settings.get("point_spacing", 0.012),
    )
    algorithm = dict(
        sample_count=sampling.sample_count,
        max_candidates=sampling.candidates_per_direction,
        approach_direction_samples=1,
        approach_alignment_angle=math.radians(sampling.approach_tolerance_deg),
    )
    algorithm.update(settings.get("algorithm", {}))
    return AntipodalGraspPoseGenerator(
        **grasp_generator_parameters(description),
        collision_geometry=geometry,
        algorithm_cfg=AntipodalGraspPoseGeneratorCfg(**algorithm),
        collision_cfg=ParallelJawGraspCollisionCfg(
            **(settings.get("collision", {}) | {"collision_threshold": clearance})
        ),
        annotation_cfg=GraspAnnotationCfg(disk_cache=False),
    )


def _directions(rng, count):
    # u points from the grasp towards the upper hemisphere; -u is hand travel.
    z = rng.uniform(0, 1, count)
    phi = rng.uniform(0, 2 * np.pi, count)
    r = np.sqrt(1 - z**2)
    return -torch.tensor(
        np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=1), dtype=torch.float32
    )


def _drop_rotations(cfg, rng):
    if cfg.support_up_axis is None:
        return Rotation.random(cfg.max_drops, random_state=rng).as_matrix()
    up = np.asarray(cfg.support_up_axis, dtype=float)
    up = up / np.linalg.norm(up)
    x, y, z = np.eye(3)
    targets = (z, x, -z)
    alignments = Rotation.concatenate(
        [Rotation.align_vectors(target[None], up[None])[0] for target in targets]
    )
    align = alignments[np.arange(cfg.max_drops) % len(targets)]
    yaw, spin = rng.uniform(-np.pi, np.pi, size=(2, cfg.max_drops, 1))
    tilt = rng.uniform(-0.12, 0.12, size=(cfg.max_drops, 1))
    return (
        Rotation.from_rotvec(yaw * z)
        * Rotation.from_rotvec(tilt * y)
        * align
        * Rotation.from_rotvec(spin * up)
    ).as_matrix()


def _support_modes(poses, centers, vertices, up_axis):
    # The table normal is world +Z. Its object-frame coordinates are row 2 of R.
    normals = poses[:, 2, :3].double()
    vertices = vertices.cpu().double()
    diameter = (vertices.amax(0) - vertices.amin(0)).norm().item()
    modes, regions = [], []
    representatives = []
    if up_axis is not None:
        up = torch.nn.functional.normalize(
            torch.as_tensor(up_axis, dtype=normals.dtype), dim=0
        )
    for normal, center in zip(normals, centers, strict=True):
        region = next(
            (
                i
                for i, (n, c) in enumerate(representatives)
                if n @ normal >= math.cos(math.radians(20))
                and (center - c).norm() <= max(0.002, diameter * 0.12)
            ),
            len(representatives),
        )
        if region == len(representatives):
            representatives.append((normal, center))
        if up_axis is None:
            mode = f"support_{region}"
        else:
            cosine = float(normal @ up)
            mode = "side_lying"
            if abs(cosine) >= math.cos(math.pi / 4):
                mode = "upright" if cosine > 0 else "inverted"
        modes.append(mode)
        regions.append(region)
    return modes, regions


def _placements(v, cfg, rng):
    rotations = _drop_rotations(cfg, rng)
    batch_size = v.sim.num_envs
    batches = [
        v.drop_objects(
            rotations[i : i + batch_size],
            cfg.drop_height_m,
            return_support_centers=True,
        )
        for i in range(0, len(rotations), batch_size)
    ]
    settled, centers = (torch.cat(field) for field in zip(*batches, strict=True))
    if not len(settled):
        raise RuntimeError(
            f"No supported, stable placement after {len(rotations)} drops"
        )
    vertices, _ = v.obj.get_semantic_mesh([0])
    modes, regions = _support_modes(settled, centers, vertices[0], cfg.support_up_axis)
    settled_rotations = settled[:, :3, :3].numpy()
    quats = Rotation.from_matrix(settled_rotations).as_quat()
    # Measure yaw with the same object-fixed tangent for every pose in a region.
    # Choosing its least vertical basis axis keeps the horizontal projection nonzero.
    regions = np.asarray(regions)
    _, first = np.unique(regions, return_index=True)
    tangent = np.abs(settled_rotations[first, 2]).argmin(axis=1)[regions]
    headings = settled_rotations[np.arange(len(settled)), :2, tangent]
    headings /= np.linalg.norm(headings, axis=1, keepdims=True)
    nearest_yaw = np.full(len(quats), -1.0)
    nearest = np.zeros(len(quats))
    threshold = math.cos(math.radians(cfg.min_rotation_deg) / 2)
    selected, mode_counts, region_counts = [], {}, {}
    candidates = list(range(len(settled)))
    while candidates and len(selected) < cfg.stable_poses:
        # Allocate one placement per mode before adding support regions and yaw variants.
        index = min(
            candidates,
            key=lambda i: (
                mode_counts.get(modes[i], 0),
                region_counts.get(regions[i], 0),
                nearest_yaw[i],
                nearest[i],
                i,
            ),
        )
        selected.append(index)
        mode_counts[modes[index]] = mode_counts.get(modes[index], 0) + 1
        region_counts[regions[index]] = region_counts.get(regions[index], 0) + 1
        same_region = regions == regions[index]
        nearest_yaw[same_region] = np.maximum(
            nearest_yaw[same_region], headings[same_region] @ headings[index]
        )
        similarity = np.abs(quats @ quats[index])
        nearest = np.maximum(nearest, similarity)
        candidates = [i for i in candidates if i != index and nearest[i] <= threshold]
    print(
        f"Retained {len(selected)} placements from {len(settled)}/{len(rotations)} stable drops: {mode_counts}",
        flush=True,
    )
    return dict(
        placements=settled[selected],
        support_mode=[modes[i] for i in selected],
        support_region=torch.tensor([regions[i] for i in selected]),
        support_normal_object=settled[selected, 2, :3],
        support_center_object=centers[selected],
        settled_drops=len(settled),
        drop_attempts=len(rotations),
    )


def _candidates(gen, mesh, placement, direction):
    vertices, triangles = mesh
    pose = placement.to(vertices)
    world, widths, costs = gen.get_grasp_candidates(
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        obj_poses=pose[None],
        approach_direction=direction.to(vertices),
    )[0]
    keep = (
        torch.isfinite(costs)
        & torch.isfinite(widths)
        & (widths > 0)
        & torch.isfinite(world).all(dim=(1, 2))
    )
    local = torch.linalg.inv(pose) @ world[keep]
    count = len(local)
    return dict(
        poses=local.cpu(),
        widths=widths[keep].cpu(),
        costs=costs[keep].cpu(),
        tested=torch.zeros(count, dtype=torch.bool),
        arena_index=torch.full((count,), -1, dtype=torch.long),
        success=torch.zeros(count, dtype=torch.bool),
        failure_stage=torch.full((count,), -1, dtype=torch.int8),
        hold=torch.full((count, len(_QUALITY_FIELDS)), float("nan")),
        disturbance=torch.full((count, len(_QUALITY_FIELDS)), float("nan")),
    )


def _save(path, value):
    with NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
        temp = Path(stream.name)
        try:
            torch.save(value, stream)
            stream.flush()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def _fingerprint(config, mesh, urdf, num_envs):
    hand = config["gripper"]
    defaults = (
        resolve_gripper_defaults(hand["model"], variant=hand.get("variant"))
        if "model" in hand
        else None
    )
    identity = [
        VALIDATION_REVISION,
        config,
        defaults,
        num_envs,
        version("dexsim-engine"),
    ]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    # Native hand contact geometry may change independently of the URDF.
    files = {mesh, urdf}
    for node in ET.parse(urdf).iterfind(".//collision/geometry/mesh"):
        files.add((urdf.parent / node.attrib["filename"]).resolve())
    for file in sorted(files):
        with file.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


_QUALITY_FIELDS = ("slip_mm", "rotation_deg", "bilateral_fraction")
_FAILURE_STAGES = ("passed", "lift", "disturbance", "hold")


def _validate_groups(v, placements, groups):
    # Store only CPU row references between simulations. Indexing both transforms
    # with the same reference keeps candidate and placement paired through retries.
    work = (
        (g, row)
        for g, (_, _, group, _) in enumerate(groups)
        for row in torch.where(~group["tested"])[0].tolist()
    )
    missing = real = slots = 0
    while batch := list(islice(work, v.sim.num_envs)):
        padded = batch + [batch[-1]] * (v.sim.num_envs - len(batch))
        poses = torch.stack([groups[g][2]["poses"][row] for g, row in padded])
        v.restore_placement(placements[[groups[g][0] for g, _ in padded]])
        results = v.validate(poses.numpy())
        real += len(batch)
        slots += len(padded)
        for arena, ((g, row), result) in enumerate(
            zip(batch, results[: len(batch)], strict=True)
        ):
            group = groups[g][2]
            if result["status"] != "ok":
                missing += 1
                continue
            group["arena_index"][row] = arena
            group["tested"][row] = True
            group["success"][row] = result["success"]
            group["failure_stage"][row] = _FAILURE_STAGES.index(
                result["failure_stage"] or "passed"
            )
            for phase in ("hold", "disturbance"):
                window = result[phase]
                if window is not None:
                    group[phase][row] = torch.tensor(
                        [window[key] for key in _QUALITY_FIELDS]
                    )
        for g in sorted({g for g, _ in batch}):
            _, _, group, path = groups[g]
            _save(path, group)
    return dict(missing=missing, real_trials=real, arena_slots=slots)


def _labels(groups, placements, directions, sampling, physics):
    # Keep the trial table in stable (placement, direction, candidate) order,
    # independent of the arena count and checkpoint boundaries.
    trials = {
        field: torch.cat([group[field] for _, _, group, _ in groups])
        for field in (
            "poses",
            "widths",
            "costs",
            "arena_index",
            "success",
            "failure_stage",
            "hold",
            "disturbance",
        )
    }
    trials["object_pose_index"] = torch.cat(
        [torch.full((len(g["poses"]),), p, dtype=torch.long) for p, _, g, _ in groups]
    )
    trials["direction_index"] = torch.cat(
        [torch.full((len(g["poses"]),), d, dtype=torch.long) for _, d, g, _ in groups]
    )
    trials["candidate_index"] = torch.cat(
        [torch.arange(len(g["poses"])) for _, _, g, _ in groups]
    )
    successful = torch.where(trials["success"])[0]
    ordered = successful[trials["costs"][successful].argsort(stable=True)]
    keep, inverse = pose_nms_indices(
        trials["poses"][ordered],
        dist_th=sampling.nms_distance_m,
        angle_th=math.radians(sampling.nms_angle_deg),
        preserve_order=True,
        return_inverse=True,
    )
    representatives = ordered[keep]
    trials["label_index"] = torch.full((len(trials["poses"]),), -1, dtype=torch.long)
    trials["label_index"][ordered] = inverse
    return dict(
        grasp_pose=trials["poses"][representatives],
        open_length=trials["widths"][representatives],
        object_pose=placements,
        object_pose_index=trials["object_pose_index"][representatives],
        approach_direction_world=directions[trials["direction_index"][representatives]],
        representative_trial_index=representatives,
        trials=trials,
        directions_world=directions,
        quality_fields=list(_QUALITY_FIELDS),
        failure_stages=list(_FAILURE_STAGES),
        physics=physics,
    )


def _support_coverage(labels, modes):
    """Count NMS representatives by the placement of their actual trial."""
    trials = labels["trials"]
    tested = trials["object_pose_index"]
    per_placement = torch.stack(
        [
            torch.bincount(rows, minlength=len(modes))
            for rows in (tested, tested[trials["success"]], labels["object_pose_index"])
        ],
        dim=1,
    )
    coverage = {}
    for mode, (total, passed, kept) in zip(modes, per_placement.tolist(), strict=True):
        counts = coverage.setdefault(
            mode, dict(placements=0, trials=0, passed=0, labels=0)
        )
        counts["placements"] += 1
        counts["trials"] += total
        counts["passed"] += passed
        counts["labels"] += kept
    return coverage


def label_grasps(
    config: dict,
    sampling: GraspSamplingCfg,
    mesh_path: Path,
    urdf: Path,
    output: Path,
    checkpoints: Path,
    num_envs: int,
    resume: bool = False,
    *,
    generator_settings: dict | None = None,
    candidate_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    """Validate sampled TCPs and write their evidence to a PyTorch file.

    Args:
        config: Resolved object, gripper and physics settings.
        sampling: Sampling budgets and NMS thresholds.
        mesh_path: Target geometry file.
        urdf: Standalone hand model.
        output: Destination label file.
        checkpoints: Working directory for placements and completed trials.
        num_envs: Maximum number of simultaneous physical trials.
        resume: Continue compatible checkpoints.
        generator_settings: Effective mounted generator settings, when available.
        candidate_transform: Invocation-specific TCP adjustment applied before validation.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    with (checkpoints / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _validate_and_save(
            config,
            sampling,
            mesh_path,
            urdf,
            output,
            checkpoints,
            num_envs,
            resume,
            generator_settings,
            candidate_transform,
        )


def _validate_and_save(
    config,
    sampling,
    mesh_path,
    urdf,
    output,
    checkpoints,
    num_envs,
    resume,
    generator_settings,
    candidate_transform,
):
    started = time.perf_counter()
    hand, options = _resolve_hand(config["gripper"], urdf)
    rigid = dict(config.get("rigid", {}))
    if (
        rigid.get("body_type", "dynamic") != "dynamic"
        or rigid.get("source") is not None
        or rigid.get("semantic_shape") is not None
    ):
        raise ValueError(
            "Labels require a dynamic target using the given mesh for sampling and simulation"
        )
    shape = dict(rigid.pop("shape", {}), fpath=str(mesh_path), shape_type="Mesh")
    rigid.setdefault("init_pos", (0.0, 0.0, 0.25))
    obj = RigidObjectCfg.from_dict(
        dict(rigid, shape=shape, body_type="dynamic", uid="target")
    )
    fingerprint = _fingerprint(
        config | {"sampling": sampling.to_dict()}, mesh_path, urdf, num_envs
    )
    manifest = checkpoints / "placements.pt"
    state = None
    if manifest.exists():
        if not resume:
            raise FileExistsError(
                f"Use --resume or choose a fresh output: {checkpoints}"
            )
        state = torch.load(manifest, map_location="cpu", weights_only=True)
        if state["fingerprint"] != fingerprint:
            raise ValueError(
                "Inputs or sampling configuration changed; choose a fresh output"
            )
    resumed = state is not None
    if output.exists():
        if state is not None:
            print(f"Already completed: {output}")
            return
        raise FileExistsError(output)
    placement_seed, direction_seed, geometry_seed = np.random.SeedSequence(
        sampling.seed
    ).spawn(3)
    physics = config.get("physics", "default")

    def scene(size):
        return GraspValidator(
            obj,
            hand,
            num_envs=size,
            gripper_urdf=urdf,
            physics=physics,
            close_wait_steps=config.get("close_wait_steps", 60),
            enable_disturbance=config.get("enable_disturbance", True),
            settle_on_init=False,
            **options,
        )

    v = None
    try:
        if state is None:
            drop_size = min(num_envs, sampling.max_drops)
            v = scene(drop_size)
            state = _placements(v, sampling, np.random.default_rng(placement_seed))
            state.update(
                fingerprint=fingerprint,
                directions=_directions(
                    np.random.default_rng(direction_seed), sampling.directions
                ),
            )
            _save(manifest, state)
            if drop_size != num_envs:
                v.close()
                v = None
        if v is None:
            v = scene(num_envs)
        placements, directions = state["placements"], state["directions"]
        seed = int(geometry_seed.generate_state(1)[0])
        np.random.seed(seed)
        torch.manual_seed(seed)
        gen = _generator(
            v,
            urdf,
            sampling,
            config.get("collision_threshold", 0.005),
            generator_settings=generator_settings,
        )
        vertices, triangles = v.obj.get_semantic_mesh([0])
        mesh = vertices[0], triangles[0]
        groups = []
        for p, placement in enumerate(placements):
            for d, direction in enumerate(directions):
                path = checkpoints / f"placement-{p}-direction-{d}.pt"
                if path.exists():
                    group = torch.load(path, map_location="cpu", weights_only=True)
                else:
                    # Per-group streams make partial candidate-generation resumes reproducible.
                    group_seed = int(
                        np.random.SeedSequence([seed, p, d]).generate_state(1)[0]
                    )
                    np.random.seed(group_seed)
                    torch.manual_seed(group_seed)
                    group = _candidates(gen, mesh, placement, direction)
                    if candidate_transform is not None:
                        group["poses"] = candidate_transform(group["poses"])
                    _save(path, group)
                groups.append((p, d, group, path))
        stats = _validate_groups(v, placements, groups)
        if stats["missing"]:
            raise RuntimeError(
                f"{stats['missing']} trials lack complete observations; retry with --resume"
            )
        labels = _labels(groups, placements, directions, sampling, physics)
        labels.update(
            {
                key: state[key]
                for key in (
                    "support_mode",
                    "support_region",
                    "support_normal_object",
                    "support_center_object",
                    "settled_drops",
                    "drop_attempts",
                )
            }
        )
        stats.update(
            support_coverage=_support_coverage(labels, state["support_mode"]),
            elapsed_s=time.perf_counter() - started,
            resumed=resumed,
            drop_arenas=min(sampling.max_drops, num_envs),
            trial_arenas=num_envs,
        )
        labels.update(
            run_stats=stats,
            root_from_tcp=v.root_tcp.detach().cpu(),
            engine_version=version("dexsim-engine"),
            sampling=sampling.to_dict(),
        )
        _save(output, labels)
        elapsed = time.perf_counter() - started
        count = len(labels["grasp_pose"])
        print(f"Support coverage: {json.dumps(stats['support_coverage'])}")
        print(
            f"Saved {count} grasp labels to {output} in {elapsed:.3f} s "
            f"({count / elapsed:.2f} labels/s); "
            f"used {stats['real_trials']}/{stats['arena_slots']} arena slots",
            flush=True,
        )
    finally:
        if v is not None:
            v.close()
