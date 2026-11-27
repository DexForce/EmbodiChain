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
and world-frame approach direction.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import fcntl
import hashlib
from itertools import islice
import json
import math
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    link_attrs_from_dict,
)
from embodichain.lab.sim.cfg.gripper import ParallelJawGripperCfg
from embodichain.lab.sim.grasp_validation import GraspValidator
from embodichain.lab.sim.utility.gripper_geometry import (
    build_gripper_collision_geometry,
    grasp_generator_parameters,
)
from embodichain.robot_registry import (
    _validate_gripper_description,
    resolve_gripper_defaults,
    resolve_robot_cfg,
)
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGenerator,
    AntipodalGraspPoseGeneratorCfg,
    GraspAnnotationCfg,
    ParallelJawGraspCollisionCfg,
)
from embodichain.utils import configclass
from embodichain.utils.nms import pose_nms_indices

__all__ = ["cli"]

_DEFAULT_NUM_ENVS = 2400
# Revision 2 validates extracted hands with their native collision meshes.
_VALIDATION_REVISION = 3


@configclass
class _SamplingCfg:
    stable_poses: int = 5
    directions: int = 64
    candidates_per_direction: int = 100
    sample_count: int = 20_000
    max_drops: int = 100
    drop_height_m: float = 0.05
    min_rotation_deg: float = 20.0
    approach_tolerance_deg: float = 5.0
    nms_distance_m: float = 0.003
    nms_angle_deg: float = 5.0
    seed: int = 0

    def __post_init__(self) -> None:
        for key in (
            "stable_poses",
            "directions",
            "candidates_per_direction",
            "sample_count",
            "max_drops",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        for key in (
            "drop_height_m",
            "min_rotation_deg",
            "approach_tolerance_deg",
            "nms_distance_m",
            "nms_angle_deg",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
            if key.endswith("_deg") and value > 180:
                raise ValueError(f"{key} must be at most 180 degrees")


def _asset(value: str | None, root: Path) -> Path:
    if not value:
        raise ValueError("Provide an asset path in the YAML or on the CLI")
    raw = Path(value).expanduser()
    local = root / raw
    path = local if raw.is_absolute() or local.exists() else Path(get_data_path(value))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


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


def _placements(v, cfg, rng):
    poses = []
    quaternions = []
    attempts = 0
    n = v.sim.num_envs
    limit = math.cos(math.radians(cfg.min_rotation_deg) / 2)
    while attempts < cfg.max_drops and len(poses) < cfg.stable_poses:
        count = min(n, cfg.max_drops - attempts)
        rotations = Rotation.random(count, random_state=rng).as_matrix()
        for pose in v.drop_objects(rotations, cfg.drop_height_m):
            q = Rotation.from_matrix(pose[:3, :3].numpy()).as_quat()
            if any(abs(float(q @ prior)) > limit for prior in quaternions):
                continue
            poses.append(pose)
            quaternions.append(q)
            if len(poses) == cfg.stable_poses:
                break
        attempts += count
    if not poses:
        raise RuntimeError(f"No supported, stable placement after {attempts} drops")
    print(
        f"Accepted {len(poses)}/{cfg.stable_poses} stable placements after {attempts} drops",
        flush=True,
    )
    return torch.stack(poses)


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
        success=torch.zeros(count, dtype=torch.bool),
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
        _VALIDATION_REVISION,
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


def _validate_placement(v, placement, groups):
    # Directions share a frozen object transform, so their pending candidates
    # can occupy adjacent arenas in the same physics rollout.
    work = (
        (d, row)
        for d, (group, _) in enumerate(groups)
        for row in torch.where(~group["tested"])[0].tolist()
    )
    missing = 0
    while batch := list(islice(work, v.sim.num_envs)):
        poses = torch.stack([groups[d][0]["poses"][row] for d, row in batch])
        padding = poses[-1:].expand(v.sim.num_envs - len(batch), -1, -1)
        v.restore_placement(placement)
        results = v.validate(torch.cat((poses, padding)).numpy())
        for (d, row), result in zip(batch, results[: len(batch)], strict=True):
            group = groups[d][0]
            if result["status"] == "ok":
                group["tested"][row] = True
                group["success"][row] = result["success"]
            else:
                missing += 1
        for d in sorted({d for d, _ in batch}):
            group, path = groups[d]
            _save(path, group)
    return missing


def _labels(groups, placements, directions, sampling):
    poses, widths, costs, source, approach = [], [], [], [], []
    for p, d, group in groups:
        ok = group["success"] & group["tested"]
        selected = group["poses"][ok]
        poses.append(selected)
        widths.append(group["widths"][ok])
        costs.append(group["costs"][ok])
        source.append(torch.full((len(selected),), p, dtype=torch.long))
        approach.append(directions[d].expand(len(selected), 3))
    grasps = torch.cat(poses)
    order = torch.argsort(torch.cat(costs), stable=True)
    keep = order[
        pose_nms_indices(
            grasps[order],
            dist_th=sampling.nms_distance_m,
            angle_th=math.radians(sampling.nms_angle_deg),
            preserve_order=True,
        )
    ]
    return dict(
        grasp_pose=grasps[keep],
        open_length=torch.cat(widths)[keep],
        object_pose=placements,
        object_pose_index=torch.cat(source)[keep],
        approach_direction_world=torch.cat(approach)[keep],
    )


def _run(
    config,
    sampling,
    mesh_path,
    urdf,
    output,
    checkpoints,
    num_envs,
    resume,
    *,
    generator_settings=None,
    candidate_transform=None,
):
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
    if output.exists():
        if state is not None:
            print(f"Already completed: {output}")
            return
        raise FileExistsError(output)
    placement_seed, direction_seed, geometry_seed = np.random.SeedSequence(
        sampling.seed
    ).spawn(3)
    v = GraspValidator(
        obj,
        hand,
        num_envs=num_envs,
        gripper_urdf=urdf,
        physics=config.get("physics", "default"),
        close_wait_steps=config.get("close_wait_steps", 60),
        enable_disturbance=config.get("enable_disturbance", True),
        settle_on_init=False,
        **options,
    )
    try:
        if state is None:
            placements = _placements(v, sampling, np.random.default_rng(placement_seed))
            directions = _directions(
                np.random.default_rng(direction_seed), sampling.directions
            )
            state = dict(
                fingerprint=fingerprint, placements=placements, directions=directions
            )
            _save(manifest, state)
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
        missing = 0
        for p, placement in enumerate(placements):
            placement_groups = []
            for d, direction in enumerate(directions):
                path = checkpoints / f"placement-{p}-direction-{d}.pt"
                if path.exists():
                    group = torch.load(path, map_location="cpu", weights_only=True)
                else:
                    group = _candidates(gen, mesh, placement, direction)
                    if candidate_transform is not None:
                        group["poses"] = candidate_transform(group["poses"])
                    _save(path, group)
                placement_groups.append((group, path))
                groups.append((p, d, group))
            missing += _validate_placement(v, placement, placement_groups)
            passed = sum(int(group["success"].sum()) for group, _ in placement_groups)
            count = sum(len(group["poses"]) for group, _ in placement_groups)
            print(
                f"Placement {p+1}/{len(placements)}: {passed}/{count} grasps passed",
                flush=True,
            )
        if missing:
            raise RuntimeError(
                f"{missing} trials lack complete observations; retry with --resume"
            )
        labels = _labels(groups, placements, directions, sampling)
        _save(output, labels)
        print(f"Saved {len(labels['grasp_pose'])} grasp labels to {output}", flush=True)
    finally:
        v.close()


def cli(argv: Sequence[str] | None = None) -> None:
    """Save physically validated object-local grasp poses and opening widths.

    Args:
        argv: Command-line arguments, or None to read the process arguments.
            Use ``--config`` for the mesh, gripper and sampling settings and
            ``--output`` for the destination PyTorch file.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain label-grasps", description=__doc__
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mesh", help="Override the target mesh path in the YAML.")
    parser.add_argument(
        "--gripper-urdf", help="Override the standalone hand URDF in the YAML."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Destination .pt file."
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        help=f"Override the YAML candidate arena count (default {_DEFAULT_NUM_ENVS}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the checkpoint next to the output.",
    )
    budgets = (
        "stable_poses",
        "directions",
        "candidates_per_direction",
        "sample_count",
        "seed",
    )
    for name in budgets:
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    args = parser.parse_args(argv)
    if args.output.suffix != ".pt":
        parser.error("Use a .pt output path")
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("gripper"), dict):
        parser.error("The YAML must contain a gripper settings mapping")
    configured_envs = config.pop("num_envs", _DEFAULT_NUM_ENVS)
    num_envs = configured_envs if args.num_envs is None else args.num_envs
    if type(num_envs) is not int or num_envs < 1:
        parser.error("num_envs must be a positive integer")
    unknown = config.keys() - {
        "physics",
        "mesh",
        "gripper_urdf",
        "gripper",
        "rigid",
        "sampling",
        "close_wait_steps",
        "enable_disturbance",
        "collision_threshold",
    }
    if unknown:
        parser.error(f"Unknown configuration fields: {sorted(unknown)}")
    params = dict(config.get("sampling", {}))
    for key in budgets:
        if getattr(args, key) is not None:
            params[key] = getattr(args, key)
    sampling = _SamplingCfg(**params)
    base = args.config.resolve().parent
    mesh = _asset(args.mesh or config.get("mesh"), Path.cwd() if args.mesh else base)
    urdf = args.gripper_urdf or config.get("gripper_urdf")
    if urdf:
        hand = _asset(urdf, Path.cwd() if args.gripper_urdf else base)
    elif model := config["gripper"].get("model"):
        preset = resolve_robot_cfg(model, variant=config["gripper"].get("variant"))
        hand = Path(preset.fpath or preset.urdf_cfg.assemble_urdf()).resolve(
            strict=True
        )
    else:
        parser.error(
            "A custom gripper description requires --gripper-urdf or gripper_urdf in YAML"
        )
    checkpoints = args.output.with_suffix(".checkpoints")
    checkpoints.mkdir(parents=True, exist_ok=True)
    with (checkpoints / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run(
            config,
            sampling,
            mesh,
            hand,
            args.output,
            checkpoints,
            num_envs,
            args.resume,
        )
