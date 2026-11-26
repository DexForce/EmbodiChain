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

"""Execute frozen grasp candidates or a full physical evaluation suite."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from importlib.metadata import version
import multiprocessing
import time
from copy import copy
import hashlib
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path
from typing import TypedDict

import numpy as np
import yaml

from embodichain.robot_registry import (
    _validate_gripper_command,
    _validate_gripper_drive_value,
    _validate_gripper_transform,
    resolve_gripper_defaults,
)

from .metrics import summarize
from ._objects import _suite_asset_root, suite_objects
from ._reporting import (
    _export_trials,
    _visual_report,
    _write_report,
    _write_visual_report,
)

__all__ = ["main"]

_BASE_PROTOCOL = dict(
    lift_m=0.10,
    clearance_m=0.02,
    hold_s=3.0,
    physics_dt_s=0.01,
    solver_substeps=4,
)
_DISTURBANCE_PROTOCOL = dict(
    amplitude_m=0.03,
    leg_s=0.30,
    cycles=1,
    command_peak_m_s2=10 / np.sqrt(3) * 0.03 / 0.30**2,
)


def _physics_protocol(physics):
    if physics == "newton":
        return dict(
            physics="newton",
            solver="dexuni",
            coupling="two_way",
            contact_evidence="positive_impulse_or_nonpositive_signed_distance",
        )
    return dict(
        physics="default",
        solver="default",
        mimic_coupling="hard",
        contact_evidence="positive_impulse",
    )


def _close_wait_steps(config: dict) -> int:
    """Read the extra closed-target dwell in physics steps."""
    steps = config.get("close_wait_steps", 0)
    if type(steps) is not int or steps < 0:
        raise ValueError("close_wait_steps must be a nonnegative integer")
    return steps


def _save(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, allow_nan=False)
    os.replace(f.name, path)


def _poses(values):
    poses = np.asarray(values, dtype=float)
    if len(poses) == 0:
        return np.empty((0, 4, 4))
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("Expected an ordered list of 4x4 matrices")
    rot = poses[:, :3, :3]
    if not (
        np.isfinite(poses).all()
        and np.allclose(poses[:, 3], [0, 0, 0, 1])
        and np.allclose(rot.transpose(0, 2, 1) @ rot, np.eye(3), atol=1e-4)
        and np.allclose(np.linalg.det(rot), 1, atol=1e-4)
    ):
        raise ValueError("Expected finite rigid object_from_tcp matrices")
    return poses


class _SuiteLoader(yaml.SafeLoader):
    """Reject repeated names before YAML construction discards a calibration."""

    def compose_mapping_node(self, anchor: str | None) -> yaml.nodes.MappingNode:
        node = super().compose_mapping_node(anchor)
        names = set()
        for key, _ in node.value:
            if key.tag == "tag:yaml.org,2002:str":
                if key.value in names:
                    raise ValueError(
                        f"Duplicate suite key {key.value!r} at line {key.start_mark.line + 1}"
                    )
                names.add(key.value)
        return node


def _load_suite(path: Path) -> tuple[dict, Path, dict | None]:
    """Read a benchmark or the underlying suite of a visual-check preset."""
    suite = yaml.load(path.read_text(), Loader=_SuiteLoader)
    if "visual_check" not in suite:
        return suite, path.parent, None
    check = suite["visual_check"]
    if (
        set(suite) != {"visual_check"}
        or not isinstance(check, dict)
        or not {"suite", "objects"}
        <= check.keys()
        <= {"suite", "objects", "sample_count"}
    ):
        raise ValueError(
            "A visual_check preset accepts suite, objects and optional sample_count"
        )
    if not isinstance(check["suite"], str) or not check["suite"].strip():
        raise ValueError("visual_check.suite must be a path to a benchmark suite")
    check.setdefault("sample_count", 2000)
    if type(check["sample_count"]) is not int or check["sample_count"] < 1:
        raise ValueError("visual_check.sample_count must be a positive integer")
    if not isinstance(check["objects"], dict) or not check["objects"]:
        raise ValueError("visual_check.objects must map categories to model IDs")
    base = path.parent / check["suite"]
    suite = yaml.load(base.read_text(), Loader=_SuiteLoader)
    if "visual_check" in suite:
        raise ValueError("visual_check.suite must reference a benchmark suite")
    return suite, base.parent, check


def _trial_protocol(physics: str, close_wait_steps: int, visual: dict | None) -> dict:
    protocol = dict(
        _BASE_PROTOCOL, close_wait_steps=close_wait_steps, **_physics_protocol(physics)
    )
    if visual is None:
        return protocol | _DISTURBANCE_PROTOCOL
    return dict(
        protocol,
        mode="visual_check",
        num_envs=1,
        candidate_limit=1,
        conditions=["base"],
        sample_count=visual["sample_count"],
    )


class _GripperOptions(TypedDict, total=False):
    gripper_urdf: str
    variant: str
    root_from_tcp: list[list[float]]
    stiffness: float
    max_effort: float
    open_qpos: dict[str, float]
    close_qpos: dict[str, float]


def _suite_grippers(
    raw: object, *, gripper_root: Path | None = None
) -> dict[str, _GripperOptions]:
    """Resolve each catalog variant and validate its TCP, drives and joint targets."""
    if not isinstance(raw, (dict, list)) or not raw:
        raise ValueError("grippers must be a nonempty list of IDs or a mapping by ID")
    entries = raw.items() if isinstance(raw, dict) else ((name, {}) for name in raw)
    grippers = {}
    for name, fields in entries:
        if not isinstance(name, str) or not name or name in grippers:
            raise ValueError("grippers must contain unique, nonempty catalog IDs")
        if not isinstance(fields, dict):
            raise ValueError(f"grippers.{name} must be a settings mapping")
        unknown = fields.keys() - {
            "variant",
            "root_from_tcp",
            "stiffness",
            "max_effort",
            "open_qpos",
            "close_qpos",
        }
        if unknown:
            raise ValueError(f"Unsupported grippers.{name} fields: {unknown}")
        options: _GripperOptions = {}
        if "variant" in fields and not isinstance(fields["variant"], str):
            raise ValueError(f"grippers.{name}.variant must be a catalog variant ID")
        defaults = resolve_gripper_defaults(name, variant=fields.get("variant"))
        for field, value in fields.items():
            try:
                if field == "variant":
                    options["variant"] = value
                elif field == "stiffness":
                    options["stiffness"] = _validate_gripper_drive_value(field, value)
                elif field == "max_effort":
                    options["max_effort"] = _validate_gripper_drive_value(field, value)
                elif field in ("open_qpos", "close_qpos"):
                    options[field] = _validate_gripper_command(
                        value, defaults["control_parts"]["eef"]
                    )
                else:
                    options["root_from_tcp"] = [
                        list(row) for row in _validate_gripper_transform(value)
                    ]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"grippers.{name}.{field}: {exc}") from exc
        if gripper_root is not None:
            options["gripper_urdf"] = str(
                (gripper_root / name / f"{name}.urdf").resolve(strict=True)
            )
        grippers[name] = options
    return grippers


def _generate(
    v,
    defaults,
    root_from_tcp,
    seed,
    direction,
    sample_count=None,
    *,
    collision_threshold=0.0,
):
    import torch
    from embodichain.lab.sim.utility.gripper_geometry import (
        build_gripper_collision_geometry,
        grasp_generator_parameters,
    )
    from embodichain.toolkits.graspkit.pg_grasp import (
        AntipodalGraspPoseGenerator,
        AntipodalGraspPoseGeneratorCfg,
        GraspAnnotationCfg,
        ParallelJawGraspCollisionCfg,
    )

    gen = AntipodalGraspPoseGenerator(
        **grasp_generator_parameters(defaults),
        algorithm_cfg=(
            AntipodalGraspPoseGeneratorCfg(sample_count=sample_count, max_candidates=1)
            if sample_count is not None
            else None
        ),
        collision_geometry=build_gripper_collision_geometry(
            defaults | {"joint_commands": v.cfg.joint_commands},
            root_from_tcp=root_from_tcp,
        ),
        collision_cfg=ParallelJawGraspCollisionCfg(
            collision_threshold=collision_threshold
        ),
        annotation_cfg=GraspAnnotationCfg(disk_cache=False),
    )
    vertices, faces = v.obj.get_semantic_mesh([0])
    np.random.seed(seed)
    torch.manual_seed(seed)
    poses, widths, cost = gen.get_grasp_candidates(
        mesh_vertices=vertices[0],
        mesh_triangles=faces[0],
        obj_poses=v.initial[None],
        approach_direction=torch.as_tensor(
            direction, device=vertices.device, dtype=vertices.dtype
        ),
    )[0]
    order = torch.argsort(cost, stable=True)
    order = order[torch.isfinite(cost[order])]
    local = (torch.linalg.inv(v.initial) @ poses[order]).cpu().tolist()
    return [
        dict(T_object_tcp=pose, cost=c, opening_m=w)
        for pose, c, w in zip(
            local, cost[order].cpu().tolist(), widths[order].cpu().tolist()
        )
    ]


def _asset_config(value, root, digests):
    if isinstance(value, list):
        return [_asset_config(v, root, digests) for v in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in ("fpath", "file_path") and isinstance(item, str):
            local = root / item
            if local.is_file():
                with local.open("rb") as stream:
                    digests[str(local.resolve())] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
                item = str(local.resolve())
        result[key] = _asset_config(item, root, digests)
    return result


def _packaged_hand_files(urdf: Path) -> dict[str, str]:
    """Fingerprint the actual URDF and collision meshes used by the two stages."""
    meshes = ET.parse(urdf).iterfind(".//collision/geometry/mesh")
    paths = {urdf.parent / mesh.attrib["filename"] for mesh in meshes}
    paths.add(urdf)
    digests = {}
    for path in paths:
        with path.open("rb") as stream:
            digests[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    return digests


def _run_case(spec, method, source, root, out, args):
    from embodichain.lab.sim.grasp_validation import GraspValidator
    from embodichain.lab.sim.cfg import RigidObjectCfg

    meta = {
        key: spec[key] for key in ("object", "category", "gripper", "seed", "placement")
    }
    case = dict(
        meta,
        physics=args.physics,
        algorithm=method,
        case_config=spec,
        count=0,
        trials=[],
    )
    digests = {}
    rigid = _asset_config(spec["rigid"], root, digests)
    variant = spec.get("variant")
    defaults = resolve_gripper_defaults(spec["gripper"], variant=variant)
    root_from_tcp = spec.get("root_from_tcp")
    close_wait_steps = _close_wait_steps(spec)
    visual = spec.get("visual_check") if args.command == "run" else None
    gripper_urdf = None
    gripper_files = {}
    if "gripper_urdf" in spec:
        gripper_urdf = (root / Path(spec["gripper_urdf"]).expanduser()).resolve(
            strict=True
        )
        gripper_files = _packaged_hand_files(gripper_urdf)
        defaults.update(
            fpath=str(gripper_urdf),
            source_sha256=gripper_files[str(gripper_urdf)],
            mesh_paths={},
            extract_subtree=False,
        )
    protocol = {
        "version": 5,
        **_physics_protocol(args.physics),
        "gripper_defaults": defaults,
        "dexsim": version("dexsim-engine"),
        "object_files": digests,
        "newton": version("newton"),
    }
    if gripper_urdf is not None:
        protocol["gripper_files"] = gripper_files
    case["protocol"] = _trial_protocol(args.physics, close_wait_steps, visual)
    case["provenance"] = protocol
    checkpoint_spec = spec
    if args.command == "run" and "open_qpos" in spec:
        # A suite spelling out its catalog targets can reuse its earlier trials.
        checkpoint_spec = spec.copy()
        for field, command in (("open_qpos", "open"), ("close_qpos", "grasp")):
            if spec.get(field) == defaults["joint_commands"][command]:
                del checkpoint_spec[field]
    encoded = json.dumps(
        [protocol, checkpoint_spec, method, source, args.command],
        sort_keys=True,
        allow_nan=False,
    ).encode()
    name = hashlib.sha256(encoded).hexdigest()
    path = out / (name + ".json")
    if args.resume and path.exists():
        prior = json.loads(path.read_text())
        if prior.get("finished"):
            return prior
    import psutil
    import torch

    clock = time.perf_counter()
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    device = torch.device("cuda:0")
    torch.cuda.init()
    initial_gpu = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    cfg = RigidObjectCfg.from_dict(rigid)
    if not args.headless:
        label = f"{spec['category']} / {spec['object']} with {spec['gripper']}"
        print(f"Preparing {label}", flush=True)
    v = GraspValidator(
        cfg,
        spec["gripper"],
        args.num_envs,
        variant=variant,
        gripper_urdf=gripper_urdf,
        physics=args.physics,
        headless=args.headless,
        rotation_shift_deg=spec.get("rotation_shift_deg"),
        root_from_tcp=root_from_tcp,
        stiffness=spec.get("stiffness"),
        max_effort=spec.get("max_effort"),
        open_qpos=spec.get("open_qpos"),
        close_qpos=spec.get("close_qpos"),
        close_wait_steps=close_wait_steps,
    )
    try:
        if args.resume and path.exists():
            case = json.loads(path.read_text())
            poses = _poses(case["poses"])
        else:
            if not args.headless and source is None:
                print(f"Generating grasp candidates for {label}", flush=True)
            candidates = (
                source
                if source is not None
                else _generate(
                    v,
                    defaults,
                    root_from_tcp,
                    spec["seed"],
                    spec.get("approach_direction", [0, 0, -1]),
                    sample_count=visual["sample_count"] if visual is not None else None,
                    collision_threshold=spec.get("collision_threshold", 0.0),
                )
            )
            if args.command == "run":
                candidates = candidates[:1] if visual is not None else candidates[:10]
            poses = _poses([c["T_object_tcp"] for c in candidates])
            case["candidate_metadata"] = [
                {k: v for k, v in c.items() if k != "T_object_tcp"} for c in candidates
            ]
            case.update(
                count=len(poses),
                poses=poses.tolist(),
                initial=v.initial.cpu().tolist(),
                trial_id_prefix=name,
            )
            _save(path, case)
        if not args.headless and len(poses) == 0:
            print(f"No grasp candidates for {label}. Skipping playback.", flush=True)
        if not np.allclose(case["initial"], v.initial.cpu().numpy(), atol=1e-4):
            raise ValueError("Settled initial state changed on resume")
        v.initial.copy_(v.initial.new_tensor(case["initial"]))
        completed = {
            (r["rank"], r["condition"]): r
            for r in case["trials"]
            if r["status"] == "ok"
        }
        conditions = ("base",) if visual is not None else ("base", "x", "y")
        work = [
            item
            for item in product(range(1, len(poses) + 1), conditions)
            if item not in completed
        ]
        for start in range(0, len(work), args.num_envs):
            batch = work[start : start + args.num_envs]
            if not args.headless:
                rank, condition = batch[0]
                print(
                    f"Viewing {label} in arena 0 " f"(candidate {rank}, {condition})",
                    flush=True,
                )
            padded = batch + [batch[-1]] * (args.num_envs - len(batch))
            results = v.validate(
                np.stack([poses[i - 1] for i, _ in padded]), [d for _, d in padded]
            )
            if not args.headless:
                trial = results[0]
                if trial["status"] != "ok":
                    outcome = trial.get("reason", trial["status"])
                elif trial["success"]:
                    outcome = "success"
                else:
                    outcome = f"failed during {trial['failure_stage']}"
                print(f"Arena 0 candidate {rank} ({condition}): {outcome}", flush=True)
            for (rank, d), result in zip(batch, results):
                completed[rank, d] = dict(
                    result, rank=rank, condition=d, trial_id=f"{name}/{rank}/{d}"
                )
            case["trials"] = list(completed.values())
            _save(path, case)
        case["finished"] = all(r["status"] == "ok" for r in case["trials"])
        case["performance"] = {
            "cost_time_ms": 1000 * (time.perf_counter() - clock),
            "cpu_delta_mb": (process.memory_info().rss - initial_rss) / 2**20,
            "gpu_delta_mb": (torch.cuda.memory_allocated(device) - initial_gpu) / 2**20,
            "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        }
        _save(path, case)
        return case
    finally:
        v.close()


def main(argv: list[str] | None = None) -> None:
    """Dispatch single-case validation or a crossed object/hand benchmark."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--case", type=Path, required=True)
    validate.add_argument("--poses", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--suite", type=Path, required=True)
    for p in (run, validate):
        p.add_argument(
            "--physics",
            choices=("default", "newton"),
            default="default",
            help="Select Default physics or Newton with the DexUni solver.",
        )
        p.add_argument("--num-envs", type=int, default=1)
        p.add_argument(
            "--headless",
            action=argparse.BooleanOptionalAction,
            default=True if p is validate else None,
            help="Show the native window with --no-headless. Visual-check presets show it by default.",
        )
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "validate":
        spec = yaml.safe_load(args.case.read_text())
        spec.setdefault("category", spec.pop("family", spec["object"]))
        spec.pop("object_holdout", None)
        if "root_from_contact_frame" in spec:
            parser.error("root_from_contact_frame is supplied by the gripper catalog")
        if spec.keys() & {"default_physics", "mimic_compliance"}:
            parser.error(
                "Remove the mimic settings from the case YAML. PhysX uses hard coupling automatically."
            )
        poses = [
            json.loads(line)
            for line in args.poses.read_text().splitlines()
            if line.strip()
        ]
        case = _run_case(
            dict(seed=0, placement="default") | spec,
            "external",
            poses,
            args.case.parent,
            args.output,
            args,
        )
        print(args.output / (case["trial_id_prefix"] + ".json"))
        if not case["finished"]:
            raise SystemExit("Incomplete observations; retry using --resume")
        return
    suite, root, visual = _load_suite(args.suite)
    if args.headless is None:
        args.headless = visual is None
    if visual is not None and args.num_envs != 1:
        parser.error("A visual check runs in a single arena; use --num-envs 1")
    close_wait_steps = _close_wait_steps(suite)
    collision_threshold = suite.get("collision_threshold", 0.0)
    if type(collision_threshold) not in (int, float) or not math.isfinite(
        collision_threshold
    ):
        raise ValueError("collision_threshold must be a finite number in metres")
    gripper_root = suite.get("gripper_root")
    grippers = _suite_grippers(
        suite["grippers"],
        gripper_root=(
            _suite_asset_root(gripper_root, root) if gripper_root is not None else None
        ),
    )
    objects = suite_objects(
        suite,
        root,
        physics=args.physics,
        representatives=visual["objects"] if visual is not None else None,
    )
    seeds = suite.get("seeds", [0])
    algorithms = suite.get("algorithms", {"antipodal": None})
    methods = list(algorithms)
    jobs = []
    started = time.perf_counter()
    for entries in (
        [(o["category"], o["id"]) for o in objects],
        seeds,
        methods,
    ):
        if not entries or len(set(entries)) != len(entries):
            raise ValueError("Suite axes must be nonempty and unique")
    if visual is not None:
        seeds, methods = seeds[:1], methods[:1]
    for obj, hand, seed, method in product(objects, grippers, seeds, methods):
        placements = list(obj.get("placements", {"default": {}}).items())
        if visual is not None:
            if not placements:
                raise ValueError(f"At least one placement is required for {obj['id']}")
            placements = placements[:1]
        for placement, overrides in placements:
            spec = dict(
                object=obj["id"],
                category=obj["category"],
                gripper=hand,
                seed=seed,
                placement=placement,
                rigid=obj["rigid"] | overrides,
                **grippers[hand],
            )
            if close_wait_steps:
                spec["close_wait_steps"] = close_wait_steps
            if visual is not None:
                spec["visual_check"] = {"sample_count": visual["sample_count"]}
            if "rotation_shift_deg" in obj:
                spec["rotation_shift_deg"] = obj["rotation_shift_deg"]
            spec["approach_direction"] = obj.get("approach_direction", [0, 0, -1])
            table = algorithms[method]
            source = None
            if table is not None:
                filename = table.format(
                    object=obj["id"],
                    category=obj["category"],
                    gripper=hand,
                    seed=seed,
                    placement=placement,
                )
                source = [
                    json.loads(line)
                    for line in (root / filename).read_text().splitlines()
                    if line.strip()
                ]
            elif collision_threshold != 0.0:
                spec["collision_threshold"] = float(collision_threshold)
            jobs.append((spec, method, source))
    # Native solver failures can terminate the process. The parent owns the
    # intended population and retries from atomically committed arena batches.
    pool = None
    cases = []
    try:
        for index, (spec, method, source) in enumerate(jobs, 1):
            current = copy(args)
            if visual is not None:
                print(
                    f"Visual check {index}/{len(jobs)}: {spec['gripper']} / {spec['category']} / {spec['object']}",
                    flush=True,
                )
            for _ in range(1 if visual is not None else 2):
                if pool is None:
                    pool = ProcessPoolExecutor(
                        max_workers=1, mp_context=multiprocessing.get_context("spawn")
                    )
                try:
                    case = pool.submit(
                        _run_case,
                        spec,
                        method,
                        source,
                        root,
                        args.output,
                        current,
                    ).result()
                    if case["finished"]:
                        break
                except Exception as exc:
                    if isinstance(exc, BrokenProcessPool):
                        pool.shutdown(wait=True)
                        pool = None
                    case = dict(
                        spec, algorithm=method, error=str(exc), count=0, trials=[]
                    )
                    if visual is not None:
                        print(f"Visual check error: {exc}", flush=True)
                current.resume = True
            cases.append(case)
    finally:
        if pool is not None:
            pool.shutdown()

    _export_trials(args.output, cases)
    protocol = _trial_protocol(args.physics, close_wait_steps, visual)
    if visual is not None:
        report = _visual_report(cases, protocol)
        report["wall_s"] = time.perf_counter() - started
        _save(args.output / "report.json", report)
        _write_visual_report(args.output, report)
        print(args.output / "report.md")
        if not report["complete"]:
            raise SystemExit("Incomplete visual checks; retry using --resume")
        return
    report = summarize(cases)
    report["protocol"] = protocol
    elapsed = time.perf_counter() - started
    report["wall_s"] = elapsed
    report["case_inventory"] = [
        {
            k: c[k]
            for k in (
                "object",
                "category",
                "gripper",
                "placement",
                "seed",
                "algorithm",
                "count",
                "trial_id_prefix",
                "performance",
                "error",
            )
            if k in c
        }
        for c in cases
    ]
    report["attempts"] = sum(len(c["trials"]) for c in cases)
    _save(args.output / "report.json", report)
    _write_report(args.output, report)
    print(args.output / "report.json")
    if not report["complete"]:
        raise SystemExit("Incomplete observations; retry using --resume")
