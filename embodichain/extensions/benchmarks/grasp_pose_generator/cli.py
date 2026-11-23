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
import os
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import yaml

from .metrics import summarize
from ._reporting import _export_trials, _write_report

__all__ = ["main"]

PROTOCOL = dict(
    lift_m=0.10,
    clearance_m=0.02,
    hold_s=3.0,
    amplitude_m=0.03,
    leg_s=0.30,
    cycles=1,
    command_peak_m_s2=10 / np.sqrt(3) * 0.03 / 0.30**2,
    physics_dt_s=0.01,
    solver_substeps=4,
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
        physics="default", solver="default", contact_evidence="positive_impulse"
    )


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


def _generate(v, gripper, seed, direction):
    import torch
    from embodichain.robot_registry import resolve_gripper_defaults
    from embodichain.lab.sim.utility.gripper_geometry import (
        build_gripper_collision_geometry,
        grasp_generator_parameters,
    )
    from embodichain.toolkits.graspkit.pg_grasp import (
        AntipodalGraspPoseGenerator,
        GraspAnnotationCfg,
    )

    defaults = resolve_gripper_defaults(gripper)
    gen = AntipodalGraspPoseGenerator(
        **grasp_generator_parameters(defaults),
        collision_geometry=build_gripper_collision_geometry(defaults),
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


def _run_case(spec, method, source, root, out, args):
    from embodichain.lab.sim.grasp_validation import GraspValidator
    from embodichain.lab.sim.cfg import RigidObjectCfg
    from embodichain.robot_registry import resolve_gripper_defaults

    meta = {
        key: spec[key]
        for key in (
            "object",
            "gripper",
            "family",
            "object_holdout",
            "seed",
            "placement",
        )
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
    protocol = {
        "version": 4,
        **_physics_protocol(args.physics),
        "gripper_defaults": resolve_gripper_defaults(spec["gripper"]),
        "dexsim": version("dexsim-engine"),
        "object_files": digests,
        "newton": version("newton"),
    }
    case["protocol"] = dict(PROTOCOL, **_physics_protocol(args.physics))
    case["provenance"] = protocol
    encoded = json.dumps(
        [protocol, spec, method, source, args.command], sort_keys=True, allow_nan=False
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
    v = GraspValidator(
        cfg,
        spec["gripper"],
        args.num_envs,
        physics=args.physics,
        headless=args.headless,
    )
    try:
        if args.resume and path.exists():
            case = json.loads(path.read_text())
            poses = _poses(case["poses"])
        else:
            candidates = (
                source
                if source is not None
                else _generate(
                    v,
                    spec["gripper"],
                    spec["seed"],
                    spec.get("approach_direction", [0, 0, -1]),
                )
            )
            if args.command == "run":
                candidates = candidates[:10]
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
        if not np.allclose(case["initial"], v.initial.cpu().numpy(), atol=1e-4):
            raise ValueError("Settled initial state changed on resume")
        v.initial.copy_(v.initial.new_tensor(case["initial"]))
        completed = {
            (r["rank"], r["condition"]): r
            for r in case["trials"]
            if r["status"] == "ok"
        }
        work = [
            item
            for item in product(range(1, len(poses) + 1), ("base", "x", "y"))
            if item not in completed
        ]
        for start in range(0, len(work), args.num_envs):
            batch = work[start : start + args.num_envs]
            if not args.headless:
                rank, condition = batch[0]
                print(
                    f"Viewing {spec['object']} with {spec['gripper']} in arena 0 "
                    f"(candidate {rank}, {condition})",
                    flush=True,
                )
            padded = batch + [batch[-1]] * (args.num_envs - len(batch))
            results = v.validate(
                np.stack([poses[i - 1] for i, _ in padded]), [d for _, d in padded]
            )
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
            default=True,
            help="Run headlessly by default. Use --no-headless to open the simulator viewer.",
        )
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "validate":
        spec = yaml.safe_load(args.case.read_text())
        poses = [
            json.loads(line)
            for line in args.poses.read_text().splitlines()
            if line.strip()
        ]
        case = _run_case(
            dict(
                family=spec["object"], object_holdout=False, seed=0, placement="default"
            )
            | spec,
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
    suite = yaml.safe_load(args.suite.read_text())
    jobs = []
    started = time.perf_counter()
    families = {}
    for entries in (
        [o["id"] for o in suite["objects"]],
        suite["grippers"],
        suite.get("seeds", [0]),
        list(suite.get("algorithms", {"antipodal": None})),
    ):
        if not entries or len(set(entries)) != len(entries):
            raise ValueError("Suite axes must be nonempty and unique")
    for obj in suite["objects"]:
        family = obj["family"]
        if (
            type(obj["holdout"]) is not bool
            or families.setdefault(family, obj["holdout"]) != obj["holdout"]
        ):
            raise ValueError("Each CAD family must belong to exactly one object split")
    for obj, hand, seed, method in product(
        suite["objects"],
        suite["grippers"],
        suite.get("seeds", [0]),
        suite.get("algorithms", {"antipodal": None}),
    ):
        for placement, overrides in obj.get("placements", {"default": {}}).items():
            spec = dict(
                object=obj["id"],
                family=obj["family"],
                object_holdout=obj["holdout"],
                gripper=hand,
                seed=seed,
                placement=placement,
                rigid=obj["rigid"] | overrides,
            )
            spec["approach_direction"] = obj.get("approach_direction", [0, 0, -1])
            table = suite.get("algorithms", {}).get(method)
            source = None
            if table is not None:
                filename = table.format(
                    object=obj["id"], gripper=hand, seed=seed, placement=placement
                )
                source = [
                    json.loads(line)
                    for line in (args.suite.parent / filename).read_text().splitlines()
                    if line.strip()
                ]
            jobs.append((spec, method, source))
    # Native solver failures can terminate the process. The parent owns the
    # intended population and retries from atomically committed arena batches.
    pool = None
    cases = []
    try:
        for spec, method, source in jobs:
            current = copy(args)
            for attempt in range(2):
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
                        args.suite.parent,
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
                current.resume = True
            cases.append(case)
    finally:
        if pool is not None:
            pool.shutdown()

    _export_trials(args.output, cases)
    held = suite.get("held_grippers", suite["grippers"])
    if not held or not set(held) <= set(suite["grippers"]):
        raise ValueError("held_grippers must select evaluated grippers")
    report = summarize(cases, held)
    report["protocol"] = dict(PROTOCOL, **_physics_protocol(args.physics))
    elapsed = time.perf_counter() - started
    report["wall_s"] = elapsed
    report["case_inventory"] = [
        {
            k: c[k]
            for k in (
                "object",
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
