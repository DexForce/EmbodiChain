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

"""Collect a fixed visual PickPlace corpus using fresh native worker batches.

Run:
    CUDA_VISIBLE_DEVICES=4 python examples/learning/bc/_support/_collect.py \
        --reference-source /path/to/pr828 --output /path/to/fresh/corpus

The pinned reference task owns physics, expert planning, actions and measured
success. A creator process starts one fresh subprocess per physical batch;
no ODS engine, learner, or inherited native world is involved.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

_REFERENCE = "51f1f0cd5a7602f23358c142997e9045323f547a"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker(request_path: Path) -> None:
    request = json.loads(request_path.read_text())
    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    sys.path.insert(0, request["reference_source"])
    import gymnasium as gym
    import h5py
    import numpy as np
    import torch

    from examples.ods.data_stream_demo.acquisition import load_acquisition_config
    from examples.ods.simple_bc_demo.observations import policy_observations
    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg, get_manager_modules
    from embodichain.lab.gym.utils.registration import discover_task_packages
    from embodichain.lab.sim import SimulationManager

    torch.set_num_threads(1)
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    config = load_acquisition_config(
        "dual",
        request["device"],
        producer_envs=request["num_envs"],
        seed=request["training_seed"],
        camera_profile="pick_place",
    )
    config["max_episode_steps"] = 600
    _write(output / "environment.json", config)
    discover_task_packages()
    cfg = config_to_cfg(config, manager_modules=get_manager_modules())
    cfg.init_rollout_buffer = True
    cfg.rollout_action_semantics = "policy_request"
    cfg.filter_dataset_saving = True
    cfg.max_episode_steps = 600
    runtime = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_uuid": request.get("gpu_uuid"),
        "gpu_name": torch.cuda.get_device_name(request["device"]),
    }
    env = None
    started = time.perf_counter()
    reports = []
    records = []
    try:
        env = gym.make(config["id"], cfg=cfg)
        raw = env.unwrapped
        for repeat in range(request.get("repeats", 1)):
            candidate = request["candidate"] + repeat * request["num_envs"]
            seed = request["seed_base"] + candidate
            torch.manual_seed(seed)
            env.reset(seed=seed, options={"save_data": False})
            before = {
                "robot_qpos": raw.robot.get_qpos().detach().cpu().tolist(),
                "cube_pose": raw.sim.get_rigid_object("cube")
                .get_local_pose()
                .detach()
                .cpu()
                .tolist(),
            }
            result = execute_demo_episode(env, episode_index=candidate)
            success, failure, metrics = raw.compute_task_state()
            physical = success.detach().cpu().tolist()
            metadata = result.to_metadata()
            after = {
                "robot_qpos": raw.robot.get_qpos().detach().cpu().tolist(),
                "cube_pose": raw.sim.get_rigid_object("cube")
                .get_local_pose()
                .detach()
                .cpu()
                .tolist(),
                "physical_success": physical,
                "physical_failure": failure.detach().cpu().tolist(),
                "metrics": {
                    name: value.detach().cpu().tolist()
                    for name, value in metrics.items()
                },
                "rollout_lengths": raw.rollout_steps.detach().cpu().tolist(),
            }
            reports.append(
                {
                    "candidate": candidate,
                    "seed": seed,
                    "before": before,
                    "demo": metadata,
                    "after": after,
                }
            )
            for row in range(request["num_envs"]):
                accepted = bool(
                    physical[row]
                    and result.success[row]
                    and result.completed_by_env[row]
                )
                record = {
                    "candidate": candidate + row,
                    "seed": seed,
                    "row": row,
                    "success": accepted,
                    "physical_success": bool(physical[row]),
                    "demo_success": bool(result.success[row]),
                    "completed": bool(result.completed_by_env[row]),
                    "terminal_reason": result.terminal_reasons[row],
                    "length": int(raw.rollout_steps[row]),
                    "world_candidate_start": request["candidate"],
                    "world_batch_index": repeat,
                    "world_group_batches": request.get("repeats", 1),
                }
                if request.get("trace"):
                    length = record["length"]
                    recorded = raw.rollout_buffer[row, :length]
                    trace_actions = recorded["actions"].detach().cpu().numpy()
                    trace_geometry = (
                        policy_observations(recorded["obs"]).detach().cpu().numpy()
                    )
                    record["trace"] = {
                        "actions_sha256": hashlib.sha256(
                            trace_actions.tobytes()
                        ).hexdigest(),
                        "geometry_sha256": hashlib.sha256(
                            trace_geometry.tobytes()
                        ).hexdigest(),
                        "actions_max_abs": float(np.abs(trace_actions).max()),
                        "finite": bool(
                            np.isfinite(trace_actions).all()
                            and np.isfinite(trace_geometry).all()
                        ),
                    }
                if accepted and request.get("save"):
                    length = record["length"]
                    data = raw.rollout_buffer[row, :length].to("cpu")
                    geometry = policy_observations(data["obs"]).numpy()
                    actions = data["actions"].numpy()
                    if (
                        length < 33
                        or not np.isfinite(geometry).all()
                        or not np.isfinite(actions).all()
                    ):
                        raise ValueError(
                            "Accepted expert payload has invalid numeric history"
                        )
                    if geometry.shape != (length, 51) or actions.shape != (length, 7):
                        raise ValueError(
                            "Accepted expert payload differs from the reference schema"
                        )
                    path = output / f"candidate_{candidate + row:06d}.h5"
                    temporary = path.with_suffix(".h5.tmp")
                    with h5py.File(temporary, "w") as handle:
                        handle.create_dataset("geometry", data=geometry)
                        handle.create_dataset("actions", data=actions)
                        for camera in ("cam_high", "cam_wrist"):
                            images = data["obs", "sensor", camera, "color"][
                                ..., :3
                            ].numpy()
                            if images.dtype != np.uint8 or images.shape != (
                                length,
                                256,
                                256,
                                3,
                            ):
                                raise ValueError(
                                    "Accepted RGB payload differs from camera calibration"
                                )
                            handle.create_dataset(
                                "images/" + camera,
                                data=images,
                                chunks=(1, 256, 256, 3),
                                compression="lzf",
                            )
                    temporary.replace(path)
                    record.update(
                        path=path.name,
                        sha256=_sha(path),
                        cube_xy=data["obs", "cube_pose"][0, :2].tolist(),
                    )
                    del data, geometry, actions
                records.append(record)
            _write(
                output / "outcomes.json",
                {
                    "records": records,
                    "batches": reports,
                    "elapsed_seconds": time.perf_counter() - started,
                    "runtime": runtime,
                },
            )
            print(
                json.dumps(
                    {
                        "candidate": candidate,
                        "completed_scenes": len(records),
                        "successes": sum(record["success"] for record in records),
                    }
                ),
                flush=True,
            )
        _write(
            output / "outcomes.json",
            {
                "records": records,
                "batches": reports,
                "elapsed_seconds": time.perf_counter() - started,
                "runtime": runtime,
            },
        )
    finally:
        if env is not None:
            env.close()
        SimulationManager.flush_cleanup_queue()
    _write(output / "cleanup.json", {"closed": True})


def _run(request: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / "request.json"
    if request_path.exists():
        previous = json.loads(request_path.read_text())
        physical_keys = {"cuda_visible_devices", "gpu_uuid"}
        if {
            key: value for key, value in previous.items() if key not in physical_keys
        } != {key: value for key, value in request.items() if key not in physical_keys}:
            raise ValueError("Existing worker request differs")
        if (output / "outcomes.json").exists() and (output / "cleanup.json").exists():
            return json.loads((output / "outcomes.json").read_text())
        raise RuntimeError(
            "Previous physical worker was incomplete; retain evidence and use a fresh corpus"
        )
    _write(request_path, request)
    if request.get("gpu_uuid") is not None:
        while True:
            snapshot = None
            for line in subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines():
                uuid, total, used, utilization = (
                    value.strip() for value in line.split(",")
                )
                if uuid == request["gpu_uuid"]:
                    snapshot = {
                        "gpu_uuid": uuid,
                        "total_mib": int(total),
                        "used_mib": int(used),
                        "utilization_percent": int(utilization),
                        "observed_unix": time.time(),
                    }
                    break
            if snapshot is None:
                raise RuntimeError("Assigned physical GPU UUID is unavailable")
            admitted = (
                snapshot["total_mib"] - snapshot["used_mib"] >= 32768
                and snapshot["utilization_percent"] <= 20
            )
            _write(
                output / "admission.json",
                {
                    **snapshot,
                    "admitted": admitted,
                    "reason": (
                        "available"
                        if admitted
                        else "busy; waiting within worker deadline"
                    ),
                },
            )
            if admitted:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Assigned GPU stayed busy within the worker deadline"
                )
            time.sleep(min(2, max(0, deadline - time.monotonic())))
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=request["reference_source"],
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    if request.get("cuda_visible_devices") is not None:
        environment["CUDA_VISIBLE_DEVICES"] = request["cuda_visible_devices"]
    with (output / "worker.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(request_path),
            ],
            cwd=request["reference_source"],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=max(0.001, deadline - time.monotonic()),
            check=True,
        )
    if not (output / "cleanup.json").exists():
        raise RuntimeError("Physical worker did not complete cleanup")
    return json.loads((output / "outcomes.json").read_text())


def main() -> None:
    """Qualify the physical expert and collect immutable accepted episodes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--qualification-seed", type=int, default=10000)
    parser.add_argument("--worker-timeout", type=float, default=1800)
    parser.add_argument("--batches-per-worker", type=int, default=1)
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        help="Physical GPU indices/UUIDs; at most one worker per GPU",
    )
    parser.add_argument("--diagnose-reuse", action="store_true")
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
        return
    if args.reference_source is None or args.output is None:
        parser.error("reference-source and output are required")
    if (
        min(args.num_envs, args.episodes, args.batches_per_worker) < 1
        or args.batches_per_worker > 16
    ):
        parser.error(
            "num-envs/episodes must be positive; batches-per-worker must be1..16"
        )
    reference = args.reference_source.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != _REFERENCE:
        parser.error("reference-source must be the frozen PR828 revision")
    args.output.mkdir(parents=True, exist_ok=True)
    base = {
        "reference_source": str(reference),
        "device": args.device,
        "training_seed": args.seed,
        "save": False,
    }
    if args.diagnose_reuse:
        report = _run(
            dict(
                base,
                output=str((args.output / "diagnostic-reuse").resolve()),
                num_envs=args.num_envs,
                candidate=0,
                seed_base=args.qualification_seed,
                repeats=3,
            ),
            timeout=args.worker_timeout,
        )
        _write(args.output / "diagnostic-reuse.json", report)
        return
    qualification_path = args.output / "expert_qualification.json"
    started = time.perf_counter()
    if qualification_path.exists():
        qualification = json.loads(qualification_path.read_text())
    else:
        report = _run(
            dict(
                base,
                output=str(
                    (args.output / "qualification" / "scalar-reference-panel").resolve()
                ),
                num_envs=1,
                candidate=0,
                seed_base=args.qualification_seed,
                repeats=40,
            ),
            timeout=args.worker_timeout,
        )
        records = report["records"]
        successes = sum(record["success"] for record in records)
        qualification = {
            "passed": successes >= 38,
            "successes": successes,
            "records": records,
            "lifecycle": "scalar-reference-panel",
            "seed_base": args.qualification_seed,
            "reference_commit": commit,
        }
        _write(qualification_path, qualification)
    if not qualification["passed"]:
        raise RuntimeError("Physical expert did not reach the unchanged 38/40 gate")
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["requested_episodes"] != args.episodes
            or manifest["training_seed"] != args.seed
        ):
            raise ValueError("Existing corpus recipe differs")
        if manifest.get("complete") is True:
            print(
                json.dumps(
                    {"phase": "already_complete", "episodes": len(manifest["episodes"])}
                ),
                flush=True,
            )
            return
    else:
        manifest = {
            "version": 1,
            "complete": False,
            "reference_commit": commit,
            "requested_episodes": args.episodes,
            "episodes": [],
            "next_candidate": 0,
            "training_seed": args.seed,
            "camera_profile": "pick_place",
            "camera_profile_version": "pick-place-v2",
            "cameras": ["cam_high", "cam_wrist"],
            "geometry_width": 51,
            "action_width": 7,
            "step_dt": 1 / 30,
            "action_semantics": "policy_request",
            "action_contracts": ["eef_pose.relative@1", "gripper.parallel@1"],
            "image_layout": "THWC",
            "image_dtype": "uint8",
            "physics": "default",
            "lifecycle": "fresh-subprocess-per-batch",
        }
        environment = json.loads(
            (
                args.output / "qualification/scalar-reference-panel/environment.json"
            ).read_text()
        )
        _write(args.output / "qualification_environment.json", environment)
        environment["num_envs"] = args.num_envs
        _write(args.output / "environment.json", environment)
        manifest["config_sha256"] = hashlib.sha256(
            json.dumps(environment, sort_keys=True).encode()
        ).hexdigest()
        _write(manifest_path, manifest)
    devices = args.gpu_ids or [os.environ.get("CUDA_VISIBLE_DEVICES", "4")]
    if len(devices) != len(set(devices)):
        parser.error("gpu-ids must be distinct")
    inventory = {}
    for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    ).splitlines():
        index, uuid = (value.strip() for value in line.split(","))
        inventory[index] = inventory[uuid] = uuid
    if any(device not in inventory for device in devices):
        parser.error("Unknown physical GPU index or UUID")
    if len({inventory[device] for device in devices}) != len(devices):
        parser.error("gpu-ids aliases must resolve to distinct physical devices")
    legacy_runtime = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "4"),
        "gpu_uuid": inventory[os.environ.get("CUDA_VISIBLE_DEVICES", "4")],
        "source": "prior_serial_launch",
    }
    legacy_batches = set()
    for record in manifest["episodes"]:
        record.setdefault(
            "world_candidate_start",
            record["candidate"] // args.num_envs * args.num_envs,
        )
        record.setdefault("world_batch_index", 0)
        record.setdefault("world_group_batches", 1)
        if "compute_device" not in record:
            record["compute_device"] = legacy_runtime
            legacy_batches.add((args.output / record["path"]).parent)
    for directory in legacy_batches:
        _write(directory / "device-audit.json", legacy_runtime)
    manifest.setdefault("preparation_runs", []).append(
        {
            "devices": [
                {"index_or_uuid": device, "uuid": inventory[device]}
                for device in devices
            ],
            "candidate_start": manifest["next_candidate"],
            "started_unix": time.time(),
            "batches_per_worker": args.batches_per_worker,
        }
    )
    manifest["lifecycle"] = "bounded-native-world-groups"
    _write(manifest_path, manifest)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        while len(manifest["episodes"]) < args.episodes:
            start = manifest["next_candidate"]
            remaining = 2 * args.episodes - start
            if remaining <= 0:
                raise RuntimeError("Accepted-episode candidate budget exhausted")
            jobs = []
            candidate = start
            needed = args.episodes - len(manifest["episodes"])
            planned = 0
            for slot, device in enumerate(devices):
                if candidate >= 2 * args.episodes or planned >= needed:
                    break
                envs = min(args.num_envs, 2 * args.episodes - candidate)
                repeats = min(
                    args.batches_per_worker,
                    (2 * args.episodes - candidate) // envs,
                    max(1, (needed - planned + envs - 1) // envs),
                )
                directory = args.output / "batches" / f"batch-{candidate:06d}"
                request = dict(
                    base,
                    save=True,
                    output=str(directory.resolve()),
                    num_envs=envs,
                    repeats=repeats,
                    candidate=candidate,
                    seed_base=args.seed,
                    cuda_visible_devices=device,
                    gpu_uuid=inventory[device],
                )
                previous_path = directory / "request.json"
                if previous_path.exists():
                    previous = json.loads(previous_path.read_text())
                    for key in (
                        "reference_source",
                        "device",
                        "training_seed",
                        "save",
                        "candidate",
                        "seed_base",
                    ):
                        if previous[key] != request[key]:
                            raise ValueError(
                                "Existing batch no longer matches the frozen corpus"
                            )
                    request = previous
                count = request["num_envs"] * request.get("repeats", 1)
                jobs.append(
                    (
                        candidate,
                        count,
                        directory,
                        pool.submit(_run, request, timeout=args.worker_timeout),
                    )
                )
                candidate += count
                planned += count
            for candidate, count, directory, future in jobs:
                report = future.result()
                if len(report["records"]) != count:
                    raise ValueError(
                        "Physical worker returned an incomplete candidate group"
                    )
                runtime = report.get("runtime", legacy_runtime)
                _write(directory / "device-audit.json", runtime)
                environment = json.loads((directory / "environment.json").read_text())
                if environment != json.loads(
                    (args.output / "environment.json").read_text()
                ):
                    raise ValueError(
                        "Physical batch configuration differs from the frozen corpus"
                    )
                for record in report["records"]:
                    if record["success"] and len(manifest["episodes"]) < args.episodes:
                        manifest["episodes"].append(
                            {
                                **record,
                                "path": str(
                                    (directory / record["path"]).relative_to(
                                        args.output
                                    )
                                ),
                                "compute_device": runtime,
                            }
                        )
                manifest["next_candidate"] = candidate + count
                _write(manifest_path, manifest)
                print(
                    json.dumps(
                        {
                            "phase": "collect",
                            "accepted": len(manifest["episodes"]),
                            "candidates": manifest["next_candidate"],
                            "seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    manifest["complete"] = True
    finished = time.time()
    requests = list((args.output / "batches").glob("*/request.json"))
    manifest["collection_seconds"] = finished - min(
        path.stat().st_mtime for path in requests
    )
    manifest["preparation_seconds"] = (
        finished
        - (args.output / "qualification/scalar-reference-panel/request.json")
        .stat()
        .st_mtime
    )
    manifest["last_invocation_seconds"] = time.perf_counter() - started
    manifest["time_origin"] = (
        "first immutable worker request timestamps; wall time includes restarts and waits"
    )
    manifest["source_id"] = (
        "sl-bc-pickplace-v1:"
        + hashlib.sha256(
            json.dumps(manifest["episodes"], sort_keys=True).encode()
        ).hexdigest()
    )
    _write(manifest_path, manifest)


if __name__ == "__main__":
    main()
