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
"""Audit assembly RL buffer capacity using CPU-only MuJoCo Warp allocations.

Run: python -m scripts.benchmark.assembly.rl_capacity
The synthetic model has nine robot DOFs and one six-DOF free part. The audit
allocates one and two CPU worlds, then extrapolates their persistent array
storage. It measures neither native GPU memory nor simulation/training speed.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import gc
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any

__all__: list[str] = []


def _array_bytes(value: Any, prefix: str = "") -> dict[str, int]:
    """Count allocated Warp array payloads without copying their contents."""
    import warp as wp

    allocations = {}
    for field in dataclasses.fields(value):
        member = getattr(value, field.name)
        name = prefix + field.name
        if isinstance(member, wp.array):
            allocations[name] = int(
                member.size * wp.types.type_size_in_bytes(member.dtype)
            )
        elif dataclasses.is_dataclass(member):
            allocations.update(_array_bytes(member, name + "."))
    return allocations


def _synthetic_model(robot_dofs: int) -> Any:
    """Build an allocation fixture, independent of robot asset availability."""
    import mujoco

    xml = (
        '<mujoco><option jacobian="dense"/><worldbody>'
        '<geom type="plane" size="0 0 .1" condim="3"/>'
    )
    xml += (
        '<body pos="0 0 .1"><joint type="hinge"/>'
        '<geom type="sphere" size=".01" mass="1" condim="3"/>'
    ) * robot_dofs
    xml += "</body>" * robot_dofs
    xml += (
        '<body pos="1 0 .5"><freejoint/>'
        '<geom type="sphere" size=".02" mass=".1" condim="3"/>'
        "</body></worldbody></mujoco>"
    )
    return mujoco.MjModel.from_xml_string(xml)


def _measure_profile(
    model: Any, name: str, capacities: tuple[int, ...]
) -> dict[str, Any]:
    """Measure CPU allocation latency and census one/two-world payloads."""
    import mujoco_warp as mjw

    contacts, constraints, assembly_sensor, grasp_sensor = capacities

    def allocate(worlds: int) -> Any:
        """Allocate the requested CPU fixture with explicit capacities."""
        return mjw.make_data(model, nworld=worlds, nconmax=contacts, njmax=constraints)

    for _ in range(2):
        data = allocate(1)
        del data
    samples_ms = []
    for _ in range(7):
        start = time.perf_counter()
        data = allocate(1)
        samples_ms.append((time.perf_counter() - start) * 1000)
        del data
    gc.collect()
    try:
        import psutil

        process = psutil.Process()
        rss_before = process.memory_info().rss
    except ImportError:
        process, rss_before = None, None
    one = allocate(1)
    cpu_delta_mb = (
        (process.memory_info().rss - rss_before) / 2**20
        if process is not None
        else None
    )
    one_bytes = _array_bytes(one)
    two = allocate(2)
    two_bytes = _array_bytes(two)
    result = {
        "profile": name,
        "nconmax_average_allocation_per_world": contacts,
        "njmax_per_world": constraints,
        "assembly_sensor_contacts_per_world": assembly_sensor,
        "grasp_sensor_contacts_per_world": grasp_sensor,
        "cost_time_ms": statistics.median(samples_ms),
        "allocation_samples_ms": samples_ms,
        "cpu_delta_mb": cpu_delta_mb,
        "gpu_delta_mb": None,
        "peak_gpu_mb": None,
        "data_one_world_bytes": sum(one_bytes.values()),
        "data_two_world_bytes": sum(two_bytes.values()),
        "data_marginal_world_bytes": sum(two_bytes.values()) - sum(one_bytes.values()),
        "contact_one_world_bytes": sum(
            size for key, size in one_bytes.items() if key.startswith("contact.")
        ),
        "constraint_one_world_bytes": sum(
            size for key, size in one_bytes.items() if key.startswith("efc.")
        ),
        "constraint_jacobian_shape": list(one.efc.J.shape),
        "overflow_shape": list(one.overflow.shape),
        "constraint_count_shape": list(one.nefc.shape),
        "batch_contact_count_shape": list(one.nacon.shape),
        "contact_pool_maximum_one_world": one.naconmax,
        "contact_pool_maximum_two_worlds": two.naconmax,
        "largest_one_world_arrays": sorted(
            one_bytes.items(), key=lambda item: -item[1]
        )[:10],
        "success_rate": None,
    }
    del one, two
    return result


def run_all_benchmarks(args: argparse.Namespace) -> Path:
    """Write one Markdown report and a machine-readable capacity estimate."""
    # This script deliberately cannot occupy the GPU used by a training run.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import warp as wp

    wp.init()
    model = _synthetic_model(args.robot_dofs)
    with wp.ScopedDevice("cpu"):
        profiles = [
            _measure_profile(model, "formal_configuration", (4096, 4096, 2048, 128)),
            _measure_profile(model, "development_candidate", (256, 1024, 256, 64)),
        ]
    projections = []
    for profile in profiles:
        for worlds in args.num_envs:
            data_bytes = (
                profile["data_one_world_bytes"]
                + (worlds - 1) * profile["data_marginal_world_bytes"]
            )
            # ContactSensor owns 11 FP32 values, two int32 IDs and one bool
            # per contact. Its separate backend query buffers are excluded.
            sensor_bytes = (
                worlds
                * (
                    profile["assembly_sensor_contacts_per_world"]
                    + profile["grasp_sensor_contacts_per_world"]
                )
                * 53
            )
            actor_rollout_bytes = worlds * 33 * 1023 * 4
            projections.append(
                {
                    "profile": profile["profile"],
                    "num_envs": worlds,
                    "estimated_mjw_data_bytes": data_bytes,
                    "sensor_tensor_bytes": sensor_bytes,
                    "actor_observation_rollout_bytes": actor_rollout_bytes,
                    "partial_storage_bytes": data_bytes
                    + sensor_bytes
                    + actor_rollout_bytes,
                    "native_gpu_capacity_status": "unmeasured",
                }
            )
    packages = {}
    for package in ("mujoco", "mujoco-warp", "warp-lang", "newton", "dexsim-engine"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "unavailable"
    notes = [
        "Synthetic allocation fixture: dense Jacobian, nine robot DOFs + one free part; no simulation steps or policy calls.",
        "CPU one-world allocation latency: two warmups, seven completed allocations, FP32 arrays. CPU retained RSS delta is separate from timing and is not peak memory.",
        "Array payload sizes come from CPU allocations at one and two worlds. Larger batches extrapolate their marginal storage, not measured GPU usage.",
        "nconmax controls the average contact allocation per world: the naconmax=nconmax*num_envs contact pool is shared across the batch, and one world may use more than nconmax contacts. njmax is a per-world constraint-row limit.",
        "Partial storage excludes native Newton model/state, collision pipeline, query buffers, solver temporary arrays, remaining rollout fields, networks, optimizer and allocators.",
        "Development capacities are candidates only. Validate real contacts, constraints, overflow, grasp/reset behavior and whole-device memory before training.",
        "Task success, native capacity, GPU memory, throughput and learning are unmeasured; no learning success rate is assigned.",
    ]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "packages": packages,
        "dtype": "float32",
        "seed": "not applicable; deterministic synthetic geometry",
        "robot_dofs": args.robot_dofs,
        "model_nv": model.nv,
        "synchronization": "synchronous CPU allocation; GPU hidden",
        "profiles": profiles,
        "projections": projections,
        "notes": notes,
    }
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "capacity.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Assembly RL CPU allocation audit",
        "",
        "## Time & Memory",
        "",
        "| profile | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb | one-world Data MiB |",
        "|---|---:|---:|---|---|---:|",
    ]
    for profile in profiles:
        delta = profile["cpu_delta_mb"]
        lines.append(
            f"| {profile['profile']} | {profile['cost_time_ms']:.3f} | "
            f"{f'{delta:.3f}' if delta is not None else 'unavailable'} | "
            f"unmeasured | unmeasured | {profile['data_one_world_bytes']/2**20:.3f} |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| profile | num_envs | success_rate | estimated Data GiB | sensor tensors GiB | actor rollout GiB | partial storage GiB |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in projections:
        values = " | ".join(
            f"{row[field]/2**30:.4f}"
            for field in (
                "estimated_mjw_data_bytes",
                "sensor_tensor_bytes",
                "actor_observation_rollout_bytes",
                "partial_storage_bytes",
            )
        )
        lines.append(
            f"| {row['profile']} | {row['num_envs']} | unmeasured | {values} |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | overall_success_rate | qualification |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| unranked | {profile['profile']} | unmeasured | CPU allocation audit only |"
        for profile in profiles
    ]
    lines += ["", "## Notes", ""] + [f"- {note}" for note in notes]
    report = directory / "README.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"Markdown report saved: {report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-dofs", type=int, default=9)
    parser.add_argument(
        "--num-envs", type=int, nargs="+", default=[1, 8, 64, 256, 1024, 12288]
    )
    parser.add_argument("--output", default="outputs/benchmarks/assembly_rl_capacity")
    args = parser.parse_args()
    if args.robot_dofs < 1 or any(worlds < 1 for worlds in args.num_envs):
        parser.error("robot-dofs and num-envs must be positive")
    run_all_benchmarks(args)
