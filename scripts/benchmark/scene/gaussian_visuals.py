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

"""Measure visual publication and completed camera rendering on identical Scenes.

Run: python scripts/benchmark/scene/gaussian_visuals.py --asset /assets/scene.ply
Pass --lod-path to compare flat and dynamic hierarchy at the same camera quality.
Each shape runs in a separate process. Native and EmbodiChain arms use the same
prepared Scene, transforms and CameraGroups; setup time is shared, not a claim
about independent native loading. Native rendering is synchronous. CPU state
transfers are included; native profiling accounts for Gaussian GPU payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

__all__ = ["main"]


def worker(args: argparse.Namespace) -> dict:
    """Measure the same prepared native scene through both publication paths."""
    import numpy as np
    import psutil
    import dexsim
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        RenderCfg,
        GaussianRenderCfg,
        DenoisingCfg,
        VisualObjectCfg,
        VisualElementCfg,
        GaussianSplatCfg,
    )
    from embodichain.lab.sim.sensors import CameraCfg

    rss = psutil.Process().memory_info().rss
    start = time.perf_counter()
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            num_envs=args.envs,
            device="cpu",
            enable_entity_gizmo=False,
            startup_summary="off",
            render_cfg=RenderCfg(
                renderer=args.renderer,
                gaussian=GaussianRenderCfg(
                    enabled=True,
                    enable_render_profiler=True,
                    lod_mode="dynamic_hierarchy" if args.mode == "lod" else "disabled",
                ),
                denoising=DenoisingCfg(window="off", offscreen="off"),
            ),
        )
    )
    cameras = []
    visual = batch = world = None
    try:
        visual = sim.add_visual_object(
            VisualObjectCfg(
                uid="scan",
                init_pos=(0, 0, 0.7),
                visuals=[
                    VisualElementCfg(
                        representation=GaussianSplatCfg(
                            file_path=str(args.asset),
                            lod_path=str(args.lod_path) if args.mode == "lod" else None,
                            lod_enabled=args.mode == "lod",
                        )
                    )
                ],
            )
        )
        for index in range(args.cameras):
            cameras.append(
                sim.add_sensor(
                    CameraCfg(
                        uid=f"camera_{index}",
                        width=160,
                        height=120,
                        intrinsics=(100, 100, 80, 60),
                        extrinsics=CameraCfg.ExtrinsicsCfg(
                            eye=(0, -3, 1.5), target=(0, 0, 0.7)
                        ),
                    )
                )
            )
        sim.prepare()
        setup = time.perf_counter() - start
        world = sim.get_world()
        handles = visual._visual_binding.handles[0]
        poses = np.stack([h.get_world_pose() for h in handles]).astype(np.float32)
        batch = sim.spawn_result.create_gaussian_splat_batch(list(handles))
        group_ids = [camera.group_id for camera in cameras]

        rng = np.random.default_rng(42)
        state = {}

        def stage_state():
            state.update(
                geometry_scales=rng.uniform(0.8, 1.2, args.envs).astype(np.float32),
                color_gain=rng.uniform(0.5, 1.5, (args.envs, 3)).astype(np.float32),
                opacity_gain=rng.uniform(0.5, 1, args.envs).astype(np.float32),
            )
            visual.set_gaussian_visual_state(
                visual.cfg.visuals[0].id,
                scale=state["geometry_scales"],
                color_gain=state["color_gain"],
                opacity_gain=state["opacity_gain"],
            )

        stage_state()

        def publish_native():
            world.update(0.0)
            batch.apply(
                world_poses=poses, visible=np.ones(args.envs, dtype=bool), **state
            )

        def render_native():
            for group in group_ids:
                world.render_camera_group([group])

        def native():
            publish_native()
            render_native()

        def wrapped():
            with sim.render_frame():
                sim.render_camera_group(group_ids)

        operations = {"native": native, "embodichain": wrapped}
        for _ in range(args.warmup):
            native()
            wrapped()
        samples = {name: [] for name in operations}
        for iteration in range(args.iterations):
            stage_state()
            for name in (
                list(operations) if iteration % 2 == 0 else list(reversed(operations))
            ):
                start = time.perf_counter()
                operations[name]()
                samples[name].append((time.perf_counter() - start) * 1000)
        # A separate pass keeps instrumentation out of the end-to-end samples.
        # Publication measures CPU submission, not completed GPU work; the
        # synchronous render call includes outstanding work and, for the host
        # wrapper, validation and capture metadata construction.
        stages = {
            name: {"publication_ms": [], "render_completion_ms": []}
            for name in operations
        }
        for iteration in range(args.iterations):
            stage_state()
            for name in (
                list(operations) if iteration % 2 == 0 else list(reversed(operations))
            ):
                if name == "native":
                    start = time.perf_counter()
                    publish_native()
                    published = time.perf_counter()
                    render_native()
                    completed = time.perf_counter()
                else:
                    with sim.render_frame():
                        start = time.perf_counter()
                        sim._ensure_render_state()
                        published = time.perf_counter()
                        sim.render_camera_group(group_ids)
                        completed = time.perf_counter()
                stages[name]["publication_ms"].append((published - start) * 1000)
                stages[name]["render_completion_ms"].append(
                    (completed - published) * 1000
                )
        snapshots = {}
        for name, operation in operations.items():
            operation()
            # Native arm bypasses the host capture ledger intentionally. Read
            # its completed native buffer without attaching stale host metadata.
            snapshots[name] = (
                cameras[0]
                ._frame_buffer.get_rgb_gpu_buffer()
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
        error = float(
            np.abs(
                snapshots["native"].astype(float)
                - snapshots["embodichain"].astype(float)
            ).mean()
        )
        profile = world.get_render_profile_data().optix_profile.gaussian_batch
        status = dict(world.get_gaussian_frame_status())
        ids = {h.get_resource_id() for h in handles}
        if len(ids) != 1:
            raise RuntimeError("Replicated asset did not share its native resource")
        if status["outcome"] != "SUCCESS":
            raise RuntimeError(f"Capture failed: {status}")
        mapped_libraries = (
            sorted(
                {
                    line.split()[-1]
                    for line in Path("/proc/self/maps").read_text().splitlines()
                    if any(
                        name in line
                        for name in (
                            "libengine.so",
                            "librender-optix.so",
                            "libDexSim.so",
                        )
                    )
                }
            )
            if sys.platform.startswith("linux")
            else []
        )
        row = {
            "native_libraries": mapped_libraries,
            "envs": args.envs,
            "cameras": args.cameras,
            "mode": args.mode,
            "asset": str(args.asset),
            "source_splats": handles[0].source_splat_count,
            "host_nodes": handles[0].host_node_count,
            "setup_s": setup,
            "unique_resources": len(ids),
            "rss_delta_mib": (psutil.Process().memory_info().rss - rss) / 2**20,
            "resident_mib": (
                float(profile.resident_bytes) / 2**20 if profile.valid else None
            ),
            "peak_workspace_mib": (
                float(profile.peak_workspace_bytes) / 2**20 if profile.valid else None
            ),
            "image_mae_uint8": error,
            "native_status": status["outcome"],
            "native_instance_input_bytes": poses.nbytes
            + args.envs
            + sum(v.nbytes for v in state.values()),
            "timings": {
                name: {
                    "median_ms": statistics.median(values),
                    "p95_ms": float(np.percentile(values, 95)),
                    "samples_ms": values,
                    "stage_medians_ms": {
                        stage: statistics.median(stage_values)
                        for stage, stage_values in stages[name].items()
                    },
                    "stage_samples_ms": stages[name],
                }
                for name, values in samples.items()
            },
        }
        return row
    finally:
        cameras.clear()
        visual = batch = world = None
        sim.destroy(exit_process=False)
        sim.flush_cleanup_queue()


def source_identity() -> str:
    """Hash the integration inputs so a matrix cannot silently mix revisions."""
    import dexsim

    root = Path(__file__).resolve().parents[3]
    paths = [
        root / name
        for name in (
            "embodichain/lab/sim/spawn/_visuals.py",
            "embodichain/lab/sim/spawn/scene.py",
            "embodichain/lab/sim/sim_manager.py",
            "embodichain/lab/sim/sensors/camera.py",
            "embodichain/lab/sim/cfg/visuals.py",
            "embodichain/lab/sim/objects/visual_object.py",
        )
    ]
    paths += [
        Path(dexsim.__file__).parent / name
        for name in ("scene/_scene.py", "spawn/adapters/dexsim_adapter.py")
    ]
    paths.append(Path(__file__).resolve())
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    """Run isolated workload shapes and emit one three-table Markdown report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--lod-path", type=Path)
    parser.add_argument("--env-counts", nargs="+", type=int, default=[1, 16, 64])
    parser.add_argument("--camera-counts", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--renderer", choices=["fast-rt", "hybrid"], default="fast-rt")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/gaussian_visuals.md")
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--cameras", type=int, default=1)
    parser.add_argument("--mode", choices=["flat", "lod"], default="flat")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1:
        parser.error("iterations and warmup must be positive")
    if args.worker:
        print("RESULT_JSON=" + json.dumps(worker(args)), flush=True)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    rows = []
    failures = []
    for mode in (["flat", "lod"] if args.lod_path else ["flat"]):
        for envs in args.env_counts:
            for cameras in args.camera_counts:
                command = [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--asset",
                    str(args.asset.resolve()),
                    "--envs",
                    str(envs),
                    "--cameras",
                    str(cameras),
                    "--mode",
                    mode,
                    "--iterations",
                    str(args.iterations),
                    "--warmup",
                    str(args.warmup),
                    "--renderer",
                    args.renderer,
                ]
                if args.lod_path:
                    command += ["--lod-path", str(args.lod_path.resolve())]
                if source_identity() != identity:
                    raise RuntimeError(
                        "Integration sources changed during benchmark; rerun the matrix."
                    )
                result = subprocess.run(command, capture_output=True, text=True)
                if source_identity() != identity:
                    raise RuntimeError(
                        "Integration sources changed during a worker; measurements rejected."
                    )
                logs = args.output.with_suffix(f".{mode}-{envs}-{cameras}.log")
                logs.write_text(result.stdout + result.stderr)
                markers = [
                    line[len("RESULT_JSON=") :]
                    for line in result.stdout.splitlines()
                    if line.startswith("RESULT_JSON=")
                ]
                if result.returncode or not markers:
                    failures.append(f"{mode}/{envs}/{cameras}: failed, see {logs.name}")
                else:
                    rows.append(json.loads(markers[-1]))
                print(
                    f"{mode} envs={envs} cameras={cameras}: {'passed' if result.returncode==0 and markers else 'FAILED'}",
                    flush=True,
                )
    import dexsim

    native = Path(dexsim.__file__).parent / "lib" / "libDexSim.so"
    notes = [
        f"Integration source SHA256: {identity}",
        f"Asset SHA256: {hashlib.sha256(args.asset.read_bytes()).hexdigest()}",
        f"Python: {sys.executable}",
        f"DexSim package: {dexsim.__file__}",
        f"Native library: {native.resolve()}",
        f"libDexSim SHA256: {hashlib.sha256(native.read_bytes()).hexdigest()}",
        f"Renderer: {args.renderer}; 160x120; RAW; CPU pose state; synchronous native CameraGroup completion; {args.warmup} warmups, {args.iterations} paired samples.",
        "Setup and process RSS retained delta are shared between arms. Native counters cover Gaussian resources/workspaces, not total GPU memory. No device-copy tracer was run; GPU transfer byte counts are unmeasured.",
        "Native and wrapper arms share the prepared Scene and CameraGroups. Mesh sampling introduces small RGB differences. No image tensors are copied in the timed loops.",
        "Stage medians come from a separate paired pass: publication is CPU submission; render completion includes outstanding GPU work and wrapper capture metadata. Stage medians need not sum to end-to-end medians. Workload changes independent scale/RGB/opacity values each paired iteration (seed 42); both arms receive the same state. Instance input bytes count native float32 rigid matrices, uniform scales, RGB/opacity gains and bool visibility, not traced device traffic or all wrapper allocations.",
    ]
    if rows:
        notes.append("Loaded libraries: " + "; ".join(rows[0]["native_libraries"]))
    for name in (
        "libengine.so",
        "librender-optix.so",
        "librender-hybrid.so",
        "libadaptation.so",
        "libcuda_compute.so",
    ):
        library = native.parent / name
        if library.exists():
            notes.append(
                f"{name} SHA256: {hashlib.sha256(library.read_bytes()).hexdigest()}"
            )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    notes.append(gpu.stdout.strip())
    if not args.lod_path:
        notes.append("Dynamic LoD unmeasured: provide --lod-path.")
    lines = [
        "# Gaussian visual integration benchmark",
        "",
        *notes,
        "",
        "## Time & Memory",
        "",
        "| Mode | Envs | Cameras | Arm | Median ms | P95 ms | Publication ms | Render completion ms | Setup s | Gaussian resident MiB | Workspace peak MiB | RSS retained delta MiB |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for name, timing in row["timings"].items():
            lines.append(
                f"| {row['mode']} | {row['envs']} | {row['cameras']} | {name} | {timing['median_ms']:.3f} | {timing['p95_ms']:.3f} | {timing['stage_medians_ms']['publication_ms']:.3f} | {timing['stage_medians_ms']['render_completion_ms']:.3f} | {row['setup_s']:.3f} | {row['resident_mib']} | {row['peak_workspace_mib']} | {row['rss_delta_mib']:.2f} |"
            )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| Mode | Envs | Cameras | Source splats | Host nodes | Unique resources | Instance input bytes | Status | RGB MAE uint8 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['envs']} | {row['cameras']} | {row['source_splats']} | {row['host_nodes']} | {row['unique_resources']} | {row['native_instance_input_bytes']} | {row['native_status']} | {row['image_mae_uint8']:.4f} |"
        )
    total = len(rows) + len(failures)
    lines += [
        "",
        "## Leaderboard",
        "",
        "| Arm | Successful workloads | Attempted workloads | Success rate |",
        "|---|---:|---:|---:|",
    ]
    for name in ["native", "embodichain"]:
        lines.append(
            f"| {name} | {len(rows)} | {total} | {len(rows)/total if total else 0:.3f} |"
        )
    lines += [
        "",
        "Workloads have different view counts; timing is not aggregated into a cross-workload speed ranking.",
        *failures,
    ]
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output.resolve(), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
