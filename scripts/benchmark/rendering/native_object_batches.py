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

"""Compare real Camera/StereoCamera/Light facades against a reference checkout.

Run: python -m scripts.benchmark.rendering.native_object_batches --num-envs 8
     --reference-root .worktrees/camera-light-baseline --renderer fast-rt

Each side runs in a fresh subprocess using the same interpreter/native SDK.
Setters alternate precomputed inputs; creation, input generation and numerical
checks are excluded from steady-state latency. Camera update includes rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _measure(
    operation: Callable[[], Any],
    synchronize: Callable[[], None],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Measure completed calls and a separate warmed retained-memory delta."""
    import psutil
    import torch

    for _ in range(warmup):
        value = operation()
        synchronize()
        del value
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        value = operation()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000)
        del value
    process = psutil.Process()
    rss = process.memory_info().rss
    allocated = torch.cuda.memory_allocated(0)
    torch.cuda.reset_peak_memory_stats(0)
    value = operation()
    synchronize()
    result = dict(
        cost_time_ms=statistics.median(samples),
        samples_ms=samples,
        cpu_delta_mb=(process.memory_info().rss - rss) / 2**20,
        gpu_delta_mb=(torch.cuda.memory_allocated(0) - allocated) / 2**20,
        peak_gpu_mb=torch.cuda.max_memory_allocated(0) / 2**20,
        incremental_peak_gpu_mb=(torch.cuda.max_memory_allocated(0) - allocated)
        / 2**20,
    )
    del value
    return result


def _worker(args: argparse.Namespace) -> None:
    """Run the imported checkout's real public facades on the selected SDK."""
    import gc
    import importlib.metadata
    import platform
    import numpy as np
    import torch
    import dexsim
    import embodichain
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import DenoisingCfg, LightCfg, RenderCfg
    from embodichain.lab.sim.sensors import CameraCfg, StereoCameraCfg

    root = Path(args.source_root).resolve()
    Path(embodichain.__file__).resolve().relative_to(root / "embodichain")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            num_envs=args.num_envs,
            device="cuda:0",
            startup_summary="off",
            render_cfg=RenderCfg(
                renderer=args.renderer,
                denoising=DenoisingCfg(window="optix", offscreen="optix"),
            ),
        )
    )
    rows = []
    checks = []

    def synchronize() -> None:
        """Join native owner publication and the participating CUDA device."""
        status = sim.get_world()._flush_render_updates()
        if status < 0:
            raise RuntimeError(f"Native render flush failed: {status}")
        torch.cuda.synchronize(0)

    def equal(actual: Any, expected: Any) -> None:
        """Compare with independently prepared expected values outside timing."""
        a = (
            actual.detach().cpu().numpy()
            if torch.is_tensor(actual)
            else np.asarray(actual)
        )
        b = (
            expected.detach().cpu().numpy()
            if torch.is_tensor(expected)
            else np.asarray(expected)
        )
        np.testing.assert_allclose(a, b, atol=2e-4, rtol=1e-5)

    try:
        cfg_type = StereoCameraCfg if args.sensor == "stereo" else CameraCfg
        cfg = cfg_type(
            uid="perf_camera",
            width=32,
            height=32,
            intrinsics=(40, 42, 16, 15),
            enable_depth=True,
        )
        start = time.perf_counter()
        camera = sim.add_sensor(cfg)
        camera_create_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        light = sim.add_light(
            LightCfg(uid="perf_light", light_type="spot", init_pos=(0, 0, 2))
        )
        light_create_ms = (time.perf_counter() - start) * 1000
        sim.prepare()
        sim.enable_physics(False)
        count = args.num_envs

        def intrinsics():
            """Read the logical left eye for either public sensor type."""
            result = camera.get_intrinsics()
            return result[0] if args.sensor == "stereo" else result

        for device in ("cpu", "cuda:0"):
            ks = [
                torch.tensor(
                    [40.0 + i * 5, 42.0 + i * 3, 16, 15], device=device
                ).repeat(count, 1)
                for i in range(2)
            ]
            poses = [torch.eye(4, device=device).repeat(count, 1, 1) for _ in range(2)]
            for i, pose in enumerate(poses):
                pose[:, 2, 3] = 2 + i * 0.2
            colors = [
                torch.tensor([0.2 + i * 0.1, 0.3, 0.4], device=device).repeat(count, 1)
                for i in range(2)
            ]
            intensities = [
                torch.full((count,), 20.0 + i * 5, device=device) for i in range(2)
            ]
            eyes = [pose[:, :3, 3] for pose in poses]
            targets = torch.zeros((count, 3), device=device)
            up = torch.tensor([0.0, 1.0, 0.0], device=device).repeat(count, 1)

            def record(name, operation, verify):
                """Alternate immutable payloads, then validate the final completed call."""
                calls = 0
                last = 0

                def callback():
                    nonlocal calls, last
                    last = calls % 2
                    calls += 1
                    return operation(last)

                metrics = _measure(callback, synchronize, args.warmup, args.repeats)
                verify(last)
                checks.append(f"{device}: {name}")
                rows.append(dict(operation=name, input_device=device, **metrics))
                return last

            last_k = record(
                "camera_intrinsics_write",
                lambda i: camera.set_intrinsics(ks[i]),
                lambda i: equal(intrinsics()[:, (0, 1, 0, 1), (0, 1, 2, 2)], ks[i]),
            )
            record(
                "camera_intrinsics_read",
                lambda _i: camera.get_intrinsics(),
                lambda _i: equal(
                    intrinsics()[:, (0, 1, 0, 1), (0, 1, 2, 2)], ks[last_k]
                ),
            )
            last_pose = record(
                "camera_pose_write",
                lambda i: camera.set_local_pose(poses[i]),
                lambda i: equal(camera.get_local_pose(to_matrix=True), poses[i]),
            )
            record(
                "camera_pose_read",
                lambda _i: camera.get_local_pose(to_matrix=True),
                lambda _i: equal(
                    camera.get_local_pose(to_matrix=True), poses[last_pose]
                ),
            )
            record(
                "camera_look_at",
                lambda i: camera.look_at(eyes[i], targets, up),
                lambda i: equal(camera.get_local_pose(to_matrix=True), poses[i]),
            )
            reset_pose = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0])).repeat(
                count, 1, 1
            )
            record(
                "camera_reset",
                lambda _i: camera.reset(),
                lambda _i: equal(camera.get_local_pose(to_matrix=True), reset_pose),
            )

            def light_fields(i):
                """Compare previous separate calls with the new fused consumer path."""
                if args.variant == "batch":
                    light.apply_state(
                        color=colors[i], intensity=intensities[i], position=eyes[i]
                    )
                else:
                    light.set_color(colors[i])
                    light.set_intensity(intensities[i])
                    light.set_local_pose(eyes[i])

            def check_light(i):
                """Read color/intensity through native getters, independent of the writer."""
                equal([entity.get_color() for entity in light._entities], colors[i])
                equal(
                    [entity.get_intensity() for entity in light._entities],
                    intensities[i],
                )
                equal(light.get_local_pose(), eyes[i])

            record("light_color_intensity_position", light_fields, check_light)
            light_pose = record(
                "light_position_write",
                lambda i: light.set_local_pose(eyes[i]),
                lambda i: equal(light.get_local_pose(), eyes[i]),
            )
            record(
                "light_pose_read",
                lambda _i: light.get_local_pose(to_matrix=True),
                lambda _i: equal(light.get_local_pose(), eyes[light_pose]),
            )
            record(
                "light_reset",
                lambda _i: light.reset(),
                lambda _i: equal(
                    light.get_local_pose(),
                    torch.tensor([0.0, 0.0, 2.0]).repeat(count, 1),
                ),
            )
        camera.set_local_pose(poses[0])
        record(
            "camera_image_update",
            lambda _i: camera.update(),
            lambda _i: equal(
                tuple(camera.get_data()["color"].shape), (count, 32, 32, 4)
            ),
        )
        setup = dict(
            camera_create_once_ms=camera_create_ms, light_create_once_ms=light_create_ms
        )
    finally:
        sim.destroy(exit_process=False)
    del camera, light, sim
    gc.collect()
    SimulationManager.flush_cleanup_queue()
    environment = dict(
        source=str(root),
        imported=embodichain.__file__,
        commit=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        source_hashes={
            p: hashlib.sha256((root / p).read_bytes()).hexdigest()
            for p in (
                "embodichain/lab/sim/sensors/camera.py",
                "embodichain/lab/sim/sensors/stereo.py",
                "embodichain/lab/sim/objects/light.py",
            )
        },
        python=platform.python_version(),
        torch=torch.__version__,
        dexsim_commit=dexsim.__commit_id__,
        dexsim_version=importlib.metadata.version("dexsim-engine"),
        gpu=torch.cuda.get_device_name(0),
        renderer=args.renderer,
        sensor=args.sensor,
        num_envs=count,
        dtype="float32",
        seed=42,
        torch_threads=1,
        warmup=args.warmup,
        repeats=args.repeats,
        denoising="optix",
        measurement_order=args.order,
    )
    Path(args.worker_output).write_text(
        json.dumps(
            dict(
                environment=environment, measurements=rows, checks=checks, setup=setup
            ),
            indent=2,
        )
        + "\n"
    )


def _report(results: dict[str, Any], path: Path) -> None:
    """Write the benchmark's three standard tables and precise measurement scope."""
    before, after = results["baseline"], results["batch"]
    lines = [
        "# Camera/light facade migration comparison",
        "",
        "```json",
        json.dumps(
            {name: run["environment"] for name, run in results.items()}, indent=2
        ),
        "```",
        "",
        "## Time & Memory",
        "",
        "| operation | input device | algorithm | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for algorithm, run in results.items():
        for r in run["measurements"]:
            lines.append(
                f"| {r['operation']} | {r['input_device']} | {algorithm} | {r['cost_time_ms']:.4f} | {r['cpu_delta_mb']:.4f} | {r['gpu_delta_mb']:.4f} | {r['peak_gpu_mb']:.4f} |"
            )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| operation | input device | baseline / batch speed ratio | success_rate |",
        "| --- | --- | ---: | ---: |",
    ]
    for a, b in zip(before["measurements"], after["measurements"], strict=True):
        assert (a["operation"], a["input_device"]) == (
            b["operation"],
            b["input_device"],
        )
        lines.append(
            f"| {a['operation']} | {a['input_device']} | {a['cost_time_ms']/b['cost_time_ms']:.3f} | 1.0 |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | overall_success_rate |",
        "| ---: | --- | ---: |",
        "| 1 | baseline | 1.0 |",
        "| 1 | batch | 1.0 |",
        "",
        "Success ties do not rank speed. State setters alternate precomputed inputs. Input generation, correctness reads and setup are excluded from repeated latency. Native render flush and CUDA synchronization finish each call. camera_image_update includes rendering; other rows exclude it. Setup timings are separately recorded single cold observations, not repeated steady-state comparisons.",
        "",
        "Memory is a separate warmed call. CPU RSS is retained delta; GPU statistics cover only the Torch allocator, excluding native render allocations. The JSON records incremental and total Torch peaks and all latency samples. Current and reference public facades run on the same native SDK in separate processes; source paths and hashes are recorded. CUDA input is not a device-residency guarantee.",
        "",
        "Single resource creation observations: "
        + json.dumps({a: r["setup"] for a, r in results.items()}),
    ]
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n")


def run_all_benchmarks() -> None:
    """Dispatch isolated before/after workers and save one report per run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--sensor", choices=("mono", "stereo"), default="mono")
    parser.add_argument(
        "--renderer", choices=("hybrid", "fast-rt", "rt"), default="hybrid"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--order", choices=("baseline-first", "batch-first"), default="baseline-first"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--variant", choices=("baseline", "batch"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if min(args.num_envs, args.warmup, args.repeats) < 1:
        parser.error("batch size, warmup and repeats must be positive")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    if args.worker_output:
        _worker(args)
        return
    root = Path(__file__).resolve().parents[3]
    reference = (
        args.reference_root or root / ".worktrees/camera-light-baseline"
    ).resolve()
    if not (reference / "embodichain").is_dir():
        parser.error("--reference-root must be a real unmodified reference checkout")
    output = (
        args.output
        or root
        / f"outputs/benchmarks/camera_light_{args.renderer}_{args.sensor}_{args.num_envs}.md"
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    variants = [("baseline", reference), ("batch", root)]
    if args.order == "batch-first":
        variants.reverse()
    for variant, source in variants:
        child = output.with_name(output.stem + f"-{variant}.json")
        log = child.with_suffix(".log")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-output",
            str(child),
            "--source-root",
            str(source),
            "--variant",
            variant,
            "--num-envs",
            str(args.num_envs),
            "--sensor",
            args.sensor,
            "--renderer",
            args.renderer,
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--order",
            args.order,
        ]
        source_paths = [source, source / "extensions"] + sorted(
            path
            for path in (source / "extensions").iterdir()
            if (path / path.name / "__init__.py").is_file()
        )
        with log.open("w") as stream:
            completed = subprocess.run(
                command,
                cwd=source,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(map(str, source_paths)),
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode:
            raise RuntimeError(f"{variant} failed ({completed.returncode}); see {log}")
        results[variant] = json.loads(child.read_text())
    assert (
        results["baseline"]["environment"]["dexsim_commit"]
        == results["batch"]["environment"]["dexsim_commit"]
    )
    _report(results, output)
    print(f"Markdown report saved: {output}")


if __name__ == "__main__":
    run_all_benchmarks()
