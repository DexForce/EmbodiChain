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
"""Measure nominal native assembly construction, zero-action steps and grasp.

Run: python -m scripts.benchmark.assembly.task_smoke --steps 25
This is a lifecycle/contact smoke test, not a learned success-rate experiment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import torch
import warp as wp

__all__: list[str] = []


def _synchronize(device: torch.device) -> None:
    """Complete both the Newton/Warp and Torch work on the tested device."""
    wp.synchronize_device(str(device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_task(task: str, steps: int, device: torch.device) -> dict[str, Any]:
    """Measure one real task while preserving its contact and reset path."""
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import make
    from embodichain.lab.sim import SimulationManager
    from embodichain.utils.utility import load_config

    root = Path(__file__).resolve().parents[3]
    config_path = (
        root
        / "extensions/embodichain_tasks/configs/tasks/manipulation/assembly"
        / task
        / "task.study.nominal.yaml"
    )
    env = None
    try:
        raw = load_config(str(config_path))
        cfg = config_to_cfg(raw, source_path=config_path)
        cfg.num_envs = 1
        cfg.seed = 0
        cfg.sim_cfg.headless = True
        cfg.sim_cfg.device = str(device)
        start = time.perf_counter()
        env = make(raw["id"], cfg=cfg)
        env.reset(seed=0)
        _synchronize(device)
        setup_ms = (time.perf_counter() - start) * 1000
        action = torch.zeros((1, 7), device=device)
        samples = []
        grasp = []
        resets = 0
        for index in range(steps):
            _synchronize(device)
            start = time.perf_counter()
            obs, _, terminated, truncated, info = env.step(action)
            _synchronize(device)
            elapsed = (time.perf_counter() - start) * 1000
            if index >= 5:
                samples.append(elapsed)
            grasp.append(bool(info["metrics"]["grasped"].item()))
            resets += int((terminated | truncated).sum().item())
        if not torch.isfinite(obs["proprioception"]).all():
            raise RuntimeError("Non-finite actor observations.")
        return {
            "task": task,
            "status": "passed",
            "num_envs": 1,
            "steps": steps,
            "seed": 0,
            "setup_including_generation_ms": setup_ms,
            "cost_time_ms": statistics.median(samples),
            "samples_ms": samples,
            "bilateral_grasp_fraction": sum(grasp) / len(grasp),
            "grasp_by_step": grasp,
            "resets": resets,
            "actor_shape": list(obs["proprioception"].shape),
            "critic_shape": list(obs["privileged"].shape),
            "contact_classes_observed": obs["contact_labels"].unique().tolist(),
            "final_metrics": {
                key: value.tolist() for key, value in info["metrics"].items()
            },
            "fidelity": env.assembly_fidelity,
            "success_rate": None,
        }
    except Exception as error:
        return {
            "task": task,
            "status": "failed",
            "error": str(error),
            "success_rate": None,
        }
    finally:
        if env is not None:
            env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def run_all_benchmarks(output: Path, steps: int, device: torch.device) -> None:
    """Write one report and machine-readable measurements for the three tasks."""
    import embodichain_tasks  # noqa: F401 -- task/functor registration

    packages = {}
    for name in ("torch", "warp-lang", "newton", "mujoco", "dexsim", "mjfit"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    rows = [
        _run_task(task, steps, device)
        for task in ("peg_insertion", "gear_meshing", "nut_threading")
    ]
    data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "packages": packages,
        "warmup_steps": 5,
        "dtype": "float32",
        "scope": "completed end-to-end control step; both Warp and Torch synchronized",
        "memory_scope": "not measured; no allocator or total-device memory claim",
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")
    lines = [
        "# Native assembly task smoke",
        "",
        "Untrained zero-action lifecycle/contact checks; learned success rates were not evaluated.",
        "",
        "## Time & Memory",
        "",
        "| task | cost_time_ms | setup_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row.get('cost_time_ms', 'unavailable')} | {row.get('setup_including_generation_ms', 'unavailable')} | unmeasured | unmeasured | unmeasured |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| task | lifecycle | bilateral_grasp_fraction | resets | success_rate |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['status']} | {row.get('bilateral_grasp_fraction', 'unavailable')} | {row.get('resets', 'unavailable')} | not evaluated |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| implementation | overall_success_rate | qualification |",
        "|---|---|---|",
        "| Native nominal assembly | not evaluated | lifecycle smoke; no learned-policy comparison |",
        "",
        "The explicit parallel-jaw model changes source linkage dynamics. These numbers do not qualify paper reproduction. GPU load from other processes can affect timing. Raw JSON records versions, fidelity, measurements and final states.",
        "",
    ]
    output.with_suffix(".md").write_text("\n".join(lines))
    print(f"Markdown report saved: {output.with_suffix('.md')}", flush=True)
    if any(row["status"] != "passed" for row in rows):
        raise SystemExit(1)


def main() -> None:
    """Select the bounded native smoke workload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/assembly_task_smoke")
    )
    args = parser.parse_args()
    if args.steps < 6:
        parser.error("--steps must be at least 6 to include warmed measurements.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.exit(
            message="Skipped: CUDA is unavailable for the native assembly smoke.\n"
        )
    run_all_benchmarks(args.output, args.steps, device)


if __name__ == "__main__":
    main()
