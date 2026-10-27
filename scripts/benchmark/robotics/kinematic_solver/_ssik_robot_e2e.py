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

"""Isolated real-Robot worker for the unified SSIK benchmark.

The maintained entry point is run_benchmark.py --solvers ssik
--ssik-robot-e2e. A separate process contains renderer/physics allocation and
teardown, keeping ordinary solver timings and memory measurements independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

__all__: list[str] = []


class _RobotIKCall:
    """Adapt arena-frame Robot calls to the existing control timing helper."""

    def __init__(self, robot: Any, part: str, solver: Any) -> None:
        self.robot, self.part, self.cfg = robot, part, solver.cfg

    def get_ik(
        self, target: torch.Tensor, seed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Include actual root-state retrieval and Robot coordinate conversion."""
        return self.robot.compute_ik(target, joint_seed=seed, name=self.part)


def _run(spec_path: Path, output: Path) -> None:
    """Initialize one engine-backed robot and record changing-target calls."""
    from embodichain.lab.sim.motion.solvers import SolverCfg
    from .ssik_benchmark import (
        _load_model,
        _measure_streaming_control,
        _versions,
        evaluate_quality,
    )

    spec = json.loads(spec_path.read_text())
    torch.set_num_threads(spec["cpu_threads"])
    logging.getLogger("ssik").setLevel(logging.WARNING)
    arms = [spec] + ([spec["partner"]] if "partner" in spec else [])
    models = [_load_model(arm["model"]) for arm in arms]
    cfg = models[0].cfg.copy()
    cfg.init_pos = [0.137, -0.251, 0.119]
    parts = [
        (
            "arm"
            if model.name in ("arx5", "flexiv_rizon4s")
            else ("left_arm" if model.name.endswith("left") else "right_arm")
        )
        for model in models
    ]
    cfg.solver_cfg = {
        part: SolverCfg.from_dict(arm["solver_cfg"]) for part, arm in zip(parts, arms)
    }
    device = torch.device(spec["device"])
    begun = time.perf_counter()
    try:
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import RenderCfg

        sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                device=str(device),
                num_envs=1,
                render_cfg=RenderCfg(renderer="no-render"),
            ),
            requires_native_renderer=False,
        )
    except (ImportError, OSError, RuntimeError) as error:
        output.write_text(
            json.dumps(
                {
                    "status": "unavailable",
                    "phase": "engine_setup",
                    "reason": repr(error),
                },
                indent=2,
            )
            + "\n"
        )
        return
    try:
        robot = sim.add_robot(cfg=cfg)
        sim.prepare()
        solvers, roots, goals, initials, limits, fixtures = [], [], [], [], [], []
        for part, arm in zip(parts, arms):
            solver = robot.get_solver(part)
            if solver is None:
                raise RuntimeError(f"Robot did not bind its configured {part} solver")
            root = robot.get_link_pose(link_name=solver.root_link_name, to_matrix=True)
            root_cpu = root.detach().cpu().double()[0]
            fixture = Path(arm["fixture"])
            with np.load(fixture, allow_pickle=False) as archive:
                local_targets = torch.from_numpy(
                    archive["trajectory_target"].copy()
                ).double()
                initial = torch.from_numpy(archive["trajectory_seed"].copy()).float()
            solvers.append(solver)
            roots.append(root_cpu)
            goals.append(root_cpu @ local_targets)
            initials.append(initial)
            limits.append(
                torch.stack((solver.lower_qpos_limits, solver.upper_qpos_limits))
                .detach()
                .cpu()
                .double()
            )
            fixtures.append(fixture)
        setup_seconds = time.perf_counter() - begun
        measured = _measure_streaming_control(
            tuple(
                _RobotIKCall(robot, part, solver)
                for part, solver in zip(parts, solvers)
            ),
            tuple(goals),
            tuple(initials),
            device,
            repeats=spec["repeats"],
            warmup=spec["warmup"],
        )
        qualities = []
        for index, (model, root, targets, initial, bound) in enumerate(
            zip(models, roots, goals, initials, limits)
        ):
            reported, joints = measured.valid[index], measured.qpos[index]
            quality = evaluate_quality(
                targets.flatten(0, 1)[: spec["repeats"]],
                joints,
                reported,
                lambda q: root @ model.fk(q),
                bound,
            )
            preceding = joints.roll(1, 0)
            starts = torch.where(
                torch.isfinite(initial), initial, bound.mean(0).float()
            ).clamp(min=bound[0].float(), max=bound[1].float())
            boundaries = torch.arange(0, spec["repeats"], targets.shape[1])
            preceding[boundaries] = starts[: len(boundaries)]
            step = (joints - preceding).abs().amax(-1)
            quality.update(
                step_violations=int((reported & (step > 0.05)).sum()),
                failed_hold_violations=int((~reported & (step != 0)).sum()),
                max_accepted_joint_step_rad=(
                    float(step[reported].max()) if reported.any() else None
                ),
            )
            qualities.append(quality)
        quality = (
            qualities[0]
            if len(qualities) == 1
            else {
                "arms": qualities,
                **{
                    key: sum(row[key] for row in qualities)
                    for key in (
                        "false_successes",
                        "step_violations",
                        "failed_hold_violations",
                    )
                },
                "success_rate": (
                    measured.latency["both_reported_success_rate"]
                    if not any(row["false_successes"] for row in qualities)
                    else 0.0
                ),
            }
        )
        record = {
            "worker_schema_version": 2,
            "status": "completed",
            "scope": (
                "robot_streaming_control"
                if len(arms) == 1
                else "robot_paired_streaming_control"
            ),
            "actual_simulation_manager": True,
            "renderer": "no-render",
            "physics_stepped": False,
            "includes": "actual engine root-state query, arena/root conversion, Robot dispatch and device completion",
            "excludes": "setup/JIT warmup, target generation, physical execution, transport from external devices",
            "setup_seconds": setup_seconds,
            "runtime": _versions(),
            "model_manifest": [model.manifest for model in models],
            "effective_limits": [bound.tolist() for bound in limits],
            "root_transform": [root.tolist() for root in roots],
            "control_part": parts,
            "fixture_sha256": [
                hashlib.sha256(fixture.read_bytes()).hexdigest() for fixture in fixtures
            ],
            "max_joint_step_rad": 0.05,
            "warmup": spec["warmup"],
            "latency": measured.latency,
            "quality": quality,
        }
        arrays = {"call_ms": np.asarray(measured.samples_ms)}
        for index, (joints, reported) in enumerate(zip(measured.qpos, measured.valid)):
            suffix = "" if len(arms) == 1 else f"_{index}"
            arrays[f"qpos{suffix}"] = joints.numpy()
            arrays[f"reported{suffix}"] = reported.numpy()
        np.savez_compressed(output.with_suffix(".npz"), **arrays)
        output.write_text(json.dumps(record, indent=2) + "\n")
    finally:
        sim.destroy(exit_process=False)


def _main() -> None:
    """Dispatch one private worker specification from the unified runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _run(args.spec, args.output)


if __name__ == "__main__":
    _main()
