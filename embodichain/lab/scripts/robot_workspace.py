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

"""Run the UR5 scene-workspace demonstration and optional Task Program trial."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections.abc import Sequence
from typing import Any

__all__ = ["main", "run_demo"]


def run_demo(
    *,
    mode: str = "scene",
    view: str = "viser",
    seed: int = 7,
    candidates: int = 3,
    output: str | Path = "/tmp/embodichain-workspace-demo",
    viser_port: int = 8080,
) -> dict[str, Any]:
    """Run real workspace sampling, physical checks and restoration.

    Args:
        mode: Independent scene services or Task Program execution.
        view: Viser, native window, offscreen video, or no renderer. Offscreen
            writes the control-cadence preview and a workspace explanation
            video with separately labelled inspection pauses.
        seed: Nonnegative reproducible sampling seed.
        candidates: Required number of distinct valid candidates.
        output: Destination for reports, cache, trajectory and optional images.
        viser_port: Browser server port.

    Returns:
        Measured report, also saved as report.json. Failed runs write an error
        report before propagating their exception.
    """
    if mode not in {"scene", "task"} or view not in {
        "viser",
        "window",
        "offscreen",
        "none",
    }:
        raise ValueError("Unsupported demo mode or view")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(candidates) is not int or candidates < 1:
        raise ValueError("candidates must be a positive integer")
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    modules_before = set(sys.modules)
    from embodichain.lab.sim import SimulationManager
    from ._robot_workspace_view import _Display
    from ._robot_workspace_scene import (
        TABLE_LOWER,
        TABLE_UPPER,
        GOAL,
        NEGATIVE,
        TABLE_Z,
        TABLE_CENTER,
        TABLE_RADIUS,
        CANDIDATE_XY_BOUNDS,
    )

    display = _Display(view, destination, viser_port)
    report = {
        "schema_version": 1,
        "mode": mode,
        "view": view,
        "seed": seed,
        "physics": "default",
        "device": "cpu",
        "accepted": False,
        "layout": {
            "table_shape": "round",
            "table_center_xy": TABLE_CENTER,
            "table_radius_m": TABLE_RADIUS,
            "table_xy_bounds": (TABLE_LOWER, TABLE_UPPER),
            "candidate_xy_bounds": CANDIDATE_XY_BOUNDS,
            "robot_base": (0.0, 0.0, TABLE_Z),
            "goal": GOAL,
            "negative": NEGATIVE,
        },
    }
    sim = None
    try:
        if mode == "task":
            from ._robot_workspace_task import _run_task

            report.update(_run_task(destination, seed, candidates, display))
        else:
            from embodichain.lab.sim.scene import declare_fragment
            from ._robot_workspace_scene import _robot_cfg, _fragment, _run_scene

            sim = SimulationManager(display.config())
            robot = sim.add_robot(cfg=_robot_cfg())
            declare_fragment(sim, _fragment())
            display.attach(sim)
            sim.prepare()
            display.start(sim)
            report.update(
                _run_scene(sim, robot, destination, seed, candidates, display)
            )
            imported_task = any(
                name.startswith("embodichain.lab.task_program")
                for name in set(sys.modules) - modules_before
            )
            report["scene_imported_task_program"] = imported_task
            imported_gym = any(
                name.startswith("embodichain.lab.gym")
                for name in set(sys.modules) - modules_before
            )
            report["scene_imported_gym"] = imported_gym
            if imported_task or imported_gym:
                raise RuntimeError(
                    "Independent scene mode imported Gym or Task Program"
                )
        if not report["accepted"]:
            raise RuntimeError("Demo acceptance checks failed; inspect report.json")
        display.close()
        if view == "offscreen":
            from ._robot_workspace_presentation import _write_walkthrough

            display.stage = "workspace_presentation"
            report["video_events"] = display.video_events
            report["presentation"] = _write_walkthrough(report, destination)
        return report
    except BaseException as error:
        report.update(
            accepted=False,
            error={
                "stage": display.stage,
                "type": type(error).__name__,
                "detail": str(error),
            },
        )
        raise
    finally:
        report["keyframes"] = display.keyframes
        try:
            (destination / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
            )
        finally:
            try:
                display.close()
            finally:
                try:
                    if sim is not None:
                        sim.destroy(exit_process=False)
                finally:
                    SimulationManager.flush_cleanup_queue()


def main(argv: Sequence[str] | None = None) -> None:
    """Parse workspace-demo arguments without initializing simulation for help.

    Args:
        argv: Explicit arguments or the process command line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scene", "task"), default="scene")
    parser.add_argument(
        "--view", choices=("viser", "window", "offscreen", "none"), default="viser"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/embodichain-workspace-demo")
    )
    parser.add_argument("--viser-port", type=int, default=8080)
    args = parser.parse_args(argv)
    report = run_demo(**vars(args))
    print(
        json.dumps(
            {"accepted": report["accepted"], "output": str(args.output)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
