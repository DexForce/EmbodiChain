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

"""Replan a saved grasp through smooth direct waypoints and physically replay it."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.tools.assemble._json_io import read_json, write_json
from scripts.tools.grasp_assemble._config import (
    load_assembly,
    execution_settings,
    motion_settings,
)

__all__ = ["main"]


def _replay(result: dict, headless: bool) -> dict:
    from scripts.tools.grasp_assemble._geometry import GraspGeometry
    from scripts.tools.grasp_assemble._robot import RobotSession

    config = result["config"]
    config["execution"] = execution_settings(config.get("execution"))
    config["motion"] = motion_settings(config.get("motion"))
    if (
        hashlib.sha256(Path(config["assembly_result"]).read_bytes()).hexdigest()
        != result["assembly_sha256"]
    ):
        raise ValueError("Assembly JSON changed after grasp planning")
    assembly = load_assembly(config)
    directory = Path(result["run_directory"])
    geometry = GraspGeometry(config, assembly, directory / "cache")
    candidate = geometry.evaluate(result["T_assemble_tcp"])
    if not candidate["accepted"]:
        raise ValueError(f"Grasp no longer valid: {candidate['reasons']}")
    session = RobotSession(config, assembly)
    try:
        plan = session.plan(candidate, geometry, directory / "replay_plan")
        write_json(directory / "replay_plan" / "plan.json", plan)
        if not plan["accepted"]:
            raise ValueError(f"Motion no longer valid: {plan}")
        print(
            f"[grasp] Planning: {plan['planning_seconds']:.2f}s; "
            f"trajectory duration: {plan['motion_metrics']['duration_seconds']:.2f}s; "
            f"weighted cost: {plan['motion_metrics']['score']:.1f}; "
            f"details: {directory / 'replay_plan' / 'plan.json'}",
            flush=True,
        )
        if not headless:
            session.show()
            print(
                "[grasp] Waypoint planning accepted. Press Enter to replay, or Ctrl+C to exit.",
                flush=True,
            )
            input()
        observed = session.replay(candidate, directory, headless)
        write_json(directory / "execution.json", observed)
        print(
            f"[grasp] Mode: {observed['grasp_mode']}; "
            f"Waypoint trajectory completed: {observed['trajectory_completed']}; "
            f"physical success: {observed['physical_success']}; "
            f"position error: {observed['position_error_m']:.4f} m; "
            f"rotation error: {observed['rotation_error_degrees']:.2f} deg; "
            f"allowed: {observed['pose_tolerances']['rotation_degrees']:.2f} deg; "
            f"details: {directory / 'execution.json'}",
            flush=True,
        )
        if not headless:
            input("Inspect the final scene, then press Enter to close... ")
        return observed
    except Exception as error:
        write_json(
            directory / "execution.json",
            {
                "trajectory_completed": False,
                "physical_success": False,
                "grasp_mode": config["execution"]["grasp_mode"],
                "execution_settings": config["execution"],
                "reason": f"{type(error).__name__}: {error}",
                "constraint_events": session._constraint_events,
                "constraint_active_at_end": session._grasp_constraint is not None,
            },
        )
        raise
    finally:
        session.close()


def main(argv: list[str] | None = None) -> int:
    """Replay an existing grasp result without regenerating meshes or grasp poses.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Zero only when the measured final assemble-object pose meets the replay tolerances.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--grasp-mode",
        choices=("contact", "fixed_constraint"),
        help="Override the saved config: contact-only grasp or a temporary physical joint.",
    )
    args = parser.parse_args(argv)
    result = read_json(args.result)
    if (
        result.get("schema") != "codex-grasp-assemble/v1"
        or result.get("success") is not True
    ):
        parser.error("Use a successful grasp result.json")
    if args.grasp_mode:
        result["config"].setdefault("execution", {})["grasp_mode"] = args.grasp_mode
    try:
        observed = _replay(result, args.headless)
        return 0 if observed["physical_success"] else 1
    finally:
        if "embodichain.lab.sim" in sys.modules:
            from embodichain.lab.sim import SimulationManager

            SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    raise SystemExit(main())
