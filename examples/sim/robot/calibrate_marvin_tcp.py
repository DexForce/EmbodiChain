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

"""Interactively visualize, adjust, and save Marvin TCP transforms."""

from __future__ import annotations

import argparse
import json
import select
import shlex
import sys
import time
from pathlib import Path

import numpy as np

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg
from embodichain.lab.sim.robots import MarvinCfg
from embodichain.lab.sim.utility.tcp_calibration import (
    TCPMarkerCalibrator,
    save_solver_tcp_overrides,
    solver_tcp_overrides,
)

_SIDES = {
    "left": ("left_arm", "left_ee"),
    "right": ("right_arm", "right_ee"),
}

_HELP = """
Commands:
  select left|right        Select the arm to edit.
  t x|y|z DISTANCE         Translate in the EE frame, in meters.
  r x|y|z ANGLE            Rotate about an EE-frame axis, in degrees.
  set x y z                Set TCP translation in the EE frame, in meters.
  reset [left|right|all]   Restore the initial TCP value.
  show [left|right|all]    Print matrices ready for solver_cfg.tcp.
  save [PATH]              Save both TCP matrices as a JSON config fragment.
  help                     Show this help.
  quit                     Save nothing and exit.

Examples:
  t z 0.001
  r x 1
  set 0 0 0.14
  save outputs/marvin_tcp.json
""".strip()


def _print_tcp(calibrators: dict[str, TCPMarkerCalibrator], side: str) -> None:
    """Print one or both calibrated TCP matrices and their config fragment."""
    selected = calibrators if side == "all" else {side: calibrators[side]}
    matrices = {
        calibrator.control_part: calibrator.tcp_transform
        for calibrator in selected.values()
    }
    print(json.dumps(solver_tcp_overrides(matrices), indent=4))


def _handle_command(
    line: str,
    calibrators: dict[str, TCPMarkerCalibrator],
    selected_side: str,
    default_output: Path,
) -> tuple[str, bool]:
    """Apply one terminal command and return ``(selected_side, should_exit)``."""
    tokens = shlex.split(line)
    if not tokens:
        return selected_side, False
    command = tokens[0].lower()

    if command in {"quit", "exit", "q"}:
        return selected_side, True
    if command in {"help", "h", "?"}:
        print(_HELP)
        return selected_side, False
    if command == "select":
        target = tokens[1].lower() if len(tokens) == 2 else ""
        if target not in calibrators:
            raise ValueError("Usage: select left|right")
        selected_side = target
        print(f"Selected {selected_side} arm.")
        return selected_side, False
    if command in {"t", "translate"}:
        if len(tokens) != 3:
            raise ValueError("Usage: t x|y|z DISTANCE")
        calibrators[selected_side].translate(tokens[1], float(tokens[2]))
        _print_tcp(calibrators, selected_side)
        return selected_side, False
    if command in {"r", "rotate"}:
        if len(tokens) != 3:
            raise ValueError("Usage: r x|y|z ANGLE_DEGREES")
        calibrators[selected_side].rotate(tokens[1], float(tokens[2]))
        _print_tcp(calibrators, selected_side)
        return selected_side, False
    if command == "set":
        if len(tokens) != 4:
            raise ValueError("Usage: set X Y Z")
        calibrators[selected_side].set_translation([float(v) for v in tokens[1:]])
        _print_tcp(calibrators, selected_side)
        return selected_side, False
    if command == "reset":
        if len(tokens) > 2:
            raise ValueError("Usage: reset [left|right|all]")
        target = tokens[1].lower() if len(tokens) == 2 else selected_side
        if target == "all":
            for calibrator in calibrators.values():
                calibrator.reset()
        elif target in calibrators:
            calibrators[target].reset()
        else:
            raise ValueError("Usage: reset [left|right|all]")
        _print_tcp(calibrators, target)
        return selected_side, False
    if command == "show":
        if len(tokens) > 2:
            raise ValueError("Usage: show [left|right|all]")
        target = tokens[1].lower() if len(tokens) == 2 else selected_side
        if target not in {*calibrators, "all"}:
            raise ValueError("Usage: show [left|right|all]")
        _print_tcp(calibrators, target)
        return selected_side, False
    if command == "save":
        output = Path(tokens[1]) if len(tokens) == 2 else default_output
        if len(tokens) > 2:
            raise ValueError("Usage: save [PATH]")
        saved_path = save_solver_tcp_overrides(
            output,
            {
                calibrator.control_part: calibrator.tcp_transform
                for calibrator in calibrators.values()
            },
        )
        print(f"Saved calibrated TCP config to {saved_path.resolve()}.")
        return selected_side, False

    raise ValueError(
        f"Unknown command {command!r}. Enter 'help' for available commands."
    )


def _run_console(
    sim: SimulationManager,
    calibrators: dict[str, TCPMarkerCalibrator],
    output: Path,
) -> None:
    """Refresh markers while processing non-blocking terminal commands."""
    selected_side = "left"
    print(_HELP)
    print("\nSelected left arm. tcp> ", end="", flush=True)
    while True:
        for calibrator in calibrators.values():
            calibrator.update()
        sim.capture_visualization_safely()

        readable, _, _ = select.select([sys.stdin], [], [], 0.03)
        if not readable:
            continue
        line = sys.stdin.readline()
        if not line:
            break
        try:
            selected_side, should_exit = _handle_command(
                line, calibrators, selected_side, output
            )
            if should_exit:
                break
        except (KeyError, ValueError) as error:
            print(f"Error: {error}")
        print(f"\nSelected {selected_side} arm. tcp> ", end="", flush=True)


def main() -> None:
    """Launch the Marvin TCP calibration scene."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Simulation device (default: cpu).",
    )
    parser.add_argument(
        "--renderer",
        choices=("auto", "hybrid", "fast-rt", "rt"),
        default="auto",
        help="DexSim renderer (default: auto).",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Physical GPU used by DexSim rendering (default: 0).",
    )
    parser.add_argument(
        "--urdf-path",
        default=None,
        help="Marvin robot_with_ee.urdf path; defaults to the data-root asset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/marvin_tcp.json"),
        help="Default JSON path used by the save command.",
    )
    args = parser.parse_args()

    sim = SimulationManager(
        SimulationManagerCfg(
            num_envs=1,
            sim_device=args.device,
            gpu_id=args.gpu_id,
            headless=False,
            render_cfg=RenderCfg(renderer=args.renderer),
            width=1280,
            height=960,
        )
    )
    sim.set_manual_update(False)
    calibrators: dict[str, TCPMarkerCalibrator] = {}
    try:
        robot_overrides: dict[str, object] = {"uid": "marvin_tcp_calibration"}
        if args.urdf_path:
            robot_overrides["urdf_path"] = args.urdf_path
        cfg = MarvinCfg.from_dict(robot_overrides)
        robot = sim.add_robot(cfg=cfg)
        if robot is None:
            raise RuntimeError("Failed to add Marvin to the simulation.")
        if not sim.open_window():
            raise RuntimeError("Failed to open the DexSim visualization window.")

        for side, (control_part, end_link_name) in _SIDES.items():
            calibrator = TCPMarkerCalibrator(
                sim,
                robot,
                control_part=control_part,
                end_link_name=end_link_name,
                tcp_transform=np.asarray(cfg.solver_cfg[control_part].tcp),
                marker_prefix=f"marvin_{side}",
            )
            calibrator.draw()
            calibrators[side] = calibrator

        time.sleep(0.2)
        _run_console(sim, calibrators, args.output)
    except KeyboardInterrupt:
        pass
    finally:
        for calibrator in calibrators.values():
            calibrator.close()
        sim.destroy()


if __name__ == "__main__":
    main()
