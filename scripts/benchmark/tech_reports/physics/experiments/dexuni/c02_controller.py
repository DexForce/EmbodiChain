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
"""Share the reference feedback planner across Python runtimes.

Only kinematic IK runs in this native 3.11 process. The requesting platform
owns every cloth and rigid-body physics step; IPC contains observed bag nodes
and returned joint/TCP targets, never commanded particle or payload poses.
An explicit trajectory JSON may adjust only the robot targets for Isaac trials.
"""

from __future__ import annotations
import argparse
import importlib.util
import json
from multiprocessing.connection import Client, Listener
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import numpy as np

__all__ = ["ReferenceController"]
REPO = Path("/home/oem/code/repos/EmbodiChain")
SDK = "/home/oem/code/repos/dexsimforembodychain/dexsim/modules/dexsim/build_Release/lib/python_package"
NEWTON = "/home/oem/miniconda3/envs/dexsim1/lib/python3.11/site-packages"


class _TrajectoryTargets:
    """Apply an explicit robot-only tip correction to reference TCP targets."""

    def __init__(self, base: Any, settings: dict[str, Any]) -> None:
        self.base = base
        self.settings = settings
        self.right_tip_grip: np.ndarray | None = None
        self.right_tip_pitch = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def command(
        self, time: float, positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        targets, openings, pitches = self.base.command(time, positions)
        cfg = self.settings
        sample_time = time + 1 / 60
        ramp = self.reference.smooth(
            (sample_time - cfg["blend_start_s"])
            / (cfg["blend_end_s"] - cfg["blend_start_s"])
        )
        release = self.reference.smooth(
            (sample_time - cfg["return_start_s"])
            / (cfg["return_end_s"] - cfg["return_start_s"])
        )
        targets = targets.copy()
        weight = ramp * (1 - release)
        targets[0] += weight * np.asarray(cfg["left_tip_translation_m"])
        pitches = pitches.copy()
        pitches[0] += weight * np.radians(cfg.get("left_tip_pitch_deg", 0.0))
        if "left_paper_opening_m" in cfg:
            openings = openings.copy()
            reference_open = self.reference.OPEN
            reference_closed = self.reference.SUPPORT_OPENING
            scale = (reference_open - cfg["left_paper_opening_m"]) / (
                reference_open - reference_closed
            )
            openings[0] = reference_open - (reference_open - openings[0]) * scale
        if cfg.get("slow_left_release", False) and 10.5 < sample_time <= 17:
            tip, _, _ = self.base.base._plan(10.5)
            start = tip[0]
            raised = start + np.array((0, 0, 0.08))
            outside = raised + np.array((-0.04, 0.13, 0))
            turned = outside + np.array((0.09, 0, -0.06))
            closed = cfg.get("left_paper_opening_m", self.reference.SUPPORT_OPENING)
            points = [
                (10.5, start, closed, np.radians(150)),
                (12.2, start, self.reference.OPEN, np.radians(150)),
                (13.2, raised, self.reference.OPEN, np.radians(105)),
                (14.2, outside, self.reference.OPEN, np.radians(105)),
                (17.0, turned, self.reference.OPEN, np.radians(95)),
            ]
            for a, b in zip(points[:-1], points[1:], strict=True):
                if sample_time <= b[0]:
                    blend = self.reference.smooth((sample_time - a[0]) / (b[0] - a[0]))
                    targets[0], openings[0], pitches[0] = tuple(
                        (1 - blend) * x + blend * y
                        for x, y in zip(a[1:], b[1:], strict=True)
                    )
                    break
        if cfg.get("assist_right_tip", False) and sample_time < 24:
            openings = openings.copy()
            targets[1], openings[1], pitches[1] = self._right_support(
                sample_time, targets[0], float(openings[0]), float(pitches[0])
            )
        return targets, openings, pitches

    def _right_support(
        self, time: float, left: np.ndarray, opening: float, pitch: float
    ) -> tuple[np.ndarray, float, float]:
        """Hold the opposite paper wall until the left rim regrasp completes."""
        demo = self.reference
        if time <= 10.5 + 1e-8:
            approach = demo.smooth((time - 1) / 2)
            point = left.copy()
            point[1] = 2 * demo.BAG_Y - point[1]
            mirrored_home = self.home[0] * np.array((1, -1, 1))
            point += (1 - approach) * (self.home[1] - mirrored_home)
            _, rotation = demo.bag_frame(demo.smooth((time - 5.5) / 4.5))
            offset = np.array(
                (self.settings.get("right_tip_rest_x_offset_m", 0.02), 0, 0)
            )
            point += approach * (rotation @ offset)
            self.right_tip_grip = point.copy()
            self.right_tip_pitch = pitch
            if time < 1:
                opening = demo.IDLE_OPENING + (
                    demo.OPEN - demo.IDLE_OPENING
                ) * demo.smooth(time)
            return point, opening, pitch
        assert self.right_tip_grip is not None
        grip = self.right_tip_grip
        closed = self.settings.get("left_paper_opening_m", demo.SUPPORT_OPENING)
        waypoints = [
            (10.5, grip, closed, self.right_tip_pitch),
            (20.5, grip, closed, self.right_tip_pitch),
            (21.3, grip, demo.OPEN, self.right_tip_pitch),
            (
                22.0,
                grip
                + np.array(
                    self.settings.get("right_release_retreat_m", [-0.06, 0, 0.06])
                ),
                demo.OPEN,
                self.right_tip_pitch,
            ),
            (
                22.8,
                grip
                + np.array(
                    self.settings.get("right_release_retreat_m", [-0.06, 0, 0.06])
                ),
                demo.OPEN,
                demo.IDLE_PITCH,
            ),
            (23.8, self.home[1], demo.OPEN, demo.IDLE_PITCH),
            (24.0, self.home[1], demo.IDLE_OPENING, demo.IDLE_PITCH),
        ]
        for a, b in zip(waypoints[:-1], waypoints[1:], strict=True):
            if time <= b[0]:
                blend = demo.smooth((time - a[0]) / (b[0] - a[0]))
                return tuple(
                    (1 - blend) * x + blend * y
                    for x, y in zip(a[1:], b[1:], strict=True)
                )
        return waypoints[-1][1:]


def _module(name: str, path: Path) -> Any:
    """Load one frozen reference module without executing its CLI."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def serve(output: Path, address: str) -> None:
    """Solve the exact reference feedback targets in its original runtime."""
    sys.path.insert(0, str(output / "source/reference"))
    demo = _module(
        "c02_original_bag_demo",
        output / "source/reference/demo_dexuni_w1_bag_packing.py",
    )
    lift = _module("c02_accepted_lift", output / "source/dexsim_c02_reference.py")
    with np.load(output / "inputs/bag.npz") as mesh:
        ctrl = demo.W1BagController(
            output / "inputs/w1_v030/robot.urdf",
            mesh["vertices"],
            int(mesh["paper_count"]),
            mesh["bottom"],
            device="cuda:0",
        )
        ctrl.plan = lift.LiftTargets(ctrl.plan, demo, mesh["rim"])
    settings = output / "inputs/controller_trajectory.json"
    if settings.exists():
        ctrl.plan = _TrajectoryTargets(ctrl.plan, json.loads(settings.read_text()))
    with Listener(address, family="AF_UNIX", authkey=b"c02-reference") as listener:
        with listener.accept() as conn:
            conn.send(
                dict(
                    joint_names=ctrl.joint_names,
                    initial=ctrl.initial_joint_positions,
                    targets=ctrl.targets,
                )
            )
            while True:
                request = conn.recv()
                if request is None:
                    conn.send(ctrl.plan.waypoints)
                    break
                t, positions = request
                q = ctrl.command(t, positions)
                conn.send((q, ctrl.targets.copy()))


class ReferenceController:
    """Use the original kinematic controller while keeping physics local."""

    def __init__(self, output: Path) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="c02_ctrl_"))
        address = str(self.directory / "socket")
        self.log = (output / "analysis/controller.log").open("w")
        env = os.environ.copy()
        env["PYTHONPATH"] = SDK + ":" + NEWTON
        self.process = subprocess.Popen(
            [
                str(REPO / ".venv/bin/python"),
                str(output / "source/c02_controller.py"),
                "--serve",
                str(output),
                "--address",
                address,
            ],
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        for _ in range(600):
            if self.process.poll() is not None:
                raise RuntimeError(
                    "Reference IK process failed; see analysis/controller.log"
                )
            if Path(address).exists():
                break
            time.sleep(0.1)
        else:
            self.process.terminate()
            raise TimeoutError("Reference IK startup")
        self.connection = Client(address, family="AF_UNIX", authkey=b"c02-reference")
        meta = self.connection.recv()
        self.joint_names = meta["joint_names"]
        self.initial_joint_positions = meta["initial"]
        self.targets = meta["targets"]
        self.current_command = self.initial_joint_positions.copy()

    def command(self, t: float, positions: np.ndarray) -> np.ndarray:
        """Return joint targets derived from this platform's observed bag."""
        self.connection.send((t, positions))
        self.current_command, self.targets = self.connection.recv()
        return self.current_command.copy()

    def close(self) -> None:
        """Save the realized lift targets and release the IK-only subprocess."""
        try:
            self.connection.send(None)
            self.waypoints = self.connection.recv()
            self.connection.close()
            self.process.wait(timeout=15)
        finally:
            if self.process.poll() is None:
                self.process.terminate()
            self.log.close()
            shutil.rmtree(self.directory, ignore_errors=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--serve", type=Path, required=True)
    p.add_argument("--address", required=True)
    a = p.parse_args()
    serve(a.serve, a.address)
