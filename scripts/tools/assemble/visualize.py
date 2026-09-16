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

"""Generate in a subprocess, then replace the scene; Enter starts a fresh job."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
from pathlib import Path
import select
import sys
import time
from typing import TYPE_CHECKING

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.tools.assemble._protocol import load_config
from scripts.tools.assemble._generation_process import run_generation
from scripts.tools.mug_rack_pose._json_io import pose_matrix

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager

__all__ = ["build_parser", "run_cycles", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build a native-viewer CLI without initializing the simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--steps", type=int, default=30, help="sim.update steps after each scene load."
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Stop after N cycles; 0 is unlimited. Default: unlimited with a window, 1 headless.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Record preview.mp4 per generation without a native window.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--renderer", choices=("auto", "hybrid", "fast-rt", "rt"), default="auto"
    )
    parser.add_argument("--record-fps", type=int, default=20)
    parser.add_argument(
        "--generation-timeout",
        type=float,
        default=1800,
        help="Maximum seconds for a complete generation subprocess.",
    )
    return parser


def run_cycles(
    generate: Callable[[int], dict],
    show: Callable[[dict], None],
    wait_for_enter: Callable[[], str],
    limit: int = 0,
) -> int:
    """Run the full pipeline again only after an explicit Enter.

    Args:
        generate: Fresh config loading and complete model generation per cycle.
        show: Replace the scene only after successful complete generation.
        wait_for_enter: Terminal input including newline; empty string means EOF.
        limit: Number of cycles, or zero for unlimited.

    Returns:
        Number of attempted complete generation cycles.
    """
    cycle = 0
    while True:
        cycle += 1
        result = generate(cycle)
        if result["success"]:
            show(result)
        else:
            print(
                f"[assemble] Generation failed; current scene retained: {result['reason']}",
                flush=True,
            )
        if limit and cycle >= limit:
            return cycle
        print(
            "[assemble] Press Enter to generate a replacement; the current scene stays visible until it is ready. q + Enter or Ctrl+C to exit.",
            flush=True,
        )
        while True:
            text = wait_for_enter()
            if text == "" or text.strip().lower() in ("q", "quit", "exit"):
                return cycle
            if not text.strip():
                break
            print(
                "[assemble] Use Enter to regenerate or q + Enter to quit.", flush=True
            )


class _Scene:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.sim: SimulationManager | None = None
        self.loaded = False

    def show(self, result: dict) -> None:
        import numpy as np
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import LightCfg, RenderCfg, physics_cfg_for_backend
        from embodichain.lab.sim.objects import RigidObjectCfg
        from embodichain.lab.sim.shapes import MeshCfg, MeshCollisionCfg
        import trimesh

        if (
            result.get("schema") != "codex-assemble/v1"
            or result.get("success") is not True
            or result.get("validation", {}).get("accepted") is not True
        ):
            raise ValueError(
                "Preview requires an independently validated assembly result"
            )
        pose = pose_matrix(result["T_base_assemble"])
        cfgs, bounds = [], []
        for role in ("base", "assemble"):
            asset = result["assets"][role]
            path = Path(asset["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
                raise ValueError(f"{role} mesh changed since pose validation")
            transform = np.eye(4) if role == "base" else pose
            mesh = trimesh.load(path, force="mesh", process=False)
            mesh.apply_transform(transform)
            bounds.append(mesh.bounds)
            cfgs.append(
                RigidObjectCfg(
                    uid=role,
                    shape=MeshCfg(
                        fpath=str(path),
                        collision=MeshCollisionCfg(approximation="triangle_mesh"),
                    ),
                    body_type="static",
                    init_local_pose=transform,
                )
            )
        if self.sim is None:
            self.sim = SimulationManager(
                SimulationManagerCfg(
                    width=960,
                    height=720,
                    headless=True,
                    physics_dt=0.01,
                    device=self.args.device,
                    physics_cfg=physics_cfg_for_backend("default"),
                    render_cfg=RenderCfg(renderer=self.args.renderer),
                    num_envs=1,
                    arena_space=2.0,
                )
            )
            self.sim.add_light(
                cfg=LightCfg(
                    uid="assembly_light",
                    light_type="direction",
                    direction=(-0.4, 0.6, -1.0),
                    intensity=3.0,
                )
            )
        if self.loaded:
            # Stop the render thread before touching native scene topology. All
            # source files and the new pose have already passed host validation.
            if self.sim.is_window_opened:
                self.sim.close_window()
            self.sim.replace_rigid_objects(cfgs)
            self.sim.reset_objects_state()
            print(
                "[assemble] New generation ready; both bodies replaced and scene state reset.",
                flush=True,
            )
        else:
            for cfg in cfgs:
                self.sim.add_rigid_object(cfg=cfg)
            self.sim.prepare()
        self.loaded = True
        boxes = np.asarray(bounds)
        low, high = boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)
        center = (low + high) / 2
        size = np.linalg.norm(high - low)
        camera = (
            tuple(center + size * np.array([0.7, -1.1, 0.55])),
            tuple(center),
            (0, 0, 1),
        )
        if not self.args.headless:
            if not self.sim.is_window_opened and not self.sim.open_window():
                raise RuntimeError("Could not open the native simulation window")
            self.sim.get_world().get_windows().set_look_at(
                eye=camera[0], look_at=camera[1], up=camera[2]
            )
        else:
            destination = str(Path(result["run_directory"]) / "preview.mp4")
            if not self.sim.start_window_record(
                save_path=destination,
                fps=self.args.record_fps,
                max_memory=512,
                look_at=camera,
            ):
                raise RuntimeError("Could not start headless recording")
        try:
            for _ in range(self.args.steps):
                self.sim.update(step=1)
                if not self.args.headless:
                    time.sleep(0.01)
        finally:
            if self.sim.is_window_recording():
                self.sim.stop_window_record()
                self.sim.wait_window_record_saves()
        print(
            f"[assemble] Loaded base + assemble and completed {self.args.steps} sim.update steps.\nResult: {result['run_directory']}/result.json",
            flush=True,
        )

    def pump(self) -> None:
        if self.sim is not None and self.loaded and not self.args.headless:
            self.sim.update(step=1)

    def wait_for_enter(self) -> str:
        while not select.select([sys.stdin], [], [], 0.05)[0]:
            self.pump()
        return sys.stdin.readline()

    def close(self) -> None:
        if self.sim is not None:
            self.sim.destroy(exit_process=False)
            self.sim = None


def _run(args: argparse.Namespace) -> bool:
    scene = _Scene(args)
    last_success = False

    def generate_cycle(cycle: int) -> dict:
        nonlocal last_success
        try:
            result = run_generation(
                args.config, cycle, args.generation_timeout, scene.pump
            )
        except Exception as error:
            # Keep the preview when a user edits the configuration incorrectly.
            result = {"success": False, "reason": f"{type(error).__name__}: {error}"}
        last_success = result["success"]
        return result

    try:
        run_cycles(
            generate_cycle,
            scene.show,
            scene.wait_for_enter,
            args.cycles,
        )
        return last_success
    finally:
        scene.close()


def main(argv: list[str] | None = None) -> int:
    """Run complete generation and native visualization, with terminal regeneration.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Zero after a successful final cycle, one for failure, or 130 on interrupt.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cycles is None:
        args.cycles = 1 if args.headless else 0
    import math

    if (
        args.steps < 1
        or args.cycles < 0
        or args.record_fps < 1
        or not math.isfinite(args.generation_timeout)
        or args.generation_timeout <= 0
    ):
        parser.error(
            "steps, record-fps and generation-timeout must be positive; cycles must be nonnegative"
        )
    # Validate before creating GPU resources or contacting Codex.
    load_config(args.config)
    try:
        return 0 if _run(args) else 1
    except KeyboardInterrupt:
        print("[assemble] Stopped.", flush=True)
        return 130
    finally:
        # _run and all facade locals must unwind before native parent cleanup.
        if "embodichain.lab.sim" in sys.modules:
            from embodichain.lab.sim import SimulationManager

            SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    raise SystemExit(main())
