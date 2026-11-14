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

"""Shared simulation and Viser orchestration for standalone previews."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from embodichain.utils.logger import log_info, log_warning

if TYPE_CHECKING:
    from embodichain.lab.scripts.preview_joint_control import (
        ArticulationPreviewController,
    )
    from embodichain.lab.sim.objects import Articulation, RigidObject
    from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg

__all__: list[str] = []


def _build_sim_cfg(
    args: argparse.Namespace, *, headless: bool | None = None
) -> SimulationManagerCfg:
    from embodichain.lab.sim.cfg import RenderCfg, physics_cfg_for_backend
    from embodichain.lab.sim.sim_manager import SimulationManagerCfg
    from embodichain.lab.visualization import visualization_cfg_from_args

    return SimulationManagerCfg(
        headless=args.headless if headless is None else headless,
        device=args.device,
        render_cfg=RenderCfg(renderer=args.renderer),
        physics_cfg=physics_cfg_for_backend(args.physics),
        gpu_id=args.gpu_id,
        num_envs=args.num_envs,
        arena_space=args.arena_space,
        visualization=visualization_cfg_from_args(args),
    )


def _setup_viser_joint_control(
    sim: SimulationManager,
    assets: Sequence[RigidObject | Articulation],
    args: argparse.Namespace,
    *,
    follow_external_targets: bool = False,
    hold_pose: bool = True,
) -> ArticulationPreviewController | None:
    if not getattr(args, "viser", False) or not getattr(args, "joint_control", True):
        return None

    from embodichain.lab.scripts.preview_joint_control import (
        ArticulationPreviewController,
    )
    from embodichain.lab.sim.objects import Articulation

    articulations = [asset for asset in assets if isinstance(asset, Articulation)]
    if not articulations:
        return None
    runtime = sim.visualization_runtime
    if runtime is None:
        log_warning("Viser is enabled but its visualization runtime is unavailable.")
        return None

    options = {"follow_external_targets": True} if follow_external_targets else {}
    if not hold_pose:
        options["hold_pose"] = False
    controller = ArticulationPreviewController(articulations, runtime, **options)
    if not controller.has_controls:
        log_warning("No supported independent articulation joints were found.")
        return None
    controller.update()
    runtime.set_joint_control_provider(controller)
    log_info("Viser articulation joint controls enabled.", color="green")
    return controller


def _publish_loaded_assets(sim: SimulationManager, args: argparse.Namespace) -> None:
    if getattr(args, "viser", False):
        sim.capture_visualization_safely(force=True)


def _step_preview_simulation(
    sim: SimulationManager,
    joint_controller: ArticulationPreviewController | None,
    *,
    step: int = 1,
) -> None:
    for _ in range(step):
        if joint_controller is not None:
            joint_controller.update()
        sim.update(step=1)


def _run_preview_loop(
    sim: SimulationManager,
    joint_controller: ArticulationPreviewController | None,
    *,
    interactive: bool,
    max_steps: int | None = None,
    advance_physics: bool | Callable[[], bool] = True,
    on_frame: Callable[[SimulationManager], bool] | None = None,
    after_step: Callable[[], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    hold_during_simulation: bool = True,
    frame_interval: float = 0.0,
) -> None:
    if not interactive and max_steps is None:
        return
    completed = 0
    try:
        while max_steps is None or completed < max_steps:
            changed = on_frame(sim) if on_frame is not None else False
            if should_stop is not None and should_stop():
                break
            advancing = (
                advance_physics() if callable(advance_physics) else advance_physics
            )
            if advancing:
                if hold_during_simulation:
                    _step_preview_simulation(sim, joint_controller)
                else:
                    sim.update(step=1)
                if after_step is not None:
                    after_step()
            else:
                if joint_controller is not None:
                    changed = bool(joint_controller.update()) or changed
                with sim.render_frame(
                    force_visualization=changed, capture_camera_images=False
                ):
                    pass
            completed += 1
            if frame_interval:
                time.sleep(frame_interval)
    except KeyboardInterrupt:
        pass
