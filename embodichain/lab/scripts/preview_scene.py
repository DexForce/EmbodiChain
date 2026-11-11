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

"""Preview ordinary USD scenes through the Lab import and display owners."""

from __future__ import annotations

import argparse
from copy import deepcopy
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from embodichain.cli.sim import add_sim_args_to_parser

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

__all__ = ["preview_scene"]


class _JointControls:
    def __init__(
        self, sim: SimulationManager, assets: Mapping[str, object], enabled: bool
    ) -> None:
        self.sim, self.assets, self.enabled = sim, assets, enabled
        self.running: bool | None = None
        self.controller: object | None = None

    def refresh(self, running: bool) -> bool:
        runtime = self.sim.visualization_runtime
        if runtime is None or self.running == running:
            return False
        from embodichain.lab.scripts._preview import _setup_viser_joint_control

        runtime.drain_joint_control_commands()
        self.running = running
        self.controller = None
        runtime.set_joint_control_provider(None)
        if self.enabled and not running:
            self.controller = _setup_viser_joint_control(
                self.sim,
                list(self.assets.values()),
                argparse.Namespace(viser=True, joint_control=True),
                hold_pose=False,
            )
        self.sim.notify_visualization_topology_changed()
        self.sim.capture_visualization_safely(force=True, capture_camera_images=False)
        return True

    def update(self) -> int:
        return 0 if self.controller is None else self.controller.update()


def preview_scene(
    usd_file: str | Path,
    *,
    sim_cfg: SimulationManagerCfg | None = None,
    simulate: bool = False,
    steps: int | None = None,
    joint_control: bool = True,
    report: str | Path | None = None,
    screenshot: str | Path | None = None,
    resolution: tuple[int, int] = (1280, 720),
    on_frame: Callable[[SimulationManager], bool] | None = None,
) -> dict[str, Any]:
    """Load a standard USD scene and own its preview lifecycle.

    Producer-specific identity and joint-state metadata are opaque. The native
    import owner resolves authored units, physical types and initial state.
    Default preview publishes that state without advancing physics. Browser
    controls and native Space/N/Home/Escape commands run on the simulation thread.

    Args:
        usd_file: USD, USDA, USDC or relocatable USDZ entry file.
        sim_cfg: Existing simulation/renderer/visualization configuration.
        simulate: Advance physics immediately rather than starting paused.
        steps: Optional finite physics-step limit; requires simulate=True.
        joint_control: Enable paused Viser joint-position editing.
        report: Optional JSON inspection/diagnostic output, also written on failure.
        screenshot: Optional final offscreen PNG, without a warmup physics step.
        resolution: Screenshot width and height.
        on_frame: Optional caller-owned playback callback before display/stepping.

    Returns:
        Portable inspection, diagnostics, step counts and output paths. Native
        objects have been destroyed before this result is returned.

    Raises:
        ValueError: Inputs or requested output modes are invalid.
        RuntimeError: Import, preparation, display or rendering fails.
    """
    source = Path(usd_file).expanduser().resolve()
    report_path = None if report is None else Path(report).expanduser().resolve()
    image_path = None if screenshot is None else Path(screenshot).expanduser().resolve()
    result: dict[str, Any] = {
        "format": "embodichain.scene-preview/v1",
        "source": str(source),
        "status": "loading",
        "stage": "input",
        "physics_steps": 0,
        "resets": 0,
        "entities": [],
        "diagnostics": [],
        "outputs": {},
    }
    sim = imported = proxy = handler = runtime = None
    try:
        if source.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
            raise ValueError(
                "Scene preview requires a .usd, .usda, .usdc or .usdz file."
            )
        if not source.is_file():
            raise FileNotFoundError(f"USD scene not found: {source}")
        if steps is not None and (type(steps) is not int or steps <= 0 or not simulate):
            raise ValueError("steps must be positive and requires simulate=True.")
        if len(resolution) != 2 or any(
            type(v) is not int or v <= 0 for v in resolution
        ):
            raise ValueError(
                "Screenshot resolution must contain two positive integers."
            )
        if image_path is not None and image_path.suffix.lower() != ".png":
            raise ValueError("Screenshot output must use .png.")
        if source in (report_path, image_path) or (
            report_path is not None and report_path == image_path
        ):
            raise ValueError(
                "Preview outputs must be distinct from the input and each other."
            )
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import UsdImportCfg, UsdSourceCfg
        from embodichain.lab.scripts._preview import _run_preview_loop
        from embodichain.lab.scripts._scene_preview import (
            PreviewControls,
            ReloadPreview,
            source_digest,
            describe_import,
            camera_view,
            screenshot as capture_image,
            install_native_controls,
        )

        cfg = (
            SimulationManagerCfg(default_ground=False)
            if sim_cfg is None
            else deepcopy(sim_cfg)
        )
        cfg.default_ground = False
        # Scene content (including cameras) is discovered after constructing
        # the manager. Reserve rendering for auto rather than upgrading an
        # already-created no-render World later; explicit no-render is honored.
        if cfg.render_cfg.renderer == "auto":
            cfg.render_cfg.renderer = "hybrid"
        is_viser = cfg.visualization.backend == "viser"
        interactive = is_viser or not cfg.headless
        if not interactive and simulate and steps is None:
            raise ValueError("Headless simulation requires a finite steps limit.")
        initial_digest = source_digest(source)
        result["source_sha256"] = initial_digest
        result["frontend"] = (
            "viser" if is_viser else "window" if interactive else "headless"
        )
        result["simulate"] = simulate
        result["default_ground"] = False
        running = simulate
        while True:
            try:
                if source_digest(source) != initial_digest:
                    raise ValueError(
                        "Source changed during preview; start a new preview for the new version."
                    )
                result["stage"] = "create"
                sim = SimulationManager(cfg)
                result["stage"] = "load"
                imported = sim.add_usd(
                    UsdImportCfg(
                        name="preview", source=UsdSourceCfg(file_path=str(source))
                    )
                )
                result["stage"] = "prepare"
                sim.prepare()
                sim.sync_render_state()
                result["entities"] = describe_import(imported)
                result["physics_backend"] = sim.physics.name
                runtime = sim.visualization_runtime
                if runtime is not None:
                    print(f"Scene preview: {runtime.endpoint}")
                    result["endpoint"] = runtime.endpoint
                controls = PreviewControls(running)
                proxy = _JointControls(sim, imported, joint_control)
                eye = target = None
                if image_path is not None or interactive:
                    result["stage"] = "camera"
                    eye, target = camera_view(sim)
                    result["camera"] = {
                        "eye": eye,
                        "target": target,
                        "up": [0.0, 0.0, 1.0],
                        "resolution": list(resolution),
                    }
                if runtime is not None:
                    runtime.publish_preview_camera(eye=tuple(eye), target=tuple(target))
                if interactive and not is_viser:
                    if not sim.open_window():
                        raise RuntimeError(
                            "Could not open the native scene-preview window."
                        )
                    sim.get_world().get_windows().set_look_at(eye, target, [0, 0, 1])
                    handler = install_native_controls(sim, controls)
                    print("Space: play/pause; N: step; Home: reload source; Esc: stop.")

                def before_frame(owner: SimulationManager) -> bool:
                    changed = bool(on_frame(owner)) if on_frame is not None else False
                    controls.update(runtime)
                    changed = (
                        proxy.refresh(controls.running or controls.pending_steps > 0)
                        or changed
                    )
                    return changed

                def committed_step() -> None:
                    controls.after_step()
                    result["physics_steps"] += 1

                result["stage"] = "preview"
                _run_preview_loop(
                    sim,
                    proxy,
                    interactive=interactive,
                    max_steps=steps if not interactive else None,
                    advance_physics=controls.advance,
                    on_frame=before_frame,
                    after_step=committed_step,
                    should_stop=lambda: controls.stopped
                    or (steps is not None and result["physics_steps"] >= steps),
                    hold_during_simulation=False,
                    frame_interval=0.01 if interactive else 0.0,
                )
                result["simulation_time"] = float(sim.simulation_time)
                if image_path is not None:
                    result["stage"] = "screenshot"
                    capture_image(sim, image_path, eye, target, resolution)
                    result["outputs"]["screenshot"] = str(image_path)
                result["status"], result["stage"] = "loaded", "complete"
                break
            except ReloadPreview:
                result["resets"] += 1
                running = False
            finally:
                if sim is not None:
                    sim.destroy(exit_process=False)
                    imported = proxy = handler = runtime = None
                    SimulationManager.flush_cleanup_queue()
                    sim = None
        return result
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        raise
    except Exception as error:
        result["status"] = "error"
        result["diagnostics"].append(
            {
                "stage": result["stage"],
                "code": type(error).__name__,
                "message": str(error),
            }
        )
        raise
    finally:
        if report_path is not None and report_path not in (source, image_path):
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = report_path.with_name(report_path.name + ".in_progress")
            try:
                temporary.write_text(
                    json.dumps(result, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(report_path)
            finally:
                temporary.unlink(missing_ok=True)


def main(
    argv: Sequence[str] | None = None, *, prog: str = "embodichain preview-scene"
) -> None:
    """Parse the lightweight USD preview CLI before importing simulation dependencies."""
    parser = argparse.ArgumentParser(
        prog=prog,
        allow_abbrev=False,
        description="Preview a standard USD scene; starts paused without physics steps.",
    )
    parser.add_argument("--usd-file", type=Path, required=True)
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Advance physics; paused joint controls are disabled while running.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Finite physics steps; requires --simulate and is mandatory for headless simulation.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write portable scene inspection and stage diagnostics to JSON.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Render a final offscreen PNG without advancing physics for warmup.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=(1280, 720),
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument(
        "--joint-control", action=argparse.BooleanOptionalAction, default=True
    )
    add_sim_args_to_parser(parser)
    args = parser.parse_args(argv)
    from embodichain.lab.scripts._preview import _build_sim_cfg

    cfg = _build_sim_cfg(args, headless=args.headless or args.viser)
    preview_scene(
        args.usd_file,
        sim_cfg=cfg,
        simulate=args.simulate,
        steps=args.steps,
        report=args.report,
        screenshot=args.screenshot,
        resolution=tuple(args.resolution),
        joint_control=args.joint_control,
    )


if __name__ == "__main__":
    main()
