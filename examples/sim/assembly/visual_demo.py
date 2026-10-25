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
"""Watch and record actual assembly physics using the precision probe runner.

The observer reads poses only. Screw translation and gear follower angles are
never prescribed. Recordings retain the real simulation timestamps; rendering
can slow their playback without changing the simulated trajectory.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
import xml.etree.ElementTree as ET

from embodichain.gen_sim.assembly import (
    AssemblyAsset,
    build_assembly,
    build_many_pegs,
    import_mjcf,
)
from embodichain.gen_sim.assembly.precision import (
    PrecisionRunCfg,
    _run_embodichain,
    _run_mujoco,
    _json_measurement,
)


def main(default_config: str = "usb_c") -> None:
    """Run the same mechanics as the benchmark with optional read-only Viser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--count", type=int, default=25, help="many_pegs grid size.")
    parser.add_argument(
        "--family", choices=["primitive", "convex"], default="primitive"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backend", choices=["newton", "default", "mujoco_cpu"], default="newton"
    )
    parser.add_argument("--route", choices=["direct", "usd"], default="direct")
    parser.add_argument("--duration", type=float, default=0.25)
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--clearance", type=float)
    parser.add_argument(
        "--max-error", type=float, help="many_pegs curved-outline chord error, m."
    )
    parser.add_argument("--contact-capacity", type=int, default=4096)
    parser.add_argument("--solver-iterations", type=int, default=100)
    parser.add_argument(
        "--peg-offset", type=float, default=0.0, help="Peg X offset, m."
    )
    parser.add_argument("--backend-contacts", action="store_true")
    parser.add_argument("--solver", choices=["pgs", "cg", "newton"])
    parser.add_argument("--use-mujoco-cpu", action="store_true")
    parser.add_argument("--solver-tolerance", type=float)
    parser.add_argument(
        "--multiccd", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--collision-colors", action="store_true")
    parser.add_argument("--viser", action="store_true")
    parser.add_argument("--native-window", action="store_true")
    parser.add_argument("--renderer", choices=["hybrid", "fast-rt"], default="hybrid")
    parser.add_argument(
        "--native-record",
        action="store_true",
        help="Record native RGB with a camera matching the DexSim window view.",
    )
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--startup-delay", type=float, default=0.0)
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    parser.add_argument("--playback-speed", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera-eye", nargs=3, type=float)
    parser.add_argument("--camera-target", nargs=3, type=float)
    parser.add_argument("--output", type=Path, default=Path("outputs/assembly_visual"))
    parser.add_argument("--record", choices=["gif", "mp4"])
    args = parser.parse_args()
    if min(args.duration, args.timestep, args.playback_speed, args.fps) <= 0:
        parser.error("Duration, timestep, playback speed and FPS must be positive.")
    if min(args.startup_delay, args.hold_seconds) < 0:
        parser.error("Startup delay and hold duration cannot be negative.")
    if args.viser and args.native_window:
        parser.error("Viser and the native window are separate viewer modes.")
    if args.native_record and not args.native_window:
        parser.error("--native-record requires --native-window.")
    if args.native_record and shutil.which("ffmpeg") is None:
        parser.error("--native-record requires FFmpeg.")
    if args.use_mujoco_cpu and args.backend != "newton":
        parser.error("--use-mujoco-cpu selects the Newton CPU integration only.")
    if args.backend == "default" and any(
        value is not None
        for value in (args.solver, args.solver_tolerance, args.multiccd)
    ):
        parser.error("Default does not accept MuJoCo solver or calibration overrides.")
    if args.solver_tolerance is not None and (
        not math.isfinite(args.solver_tolerance) or args.solver_tolerance < 0
    ):
        parser.error("Solver tolerance must be finite and non-negative.")
    if args.max_error is not None and args.config != "many_pegs":
        parser.error("--max-error applies to generated many_pegs grids only.")
    if min(args.contact_capacity, args.solver_iterations) <= 0:
        parser.error("Contact capacity and solver iterations must be positive.")
    if args.backend == "mujoco_cpu" and (
        args.route != "direct" or args.viser or args.native_window
    ):
        parser.error("CPU references use direct MJCF; use --record for visualization.")
    args.output.mkdir(parents=True, exist_ok=True)
    overrides = {} if args.clearance is None else {"clearance": args.clearance}
    asset = (
        AssemblyAsset.load(args.config)
        if args.config.endswith(".json")
        else (
            build_many_pegs(
                args.count,
                seed=args.seed,
                family=args.family,
                geometry_parameters=(
                    None if args.max_error is None else {"max_error": args.max_error}
                ),
                **overrides,
            )
            if args.config == "many_pegs"
            else build_assembly(args.config, seed=args.seed, **overrides)
        )
    )
    if args.config.endswith(".json") and overrides:
        parser.error("A saved manifest already fixes its clearance.")
    if args.peg_offset:
        if asset.config["kind"] != "peg":
            parser.error("--peg-offset applies to peg assets only.")
        # Recompile both representations from the same changed source initial
        # condition, retaining their original geometry, mass and contacts.
        xml = ET.fromstring(asset.source_mjcf)
        body = xml.find("./worldbody/body[@name='peg']")
        if body is None:
            raise ValueError("Source peg must attach directly to world.")
        position = next(b.position.copy() for b in asset.bodies if b.name == "peg")
        position[0] += args.peg_offset
        body.set("pos", " ".join(map(str, position)))
        asset = import_mjcf(
            ET.tostring(xml, encoding="unicode"),
            config={**asset.config, "demo_peg_offset_m": args.peg_offset},
            seed=asset.seed,
            generator_revision=asset.generator_revision,
        )
    if args.collision_colors:
        piece = 0
        for body in asset.bodies:
            for geom in body.geometries:
                geom.visible = geom.collision
                if geom.collision:
                    geom.rgba = [
                        *colorsys.hsv_to_rgb((piece * 0.61803398875) % 1, 0.65, 0.9),
                        1.0,
                    ]
                    piece += 1
    asset_path = asset.save(args.output / "asset.json").resolve()
    cfg = PrecisionRunCfg(
        backend=args.backend,
        route=args.route,
        duration=args.duration,
        timestep=args.timestep,
        source_contact=not args.backend_contacts,
        solver=args.solver,
        use_mujoco_cpu=args.use_mujoco_cpu,
        solver_tolerance=args.solver_tolerance,
        enable_multiccd=args.multiccd,
        contact_capacity=args.contact_capacity,
        solver_iterations=args.solver_iterations,
        work_dir=str(args.output),
    )
    selected_solver = (
        cfg.solver if cfg.solver is not None else asset.source_options.get("solver", 2)
    )
    solver_label = {
        0: "PGS",
        1: "CG",
        2: "Newton",
        "pgs": "PGS",
        "cg": "CG",
        "newton": "Newton",
    }.get(selected_solver, str(selected_solver))
    executor = (
        f"Newton / MuJoCo CPU {solver_label}"
        if args.backend == "newton" and cfg.use_mujoco_cpu
        else (
            f"Newton / Warp {solver_label}"
            if args.backend == "newton"
            else (
                f"MuJoCo CPU {solver_label}"
                if args.backend == "mujoco_cpu"
                else "Default"
            )
        )
    )
    steps = round(cfg.duration / cfg.timestep)
    if steps < 1:
        parser.error("Duration must include at least one timestep.")
    stride = max(1, round(args.playback_speed / args.fps / cfg.timestep))
    trace: dict[str, Any] = {
        "asset_manifest": asset_path.name,
        "asset_sha256": asset.content_hash,
        "kind": asset.config["kind"],
        "backend": args.backend,
        "route": args.route,
        "timestep": cfg.timestep,
        "run_config": cfg.to_dict(),
        "display_config": {
            "executor": executor,
            "solver": solver_label if args.backend != "default" else None,
            "fps": args.fps,
            "requested_playback_speed": args.playback_speed,
            "collision_colors": args.collision_colors,
            "native_window": args.native_window,
            "renderer": args.renderer if args.native_window else None,
            "capture_source": (
                "dexsim_native_camera_matching_window" if args.native_record else None
            ),
        },
        "sample_steps": [],
        "frames": [],
    }
    start = 0.0
    window = None
    camera = None
    native_capture_count = 0
    native_frames = args.output / "native_frames"
    if args.native_record:
        native_frames.mkdir(parents=True, exist_ok=True)

    def observe(host: Any, state: Any, step: int) -> None:
        nonlocal start, window, camera, native_capture_count
        if step not in {0, steps} and step % stride:
            return
        if args.backend == "mujoco_cpu":
            import mujoco

            mujoco.mj_kinematics(host, state)
            poses = {
                body.name: [
                    *state.xpos[host.body(body.name).id].tolist(),
                    *state.xquat[host.body(body.name).id][[1, 2, 3, 0]].tolist(),
                ]
                for body in asset.bodies
            }
        else:
            poses = {
                body.name: (
                    state[body.name].get_link_pose(body.name)
                    if body.joint.kind == "hinge"
                    else state[body.name].get_local_pose()
                )[0]
                .detach()
                .cpu()
                .tolist()
                for body in asset.bodies
            }
        trace["sample_steps"].append(step)
        trace["frames"].append(poses)
        if args.viser or args.native_window:
            if step == 0:
                if args.native_window:
                    import numpy as np

                    if not host.open_window():
                        raise RuntimeError("DexSim native window did not open.")
                    window = host.get_world().get_windows()
                    target = [0.0, 0.0, 0.02]
                    if asset.config["kind"] == "many_pegs":
                        positions = np.asarray([b.position for b in asset.bodies])
                        center = (positions.min(axis=0) + positions.max(axis=0)) / 2
                        span = max(float(np.ptp(positions[:, :2], axis=0).max()), 0.08)
                        target = [*center[:2].tolist(), 0.015]
                        eye = (
                            np.asarray(target) + span * np.array([0.78, -1.05, 1.55])
                        ).tolist()
                    elif asset.config["kind"] == "gear":
                        center = np.mean([b.position for b in asset.bodies], axis=0)
                        target = center.tolist()
                        eye = (center + np.array([0.025, -0.09, 0.12])).tolist()
                    else:
                        eye = (
                            [0.065, -0.09, 0.07]
                            if asset.config["kind"] == "peg"
                            else [0.038, -0.055, 0.045]
                        )
                    eye = args.camera_eye or eye
                    target = args.camera_target or target
                    window.set_look_at(
                        eye=np.asarray(eye, dtype=np.float32),
                        look_at=np.asarray(target, dtype=np.float32),
                        up=np.asarray([0, 0, 1], dtype=np.float32),
                    )
                    if args.native_record:
                        from embodichain.lab.sim.sensors import CameraCfg

                        camera = host.add_sensor(
                            sensor_cfg=CameraCfg(
                                uid="assembly_preview",
                                width=1280,
                                height=800,
                                near=0.01,
                                far=100.0,
                                # Native window: 28 mm lens, 24 mm sensor height.
                                intrinsics=(
                                    933.3333333333334,
                                    933.3333333333334,
                                    640.0,
                                    400.0,
                                ),
                                extrinsics=CameraCfg.ExtrinsicsCfg(
                                    eye=tuple(eye),
                                    target=tuple(target),
                                    up=(0.0, 0.0, 1.0),
                                ),
                            )
                        )
                        trace["display_config"]["camera"] = {
                            "eye": eye,
                            "target": target,
                            "intrinsics": [
                                933.3333333333334,
                                933.3333333333334,
                                640.0,
                                400.0,
                            ],
                            "near": 0.01,
                            "far": 100.0,
                        }
                    host.sync_render_state()
                    print("Live physics: DexSim native window", flush=True)
                else:
                    print(
                        f"Live physics: http://127.0.0.1:{args.viser_port}", flush=True
                    )
                    host.capture_visualization(force=True, capture_camera_images=False)
                time.sleep(args.startup_delay)
                start = time.perf_counter()
            if args.viser:
                host.capture_visualization(force=True, capture_camera_images=False)
            if window is not None:
                window.set_window_title(
                    f"mjfit {asset.config['kind']} | {executor} | "
                    f"simulation t={step * cfg.timestep:.3f}s"
                )
                if camera is not None:
                    import numpy as np
                    from PIL import Image

                    # Hybrid/FastRT's window PixelBuffer is unavailable in this
                    # SDK. This native camera uses the same scene and viewpoint;
                    # a synchronous sensor read never advances physics.
                    # Initialize render buffers at the unchanged initial pose.
                    for _ in range(4 if step == 0 else 1):
                        camera.update()
                    if step == 0:
                        trace["native_objects"] = {}
                        for name, obj in state.items():
                            entity = obj._entities[0]
                            source_body = next(
                                b for b in asset.bodies if b.name == name
                            )
                            hinged = source_body.joint.kind == "hinge"
                            body = (
                                entity.get_render_body(name)
                                if hinged
                                else entity.get_render_body()
                            )
                            if body is None:
                                raise RuntimeError(
                                    f"{name}: native visual body is missing."
                                )
                            node = body.render_node() if hinged else entity
                            visible = bool(node.is_visible())
                            if not visible:
                                # File loading can finish its GPU upload after
                                # Spawn binding. Publish visibility once ready,
                                # without advancing the physical initial state.
                                node.set_visible(True)
                                ready_deadline = time.monotonic() + 2.0
                                while not visible and time.monotonic() < ready_deadline:
                                    camera.update()
                                    time.sleep(0.05)
                                    visible = bool(node.is_visible())
                            if not visible and hasattr(body, "build"):
                                body.build(rebuild=True)
                                for _ in range(4):
                                    camera.update()
                                node.set_visible(True)
                                camera.update()
                                visible = bool(node.is_visible())
                                trace.setdefault("render_compatibility", []).append(
                                    {"body": name, "method": "native_build"}
                                )
                            trace["native_objects"][name] = {
                                "visible": visible,
                                "render_meshes": body.get_mesh_count(),
                                "collision_query_shapes": (
                                    None if hinged else len(obj.get_collision_shapes())
                                ),
                                "source_descriptor_collisions": len(
                                    entity.get_link_desc(name).collisions
                                    if hinged
                                    else entity.object_desc.collisions
                                ),
                                "configured_collision_parts": sum(
                                    g.collision for g in source_body.geometries
                                ),
                                "backend_collision_parts": (
                                    sum(
                                        len(link.shape_ids)
                                        for link in entity._physics_binding.links
                                        if link.name == name
                                    )
                                    if hinged
                                    and hasattr(entity._physics_binding, "links")
                                    else (
                                        len(entity._physics_binding.shape_ids)
                                        if hasattr(entity._physics_binding, "shape_ids")
                                        else None
                                    )
                                ),
                            }
                            actual = trace["native_objects"][name][
                                "backend_collision_parts"
                            ]
                            expected = trace["native_objects"][name][
                                "configured_collision_parts"
                            ]
                            if actual is not None and actual != expected:
                                raise RuntimeError(
                                    f"{name}: native collision count changed ({actual} != {expected})."
                                )
                            if not visible:
                                raise RuntimeError(
                                    f"{name}: native visual body is not GPU-ready."
                                )
                    picture = (
                        camera.get_data()["color"][0, :, :, :3].cpu().numpy().copy()
                    )
                    if not np.any(picture):
                        raise RuntimeError("Native camera returned an all-black frame.")
                    Image.fromarray(picture).save(
                        native_frames / f"{native_capture_count:06d}.png"
                    )
                    native_capture_count += 1
            if step == 0:
                # Asset upload and camera warm-up are outside playback time.
                start = time.perf_counter()
            remaining = (
                start + step * cfg.timestep / args.playback_speed - time.perf_counter()
            )
            if remaining > 0:
                time.sleep(remaining)
            if step == steps and args.hold_seconds:
                print(
                    "Physics finished; holding the final scene for inspection.",
                    flush=True,
                )
                deadline = time.perf_counter() + args.hold_seconds
                while time.perf_counter() < deadline:
                    time.sleep(min(0.1, max(0.0, deadline - time.perf_counter())))

    visualization = None
    if args.viser:
        from embodichain.lab.visualization import VisualizationCfg, ViserServerCfg

        visualization = VisualizationCfg(
            backend="viser",
            allow_commands=False,
            scene_fps=args.fps,
            viser_server=ViserServerCfg(
                port=args.viser_port, label="mjfit native physics"
            ),
        )
    try:
        result = (
            _run_mujoco(asset, cfg, on_step=observe)
            if args.backend == "mujoco_cpu"
            else _run_embodichain(
                asset,
                cfg,
                visualization=visualization,
                on_step=observe,
                native_window=args.native_window,
                renderer=args.renderer,
            )
        )
    except (ImportError, RuntimeError, NotImplementedError, ValueError) as error:
        result = {"status": "failed", "reason": str(error), "metrics": {}}
    result = _json_measurement(result)
    trace.update(result)
    trace["native_captured_frames"] = native_capture_count
    trace["timing_scope"] = (
        "Observed visual demo; timing includes pose reads and optional pacing."
    )
    trace_path = args.output / "trace.json"
    trace_path.write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"Recorded physical poses: {trace_path.resolve()}")
    if args.native_record and native_capture_count:
        movie = args.output / "native_window.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(args.fps),
                "-i",
                str(native_frames / "%06d.png"),
                "-c:v",
                "libx264",
                "-frames:v",
                str(native_capture_count),
                "-pix_fmt",
                "yuv420p",
                str(movie),
            ],
            check=True,
        )
        print(f"DexSim native RGB recording (same window viewpoint): {movie.resolve()}")
    if args.record and trace["frames"]:
        from _render_trace import render_trace

        print(
            render_trace(
                trace_path,
                args.output / f"demo.{args.record}",
                fps=args.fps,
                collision_colors=args.collision_colors,
            )
        )
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
