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
"""Four local WorldParticle checkpoints, stepped by DexSim through one Provider."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from ._demo_assets import PROFILES, add_boundary_visuals
from embodichain_models.dynamics.worldparticle.data import load_sample


def _demo(default_scene="fancy_fluid") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=tuple(PROFILES), default=default_scene)
    parser.add_argument(
        "--assets", type=Path, default=Path("outputs/worldparticle/assets")
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--start-frame", type=int, help="Override config frame_start after task stride"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--headless", action="store_true", help="No window; no pacing or final hold"
    )
    parser.add_argument(
        "--renderer", choices=("OFFLINERT", "HYBRID", "FASTRT"), default="OFFLINERT"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Display pacing only; checkpoint timestep is unchanged",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=-1.0,
        help="After inference: -1 keeps final frame until window closes/Ctrl+C; 0 exits",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        help="Save actual native camera frames for diagnosis",
    )
    parser.add_argument(
        "--once", action="store_true", help="Run once and hold the final frame"
    )
    parser.add_argument(
        "--show-boundary-cloud",
        action="store_true",
        help="Show the original boundary sample surfels",
    )
    args = parser.parse_args()
    profile = PROFILES[args.scene]
    checkpoint = (
        args.checkpoint
        or args.assets / "checkpoints" / profile["checkpoint"] / "final.ckpt"
    )
    config_path = (
        args.config
        or args.assets / "checkpoints" / profile["checkpoint"] / "config.yaml"
    )
    sample_path = args.sample or args.assets / profile["sample"]
    steps = profile["steps"] if args.steps is None else args.steps
    if steps < 1 or args.fps < 0 or min(args.width, args.height) < 1:
        parser.error("steps/window size must be positive and fps must be nonnegative")
    output = (
        args.output
        or Path("outputs/worldparticle/demos") / args.scene / "prediction.npz"
    )
    for path in (checkpoint, config_path, sample_path):
        if not path.is_file():
            parser.error(f"Missing local asset: {path}")
    with config_path.open() as stream:
        model_cfg = yaml.safe_load(stream)
    frame_start = (
        int(model_cfg.get("frame_start", 0))
        if args.start_frame is None
        else args.start_frame
    )
    frame_end = model_cfg.get("frame_end")
    initial, velocity, raw, box, normals, faces, reference = load_sample(
        sample_path, profile, frame_start=frame_start, frame_end=frame_end
    )
    rotation = (
        np.eye(3, dtype=np.float32)
        if profile["up"] == "z"
        else np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], np.float32
        )
    )
    q, qd = initial @ rotation.T, velocity @ rotation.T
    import dexsim
    from dexsim.engine.newton_physics import NewtonCfg, ProviderNewtonSolverCfg
    from embodichain.lab.sim.physics.worldparticle import WorldParticleProvider
    from dexsim.spawn import (
        SceneBuilder,
        FluidRenderMode,
        GeometryDesc,
        MaterialDesc,
        LightDesc,
        CameraDesc,
    )

    cfg = dexsim.WorldConfig()
    cfg.open_windows = not args.headless
    cfg.init_ground = False
    cfg.use_default_physics = False
    cfg.backend = (
        dexsim.types.Backend.OPENGL
        if args.renderer == "OFFLINERT"
        else dexsim.types.Backend.VULKAN
    )
    cfg.win_config.width = args.width
    cfg.win_config.height = args.height
    cfg.raytrace_config.spp = 4
    rt_mode = (
        dexsim.types.RTRenderMode.OPTIX_DENOISE
        if args.renderer == "OFFLINERT"
        else dexsim.types.RTRenderMode.RAW
    )
    cfg.set_rt_render_modes(window=rt_mode, offscreen=rt_mode)
    cfg.renderer = (
        dexsim.types.Renderer.NORENDER
        if args.headless
        else getattr(dexsim.types.Renderer, args.renderer)
    )
    cfg.newton_cfg = NewtonCfg(
        device="cuda:0",
        dt=model_cfg["timestep"],
        num_substeps=1,
        gravity=rotation @ np.asarray(model_cfg["gravity"]),
        sync_to_dexsim=not args.headless,
        collision_pipeline_cfg=None,
        solver_cfg=ProviderNewtonSolverCfg(
            provider_factory=WorldParticleProvider,
            options=dict(
                checkpoint=str(checkpoint),
                config=str(config_path),
                inputs=str(sample_path),
                model_to_world=rotation.tolist(),
            ),
        ),
    )
    cfg.newton_cfg.use_cuda_graph = False
    world = dexsim.World(cfg)
    builder = SceneBuilder(world)
    material = MaterialDesc(
        name="prediction",
        base_color=(0.18, 0.55, 0.95, 1.0),
        roughness=0.45,
        double_sided=True,
    )
    if args.headless:
        material = None
    if faces is not None:
        builder.add_clothbody(
            "prediction",
            GeometryDesc.mesh(vertices=q, triangles=faces),
            particle_masses=raw[:, 0].astype(np.float32),
            particle_radius=model_cfg["particle_radius"],
            material=material,
        )
    else:
        # FluidDesc is the public unconnected-particle carrier. Duck's learned
        # feature is not interpreted as mass or as a new SPH material model.
        mass, density = (
            (float(raw[0, 0]), float(raw[0, 1]))
            if args.scene == "fancy_fluid"
            else (
                (float(raw[0, 0]), 1000.0)
                if args.scene == "deep_fluid"
                else (1.0, 1000.0)
            )
        )
        if args.scene == "fancy_fluid" and not np.allclose(raw[:, :2], raw[0, :2]):
            raise ValueError("Fancy Fluid demo requires uniform mass/density")
        particle_material = (
            None
            if args.headless
            else MaterialDesc(
                name="neural_particles",
                base_color=(0.18, 0.58, 0.9, 1.0),
                roughness=0.55,
                metallic=0.05,
            )
        )
        builder.add_fluid(
            "prediction",
            q,
            density=density,
            particle_spacing=(mass / density) ** (1.0 / 3.0),
            particle_radius=model_cfg["particle_radius"],
            material=particle_material,
            render_radius=model_cfg["particle_radius"] * 1.6,
            render_mode=FluidRenderMode.PARTICLES,
        )
    if not args.headless:
        add_boundary_visuals(
            builder,
            box,
            normals,
            rotation,
            args.scene,
            model_cfg["particle_radius"],
            show_cloud=args.show_boundary_cloud,
        )
        pose = np.eye(4)
        pose[:3, 3] = q.mean(0) + [2.0, -3.0, 4.0]
        builder.add_light(
            LightDesc(
                name="key",
                light_type=dexsim.render.LightType.POINT,
                pose=pose,
                intensity=150.0,
            )
        )
    if args.capture_dir and not args.headless:
        builder.add_camera(
            CameraDesc(
                name="viewer",
                width=args.width,
                height=args.height,
                headless=True,
                near=0.01,
                far=100.0,
                fov=45.0,
            )
        )
    scene = builder.finalize()
    window = camera = handle = None
    try:
        if not args.headless:
            scene.set_environment_light_emission([0.8, 0.87, 1.0])
            scene.set_environment_light_intensity(80)
        handle = scene.get_particle_set("prediction")
        handle.set_particle_velocities(qd)
        if not args.headless:
            world.open_window()
            window = world.get_windows()
            framing = np.concatenate((q, box @ rotation.T))
            span = max(float(np.ptp(framing, axis=0).max()), 0.8)
            center = (framing.min(0) + framing.max(0)) / 2
            eye = center + span * np.array([0.9, -1.1, 0.7])
            window.set_look_at(eye=eye, look_at=center, up=np.array([0.0, 0.0, 1.0]))
            if args.capture_dir:
                camera = scene.get_camera("viewer")
                camera.look_at(eye, center)
                camera.open_camera()
                camera.render()
            print(
                f"{profile['title']} — live inference; close window or Ctrl+C to exit",
                flush=True,
            )
        actual_initial = handle.get_particle_positions().numpy() @ rotation
        np.testing.assert_allclose(actual_initial, initial, rtol=0, atol=1e-7)
        frames, velocities = [actual_initial.copy()], [velocity.copy()]
        if not args.headless:
            time.sleep(2.0)  # Let the initial scene become visible before inference.
        started = time.perf_counter()
        for step in range(steps):
            if not args.headless and not world.is_window_initialized():
                break
            before = time.perf_counter()
            world.update()
            if camera is not None:
                camera.render()
                if args.capture_dir and step in (0, steps // 2, steps - 1):
                    for _ in range(4):
                        camera.render()
                    from PIL import Image

                    args.capture_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.asarray(camera.get_rgb_map())[..., :3]).save(
                        args.capture_dir / f"frame-{step+1:04d}.png"
                    )
            frames.append(handle.get_particle_positions().numpy() @ rotation)
            velocities.append(handle.get_particle_velocities().numpy() @ rotation)
            if not (
                np.isfinite(frames[-1]).all() and np.isfinite(velocities[-1]).all()
            ):
                raise RuntimeError(f"Nonfinite neural state at step {step+1}")
            if step == 0 or (step + 1) % 20 == 0:
                print(f"{args.scene}: {step+1}/{steps}", flush=True)
            if not args.headless and args.fps:
                time.sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - before)))
        elapsed = time.perf_counter() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            position=np.stack(frames),
            velocity=np.stack(velocities),
            surface_faces=np.empty((0, 3), np.int32) if faces is None else faces,
        )
        report = dict(
            scene=args.scene,
            checkpoint=str(checkpoint),
            steps=len(frames) - 1,
            particles=len(initial),
            all_finite=True,
            initial_state_only=True,
            source_initial_frame=frame_start * profile["stride"],
            source_frame_stride=profile["stride"],
            window_size=None if args.headless else [args.width, args.height],
            reference_trajectory=profile["reference"],
            generated_initial_conditions=args.scene == "deep_fluid",
            elapsed_seconds=elapsed,
            display_included=not args.headless,
        )
        if reference is not None:
            count = min(len(reference), len(frames))
            report.update(
                reference_frames_compared=count,
                position_mse=float(
                    np.mean((np.stack(frames[:count]) - reference[:count]) ** 2)
                ),
            )
        output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved {output}\n{json.dumps(report,indent=2)}", flush=True)
        if not args.headless and args.hold_seconds != 0:
            deadline = time.monotonic() + args.hold_seconds
            replay_step = 0
            while world.is_window_initialized() and (
                args.hold_seconds < 0 or time.monotonic() < deadline
            ):
                if args.once:
                    time.sleep(0.05)
                    continue
                if replay_step == 0:
                    time.sleep(1.0)
                    scene.reset()
                    handle.set_particle_velocities(qd)
                    print("Restarting live inference", flush=True)
                before = time.perf_counter()
                world.update()
                replay_step = (replay_step + 1) % steps
                if args.fps:
                    time.sleep(
                        max(0.0, 1.0 / args.fps - (time.perf_counter() - before))
                    )
    except KeyboardInterrupt:
        print("Stopped by user", flush=True)
    finally:
        if camera is not None:
            camera.close_camera()
        window = None
        handle = None
        scene.close()


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        sys.argv.pop(1)
        from ._prepare import main as prepare_main

        prepare_main()
    else:
        if len(sys.argv) > 1 and sys.argv[1] == "demo":
            sys.argv.pop(1)
        _demo()


if __name__ == "__main__":
    main()
