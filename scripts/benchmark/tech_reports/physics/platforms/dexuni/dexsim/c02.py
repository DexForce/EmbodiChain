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
"""Record the supplied W1 bag demo without replacing its scene or controller.

Run with the staged DexSim/Newton environment. --lift adds a rim-contact
lifting tail after the unmodified 50-second reference task.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

__all__ = ["main"]

ROOT = Path(__file__).resolve().parents[6]
REFERENCE = Path(
    "/home/oem/code/repos/dexsimforembodychain/dexsim/modules/dexsim/examples/python/physics/multiphys"
)
ASSETS = Path("/home/oem/.cache/dexsim/resources/mjvbd_v2/w1_bag_packing")


def sha(path: Path) -> str:
    """Return the content hash used by execution manifests."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    """Save small metadata with NumPy scalar/array conversion."""
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item(),
        )
        + "\n"
    )


def phase(time: float) -> str:
    """Describe the original timing and optional contact-only lifting tail."""
    for end, label in (
        (4.5, "approach paper bag"),
        (10.5, "stand paper bag upright"),
        (24, "left hand regrasp rim"),
        (36, "pack can"),
        (48, "pack carton"),
        (50, "settle original demo"),
        (56, "right hand regrasp opposite rim"),
        (58, "lift loaded bag"),
        (60, "hold loaded bag"),
        (62, "lower loaded bag"),
    ):
        if time < end:
            return label
    return "settle"


class LiftTargets:
    """Delegate the complete original planner, then add a physical rim regrasp."""

    def __init__(self, base: Any, reference: Any, rim: np.ndarray) -> None:
        self.base, self.reference, self.rim = base, reference, rim
        self.waypoints: list[tuple] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def command(
        self, time: float, positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if time < 50 - 1e-8:
            return self.base.command(time, positions)
        if self.waypoints is None:
            targets, openings, pitches = self.base._plan(50.0)
            rest = self.base.rest[self.rim]
            ids = self.rim[(rest[:, 1] < -0.159) & (abs(rest[:, 0]) < 0.025)]
            assert len(ids) > 0
            right = positions[ids].mean(0) + np.array((0, 0, 0.005))
            left = targets[0].copy()
            self.waypoints = [
                (50.0, targets.copy(), openings.copy(), pitches.copy()),
                (
                    52.0,
                    np.array((left, right + (0, -0.10, 0.10))),
                    np.array((0.0035, 0.045)),
                    np.radians((95, 95)),
                ),
                (
                    54.0,
                    np.array((left, right + (0, 0, 0.10))),
                    np.array((0.0035, 0.020)),
                    np.radians((95, 95)),
                ),
                (
                    55.0,
                    np.array((left, right)),
                    np.array((0.0035, 0.020)),
                    np.radians((95, 95)),
                ),
                (
                    56.0,
                    np.array((left, right)),
                    np.array((0.0035, 0.0035)),
                    np.radians((95, 95)),
                ),
                (
                    58.0,
                    np.array((left, right)) + (0, 0, 0.08),
                    np.array((0.0035, 0.0035)),
                    np.radians((95, 95)),
                ),
                (
                    60.0,
                    np.array((left, right)) + (0, 0, 0.08),
                    np.array((0.0035, 0.0035)),
                    np.radians((95, 95)),
                ),
                (
                    62.0,
                    np.array((left, right)),
                    np.array((0.0035, 0.0035)),
                    np.radians((95, 95)),
                ),
                (
                    64.0,
                    np.array((left, right)),
                    np.array((0.0035, 0.0035)),
                    np.radians((95, 95)),
                ),
            ]
        t = time + 1 / 60
        for a, b in zip(self.waypoints[:-1], self.waypoints[1:], strict=True):
            if t <= b[0]:
                weight = self.reference.smooth((t - a[0]) / (b[0] - a[0]))
                return tuple(
                    (1 - weight) * x + weight * y
                    for x, y in zip(a[1:], b[1:], strict=True)
                )
        return self.waypoints[-1][1:]


def main() -> None:
    """Run the exact reference build/controller/checks and record real states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--until", type=float, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--lift", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    source.mkdir()
    reference_source = source / "reference"
    reference_source.mkdir()
    for name in ("demo_dexuni_w1_bag_packing.py", "dexuni_asset_utils.py"):
        shutil.copy2(REFERENCE / name, reference_source / name)
    shutil.copy2(Path(__file__), source / "c02.py")
    shutil.copytree(ASSETS, out / "inputs")
    sys.path.insert(0, str(reference_source))
    spec = importlib.util.spec_from_file_location(
        "c02_original_bag_demo", reference_source / "demo_dexuni_w1_bag_packing.py"
    )
    demo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = demo
    spec.loader.exec_module(demo)
    import dexsim
    import warp as wp
    from dexsim.engine.newton_physics.backend_registry import get_newton_backend

    runtime_args = SimpleNamespace(
        device=args.device,
        assets=out / "inputs",
        w1_urdf=out / "inputs/w1_v030/robot.urdf",
        disable_cuda_graph=False,
    )
    with np.load(runtime_args.assets / "bag.npz") as a:
        mesh = dict(a)
    controller = demo.W1BagController(
        runtime_args.w1_urdf,
        mesh["vertices"],
        int(mesh["paper_count"]),
        mesh["bottom"],
        device=args.device,
    )
    if args.lift:
        controller.plan = LiftTargets(controller.plan, demo, mesh["rim"])
    config = dexsim.WorldConfig()
    config.use_default_physics = False
    config.open_windows = False
    config.backend = dexsim.types.Backend.VULKAN
    config.renderer = dexsim.types.Renderer.OFFLINERT
    config.raytrace_config.spp = 1
    config.set_rt_render_modes(
        window=dexsim.types.RTRenderMode.DLSS_RR,
        offscreen=dexsim.types.RTRenderMode.DLSS_RR,
    )
    config.dlss_config.dlss_quality = 3
    config.newton_cfg = demo.solver_config(runtime_args)
    preview = out / "preview"
    preview.mkdir()
    analysis = out / "analysis"
    analysis.mkdir()
    manifest = dict(
        task="C02",
        scene="supplied W1 paper-bag demo",
        status="running",
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        reference=str(REFERENCE / "demo_dexuni_w1_bag_packing.py"),
        reference_sha256=sha(reference_source / "demo_dexuni_w1_bag_packing.py"),
        native_build=dexsim.__version__,
        newton_module=__import__("newton").__file__,
        warp_version=wp.__version__,
        original_assets=True,
        original_scene_builder=True,
        original_solver_config=True,
        original_controller=True,
        physical_changes_to_reference=[],
        lift_tail=args.lift,
        performance_measurements=False,
        user_scene_approved=False,
    )
    write_json(out / "manifest.json", manifest)
    world = dexsim.World(config)
    world.set_manual_update(True)
    scene = camera = writer = None
    states = {}
    commands = []
    checks = None
    reference_pass = False
    try:
        scene, control = demo.build_scene(world, runtime_args, controller)
        world.enable_physics(True)
        backend = get_newton_backend(world)
        checks = demo.W1BagChecks(scene, controller, runtime_args.assets)
        manifest.update(
            joint_state_order=list(controller.joint_names),
            link_state_order=list(checks.names),
            table_centre=demo.TABLE_CENTER,
            table_half=demo.TABLE_HALF,
            original_handle_paper_check_stride=10,
        )
        model = backend.model
        assert np.all(
            model.particle_inv_mass.numpy() > 0
        ), "Original paper must remain unpinned"
        initial = {}
        for key in (
            "particle_q",
            "particle_mass",
            "particle_inv_mass",
            "particle_radius",
            "tri_indices",
            "tri_poses",
            "tri_materials",
            "edge_indices",
            "edge_rest_angle",
            "edge_bending_properties",
            "shape_transform",
            "shape_scale",
            "shape_body",
            "body_mass",
            "body_inertia",
        ):
            value = getattr(model, key, None)
            if value is not None and hasattr(value, "numpy"):
                initial[key] = value.numpy()
        np.savez_compressed(analysis / "initial_model.npz", **initial)
        if not args.no_video:
            camera = scene.add_camera(
                demo.spawn.CameraDesc(
                    "preview",
                    resolution=(1280, 720),
                    near=0.01,
                    far=10.0,
                    headless=True,
                )
            )
            camera.set_perspective(40.0, 1280 / 720, 0.01, 10.0)
            camera.look_at((2.75, -0.45, 2.20), (0.44, -0.03, 1.08), (0, 0, 1))
            camera.open_camera()
            writer = imageio.get_writer(
                preview / "dexsim.mp4",
                fps=30,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )
        total = 64.0 if args.lift else 50.0
        last = round(min(total, args.until if args.until is not None else total) * 60)
        current = controller.initial_joint_positions.copy()
        for frame in range(last + 1):
            if frame:
                current = controller.command(
                    (frame - 1) / 60, checks.bag.get_particle_positions().numpy()
                )
                control.set_target(current)
                world.update(1 / 60)
            checks.update(frame)
            actual = float(scene.get_simulation_time())
            assert abs(actual - frame / 60) < 1e-7
            q = checks.joints[0, checks.joint_order].copy()
            assert np.max(abs(q - current)) < 2e-5
            tcp = np.array(
                [
                    wp.transform_point(
                        wp.transform(*checks.body[0, body, [4, 5, 6, 0, 1, 2, 3]]),
                        wp.vec3(*centre),
                    )
                    for body, centre in zip(
                        checks.wrists, controller.pinch_centers, strict=True
                    )
                ]
            )
            sample = dict(
                time=np.asarray(actual),
                bag=checks.bag.get_particle_positions().numpy(),
                qpos=q,
                links=checks.body[0],
                snack_pose=checks.snack_pose,
                tcp=tcp,
                tcp_targets=controller.targets,
                bag_position=checks.position,
                bag_rotation=checks.rotation,
                loaded=checks.loaded,
                shape_error=np.asarray(checks.shape_error),
            )
            for k, v in sample.items():
                assert np.isfinite(v).all(), k
                states.setdefault(k, []).append(np.array(v, copy=True))
            commands.append(current.copy())
            if frame == 3000:
                checks.final()
                reference_pass = True
            if writer is not None and frame % 2 == 0:
                backend.sync_to_dexsim(world)
                world._flush_render_updates()
                camera.render()
                image = Image.fromarray(np.asarray(camera.get_rgb_map())[..., :3])
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 1280, 36), fill="black")
                draw.text(
                    (12, 10),
                    f"C02 | original W1 paper-bag demo | {actual:.2f}s | {phase(actual)}",
                    fill="white",
                )
                writer.append_data(np.asarray(image))
                if frame in (0, 600, 1200, 2100, 2820, 3000, 3360, 3480, 3600, 3840):
                    image.save(preview / f"still_{actual:05.2f}s.jpg")
            if frame % 120 == 0:
                print(
                    f"C02 reference frame={frame}/{last} actual={actual:.2f}s upright={checks.rotation[2,2]:.5f} loaded={checks.loaded.tolist()}",
                    flush=True,
                )
        manifest.update(
            status="completed",
            actual_time_s=float(scene.get_simulation_time()),
            actual_steps=int(backend.step_index),
            reference_final_checks_passed=reference_pass,
            effective=dict(scene.get_solver_diagnostics()),
            max_ik_error_m=checks.peak_ik,
            max_lip_bow_m=checks.peak_lip,
            peak_snack_z=checks.peak_z,
        )
        if args.lift and isinstance(controller.plan, LiftTargets):
            write_json(out / "lift_waypoints.json", controller.plan.waypoints)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if writer is not None:
            writer.close()
        if states:
            np.savez_compressed(
                preview / "states.npz", **{k: np.stack(v) for k, v in states.items()}
            )
            np.savez_compressed(
                out / "inputs/joint_commands.npz",
                time=np.asarray(states["time"]),
                joint_names=np.array(controller.joint_names),
                targets=np.stack(commands),
            )
        manifest.update(
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            input_sha256={
                str(p.relative_to(out / "inputs")): sha(p)
                for p in (out / "inputs").rglob("*")
                if p.is_file()
            },
            source_sha256={
                str(p.relative_to(source)): sha(p) for p in source.rglob("*.py")
            },
        )
        if scene is not None:
            manifest["effective"] = dict(scene.get_solver_diagnostics())
        write_json(out / "manifest.json", manifest)
        if camera is not None:
            camera.close_camera()
        if scene is not None:
            scene.close()


if __name__ == "__main__":
    main()
