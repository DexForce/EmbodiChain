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

"""Slice the packaged MPM loaf using a prescribed tabletop fixture.

Use --headless for physics, --frames-dir for native OfflineRT camera images,
and --no-damage for a matched intact-material control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import dexsim
import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dexsim.engine.newton_physics import RealtimeJointTargetControl
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.cfg import DenoisingCfg, LightCfg, RenderCfg
from embodichain.lab.sim.sensors import CameraCfg
from examples.sim.mpm._scene import DemoScene

if __package__:
    from ._dough.appearance import LoafSkin
    from ._dough.cohesion import BreadCut, reference_example
else:
    from _dough.appearance import LoafSkin
    from _dough.cohesion import BreadCut, reference_example


def add_fixture(
    sim: SimulationManager, scene: DemoScene, initial_target: np.ndarray
) -> RealtimeJointTargetControl:
    """Bind the packaged three-axis fixture and its authored drives."""
    options = dict(scene.spec["fixture_options"])
    options["newton_collision"] = dexsim.spawn.NewtonCollisionDesc(
        **options["newton_collision"]
    )
    options["newton_joint_drives"] = {
        name: dexsim.spawn.NewtonJointDesc(**drive)
        for name, drive in options["newton_joint_drives"].items()
    }
    sim._spawn_scene.builder.add_urdf(
        "fixture", str(scene.path("fixture_urdf")), **options
    )
    drive = RealtimeJointTargetControl(
        "arena_0/fixture",
        initial_target.tolist(),
        joint_names=scene.spec["joint_order"],
    )
    sim.add_runtime_control(drive)
    return drive


def camera_setup(sim):
    sim.set_emission_light([1.0, 0.97, 0.91], 24.0)
    sim.add_light(
        LightCfg(
            uid="key",
            light_type="sun",
            direction=(-0.52, 0.65, -1.0),
            intensity=12.0,
            color=(1.0, 0.91, 0.79),
        )
    )
    sim.add_light(
        LightCfg(
            uid="fill",
            light_type="direction",
            direction=(0.32, -0.4, -0.9),
            intensity=5.0,
            color=(0.78, 0.89, 1.0),
        )
    )
    camera = sim.add_sensor(
        CameraCfg(
            uid="bread_camera",
            width=960,
            height=540,
            near=0.005,
            far=4.0,
            intrinsics=(850.0, 850.0, 480.0, 270.0),
            extrinsics=CameraCfg.ExtrinsicsCfg(
                eye=(0.280, -0.210, 0.250),
                target=(-0.002, -0.010, 0.065),
                up=(0.0, 0.0, 1.0),
            ),
        )
    )
    sim.prepare()
    return camera


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-damage", action="store_true")
    parser.add_argument("--dexsim-examples", type=Path)
    parser.add_argument(
        "--frames-dir",
        type=Path,
        help="Save native camera frames at 30 Hz and their measured states.",
    )
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    ref = reference_example(args.dexsim_examples)
    scene = DemoScene("bread_cut")
    points = scene.positions @ scene.pose[:3, :3].T + scene.pose[:3, 3]
    cut_y = scene.spec["cut_y_m"]
    board_top = scene.spec["board_top_m"]
    spacing = 2 * scene.radius
    if not np.all(scene.masses == scene.masses[0]):
        raise ValueError("The bread cohesion model requires uniform particle mass.")
    with np.load(scene.path("fixture_motion"), allow_pickle=False) as motion:
        times = motion["times"]
        targets = motion["target_positions"]
    frames = scene.steps + 1
    if times.shape != (frames,) or targets.shape != (
        frames,
        len(scene.spec["joint_order"]),
    ):
        raise ValueError(
            "Fixture motion does not match the demo step and joint counts."
        )

    draw = not args.headless or args.frames_dir is not None
    sim = scene.create_simulation(
        headless=args.headless,
        render_cfg=RenderCfg(
            renderer="rt" if draw else "no-render",
            spp=16,
            denoising=DenoisingCfg(window="off", offscreen="optix"),
        ),
    )
    try:
        drive = add_fixture(sim, scene, targets[0])
        scene.add_environment(sim)
        bread = scene.add_material(sim)
        if draw:
            desc = sim._spawn_scene.builder.particle_sets[0]
            desc.render_mode = dexsim.spawn.MPMRenderMode.PARTICLES
            desc.render_radius = (
                0.00001  # The particle-driven skin supplies the optical surface.
            )
        cut = BreadCut(
            ref,
            points,
            not args.no_damage,
            mass=float(scene.masses[0]),
            spacing=spacing,
            yield_stress=scene.material["yield_stress"],
            tensile_yield_ratio=scene.material["tensile_yield_ratio"],
        )
        sim.add_runtime_control(cut)
        sim.prepare()
        camera = camera_setup(sim) if draw else None
        skin = LoafSkin(sim, scene) if draw else None
        if not args.headless:
            sim.open_window()
            sim.viewer.set_look_at(
                eye=np.array([0.310, -0.28, 0.210]),
                look_at=np.array([0.0, 0.0, 0.068]),
                up=np.array([0.0, 0.0, 1.0]),
            )
        if args.frames_dir:
            args.frames_dir.mkdir(parents=True, exist_ok=True)
        tool = sim.spawn_result.get_articulation("arena_0/fixture")
        native = sim.spawn_result.get_mpm_particle_set("arena_0/bread")
        mass0 = native.get_inverse_masses().numpy().copy()
        initial_n = len(cut.network.components()[1])
        assert initial_n == 1
        near = np.abs(points[:, 1] - cut_y) < 0.015
        left = points[:, 1] < cut_y
        right = points[:, 1] > cut_y
        sides = np.where(left, -1.0, 1.0)
        before_cut = points.copy()
        history = []
        snapshots = []
        all_labels = []
        tool_poses = []
        for frame, t in enumerate(times):
            if frame:
                drive.set_target(targets[frame].tolist())
                sim.update(step=1, render_final_step=not args.headless)
            if frame % 2 and args.headless:
                continue
            live_q = bread.data.nodal_pos_w[0]
            labels, sizes = cut.network.components()
            if skin is not None:
                separated = len(sizes) == 2 and sizes.min() >= 0.1 * len(points)
                if separated and (
                    np.unique(labels[left]).size != 1
                    or np.unique(labels[right]).size != 1
                ):
                    raise RuntimeError(
                        "Measured material components diverged from the authored single-cut skin."
                    )
                skin.update(live_q, separated)
            if frame % 2:
                continue
            q = live_q.cpu().numpy()
            if not np.isfinite(q).all():
                raise RuntimeError("Nonfinite bread positions")
            if frame == 48:
                before_cut = q.copy()
            pose = tool.get_link_pose("utensil")
            surface = float(
                np.quantile(q[right & near, 1], 0.05)
                - np.quantile(q[left & near, 1], 0.95)
            )
            history.append(
                [
                    t,
                    surface,
                    float(2 * np.mean((q[near, 1] - points[near, 1]) * sides[near])),
                    float(np.quantile(q[:, 2], 0.98)),
                    len(sizes),
                ]
            )
            if args.trace:
                snapshots.append(q)
                all_labels.append(labels)
                tool_poses.append(pose)
            if args.frames_dir:
                with sim.render_frame():
                    sim.sync_render_state()
                    sim.render_camera_group(list(camera.group_ids))
                    camera.update(fetch_only=True)
                    rgb = (
                        camera.get_data()["color"][0, :, :, :3]
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                Image.fromarray(rgb).save(
                    args.frames_dir / f"{frame//2:04}.png", compress_level=1
                )
            if frame % 60 == 0:
                print(
                    f"{t:.1f}s: pieces={sizes.tolist()}, opening={surface*1000:.1f} mm",
                    flush=True,
                )
        h = np.asarray(history)
        base = h[(h[:, 0] >= 0.5) & (h[:, 0] <= 0.8), 1].mean()
        final = h[h[:, 0] >= 7.3]
        loose = ref.count_detached_samples(q, labels, spacing)
        results = dict(
            damage=cut.enabled,
            particles=len(points),
            components=sizes.tolist(),
            detached=int(loose),
            residual_mm=float(1000 * (final[:, 1].min() - base)),
            surface_gap_mm=float(1000 * final[:, 1].min()),
            mass_kg=float((1 / mass0).sum()),
            mass_preserved=bool(
                np.array_equal(mass0, native.get_inverse_masses().numpy())
            ),
            solver=sim.spawn_result.get_solver_diagnostics()["dexuni_solver"],
            sdk=dexsim.__version__,
        )
        loaded = h[(h[:, 0] >= 5.1) & (h[:, 0] <= 6.25)]
        results["recoil_mm"] = float(1000 * (loaded[:, 1].max() - final[:, 1].mean()))
        affected = (cut.damage.numpy() > 0) & (points[:, 2] > board_top + 0.045)
        results["retained_notch_mm"] = (
            float(1000 * np.mean(before_cut[affected, 2] - q[affected, 2]))
            if affected.any()
            else None
        )
        results["reference_opening_mm"] = float(1000 * final[:, 2].mean())
        if args.frames_dir:
            camera.look_at(
                torch.tensor([[0.116, -0.166, 0.250]], device=sim.device),
                torch.tensor([[0.000, -0.032, 0.051]], device=sim.device),
            )
            camera.set_intrinsics(
                torch.tensor([[1050.0, 1050.0, 480.0, 270.0]], device=sim.device)
            )
            with sim.render_frame():
                sim.sync_render_state()
                sim.render_camera_group(list(camera.group_ids))
                camera.update(fetch_only=True)
                Image.fromarray(
                    camera.get_data()["color"][0, :, :, :3].contiguous().cpu().numpy()
                ).save(args.frames_dir / "detail.png")
        if args.trace:
            np.savez_compressed(
                args.trace,
                xyz=snapshots,
                labels=all_labels,
                tool=tool_poses,
                history=h,
                dt=2 * scene.physics.physics_dt,
                spacing=spacing,
                metrics=json.dumps(results),
            )
        print(json.dumps(results), flush=True)
        if (
            not results["mass_preserved"]
            or loose
            or (
                cut.enabled
                and (
                    len(sizes) != 2
                    or sizes.min() < 0.10 * len(points)
                    or results["residual_mm"] < 2.0
                    or results["recoil_mm"] < 0.5
                    or results["reference_opening_mm"] < 2.0
                    or (results["retained_notch_mm"] or 0.0) < 1.5
                    or np.any(final[:, 4] != 2)
                )
            )
            or (not cut.enabled and len(sizes) != 1)
        ):
            raise RuntimeError("Bread cutting acceptance failed")
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
