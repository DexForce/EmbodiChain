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

"""Slice a rounded MPM loaf using DexSim's bread demo and a prescribed tabletop fixture.

Use --headless for physics, --frames-dir for native OfflineRT camera images,
and --no-damage for a matched intact-material control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import torch
import numpy as np
from PIL import Image
import dexsim
from dexsim.engine.newton_physics import RealtimeJointTargetControl
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    DenoisingCfg,
    LightCfg,
    NewtonPhysicsCfg,
    RenderCfg,
    RigidObjectCfg,
    VolumeDeformableObjectCfg,
)
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.sensors import CameraCfg

if __package__:
    from ._dough.appearance import LoafSkin, textures
    from ._dough.cohesion import BreadCut, reference_example
else:
    from _dough.appearance import LoafSkin, textures
    from _dough.cohesion import BreadCut, reference_example

BOARD = 0.026
CUT_Y = -0.035
START = np.array([-0.14, CUT_Y, BOARD + 0.140])
DT = 1 / 60
STEPS = 480


def travel(t: float) -> np.ndarray:
    """Prescribe approach, alternating saw strokes, opening, and withdrawal."""
    times = (0.0, 0.8, 1.5, 2.9, 3.9, 4.8, 5.1, 6.0, 7.1, 8.0)
    poses = np.array(
        [
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, -0.02),
            (0.015, 0, -0.048),
            (-0.010, 0, -0.071),
            (0, 0, -0.088),
            (0, 0, -0.088),
            (0, -0.018, -0.088),
            (-0.040, -0.040, 0.043),
            (-0.075, -0.050, 0.071),
        ]
    )
    k = min(len(times) - 2, max(0, int(np.searchsorted(times, t, side="right") - 1)))
    u = np.clip((t - times[k]) / (times[k + 1] - times[k]), 0.0, 1.0)
    return poses[k] + (poses[k + 1] - poses[k]) * (u * u * (3.0 - 2.0 * u))


def fixture(sim, ref, directory):
    """Mount the reference utensil on three translation coordinates."""
    vertices, faces = ref.blade_mesh()
    mesh = directory / "utensil.obj"
    with mesh.open("w") as f:
        for x, y, z in vertices:
            f.write(f"v {x} {y} {z}\n")
        for a, b, c in faces + 1:
            f.write(f"f {a} {b} {c}\n")
    blocks = ['<robot name="fixture"><link name="base"/>']
    for i, (axis, name) in enumerate(zip(("1 0 0", "0 1 0", "0 0 1"), ("x", "y", "z"))):
        child = ("stage_x", "stage_y", "utensil")[i]
        parent = ("base", "stage_x", "stage_y")[i]
        origin = " ".join(map(str, START)) if i == 0 else "0 0 0"
        visual = (
            ""
            if i < 2
            else f"""<visual><geometry><mesh filename="{mesh.name}"/></geometry>
          <material name="steel"><color rgba=".67 .72 .76 1"/></material></visual>
          <collision><geometry><mesh filename="{mesh.name}"/></geometry></collision>
          <visual><geometry><box size=".124 .021 .026"/></geometry><material name="walnut"><color rgba=".36 .17 .075 1"/></material></visual>"""
        )
        blocks.append(
            f"""<link name="{child}"><inertial><mass value=".25"/><inertia ixx=".001" iyy=".001" izz=".001" ixy="0" ixz="0" iyz="0"/></inertial>{visual}</link>
            <joint name="{name}" type="prismatic"><parent link="{parent}"/><child link="{child}"/><origin xyz="{origin}"/><axis xyz="{axis}"/><limit lower="-.3" upper=".3" effort="500" velocity="2"/></joint>"""
        )
    path = directory / "fixture.urdf"
    path.write_text("".join(blocks) + "</robot>")
    sim._spawn_scene.builder.add_urdf(
        "fixture",
        str(path),
        fix_root_link=True,
        per_env=True,
        newton_gravity_compensation=1.0,
        newton_collision=dexsim.spawn.NewtonCollisionDesc(
            mu=0.03, gap=0.002, margin=0.0
        ),
        newton_joint_drives={
            name: dexsim.spawn.NewtonJointDesc(
                target_ke=2e4, target_kd=140.0, target_mode=1
            )
            for name in ("x", "y", "z")
        },
    )
    drive = RealtimeJointTargetControl(
        "arena_0/fixture", [0.0, 0.0, 0.0], joint_names=["x", "y", "z"]
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
    points = (ref.loaf_points() - ref.LOAF_CENTER + np.array([0, 0, BOARD])).astype(
        np.float32
    )
    draw = not args.headless or args.frames_dir is not None
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=args.headless,
            enable_ground_plane=False,
            startup_summary="off",
            render_cfg=RenderCfg(
                renderer="rt" if draw else "no-render",
                spp=16,
                denoising=DenoisingCfg(window="off", offscreen="optix"),
            ),
            physics_cfg=NewtonPhysicsCfg(
                device="cuda:0",
                physics_dt=DT,
                num_substeps=6,
                collision_cfg=None,
                solver_cfg={
                    "solver_type": "dexuni",
                    "coupling": "one_way",
                    "particle_options": {
                        "rigid_iterations": 60,
                        "mpm_options": {
                            "voxel_size": 0.006,
                            "grid_type": "fixed",
                            "grid_padding": 20,
                            "max_active_cell_count": 8192,
                            "strain_basis": "P0",
                            "collider_basis": "pic128",
                            "transfer_scheme": "pic",
                            "max_iterations": 50,
                            "solver": "gs",
                            "air_drag": 0.0,
                        },
                    },
                },
            ),
        )
    )
    directory = tempfile.TemporaryDirectory(prefix="bread-fixture-")
    try:
        drive = fixture(sim, ref, Path(directory.name))
        if draw:
            textures(Path(directory.name))
        wood = VisualMaterialCfg(
            base_color_texture=(
                str(Path(directory.name) / "board.png") if draw else None
            ),
            base_color=[1.0, 1.0, 1.0, 1.0],
            metallic=0.0,
            roughness=0.89,
        )
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="board",
                body_type="static",
                init_pos=(0.0, 0.0, BOARD / 2),
                shape=CubeCfg(size=(0.35, 0.34, BOARD), visual_material=wood),
            )
        )
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="table",
                body_type="static",
                init_pos=(0.0, 0.0, -0.016),
                shape=CubeCfg(
                    size=(8.0, 8.0, 0.03),
                    visual_material=VisualMaterialCfg(
                        base_color=[0.32, 0.39, 0.38, 1.0], metallic=0.0, roughness=0.96
                    ),
                ),
            )
        )
        bread = sim.add_deformable_object(
            VolumeDeformableObjectCfg.from_dict(
                {
                    "uid": "bread",
                    "particle_radius": ref.SPACING / 2,
                    "model": {
                        "kind": "material_point",
                        "sampling": {"kind": "points", "points": points},
                        "physics": {
                            "particle_masses": 280 * ref.SPACING**3,
                            "young_modulus": 4000.0,
                            "poisson_ratio": 0.25,
                            "damping": 0.015,
                            "yield_pressure": 5000.0,
                            "yield_stress": 3000.0,
                            "tensile_yield_ratio": 0.3,
                            "friction": 0.4,
                        },
                    },
                }
            )
        )
        if draw:
            desc = sim._spawn_scene.builder.particle_sets[0]
            desc.render_mode = dexsim.spawn.MPMRenderMode.PARTICLES
            desc.render_radius = (
                0.00001  # The particle-driven skin supplies the optical surface.
            )
        cut = BreadCut(ref, points, not args.no_damage)
        sim.add_runtime_control(cut)
        sim.prepare()
        camera = camera_setup(sim) if draw else None
        skin = (
            LoafSkin(sim, points, BOARD, CUT_Y, Path(directory.name)) if draw else None
        )
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
        near = np.abs(points[:, 1] - CUT_Y) < 0.015
        sides = np.where(points[:, 1] < CUT_Y, -1.0, 1.0)
        before_cut = points.copy()
        history = []
        snapshots = []
        all_labels = []
        tool_poses = []
        for frame in range(STEPS + 1):
            t = frame * DT
            if frame:
                drive.set_target(travel(t).tolist())
                sim.update(step=1, render_final_step=not args.headless)
            if frame % 2 and args.headless:
                continue
            live_q = bread.data.nodal_pos_w[0]
            labels, sizes = cut.network.components()
            if skin is not None:
                separated = len(sizes) == 2 and sizes.min() >= 0.1 * len(points)
                if separated and (
                    np.unique(labels[points[:, 1] < CUT_Y]).size != 1
                    or np.unique(labels[points[:, 1] > CUT_Y]).size != 1
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
                np.quantile(q[(points[:, 1] > CUT_Y) & near, 1], 0.05)
                - np.quantile(q[(points[:, 1] < CUT_Y) & near, 1], 0.95)
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
        loose = ref.count_detached_samples(q, labels, ref.SPACING)
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
        affected = (cut.damage.numpy() > 0) & (points[:, 2] > BOARD + 0.045)
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
                dt=2 * DT,
                spacing=ref.SPACING,
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
        directory.cleanup()


if __name__ == "__main__":
    main()
