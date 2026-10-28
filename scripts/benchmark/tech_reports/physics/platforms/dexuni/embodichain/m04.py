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
"""Replay the approved M04 T-shirt fold through EmbodiChain public scene APIs.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/embodichain/m04.py
Use --pilot-performance for a separate single no-camera timing probe.
Truncated diagnostics require --out <scratch>.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))
import dexsim
import newton
import warp as wp
from dexsim.engine.newton_physics import (
    KinematicJointTrajectoryControl,
    ParticleDragControl,
)
from dexsim.engine.newton_physics.backend_registry import get_newton_backend
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    LightCfg,
    NewtonCollisionPropertiesCfg,
    NewtonPhysicsCfg,
    NewtonRigidBodyMaterialCfg,
    RenderCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    SurfaceDeformableObjectCfg,
    SurfaceDeformablePhysicsCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_common import (
    OUTPUT,
    supports,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_replay import (
    CAMERA,
    load_inputs,
    finger_schedule,
    shape_damping,
    texture_uv,
    link_matrices,
    audit_cloth,
    record_preview,
)

from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_pilot import (
    pilot_directory,
    record_episode,
)

from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_measurement import (
    measurement_directory,
)

__all__ = ["main"]


def main() -> None:
    """Build the unchanged garment, replay approved commands and record states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-video", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot-performance", action="store_true")
    mode.add_argument("--measure", action="store_true")
    parser.add_argument("--repeat", type=int, choices=range(1, 6))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--until", type=float)
    args = parser.parse_args()
    if args.until is not None and args.out is None:
        parser.error("Truncated diagnostics require --out <scratch>")
    if args.pilot_performance:
        if args.out is not None or args.until is not None:
            parser.error(
                "Pilot uses the full accepted episode and its separate fixed output"
            )
        args.no_video = True
    if args.measure:
        if args.repeat is None or args.out is not None or args.until is not None:
            parser.error(
                "Measurement requires --repeat 1..5 and the full accepted configuration"
            )
        args.no_video = True
    elif args.repeat is not None:
        parser.error("--repeat requires --measure")
    inputs = load_inputs()
    cfg, mesh, commands = inputs["config"], inputs["mesh"], inputs["commands"]
    sol, c, cloth = cfg["solver"], cfg["contact"], cfg["cloth"]
    output = (
        pilot_directory("embodichain")
        if args.pilot_performance
        else (
            measurement_directory("embodichain", args.repeat)
            if args.measure
            else args.out or OUTPUT / "preview/embodichain"
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    width, height = CAMERA["resolution"]
    sim = SimulationManager(
        SimulationManagerCfg(
            num_envs=1,
            headless=True,
            device="cuda:0",
            width=width,
            height=height,
            enable_entity_gizmo=False,
            robot_ik_gizmo=None,
            render_cfg=RenderCfg(
                renderer="no-render" if args.no_video else "rt", spp=1
            ),
            physics_cfg=NewtonPhysicsCfg(
                device="cuda:0",
                physics_dt=sol["dt"],
                num_substeps=sol["num_substeps"],
                use_cuda_graph=sol["use_cuda_graph"],
                collision_cfg=None,
                solver_cfg=dict(
                    solver_type="dexuni",
                    mujoco_articulation_paths=("arena_0/w1",),
                    joint_mode=sol["joint_mode"],
                    coupling=sol["coupling"],
                    contact_mode=sol["contact_mode"],
                    vbd_preset=sol["vbd_preset"],
                    particle_coloring="newton_mcs",
                    vbd_options=dict(
                        iterations=sol["vbd_options"]["iterations"],
                        particle_enable_multilevel_correction=False,
                        particle_jacobi_polish_iterations=1,
                        particle_enable_self_contact=True,
                        particle_self_contact_radius=sol["self_contact_radius"],
                        particle_self_contact_margin=sol["self_contact_margin"],
                        particle_topological_contact_filter_threshold=1,
                        particle_rest_shape_contact_exclusion_radius=0.03,
                    ),
                    collision_options=dict(soft_contact_margin=c["soft_margin"]),
                ),
            ),
        )
    )

    def attrs(ke: float, mu: float) -> RigidBodyPhysicsCfg:
        return RigidBodyPhysicsCfg(
            collision_props=NewtonCollisionPropertiesCfg(
                margin=c["margin"], gap=c["gap"]
            ),
            material_props=NewtonRigidBodyMaterialCfg(
                ke=ke, kd=shape_damping(ke), dynamic_friction=mu
            ),
        )

    robot = garment = camera = None
    try:
        robot = sim.add_articulation(
            ArticulationCfg(
                uid="w1",
                fpath=str(OUTPUT / "inputs/w1_v030/robot.urdf"),
                init_pos=tuple(cfg["robot"]["base_position"]),
                build_pk_chain=False,
                asset_physics_mode="overlay",
                attrs=attrs(c["robot_ke"], c["open_mu"]),
                root_props=ArticulationRootPropertiesCfg(
                    fixed_base=True,
                    self_collision_enabled=True,
                    newton_mimic_stabilizer=False,
                    newton_force_sdf=True,
                ),
            )
        )
        sim.add_runtime_control(
            lambda arena, index: KinematicJointTrajectoryControl(
                f"{arena}/w1",
                commands["targets"],
                fps=1 / sol["dt"],
                joint_names=inputs["names"],
                initialize_pose=True,
            )
        )
        sim.add_runtime_control(
            lambda arena, index: ParticleDragControl(cloth["air_drag_rate"])
        )
        sim.register_contact_material_schedule(
            "w1",
            {
                "dynamic_friction": finger_schedule(cfg),
                "stiffness": ((0.0, c["robot_ke"]),),
                "damping": ((0.0, shape_damping(c["robot_ke"])),),
            },
            link_names=tuple(
                f"{side}_finger{k}" for side in ("left", "right") for k in (1, 2)
            ),
        )
        for name, center, size in supports(cfg):
            color = (0.53, 0.48, 0.39, 1) if name == "table" else (0.30, 0.31, 0.33, 1)
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid=name,
                    shape=CubeCfg(
                        size=list(size),
                        visual_material=VisualMaterialCfg(
                            uid=name, base_color=list(color)
                        ),
                    ),
                    body_type="static",
                    init_pos=center,
                    attrs=attrs(c["support_ke"], c["table_mu"]),
                )
            )
        tex = OUTPUT / "inputs/shirt/FabricGray"
        garment = sim.add_deformable_object(
            SurfaceDeformableObjectCfg(
                uid="cloth",
                shape=MeshCfg(
                    vertices=mesh["initial_positions"],
                    triangles=mesh["triangles"],
                    uv_coords=texture_uv(mesh["vertices"]),
                    visual_material=VisualMaterialCfg(
                        uid="FabricGray",
                        base_color=[1, 1, 1, 1],
                        roughness=0.8,
                        base_color_texture=str(tex / "FabricGray_Color.png"),
                        normal_texture=str(tex / "FabricGray_Normal.png"),
                        roughness_texture=str(tex / "FabricGray_Roughness.png"),
                    ),
                ),
                particle_radius=cloth["particle_radius"],
                attrs=SurfaceDeformablePhysicsCfg(
                    density=cloth["surface_density"],
                    edge_rest_angle=cloth["edge_rest_angle"],
                    surface_props={
                        **{
                            key: cloth[key]
                            for key in (
                                "tri_ke",
                                "tri_ka",
                                "tri_kd",
                                "edge_ke",
                                "edge_kd",
                            )
                        },
                        "tri_drag": 0.0,
                        "tri_lift": 0.0,
                    },
                    contact_material=dict(
                        stiffness=c["particle_ke"],
                        damping=c["particle_kd"],
                        dynamic_friction=c["particle_mu"],
                    ),
                ),
            )
        )
        focal = width / 2 / np.tan(np.deg2rad(CAMERA["horizontal_fov_deg"]) / 2)
        if not args.no_video:
            sim.add_light(
                LightCfg(
                    uid="key",
                    light_type="point",
                    init_pos=(1.5, -1, 3),
                    intensity=60,
                    enable_shadow=False,
                )
            )
            camera = sim.add_sensor(
                CameraCfg(
                    uid="preview",
                    width=width,
                    height=height,
                    near=0.01,
                    far=10,
                    intrinsics=(focal, focal, width / 2, height / 2),
                    enable_depth=False,
                    enable_mask=False,
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=CAMERA["eye"], target=CAMERA["target"], up=(0, 0, 1)
                    ),
                )
            )
        sim.prepare()
        ids = [robot.joint_names.index(n) for n in inputs["names"]]
        start = torch.as_tensor(
            commands["targets"][:1], dtype=torch.float32, device=sim.device
        )
        robot.set_qpos(start, joint_ids=ids, target=False)
        robot.set_qpos(start, joint_ids=ids, target=True)
        robot.clear_dynamics()
        if camera is not None:
            camera.set_intrinsics(
                torch.tensor(
                    [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
                    device=sim.device,
                    dtype=torch.float32,
                )
            )
            for material in garment.get_visual_material_inst():
                if material is not None:
                    material.mat.set_double_sided(True)
        particles = sim.spawn_result.get_particle_set("arena_0/cloth")
        initial = garment.data.nodal_pos_w[0].cpu().numpy()
        inv = particles.get_inverse_masses().numpy()
        audit = audit_cloth(inputs, initial, inv)
        backend = get_newton_backend(sim.get_world())
        model = backend.solver.vbd_solver.model
        fields = (
            "particle_mass",
            "tri_indices",
            "tri_materials",
            "edge_rest_angle",
            "edge_bending_properties",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_mu",
            "shape_margin",
        )
        np.savez_compressed(
            output / "initial_model.npz",
            particles=initial,
            inverse_mass=inv,
            shape_label=np.asarray(model.shape_label),
            **{k: getattr(model, k).numpy() for k in fields},
        )
        links = list(robot.link_names)
        hands = [
            links.index(f"{side}_{name}")
            for side in ("left", "right")
            for name in ("gripper_base", "finger1", "finger2")
        ]
        tcp_local = np.r_[cfg["robot"]["tcp_offset"], 1]

        def readings() -> dict[str, np.ndarray]:
            matrices = link_matrices(robot.body_data.body_link_pose[0].cpu().numpy())
            hand = matrices[hands]
            return dict(
                time=np.asarray(sim.spawn_result.get_simulation_time()),
                qpos=robot.get_qpos()[0, ids].cpu().numpy(),
                particles=garment.data.nodal_pos_w[0].cpu().numpy(),
                velocities=garment.data.nodal_vel_w[0].cpu().numpy(),
                tcp=(hand[[0, 3]] @ tcp_local)[:, :3],
                hand_poses=hand,
                link_poses=matrices,
            )

        def step() -> None:
            sim.update(physics_dt=sol["dt"], step=1, render_final_step=False)

        def capture() -> np.ndarray:
            with sim.render_frame():
                sim.sync_render_state()
                camera.update()
                return camera.get_data()["color"][0, ..., :3].cpu().numpy().copy()

        def diagnostics() -> dict[str, Any]:
            d = dict(sim.spawn_result.get_solver_diagnostics())
            assert d["vbd_preset"] == "surface-fast" and d["vbd"]["iterations"] == 16
            assert not d["vbd"]["multilevel_active"]
            return dict(
                d,
                versions=dict(
                    dexsim=dexsim.__version__,
                    newton=newton.__version__,
                    warp=wp.__version__,
                ),
                air_drag_rate=cloth["air_drag_rate"],
                finger_friction_keyframes=finger_schedule(cfg),
            )

        if args.pilot_performance or args.measure:
            record_episode(
                "embodichain",
                inputs,
                output,
                step,
                readings,
                lambda: wp.synchronize_device("cuda:0"),
                diagnostics,
                audit,
                links,
                repeat=args.repeat if args.measure else None,
            )
            return

        record_preview(
            "embodichain",
            inputs,
            output,
            step,
            readings,
            None if args.no_video else capture,
            diagnostics,
            audit,
            links,
            until=args.until,
        )
    finally:
        robot = garment = camera = particles = backend = model = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
