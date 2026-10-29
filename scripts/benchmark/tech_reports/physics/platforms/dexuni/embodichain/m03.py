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
"""Replay approved four-layer M03 through public EmbodiChain scene APIs.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/embodichain/m03.py
No performance samples are collected; --no-video is a full diagnostic episode.
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))
from dexsim.engine.newton_physics import KinematicJointTrajectoryControl
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
    VisualObjectCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m03_common import (
    OUTPUT,
    CAMERAS,
    load_inputs,
    boxes,
    audit_layers,
    record_preview,
)

__all__ = ["main"]


def main() -> None:
    """Build four independent dynamic layers and replay the frozen named joints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    init_started = time.perf_counter()
    if args.measure:
        args.no_video = True
    inputs = load_inputs()
    root = inputs["output_root"]
    cfg = inputs["config"]
    sol = cfg["solver"]
    contact = cfg["contact"]
    cloth = cfg["cloth"]
    output = root / "preview/embodichain"
    output.mkdir(parents=True, exist_ok=True)
    if args.measure:
        from scripts.benchmark.tech_reports.physics.experiments.dexuni.m03_measurement import (
            measurement_directory,
        )

        output = measurement_directory("embodichain", args.repeat)
    sim = SimulationManager(
        SimulationManagerCfg(
            num_envs=1,
            headless=True,
            device="cuda:0",
            width=960,
            height=720,
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
                solver_cfg={
                    "solver_type": "dexuni",
                    "mujoco_articulation_paths": ["arena_0/w1"],
                    "joint_mode": sol["joint_mode"],
                    "contact_mode": sol["contact_mode"],
                    "coupling": sol["coupling"],
                    "vbd_preset": sol["vbd_preset"],
                    "particle_coloring_include_bending": sol[
                        "particle_coloring_include_bending"
                    ],
                    "particle_coloring_balance_colors": sol[
                        "particle_coloring_balance_colors"
                    ],
                    "vbd_options": {
                        "particle_enable_self_contact": True,
                        "particle_self_contact_radius": sol["self_contact_radius"],
                        "particle_self_contact_margin": sol["self_contact_margin"],
                        "particle_rest_shape_contact_exclusion_radius": sol[
                            "rest_shape_exclusion_radius"
                        ],
                        "particle_topological_contact_filter_threshold": 1,
                        "particle_vertex_contact_buffer_size": 64,
                        "particle_edge_contact_buffer_size": 64,
                        "particle_collision_detection_interval": -1,
                    },
                    "collision_options": {
                        "soft_contact_margin": contact["soft_contact_margin"]
                    },
                },
            ),
        )
    )

    def attrs() -> RigidBodyPhysicsCfg:
        return RigidBodyPhysicsCfg(
            collision_props=NewtonCollisionPropertiesCfg(margin=0.001, gap=0.002),
            material_props=NewtonRigidBodyMaterialCfg(
                ke=contact["shape_ke"],
                kd=contact["shape_kd"],
                dynamic_friction=contact["table_friction"],
            ),
        )

    robot = None
    layers = []
    cameras = []
    try:
        robot = sim.add_articulation(
            ArticulationCfg(
                uid="w1",
                fpath=str(root / "inputs/w1_v030/robot.urdf"),
                init_pos=tuple(cfg["robot"]["base_position"]),
                build_pk_chain=False,
                asset_physics_mode="overlay",
                attrs=attrs(),
                root_props=ArticulationRootPropertiesCfg(
                    fixed_base=True, newton_mimic_stabilizer=False
                ),
            )
        )
        sim.add_runtime_control(
            lambda arena, index: KinematicJointTrajectoryControl(
                f"{arena}/w1",
                inputs["targets"],
                fps=1 / sol["dt"],
                joint_names=inputs["names"],
                initialize_pose=True,
            )
        )
        sim.register_contact_material_schedule(
            "w1",
            {
                "dynamic_friction": ((0.0, contact["gripper_friction"]),),
                "stiffness": ((0.0, contact["gripper_ke"]),),
                "damping": ((0.0, contact["gripper_kd"]),),
            },
            link_names=tuple(
                f"{side}_finger{k}" for side in ("left", "right") for k in (1, 2)
            ),
        )
        for name, center, size, color, collision in boxes(cfg):
            shape = CubeCfg(
                size=list(size),
                visual_material=VisualMaterialCfg(uid=name, base_color=[*color, 1]),
            )
            if collision:
                sim.add_rigid_object(
                    RigidObjectCfg(
                        uid=name,
                        shape=shape,
                        body_type="static",
                        init_pos=center,
                        attrs=attrs(),
                    )
                )
            elif not args.no_video:
                sim.add_visual_object(
                    VisualObjectCfg(uid=name, shape=shape, init_pos=center)
                )
        for k in range(4):
            layers.append(
                sim.add_deformable_object(
                    SurfaceDeformableObjectCfg(
                        uid=f"cloth_{k}",
                        shape=MeshCfg(
                            vertices=inputs["mesh"]["initial_positions"][k],
                            triangles=inputs["mesh"]["triangles"],
                            visual_material=VisualMaterialCfg(
                                uid=f"cloth_{k}",
                                base_color=[*cloth["colors"][k], 1],
                                roughness=0.8,
                            ),
                        ),
                        particle_radius=cloth["particle_radius"],
                        attrs=SurfaceDeformablePhysicsCfg(
                            density=cloth["surface_density"],
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
                            contact_material={
                                "stiffness": contact["particle_ke"],
                                "damping": contact["particle_kd"],
                                "dynamic_friction": contact["particle_friction"],
                            },
                        ),
                    )
                )
            )
        if not args.no_video:
            sim.add_light(
                LightCfg(
                    uid="key",
                    light_type="point",
                    init_pos=(1, -1, 3),
                    intensity=70,
                    enable_shadow=False,
                )
            )
            for name, ((w, h), eye, target) in CAMERAS.items():
                focal = h / 2 / np.tan(np.deg2rad(48) / 2)
                cameras.append(
                    sim.add_sensor(
                        CameraCfg(
                            uid=name,
                            width=w,
                            height=h,
                            near=0.01,
                            far=10.0,
                            intrinsics=(focal, focal, w / 2, h / 2),
                            enable_depth=False,
                            enable_mask=False,
                            extrinsics=CameraCfg.ExtrinsicsCfg(
                                eye=eye, target=target, up=(0, 0, 1)
                            ),
                        )
                    )
                )
        sim.prepare()
        ids = [robot.joint_names.index(n) for n in inputs["names"]]
        start = torch.as_tensor(
            inputs["targets"][:1], dtype=torch.float32, device=sim.device
        )
        robot.set_qpos(start, joint_ids=ids, target=False)
        robot.set_qpos(start, joint_ids=ids, target=True)
        robot.clear_dynamics()
        for layer in layers:
            for material in layer.get_visual_material_inst():
                if material is not None:
                    material.mat.set_double_sided(True)
        for camera in cameras:
            w, h = camera.cfg.width, camera.cfg.height
            focal = h / 2 / np.tan(np.deg2rad(48) / 2)
            camera.set_intrinsics(
                torch.tensor(
                    [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]],
                    device=sim.device,
                    dtype=torch.float32,
                )
            )
        initial = np.stack([o.data.nodal_pos_w[0].cpu().numpy() for o in layers])
        inverse = np.stack(
            [
                sim.spawn_result.get_particle_set(f"arena_0/cloth_{k}")
                .get_inverse_masses()
                .numpy()
                for k in range(4)
            ]
        )
        audit = audit_layers(output, inputs, initial, inverse)
        tcp_local = np.array([*cfg["robot"]["tcp_offset"], 1])

        def readings() -> dict[str, np.ndarray]:
            return dict(
                time=np.asarray(sim.spawn_result.get_simulation_time()),
                qpos=robot.get_qpos()[0, ids].cpu().numpy(),
                particles=np.stack(
                    [o.data.nodal_pos_w[0].cpu().numpy() for o in layers]
                ),
                velocities=np.stack(
                    [o.data.nodal_vel_w[0].cpu().numpy() for o in layers]
                ),
                tcp=np.stack(
                    [
                        (
                            robot.get_link_pose(f"{side}_gripper_base", to_matrix=True)[
                                0
                            ]
                            .cpu()
                            .numpy()
                            @ tcp_local
                        )[:3]
                        for side in ("left", "right")
                    ]
                ),
            )

        def step(frame: int) -> None:
            sim.update(physics_dt=sol["dt"], step=1, render_final_step=False)

        def capture() -> list[np.ndarray]:
            with sim.render_frame():
                sim.sync_render_state()
                views = []
                for camera in cameras:
                    camera.update()
                    views.append(
                        camera.get_data()["color"][0, ..., :3].cpu().numpy().copy()
                    )
                return views

        def diagnostics() -> dict:
            d = dict(sim.spawn_result.get_solver_diagnostics())
            v = d["vbd"]
            assert d["vbd_preset"] == "surface-fast" and v["iterations"] == 8
            assert (
                v["particle_enable_self_contact"]
                and v["particle_enable_batched_jacobi"]
                and v["particle_chebyshev_enabled"]
                and v["multilevel_active"]
            )
            return dict(
                d,
                outer_steps_per_command=1,
                simulated_elapsed_s=d["sim_time"],
                vbd_iterations=v["iterations"],
                cameras={
                    c.cfg.uid: {
                        "intrinsics": c.get_intrinsics()[0].cpu().tolist(),
                        "pose": c.get_arena_pose(to_matrix=True)[0].cpu().tolist(),
                    }
                    for c in cameras
                },
            )

        if args.measure:
            import warp as wp
            from scripts.benchmark.tech_reports.physics.experiments.dexuni.m03_measurement import (
                record_episode,
            )

            record_episode(
                "embodichain",
                args.repeat,
                inputs,
                output,
                step,
                readings,
                lambda: wp.synchronize_device("cuda:0"),
                diagnostics,
                audit,
                time.perf_counter() - init_started,
            )
        else:
            record_preview(
                "embodichain",
                inputs,
                step,
                readings,
                None if args.no_video else capture,
                diagnostics,
                audit,
                Path(__file__),
            )

    finally:
        robot = None
        layers = []
        cameras = []
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
