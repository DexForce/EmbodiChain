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
"""Preview the approved W1 rigid/soft squeeze-grasp through EmbodiChain.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/embodichain/s04.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))

import numpy as np
from dexsim.engine.newton_physics import RealtimeJointTargetControl
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    LightCfg,
    MassPropertiesCfg,
    NewtonCollisionPropertiesCfg,
    NewtonJointDrivePropertiesCfg,
    NewtonPhysicsCfg,
    NewtonRigidBodyMaterialCfg,
    RenderCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    VolumeDeformableObjectCfg,
    VolumeDeformablePhysicsCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_common import (
    OUTPUT,
    lame_to_youngs,
    load_inputs,
    record_preview,
    static_boxes,
)

from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_measurement import (
    measurement_directory,
    record_measurement,
)

__all__ = ["main"]


def main() -> None:
    """Create all objects through SimulationManager and render a dynamic episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    output = (
        measurement_directory("embodichain", args.repeat)
        if args.measure
        else OUTPUT / "preview/embodichain"
    )
    started = time.perf_counter()
    inputs = load_inputs()
    cfg, mesh, names = inputs["config"], inputs["mesh"], inputs["names"]
    width, height = cfg["camera"]["resolution"]
    sim = SimulationManager(
        SimulationManagerCfg(
            num_envs=1,
            headless=True,
            device="cuda:0",
            width=width,
            height=height,
            enable_entity_gizmo=False,
            robot_ik_gizmo=None,
            render_cfg=(
                RenderCfg(renderer="no-render")
                if args.measure
                else RenderCfg(renderer="rt", spp=4)
            ),
            physics_cfg=NewtonPhysicsCfg(
                device="cuda:0",
                physics_dt=cfg["dt"],
                num_substeps=cfg["substeps"],
                use_cuda_graph=cfg["use_cuda_graph"],
                collision_cfg=None,
                solver_cfg={
                    "solver_type": "dexuni",
                    "request_body_acceleration": False,
                    "mujoco_articulation_paths": ["arena_0/w1"],
                    "joint_mode": "dynamic",
                    "contact_mode": "full",
                    "coupling": "two_way",
                    "coupling_options": {
                        "mass_scale": cfg["proxy_mass_scale"],
                        "proxy_relaxation": cfg["proxy_relaxation"],
                    },
                    "vbd_options": {
                        "iterations": cfg["vbd_iterations"],
                        "particle_enable_uniform_tet_specialization": True,
                        "friction_epsilon": 1e-4,
                        "rigid_contact_hard": False,
                        "rigid_contact_history": False,
                        "rigid_body_contact_buffer_size": 4096,
                    },
                    "collision_options": {
                        "broad_phase": "nxn",
                        "include_static_kinematic_pairs": False,
                        "soft_contact_margin": 0.003,
                    },
                },
            ),
        )
    )
    fingers = [name for name in names if "FINGER" in name]

    def attrs(mu: float, density: float | None = None) -> RigidBodyPhysicsCfg:
        return RigidBodyPhysicsCfg(
            mass_props=(
                MassPropertiesCfg(density=density, recompute_inertia=True)
                if density
                else None
            ),
            collision_props=NewtonCollisionPropertiesCfg(gap=cfg["shape_gap"]),
            material_props=NewtonRigidBodyMaterialCfg(
                ke=cfg["shape_ke"], kd=cfg["shape_kd"], dynamic_friction=mu
            ),
        )

    def material(name: str, color: tuple) -> VisualMaterialCfg:
        return VisualMaterialCfg(uid=name, base_color=[*color, 1.0])

    try:
        robot = sim.add_articulation(
            ArticulationCfg(
                uid="w1",
                fpath=str(OUTPUT / "inputs" / cfg["robot_urdf"]),
                build_pk_chain=False,
                asset_physics_mode="overlay",
                attrs=attrs(1.0),
                root_props=ArticulationRootPropertiesCfg(
                    fixed_base=True,
                    newton_gravity_compensation=1.0,
                    newton_mimic_stabilizer=False,
                ),
                joint_drive_props=NewtonJointDrivePropertiesCfg(
                    target_mode="position_velocity",
                    stiffness={
                        name: cfg["finger_kp"] if name in fingers else cfg["arm_kp"]
                        for name in names
                    },
                    damping={
                        name: cfg["finger_kd"] if name in fingers else cfg["arm_kd"]
                        for name in names
                    },
                    armature={name: cfg["finger_armature"] for name in fingers},
                    passive_damping={
                        name: cfg["finger_passive_damping"] for name in fingers
                    },
                ),
            )
        )
        (control,) = sim.add_runtime_control(
            lambda arena, _: RealtimeJointTargetControl(
                f"{arena}/w1",
                inputs["targets"][0],
                joint_names=names,
                initialize_pose=True,
            )
        )
        for name, center, size in static_boxes():
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid=name,
                    shape=CubeCfg(
                        size=list(size), visual_material=material(name, (0.5, 0.5, 0.5))
                    ),
                    body_type="static",
                    init_pos=center,
                    attrs=attrs(0.8),
                )
            )
        rigid = sim.add_rigid_object(
            RigidObjectCfg(
                uid="rigid_block",
                shape=CubeCfg(
                    size=cfg["rigid_size"],
                    visual_material=material("rigid", (0.18, 0.55, 0.78)),
                ),
                body_type="dynamic",
                init_pos=tuple(cfg["rigid_center"]),
                attrs=attrs(1.0, cfg["rigid_density"]),
            )
        )
        youngs, poisson = lame_to_youngs(cfg["soft_mu"], cfg["soft_lambda"])
        soft = sim.add_deformable_object(
            VolumeDeformableObjectCfg(
                uid="soft_cube",
                shape=MeshCfg(
                    vertices=mesh["particle_q"],
                    triangles=mesh["tri_indices"],
                    visual_material=material("soft", (0.9, 0.4, 0.2)),
                ),
                tetrahedra=mesh["tet_indices"],
                particle_radius=cfg["particle_radius"],
                attrs=VolumeDeformablePhysicsCfg(
                    youngs=youngs,
                    poissons=poisson,
                    density=cfg["soft_density"],
                    elasticity_damping=cfg["soft_damping"],
                    contact_material={
                        "stiffness": cfg["particle_ke"],
                        "damping": cfg["particle_kd"],
                        "dynamic_friction": cfg["particle_mu"],
                    },
                ),
            )
        )
        camera = None
        if not args.measure:
            sim.add_light(
                LightCfg(
                    uid="key",
                    light_type="point",
                    init_pos=(1.5, -1.0, 3.0),
                    intensity=50.0,
                    enable_shadow=False,
                )
            )
            fx = width / 2 / np.tan(np.deg2rad(cfg["camera"]["hfov"]) / 2)
            camera = sim.add_sensor(
                CameraCfg(
                    uid="preview",
                    width=width,
                    height=height,
                    near=0.01,
                    far=10.0,
                    intrinsics=(fx, fx, width / 2, height / 2),
                    enable_depth=False,
                    enable_mask=False,
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=tuple(cfg["camera"]["eye"]),
                        target=tuple(cfg["camera"]["target"]),
                        up=(0, 0, 1),
                    ),
                )
            )
        sim.prepare()
        if not args.measure:
            import torch

            camera.set_intrinsics(
                torch.tensor(
                    [[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]],
                    device=sim.device,
                )
            )
        ids = [robot.joint_names.index(name) for name in names]
        # Articulation.reset() during prepare() writes cfg.init_qpos (zeros by
        # default) over the control's initial pose, so the arms would start at
        # zero and be pulled to frame 0 by the drives. Start at frame 0 instead,
        # as Isaac Lab does with write_joint_position_to_sim.
        import torch

        start = torch.as_tensor(
            inputs["targets"][0], dtype=torch.float32, device=sim.device
        ).unsqueeze(0)
        robot.set_qpos(start, joint_ids=ids, target=False)
        robot.set_qpos(start, joint_ids=ids, target=True)
        robot.clear_dynamics()
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend

        backend = get_newton_backend(sim.get_world())
        model = backend.model
        audit = output
        audit.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            audit / "model.npz",
            **{
                name: getattr(model, name).numpy()
                for name in (
                    "particle_mass",
                    "particle_q",
                    "particle_radius",
                    "tet_indices",
                    "tet_materials",
                    "tri_indices",
                    "joint_target_ke",
                    "joint_target_kd",
                    "joint_armature",
                    "joint_effort_limit",
                    "joint_damping",
                    "body_mass",
                    "body_inertia",
                )
                if hasattr(getattr(model, name, None), "numpy")
            },
        )

        def set_target(frame: int) -> None:
            control.set_target(inputs["targets"][frame])

        def advance() -> None:
            sim.update(physics_dt=cfg["dt"], step=1, render_final_step=not args.measure)

        def step(frame: int) -> None:
            set_target(frame)
            advance()

        def readings() -> dict[str, np.ndarray]:
            return {
                "qpos": robot.get_qpos()[0, ids].cpu().numpy(),
                "particles": soft.data.nodal_pos_w[0].cpu().numpy(),
                "rigid_pose": rigid.get_local_pose(to_matrix=True)[0].cpu().numpy(),
                "tcp": np.stack(
                    [
                        (
                            robot.get_link_pose(f"{side}_gripper_base", to_matrix=True)[
                                0
                            ]
                            .cpu()
                            .numpy()
                            @ np.array([0, 0, 0.125, 1])
                        )[:3]
                        for side in ["left", "right"]
                    ]
                ),
            }

        def capture() -> np.ndarray:
            sim.sync_render_state()
            camera.update()
            return camera.get_data()["color"][0, ..., :3].cpu().numpy()

        def diagnostics() -> dict:
            result = sim.spawn_result.get_solver_diagnostics()
            return dict(
                result,
                outer_steps_per_command=1,
                simulated_elapsed_s=result["sim_time"],
            )

        if args.measure:
            import warp as wp

            record_measurement(
                "embodichain",
                output,
                inputs,
                set_target,
                advance,
                readings,
                lambda: wp.synchronize_device("cuda:0"),
                diagnostics,
                time.perf_counter() - started,
            )
        else:
            record_preview(
                "embodichain",
                inputs,
                step,
                readings,
                capture,
                lambda: dict(
                    diagnostics(),
                    camera_intrinsics=camera.get_intrinsics()[0].cpu().tolist(),
                    camera_pose=camera.get_arena_pose(to_matrix=True)[0].cpu().tolist(),
                ),
            )
    finally:
        sim.destroy(exit_process=False)


if __name__ == "__main__":
    main()
