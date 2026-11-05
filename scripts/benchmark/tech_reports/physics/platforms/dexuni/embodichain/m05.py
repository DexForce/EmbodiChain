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
"""Replay accepted M05 through EmbodiChain with a real sealed gas cavity.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/embodichain/m05.py
No performance measurements. --no-video --out <scratch> supports diagnostics.
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
from dexsim.engine.newton_physics import KinematicJointTrajectoryControl
from dexsim.engine.newton_physics.backend_registry import get_newton_backend
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    LightCfg,
    MassPropertiesCfg,
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
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m05_common import OUTPUT
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m05_replay import (
    load_inputs,
    record_preview,
)

__all__ = ["main"]


def main() -> None:
    """Build the exact frozen mesh, material schedules and named joint replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--until", type=float)
    args = parser.parse_args()
    inputs = load_inputs()
    scene = inputs["manifest"]["scene"]
    contact = scene["contact"]
    membrane = scene["bag_membrane"]
    joints, mesh = inputs["joints"], inputs["mesh"]
    camera_spec = inputs["manifest"]["camera"]
    width, height = camera_spec["resolution"]
    output = args.out or OUTPUT / "preview/embodichain"
    if args.until is not None and args.out is None:
        parser.error("Truncated diagnostics require a separate --out scratch directory")
    native_solver = inputs["manifest"]["effective"]
    solver_cfg = dict(
        solver_type="dexuni",
        mujoco_articulation_paths=("arena_0/w1",),
        joint_mode="kinematic",
        contact_mode="full",
        coupling="one_way",
        particle_coloring_include_bending=True,
        particle_coloring_balance_colors=True,
        full_surface_contact_targets={
            "arena_0/w1": tuple(contact["full_surface_contact_links"]),
            "arena_0/rigid_block": None,
            "arena_0/table": None,
        },
        vbd_options=dict(
            pneumatic_enable_incremental_volume=True,
            iterations=scene["solver"]["iterations"],
            friction_epsilon=contact["friction_epsilon"],
            rigid_body_contact_buffer_size=4096,
            rigid_body_particle_contact_buffer_size=8192,
            particle_enable_self_contact=False,
            rigid_contact_hard=False,
            rigid_contact_history=False,
            rigid_contact_stick_freeze_translation_eps=0.0,
            rigid_contact_stick_freeze_angular_eps=0.0,
        ),
        collision_options=dict(
            broad_phase="nxn",
            soft_contact_margin=contact["soft_contact_margin"],
            include_static_kinematic_pairs=False,
            enable_rigid_soft_full_surface_contact=True,
            contact_matching="disabled",
        ),
    )
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
                physics_dt=native_solver["dt"],
                num_substeps=scene["solver"]["substeps"],
                use_cuda_graph=True,
                collision_cfg=None,
                solver_cfg=solver_cfg,
            ),
        )
    )

    def attrs(
        ke: float, mu: float, density: float | None = None
    ) -> RigidBodyPhysicsCfg:
        return RigidBodyPhysicsCfg(
            collision_props=NewtonCollisionPropertiesCfg(
                margin=contact["margin"], gap=contact["gap"]
            ),
            material_props=NewtonRigidBodyMaterialCfg(
                ke=ke, kd=contact["particle_kd"], dynamic_friction=mu
            ),
            mass_props=None if density is None else MassPropertiesCfg(density=density),
        )

    robot = bag = block = camera = None
    try:
        robot = sim.add_articulation(
            ArticulationCfg(
                uid="w1",
                fpath=str(OUTPUT / "inputs/w1_v030/robot.urdf"),
                init_pos=tuple(scene["robot_base_position"]),
                asset_physics_mode="overlay",
                build_pk_chain=False,
                attrs=attrs(contact["robot_ke_other_shapes"], contact["hand_mu"]),
                root_props=ArticulationRootPropertiesCfg(
                    fixed_base=True,
                    newton_mimic_stabilizer=False,
                    newton_force_sdf=contact["robot_force_sdf"],
                ),
            )
        )
        sim.add_runtime_control(
            lambda arena, index: KinematicJointTrajectoryControl(
                f"{arena}/w1",
                joints["targets"],
                fps=60,
                joint_names=joints["joint_names"].tolist(),
                initialize_pose=True,
            )
        )
        for name, center, size in scene["supports"]:
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid=name,
                    shape=CubeCfg(
                        size=size,
                        visual_material=VisualMaterialCfg(
                            uid=name, base_color=[0.5, 0.5, 0.5, 1.0]
                        ),
                    ),
                    body_type="static",
                    init_pos=center,
                    attrs=attrs(contact["particle_ke"], contact["table_mu"]),
                )
            )
        size = scene["block_size_m"]
        block = sim.add_rigid_object(
            RigidObjectCfg(
                uid="rigid_block",
                shape=CubeCfg(
                    size=[size] * 3,
                    visual_material=VisualMaterialCfg(
                        uid="rigid_block", base_color=[0.18, 0.55, 0.78, 1.0]
                    ),
                ),
                body_type="dynamic",
                init_pos=(
                    *scene["block_start_xy"],
                    scene["table_z"] + size / 2 + 2 * contact["margin"],
                ),
                attrs=attrs(
                    scene["block_ke"],
                    scene["block_mu"],
                    scene["block_mass_kg"] / size**3,
                ),
            )
        )
        bag = sim.add_deformable_object(
            SurfaceDeformableObjectCfg(
                uid="bag",
                shape=MeshCfg(
                    vertices=mesh["vertices"],
                    triangles=mesh["triangles"],
                    visual_material=VisualMaterialCfg(
                        uid="bag", base_color=[0.86, 0.68, 0.34, 1.0], roughness=0.62
                    ),
                ),
                particle_radius=membrane["particle_radius"],
                validate_mesh=True,
                attrs=SurfaceDeformablePhysicsCfg(
                    density=membrane["surface_density"],
                    surface_props={
                        key: membrane[key]
                        for key in ("tri_ke", "tri_ka", "tri_kd", "edge_ke", "edge_kd")
                    },
                    contact_material={
                        "stiffness": contact["particle_ke"],
                        "damping": contact["particle_kd"],
                        "dynamic_friction": contact["particle_mu"],
                    },
                    pneumatic={
                        **scene["bag_gas"],
                        "mode": scene["bag_gas"]["mode"].lower(),
                    },
                ),
            )
        )
        sim.register_contact_material_schedule(
            "w1",
            {"stiffness": ((0.0, contact["hand_ke_left_fingers"]),)},
            link_names=("left_finger1", "left_finger2"),
        )
        release_track = (
            (0.0, contact["particle_mu"]),
            (contact["release_time_s"], contact["release_friction"]),
        )
        sim.register_contact_material_schedule(
            "w1",
            {
                "dynamic_friction": release_track,
                "stiffness": ((0.0, contact["hand_ke_right_fingers"]),),
            },
            link_names=("right_finger1", "right_finger2"),
        )
        sim.register_particle_contact_material_schedule(
            {
                "stiffness": ((0.0, contact["particle_ke"]),),
                "damping": ((0.0, contact["particle_kd"]),),
                "dynamic_friction": release_track,
            },
            uid="bag",
        )
        # The accepted native worker passes its computed "vfov" to the SDK's
        # horizontal-FOV API. Match that actual framing through public K I/O;
        # OfflineRT's projection getter retains its initial matrix after FOV I/O.
        native_fov = 2 * np.arctan(
            np.tan(np.deg2rad(camera_spec["hfov"]) / 2) * height / width
        )
        focal = width / 2 / np.tan(native_fov / 2)
        if not args.no_video:
            sim.add_light(
                LightCfg(
                    uid="key_light",
                    intensity=50.0,
                    color=(1, 1, 1),
                    init_pos=(1.5, -1.0, 3.0),
                    light_type="point",
                    enable_shadow=False,
                )
            )
            camera = sim.add_sensor(
                CameraCfg(
                    uid="preview",
                    width=width,
                    height=height,
                    near=0.01,
                    far=10.0,
                    intrinsics=(focal, focal, width / 2, height / 2),
                    enable_depth=False,
                    enable_mask=False,
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=camera_spec["eye"],
                        target=camera_spec["target"],
                        up=(0, 0, 1),
                    ),
                )
            )
        sim.prepare()
        sim.set_ground_plane_visibility(False)
        ids = [robot.joint_names.index(name) for name in joints["joint_names"]]
        initial_q = torch.as_tensor(
            joints["targets"][:1], dtype=torch.float32, device=sim.device
        )
        robot.set_qpos(initial_q, joint_ids=ids, target=False)
        robot.set_qpos(initial_q, joint_ids=ids, target=True)
        robot.clear_dynamics()
        for material in bag.get_visual_material_inst():
            if material is not None:
                material.mat.set_double_sided(True)
        if camera is not None:
            camera.set_intrinsics(
                torch.tensor(
                    [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
                    dtype=torch.float32,
                    device=sim.device,
                )
            )
        initial = bag.data.nodal_pos_w[0].cpu().numpy()
        inverse = (
            sim.spawn_result.get_particle_set("arena_0/bag")
            .get_inverse_masses()
            .numpy()
        )
        if not np.allclose(initial, mesh["vertices"], atol=1e-5, rtol=0):
            raise RuntimeError("M05 authored node order or initial position changed")
        if not np.isfinite(inverse).all() or np.any(inverse <= 0):
            raise RuntimeError("All M05 bag nodes must remain dynamic")
        backend = get_newton_backend(sim.get_world())
        pipeline = backend.solver.backend.pipeline
        face_pairs = pipeline.soft_face_rigid_pairs.numpy()
        edge_pairs = pipeline.soft_edge_rigid_pairs.numpy()
        selected = set(face_pairs[:, 1]) | set(edge_pairs[:, 1])
        full_surface = {}
        for name in ("table", "rigid_block", *contact["full_surface_contact_links"]):
            found = {
                i for i, label in enumerate(pipeline.model.shape_label) if name in label
            }
            if not found or not found <= selected:
                raise RuntimeError(f"Missing M05 full-surface contact for {name}")
            full_surface[name] = True
        model = backend.solver.model
        gas_model = model.pneumatic
        gas_audit = {
            name: getattr(gas_model, name).numpy()
            for name in (
                "mode",
                "rest_volume",
                "reference_absolute_pressure",
                "ambient_pressure",
                "bulk_damping",
                "max_absolute_pressure",
            )
        }
        for name in (
            "reference_absolute_pressure",
            "ambient_pressure",
            "bulk_damping",
            "max_absolute_pressure",
        ):
            if not np.allclose(gas_audit[name], scene["bag_gas"][name], rtol=1e-6):
                raise RuntimeError(f"M05 effective gas parameter changed: {name}")
        if not np.allclose(gas_audit["rest_volume"], mesh["rest_volume"], rtol=1e-5):
            raise RuntimeError("M05 effective reference cavity volume changed")
        if not np.array_equal(gas_audit["mode"], [0]):
            raise RuntimeError("M05 effective gas law must be isothermal")
        effective_mass = block.get_mass().cpu().numpy().reshape(-1)
        if len(effective_mass) != 1 or not np.allclose(
            effective_mass, scene["block_mass_kg"]
        ):
            raise RuntimeError(
                f"M05 dynamic block mass changed: {effective_mass.tolist()}"
            )
        audit = dict(
            bag_nodes=len(inverse),
            bag_mass_kg=float((1 / inverse).sum()),
            pinned_nodes=0,
            full_surface_contact_verified=full_surface,
            face_candidate_pairs=len(face_pairs),
            edge_candidate_pairs=len(edge_pairs),
            block_initial_pose=block.get_local_pose(to_matrix=True)[0].cpu().numpy(),
            dexsim_version=dexsim.__version__,
            newton_version=newton.__version__,
            sdf_source_request=True,
            initialized_full_surface_pipeline=True,
            pneumatic=gas_audit,
            effective_block_mass_kg=float(effective_mass[0]),
            render_camera=dict(
                reference_nominal_hfov_deg=camera_spec["hfov"],
                effective_hfov_deg=float(np.rad2deg(native_fov)),
                focal_pixels=float(focal),
                ground_plane_visible=False,
                scope="Match accepted native horizontal-FOV API framing",
            ),
        )
        # Frozen demo's dexuni_w1_two_way_utils.TCP, in the gripper base frame.
        tcp_local = np.array((0.0, 0.0, 0.125, 1.0))

        def readings() -> dict[str, np.ndarray]:
            hand = [
                robot.get_link_pose(name, to_matrix=True)[0].cpu().numpy()
                for name in ("left_gripper_base", "left_finger1", "left_finger2")
            ]
            right = (
                robot.get_link_pose("right_gripper_base", to_matrix=True)[0]
                .cpu()
                .numpy()
            )
            return dict(
                time=np.asarray(sim.spawn_result.get_simulation_time()),
                qpos=robot.get_qpos()[0, ids].cpu().numpy(),
                bag=bag.data.nodal_pos_w[0].cpu().numpy(),
                block_pose=block.get_local_pose(to_matrix=True)[0].cpu().numpy(),
                tcp=np.stack([(hand[0] @ tcp_local)[:3], (right @ tcp_local)[:3]]),
                left_hand_poses=np.stack(hand),
            )

        def step() -> None:
            sim.update(physics_dt=native_solver["dt"], step=1, render_final_step=False)

        def capture() -> np.ndarray:
            with sim.render_frame():
                sim.sync_render_state()
                camera.update()
                return camera.get_data()["color"][0, ..., :3].cpu().numpy().copy()

        def diagnostics() -> dict[str, Any]:
            value = dict(sim.spawn_result.get_solver_diagnostics())
            value.update(
                dexsim_version=dexsim.__version__,
                newton_version=newton.__version__,
                outer_steps_per_command=1,
            )
            return value

        record_preview(
            "EmbodiChain",
            inputs,
            output,
            step,
            readings,
            None if args.no_video else capture,
            diagnostics,
            audit,
            until=args.until,
        )
    finally:
        robot = bag = block = camera = None
        backend = pipeline = model = gas_model = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
