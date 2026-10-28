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
"""Preview the approved W1 one-way single cloth fold (M02) through EmbodiChain.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/embodichain/m02.py
Replays the frozen DexSim joint trajectory kinematically on the same cloth mesh,
with the same DexUni settings as the approved run (soft contact, surface-fast).
Formal timing: add ``--measure --repeat N`` (no camera, no rendering).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))

import numpy as np
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
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m02_common import (
    OUTPUT,
    TABLE_Z,
    load_inputs,
    record_preview,
)

__all__ = ["main"]


def solver_cfg(cfg: dict) -> dict:
    """The DexUni solver settings of the approved native run."""
    if cfg["vbd_preset"] != "surface-fast" or cfg["contact_mode"] != "soft":
        raise ValueError("M02 was approved with soft contact and surface-fast")
    return {
        "solver_type": "dexuni",
        "mujoco_articulation_paths": ["arena_0/w1"],
        "joint_mode": cfg["joint_mode"],
        "contact_mode": cfg["contact_mode"],
        "coupling": cfg["coupling"],
        "vbd_preset": cfg["vbd_preset"],
        # No explicit iteration count: the preset sets it, as in the native run.
        "vbd_options": {
            "friction_epsilon": cfg["friction_epsilon"],
            "rigid_contact_hard": False,
            "rigid_contact_history": False,
            "rigid_body_contact_buffer_size": 4096,
            **cfg["self_contact"],
        },
        # The kinematic soft-only backend accepts only the contact margin.
        "collision_options": {"soft_contact_margin": cfg["soft_contact_margin"]},
    }


def main() -> None:
    """Create all objects through SimulationManager and record one episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    started = time.perf_counter()
    output = None
    if args.measure:
        from scripts.benchmark.tech_reports.physics.experiments.dexuni.m02_measurement import (
            measurement_directory,
        )

        output = measurement_directory("embodichain", args.repeat)
    inputs = load_inputs()
    cfg, mesh, names = inputs["config"], inputs["mesh"], inputs["names"]
    cloth_cfg = cfg["cloth"]
    camera_cfg = cfg["camera"]
    width, height = camera_cfg["resolution"]
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
                solver_cfg=solver_cfg(cfg),
            ),
        )
    )

    def attrs(mu: float) -> RigidBodyPhysicsCfg:
        return RigidBodyPhysicsCfg(
            collision_props=NewtonCollisionPropertiesCfg(
                margin=cfg["shape_margin"], gap=cfg["shape_gap"]
            ),
            material_props=NewtonRigidBodyMaterialCfg(
                ke=cfg["shape_ke"], kd=cfg["shape_kd"], dynamic_friction=mu
            ),
        )

    def material(name: str, color) -> VisualMaterialCfg:
        return VisualMaterialCfg(uid=name, base_color=[*color, 1.0])

    try:
        robot = sim.add_articulation(
            ArticulationCfg(
                uid="w1",
                fpath=str(OUTPUT / "inputs" / cfg["robot_urdf"]),
                build_pk_chain=False,
                asset_physics_mode="overlay",
                attrs=attrs(cfg["robot_mu"]),
                root_props=ArticulationRootPropertiesCfg(
                    fixed_base=True, newton_mimic_stabilizer=False
                ),
            )
        )
        sim.add_runtime_control(
            lambda arena, _: KinematicJointTrajectoryControl(
                f"{arena}/w1",
                inputs["targets"],
                fps=1 / cfg["dt"],
                joint_names=names,
                initialize_pose=True,
            )
        )
        boxes = [("table", (0.46, 0.0, TABLE_Z - 0.025), (0.6, 1.64, 0.05))] + [
            (f"leg_{i}_{j}", (x, y, (TABLE_Z - 0.05) / 2), (0.05, 0.05, TABLE_Z - 0.05))
            for i, x in enumerate((0.21, 0.71))
            for j, y in enumerate((-0.74, 0.74))
        ]
        for name, center, size in boxes:
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid=name,
                    shape=CubeCfg(
                        size=list(size), visual_material=material(name, (0.5, 0.5, 0.5))
                    ),
                    body_type="static",
                    init_pos=center,
                    attrs=attrs(cfg["static_mu"]),
                )
            )
        cloth = sim.add_deformable_object(
            SurfaceDeformableObjectCfg(
                uid="cloth",
                shape=MeshCfg(
                    vertices=mesh["vertices"],
                    triangles=mesh["triangles"],
                    visual_material=VisualMaterialCfg(
                        uid="cloth",
                        base_color=[*cloth_cfg["color"], 1.0],
                        roughness=0.7,
                    ),
                ),
                particle_radius=cloth_cfg["particle_radius"],
                attrs=SurfaceDeformablePhysicsCfg(
                    density=cloth_cfg["surface_density"],
                    surface_props={
                        key: cloth_cfg[key]
                        for key in (
                            "tri_ke",
                            "tri_ka",
                            "tri_kd",
                            "tri_drag",
                            "tri_lift",
                            "edge_ke",
                            "edge_kd",
                        )
                    },
                    contact_material={
                        "stiffness": cloth_cfg["contact_ke"],
                        "damping": cloth_cfg["contact_kd"],
                        "dynamic_friction": cloth_cfg["contact_mu"],
                    },
                ),
            )
        )
        camera = None
        fx = width / 2 / np.tan(np.deg2rad(camera_cfg["hfov"]) / 2)
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
                        eye=tuple(camera_cfg["eye"]),
                        target=tuple(camera_cfg["target"]),
                        up=(0, 0, 1),
                    ),
                )
            )
        sim.prepare()
        import torch

        if camera is not None:
            for inst in cloth.get_visual_material_inst():
                if inst is None:
                    raise RuntimeError("Cloth render material was not bound")
                inst.mat.set_double_sided(True)
            camera.set_intrinsics(
                torch.tensor(
                    [[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]],
                    device=sim.device,
                )
            )
        ids = [robot.joint_names.index(name) for name in names]
        # Articulation.reset() during prepare() writes cfg.init_qpos (zeros) over
        # the control's initial pose; start at frame 0 as the native run does.
        start = torch.as_tensor(
            inputs["targets"][0], dtype=torch.float32, device=sim.device
        ).unsqueeze(0)
        robot.set_qpos(start, joint_ids=ids, target=False)
        robot.set_qpos(start, joint_ids=ids, target=True)
        robot.clear_dynamics()
        initial = cloth.data.nodal_pos_w[0].cpu().numpy()
        if not np.allclose(initial, mesh["vertices"], atol=1e-6):
            raise RuntimeError("Cloth node order/positions differ from the frozen mesh")
        tcp_local = np.array([*cfg["tcp_offset"], 1.0])

        def readings() -> dict[str, np.ndarray]:
            return {
                "time": np.asarray(sim.spawn_result.get_simulation_time()),
                "qpos": robot.get_qpos()[0, ids].cpu().numpy(),
                "particles": cloth.data.nodal_pos_w[0].cpu().numpy(),
                "tcp": np.stack(
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
            }

        def step(frame: int) -> None:
            sim.update(physics_dt=cfg["dt"], step=1, render_final_step=False)
            qpos = robot.get_qpos()[0, ids].cpu().numpy()
            if not np.allclose(qpos, inputs["targets"][frame], atol=1e-5):
                raise RuntimeError(f"Kinematic joint tracking failed at frame {frame}")

        def capture() -> np.ndarray:
            with sim.render_frame():
                sim.sync_render_state()
                camera.update()
                return camera.get_data()["color"][0, ..., :3].cpu().numpy().copy()

        def diagnostics() -> dict:
            result = sim.spawn_result.get_solver_diagnostics()
            vbd = result["vbd"]
            if not (
                result["vbd_preset"] == "surface-fast"
                and vbd["particle_enable_batched_jacobi"]
                and vbd["particle_chebyshev_enabled"]
                and vbd["multilevel_active"]
                and vbd["particle_enable_self_contact"]
            ):
                raise RuntimeError("The surface-fast policy is not active")
            result = dict(
                result,
                outer_steps_per_command=1,
                simulated_elapsed_s=result["sim_time"],
                vbd_iterations=vbd["iterations"],
            )
            if camera is not None:
                result.update(
                    camera_intrinsics=camera.get_intrinsics()[0].cpu().tolist(),
                    camera_pose=camera.get_arena_pose(to_matrix=True)[0].cpu().tolist(),
                )
            return result

        if args.measure:
            import warp as wp
            from scripts.benchmark.tech_reports.physics.experiments.dexuni.m02_measurement import (
                record_episode,
            )

            record_episode(
                "embodichain",
                output,
                inputs,
                lambda: sim.update(
                    physics_dt=cfg["dt"], step=1, render_final_step=False
                ),
                readings,
                lambda: wp.synchronize_device("cuda:0"),
                diagnostics,
                time.perf_counter() - started,
            )
        else:
            record_preview("embodichain", inputs, step, readings, capture, diagnostics)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
