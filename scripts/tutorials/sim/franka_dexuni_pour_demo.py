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
"""Drive a live Franka grasp/pour/place with DexUni and one-way DFSPH.

DexUni advances the MuJoCo-owned robot and cup and then solves the water
against their live boundaries. Coupling is one-way.

Defaults: DFSPH at 480 Hz (two steps per 1/240 s physics step) with 15 MuJoCo
substeps each (7,200 Hz). The task config supplies the cup contact settings.
Requires a DexSim build with ``DexUniSolverCfg.sph_cfg``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
import torch

from embodichain.lab.scripts.run_fluid_task_program import (
    _configure,
    _cup_qualification,
)
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import (
    AxisAlignAffordance,
    ControlPartCommandProfile,
    GraspGoal,
    HeldObjectPoseGoal,
    MoveHeldObjectOptions,
    MotionPolicy,
    ObjectSemantics,
    PickUpOptions,
    PlaceGoal,
    PlaceOptions,
    PourGoal,
    PourOptions,
    create_simulation_atomic_action_engine,
)
from embodichain.lab.sim.motion.motion_generator import MotionGenerator, MotionGenCfg
from embodichain.lab.sim.motion.planners import ToppraPlannerCfg
from embodichain.lab.sim.sensors import CameraCfg

__all__ = ["main"]

ROOT = Path(__file__).resolve().parents[3]
TASK = (
    ROOT
    / "extensions/embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/task.franka.fluid.yaml"
)


def _use_dexuni(cfg, args) -> None:
    """Apply the requested DexUni and DFSPH substep counts."""
    physics = cfg.sim_cfg.physics_cfg
    solver = dict(physics.solver_cfg)
    physics.num_substeps = args.fluid_substeps
    solver["sph_rigid_substeps"] = args.rigid_substeps
    physics.solver_cfg = solver


def main() -> None:
    """Plan fresh joint commands and solve the robot, cup and fluid online."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/fluid-dexuni-online"
    )
    parser.add_argument("--spacing", type=float, default=0.0015)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--renderer", choices=["fast-rt", "no-render"], default="fast-rt"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--fluid-substeps",
        type=int,
        default=2,
        help="DFSPH steps per 1/240 s physics step",
    )
    parser.add_argument(
        "--rigid-substeps",
        type=int,
        default=15,
        help="MuJoCo substeps per DFSPH step",
    )
    args = parser.parse_args()
    if args.fluid_substeps < 1 or args.rigid_substeps < 1:
        parser.error("substep counts must be positive")
    if args.seconds is not None and (
        not np.isfinite(args.seconds) or args.seconds <= 0
    ):
        parser.error("seconds must be finite and positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _, cfg = _configure(
        TASK,
        num_envs=args.num_envs,
        renderer=args.renderer,
        output=output,
        particle_spacing=args.spacing,
    )
    _use_dexuni(cfg, args)
    cfg.sim_cfg.num_envs = cfg.num_envs
    sim = SimulationManager(cfg.sim_cfg)
    writers = []
    robot = cup = camera = engine = None
    try:
        robot = sim.add_robot(cfg.robot)
        for object_cfg in [*cfg.background, *cfg.rigid_object]:
            sim.add_rigid_object(object_cfg)
        for fluid_cfg in cfg.fluid_object:
            sim.add_fluid_object(fluid_cfg)
        if sim.has_native_renderer:
            for light_cfg in cfg.light.direct:
                sim.add_light(light_cfg)
        sim.prepare()
        robot.build_pk_serial_chain()
        sim.reset_objects_state()
        cup = sim.get_rigid_object("cup")
        # Live gravity/contact settling before observing the planning scene.
        sim.update(step=24, render_final_step=False)
        initial_pose = cup.get_local_pose(to_matrix=True).clone()
        fill_pose = cup.get_local_pose().clone()
        fill_pose[:, :3] += sim.arena_offsets
        sim.get_fluid_object("water").reset_to_pose(fill_pose)
        object_to_eef = torch.tensor(
            [[0, 0, -1, 0.065], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
            device=sim.device,
        )
        semantics = ObjectSemantics(
            entity_id="cup",
            label="cup",
            geometry={},
            affordance=AxisAlignAffordance(
                mesh_vertices=cup.get_vertices(env_ids=[0], scale=True)[0],
                mesh_triangles=cup.get_triangles(env_ids=[0])[0],
                internal_axis=torch.tensor([1.0, 0, 0], device=sim.device),
            ),
        )
        motion = MotionGenerator(
            MotionGenCfg(
                planner_cfg=ToppraPlannerCfg(
                    robot_uid=robot.uid, sim_instance_id=sim.instance_id
                )
            )
        )
        engine = create_simulation_atomic_action_engine(
            motion,
            (cup,),
            control_profiles={
                "eef": ControlPartCommandProfile.joint_positions(
                    open=torch.tensor([0.04, 0.04], device=sim.device),
                    grasp=torch.zeros(2, device=sim.device),
                )
            },
        )
        bindings = {"primary": {"motion": "arm", "grasp": "eef"}}
        policy = MotionPolicy(
            strategy="ik_interp", sample_count=160, dynamic_collision_mode="off"
        )
        receiver_pose = sim.get_rigid_object("receiver").get_local_pose(to_matrix=True)
        receiver_to_cup = torch.eye(4, device=sim.device)
        receiver_to_cup[:3, 3] = torch.tensor([0.005, 0.09, 0.17], device=sim.device)
        pour_pose = receiver_pose @ receiver_to_cup
        invocations = (
            engine.make_invocation(
                "pick_up",
                GraspGoal(semantics),
                control_parts=bindings,
                motion_policy=policy,
                skill_options=PickUpOptions(
                    fixed_object_to_eef=object_to_eef,
                    approach_direction=torch.tensor([-1.0, 0, 0], device=sim.device),
                    pre_grasp_distance=0.08,
                    lift_height=0.12,
                    hand_interp_steps=20,
                    grasp_settle_steps=20,
                ),
            ),
            engine.make_invocation(
                "move_held_object",
                HeldObjectPoseGoal(pour_pose),
                control_parts=bindings,
                motion_policy=policy,
                skill_options=MoveHeldObjectOptions(),
            ),
            engine.make_invocation(
                "pour",
                PourGoal(),
                control_parts=bindings,
                motion_policy=policy,
                skill_options=PourOptions(rotate_angle=np.deg2rad(80)),
            ),
            engine.make_invocation(
                "place",
                PlaceGoal(initial_pose @ object_to_eef),
                control_parts=bindings,
                motion_policy=policy,
                skill_options=PlaceOptions(
                    hand_interp_steps=20, release_settle_steps=20
                ),
            ),
        )
        control_dt = cfg.sim_steps_per_control * sim.sim_config.physics_dt
        plan = engine.compile(
            invocations, engine.initial_context(control_dt=control_dt)
        )
        if not bool(plan.plan_success.all()):
            raise RuntimeError("Atomic Action planning failed.")
        if args.renderer != "no-render":
            params = cfg.events.record_camera.params
            camera = sim.add_sensor(
                CameraCfg(
                    uid="fluid_sim_record",
                    width=params["resolution"][0],
                    height=params["resolution"][1],
                    intrinsics=params["intrinsics"],
                    visualization_role="record",
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=params["eye"], target=params["target"], up=params["up"]
                    ),
                )
            )
            writers = [
                imageio.get_writer(
                    output / f"fluid_sim_env{i}.mp4",
                    fps=1 / control_dt,
                    codec="libx264",
                    quality=8,
                    macro_block_size=1,
                )
                for i in range(sim.num_envs)
            ]
            if not args.headless:
                sim.open_window()
        positions = plan.trajectory.positions
        poses = [cup.get_local_pose().cpu().numpy().copy()]
        samples = []
        started = time.perf_counter()
        count = positions.shape[1] + round(2 / control_dt)
        if args.seconds is not None:
            count = min(count, int(np.ceil(args.seconds / control_dt)))
        for i in range(count):
            # Set targets only. Every measured joint/cup pose comes from physics.
            robot.set_qpos(positions[:, min(i, positions.shape[1] - 1)], target=True)
            sim.update(step=cfg.sim_steps_per_control, render_final_step=False)
            poses.append(cup.get_local_pose().cpu().numpy().copy())
            if camera is not None:
                with sim.render_frame():
                    sim.sync_render_state()
                    camera.update()
                    rgb = camera.get_data()["color"][..., :3].cpu().numpy()
                for env_id, writer in enumerate(writers):
                    writer.append_data(rgb[env_id])
            if i % round(1 / control_dt) == 0 or i == count - 1:
                fluid = sim.get_fluid_object("water")
                state = fluid.data.particle_state_w
                if not bool(torch.isfinite(state).all()):
                    raise RuntimeError("Non-finite online fluid state.")
                local = state[..., :3] - sim.arena_offsets[:, None, :]
                receiver_pose = sim.get_rigid_object("receiver").get_local_pose(
                    to_matrix=True
                )
                receiver = torch.bmm(
                    local - receiver_pose[:, None, :3, 3], receiver_pose[:, :3, :3]
                )
                received = (
                    (receiver[..., 2] > 0.0045)
                    & (receiver[..., 2] < 0.0675)
                    & (
                        torch.linalg.vector_norm(
                            receiver - torch.tensor([0, 0, 0.088], device=sim.device),
                            dim=-1,
                        )
                        < 0.093
                    )
                )
                sample = {
                    "time_s": (i + 1) * control_dt,
                    "cup_pose": poses[-1].tolist(),
                    "received_ml": (received.sum(dim=1) * args.spacing**3 * 1e6)
                    .cpu()
                    .tolist(),
                }
                samples.append(sample)
                print(json.dumps(sample), flush=True)
        report = {
            "coupling": "online one-way rigid-to-fluid",
            "controller": "fresh Atomic Action joint targets",
            "cup": _cup_qualification(poses),
            "num_envs": sim.num_envs,
            "particle_count_per_env": sim.get_fluid_object("water").num_particles,
            "control_dt": control_dt,
            "simulated_seconds": count * control_dt,
            "wall_seconds": time.perf_counter() - started,
            "samples": samples,
        }
        report["accepted"] = bool(
            all(report["cup"]["accepted"])
            and all(v >= 50 for v in samples[-1]["received_ml"])
        )
        (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["cup"]), flush=True)
        if args.seconds is None and not report["accepted"]:
            raise RuntimeError(
                "Online Sim pouring failed physical qualification; inspect validation.json."
            )
    finally:
        for writer in writers:
            writer.close()
        robot = cup = camera = engine = None
        sim.destroy(exit_process=False)
        del sim
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
