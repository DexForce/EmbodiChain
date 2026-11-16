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

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import pytest


def _check_place_keyframes(env, generator):
    from dataclasses import replace
    from embodichain.lab.sim.atomic_skills import (
        AtomicSkillEngine,
        ControlPartCommandProfile,
        MotionPolicy,
        SkillInvocation,
    )
    from embodichain.lab.sim.atomic_skills.primitives.place import (
        PlaceGoal,
        PlaceOptions,
    )
    from embodichain.lab.sim.atomic_skills.primitives._helpers import (
        split_joint_trajectory_at_pose,
    )
    from embodichain.lab.sim.atomic_skills.trajectory_ops import split_three_segments
    from embodichain.lab.sim.motion.planners.utils import PlanState
    import torch

    hand_limits = env.robot.get_qpos_limits(name="eef")[0]
    engine = AtomicSkillEngine(
        generator,
        env_ids=(0,),
        control_profiles={
            "eef": ControlPartCommandProfile.joint_positions(
                open=hand_limits[:, 0], grasp=hand_limits[:, 1]
            )
        },
    )
    binding = engine.bind_control_parts(
        "place", {"primary": {"motion": "arm", "grasp": "eef"}}
    )
    context = engine.initial_context(control_dt=0.04)
    arm_ids = env.robot.get_joint_ids("arm")
    start = context.robot.qpos[:, arm_ids]
    start_pose = env.robot.compute_fk(
        qpos=start, name="arm", to_matrix=True, env_ids=(0,)
    )
    release = start_pose.clone()
    release[:, 0, 3] += 0.025
    approach = release.clone()
    approach[:, 2, 3] += 0.03
    # Four explicit quarter-points belong to each original translation segment.
    points = []
    for first, last in (
        (start_pose, approach),
        (approach, release),
        (release, approach),
    ):
        for alpha in (0.25, 0.5, 0.75, 1.0):
            pose = last.clone()
            pose[:, :3, 3] = first[:, :3, 3] + alpha * (
                last[:, :3, 3] - first[:, :3, 3]
            )
            points.append(PlanState.from_xpos(pose))
    options = PlaceOptions(cartesian_waypoint_count=4, lift_height=0.03)
    for before, after in (("joint", "cartesian"), ("cartesian", "joint")):
        policy = MotionPolicy(
            strategy="ik_interp",
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            sample_count=125,
            velocity_limit=None,
            acceleration_limit=None,
            phase_paths={"pre_place": before, "place": after, "retreat": after},
        )
        invocation = SkillInvocation(
            skill_id="place",
            goal=PlaceGoal(release),
            binding=binding,
            motion_policy=policy,
            skill_options=options,
        )
        plan = engine.plan(invocation, context)
        down_count, _, back_count = split_three_segments(
            policy.sample_count,
            options.hand_interp_steps,
            first_segment_name="approach",
            third_segment_name="back",
        )
        reference_options = policy.to_motion_gen_options(
            start_qpos=start,
            control_part="arm",
            sample_count=down_count + back_count,
            interpolation_dt=context.control_dt,
        )
        reference_options.waypoint_modes = (before,) * 4 + (after,) * 8
        reference = generator.generate(points, options=reference_options)
        assert plan.success_all and reference.success.all()
        down, back = split_joint_trajectory_at_pose(
            reference.positions,
            release,
            robot=env.robot,
            control_part="arm",
            first_sample_count=down_count,
            second_sample_count=back_count,
            dt=reference.dt,
            control_dt=context.control_dt,
            arrival_indices=reference.waypoint_indices,
            success=reference.success,
            split_target_index=7,
            env_ids=(0,),
        )
        trajectory = plan.joint_trajectory
        for segment, expected in zip(
            (plan.segments[0], plan.segments[-1]), (down, back)
        ):
            torch.testing.assert_close(
                trajectory.positions[:, segment.start : segment.stop, arm_ids],
                expected,
                rtol=0,
                atol=1e-6,
            )
    with pytest.raises(ValueError, match="cartesian_approach and phase_paths"):
        engine.plan(
            replace(
                invocation, skill_options=replace(options, cartesian_approach=True)
            ),
            context,
        )


def _probe(output):
    import torch
    from embodichain.lab.expansion import MeasuredExpansionValidator
    from embodichain.lab.expansion.program import TaskProgramBatchExecutor
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import (
        REGISTERED_ENVS,
        discover_task_packages,
        execute_init_hooks,
    )
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.motion.motion_generator import (
        MotionGenerator,
        MotionGenCfg,
        MotionGenOptions,
    )
    from embodichain.lab.sim.motion.planners.trapezoidal_planner import (
        TrapezoidalPlannerCfg,
    )
    from embodichain.lab.sim.motion.planners.utils import PlanState
    from embodichain.lab.task_program.integrations.simulation.feasibility_geometry import (
        collision_bodies_from_simulation,
    )
    from embodichain.utils.utility import load_config

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml"
    )
    discover_task_packages()
    execute_init_hooks()
    cfg = config_to_cfg(load_config(path), source_path=path)
    cfg.num_envs = cfg.sim_cfg.num_envs = 2
    cfg.sim_cfg.device, cfg.sim_cfg.headless = "cpu", True
    cfg.sim_cfg.render_cfg.renderer = "no-render"
    cfg.sensor, cfg.enable_sensor, cfg.dataset = [], False, None
    cfg.init_rollout_buffer = cfg.record_trajectory = False
    cfg.task_program.program.count = 1
    env = REGISTERED_ENVS[load_config(path)["id"]].make(cfg=cfg)
    try:
        env.reset(seed=0)
        generator = MotionGenerator(
            MotionGenCfg(
                planner_cfg=TrapezoidalPlannerCfg(
                    robot_uid=env.robot.uid, sim_instance_id=env.sim.instance_id
                )
            ),
            env_ids=(0,),
        )
        start = env.robot.get_qpos(name="arm")[0:1].clone()
        states = []
        for index in range(3):
            goal = start.clone()
            goal[:, 0] += 0.001 * (index + 1)
            pose = env.robot.compute_fk(
                qpos=goal, name="arm", to_matrix=True, env_ids=(0,)
            )
            states.append(PlanState.from_xpos(pose))
        base = dict(
            strategy="ik_interp",
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            interpolation_dt=0.04,
            start_qpos=start,
            control_part="arm",
        )
        ordinary = generator.generate(states, MotionGenOptions(sample_count=25, **base))
        first = generator.generate(states[:1], MotionGenOptions(sample_count=9, **base))
        tail = generator.generate(
            states[1:],
            MotionGenOptions(
                sample_count=17,
                **{**base, "start_qpos": first.positions[:, -1], "is_linear": True},
            ),
        )
        before = env.sim.physics_step_index
        mixed = generator.generate(
            states,
            MotionGenOptions(
                sample_count=25,
                waypoint_modes=("joint", "cartesian", "cartesian"),
                **base,
            ),
        )
        assert mixed.success.all() and first.success.all() and tail.success.all()
        torch.testing.assert_close(
            mixed.positions,
            torch.cat((first.positions, tail.positions[:, 1:]), dim=1),
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            mixed.dt, torch.cat((first.dt, tail.dt[:, 1:]), dim=1), rtol=0, atol=0
        )
        after = generator.generate(states, MotionGenOptions(sample_count=25, **base))
        torch.testing.assert_close(
            ordinary.positions, after.positions, rtol=0, atol=1e-6
        )
        assert env.sim.physics_step_index == before
        _check_place_keyframes(env, generator)
        assert env.sim.physics_step_index == before
        program = env.compile_task_program(cfg.task_program)
        frames = sorted({b.frame_uid for b in collision_bodies_from_simulation(env)})
        value = float(env.robot.get_qpos()[0, 0])
        name = env.robot.joint_names[0]
        configuration = {
            "initial_checks": [
                {
                    "kind": "joint_range",
                    "joint": name,
                    "lower": value - 0.1,
                    "upper": value + 0.1,
                }
            ],
            "final_checks": [
                {
                    "kind": "joint_range",
                    "joint": name,
                    "lower": value + 1,
                    "upper": value + 2,
                }
            ],
            "events": [],
            "mounting_contacts": [],
            "allowed_contacts": [
                (a, b) for i, a in enumerate(frames) for b in frames[i + 1 :]
            ],
        }
        for backend in ("geometry", "native_contacts"):
            check = MeasuredExpansionValidator(
                env, configuration, Path(output) / backend, backend=backend
            )
            try:
                check.prepare()
                TaskProgramBatchExecutor(env, program).run(
                    (0, 1), max_steps=3, observe_step=check.sample
                )
                assert set(check.results) == set(check.terminal) == {0, 1}
                assert all(
                    row.status == "passed" for row in check.results.values()
                ), check.results
                peer = check.terminal[1]
                check.prepare(env_ids=(0,))
                assert check.terminal[1] is peer and 0 not in check.terminal
                if backend == "native_contacts":
                    assert check.rows[1].substeps > 0 and check.rows[0].substeps == 0
            finally:
                check.close()
    finally:
        env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.subprocess_sim
def test_explicit_phase_paths_and_selected_row_acceptance(tmp_path):
    result = subprocess.run(
        [sys.executable, __file__, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "SDL_VIDEODRIVER": "offscreen", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-5000:]


if __name__ == "__main__":
    _probe(sys.argv[1])
