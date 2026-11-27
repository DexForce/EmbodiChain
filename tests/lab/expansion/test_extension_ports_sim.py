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
from unittest.mock import patch
import pytest


def _recorded_grasp_labels(cache, _key, _urdf, vertices, _triangles, _signature):
    """Replay physical labels for these CPU expansion probes; keep live ranking and IK."""
    import torch
    import yaml

    path = Path(__file__).parent / "fixtures/validated_grasps.yaml"
    entry = yaml.safe_load(path.read_text())[cache.hand["root_link"]]
    torch.testing.assert_close(
        torch.stack((vertices.amin(0), vertices.amax(0))),
        vertices.new_tensor(entry["mesh_bounds"]),
        atol=1e-6,
        rtol=0,
    )
    return {
        field: torch.tensor(entry[field], dtype=torch.float32)
        for field in ("grasp_pose", "open_length")
    }


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
    for period, steps, expected in ((0.04, 0, 3), (0.02, 0, 6), (0.04, 5, 5)):
        timed = engine.plan(
            replace(
                invocation,
                skill_options=replace(
                    options, release_settle_duration=0.11, release_settle_steps=steps
                ),
            ),
            engine.initial_context(control_dt=period),
        )
        release = timed.segment("release")
        assert release.stop - release.start == options.hand_interp_steps + expected
        hand_ids = env.robot.get_joint_ids("eef")
        torch.testing.assert_close(
            timed.joint_trajectory.positions[
                :, release.stop - expected : release.stop, hand_ids
            ],
            hand_limits[None, :, 0].expand(1, expected, -1),
            rtol=0,
            atol=0,
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
        / "embodichain/extensions/tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml"
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
            selected = dict(configuration)
            if backend == "native_contacts":
                selected["contact_hold"] = {
                    "entity": "cube",
                    "links": ["gripper_finger1_link_1", "gripper_finger2_link_1"],
                    "duration_s": env.step_dt * 1.4,
                    "max_translation_drift_m": 0.001,
                    "max_rotation_drift_rad": 0.001,
                }
            check = MeasuredExpansionValidator(
                env, selected, Path(output) / backend, backend=backend
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
                if backend == "geometry":
                    from embodichain.lab.gym.envs.types import ControllerAction

                    # A real collider overlap in only one selected arena must
                    # not reject the safe peer, including reordered selection.
                    forbidden = {
                        **configuration,
                        "allowed_contacts": [
                            pair
                            for pair in configuration["allowed_contacts"]
                            if not (
                                "cube" in pair
                                and any(
                                    name.startswith(env.robot.uid + "/")
                                    for name in pair
                                )
                            )
                        ],
                    }
                    measured = MeasuredExpansionValidator(
                        env,
                        forbidden,
                        Path(output) / "geometry-selected",
                        backend="geometry",
                    )
                    cube = env.sim.get_rigid_object("cube")
                    original = cube.get_local_pose(to_matrix=True)[1:2].clone()
                    try:
                        measured.prepare()
                        body = next(
                            b
                            for b in collision_bodies_from_simulation(env, env_id=1)
                            if b.robot and b.kind != "halfspace"
                        )
                        link = body.frame_uid.split("/", 1)[1]
                        pose = env.robot.get_link_pose(link, to_matrix=True)[1:2]
                        pose = pose @ torch.tensor(body.local_pose, dtype=pose.dtype)
                        before_query = env.sim.physics_step_index
                        cube.set_local_pose(pose, env_ids=[1])
                        measured.sample((1, 0), ControllerAction(env.robot.get_qpos()))
                        assert measured.results[0].status == "passed"
                        assert measured.results[1].status == "failed"
                        assert env.sim.physics_step_index == before_query
                    finally:
                        cube.set_local_pose(original, env_ids=[1])
                        measured.close()
                if backend == "native_contacts":
                    assert check.rows[1].substeps > 0 and check.rows[0].substeps == 0
                    # The clock uses the same native physics intervals that
                    # supplied contact evidence, independently of World time.
                    report = check.rows[1]._hold_report()
                    assert report["samples"] == report["required_samples"] == 3
                    assert report["elapsed_duration_s"] == pytest.approx(
                        2 * env.step_dt
                    )
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


def _check_coordinated_grasp_targets(env):
    """Real selected-row IK keeps commanded preload on contact-blocked hands."""
    import torch
    from embodichain.lab.sim.atomic_skills import (
        AntipodalAffordance,
        AtomicSkillEngine,
        ControlPartCommandProfile,
        CoordinatedPlacementGoal,
        CoordinatedPlacementOptions,
        HeldObjectState,
        MotionPolicy,
        ObjectSemantics,
        SkillInvocation,
        TaskState,
    )
    from embodichain.lab.sim.motion.motion_generator import (
        MotionGenerator,
        MotionGenCfg,
    )
    from embodichain.lab.sim.motion.planners.trapezoidal_planner import (
        TrapezoidalPlannerCfg,
    )

    rows = (3, 1)
    generator_cfg = MotionGenCfg(
        planner_cfg=TrapezoidalPlannerCfg(
            robot_uid=env.robot.uid, sim_instance_id=env.sim.instance_id
        )
    )
    generator = MotionGenerator(generator_cfg, env_ids=rows)
    profiles = {}
    for part in ("left_eef", "right_eef"):
        limits = env.robot.get_qpos_limits(name=part)[0]
        profiles[part] = ControlPartCommandProfile.joint_positions(
            open=limits[:, 0], grasp=limits[:, 1]
        )
    engine = AtomicSkillEngine(generator, env_ids=rows, control_profiles=profiles)
    binding = engine.bind_control_parts(
        "coordinated_placement",
        {
            "placing": {"motion": "left_arm", "grasp": "left_eef"},
            "support": {"motion": "right_arm", "grasp": "right_eef"},
        },
    )
    poses, held = {}, {}
    for slot, arm in (("placing", "left_arm"), ("support", "right_arm")):
        poses[slot] = env.robot.compute_fk(
            qpos=env.robot.get_qpos(name=arm)[list(rows)],
            name=arm,
            to_matrix=True,
            env_ids=rows,
        )
        held[binding.endpoint(slot, "motion").task_state_key] = HeldObjectState(
            semantics=ObjectSemantics(
                entity_id=slot,
                label=slot,
                geometry={},
                affordance=AntipodalAffordance(),
            ),
            object_to_eef=torch.eye(4).repeat(2, 1, 1),
            grasp_xpos=poses[slot],
        )
    context = engine.initial_context(
        task=TaskState(batch_size=2, device="cpu", held_objects=held), control_dt=0.04
    )
    before = env.sim.physics_step_index
    # The stronger solver capability must preserve actual preceding-seed IK,
    # including a failed peer, without losing non-contiguous physical rows.
    start = context.robot.qpos[:, env.robot.get_joint_ids("right_arm")]
    nodes = start[:, None].repeat(1, 4, 1)
    nodes[:, :, 0] += torch.tensor([0.0, 0.01, 0.02, 0.03])
    targets = torch.stack(
        [
            env.robot.compute_fk(
                qpos=node, name="right_arm", to_matrix=True, env_ids=rows
            )
            for node in nodes.unbind(1)
        ],
        dim=1,
    )
    for failed_peer in (False, True):
        requested = targets.clone()
        if failed_peer:
            requested[1, 2, 0, 3] += 100
        seed, points = start, []
        accepted = torch.ones(2, dtype=torch.bool)
        for pose in requested.unbind(1):
            valid, point = env.robot.compute_ik(
                pose=pose, name="right_arm", joint_seed=seed, env_ids=rows
            )
            seed = torch.where(valid[:, None], point, seed)
            points.append(seed)
            accepted &= valid
        expected = torch.where(
            accepted[:, None, None], torch.stack(points, 1), start[:, None]
        )
        valid, actual = generator._solve_seeded_cartesian_waypoints(
            requested, start, "right_arm"
        )
        assert torch.equal(valid, accepted)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    from embodichain.lab.expansion._planning_batches import batched_planning

    class PathQuery:
        # A numerical request producer, with the real Robot/SSIK implementation.
        def __init__(self, index):
            self.index = index

        def next(self):
            index = self.index
            single = MotionGenerator(generator_cfg, env_ids=(rows[index],))
            return single._solve_seeded_cartesian_waypoints(
                requested[index : index + 1], start[index : index + 1], "right_arm"
            )

    with batched_planning(env.robot, seed=2) as broker:
        answers = broker.gather({row: PathQuery(i) for i, row in enumerate(rows)})
        assert broker.stats["max_group"] == 2
    for index, row in enumerate(rows):
        assert torch.equal(answers[row][0], accepted[index : index + 1])
        torch.testing.assert_close(
            answers[row][1], expected[index : index + 1], rtol=0, atol=1e-6
        )
    for release in (False, True):
        plan = engine.plan(
            SkillInvocation(
                skill_id="coordinated_placement",
                binding=binding,
                goal=CoordinatedPlacementGoal(
                    placing_object_target_pose=poses["placing"],
                    support_object_target_pose=poses["support"],
                ),
                motion_policy=MotionPolicy(sample_count=30, strategy="ik_interp"),
                skill_options=CoordinatedPlacementOptions(
                    release=release, lift_height=0, hand_interp_steps=4, retreat_steps=4
                ),
            ),
            context,
        )
        assert plan.success_all, plan.diagnostics
        for part in ("left_eef", "right_eef"):
            ids = env.robot.get_joint_ids(part)
            expected = env.robot.get_qpos_limits(name=part)[0, :, 1]
            actual = plan.joint_trajectory.positions[:, :, ids]
            if part == "left_eef" and release:
                actual = actual[:, : plan.segments[0].stop]
            assert not torch.equal(context.robot.qpos[:, ids], expected.expand(2, -1))
            torch.testing.assert_close(
                actual, expected.expand_as(actual), rtol=0, atol=0
            )
    assert env.sim.physics_step_index == before


def _probe_handover_rows():
    """Plan and consume selected rows in a larger real handover World."""
    import torch
    from embodichain.lab.expansion.program import collect_program_batch
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import (
        REGISTERED_ENVS,
        discover_task_packages,
        execute_init_hooks,
    )
    from embodichain.lab.sim import SimulationManager
    from embodichain.utils.utility import load_config

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "embodichain/extensions/tasks/configs/tasks/manipulation/hand_over/task.tianji_marvin.yaml"
    )
    discover_task_packages()
    execute_init_hooks()
    cfg = config_to_cfg(load_config(path), source_path=path)
    cfg.num_envs = cfg.sim_cfg.num_envs = 4
    cfg.sim_cfg.device, cfg.sim_cfg.headless = "cpu", True
    cfg.sim_cfg.render_cfg.renderer = "no-render"
    cfg.sensor, cfg.enable_sensor, cfg.dataset = [], False, None
    cfg.init_rollout_buffer = cfg.record_trajectory = False
    env = REGISTERED_ENVS[load_config(path)["id"]].make(cfg=cfg)
    adapters = []
    try:
        env.reset(seed=2)
        _check_coordinated_grasp_targets(env)
        program = env.task_program_adapter.compile(cfg.task_program)
        bridges = {}
        for row in (3, 1):
            adapter = env.task_program_adapter.for_env_ids((row,))
            adapters.append(adapter)
            bridges[row] = adapter.create_bridge(program, isolate_policy_sessions=True)
        observed = []
        result = collect_program_batch(
            env,
            bridges,
            max_steps=2,
            observe_step=lambda ids, action, observation: observed.append(ids),
        )
        # The original FK omitted selected IDs and rejected both plans at zero
        # commands. Budget cancellation must instead follow real commands.
        assert result.env_ids == (3, 1)
        assert all(length >= 1 for length in result.lengths), result.lengths
        assert observed and set(observed[0]) == {3, 1}
        assert not any(result.accepted)
    finally:
        for adapter in adapters:
            adapter.close()
        env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.subprocess_sim
def test_selected_handover_rows_preserve_world_batch(tmp_path):
    result = subprocess.run(
        [sys.executable, __file__, "handover"],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "SDL_VIDEODRIVER": "offscreen", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-5000:]


if __name__ == "__main__":
    from embodichain.lab.sim.utility._validated_grasps import ValidatedGraspCache

    with patch.object(ValidatedGraspCache, "_load_labels", _recorded_grasp_labels):
        if sys.argv[1] == "handover":
            _probe_handover_rows()
        else:
            _probe(sys.argv[1])
