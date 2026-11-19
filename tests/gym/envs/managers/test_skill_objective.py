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

"""Measured placement windows and physical failure cases independent of planners."""

from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch

from embodichain.lab.gym.envs.managers.cfg import ObservationCfg
from embodichain.lab.gym.envs.managers.skill_objective import SkillObjectiveObservation


def _setup(*, skill="place", goal=None, freeze_reference=False):
    pose = torch.eye(4)[None].repeat(2, 1, 1)
    pose[:, :3, 3] = torch.tensor([0.75, -0.1, 0.95])
    hand = torch.eye(4)[None].repeat(2, 1, 1)
    hand[:, 2, 3] = 1.6
    shape = SimpleNamespace(
        vertices=None,
        half_extents=torch.tensor([0.025] * 3),
        local_pose=torch.eye(4),
        radius=None,
    )
    data = {
        "user_ids": torch.tensor([[[4, 1]], [[4, 1]]]),
        "normal": torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]]),
        "impulse": torch.ones(2, 1),
        "is_valid": torch.ones(2, 1, dtype=torch.bool),
        "impulse_valid": torch.ones(2, 1, dtype=torch.bool),
    }

    class SampledSensor:
        """Replay completed fake physics samples, including sensor observer hooks."""

        contact_capabilities = SimpleNamespace(impulse=True)
        uid = "test_sensor"

        def __init__(self):
            self.observers = {}
            self.dropped_contacts_mask = torch.zeros(2, dtype=torch.bool)
            self.last_step = 0

        @property
        def requires_substep_update(self):
            return bool(self.observers)

        def add_sampling_observer(self, name, observer):
            self.observers[name] = observer

        def get_data(self):
            return data

        def begin_control_step(self):
            for observer in self.observers.values():
                observer.begin_control_step()

        def update_physics_step(self, dt):
            for observer in self.observers.values():
                observer.after_sample(dt, observer.before_sample())
            self.last_step = sim.physics_step_index

        def update(self):
            # Most unit cases directly advance a fake clock. Publish those
            # completed samples once; repeated queries cannot resample them.
            end = sim.physics_step_index
            if end != self.last_step:
                sim.physics_step_index = self.last_step
                self.begin_control_step()
                for step in range(self.last_step + 1, end + 1):
                    sim.physics_step_index = step
                    self.update_physics_step(
                        env.step_dt / env.cfg.sim_steps_per_control
                    )
                sim.physics_step_index = end

    sensor = SampledSensor()
    sensor.get_actor_ids = lambda uid, links=None: (
        torch.tensor([[2, 3], [2, 3]])
        if links
        else torch.full((2, 1), 1 if uid == "cargo" else 4)
    )
    robot = SimpleNamespace(
        uid="robot",
        get_joint_ids=lambda part: [0],
        get_qpos=lambda part: torch.zeros(2, 1),
        compute_fk=lambda *args, **kw: hand,
        get_link_pose=lambda *args, **kw: hand,
        get_link_collision_shapes=lambda *args: [shape],
    )
    cargo = SimpleNamespace(
        get_local_pose=lambda **kw: pose,
        get_collision_shapes=lambda *args, **kw: [shape],
        body_data=SimpleNamespace(lin_vel=torch.zeros(2, 3), ang_vel=torch.zeros(2, 3)),
    )
    reference_pose = torch.eye(4)[None].repeat(2, 1, 1)
    reference = SimpleNamespace(get_local_pose=lambda **kw: reference_pose)
    sim = SimpleNamespace(
        physics_step_index=0,
        get_robot=lambda uid: robot,
        get_rigid_object=lambda uid: reference if uid == "reference" else cargo,
        add_sensor=lambda cfg: sensor,
    )
    env = SimpleNamespace(
        sensors={},
        num_envs=2,
        device=torch.device("cpu"),
        sim=sim,
        step_dt=0.05,
        cfg=SimpleNamespace(sim_steps_per_control=1),
    )
    objective = {
        "skill_id": skill,
        "binding": {
            "robot": "robot",
            "motion_part": "arm",
            "object": "cargo",
            "support": "base",
            "pad_links": ["pad_a", "pad_b"],
            "hand_links": ["pad_a", "pad_b"],
        },
        "goal": {"position": [0.75, -0.1, 0.95]} if goal is None else goal,
    }
    term = SkillObjectiveObservation(
        ObservationCfg(
            name="test",
            params={"objective": objective, "freeze_reference": freeze_reference},
        ),
        env,
    )
    return env, term, data, pose


def test_place_requires_continuous_supported_separated_window_and_queries_do_not_advance():
    env, term, data, pose = _setup()
    assert not term(env, {}).values[:, 1].any()
    term.initialize_observation(None, [0, 1])
    # Row 1 has no support. It stays near the target but cannot pass.
    data["is_valid"][1] = False
    for step in range(1, 5):
        env.sim.physics_step_index = step
        sample = term(env, {})
        repeated = term(env, {})
        torch.testing.assert_close(sample.values, repeated.values)
        assert sample.values[:, 1].bool().tolist() == [step == 4, False]
        term.advance_observation()
    term.reset([0])
    assert term._stable_time.tolist() == [0.0, 0.0]
    assert term._initialized.tolist() == [False, True]


def test_contact_overflow_and_velocity_break_the_acceptance_window():
    env, term, data, pose = _setup()
    term(env, {})
    term.initialize_observation(None, [0, 1])
    for step in range(1, 4):
        env.sim.physics_step_index = step
        term(env, {})
        term.advance_observation()
    env.sim.physics_step_index = 4
    term.sensor.dropped_contacts_mask[0] = True
    term.object.body_data.lin_vel[1, 0] = 0.5
    sample = term(env, {})
    assert sample.valid.tolist() == [False, True]
    assert sample.values[:, 1].tolist() == [0.0, 0.0]
    term.advance_observation()
    assert term._stable_time.tolist() == [0.0, 0.0]


def test_control_gap_cannot_count_as_consecutive_stability():
    env, term, data, pose = _setup()
    term(env, {})
    term.initialize_observation(None, [0, 1])
    env.sim.physics_step_index = 10
    term(env, {})
    term.advance_observation()
    assert term._stable_time.tolist() == [0.0, 0.0]


def test_distinct_goals_on_one_robot_do_not_alias_raw_capture():
    from embodichain.lab.gym.envs.managers.observation_manager import (
        _ObservationCapture,
    )

    env, first, _, _ = _setup()
    objective = {**first.objective, "goal": {"position": [0.75, -0.1, 1.1]}}
    second = SkillObjectiveObservation(
        ObservationCfg(name="other_goal", params={"objective": objective}), env
    )
    with _ObservationCapture((0, 0), None, episode_owner=object()) as capture:
        a = capture.acquire(("first",), lambda: first(env, {}))
        b = capture.acquire(("second",), lambda: second(env, {}))
    assert a.descriptor != b.descriptor
    assert not torch.equal(a.values, b.values)


def test_each_repeated_occurrence_requires_a_new_stability_window():
    env, term, _, _ = _setup()
    current = [0]
    env.task_program_progress = SimpleNamespace(
        ordered_progress=lambda: (torch.full((2,), current[0]), 2),
        indices=lambda stage: (0, 1),
    )
    term.stage = "repeat"
    term(env, {})
    term.initialize_observation(None, [0, 1])
    for step in range(1, 5):
        env.sim.physics_step_index = step
        sample = term(env, {})
        term.advance_observation()
    assert sample.values[:, 1].bool().all()
    current[0] = 1
    env.sim.physics_step_index = 5
    assert not term(env, {}).values[:, 1].bool().any()
    term.advance_observation()
    torch.testing.assert_close(term._stable_time, torch.zeros(2))


@pytest.mark.parametrize("frame", ["parent", "world_axes"])
def test_program_reference_pose_freezes_per_row_and_call_and_reseeds_on_reset(frame):
    env, term, data, pose = _setup(
        skill="move_held_object",
        freeze_reference=True,
        goal={
            "position": [0.1, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "reference": "reference",
            "frame": frame,
        },
    )
    reference = term.reference.get_local_pose(to_matrix=True)
    reference[:, :3, 3] = torch.tensor([1.0, 0.0, 1.0])
    reference[:, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    pose[:, :3, 3] = torch.tensor(
        [1.0, 0.1, 1.0] if frame == "parent" else [1.1, 0.0, 1.0]
    )
    if frame == "parent":
        pose[:, :3, :3] = reference[:, :3, :3]
    completed = torch.zeros(2, dtype=torch.long)
    env.task_program_progress = SimpleNamespace(
        ordered_progress=lambda: (completed, 2), indices=lambda stage: (0, 1)
    )
    term.stage = "repeat"
    data.update(
        user_ids=torch.tensor([[[1, 2], [1, 3]]] * 2),
        normal=torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]] * 2),
        impulse=torch.ones(2, 2),
        is_valid=torch.ones(2, 2, dtype=torch.bool),
        impulse_valid=torch.ones(2, 2, dtype=torch.bool),
    )
    term(env, {})
    term.initialize_observation(None, [0, 1])

    def advance():
        env.sim.physics_step_index += 1
        sample = term(env, {})
        term.advance_observation()
        return sample

    for _ in range(3):
        sample = advance()
    assert sample.values[:, 1].bool().all()
    reference[:, 0, 3] = 2.0
    reference[:, :3, :3] = torch.eye(3)
    assert advance().values[:, 1].bool().all()
    # Following the moved reference cannot satisfy the original call target.
    hand = term.robot.compute_fk(None)
    destination = torch.tensor([2.1, 0.0, 1.0])
    hand[0, :3, 3] += destination - pose[0, :3, 3]
    pose[0, :3, 3] = destination
    pose[0, :3, :3] = torch.eye(3)
    sample = advance()
    assert sample.valid.all()
    assert sample.values[:, 1].bool().tolist() == [False, True]
    # Only row 0 enters the next call and resolves its new reference pose.
    completed[0] = 1
    assert advance().values[:, 1].bool().tolist() == [False, True]
    for _ in range(2):
        sample = advance()
    assert sample.values[:, 1].bool().all()
    term.reset([0])
    completed[0] = 0
    reference[0, 0, 3] = 3.0
    hand[0, 0, 3] += 1.0
    pose[0, 0, 3] = 3.1
    term(env, {})
    term.initialize_observation(None, [0])
    for _ in range(3):
        sample = advance()
    assert sample.values[:, 1].bool().all()


def test_inactive_place_previews_current_target_and_orientation_until_activation():
    env, term, _, pose = _setup(
        freeze_reference=True,
        goal={
            "position": [0.0, 0.0, 0.0],
            "reference": "reference",
            "frame": "world_axes",
            "preserve_orientation": True,
        },
    )
    reference = term.reference.get_local_pose(to_matrix=True)
    reference[:] = pose
    pose[:, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    completed = [0]
    env.task_program_progress = SimpleNamespace(
        ordered_progress=lambda: (torch.full((2,), completed[0]), 2),
        indices=lambda stage: (1,),
    )
    term.stage = "next"
    sample = term(env, {})
    torch.testing.assert_close(sample.values[:, 0], torch.full((2,), 2 / 3))
    term.initialize_observation(None, [0, 1])
    reference[:, 0, 3] += 0.2
    pose[:, 0, 3] += 0.2
    pose[:, :2, :2] = -torch.eye(2)
    env.sim.physics_step_index = 1
    sample = term(env, {})
    term.advance_observation()
    torch.testing.assert_close(sample.values[:, 0], torch.full((2,), 2 / 3))
    completed[0] = 1
    env.sim.physics_step_index = 2
    sample = term(env, {})
    torch.testing.assert_close(sample.values[:, 0], torch.full((2,), 2 / 3))
    assert not sample.values[:, 1].bool().any()


def test_pick_cannot_succeed_without_measured_lift_even_with_bilateral_contact():
    from embodichain.lab.sim.atomic_skills.reward_profiles import (
        default_skill_reward_profile,
    )

    env, term, data, _ = _setup()
    term.profile = default_skill_reward_profile("pick_up")
    data.update(
        user_ids=torch.tensor([[[1, 2], [1, 3]]] * 2),
        normal=torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]] * 2),
        impulse=torch.ones(2, 2),
        is_valid=torch.ones(2, 2, dtype=torch.bool),
        impulse_valid=torch.ones(2, 2, dtype=torch.bool),
    )
    term(env, {})
    term.initialize_observation(None, [0, 1])
    for step in range(1, 5):
        env.sim.physics_step_index = step
        sample = term(env, {})
        term.advance_observation()
    assert (term._held_time >= 0.1).all()
    assert not sample.values[:, 1].bool().any()


def test_final_objective_remains_measured_after_all_program_milestones():
    env, term, _, _ = _setup()
    completed = [0]
    env.task_program_progress = SimpleNamespace(
        ordered_progress=lambda: (torch.full((2,), completed[0]), 1),
        indices=lambda stage: (0,),
    )
    term.stage = "final"
    term(env, {})
    term.initialize_observation(None, [0, 1])
    for step in range(1, 5):
        env.sim.physics_step_index = step
        term(env, {})
        term.advance_observation()
    completed[0] = 1
    env.sim.physics_step_index = 5
    assert term(env, {}).values[:, 1].bool().all()
    term.advance_observation()
    assert (term._stable_time >= 0.2).all()
    # Losing support must revoke current physical success even after achievement.
    term.sensor.get_data()["is_valid"].zero_()
    term.object.get_local_pose()[:, 2, 3] += 0.005
    env.sim.physics_step_index = 6
    assert not term(env, {}).values[:, 1].bool().any()


def test_objective_identity_binds_goal_and_acceptance_thresholds():
    from copy import deepcopy
    from embodichain.lab.gym.envs.managers.skill_objective import _success_spec

    _, term, _, _ = _setup()
    changed = deepcopy(term.objective)
    changed["acceptance"]["stable_duration"] = 0.4
    assert (
        _success_spec("skill/state", term.objective).identity
        != _success_spec("skill/state", changed).identity
    )


def test_named_measured_template_changes_shaping_without_changing_acceptance(
    monkeypatch,
):
    from embodichain.lab.sim.atomic_skills.reward_profiles import (
        SkillRewardProfile,
        register_skill_reward_profile,
        _PROFILES,
    )

    env, term, _, _ = _setup()
    monkeypatch.setitem(_PROFILES, "place", dict(_PROFILES["place"]))
    register_skill_reward_profile(
        "pose_only", SkillRewardProfile("place", 2, ("pose",), "mean")
    )
    objective = {**term.objective, "reward_profile": "pose_only"}
    other = SkillObjectiveObservation(
        ObservationCfg(name="custom_goal", params={"objective": objective}), env
    )
    a, b = term(env, {}), other(env, {})
    assert (b.values[:, 0] > a.values[:, 0]).all()
    torch.testing.assert_close(a.values[:, 1], b.values[:, 1])


def test_sleeping_support_witness_requires_unchanged_physical_geometry():
    env, term, data, pose = _setup()
    term(env, {})
    term.initialize_observation(None, [0, 1])
    env.sim.physics_step_index = 1
    term(env, {})
    term.advance_observation()
    # A sleeping native body may stop emitting positive contact rows.
    data["is_valid"].zero_()
    for step in range(2, 5):
        env.sim.physics_step_index = step
        sample = term(env, {})
        term.advance_observation()
    assert sample.values[:, 1].bool().all()
    pose[0, 2, 3] += 0.005
    env.sim.physics_step_index = 5
    sample = term(env, {})
    assert sample.values[:, 1].bool().tolist() == [False, True]


@pytest.mark.parametrize("skill", ["place", "move_held_object"])
def test_transient_final_substep_contact_cannot_satisfy_a_physical_window(skill):
    from embodichain.lab.gym.envs.base_env import BaseEnv
    from embodichain.lab.sim.atomic_skills.reward_profiles import (
        default_skill_reward_profile,
    )

    env, term, data, _ = _setup()
    term.profile = default_skill_reward_profile(skill)
    if skill == "move_held_object":
        data.update(
            user_ids=torch.tensor([[[1, 2], [1, 3]]] * 2),
            normal=torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]] * 2),
            impulse=torch.ones(2, 2),
            is_valid=torch.zeros(2, 2, dtype=torch.bool),
            impulse_valid=torch.ones(2, 2, dtype=torch.bool),
        )
    else:
        data["is_valid"].zero_()
    env.cfg.sim_steps_per_control = 4
    env.physics_dt = env.step_dt / 4
    steady = [False]

    def update(dt, steps, *, after_substep=None, **kwargs):
        for index in range(steps):
            env.sim.physics_step_index += 1
            supported = steady[0] or index == steps - 1
            data["is_valid"].fill_(supported)
            term.object.body_data.lin_vel[:, 0] = 0.0 if supported else 0.5
            if after_substep:
                after_substep(dt)

    env.sim.update = update
    term(env, {})
    term.initialize_observation(None, [0, 1])
    for _ in range(4):
        BaseEnv._advance_physics(env)
        sample = term(env, {})
        assert not sample.values[:, 1].bool().any()
        term.advance_observation()
    duration = term._stable_time if skill == "place" else term._held_time
    torch.testing.assert_close(duration, torch.zeros(2))
    # A subsequent complete interval sequence really proves the minimum dwell.
    steady[0] = True
    for _ in range(4):
        BaseEnv._advance_physics(env)
        sample = term(env, {})
        term.advance_observation()
    assert sample.values[:, 1].bool().all()


@pytest.mark.parametrize("uid", ["cargo", "base"])
def test_replaced_object_handles_reject_before_measuring_stale_geometry(uid):
    env, term, _, _ = _setup()
    previous = env.sim.get_rigid_object
    replacement = SimpleNamespace(
        get_local_pose=lambda **kw: torch.eye(4)[None].repeat(2, 1, 1)
    )
    env.sim.get_rigid_object = lambda name: (
        replacement if name == uid else previous(name)
    )
    with pytest.raises(RuntimeError, match="replaced.*rebuild"):
        term(env, {})


def test_unobserved_physics_substep_invalidates_window_evidence():
    env, term, _, _ = _setup()
    term(env, {})
    term.initialize_observation(None, [0, 1])
    env.cfg.sim_steps_per_control = 4
    term.sensor.begin_control_step()
    for step in (1, 3, 4):
        env.sim.physics_step_index = step
        term.sensor.update_physics_step(env.step_dt / 4)
    sample = term(env, {})
    assert not sample.valid.any()
    assert not sample.values[:, 1].bool().any()


def test_post_physics_pose_write_cannot_reuse_contact_window():
    env, term, _, pose = _setup()
    term(env, {})
    term.initialize_observation(None, [0, 1])
    for step in range(1, 5):
        env.sim.physics_step_index = step
        term(env, {})
        term.advance_observation()
    env.sim.physics_step_index = 5
    term.sensor.update()  # Fully sampled native boundary, before interval events.
    pose[0, 2, 3] += 0.005
    sample = term(env, {})
    assert sample.valid.tolist() == [False, True]
    assert sample.values[:, 1].bool().tolist() == [False, True]


def test_post_physics_gripper_write_cannot_reuse_bilateral_contact_window():
    from embodichain.lab.sim.atomic_skills.reward_profiles import (
        default_skill_reward_profile,
    )

    env, term, data, _ = _setup()
    term.profile = default_skill_reward_profile("move_held_object")
    hand_pose = term.robot.get_link_pose("pad_a").clone()
    term.robot.get_link_pose = lambda *args, **kwargs: hand_pose
    data.update(
        user_ids=torch.tensor([[[1, 2], [1, 3]]] * 2),
        normal=torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]] * 2),
        impulse=torch.ones(2, 2),
        is_valid=torch.ones(2, 2, dtype=torch.bool),
        impulse_valid=torch.ones(2, 2, dtype=torch.bool),
    )
    term(env, {})
    term.initialize_observation(None, [0, 1])
    for step in range(1, 4):
        env.sim.physics_step_index = step
        term(env, {})
        term.advance_observation()
    env.sim.physics_step_index = 4
    term.sensor.update()
    # An interval event moves only the gripper; arm FK and the cargo stay fixed.
    hand_pose[0, 0, 3] += 1.0
    sample = term(env, {})
    assert sample.valid.tolist() == [False, True]
    assert sample.values[:, 1].bool().tolist() == [False, True]
    term.advance_observation()
    assert term._held_time[0] == 0
    # Diagnostic hosts may continue without auto-reset: reacquisition must start
    # a new window rather than restoring the invalidated hold.
    hand_pose[0, 0, 3] -= 1.0
    env.sim.physics_step_index = 5
    sample = term(env, {})
    assert sample.valid.all()
    assert sample.values[:, 1].bool().tolist() == [False, True]
