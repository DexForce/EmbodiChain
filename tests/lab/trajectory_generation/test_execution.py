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

"""Causal execution through real Gym control ports and deterministic CPU physics."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
from embodichain.lab.gym.utils.profiler import EnvProfiler
from embodichain.lab.sim.motion.expansion import (
    CandidateIdentity,
    CandidateTrajectoryBatch,
    SceneCase,
    TrajectoryPhase,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.trajectory_generation.execution import QposRolloutExecutor


def _passed(*args) -> ValidationResult:
    return ValidationResult((ValidationCheck("task_success", "passed"),))


class _Robot:
    def __init__(self) -> None:
        self.joint_names = ("arm", "locked")
        self.active_joint_ids = [0]
        self.qpos = torch.tensor([[0.0, 0.3], [0.0, 0.3]])
        self.target = self.qpos.clone()
        self.target_velocity = torch.zeros_like(self.qpos)
        self.force = torch.zeros_like(self.qpos)
        self.pose = torch.eye(4).repeat(2, 1, 1)
        self.commands = []
        self.offset = 0.0

    def get_qpos(self, target=False):
        return self.target if target else self.qpos

    def get_qpos_limits(self):
        return torch.tensor([-1.0, 1.0]).repeat(2, 2, 1)

    def get_qvel(self, target=False):
        assert target
        return self.target_velocity

    def get_qf(self):
        return self.force

    def get_local_pose(self, to_matrix=False):
        assert to_matrix
        return self.pose

    def set_qpos(self, qpos, *, joint_ids, target=True):
        assert target
        self.target[:, joint_ids] = qpos + self.offset
        self.commands.append(self.target.clone())

    def set_qvel(self, qvel, *, joint_ids, target=True):
        self.target_velocity[:, joint_ids] = qvel

    def set_qf(self, qf, *, joint_ids):
        self.force[:, joint_ids] = qf


class _Sim:
    num_envs = 2
    device = torch.device("cpu")

    def __init__(self, robot):
        self.robot = robot
        self.simulation_time = 11.0
        self.updates = []
        self.fail_at = None
        self.clock_scale = 1.0
        self.after_update = lambda: None
        self.cube = SimpleNamespace(pose=torch.eye(4).repeat(2, 1, 1))
        self.cube.get_local_pose = lambda to_matrix: self.cube.pose

    def update(self, dt, steps):
        self.updates.append((dt, steps))
        if len(self.updates) == self.fail_at:
            raise RuntimeError("physics failed")
        self.robot.qpos += 0.5 * (self.robot.target - self.robot.qpos)
        self.simulation_time += dt * steps * self.clock_scale
        self.after_update()

    def get_rigid_object_uid_list(self):
        return ["cube"]

    def get_rigid_object(self, uid):
        assert uid == "cube"
        return self.cube


class _Env(EmbodiedEnv):
    """Use BaseEnv.step and normal EmbodiedEnv action/demo machinery unchanged."""

    def __init__(self, sim, robot):
        self.sim, self.robot = sim, robot
        self._num_envs = 2
        self.active_joint_ids = [0]
        self.cfg = SimpleNamespace(
            sim_steps_per_control=2,
            events=False,
            dataset=False,
            ignore_terminations=False,
            trajectory_auto_save=False,
        )
        self.sim_cfg = SimpleNamespace(physics_dt=0.01)
        self.max_episode_steps = 100
        self._profiler = EnvProfiler(None, sim.device)
        self._elapsed_steps = torch.zeros(2, dtype=torch.long)
        self._traj_buffer = None
        self.rollout_buffer = None
        self.rollout_steps = torch.zeros(2, dtype=torch.long)
        self._demo_steps = torch.zeros(2, dtype=torch.long)
        self._demo_active_segment_ids = torch.zeros(2, dtype=torch.long)
        self._demo_active_mask = torch.ones(2, dtype=torch.bool)
        self._demo_segment_participants = torch.zeros(2, dtype=torch.bool)
        self.obs_calls = 0
        self.fail_observation_at = None
        self.terminate_after = [None, None]
        self.post_target_offset = 0.0
        self.action_manager = Mock(process_action=Mock(side_effect=self._post))
        self.recorded_masks = []

    def _post(self, action, mode):
        assert mode == "post", "ControllerAction must bypass policy preprocessing"
        self.robot.target[:, 0] += self.post_target_offset
        return action + 8.0  # Postprocessor output is not the command label.

    def get_obs(self, **kwargs):
        self.obs_calls += 1
        if self.obs_calls == self.fail_observation_at:
            raise RuntimeError("camera unavailable")
        return TensorDict({"state": self.robot.qpos.clone()}, batch_size=[2])

    def get_info(self, **kwargs):
        return {
            "success": torch.tensor(
                [
                    limit is not None and len(self.sim.updates) >= limit
                    for limit in self.terminate_after
                ]
            ),
            "fail": torch.zeros(2, dtype=torch.bool),
        }

    def get_reward(self, **kwargs):
        return torch.zeros(2)

    def _extend_reward(self, rewards, **kwargs):
        return rewards

    def check_truncated(self, **kwargs):
        return torch.zeros(2, dtype=torch.bool)

    def is_task_success(self, **kwargs):
        return torch.ones(2, dtype=torch.bool)

    def _hook_after_sim_step(self, **kwargs):
        self.recorded_masks.append(self._demo_active_mask.clone())
        self._demo_steps += self._demo_active_mask


@pytest.fixture(params=[False, True], ids=["pure_sim", "gym"])
def scene(request):
    robot = _Robot()
    sim = _Sim(robot)
    env = _Env(sim, robot) if request.param else None
    cases = tuple(
        SceneCase(f"case-{i}", f"initial-{i}", "scene", "task", "robot")
        for i in range(2)
    )
    binding = SimpleNamespace(cases=cases)
    initial = env.get_obs() if env is not None else None
    snapshots = tuple(
        SimpleNamespace(
            root_pose=robot.pose[i].clone(),
            entity_poses={"cube": sim.cube.pose[i].clone()},
        )
        for i in range(2)
    )
    host = SimpleNamespace(
        adapter=SimpleNamespace(robot=robot, sim=sim, atol=1e-5),
        env=env,
        profile=SimpleNamespace(physics_dt=0.01),
        assert_current=Mock(),
        verify_initial=Mock(return_value=_passed()),
        snapshots=Mock(return_value=snapshots),
        initial_observation=lambda value: (
            initial.clone() if initial is not None else None
        ),
    )
    if env is not None:
        env.acquire_generation_lease(host)
        env._generation_prepared = True
    yield host, binding
    if env is not None:
        env.release_generation_lease(host)


def _candidate(slot, values=(0.0, 0.2, 0.4, 0.6)):
    positions = torch.tensor([[value, 0.3] for value in values]).unsqueeze(0)
    return CandidateTrajectoryBatch(
        positions=positions,
        dt=torch.tensor([[0.0] + [0.02] * (len(values) - 1)], dtype=torch.float64),
        valid_length=torch.tensor([len(values)]),
        identities=(
            CandidateIdentity(
                f"case-{slot}",
                f"initial-{slot}",
                f"candidate-{slot}",
                "family",
                "source",
                "rev",
                "template",
            ),
        ),
        joint_names=("arm", "locked"),
        phases=((TrajectoryPhase("move", 0, len(values), "free"),),),
    )


def _executor(host, **kwargs):
    kwargs.setdefault("validator", _passed)
    kwargs.setdefault("observe", lambda: {"state": host.adapter.robot.qpos})
    return QposRolloutExecutor(host, control_dt=0.02, **kwargs)


def _run(executor, binding, candidates, **kwargs):
    kwargs.setdefault("on_started", Mock())
    return executor.execute(
        binding,
        candidates,
        {
            value.identities[0].candidate_id: (f"episode-{i}", f"commit-{i}")
            for i, value in enumerate(candidates)
            if value is not None
        },
        **kwargs,
    )


def test_causal_targets_terminal_frame_ragged_rows_and_owned_payload(scene):
    host, binding = scene
    robot, sim, env = host.adapter.robot, host.adapter.sim, host.env
    if env is not None:
        env.post_target_offset = 0.01
    validator = Mock(
        side_effect=lambda *args: (
            _passed() if len(sim.updates) == 3 else pytest.fail("early validation")
        )
    )
    started = Mock()
    episodes = _run(
        _executor(host, validator=validator),
        binding,
        [_candidate(0, (0.0, 0.2)), _candidate(1)],
        on_started=started,
    )
    assert [len(value.actions) for value in episodes] == [1, 3]
    assert started.call_count == 2
    assert sim.updates == [(0.01, 2)] * 3
    assert all(episode.validation.accepted for episode in episodes)
    assert episodes[0].observations["state"].shape == (2, 2)
    torch.testing.assert_close(episodes[1].actions[:, 0], torch.tensor([0.2, 0.4, 0.6]))
    torch.testing.assert_close(
        episodes[1].observations["joint_positions"][:, 0],
        torch.tensor([0.0, 0.1, 0.25, 0.425]),
    )
    torch.testing.assert_close(
        episodes[1].timestamps, torch.tensor([0, 0.02, 0.04, 0.06], dtype=torch.float64)
    )
    assert robot.commands[2][0, 0] == pytest.approx(0.1)  # measured hold
    assert robot.target[1, 0] == robot.qpos[1, 0]  # final safe hold
    if env is not None:
        assert env.obs_calls == 4  # initial preparation + actual transitions
        assert env._demo_steps.tolist() == [1, 3]
        assert [mask.tolist() for mask in env.recorded_masks] == [
            [True, True],
            [False, True],
            [False, True],
        ]
        assert env.action_manager.process_action.call_count == 3
        assert env._generation_command_observer is None
    robot.qpos.zero_()
    assert episodes[1].observations["state"][-1, 0] == pytest.approx(0.425)


def test_idle_row_never_starts_or_records(scene):
    host, binding = scene
    started = Mock()
    episodes = _run(_executor(host), binding, [None, _candidate(1)], on_started=started)
    assert episodes[0] is None
    assert episodes[1].validation.accepted
    started.assert_called_once()
    assert host.adapter.robot.qpos[0, 0] == 0


@pytest.mark.parametrize("stop_after", [0, 1])
def test_stop_counts_only_submitted_commands_and_rejects_partial_episode(
    scene, stop_after
):
    host, binding = scene
    started = Mock()
    executor = _executor(host)
    episodes = _run(
        executor,
        binding,
        [_candidate(0), None],
        on_started=started,
        should_stop=lambda: len(host.adapter.sim.updates) >= stop_after,
    )
    assert started.call_count == int(stop_after > 0)
    assert executor.last_failures == {
        "candidate-0": "execution cancelled by should_stop"
    }
    if stop_after == 0:
        assert episodes == (None, None)
    else:
        assert len(episodes[0].actions) == 1
        assert not episodes[0].validation.accepted
        assert episodes[0].phases[0].stop_index == 2


@pytest.mark.parametrize("fail_at", [1, 2])
def test_physics_exception_counts_attempt_without_fabricating_transition(
    scene, fail_at
):
    host, binding = scene
    host.adapter.sim.fail_at = fail_at
    started = Mock()
    executor = _executor(host)
    episodes = _run(executor, binding, [_candidate(0), None], on_started=started)
    started.assert_called_once()
    assert executor.last_failures == {"candidate-0": "RuntimeError: physics failed"}
    if fail_at == 1:
        assert episodes[0] is None
    else:
        assert len(episodes[0].actions) == 1
        assert not episodes[0].validation.accepted
    assert host.adapter.robot.target[0, 0] == host.adapter.robot.qpos[0, 0]


def test_observation_failure_after_command_still_counts_attempt(scene):
    host, binding = scene
    if host.env is not None:
        host.env.fail_observation_at = 2
        executor = _executor(host)
    else:
        observe = Mock(
            side_effect=[
                {"state": host.adapter.robot.qpos.clone()},
                RuntimeError("camera unavailable"),
            ]
        )
        executor = _executor(host, observe=observe)
    started = Mock()
    assert _run(executor, binding, [_candidate(0), None], on_started=started) == (
        None,
        None,
    )
    started.assert_called_once()
    assert executor.last_failures == {"candidate-0": "RuntimeError: camera unavailable"}


def test_changed_controller_target_is_recorded_and_rejected(scene):
    host, binding = scene
    host.adapter.robot.offset = 0.05
    episode = _run(_executor(host), binding, [_candidate(0), None])[0]
    torch.testing.assert_close(episode.actions[:, 0], torch.tensor([0.25, 0.45, 0.65]))
    assert not episode.validation.accepted


def test_changed_obstacle_or_root_never_passes_static_collision_contract(scene):
    host, binding = scene
    host.adapter.sim.after_update = lambda: host.adapter.sim.cube.pose[0, 0, 3].add_(
        0.01
    )
    episodes = _run(_executor(host), binding, [_candidate(0), _candidate(1)])
    assert not episodes[0].validation.accepted
    assert episodes[1].validation.accepted
    assert (
        next(
            check
            for check in episodes[0].validation.checks
            if check.check_id == "fixed_collision_world"
        ).status
        == "failed"
    )


def test_actual_clock_disagreement_rejects_without_fabricated_timestamps(scene):
    host, binding = scene
    host.adapter.sim.clock_scale = 2
    started = Mock()
    episodes = _run(_executor(host), binding, [_candidate(0), None], on_started=started)
    assert episodes == (None, None)
    started.assert_called_once()


@pytest.mark.parametrize("fault", ["initial", "inactive", "timing", "case", "budget"])
def test_preflight_failure_submits_no_command(scene, fault):
    host, binding = scene
    candidate = _candidate(0)
    kwargs = {}
    if fault == "initial":
        candidate.positions[0, 0, 0] = 0.1
    elif fault == "inactive":
        candidate.positions[0, 1, 1] = 0.4
    elif fault == "timing":
        candidate.dt[0, 1] = 0.03
    elif fault == "case":
        candidate = _candidate(1)
    else:
        kwargs["max_episode_bytes"] = 1
    with pytest.raises(ValueError):
        _run(_executor(host, **kwargs), binding, [candidate, None])
    assert not host.adapter.sim.updates
    assert not host.adapter.robot.commands


def test_exhaustion_requires_explicit_final_task_success(scene):
    host, binding = scene
    validator = lambda *args: ValidationResult(
        (ValidationCheck("task_success", "failed"),)
    )
    episode = _run(
        _executor(host, validator=validator), binding, [_candidate(0), None]
    )[0]
    assert len(episode.actions) == 3
    assert not episode.validation.accepted


def test_early_gym_termination_freezes_terminal_observation_and_holds_row(scene):
    host, binding = scene
    if host.env is None:
        pytest.skip("Gym-specific terminal semantics")
    host.env.terminate_after[0] = 1
    episodes = _run(_executor(host), binding, [_candidate(0), _candidate(1)])
    assert len(episodes[0].actions) == 1
    assert not episodes[0].validation.accepted
    assert len(episodes[1].actions) == 3
    assert episodes[1].validation.accepted
    assert episodes[0].observations["state"][-1, 0] == pytest.approx(0.1)
    assert host.env._demo_steps.tolist() == [1, 3]


def test_schema_growth_is_rejected_before_copying_or_retaining_changed_frame(scene):
    host, binding = scene
    if host.env is None:

        def observe():
            return {
                "state": (
                    torch.zeros(2, 3)
                    if host.adapter.sim.updates
                    else host.adapter.robot.qpos
                )
            }

        executor = _executor(host, observe=observe)
    else:
        executor = _executor(
            host,
            encode_observation=lambda raw: {
                "state": torch.zeros(2, 3) if host.adapter.sim.updates else raw["state"]
            },
        )
    started = Mock()
    assert _run(executor, binding, [_candidate(0), None], on_started=started) == (
        None,
        None,
    )
    started.assert_called_once()


def test_oversized_validator_metadata_rejects_with_a_bounded_failure(scene):
    host, binding = scene
    validator = lambda *args: ValidationResult(
        (ValidationCheck("task_success", "passed", "x" * 70000),)
    )
    episode = _run(
        _executor(host, validator=validator, max_episode_bytes=70000),
        binding,
        [_candidate(0), None],
    )[0]
    assert not episode.validation.accepted
    assert len(episode.validation.checks[0].detail) < 100


def test_running_guard_covers_final_task_validator(scene):
    host, binding = scene
    executor = _executor(host)

    def validate(*args):
        with pytest.raises(RuntimeError, match="serialized"):
            _run(executor, binding, [_candidate(0), None])
        return _passed()

    executor.validator = validate
    assert _run(executor, binding, [_candidate(0), None])[0].validation.accepted


@pytest.mark.requires_sim
def test_real_cpu_sim_qpos_rollout_records_measured_state_and_time(
    tmp_path, monkeypatch
):
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RobotCfg, RigidObjectCfg
    from embodichain.lab.sim.shapes import CubeCfg
    from embodichain.lab.trajectory_generation.initial_state import (
        FixedSceneHost,
        InitialStateProfile,
    )
    from embodichain.lab.trajectory_generation.integrations.sim import (
        SimInitialStateAdapter,
    )

    monkeypatch.setenv("EMBODICHAIN_SIM_EXIT_PROCESS", "0")
    urdf = tmp_path / "execution_robot.urdf"
    urdf.write_text(
        '<robot name="execution_robot">'
        '<link name="base"><inertial><mass value="1"/>'
        '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>'
        "</inertial></link>"
        '<link name="tip"><inertial><mass value="0.1"/>'
        '<inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>'
        "</inertial></link>"
        '<joint name="joint" type="revolute"><parent link="base"/><child link="tip"/>'
        '<origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>'
        '<limit lower="-1" upper="1" effort="10" velocity="2"/></joint></robot>',
        encoding="utf8",
    )
    sim = SimulationManager(
        SimulationManagerCfg(headless=True, sim_device="cpu", num_envs=1)
    )
    try:
        sim.enable_physics(True)
        robot = sim.add_robot(RobotCfg(uid="robot", fpath=str(urdf), fix_base=True))
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="cube",
                shape=CubeCfg(size=[0.05, 0.05, 0.05]),
                init_pos=[2, 0, 1],
                body_type="static",
            )
        )
        adapter = SimInitialStateAdapter(sim, robot)
        profile = InitialStateProfile(
            profile_id="test",
            prepare=lambda: None,
            signature=lambda: "fixed",
            verify=lambda cases: _passed(),
        )
        with FixedSceneHost(adapter, profile) as host:
            binding = host.acquire_case(
                [SceneCase("case", "initial", "fixed", "move", "robot")]
            )
            initial = robot.get_qpos().clone()
            candidate = CandidateTrajectoryBatch(
                positions=torch.stack((initial, initial + 0.01, initial + 0.02), dim=1),
                dt=torch.tensor([[0, 0.02, 0.02]], dtype=torch.float64),
                valid_length=torch.tensor([3]),
                identities=(
                    CandidateIdentity(
                        "case",
                        "initial",
                        "candidate",
                        "family",
                        "source",
                        "rev",
                        "template",
                    ),
                ),
                joint_names=tuple(robot.joint_names),
            )
            executor = QposRolloutExecutor(
                host,
                control_dt=0.02,
                observe=lambda: {"state": robot.get_qpos()},
                validator=_passed,
            )
            started = Mock()
            episode = executor.execute(
                binding,
                [candidate],
                {"candidate": ("episode", "commit")},
                on_started=started,
            )[0]
            assert episode is not None
            assert episode.validation.accepted, episode.validation.checks
            assert episode.actions.shape == (2, 1)
            torch.testing.assert_close(episode.actions, candidate.positions[0, 1:])
            torch.testing.assert_close(
                episode.observations["joint_positions"][-1], robot.get_qpos()[0]
            )
            torch.testing.assert_close(
                episode.timestamps, torch.tensor([0, 0.02, 0.04], dtype=torch.float64)
            )
            started.assert_called_once()
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


def test_idle_rows_clear_residual_velocity_targets_and_effort_before_physics(scene):
    host, binding = scene
    robot = host.adapter.robot
    robot.target_velocity.fill_(0.7)
    robot.force.fill_(0.8)

    def verify_idle_hold():
        assert robot.target_velocity[0, 0] == 0
        assert robot.force[0, 0] == 0

    host.adapter.sim.after_update = verify_idle_hold
    assert _run(_executor(host), binding, [None, _candidate(1)])[1].validation.accepted


def test_started_callback_failure_counts_each_submitted_row_and_stops_before_physics(
    scene,
):
    host, binding = scene
    started = Mock(side_effect=RuntimeError("accounting failed"))
    episodes = _run(
        _executor(host), binding, [_candidate(0), _candidate(1)], on_started=started
    )
    assert episodes == (None, None)
    assert started.call_count == 2
    assert not host.adapter.sim.updates
    torch.testing.assert_close(host.adapter.robot.target, host.adapter.robot.qpos)


def test_measured_inactive_joint_movement_rejects_the_row(scene):
    host, binding = scene
    host.adapter.sim.after_update = lambda: host.adapter.robot.qpos[0, 1].add_(0.01)
    episode = _run(_executor(host), binding, [_candidate(0), None])[0]
    assert not episode.validation.accepted
    assert "inactive" in episode.validation.checks[0].detail


def test_task_validator_cannot_overwrite_execution_check(scene):
    host, _ = scene
    with pytest.raises(ValueError, match="execution check"):
        _executor(host, validator_id="execution_complete")


def test_last_failures_are_bounded_read_only_owned_and_cleared_for_the_next_batch(
    scene,
):
    host, binding = scene
    host.adapter.sim.update = Mock(side_effect=RuntimeError("x" * 2000))
    executor = _executor(host)
    assert _run(executor, binding, [_candidate(0), None]) == (None, None)
    previous = executor.last_failures
    assert len(previous["candidate-0"]) == 512
    with pytest.raises(TypeError):
        previous["candidate-0"] = "changed"
    assert _run(executor, binding, [None, None]) == (None, None)
    assert executor.last_failures == {}
    assert len(previous["candidate-0"]) == 512
