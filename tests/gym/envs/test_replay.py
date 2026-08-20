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

import gc
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from embodichain.data import get_data_path
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.wrapper import ReplayWrapper
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import JointDrivePropertiesCfg, RigidObjectCfg, RobotCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.gym.envs.managers.actions import DeltaQposTerm
from embodichain.lab.gym.envs.managers.cfg import ActionTermCfg

pytestmark = [pytest.mark.requires_sim, pytest.mark.slow]


@register_env("ReplayTest-v1", max_episode_steps=100, override=True)
class ReplayTestEnv(EmbodiedEnv):
    """UR10 + a dynamic rigid cube, used for record/replay integration tests."""

    def __init__(
        self,
        record_trajectory: bool = True,
        num_envs: int = 2,
        device: str = "cpu",
        **kwargs,
    ):
        cfg = EmbodiedEnvCfg()
        cfg.num_envs = num_envs
        cfg.max_episode_steps = 100
        cfg.sim_cfg = SimulationManagerCfg(headless=True, sim_device=device)
        cfg.robot = RobotCfg(
            uid="UR10",
            fpath=get_data_path("UniversalRobots/UR10/UR10.urdf"),
            init_pos=(0.0, 0.0, 1.0),
            drive_pros=JointDrivePropertiesCfg(drive_type="force"),
        )
        cfg.rigid_object = [
            RigidObjectCfg(
                uid="cube",
                shape=CubeCfg(size=[0.03, 0.03, 0.03]),
                init_pos=(0.0, 0.0, 0.5),
                body_type="dynamic",
            )
        ]
        cfg.record_trajectory = record_trajectory
        cfg.trajectory_auto_save = False
        cfg.init_rollout_buffer = True
        super().__init__(cfg, **kwargs)


def _drive(env, num_steps: int = 5) -> list:
    """Step env with a smooth sinusoidal action list; return the actions."""
    init_qpos = env.robot.get_qpos()
    actions = []
    for i in range(num_steps):
        t = i / max(num_steps - 1, 1)
        offset = torch.zeros_like(init_qpos)
        offset[:, 0] = torch.sin(torch.tensor(t * 2.0 * np.pi)) * 0.2
        actions.append(init_qpos + offset)
    for a in actions:
        env.step(a)
    return actions


def test_record_trajectory_populates_states():
    env = ReplayTestEnv(record_trajectory=True, num_envs=2, device="cpu")
    try:
        env.reset()
        initial_qpos = env.robot.get_qpos().clone()
        _drive(env, num_steps=5)
        assert env._traj_buffer is not None
        assert "states" in env._traj_buffer.keys()
        assert env._traj_steps.tolist() == [5, 5]
        states = env._traj_buffer["states"]
        assert tuple(states["robot"]["qpos"].shape) == (2, 100, 6)
        assert tuple(states["rigid_objects"]["cube"]["pose"].shape) == (2, 100, 7)
        # Saved rows are pre-action states. The post-final state is only the
        # pending seed for a possible next action and is excluded on save.
        assert torch.allclose(states["robot"]["qpos"][:, 0], initial_qpos)
        pending = states["robot"]["qpos"][:, env._traj_steps[0].item()]
        assert torch.allclose(pending, env.robot.get_qpos(), atol=1e-5)
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()


def test_expert_rollout_pairs_action_with_pre_action_observation():
    """Expert frame t keeps s_t while the returned s_(t+1) seeds frame t+1."""
    env = ReplayTestEnv(record_trajectory=False, num_envs=1, device="cpu")
    try:
        initial_obs, _ = env.reset()
        initial_qpos = initial_obs["robot"]["qpos"].clone()
        action = initial_qpos.clone()
        action[:, 0] += 0.2

        next_obs, _, _, _, _ = env.step(action)

        assert env.rollout_steps.tolist() == [1]
        assert torch.allclose(
            env.rollout_buffer["obs"]["robot"]["qpos"][0, 0],
            initial_qpos[0],
        )
        assert torch.allclose(
            env.rollout_buffer["obs"]["robot"]["qpos"][0, 1],
            next_obs["robot"]["qpos"][0],
        )
        assert env.rollout_buffer["valid"][0, 0]
        assert not env.rollout_buffer["valid"][0, 1]
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()


def test_save_trajectory_round_trip(tmp_path):
    env = ReplayTestEnv(record_trajectory=True, num_envs=2, device="cpu")
    try:
        env.reset()
        initial_qpos = env.robot.get_qpos().clone()
        n = 4
        _drive(env, num_steps=n)
        path = tmp_path / "traj.pt"
        env.save_trajectory(str(path))
        assert path.exists()

        from embodichain.lab.gym.utils.gym_utils import load_trajectory

        data = load_trajectory(str(path))
        assert data["meta"]["lengths"] == [n, n]
        assert data["meta"]["num_steps"] == n
        assert data["meta"]["num_envs"] == 2
        assert data["meta"]["dt"] == pytest.approx(env.step_dt)
        assert data["meta"]["physics_dt"] == pytest.approx(env.physics_dt)
        assert data["meta"]["sim_steps_per_control"] == 4
        assert data["meta"]["step_dt"] == pytest.approx(env.step_dt)
        assert data["meta"]["control_frequency"] == pytest.approx(env.control_frequency)
        assert tuple(data["states"]["robot"]["qpos"].shape) == (2, n, 6)
        assert torch.allclose(data["states"]["robot"]["qpos"][:, 0], initial_qpos)
        assert data["actions"].shape == (2, n, 6)
        assert "initial_states" not in data
        assert "cube" in data["states"]["rigid_objects"].keys()
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()


def test_no_auto_reset_when_replay_flag_set():
    env = ReplayTestEnv(record_trajectory=False, num_envs=2, device="cpu")
    try:
        env.reset()
        action = env.robot.get_qpos()  # hold position
        # Force a "done" every step so the auto-reset path would normally fire.
        success = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        fail = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env.compute_task_state = lambda **kwargs: (success, fail, {})
        env.cfg.ignore_terminations = False
        env.max_episode_steps = 10_000  # avoid time-limit truncation

        reset_calls = [0]
        orig_reset = env.reset

        def counting_reset(*a, **k):
            reset_calls[0] += 1
            return orig_reset(*a, **k)

        env.reset = counting_reset

        # With the guard on, stepping must NOT auto-reset even though dones=True.
        env._replay_no_auto_reset = True
        env.step(action)
        assert reset_calls[0] == 0

        # With the guard off, the same step triggers an auto-reset.
        env._replay_no_auto_reset = False
        env.step(action)
        assert reset_calls[0] == 1
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()


def test_kinematic_replay_reproduces_recorded_states(tmp_path):
    # --- Record ---
    env = ReplayTestEnv(record_trajectory=True, num_envs=2, device="cpu")
    try:
        env.reset()
        n = 5
        _drive(env, num_steps=n)
        path = tmp_path / "traj.pt"
        env.save_trajectory(str(path))
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    recorded = torch.load(path, weights_only=False)
    rec_states = recorded["states"]

    # --- Replay (kinematic) ---
    env2 = ReplayTestEnv(record_trajectory=False, num_envs=2, device="cpu")
    env2 = ReplayWrapper(env2, str(path), mode="kinematic")
    try:
        env2.reset()
        assert torch.allclose(
            env2.env.robot.get_qpos(),
            rec_states["robot"]["qpos"][:, 0],
            atol=1e-4,
        )
        for i in range(n):
            obs, reward, term, trunc, info = env2.step(None)
            inner = env2.env  # the wrapped ReplayTestEnv
            assert torch.allclose(
                inner.robot.get_qpos(), rec_states["robot"]["qpos"][:, i], atol=1e-4
            )
            assert torch.allclose(
                inner.robot.get_local_pose(),
                rec_states["robot"]["root_pose"][:, i],
                atol=1e-4,
            )
            assert torch.allclose(
                inner.sim.get_rigid_object("cube").get_local_pose(),
                rec_states["rigid_objects"]["cube"]["pose"][:, i],
                atol=1e-4,
            )
            assert torch.all(reward == 0)
        assert bool(trunc.all())  # truncated after consuming all steps
    finally:
        env2.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()


def test_dynamic_replay_tracks_recorded_states(tmp_path):
    env = ReplayTestEnv(record_trajectory=True, num_envs=2, device="cpu")
    try:
        env.reset()
        n = 5
        _drive(env, num_steps=n)
        recorded_final_qpos = env.robot.get_qpos().clone()
        path = tmp_path / "traj.pt"
        env.save_trajectory(str(path))
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    rec = torch.load(path, weights_only=False)
    rec_states = rec["states"]

    env2 = ReplayTestEnv(record_trajectory=False, num_envs=2, device="cpu")
    env2 = ReplayWrapper(env2, str(path), mode="dynamic")
    try:
        env2.reset()
        for i in range(n):
            obs, reward, term, trunc, info = env2.step(None)
            inner = env2.env
            # Robot qpos is driven by the recorded action target -> tracks closely.
            expected = (
                rec_states["robot"]["qpos"][:, i + 1]
                if i + 1 < n
                else recorded_final_qpos
            )
            assert torch.allclose(inner.robot.get_qpos(), expected, atol=0.05)
        # Dynamic replay keeps the auto-reset guard engaged.
        assert inner._replay_no_auto_reset is True
    finally:
        env2.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()


def test_single_env_trajectory_broadcasts_to_many(tmp_path):
    env = ReplayTestEnv(record_trajectory=True, num_envs=1, device="cpu")
    try:
        env.reset()
        _drive(env, num_steps=3)
        path = tmp_path / "traj.pt"
        env.save_trajectory(str(path))
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    env2 = ReplayTestEnv(record_trajectory=False, num_envs=2, device="cpu")
    env2 = ReplayWrapper(env2, str(path), mode="kinematic")
    try:
        env2.reset()
        obs, _, _, trunc, _ = env2.step(None)
        assert obs["robot"]["qpos"].shape[0] == 2  # broadcast to 2 envs
    finally:
        env2.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()


def test_close_restores_physics(tmp_path):
    env = ReplayTestEnv(record_trajectory=True, num_envs=1, device="cpu")
    try:
        env.reset()
        _drive(env, num_steps=2)
        path = tmp_path / "traj.pt"
        env.save_trajectory(str(path))
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    env2 = ReplayTestEnv(record_trajectory=False, num_envs=1, device="cpu")
    env2 = ReplayWrapper(env2, str(path), mode="kinematic")
    inner = env2.env
    closed = False
    try:
        with patch.object(
            inner.sim, "enable_physics", wraps=inner.sim.enable_physics
        ) as spy:
            env2.reset()
            env2.close()
            closed = True
        spy.assert_any_call(True)
        assert inner._replay_no_auto_reset is False
    finally:
        if not closed:
            env2.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()


def test_replay_respects_per_env_lengths(tmp_path):
    """Replay truncates each env at its own recorded length."""
    env = ReplayTestEnv(record_trajectory=True, num_envs=2, device="cpu")
    try:
        env.reset()
        _drive(env, num_steps=5)
        path = tmp_path / "traj.pt"
        env.save_trajectory(str(path))
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    # Override lengths: env0 -> 3 steps, env1 -> 5 steps.
    data = torch.load(path, weights_only=False)
    data["meta"]["lengths"] = [3, 5]
    torch.save(data, path)

    env2 = ReplayTestEnv(record_trajectory=False, num_envs=2, device="cpu")
    env2 = ReplayWrapper(env2, str(path), mode="kinematic")
    try:
        env2.reset()
        trunc = torch.zeros(2, dtype=torch.bool)
        for step_i in range(5):
            _, _, _, t, _ = env2.step(None)
            trunc = trunc | t
            if step_i == 2:  # after 3 steps (0, 1, 2)
                assert bool(t[0]) and not bool(
                    t[1]
                ), f"after step 3: env0 should be truncated, env1 not; got {t.tolist()}"
        assert bool(trunc[0]) and bool(trunc[1])  # both eventually done
    finally:
        env2.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()


def test_auto_save_at_episode_end(tmp_path):
    save_dir = tmp_path / "trajs"
    env = ReplayTestEnv(record_trajectory=True, num_envs=2, device="cpu")
    env.cfg.trajectory_save_dir = str(save_dir)
    env.cfg.trajectory_auto_save = True
    try:
        env.reset()
        _drive(env, num_steps=4)
        # Trigger an episode-end reset for env 0 only.
        env.reset(options={"reset_ids": torch.tensor([0])})
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    files = list(save_dir.glob("*.pt"))
    env0_files = [f for f in files if f.name.startswith("traj_env0_")]
    assert len(env0_files) == 1, f"expected 1 auto-saved file for env 0, got {files}"
    data = torch.load(env0_files[0], weights_only=False)
    assert data["meta"]["env_ids"] == [0]
    assert data["meta"]["lengths"] == [4]


def test_close_discards_uncommitted_trajectory(tmp_path):
    save_dir = tmp_path / "trajs"
    env = ReplayTestEnv(record_trajectory=True, num_envs=2, device="cpu")
    env.cfg.trajectory_save_dir = str(save_dir)
    env.cfg.trajectory_auto_save = True
    try:
        env.reset()
        _drive(env, num_steps=3)
        env.close()
    finally:
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    files = list(save_dir.glob("*.pt"))
    assert files == []


def test_async_envs_do_not_corrupt_recording():
    """env0 terminates early; env1 keeps recording without being overwritten."""
    env = ReplayTestEnv(record_trajectory=True, num_envs=2, device="cpu")
    try:
        env.reset()
        _drive(env, num_steps=3)
        # env0 "finishes" its episode at step 3 -> its counter resets to 0.
        env.reset(options={"reset_ids": torch.tensor([0])})
        # env1 continues for 2 more steps; env0 records a new episode from 0.
        _drive(env, num_steps=2)
        # env1 should have 5 recorded steps; env0 should have 2 (not overwrite env1).
        assert env._traj_steps.tolist() == [2, 5]
        # env1 step 4 (the 5th) was recorded and is intact.
        env1_qpos_step4 = env._traj_buffer["states"]["robot"]["qpos"][1, 4]
        assert not torch.all(env1_qpos_step4 == 0)
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()


@register_env("ReplayDeltaTask-v1", max_episode_steps=100, override=True)
class ReplayDeltaEnv(EmbodiedEnv):
    """UR10 + dynamic cube with a delta-qpos ActionManager for replay tests."""

    def __init__(
        self,
        record_trajectory: bool = True,
        num_envs: int = 1,
        device: str = "cpu",
        **kwargs,
    ):
        cfg = EmbodiedEnvCfg()
        cfg.num_envs = num_envs
        cfg.max_episode_steps = 100
        cfg.sim_cfg = SimulationManagerCfg(headless=True, sim_device=device)
        cfg.robot = RobotCfg(
            uid="UR10",
            fpath=get_data_path("UniversalRobots/UR10/UR10.urdf"),
            init_pos=(0.0, 0.0, 1.0),
            drive_pros=JointDrivePropertiesCfg(drive_type="force"),
        )
        cfg.rigid_object = [
            RigidObjectCfg(
                uid="cube",
                shape=CubeCfg(size=[0.03, 0.03, 0.03]),
                init_pos=(0.0, 0.0, 0.5),
                body_type="dynamic",
            )
        ]
        cfg.actions = {
            "arm": ActionTermCfg(func=DeltaQposTerm, mode="pre", params={"scale": 1.0})
        }
        cfg.record_trajectory = record_trajectory
        cfg.trajectory_auto_save = False
        super().__init__(cfg, **kwargs)


def test_dynamic_replay_with_action_manager(tmp_path):
    """Dynamic replay feeds pre-process (delta) action; ActionManager re-applies it."""
    env = ReplayDeltaEnv(record_trajectory=True, num_envs=1, device="cpu")
    try:
        env.reset()
        init_qpos = env.robot.get_qpos().clone()
        deltas = []
        recorded_post_qpos = []
        for i in range(4):
            d = torch.zeros_like(init_qpos)
            d[:, 0] = 0.05 * (i + 1)
            deltas.append(d)
            env.step(d)
            recorded_post_qpos.append(env.robot.get_qpos().clone())
        path = tmp_path / "delta.pt"
        env.save_trajectory(str(path))
        # Recorded action must be the raw delta (pre-process), not the resolved qpos.
        rec = torch.load(path, weights_only=False)
        assert torch.allclose(rec["actions"][0, 0], deltas[0][0], atol=1e-6)
        assert torch.allclose(rec["states"]["robot"]["qpos"][:, 0], init_qpos)
        for step in range(1, len(deltas)):
            assert torch.allclose(
                rec["states"]["robot"]["qpos"][:, step],
                recorded_post_qpos[step - 1],
                atol=1e-5,
            )
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    env2 = ReplayDeltaEnv(record_trajectory=False, num_envs=1, device="cpu")
    env2 = ReplayWrapper(env2, str(path), mode="dynamic")
    try:
        env2.reset()
        assert torch.allclose(
            env2.env.robot.get_qpos(),
            rec["states"]["robot"]["qpos"][:, 0],
            atol=1e-5,
        )
        for step in range(len(deltas)):
            obs, reward, term, trunc, info = env2.step(None)
            assert torch.allclose(
                env2.env.robot.get_qpos(), recorded_post_qpos[step], atol=1e-3
            )
        # Auto-reset guard stays engaged during dynamic replay.
        assert env2.env._replay_no_auto_reset is True
    finally:
        env2.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()


def test_control_mode_scrubs_to_recorded_state(tmp_path):
    """Control mode jumps to arbitrary steps and sets the recorded state."""
    env = ReplayTestEnv(record_trajectory=True, num_envs=2, device="cpu")
    try:
        env.reset()
        _drive(env, num_steps=5)
        path = tmp_path / "traj.pt"
        env.save_trajectory(str(path))
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    rec = torch.load(path, weights_only=False)
    rec_states = rec["states"]

    env2 = ReplayTestEnv(record_trajectory=False, num_envs=2, device="cpu")
    rw = ReplayWrapper(env2.unwrapped, str(path), mode="control")
    try:
        rw.reset()
        # State index t is the state immediately before action t.
        rw.go_to_step(3)
        assert torch.allclose(
            rw.env.robot.get_qpos(), rec_states["robot"]["qpos"][:, 3], atol=1e-4
        )
        # out-of-range clamps to last step
        rw.go_to_step(999)
        max_step = int(rw._lengths.min().item()) - 1
        assert torch.allclose(
            rw.env.robot.get_qpos(), rec_states["robot"]["qpos"][:, max_step], atol=1e-4
        )
        # jump back to the first pre-action state
        rw.go_to_step(0)
        assert torch.allclose(
            rw.env.robot.get_qpos(),
            rec_states["robot"]["qpos"][:, 0],
            atol=1e-4,
        )
    finally:
        rw.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()


def test_control_loop_uses_single_keys_and_auto_can_be_interrupted():
    """Single-key commands act immediately and interrupt auto playback."""
    from embodichain.lab.scripts.run_env import _run_replay_control_loop

    class FakeControlInput:
        single_key = True

        def __init__(self):
            self.keys = iter(["n", "p", "a", None, "p", "q"])
            self.timeouts = []

        def read_key(self, timeout=None):
            if timeout is not None:
                self.timeouts.append(timeout)
            return next(self.keys)

    class FakeReplayEnv:
        def __init__(self):
            self._lengths = torch.tensor([5])
            self.env = SimpleNamespace(
                sim_cfg=SimpleNamespace(physics_dt=0.01),
                cfg=SimpleNamespace(sim_steps_per_control=4),
            )
            self.visited_steps = []

        def go_to_step(self, step):
            self.visited_steps.append(step)

    replay_env = FakeReplayEnv()
    control_input = FakeControlInput()
    _run_replay_control_loop(replay_env, control_input)

    # n and p execute as individual key reads. During auto, p interrupts at
    # step 2 and is then applied immediately, moving back to step 1.
    assert replay_env.visited_steps == [0, 1, 0, 1, 2, 1]
    assert control_input.timeouts == [0.04, 0.04]


def test_run_env_replay_function(tmp_path):
    """run_env.replay() drives the full trajectory without error (kinematic + dynamic)."""
    from embodichain.lab.scripts.run_env import replay

    # record
    env = ReplayTestEnv(record_trajectory=True, num_envs=1, device="cpu")
    try:
        env.reset()
        _drive(env, num_steps=4)
        path = tmp_path / "traj.pt"
        env.save_trajectory(str(path))
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    # replay() borrows the environment; the caller remains responsible for close().
    env_k = ReplayTestEnv(record_trajectory=False, num_envs=1, device="cpu")
    try:
        replay(env_k, str(path), mode="kinematic")
    finally:
        env_k.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()

    # dynamic replay on a fresh env
    env_d = ReplayTestEnv(record_trajectory=False, num_envs=1, device="cpu")
    try:
        replay(env_d, str(path), mode="dynamic")
    finally:
        env_d.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()
