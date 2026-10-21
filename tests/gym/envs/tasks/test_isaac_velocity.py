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

"""Compare packaged velocity definitions with executed Isaac reference values."""

from __future__ import annotations

import json
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dexsim.spawn import (
    CollisionDesc,
    GeometryDesc,
    NewtonCollisionDesc,
    ObjectDesc,
    RenderDesc,
)

from embodichain.lab.sim.sim_manager import SimulationManager
from embodichain_tasks.locomotion.velocity.isaac.curriculum import (
    ReferenceTerrainCurriculum,
)
from embodichain_tasks.locomotion.velocity.isaac.config import load_config
from embodichain_tasks.locomotion.velocity.isaac.environment import (
    load_reference_env_cfg,
)
from embodichain_tasks.locomotion.velocity.isaac.rsl_rl import ReferenceRslRlEnv
from embodichain_tasks.locomotion.velocity.isaac.runtime import IsaacVelocityEnv
from embodichain_tasks.locomotion.velocity.isaac.terrain import ReferenceHeightScanner
from embodichain_tasks.locomotion.velocity.isaac.mdp import (
    IsaacVelocityState,
    action_target,
    build_observation,
    compute_rewards,
    compute_termination,
    dc_motor_effort,
    update_contact_times,
)

__all__ = []


@pytest.mark.no_sim
def test_reference_observation_preserves_reset_contact_rows() -> None:
    """Observation reads preserve cleared contact rows and update other sensors."""
    calls = []
    contacts = SimpleNamespace(
        update=lambda **kwargs: pytest.fail("reset contacts must not be queried"),
        get_data=lambda: {"is_valid": torch.zeros((2, 1), dtype=torch.bool)},
    )
    camera = SimpleNamespace(
        requires_substep_update=False,
        update=lambda **kwargs: calls.append(kwargs),
        get_data=lambda: torch.ones((2, 1)),
    )
    env = SimpleNamespace(
        sim=SimpleNamespace(render_camera_group=lambda ids: calls.append(ids)),
        _camera_group_ids=[3],
        num_envs=2,
        device=torch.device("cpu"),
        sensors={"locomotion_contacts": contacts, "camera": camera},
        _reference_sensor=contacts,
    )
    obs = IsaacVelocityEnv._get_sensor_obs(env)
    assert not obs["locomotion_contacts"]["is_valid"].any()
    assert obs["camera"].eq(1).all()
    assert calls == [[3], {"fetch_only": True}]


@pytest.mark.no_sim
def test_reference_uses_control_frame_contacts_without_parent_histories() -> None:
    """Reference normal-force histories replace the generic substep histories."""
    from embodichain_tasks.locomotion.velocity._embodichain import (
        EmbodiChainVelocityEnv,
    )

    registered = []
    sensor = SimpleNamespace(
        get_actor_ids=lambda *args: torch.tensor([[1]]),
        create_history=lambda name, *args, **kwargs: registered.append(name),
    )
    env = object.__new__(EmbodiChainVelocityEnv)
    env.get_sensor = lambda name: sensor
    env.robot = SimpleNamespace(uid="robot")
    env._state_fields = frozenset()
    env._initialize_contact_histories()
    assert registered == ["feet", "illegal"]
    registered.clear()
    reference = object.__new__(IsaacVelocityEnv)
    reference.get_sensor = lambda name: sensor
    reference._initialize_contact_histories()
    assert registered == []


@pytest.mark.no_sim
@pytest.mark.parametrize("native_go2", [False, True])
def test_joint_acceleration_matches_reference_update_interval(
    monkeypatch, native_go2
) -> None:
    """Match G1's final 5 ms derivative and Go2's fused 20 ms derivative."""
    from embodichain_tasks.locomotion.velocity._embodichain import (
        EmbodiChainVelocityEnv,
    )

    env = object.__new__(IsaacVelocityEnv)
    env.sim_cfg = SimpleNamespace(physics_dt=0.005)
    env.cfg = SimpleNamespace(sim_steps_per_control=4)
    env.policy_joint_ids = torch.arange(2)
    env.explicit_pd_effort_control = native_go2
    velocities = torch.tensor(
        [[[0.0, 0.0]], [[4.0, -2.0]], [[-4.0, 3.0]], [[2.0, -1.0]], [[3.0, 1.0]]]
    )
    step = 0
    sampled_steps = []

    def fetch_state():
        sampled_steps.append(step)
        return {"qpos": torch.zeros(1, 2), "qvel": velocities[step]}

    def advance(dt, count, **kwargs):
        nonlocal step
        assert dt == 0.005 and count == 1
        step += 1

    def update_control_state(self, **kwargs):
        current = fetch_state()["qvel"]
        self._joint_acceleration.copy_(
            (current - self._previous_joint_velocity) / self.step_dt
        )
        self._previous_joint_velocity.copy_(current)

    env.robot = SimpleNamespace(
        body_data=SimpleNamespace(fetch_state=fetch_state), set_qf=lambda *a, **kw: None
    )
    env.sim = SimpleNamespace(update=advance)
    env._position_target = torch.zeros(1, 2)
    env._kp = torch.ones(2)
    env._kd = torch.ones(2)
    env._effort_limit = torch.full((2,), 100.0)
    env._velocity_limit = torch.full((2,), 30.0)
    env._applied_effort = torch.zeros(1, 2)
    env._previous_joint_velocity = torch.zeros(1, 2)
    env._previous_physics_joint_velocity = torch.zeros(1, 2)
    env._joint_acceleration = torch.zeros(1, 2)
    env._sample_reference_contacts = lambda: None
    monkeypatch.setattr(
        EmbodiChainVelocityEnv, "_update_sim_state", update_control_state
    )
    env._advance_physics()
    assert sampled_steps == ([0, 1, 2, 3] if native_go2 else [3])
    torch.testing.assert_close(env._applied_effort, -velocities[3])
    env._update_sim_state()
    expected = [[150.0, 50.0]] if native_go2 else [[200.0, 400.0]]
    torch.testing.assert_close(env._joint_acceleration, torch.tensor(expected))


@pytest.mark.no_sim
def test_control_state_read_is_reused_by_reference_managers() -> None:
    """Acceleration and task managers consume one live read per physics frame."""
    env = object.__new__(IsaacVelocityEnv)
    env._num_envs = 1
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env.sim_cfg = SimpleNamespace(physics_dt=0.005)
    env.cfg = SimpleNamespace(sim_steps_per_control=4, events=None)
    env._state_fields = frozenset({"joint_acc"})
    env.explicit_pd_effort_control = True
    env.policy_joint_ids = torch.arange(2)
    env.foot_link_ids = torch.arange(1)
    env.foot_link_names = ("foot",)
    env._reference_events_deferred = True
    env._height_scanner = None
    env._episode_step = torch.zeros(1, dtype=torch.long)
    env._global_control_step = 0
    env._gait_process = torch.zeros(1)
    env._gait_frequency = torch.zeros(1)
    env.command = torch.zeros(1, 3)
    env.reward_command = torch.zeros(1, 3)
    for name in (
        "_joint_acceleration",
        "_previous_joint_velocity",
        "_applied_effort",
        "locomotion_action",
        "last_locomotion_action",
    ):
        setattr(env, name, torch.zeros(1, 2))
    env._normal_history = torch.zeros(1, 3, 1, 3)
    for name in ("_air_time", "_contact_time", "_last_air_time"):
        setattr(env, name, torch.zeros(1, 1))
    velocity = torch.ones(1, 2)
    reads = []

    def fetch_state():
        reads.append(1)
        return dict(
            qpos=torch.zeros(1, 2),
            qvel=velocity.clone(),
            root_pose=torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
            root_lin_vel=torch.zeros(1, 3),
            root_ang_vel=torch.zeros(1, 3),
        )

    env.robot = SimpleNamespace(
        body_data=SimpleNamespace(
            fetch_state=fetch_state, body_link_vel=torch.zeros(1, 1, 6)
        )
    )
    for frame in (1, 2):
        velocity.fill_(frame)
        env._update_sim_state()
        state = env.get_velocity_locomotion_state()
        assert len(reads) == frame
        torch.testing.assert_close(state.joint_vel, velocity)
        torch.testing.assert_close(state.joint_acc, torch.full((1, 2), 50.0))


@pytest.mark.no_sim
@pytest.mark.parametrize("task, dimension", [("g1_isaac_rough", 310)])
@pytest.mark.parametrize("cached", [False, True])
def test_executed_isaac_reference(task: str, dimension: int, cached: bool) -> None:
    """Use golden values exported by the actual Isaac MDP functions."""
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / f"{task}.json").read_text()
    )
    state = IsaacVelocityState(
        **{
            name: torch.tensor(value, dtype=torch.float32)
            for name, value in fixture["state"].items()
        }
    )
    config = load_config(task)
    from embodichain_tasks.locomotion.velocity.isaac.mdp import IsaacVelocityConstants

    constants = (
        IsaacVelocityConstants.from_config(config, state.joint_pos) if cached else None
    )
    noise = {name: torch.tensor(value) for name, value in fixture["noise"].items()}
    observed = build_observation(config, state, noise=noise, constants=constants)
    assert observed.shape == (4, dimension)
    torch.testing.assert_close(
        observed, torch.tensor(fixture["observation"]), rtol=1e-6, atol=1e-6
    )
    terminated, truncated = compute_termination(config, state)
    assert torch.equal(terminated, torch.tensor(fixture["terminated"]))
    assert not truncated.any()
    reward = compute_rewards(config, state, terminated, constants=constants)
    for name, expected in fixture["rewards"].items():
        torch.testing.assert_close(
            reward.raw_terms[name], torch.tensor(expected), rtol=1e-6, atol=1e-6
        )
    expected_total = sum(
        torch.tensor(value) * config.data["rewards"][name]["weight"] * 0.02
        for name, value in fixture["rewards"].items()
    )
    torch.testing.assert_close(reward.total, expected_total, rtol=1e-6, atol=1e-6)
    state.episode_step[0] = 1000
    assert compute_termination(config, state)[1].tolist() == [True, False, False, False]


@pytest.mark.no_sim
@pytest.mark.parametrize("task", ["g1_isaac_rough"])
def test_reference_action_retains_large_policy_output(task: str) -> None:
    """Outputs above one must reach the reference offset mapping unchanged."""
    config = load_config(task)
    action = torch.full((2, config.action_dim), 3.0)
    target = action_target(config, action)
    expected = torch.tensor(config.default_joint_position) + 3 * torch.tensor(
        config.action_scale
    )
    torch.testing.assert_close(target, expected.expand_as(target))


@pytest.mark.no_sim
def test_motor_brakes_at_overspeed() -> None:
    """The four-quadrant envelope retains opposing torque beyond free speed."""
    result = dc_motor_effort(
        torch.tensor([100.0, 100.0, -100.0, -100.0]),
        torch.tensor([0.0, 30.0, -30.0, -90.0]),
        torch.tensor(20.0),
        torch.tensor(30.0),
        torch.tensor(20.0),
    )
    torch.testing.assert_close(result, torch.tensor([20.0, 0.0, 0.0, 20.0]))


@pytest.mark.no_sim
def test_contact_transition_keeps_elapsed_air_and_contact_time() -> None:
    """Landing and liftoff include the elapsed sensor interval in the last mode."""
    air = torch.tensor([[0.1, 0.0]])
    contact = torch.tensor([[0.0, 0.2]])
    last_air = torch.zeros_like(air)
    last_contact = torch.zeros_like(air)
    update_contact_times(
        torch.tensor([[[0.0, 0.0, 0.00001], [0.0, 0.0, 0.0]]]),
        0.02,
        air,
        contact,
        last_air,
        last_contact,
    )
    torch.testing.assert_close(air, torch.tensor([[0.0, 0.02]]))
    torch.testing.assert_close(contact, torch.tensor([[0.02, 0.0]]))
    torch.testing.assert_close(last_air, torch.tensor([[0.12, 0.0]]))
    torch.testing.assert_close(last_contact, torch.tensor([[0.0, 0.22]]))


@pytest.mark.no_sim
def test_ground_replacement_updates_declared_geometry() -> None:
    """A shallow scene declaration must retain the modified ground descriptors."""
    original = GeometryDesc.plane(10.0)
    descriptor = ObjectDesc(
        name="ground",
        collisions=[CollisionDesc.from_geometry(original)],
        renders=[RenderDesc.from_geometry(original)],
    )
    declared = copy(descriptor)
    sim = object.__new__(SimulationManager)
    sim._spawn_default_plane_descriptor = descriptor
    mesh = GeometryDesc.mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=np.float32),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
    )
    collision = NewtonCollisionDesc(heightfield_resolution=0.1)
    sim.configure_ground_plane_collision(newton=collision, geometry=mesh)
    assert declared.collisions[0].geometry_type == mesh.geometry_type
    assert declared.collisions[0].newton is collision
    np.testing.assert_array_equal(declared.collisions[0].vertices, mesh.vertices)
    np.testing.assert_array_equal(declared.renders[0].triangles, mesh.triangles)
    sim._spawn_scene = SimpleNamespace(
        builder=SimpleNamespace(is_finalized=True, result=object())
    )
    with pytest.raises(RuntimeError, match="before prepare"):
        sim.configure_ground_plane_collision(enabled=False)


@pytest.mark.no_sim
def test_rsl_wrapper_preserves_actions_and_timeout_bootstrapping() -> None:
    """Failures end episodes while only timeouts bootstrap the value estimate."""
    actions = torch.full((2, 12), 4.0)
    received = []

    def step(value: torch.Tensor, *, policy_observations_only: bool) -> tuple:
        assert policy_observations_only
        received.append(value)
        return (
            {"policy": torch.zeros(2, 48)},
            torch.ones(2),
            torch.tensor([True, False]),
            torch.tensor([False, True]),
            {},
        )

    inner = SimpleNamespace(
        num_envs=2,
        velocity_task_config=SimpleNamespace(action_dim=12),
        max_episode_steps=1000,
        device=torch.device("cpu"),
        cfg=None,
        step=step,
    )
    wrapped = ReferenceRslRlEnv(inner)
    _, _, done, extras = wrapped.step(actions)
    assert received[0] is actions
    assert done.tolist() == [True, True]
    assert extras["time_outs"].tolist() == [False, True]


@pytest.mark.no_sim
def test_height_scanner_keeps_yaw_frame_and_xy_order() -> None:
    """Check all 187 rays against an analytic inclined plane after a yaw rotation."""
    vertices = np.array(
        [[-5, -5, -1.5], [5, -5, -0.5], [5, 5, 1.5], [-5, 5, 0.5]], dtype=np.float32
    )
    scanner = ReferenceHeightScanner(
        vertices,
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        1,
        torch.device("cpu"),
    )
    pose = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 2**-0.5, 2**-0.5]])
    result = scanner.compute(pose)
    x, y = torch.meshgrid(
        torch.arange(-0.8, 0.8 + 1e-9, 0.1),
        torch.arange(-0.5, 0.5 + 1e-9, 0.1),
        indexing="xy",
    )
    expected = 0.5 + 0.1 * y.flatten() - 0.2 * x.flatten()
    torch.testing.assert_close(result[0], expected, atol=5e-6, rtol=0)


@pytest.mark.no_sim
def test_height_scanner_queries_physics_heightfield_instead_of_source_mesh() -> None:
    """Read the collision surface when its height differs from the source mesh."""
    import newton

    heightfield = newton.Heightfield(
        data=np.ones((3, 3), dtype=np.float32),
        nrow=3,
        ncol=3,
        hx=2.0,
        hy=2.0,
        min_z=1.0,
        max_z=1.0,
    )
    builder = newton.ModelBuilder()
    builder.add_shape_heightfield(heightfield=heightfield)
    model = builder.finalize(device="cpu")
    # The source mesh is at z=0; the collision heightfield is at z=1 m.
    vertices = np.array(
        [[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], dtype=np.float32
    )
    scanner = ReferenceHeightScanner(
        vertices,
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        1,
        torch.device("cpu"),
        physics_model=model,
        physics_state=model.state(),
    )
    pose = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    expected = torch.full((1, 187), -0.5)  # torso_z - terrain_z - 0.5 m
    torch.testing.assert_close(scanner.compute(pose), expected, atol=5e-6, rtol=0)


@pytest.mark.no_sim
def test_contact_impulse_uses_internal_solver_interval() -> None:
    """0.025 N s from a 2.5 ms solver interval is 10 N; invalid slots stay zero."""
    env = object.__new__(IsaacVelocityEnv)
    env._num_envs = 1
    env._device = torch.device("cpu")
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env.sim_cfg = SimpleNamespace(
        physics_dt=0.005, physics_cfg=SimpleNamespace(num_substeps=2)
    )
    env.cfg = SimpleNamespace(sim_cfg=env.sim_cfg, sim_steps_per_control=4)
    data = {
        "user_ids": torch.tensor([[[0, 1], [0, 1]]]),
        "is_valid": torch.tensor([[True, False]]),
        "normal": torch.tensor([[[0.0, 0.0, 1.0], [float("nan"), 0.0, 0.0]]]),
        "impulse": torch.tensor([[0.025, float("nan")]]),
    }
    env._reference_sensor = SimpleNamespace(update=lambda: None, get_data=lambda: data)
    env._sorted_keys = torch.tensor([1])
    env._sorted_rows = torch.tensor([0])
    env._normal_force = torch.zeros(1, 1, 3)
    env._normal_history = torch.zeros(1, 3, 1, 3)
    env._air_time = torch.zeros(1, 1)
    env._contact_time = torch.zeros(1, 1)
    env._last_air_time = torch.zeros(1, 1)
    env._last_contact_time = torch.zeros(1, 1)
    env._sample_reference_contacts()
    torch.testing.assert_close(env._normal_force, torch.tensor([[[0.0, 0.0, 10.0]]]))
    torch.testing.assert_close(env._contact_time, torch.tensor([[0.02]]))


@pytest.mark.no_sim
def test_rough_curriculum_matches_executed_isaac_resets() -> None:
    """Match source assignments, threshold cases, selected resets and overflow."""
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/terrain_curriculum.json").read_text()
    )
    origins = torch.tensor(fixture["origins"], dtype=torch.float32)
    curriculum = ReferenceTerrainCurriculum(
        origins, 20, 5, 8.0, 20.0, torch.Generator().manual_seed(fixture["seed"])
    )
    torch.testing.assert_close(
        curriculum.levels, torch.tensor(fixture["initial"]["levels"]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        curriculum.types, torch.tensor(fixture["initial"]["types"]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        curriculum.env_origins,
        torch.tensor(fixture["initial"]["env_origins"]),
        rtol=0,
        atol=0,
    )
    for case in fixture["cases"]:
        ids = torch.tensor(case["ids"])
        if case["stage"] in (4, 5):
            level = 9 if case["stage"] == 4 else 0
            curriculum.levels[ids] = level
            curriculum.env_origins[ids] = origins[level, curriculum.types[ids]]
        untouched = torch.ones(20, dtype=torch.bool)
        untouched[ids] = False
        before_levels = curriculum.levels.clone()
        before_origins = curriculum.env_origins.clone()
        curriculum.update(
            ids, torch.tensor(case["positions"]), torch.tensor(case["commands"])
        )
        torch.testing.assert_close(
            curriculum.levels, torch.tensor(case["levels"]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            curriculum.env_origins, torch.tensor(case["env_origins"]), rtol=0, atol=0
        )
        assert torch.equal(curriculum.levels[untouched], before_levels[untouched])
        assert torch.equal(curriculum.env_origins[untouched], before_origins[untouched])


@pytest.mark.no_sim
def test_rough_reset_curriculum_uses_terminal_step_command(monkeypatch) -> None:
    """A timeout must use the command that generated the terminal reward."""
    origins = torch.zeros((10, 1, 3))
    curriculum = ReferenceTerrainCurriculum(
        origins, 1, 5, 8.0, 20.0, torch.Generator().manual_seed(42)
    )
    curriculum.levels[:] = 3
    env = object.__new__(IsaacVelocityEnv)
    env._device = torch.device("cpu")
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env._num_envs = 1
    env._reference_terrain_course = curriculum
    env.robot = SimpleNamespace(
        body_data=SimpleNamespace(
            root_pose=torch.tensor([[2.0, 0.0, 0.74, 0.0, 0.0, 0.0, 1.0]])
        )
    )
    env.reward_command = torch.tensor([[0.4, 0.0, 0.0]])
    env.command = torch.tensor([[0.1, 0.0, 0.0]])

    class ResetRecorded(Exception):
        pass

    def stop_after_curriculum(self, **kwargs):
        raise ResetRecorded

    from embodichain_tasks.locomotion.velocity._embodichain import (
        EmbodiChainVelocityEnv,
    )

    monkeypatch.setattr(
        EmbodiChainVelocityEnv, "_initialize_episode", stop_after_curriculum
    )
    with pytest.raises(ResetRecorded):
        env._initialize_episode()
    # Two meters is below the old command's four-meter threshold; the next
    # command's one-meter threshold would incorrectly keep the level unchanged.
    assert curriculum.levels.item() == 2


@pytest.mark.no_sim
def test_reference_step_orders_events_after_rewards_and_resets(monkeypatch) -> None:
    """Commands and pushes must precede only the returned observation."""
    from embodichain_tasks.locomotion.velocity._embodichain import (
        EmbodiChainVelocityEnv,
    )

    events = []
    env = object.__new__(IsaacVelocityEnv)

    def parent_step(self, action, **kwargs):
        self._advance_command_timer()
        self._update_heading_commands()
        events.extend(["rewards", "reset"])
        return (
            {},
            torch.ones(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {},
        )

    monkeypatch.setattr(EmbodiChainVelocityEnv, "step", parent_step)
    monkeypatch.setattr(
        EmbodiChainVelocityEnv,
        "_advance_command_timer",
        lambda self: events.append("commands"),
    )
    monkeypatch.setattr(
        EmbodiChainVelocityEnv,
        "_update_heading_commands",
        lambda self: events.append("heading"),
    )
    monkeypatch.setattr(
        IsaacVelocityEnv,
        "_apply_reference_interval_events",
        lambda self: events.append("push"),
    )
    monkeypatch.setattr(
        IsaacVelocityEnv,
        "get_obs",
        lambda self, **kwargs: events.append("observation") or {},
    )
    env.step(torch.zeros(1, 37))
    assert events == ["rewards", "reset", "commands", "heading", "push", "observation"]
    assert not env._reference_events_deferred


@pytest.mark.no_sim
@pytest.mark.parametrize("recording", [False, True])
@pytest.mark.parametrize("task_managers", [False, True])
def test_training_step_defers_only_unused_observations(
    monkeypatch, recording, task_managers
) -> None:
    from embodichain_tasks.locomotion.velocity._embodichain import (
        EmbodiChainVelocityEnv,
    )

    env = object.__new__(IsaacVelocityEnv)
    env.cfg = SimpleNamespace(events=SimpleNamespace())
    env._num_envs = 2
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env.event_manager = SimpleNamespace(available_modes=[])
    env.rollout_buffer = object() if recording else None
    env.dataset_manager = None
    env._traj_buffer = None
    env._reference_task_managers_only = task_managers
    calls = []
    raw_reads = []
    generator = torch.Generator().manual_seed(42)

    def observe(self, **kwargs):
        calls.append("observe")
        noise = (
            torch.rand(2, 3, generator=generator)
            if not self._reference_events_deferred
            else 0
        )
        return {"policy": torch.ones(2, 3) + noise}

    def raw_observe(self, **kwargs):
        raw_reads.append(True)
        return observe(self, **kwargs)

    def parent_step(self, action, **kwargs):
        self.get_obs()
        self.get_obs()
        return (
            {},
            torch.ones(2),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            {},
        )

    monkeypatch.setattr(EmbodiChainVelocityEnv, "get_obs", raw_observe)
    monkeypatch.setattr(
        IsaacVelocityEnv, "_extend_obs", lambda self, obs, **kwargs: observe(self)
    )
    monkeypatch.setattr(EmbodiChainVelocityEnv, "step", parent_step)
    monkeypatch.setattr(IsaacVelocityEnv, "_advance_command_timer", lambda self: None)
    monkeypatch.setattr(IsaacVelocityEnv, "_update_heading_commands", lambda self: None)
    monkeypatch.setattr(
        IsaacVelocityEnv, "_apply_reference_interval_events", lambda self: None
    )

    obs, *_ = env.step(torch.zeros(2, 12), policy_observations_only=True)

    expected_generator = torch.Generator().manual_seed(42)
    expected = torch.ones(2, 3) + torch.rand(2, 3, generator=expected_generator)
    torch.testing.assert_close(obs["policy"], expected, rtol=0, atol=0)
    assert torch.equal(generator.get_state(), expected_generator.get_state())
    assert len(calls) == (3 if recording or not task_managers else 1)
    assert len(raw_reads) == (3 if recording or not task_managers else 0)
    assert not env._reference_defer_observations


@pytest.mark.no_sim
@pytest.mark.parametrize("task", ["g1_isaac_rough"])
def test_reference_reset_writes_once_and_preserves_sampled_state(monkeypatch, task):
    """Deferred parent writes preserve final states, RNG and untouched rows."""
    from embodichain.lab.gym.envs import EmbodiedEnv
    from embodichain_tasks.locomotion.velocity._embodichain import (
        EmbodiChainVelocityEnv,
    )

    monkeypatch.setattr(EmbodiedEnv, "_initialize_episode", lambda *a, **kw: None)
    config = load_config(task)
    rows = torch.tensor([1, 3])

    def make_env(legacy: bool):
        env = object.__new__(IsaacVelocityEnv)
        env._device = torch.device("cpu")
        env._num_envs = 4
        env.sim = SimpleNamespace(device=env._device)
        env.cfg = SimpleNamespace(reference_deterministic_reset=False)
        env.velocity_task_config = config
        env.policy_joint_ids = torch.arange(config.action_dim)
        env._reference_terrain_course = None
        env._reference_initial_states = None
        env._reference_events_deferred = False
        env._locomotion_generator = torch.Generator().manual_seed(42)
        env.command = torch.ones((4, 3))
        env.reward_command = torch.zeros((4, 3))
        env._resample_commands = lambda ids: env.command.__setitem__(
            ids, torch.rand((len(ids), 3), generator=env._locomotion_generator)
        )
        env._update_heading_commands = lambda: None
        for name in (
            "_episode_step",
            "_previous_joint_velocity",
            "_joint_acceleration",
            "_previous_root_velocity_w",
            "_root_acceleration_w",
            "_filtered_base_lin_vel_b",
            "_filtered_base_ang_vel_b",
            "_foot_peak_height",
            "_foot_swing_height_cost",
            "_position_target",
            "_normal_force",
            "_normal_history",
            "_air_time",
            "_contact_time",
            "_last_air_time",
            "_last_contact_time",
            "_applied_effort",
            "_previous_physics_joint_velocity",
            "_push_remaining",
        ):
            setattr(
                env,
                name,
                (
                    torch.zeros((4, config.action_dim))
                    if name == "_position_target"
                    else torch.zeros(4)
                ),
            )
        pose = torch.zeros((4, 7))
        pose[:, 6] = 1
        writes = []

        def set_state(**fields):
            writes.append(
                {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in fields.items()
                }
            )
            pose[fields["env_ids"]] = fields["root_pose"]

        env.robot = SimpleNamespace(
            set_state=set_state, body_data=SimpleNamespace(root_pose=pose)
        )
        if legacy:

            def parent_write(ids, root_pose, joints):
                env._reference_reset_pose = root_pose.clone()
                EmbodiChainVelocityEnv._write_episode_initial_state(
                    env, ids, root_pose, joints
                )

            env._write_episode_initial_state = parent_write
        return env, writes

    legacy, old_writes = make_env(True)
    current, new_writes = make_env(False)
    legacy._initialize_episode(rows)
    current._initialize_episode(rows)
    assert len(old_writes) == 2
    assert len(new_writes) == 1
    for name, value in old_writes[-1].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(new_writes[-1][name], value, rtol=0, atol=0)
        else:
            assert new_writes[-1][name] == value
    assert torch.equal(
        current._locomotion_generator.get_state(),
        legacy._locomotion_generator.get_state(),
    )
    torch.testing.assert_close(
        current.robot.body_data.root_pose[[0, 2]],
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).expand(2, -1),
    )


@pytest.mark.gpu
@pytest.mark.no_sim
def test_height_scanner_graph_replays_changed_poses() -> None:
    """Captured rays read current poses and match the eager physics query."""
    import newton
    import warp as wp

    device = torch.device("cuda:0")
    vertices = np.array(
        [[-3, -3, 0], [3, -3, 0], [3, 3, 0], [-3, 3, 0]], dtype=np.float32
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    builder = newton.ModelBuilder()
    builder.add_shape_plane()
    model = builder.finalize(str(device))
    state = model.state()
    scanner = ReferenceHeightScanner(
        vertices, triangles, 2, device, physics_model=model, physics_state=state
    )
    previous = None
    graph = None
    for shift in (-0.07, 0.0, 0.11):
        pose = torch.tensor(
            [[shift, 0, 0.8 + shift, 0, 0, 0, 1], [0, shift, 1.1 + shift, 0, 0, 0, 1]],
            device=device,
        )
        captured = scanner.compute(pose).clone()
        torch.cuda.synchronize()
        if graph is None:
            graph = scanner._graph
        assert graph is not None and scanner._graph is graph
        with wp.ScopedStream(scanner._scan_stream):
            scanner._compute_height()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, scanner.height, rtol=0, atol=0)
        assert bool(torch.isfinite(captured).all())
        if previous is not None:
            assert not torch.equal(captured, previous)
        previous = captured


@pytest.mark.no_sim
def test_curriculum_checkpoint_restores_assignments_and_rejects_other_maps() -> None:
    origins = torch.arange(36, dtype=torch.float32).reshape(4, 3, 3)

    def make(count=6):
        return ReferenceTerrainCurriculum(
            origins.clone(), count, 3, 8.0, 20.0, torch.Generator().manual_seed(4)
        )

    source = make()
    source.levels.fill_(3)
    source.env_origins.copy_(source.origins[source.levels, source.types])
    saved = source.state_dict()
    target = make()
    target.load_state_dict(saved)
    torch.testing.assert_close(target.levels, source.levels)
    torch.testing.assert_close(target.env_origins, source.env_origins)
    torch.testing.assert_close(
        target.generator.get_state(), source.generator.get_state()
    )
    with pytest.raises(ValueError, match="instance count"):
        make(3).load_state_dict(saved)
    different_map = make()
    different_map.origins.add_(1)
    with pytest.raises(ValueError, match="map differs"):
        different_map.load_state_dict(saved)
    env = SimpleNamespace(
        _reference_terrain_course=target, num_envs=6, device=torch.device("cpu")
    )
    env.validate_training_state = (
        lambda state: IsaacVelocityEnv.validate_training_state(env, state)
    )
    IsaacVelocityEnv.load_training_state(env, {"terrain_course": saved})
    assert env._reference_resume_reset.all()

    before = target.state_dict()
    invalid = {**saved, "levels": torch.full_like(saved["levels"], -1)}
    with pytest.raises(ValueError, match="out of range"):
        env.validate_training_state({"terrain_course": invalid})
    for key, value in before.items():
        torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)
