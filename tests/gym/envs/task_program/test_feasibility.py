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

"""Configured physical evidence and safe joint-feasibility workflow contracts."""

from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest
import torch

from embodichain.lab.task_program.integrations.feasibility import (
    MeasuredTaskMonitor,
    build_feasibility_host,
)
from embodichain.lab.task_program.integrations.simulation.feasibility import (
    save_feasibility_episode,
)
from embodichain.lab.sim.motion.expansion import (
    CandidateIdentity,
    ExpertEpisode,
    ValidationCheck,
    ValidationResult,
)


class Cube:
    def __init__(self):
        self.position = torch.tensor([[0.0, 0.0, 0.025]])
        self.body_state = torch.zeros(1, 13)

    def get_local_pose(self):
        return torch.cat((self.position, torch.tensor([[0.0, 0.0, 0.0, 1.0]])), dim=1)


def environment():
    cube = Cube()
    env = SimpleNamespace(
        sim=SimpleNamespace(
            get_rigid_object_uid_list=lambda: ["cube"],
            get_rigid_object=lambda uid: cube,
        ),
        robot=SimpleNamespace(
            joint_names=["finger"], get_qpos=lambda: torch.zeros(1, 1)
        ),
    )
    return env, cube


def test_monitor_requires_measured_lift_then_stable_terminal_and_rejects_reversed_order():
    goal = {
        "kind": "pose_near",
        "entity": "cube",
        "position": [1.0, 0.0, 0.025],
        "tolerance_m": 0.03,
    }
    lift = {"kind": "lifted", "entity": "cube", "height_m": 0.05}
    monitor = MeasuredTaskMonitor(
        (goal,),
        (
            {"event_id": "lift", "conditions": [lift]},
            {"event_id": "place", "conditions": [goal], "consecutive_steps": 2},
        ),
    )
    env, cube = environment()
    monitor.reset(env)
    assert not monitor.validation(env).accepted
    cube.position[0, 2] = 0.1
    monitor.observe(env, None)
    cube.position[0] = torch.tensor([1.0, 0.0, 0.025])
    monitor.observe(env, None)
    assert not monitor.validation(env).accepted
    monitor.observe(env, None)
    assert monitor.validation(env).accepted
    env, cube = environment()
    monitor.reset(env)
    cube.position[0] = torch.tensor([1.0, 0.0, 0.025])
    monitor.observe(env, None)
    monitor.observe(env, None)
    cube.position[0, 2] = 0.1
    monitor.observe(env, None)
    cube.position[0, 2] = 0.025
    monitor.observe(env, None)
    monitor.observe(env, None)
    assert not monitor.validation(env).accepted


def test_monitor_reads_physical_velocity_and_closed_schema():
    env, cube = environment()
    monitor = MeasuredTaskMonitor(
        (
            {
                "kind": "speed_below",
                "entity": "cube",
                "linear_m_s": 0.03,
                "angular_rad_s": 0.2,
            },
        ),
        (),
    )
    monitor.reset(env)
    assert monitor.validation(env).accepted
    cube.body_state[0, 7] = 0.1
    assert not monitor.validation(env).accepted
    with pytest.raises(ValueError, match="Unknown"):
        MeasuredTaskMonitor(
            (
                {
                    "kind": "lifted",
                    "entity": "cube",
                    "height_m": 0.05,
                    "lower_threshold_on_failure": True,
                },
            ),
            (),
        )
    with pytest.raises(ValueError, match="workflow"):
        build_feasibility_host(
            {"python_factory": "os.system"}, output_directory="/tmp/unused"
        )


def test_episode_writer_confirms_exact_idempotent_evidence(tmp_path):
    identity = CandidateIdentity(
        "scene", "state", "candidate", "family", "source", "revision", "template"
    )
    episode = ExpertEpisode(
        identity,
        {"joint_positions": torch.tensor([[0.0], [0.2]])},
        torch.tensor([[0.2]]),
        torch.tensor([0.0, 0.04]),
        "qpos",
        ValidationResult((ValidationCheck("task_success", "passed"),)),
        "episode",
        "commit",
    )
    first = save_feasibility_episode(episode, tmp_path)
    second = save_feasibility_episode(episode, tmp_path)
    assert first == second and first.confirmed
    with np.load(first.storage_id, allow_pickle=False) as saved:
        assert saved["actions"] == pytest.approx(np.array([[0.2]]))
        assert saved["observation/joint_positions"] == pytest.approx(
            np.array([[0.0], [0.2]])
        )


def test_host_uses_restored_identity_and_releases_resources_on_sink_error(
    monkeypatch, tmp_path
):
    from embodichain.lab.sim.scene.feasibility import (
        TaskFeasibilityRequest,
        TaskOperationMode,
        TaskPhase,
        SceneRobotFeasibilityCfg,
        prepare_feasibility,
        FeasibilityBudget,
        FeasibilityCheck,
        TaskPlan,
        CollisionBody,
    )
    from embodichain.lab.task_program.integrations.simulation import (
        feasibility as module,
    )
    from embodichain.lab.task_program.integrations.scene_expansion import (
        ScenePreparationResult,
    )
    from embodichain.lab.sim import SimulationManager

    monkeypatch.setattr(SimulationManager, "flush_cleanup_queue", lambda: None)
    checks = ValidationResult(
        (ValidationCheck("semantics", "passed"), ValidationCheck("geometry", "passed"))
    )

    class Env:
        step_dt = 0.04

        def __init__(self):
            self.q = torch.zeros(1, 1)
            self.closed = 0
            self.resets = 0
            cube = SimpleNamespace(
                body_type="dynamic", get_local_pose=lambda **kwargs: torch.eye(4)[None]
            )
            self.sim = SimpleNamespace(
                get_rigid_object_uid_list=lambda: ["cube"],
                get_rigid_object=lambda uid: cube,
            )
            self.robot = SimpleNamespace(get_qpos=lambda target=False: self.q.clone())

        def step(self, action, **kwargs):
            self.q = torch.tensor([[0.2]])
            return {"camera": {"rgb": torch.full((1, 2, 2, 3), 99.0)}}, None

        def reset(self, options, seed=None):
            assert options == {"save_data": False}
            self.resets += 1
            return {"camera": {"rgb": torch.full((1, 2, 2, 3), float(self.resets))}}, {}

        def close(self, exit_process):
            self.closed += 1

    env = Env()

    class SceneHost:
        def __init__(self, environment, **kwargs):
            self.initial_state = object()
            self.env = environment

        def __call__(self, variant):
            self.env.reset(options={"save_data": False})
            return ScenePreparationResult("before_restore", checks)

        def restore_initial_state(self, snapshot):
            self.env.reset(options={"save_data": False})
            return ScenePreparationResult("actual_restored", checks)

    monkeypatch.setattr(module, "SimulationSceneExpansionHost", SceneHost)

    class Backend:
        initial_qpos = np.zeros(1)
        joint_limits = np.array([[-1.0, 1.0]])
        joint_names = ("joint",)

        def __init__(self, *args, **kwargs):
            pass

        def frame_poses(self, *args, **kwargs):
            return {}

    monkeypatch.setattr(module, "RobotFeasibilityBackend", Backend)
    monkeypatch.setattr(
        module,
        "plan_task_mode",
        lambda *args: TaskPlan(
            (
                FeasibilityCheck("kinematics", "passed"),
                FeasibilityCheck("motion", "passed"),
            ),
            (np.array([[0.0], [0.2]]),),
            ("move",),
        ),
    )

    def execute(environment, variant, *, prepare_scene, measured_validator):
        actual = prepare_scene(variant)
        environment.step(torch.tensor([[0.2]]))
        assert measured_validator().accepted
        return SimpleNamespace(
            accepted=True,
            program_success=True,
            program_completed=True,
            execution=SimpleNamespace(to_metadata=lambda: {"completed": True}),
            to_metadata=lambda: {"initial_state_id": actual.initial_state_id},
        )

    monkeypatch.setattr(module, "execute_scene_variant", execute)
    req = TaskFeasibilityRequest(
        "task",
        "robot",
        "policy",
        "validator",
        "scene",
        {},
        (),
        (TaskOperationMode("runtime", (TaskPhase("move"),)),),
    )
    captured = []

    def sink(episode):
        captured.append(episode)
        raise ValueError("existing artifact differs")

    checker = SimpleNamespace(
        bodies=(CollisionBody("body", "body", "sphere", (0.1,), robot=True),),
        metadata={"backend": "test-query", "clearance_scope": "test"},
        check=lambda *args, **kwargs: (FeasibilityCheck("collision", "passed"), 0.1),
    )
    host = module.TaskProgramFeasibilityHost(
        environment_factory=lambda deployment: env,
        initial_validator=lambda *args: checks,
        measured_validator=lambda *args: ValidationResult(
            (ValidationCheck("task", "passed"),)
        ),
        collision_factory=lambda env: checker,
        runtime_phase=lambda *args: TaskPhase("move"),
        sink=sink,
        observation_keys=("camera/rgb",),
    )
    budget = FeasibilityBudget(SceneRobotFeasibilityCfg(max_candidates=1))
    host.prepare(req, budget)
    candidate = prepare_feasibility(req).candidate((), mode_id="runtime", ordinal=0)
    with pytest.raises(ValueError, match="existing artifact differs"):
        host.evaluate(candidate, budget)
    assert env.closed == 1 and env.resets >= 1 and host._env is None
    assert captured[0].identity.initial_state_id == "actual_restored"
    assert captured[0].metadata["task_program"]["initial_state_id"] == "actual_restored"
    assert captured[0].observations["joint_positions"].shape == (2, 1)
    image_history = captured[0].observations["env/camera/rgb"]
    assert image_history[:, 0, 0, 0].tolist() == [3.0, 99.0]
    assert captured[0].metadata["terminal_poses"]["cube"][0][0] == 1.0
    assert host.session_snapshot["counts"]["committed"] == 0


def test_required_observations_fail_closed_for_missing_or_oversized_frames():
    from embodichain.lab.task_program.integrations.simulation.feasibility import (
        _selected_observations,
    )
    from embodichain.lab.sim.scene.feasibility import (
        UnsupportedFeasibilityError,
        FeasibilityBudgetExhausted,
    )

    observations = {"camera": {"rgb": torch.zeros(1, 2, 2, 3, dtype=torch.uint8)}}
    with pytest.raises(UnsupportedFeasibilityError, match="unavailable"):
        _selected_observations(observations, ("camera/depth",), 1024)
    with pytest.raises(FeasibilityBudgetExhausted, match="recording_budget"):
        _selected_observations(observations, ("camera/rgb",), 8)


def test_preparation_physics_updates_honor_deadline_and_restore_cleanup_port():
    from embodichain.lab.task_program.integrations.simulation.feasibility import (
        _bounded_simulation_updates,
    )
    from embodichain.lab.sim.scene.feasibility import (
        FeasibilityBudget,
        SceneRobotFeasibilityCfg,
        FeasibilityBudgetExhausted,
    )

    clock = SimpleNamespace(now=0.0)

    def update():
        clock.now += 1.0
        return "updated"

    env = SimpleNamespace(sim=SimpleNamespace(update=update))
    budget = FeasibilityBudget(
        SceneRobotFeasibilityCfg(max_wall_time_s=1.0), clock=lambda: clock.now
    )
    with pytest.raises(FeasibilityBudgetExhausted, match="wall_time"):
        with _bounded_simulation_updates(env, budget):
            env.sim.update()
    assert env.sim.update() == "updated"
