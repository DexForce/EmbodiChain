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

"""Provider-free contracts for the existing-mesh grasp/assembly harness."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import trimesh

from scripts.tools.assemble._json_io import read_json, write_json
from scripts.tools.grasp_assemble._config import (
    load_config,
    load_assembly,
    action_schema,
    execution_settings,
    motion_settings,
)


@pytest.fixture
def job(tmp_path: Path) -> Path:
    assets = {}
    for role in ("base", "assemble"):
        path = tmp_path / f"{role}.obj"
        trimesh.creation.box([0.05, 0.05, 0.1]).export(path)
        assets[role] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    write_json(
        tmp_path / "assembly.json",
        {
            "schema": "codex-assemble/v1",
            "success": True,
            "validation": {"accepted": True},
            "assets": assets,
            "T_base_assemble": np.eye(4).tolist(),
        },
    )
    path = tmp_path / "job.json"
    write_json(
        path,
        {
            "assembly_result": "assembly.json",
            "output_dir": "out",
            "T_world_base": np.eye(4).tolist(),
            "T_world_assemble_initial": np.eye(4).tolist(),
        },
    )
    return path


def test_paths_and_verified_asset_identity(job: Path) -> None:
    config = load_config(job)
    assert config["assembly_result"] == str(job.parent / "assembly.json")
    assert config["output_dir"] == str(job.parent / "out")
    assert load_assembly(config)["success"]
    (job.parent / "assemble.obj").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        load_assembly(config)


def test_failed_assembly_is_not_a_grasp_input(job: Path) -> None:
    source = job.parent / "assembly.json"
    write_json(source, read_json(source) | {"success": False})
    with pytest.raises(ValueError, match="successful"):
        load_assembly(load_config(job))


@pytest.mark.parametrize(
    "change",
    [
        {"robot": "unknown"},
        {"extra": True},
        {"codex": {"max_turns": True}},
        {"motion": {"clearance": -1}},
        {"motion": {"strategy": "unknown"}},
        {"motion": {"strategy": "ik_interp"}},
        {"motion": {"strategy": "motion_gen"}},
        {"motion": {"sample_count": 100}},
        {"motion": {"cartesian_step": 0}},
        {"motion": {"rotation_step_degrees": -1}},
        {"motion": {"velocity_limit": 0}},
        {"motion": {"acceleration_limit": True}},
        {"motion": {"jerk_limit": -1}},
        {"motion": {"hand_duration": 0}},
        {"motion": {"joint_cost_weights": [1, 1, 1, 1, 1]}},
        {"motion": {"joint_cost_weights": [1, 1, 1, 1, 1, -0.2]}},
        {"motion": {"joint_cost_weights": [1, 1, 1, 1, 1, True]}},
        {"motion": {"joint_cost_weights": [1, 1, 1, 1, 1, "0.2"]}},
        {"motion": {"joint_cost_weights": [0, 0, 0, 0, 0, 0]}},
        {"execution": {"grasp_mode": "teleport"}},
        {"execution": {"release_hold_steps": True}},
        {"execution": {"settle_steps": 0}},
        {"execution": {"extra": 1}},
        {"execution": {"rotation_tolerance_degrees": 0}},
        {"execution": {"rotation_tolerance_degrees": 181}},
        {"execution": {"rotation_tolerance_degrees": True}},
        {"layout": {"mode": "unknown"}},
        {"layout": {"x_bounds": [1, -1]}},
        {"layout": {"min_candidates": True}},
        {"layout": {"base_min_robot_distance": 0.9}},
        {"layout": {"max_joint_step_degrees": -1}},
        {"layout": {"mode": "optimize", "min_candidates": 12}},
        {"T_world_base": np.diag([1, -1, -1, 1]).tolist()},
        {"T_world_assemble_initial": np.diag([1, 1, 2, 1]).tolist()},
    ],
)
def test_invalid_grasp_settings_rejected(job: Path, change: dict) -> None:
    write_json(job, read_json(job) | change)
    with pytest.raises(ValueError):
        load_config(job)


def test_legacy_config_keeps_contact_only_execution(job: Path) -> None:
    assert load_config(job)["execution"] == execution_settings()
    assert execution_settings()["grasp_mode"] == "contact"
    assert load_config(job)["layout"]["mode"] == "fixed"
    assert load_config(job)["execution"]["rotation_tolerance_degrees"] == 20.0
    assert load_config(job)["motion"]["joint_cost_weights"] == [1, 1, 1, 1, 1, 0.2]


def test_pose_tolerance_and_joint_weights_round_trip_without_mutating_defaults(
    job: Path,
) -> None:
    weights = [1, 1, 1, 1, 1, 0]
    write_json(
        job,
        read_json(job)
        | {
            "motion": {"joint_cost_weights": weights},
            "execution": {"rotation_tolerance_degrees": 25},
        },
    )
    config = load_config(job)
    assert config["execution"]["rotation_tolerance_degrees"] == 25
    assert config["motion"]["joint_cost_weights"] == weights
    config["motion"]["joint_cost_weights"][-1] = 0.5
    assert motion_settings()["joint_cost_weights"] == [1, 1, 1, 1, 1, 0.2]
    assert read_json(job)["motion"]["joint_cost_weights"] == weights


@pytest.mark.parametrize(
    "value",
    [
        1,
        [],
        {"velocity_limit": float("nan")},
        {"joint_cost_weights": [1, 1, 1, 1, 1, float("inf")]},
        {"joint_cost_weights": [1, 1, 1, 1, 1, float("nan")]},
    ],
)
def test_motion_settings_reject_invalid_runtime_values(value: object) -> None:
    with pytest.raises(ValueError):
        motion_settings(value)


@pytest.mark.parametrize("angle", [float("nan"), float("inf"), -1, "20"])
def test_execution_rejects_invalid_rotation_tolerance(angle: object) -> None:
    with pytest.raises(ValueError, match="rotation_tolerance_degrees"):
        execution_settings({"rotation_tolerance_degrees": angle})


def test_flip_path_preserves_target_frame_and_clears_rack() -> None:
    pytest.importorskip("fcl")
    pytest.importorskip("manifold3d")
    from scripts.tools.grasp_assemble._geometry import (
        interpolate_poses,
        object_waypoints,
    )
    from scipy.spatial.transform import Rotation

    initial = np.eye(4)
    initial[:3, 3] = [0.3, -0.3, 0]
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("z", 0.7).as_matrix()
    base[:3, 3] = [0.4, 0.2, 0]
    relative = np.diag([1, -1, -1, 1.0])
    relative[:3, 3] = [0.1, 0, 0.35]
    target = base @ relative
    rack = trimesh.creation.box([0.2, 0.2, 0.35])
    rack.apply_translation([0.4, 0.2, 0.175])
    mug = trimesh.creation.box([0.07, 0.07, 0.10])
    mug.apply_translation([0, 0, 0.05])
    waypoints = object_waypoints(initial, target, rack, mug, 0.08)
    np.testing.assert_allclose(waypoints[-1], target)
    for pose in waypoints[:-1]:
        world = mug.vertices @ pose[:3, :3].T + pose[:3, 3]
        assert world[:, 2].min() >= rack.bounds[1, 2] + 0.079
    grasp = np.eye(4)
    grasp[:3, 3] = [0.06, 0, 0.05]
    np.testing.assert_allclose((target @ grasp) @ np.linalg.inv(grasp), base @ relative)
    interpolated = interpolate_poses(waypoints[0], waypoints[-3])
    for pose in interpolated:
        assert np.linalg.det(pose[:3, :3]) == pytest.approx(1)


def _task_plan() -> dict:
    return {
        "assemble_initial_rpy_degrees": [0, 0, 0],
        "clearance": 0.08,
        "pre_grasp_distance": 0.06,
        "insertion_direction_base": [0, 0, -1],
        "insertion_distance": 0.10,
        "retract_direction_base": [0, 0, 1],
        "retract_distance": 0.08,
        "reason": "Clear approach and withdrawal for this assembly.",
    }


def _action(
    name: str,
    pose: object = None,
    candidate_id: int | None = None,
    layout: dict | None = None,
) -> dict:
    return {
        "action": name,
        "reason": "Test decision",
        "T_assemble_tcp": pose,
        "candidate_id": candidate_id,
        "layout": layout,
        "task_plan": _task_plan() if name == "evaluate" else None,
    }


def test_grasp_finish_requires_geometry_and_robot_acceptance(
    job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fcl")
    pytest.importorskip("manifold3d")
    from scripts.tools.grasp_assemble import generate, _geometry, _robot

    closed = []

    class Geometry:
        target = np.eye(4)
        parts = [trimesh.creation.box()]

        def __init__(self, *args: object) -> None:
            pass

        def set_layout(self, config: dict) -> None:
            self.config = config

        def evaluate(self, pose: object) -> dict:
            return {"accepted": True, "reasons": [], "T_assemble_tcp": pose}

    class Robot:
        def __init__(self, *args: object) -> None:
            self.calls = 0

        def set_layout(self, config: dict) -> None:
            self.config = config

        def plan(self, *args: object) -> dict:
            self.calls += 1
            return {"accepted": self.calls > 1}

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(_geometry, "GraspGeometry", Geometry)
    monkeypatch.setattr(_robot, "RobotSession", Robot)
    actions = iter(
        [
            _action("evaluate", np.eye(4).tolist()),
            _action("finish", candidate_id=1),
            _action("evaluate", np.eye(4).tolist()),
            _action("finish", candidate_id=3),
        ]
    )
    result = generate.run_harness(load_config(job), decide=lambda *args: next(actions))
    assert result["success"] and result["physical_success"] is None
    assert "error" in result["trace"][1]["observation"]
    assert result["selected"]["candidate_id"] == 3
    assert closed == [True]
    assert read_json(Path(result["run_directory"]) / "result.json")["success"]


def test_codex_transport_accepts_grasp_specific_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.tools.assemble import _protocol

    monkeypatch.setattr(_protocol.shutil, "which", lambda value: "/test/codex")

    def run(
        command: list[str], cwd: Path, prefix: Path, timeout: float, stdin: str
    ) -> None:
        assert read_json(cwd / "action.schema.json") == action_schema()
        assert "--sandbox" in command and "read-only" in command
        write_json(prefix.with_suffix(".json"), _action("fail"))

    monkeypatch.setattr(_protocol, "run_process", run)
    response = _protocol.codex_action(
        "prompt",
        tmp_path,
        1,
        {"model": None, "timeout_seconds": 10},
        response_schema=action_schema(),
    )
    assert response["action"] == "fail"


def test_robot_approach_rejects_contact_with_upright_mug() -> None:
    pytest.importorskip("fcl")
    from types import SimpleNamespace
    import torch
    from scripts.tools.grasp_assemble._robot import RobotSession

    mug = trimesh.creation.box([0.04, 0.04, 0.06])
    mug.apply_translation([0, 0, 0.05])
    rack = mug.copy()
    rack.apply_translation([1, 0, 0])
    matrices = torch.eye(4).repeat(2, 1, 1)
    robot = SimpleNamespace(
        joint_names=["joint"],
        link_names=["wrist_link"],
        pk_chain=SimpleNamespace(
            forward_kinematics=lambda _: {
                "wrist_link": SimpleNamespace(get_matrix=lambda: matrices)
            }
        ),
        get_local_pose=lambda **_: torch.eye(4)[None],
        get_link_vert_face=lambda _: (
            torch.tensor(mug.vertices.copy()),
            torch.tensor(mug.faces.copy()),
        ),
    )
    session = RobotSession.__new__(RobotSession)
    session.robot = robot
    session.trajectory = {
        "positions": torch.zeros(1, 2, 1),
        "phases": {
            "close": {"start": 1, "stop": 2},
            "lift": {"start": 1, "stop": 2},
            "release": {"start": 1, "stop": 2},
        },
    }
    geometry = SimpleNamespace(assemble_mesh=mug, base_world=rack, initial=np.eye(4))
    report = session._check_trajectory({}, geometry)
    assert not report["accepted"]
    assert "initial assemble object before closure" in report["reason"]


def test_planned_mimic_columns_follow_parent_without_mutating_plan() -> None:
    pytest.importorskip("fcl")
    from types import SimpleNamespace
    import torch
    from scripts.tools.grasp_assemble._robot import RobotSession

    original = torch.tensor([[[0.01, 0.0], [0.02, 0.0]]])
    session = RobotSession.__new__(RobotSession)
    session.trajectory = {"positions": original}
    session.robot = SimpleNamespace(
        mimic_ids=[1], mimic_parents=[0], mimic_multipliers=[-1], mimic_offsets=[0.04]
    )
    resolved = session._joint_positions()
    torch.testing.assert_close(resolved[0, :, 1], torch.tensor([0.03, 0.02]))
    assert original[..., 1].count_nonzero() == 0


@pytest.fixture
def replay_session():
    pytest.importorskip("fcl")
    from types import SimpleNamespace
    import torch
    from scripts.tools.grasp_assemble._robot import RobotSession

    session = RobotSession.__new__(RobotSession)
    session.config = {"motion": {"duration_scale": 3.0}}
    session._grasp_constraint = None
    session._constraint_events = []
    session.robot = SimpleNamespace(
        get_qpos=lambda **_: torch.zeros(1, 2),
        compute_fk=lambda *args, **kwargs: torch.eye(4)[None],
    )
    session.assemble = SimpleNamespace(
        get_local_pose=lambda **_: torch.eye(4)[None],
        clear_dynamics=lambda: None,
    )
    session.sim = SimpleNamespace(
        start_window_record=lambda **_: False,
        update=lambda **_: None,
        sim_config=SimpleNamespace(physics_dt=0.01),
    )
    # Timing is nonuniform and already includes duration_scale from planning.
    intervals = torch.zeros((1, 18))
    intervals[:, 1:] = torch.linspace(0.01, 0.03, 17) * 3
    session.trajectory = {
        "positions": torch.zeros(1, 18, 2),
        "velocities": torch.full((1, 18, 2), 0.2),
        "accelerations": torch.zeros(1, 18, 2),
        "jerks": torch.zeros(1, 18, 2),
        "dt": intervals,
        "phases": {
            "approach": {"start": 0, "stop": 3},
            "close": {"start": 2, "stop": 5},
            "lift": {"start": 4, "stop": 7},
            "rotate": {"start": 6, "stop": 10},
            "transfer": {"start": 9, "stop": 13},
            "place": {"start": 12, "stop": 15},
            "release": {"start": 14, "stop": 17},
            "retract": {"start": 16, "stop": 18},
        },
    }
    return session


def test_replay_preserves_timing_and_measures_physical_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, replay_session
) -> None:
    import torch
    from embodichain.lab.sim.motion import execution

    commands, holds = [], []
    monkeypatch.setattr(
        execution,
        "play_joint_trajectory",
        lambda *args, **kwargs: commands.append(kwargs),
    )
    session = replay_session
    planned_intervals = session.trajectory["dt"].clone()
    session.sim.update = lambda step: holds.append(step)
    target = np.eye(4)
    target[0, 3] = 0.5
    report = session.replay({"T_world_assemble_target": target}, tmp_path, True)
    assert report["trajectory_completed"]
    assert not report["physical_success"]
    assert report["pose_tolerances"] == {"position_m": 0.015, "rotation_degrees": 20.0}
    assert report["position_error_m"] == pytest.approx(0.5)
    assert [item["phase"] for item in report["phase_observations"]] == [
        "grasp_closed",
        "before_release",
        "retracted",
    ]
    assert len(commands) == 3
    assert sum(float(cmd["dt"].sum()) for cmd in commands) == pytest.approx(
        float(session.trajectory["dt"].sum())
    )
    assert all(cmd["dt"][0, 0] == 0 for cmd in commands)
    assert all(cmd["cfg"].joint_command_mode == "position_velocity" for cmd in commands)
    torch.testing.assert_close(
        torch.cat([cmd["dt"][:, 1:] for cmd in commands], dim=1),
        planned_intervals[:, 1:],
    )
    torch.testing.assert_close(session.trajectory["dt"], planned_intervals)
    assert holds == [100, 240]
    assert report["grasp_mode"] == "contact"
    assert report["constraint_events"] == []
    assert not report["constraint_active_at_end"]


@pytest.mark.parametrize(
    "angle, tolerance, expected",
    [
        (14.37, None, True),
        (19.9, None, True),
        (20.1, None, False),
        (14.37, 10, False),
        (25, 30, True),
    ],
)
def test_replay_uses_configured_final_rotation_tolerance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    replay_session,
    angle: float,
    tolerance: float | None,
    expected: bool,
) -> None:
    import torch
    from scipy.spatial.transform import Rotation
    from embodichain.lab.sim.motion import execution

    monkeypatch.setattr(
        execution, "play_joint_trajectory", lambda *args, **kwargs: None
    )
    session = replay_session
    if tolerance is not None:
        session.config["execution"] = {"rotation_tolerance_degrees": tolerance}
    observed = np.eye(4)
    observed[:3, :3] = Rotation.from_euler("x", angle, degrees=True).as_matrix()
    session.assemble.get_local_pose = lambda **_: torch.tensor(observed)[None]
    report = session.replay({"T_world_assemble_target": np.eye(4)}, tmp_path, True)
    assert report["trajectory_completed"]
    assert report["physical_success"] is expected
    assert report["rotation_error_degrees"] == pytest.approx(angle)
    assert report["pose_tolerances"]["rotation_degrees"] == (
        20.0 if tolerance is None else tolerance
    )


@pytest.mark.parametrize("fail_during_transfer", [False, True])
def test_constraint_spans_lift_rotation_transfer_and_placement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    replay_session,
    fail_during_transfer: bool,
) -> None:
    from types import SimpleNamespace
    from embodichain.lab.sim.motion import execution

    session = replay_session
    session.config["execution"] = {"grasp_mode": "fixed_constraint"}
    phases, holds, removed = [], [], []
    session.sim.get_env = lambda _: SimpleNamespace(remove_constraint=removed.append)
    session.sim.update = lambda step: holds.append((step, session._grasp_constraint))

    def attach(candidate: dict) -> None:
        session._grasp_constraint = "temporary"
        session._constraint_events.append(
            {"event": "attached", "phase": "grasp_closed"}
        )

    def play(*args, **kwargs) -> None:
        phases.append(session._grasp_constraint)
        if fail_during_transfer and len(phases) == 2:
            raise RuntimeError("transfer failed")

    monkeypatch.setattr(session, "_attach_grasp", attach)
    monkeypatch.setattr(execution, "play_joint_trajectory", play)
    if fail_during_transfer:
        with pytest.raises(RuntimeError, match="transfer failed"):
            session.replay({"T_world_assemble_target": np.eye(4)}, tmp_path, True)
        assert session._constraint_events[-1]["phase"] == "cleanup"
    else:
        report = session.replay({"T_world_assemble_target": np.eye(4)}, tmp_path, True)
        assert phases == [None, "temporary", None]
        assert holds == [(100, "temporary"), (240, None)]
        assert report["constraint_events"][-1]["phase"] == "before_release"
        assert not report["constraint_active_at_end"]
        assert report["grasp_mode"] == "fixed_constraint"
    assert session._grasp_constraint is None
    session._release_grasp()
    assert removed == ["temporary"]


@pytest.fixture
def attachment_session(replay_session):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from scipy.spatial.transform import Rotation
    import torch
    from scripts.tools.grasp_assemble._geometry import TCP_Z

    session = replay_session
    palm, cup = np.eye(4), np.eye(4)
    palm[:3, :3] = Rotation.from_euler("xyz", [0.4, -0.3, 0.2]).as_matrix()
    palm[:3, 3] = [-0.4, -0.2, 0.1]
    cup[:3, 3] = [-0.45, -0.25, 0]
    tcp_offset = np.eye(4)
    tcp_offset[2, 3] = TCP_Z
    candidate = {
        "T_assemble_tcp": np.linalg.inv(cup) @ palm @ tcp_offset,
        "finger_contact_qpos": [0.027, 0.028],
    }
    session.robot.get_link_pose = lambda *args, **_: torch.from_numpy(palm)[None]
    session.robot.get_qpos = lambda **_: torch.tensor([[0.029, 0.030]])
    session.robot.joint_names = [f"gripper_finger{i}_joint_1" for i in (1, 2)]
    session.robot._entities = [
        SimpleNamespace(
            render_articulation=SimpleNamespace(get_physical_body=lambda _: "palm_body")
        )
    ]
    session.assemble.get_local_pose = lambda **_: torch.from_numpy(cup)[None]
    session.assemble._entities = [
        SimpleNamespace(
            native=lambda: SimpleNamespace(get_rigid_body=lambda: "cup_body")
        )
    ]
    arena = SimpleNamespace(
        create_fixed_constraint_bodies=Mock(return_value=object()),
        remove_constraint=Mock(),
    )
    session.sim.get_env = lambda _: arena
    return session, candidate, arena, palm, cup


def test_physical_constraint_preserves_measured_grasp_frames(
    attachment_session,
) -> None:
    session, candidate, arena, palm, cup = attachment_session
    # A small tracking offset must be retained, not corrected by snapping the mug.
    candidate["T_assemble_tcp"][0, 3] += 0.001
    session._attach_grasp(candidate)
    name, body0, body1, frame0, frame1 = (
        arena.create_fixed_constraint_bodies.call_args.args
    )
    assert (name, body0, body1) == ("grasp_assemble_hold", "palm_body", "cup_body")
    np.testing.assert_allclose(palm @ frame0, cup @ frame1, atol=1e-7)
    assert session._constraint_events[0]["position_drift_m"] == pytest.approx(0.001)
    with pytest.raises(RuntimeError, match="already active"):
        session._attach_grasp(candidate)
    session._release_grasp("before_release")
    session._release_grasp()
    arena.remove_constraint.assert_called_once_with(name)


@pytest.mark.parametrize("invalid", ["drift", "open_fingers", "native_failure"])
def test_invalid_attachment_is_rejected(attachment_session, invalid: str) -> None:
    import torch

    session, candidate, arena, _, _ = attachment_session
    if invalid == "drift":
        candidate["T_assemble_tcp"][0, 3] += 0.05
    elif invalid == "open_fingers":
        session.robot.get_qpos = lambda **_: torch.zeros(1, 2)
    else:
        arena.create_fixed_constraint_bodies.return_value = None
    with pytest.raises(RuntimeError):
        session._attach_grasp(candidate)
    assert session._grasp_constraint is None
    assert session._constraint_events == []
    if invalid != "native_failure":
        arena.create_fixed_constraint_bodies.assert_not_called()


@pytest.mark.parametrize("extra_approach", [False, True])
def test_time_path_groups_carry_without_stopping_at_internal_waypoints(
    monkeypatch: pytest.MonkeyPatch,
    extra_approach: bool,
) -> None:
    pytest.importorskip("fcl")
    from types import SimpleNamespace
    import torch
    from scripts.tools.grasp_assemble import _trajectory
    from scripts.tools.grasp_assemble._robot import RobotSession

    session = RobotSession.__new__(RobotSession)
    session.config = {"motion": motion_settings()}
    session.sim = SimpleNamespace(
        device="cpu", sim_config=SimpleNamespace(physics_dt=0.01)
    )
    session.robot = SimpleNamespace(
        joint_names=[f"joint{i}" for i in range(1, 7)]
        + [f"gripper_finger{i}_joint_1" for i in (1, 2)],
        get_joint_ids=lambda **_: list(range(6)),
    )
    session.seed_qpos = torch.zeros(1, 8)
    names = ["approach", "lift", "rotate", "transfer", "place", "retract"]
    if extra_approach:
        names.insert(-2, "approach_target")
    q = np.linspace(0, 0.6, 2 * len(names) + 1)[:, None] * np.array(
        [1, 0.3, -0.4, 0.2, 0.5, 0.6]
    )
    ranges = {name: (2 * i, 2 * i + 3) for i, name in enumerate(names)}
    calls = []
    original = _trajectory.time_parameterize

    def timed(path, **kwargs):
        calls.append(path.copy())
        return original(path, **kwargs)

    monkeypatch.setattr(_trajectory, "time_parameterize", timed)
    result, reports = session._time_path(q, ranges, {"gripper_close_qpos": 0.03})
    assert len(calls) == 3
    for actual, expected in zip(calls, (q[:3], q[2:-2], q[-3:])):
        np.testing.assert_array_equal(actual, expected)
    assert list(reports) == ["approach", "carry", "retract"]
    assert reports["carry"]["phases"] == names[1:-1]
    phases = result["phases"]
    for name in names[1:-2]:
        step = phases[name]["stop"] - 1
        assert float(torch.linalg.vector_norm(result["velocities"][0, step, :6])) > 0.01
    for name in ("approach", "carry", "retract"):
        endpoints = [phases[name]["start"], phases[name]["stop"] - 1]
        assert torch.all(result["velocities"][0, endpoints, :6] == 0)
        assert torch.all(result["accelerations"][0, endpoints, :6] == 0)
    for name in ("close", "release"):
        start, stop = phases[name]["start"], phases[name]["stop"]
        arm = result["positions"][0, start:stop, :6]
        torch.testing.assert_close(arm, arm[:1].expand_as(arm))
    assert phases["carry"]["start"] == phases["close"]["stop"] - 1
    assert phases["carry"]["stop"] - 1 == phases["release"]["start"]


def test_endpoint_checks_cover_each_cartesian_waypoint(replay_session) -> None:
    import torch

    session = replay_session
    session.robot.get_joint_ids = lambda **_: [0]

    def fk(qpos, **kwargs):
        pose = torch.eye(4)[None]
        pose[:, 0, 3] = qpos[:, 0]
        return pose

    session.robot.compute_fk = fk
    session.trajectory["positions"][0, :, 0] = torch.arange(18) / 10
    expected_indices = {"grasp": 2, "lift": 6, "rotate": 9, "hover": 12, "place": 14}
    waypoints = {}
    for name, step in expected_indices.items():
        target = np.eye(4)
        target[0, 3] = step / 10
        waypoints[name] = {"T_world_tcp": target, "T_world_assemble": np.eye(4)}
    # Keep only one waypoint displaced to expose a missed or misindexed endpoint.
    waypoints["rotate"]["T_world_tcp"][0, 3] += 0.2
    report = session._endpoint_errors({"waypoints": waypoints})
    assert set(report) == set(expected_indices)
    for name in expected_indices:
        expected = 0.2 if name == "rotate" else 0
        assert report[name]["position_m"] == pytest.approx(expected, abs=1e-7)


def test_ik_path_crosses_wrapped_angles_without_a_full_turn() -> None:
    from scripts.tools.grasp_assemble._path import select_ik_paths

    layers = [np.deg2rad([[angle]]) for angle in (179, -179, -177)]
    paths = select_ik_paths(layers, np.deg2rad([178]), np.deg2rad([[-360, 360]]), 5)
    assert len(paths) == 1
    np.testing.assert_allclose(np.rad2deg(paths[0][:, 0]), [179, 181, 183])


def test_ik_path_avoids_greedy_branch_dead_end() -> None:
    from scripts.tools.grasp_assemble._path import select_ik_paths

    # The posture nearest home reaches 0.05, then cannot reach 0.65 within 20 deg.
    layers = [np.array([[0.0], [0.4]]), np.array([[0.05], [0.5]]), np.array([[0.65]])]
    paths = select_ik_paths(layers, np.zeros(1), np.array([[-1, 1]]), 20)
    assert len(paths) == 1
    np.testing.assert_allclose(paths[0][:, 0], [0.4, 0.5, 0.65])


def test_ik_path_keeps_long_feasible_route_when_short_winding_hits_joint_limit() -> (
    None
):
    from scripts.tools.grasp_assemble._path import select_ik_paths

    # Both routes reach the same physical endpoint. The shorter route winds
    # through +pi, which the [-3, 3] joint cannot cross; its cheap prefix must
    # not discard the longer route that stays inside the limits throughout.
    feasible = np.linspace(2.9, -2.9, 31)
    blocked = np.linspace(2.9, -2.9 + 2 * np.pi, 31)
    layers = [
        np.array([[long_angle], [short_angle]])
        for long_angle, short_angle in zip(feasible, blocked)
    ]
    paths = select_ik_paths(layers, np.array([2.9]), np.array([[-3, 3]]), 12)
    assert paths
    np.testing.assert_allclose(paths[0][:, 0], feasible)
    assert np.max(np.abs(np.diff(paths[0], axis=0))) <= np.deg2rad(12)


def test_ik_paths_are_capped_after_deduplicating_full_turn_equivalents() -> None:
    from scripts.tools.grasp_assemble._path import select_ik_paths

    physical_solutions = np.linspace(0, 0.8, 9)[:, None]
    paths = select_ik_paths(
        [physical_solutions], np.zeros(1), np.array([[-2 * np.pi, 2 * np.pi]]), 12
    )
    assert len(paths) == 8
    np.testing.assert_allclose(
        [path[0, 0] for path in paths], physical_solutions[:8, 0]
    )


def test_ik_paths_use_physical_limits_when_resolving_full_turn_aliases() -> None:
    from scripts.tools.grasp_assemble._path import select_ik_paths

    layers = [np.array([[-0.3], [2 * np.pi - 0.3]]), np.array([[-0.2]])]
    paths = select_ik_paths(layers, np.array([6.0]), np.array([[5, 7]]), 10)
    assert len(paths) == 1
    np.testing.assert_allclose(paths[0][:, 0], np.array([-0.3, -0.2]) + 2 * np.pi)
    assert select_ik_paths(layers, np.zeros(1), np.array([[0, 1]]), 10) == []


def test_float32_ik_aliases_do_not_multiply_search_states() -> None:
    from itertools import product
    from scripts.tools.grasp_assemble._path import _bounded_layer

    # One six-joint physical solution has 64 full-turn aliases inside +/-2pi.
    # The analytical solver's float32 wrapping noise must not expand those
    # aliases again as though they represented independent physical branches.
    shifts = np.array(list(product((-1, 0), repeat=6)))
    base = np.array([0.123, 0.234, 0.345, 0.456, 0.567, 0.678])
    values = (base + 2 * np.pi * shifts).astype(np.float32).astype(float)
    limits = np.tile([-2 * np.pi, 2 * np.pi], (6, 1))
    bounded = _bounded_layer(values, limits)
    assert bounded.shape == (64, 6)


def test_ik_paths_rank_reachable_end_branches_by_joint_travel() -> None:
    from scripts.tools.grasp_assemble._path import select_ik_paths

    layers = [np.zeros((1, 1)), np.array([[0.5], [0.1]])]
    paths = select_ik_paths(layers, np.zeros(1), np.array([[-1, 1]]), 40)
    assert len(paths) == 2
    np.testing.assert_allclose([path[-1, 0] for path in paths], [0.1, 0.5])


@pytest.mark.parametrize(
    "layers",
    [[], [np.empty((0, 1))], [np.array([[0.0]]), np.array([[1.0]])]],
)
def test_ik_path_returns_no_candidate_when_search_fails(layers: list) -> None:
    from scripts.tools.grasp_assemble._path import select_ik_paths

    assert select_ik_paths(layers, np.zeros(1), np.array([[-2, 2]]), 10) == []


@pytest.mark.parametrize(
    "layers,seed,limits,max_step",
    [
        ([np.array([0.0])], np.zeros(1), np.array([[-1, 1]]), 10),
        ([np.array([[float("nan")]])], np.zeros(1), np.array([[-1, 1]]), 10),
        ([np.zeros((1, 2))], np.zeros(1), np.array([[-1, 1]]), 10),
        ([np.zeros((1, 1))], np.zeros((1, 1)), np.array([[-1, 1]]), 10),
        ([np.zeros((1, 1))], np.zeros(1), np.array([-1, 1]), 10),
        ([np.zeros((1, 1))], np.zeros(1), np.array([[1, -1]]), 10),
        ([np.zeros((1, 1))], np.zeros(1), np.array([[1, 1]]), 10),
        ([np.zeros((1, 1))], np.zeros(1), np.array([[-1, 1]]), 0),
        ([np.zeros((1, 1))], np.zeros(1), np.array([[-1, 1]]), float("nan")),
    ],
)
def test_ik_path_rejects_malformed_inputs(
    layers: list, seed: np.ndarray, limits: np.ndarray, max_step: float
) -> None:
    from scripts.tools.grasp_assemble._path import select_ik_paths

    with pytest.raises(ValueError):
        select_ik_paths(layers, seed, limits, max_step)


def _layout_proposal(base_x: float = -0.55) -> dict:
    return {
        "base_xy": [base_x, -0.1],
        "assemble_xy": [-0.35, -0.3],
        "base_yaw_offset_degrees": 0,
        "assemble_yaw_offset_degrees": 30,
    }


def test_layout_resolves_robot_frame_preserving_height_and_seed(job: Path) -> None:
    from scripts.tools.grasp_assemble._layout import resolve_layout, layout_record
    from scipy.spatial.transform import Rotation

    config = load_config(job)
    config["layout"]["mode"] = "optimize"
    config["T_world_assemble_initial"][2][3] = 0.02
    resolved = resolve_layout(config, _layout_proposal())
    initial = np.asarray(resolved["T_world_assemble_initial"])
    np.testing.assert_allclose(initial[:3, 3], [-0.35, -0.3, 0.02])
    np.testing.assert_allclose(
        initial[:3, :3], Rotation.from_euler("z", 30, degrees=True).as_matrix()
    )
    assert config["T_world_assemble_initial"][0][3] == 0
    record = layout_record(resolved, _layout_proposal())
    np.testing.assert_allclose(record["T_robot_assemble_initial"], initial)
    np.testing.assert_allclose(record["T_world_robot"], np.eye(4))


@pytest.mark.parametrize(
    "change",
    [
        {"base_xy": [-0.4, 0]},  # Too close to the robot base.
        {"assemble_xy": [-0.9, -0.3]},  # Outside workspace.
        {"assemble_xy": [-0.55, -0.12]},  # Objects overlap in their initial layout.
        {"assemble_yaw_offset_degrees": 100},
        {"base_xy": [True, -0.2]},
        {"base_xy": [float("nan"), 0]},
        {"base_xy": [0]},
        {"extra": 1},
    ],
)
def test_layout_rejects_invalid_proposals(job: Path, change: dict) -> None:
    from scripts.tools.grasp_assemble._layout import resolve_layout

    config = load_config(job)
    config["layout"]["mode"] = "optimize"
    with pytest.raises(ValueError):
        resolve_layout(config, _layout_proposal() | change)


def test_fixed_layout_cannot_be_moved_by_model(job: Path) -> None:
    from scripts.tools.grasp_assemble._layout import resolve_layout

    config = load_config(job)
    assert resolve_layout(config, None) == config
    with pytest.raises(ValueError, match="fixed mode"):
        resolve_layout(config, _layout_proposal())


def test_motion_score_counts_real_full_turns(job: Path) -> None:
    from scripts.tools.grasp_assemble._layout import motion_metrics, motion_rejection

    metrics = motion_metrics(
        np.array([[0, 0], [2 * np.pi, 0], [0, 0]]), ["arm", "wrist"]
    )
    assert metrics["joint_travel_degrees"] == pytest.approx([720, 0])
    assert metrics["max_joint_range_degrees"] == pytest.approx(360)
    assert metrics["score"] == pytest.approx(1440)
    config = load_config(job)
    assert motion_rejection(metrics, config) is None
    config["layout"]["mode"] = "optimize"
    assert "max_joint_step_degrees" in motion_rejection(metrics, config)


def test_layout_search_selects_best_and_restores_its_scene(
    job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fcl")
    pytest.importorskip("manifold3d")
    from scripts.tools.grasp_assemble import generate, _geometry, _robot

    configurations = []

    class Geometry:
        target = np.eye(4)
        parts = [trimesh.creation.box()]

        def __init__(self, config, *args):
            self.set_layout(config)

        def set_layout(self, config):
            self.config = config
            configurations.append(config)

        def evaluate(self, pose):
            return {"accepted": True, "reasons": [], "T_assemble_tcp": pose}

    class Robot:
        def __init__(self, *args):
            pass

        def set_layout(self, config):
            self.config = config

        def plan(self, *args):
            score = 100 if self.config["T_world_base"][0][3] == -0.55 else 200
            return {"accepted": True, "motion_metrics": {"score": score}}

        def close(self):
            pass

    monkeypatch.setattr(_geometry, "GraspGeometry", Geometry)
    monkeypatch.setattr(_robot, "RobotSession", Robot)
    config = load_config(job)
    config["layout"].update(mode="optimize", min_candidates=2)
    second = _action("evaluate", np.eye(4).tolist(), layout=_layout_proposal(-0.6))
    second["task_plan"]["clearance"] = 0.14
    second["task_plan"]["insertion_direction_base"] = [1, 0, 0]
    actions = iter(
        [
            _action("evaluate", np.eye(4).tolist(), layout=_layout_proposal()),
            _action("finish", candidate_id=1),  # Too few distinct layouts.
            second,
            _action("finish", candidate_id=3),  # Accepted but worse than candidate 1.
            _action("finish", candidate_id=1),
        ]
    )
    result = generate.run_harness(config, decide=lambda *args: next(actions))
    assert result["success"]
    assert "distinct layouts" in result["trace"][1]["observation"]["error"]
    assert "best accepted candidate 1" in result["trace"][3]["observation"]["error"]
    assert result["selected"]["candidate_id"] == 1
    assert result["layout_search"]["evaluated_layouts"] == 2
    assert result["config"]["T_world_base"][0][3] == -0.55
    assert configurations[-1] == result["config"]
    assert result["config"]["motion"]["clearance"] == 0.08
    assert result["config"]["task_plan"] == result["task_plan"]
    assert result["task_plan"]["insertion_direction_base"] == [0, 0, -1]
    assert result["selected"]["proposed_task_plan"] == _task_plan()
    assert result["input_config"]["T_world_base"][0][3] == 0
    assert (
        read_json(Path(result["run_directory"]) / "result.json")["config"]
        == result["config"]
    )


def test_robot_layout_update_moves_both_objects_and_invalidates_plan(
    replay_session,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock
    import torch

    session = replay_session
    session.sim.device = "cpu"
    session.base = SimpleNamespace(set_local_pose=Mock(), clear_dynamics=Mock())
    session.assemble = SimpleNamespace(set_local_pose=Mock(), clear_dynamics=Mock())
    rack, mug = np.eye(4), np.eye(4)
    rack[:3, 3] = [-0.55, -0.1, 0]
    mug[:3, 3] = [-0.35, -0.3, 0]
    config = {"T_world_base": rack.tolist(), "T_world_assemble_initial": mug.tolist()}
    session.set_layout(config)
    torch.testing.assert_close(
        session.base.set_local_pose.call_args.args[0],
        torch.tensor(rack, dtype=torch.float32)[None],
    )
    torch.testing.assert_close(
        session.assemble.set_local_pose.call_args.args[0],
        torch.tensor(mug, dtype=torch.float32)[None],
    )
    assert session.trajectory is None
    session._grasp_constraint = "active"
    with pytest.raises(RuntimeError, match="holding the assemble object"):
        session.set_layout(config)
