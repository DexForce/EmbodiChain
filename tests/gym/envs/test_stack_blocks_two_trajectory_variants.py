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

"""Trajectory variant expansion inside the official stack-blocks task.

These tests drive the task's own planning helpers directly. Building the
environment needs a simulator, but the behavior worth pinning here is pure:
where the fixed phases land, that contacts survive augmentation, and that the
solver Jacobian columns are permuted into the task's joint order.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.atomic_actions.plans import TrajectorySegment
from embodichain_tasks.manipulation.tableware.stack_blocks_two import (
    PICK_SAMPLE_INTERVAL,
    PLACE_SAMPLE_INTERVAL,
    StackBlocksTwoEnv,
)

ARM_JOINTS = 6
HAND_JOINTS = 2
TOTAL_JOINTS = ARM_JOINTS + HAND_JOINTS
APPROACH_STOP = 48
CLOSE_STOP = 58
HOLD_STEPS = 45
PICK_TOTAL = PICK_SAMPLE_INTERVAL + HOLD_STEPS
TOTAL_SAMPLES = PICK_TOTAL + PLACE_SAMPLE_INTERVAL
STEP_DT = 0.02

CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "embodichain_tasks"
    / "configs"
    / "tasks"
    / "manipulation"
    / "tableware"
    / "stack_blocks_two"
    / "env.json"
)


class _Compiled:
    """Stand-in exposing the named segments of one compiled atomic action."""

    def __init__(self, segments: dict[str, tuple[int, int]]) -> None:
        self._segments = segments

    def segment(self, action_index: int, name: str) -> TrajectorySegment:
        assert action_index == 0
        start, stop = self._segments[name]
        return TrajectorySegment(name=name, start=start, stop=stop)


def pick_compiled() -> _Compiled:
    return _Compiled(
        {
            "approach": (0, APPROACH_STOP),
            "close": (APPROACH_STOP, CLOSE_STOP),
            "lift": (CLOSE_STOP, PICK_SAMPLE_INTERVAL),
        }
    )


def place_compiled() -> _Compiled:
    return _Compiled(
        {
            "approach": (0, APPROACH_STOP),
            "release": (APPROACH_STOP, CLOSE_STOP),
            "retract": (CLOSE_STOP, PLACE_SAMPLE_INTERVAL),
        }
    )


def nominal_trajectory(num_envs: int) -> torch.Tensor:
    """Build a batch whose gripper is constant inside every retimed phase."""
    trajectory = torch.zeros(num_envs, TOTAL_SAMPLES, TOTAL_JOINTS)
    for env_id in range(num_envs):
        for joint in range(ARM_JOINTS):
            trajectory[env_id, :, joint] = torch.linspace(
                0.0, 0.4 + 0.05 * env_id + 0.02 * joint, TOTAL_SAMPLES
            )
    hand = torch.zeros(TOTAL_SAMPLES)
    hand[:APPROACH_STOP] = 0.05
    hand[APPROACH_STOP:CLOSE_STOP] = torch.linspace(
        0.05, 0.0, CLOSE_STOP - APPROACH_STOP
    )
    release_start = PICK_TOTAL + APPROACH_STOP
    release_stop = PICK_TOTAL + CLOSE_STOP
    hand[release_start:release_stop] = torch.linspace(
        0.0, 0.05, release_stop - release_start
    )
    hand[release_stop:] = 0.05
    trajectory[:, :, ARM_JOINTS:] = hand[None, :, None]
    return trajectory


class _TaskStub:
    """Carries only the state the expansion helpers read, with real methods."""

    _initialize_trajectory_variants = StackBlocksTwoEnv._initialize_trajectory_variants
    _resolve_jacobian_columns = StackBlocksTwoEnv._resolve_jacobian_columns
    _reference_jacobians = StackBlocksTwoEnv._reference_jacobians
    _variant_phases = StackBlocksTwoEnv._variant_phases
    _expand_trajectory_variants = StackBlocksTwoEnv._expand_trajectory_variants

    def __init__(self, num_envs: int) -> None:
        limits = torch.stack(
            (torch.full((TOTAL_JOINTS,), -3.0), torch.full((TOTAL_JOINTS,), 3.0)),
            dim=1,
        )
        self.num_envs = num_envs
        self.step_dt = STEP_DT
        self.robot = SimpleNamespace(
            uid="cobotmagic",
            joint_names=[f"joint_{index}" for index in range(TOTAL_JOINTS)],
            get_qpos_limits=lambda: limits.unsqueeze(0),
        )
        self._arm_joint_ids = list(range(ARM_JOINTS))


def task(num_envs: int, settings: dict | None = None) -> _TaskStub:
    """Build a stand-in task and run its extension decoding."""
    instance = _TaskStub(num_envs)
    if settings is not None:
        instance.trajectory_variants = settings
    instance._initialize_trajectory_variants()
    return instance


def enabled_settings(**overrides: object) -> dict:
    settings = {
        "seed": 4,
        "factors": {
            "spatial": {
                "enabled": True,
                "method": "via_points",
                "joint_offset_scale": 0.03,
                "via_count": 3,
            },
            "timing": {
                "enabled": True,
                "duration_scales": [1.0, 1.25],
                "profiles": ["uniform", "ease_in"],
            },
        },
        "coverage": {"target_per_cell": 4},
    }
    settings.update(overrides)
    return settings


def test_packaged_config_ships_the_extension_disabled() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    settings = config["env"]["extensions"]["trajectory_variants"]
    assert settings["enabled"] is False
    instance = task(2, settings)
    assert instance._variant_cfg is None

    enabled = dict(settings)
    enabled["enabled"] = True
    del enabled["factors"]["ik"]
    ready = task(2, enabled)
    assert ready._variant_cfg is not None
    assert ready._variant_cfg.factors.spatial.method == "via_points"
    assert ready._variant_cfg.factors.timing.profiles == (
        "uniform",
        "ease_in",
        "ease_out",
    )


def test_absent_or_rejected_settings_leave_planning_unchanged() -> None:
    assert task(2)._variant_cfg is None
    assert task(2, {"enabled": False})._variant_cfg is None
    with pytest.raises(TypeError, match="must be a mapping"):
        task(2, ["not", "a", "mapping"])
    with pytest.raises(ValueError):
        task(2, {"factors": {"spatial": {"method": "teleport"}}})


def test_phases_follow_the_compiled_segments_and_lock_every_contact() -> None:
    phases = task(2)._variant_phases(
        pick_compiled(),
        place_compiled(),
        hold_steps=HOLD_STEPS,
        place_offset=PICK_TOTAL,
    )
    assert [
        (phase.phase_id, phase.start_index, phase.stop_index) for phase in phases
    ] == [
        ("pick_approach", 0, APPROACH_STOP),
        ("grasp_close", APPROACH_STOP, CLOSE_STOP),
        ("grasp_hold", CLOSE_STOP, CLOSE_STOP + HOLD_STEPS),
        ("lift", PICK_SAMPLE_INTERVAL + HOLD_STEPS - 32, PICK_TOTAL),
        ("place_approach", PICK_TOTAL, PICK_TOTAL + APPROACH_STOP),
        ("place_release", PICK_TOTAL + APPROACH_STOP, PICK_TOTAL + CLOSE_STOP),
        ("place_retract", PICK_TOTAL + CLOSE_STOP, TOTAL_SAMPLES),
    ]
    locked = {phase.phase_id: phase for phase in phases if phase.kind != "free"}
    assert set(locked) == {"grasp_close", "grasp_hold", "place_release"}
    assert all(not phase.allowed_operators for phase in locked.values())
    # Only motion after the grasp may be retimed; the executor clears the held
    # block's dynamics at one shared step index.
    retimed = {
        phase.phase_id for phase in phases if "retime" in phase.allowed_operators
    }
    assert retimed == {"lift", "place_approach", "place_retract"}
    assert "retime" not in phases[0].allowed_operators


def test_mismatched_grasp_hold_index_fails_loudly() -> None:
    shifted = _Compiled(
        {
            "approach": (0, APPROACH_STOP),
            "close": (APPROACH_STOP, CLOSE_STOP + 3),
            "lift": (CLOSE_STOP + 3, PICK_SAMPLE_INTERVAL),
        }
    )
    with pytest.raises(RuntimeError, match="grasp-hold insertion index"):
        task(2)._variant_phases(
            shifted,
            place_compiled(),
            hold_steps=HOLD_STEPS,
            place_offset=PICK_TOTAL,
        )


def test_expansion_varies_rows_while_preserving_contacts_and_row_zero() -> None:
    num_envs = 5
    instance = task(num_envs, enabled_settings(enabled=True))
    trajectory = nominal_trajectory(num_envs)
    phases = instance._variant_phases(
        pick_compiled(),
        place_compiled(),
        hold_steps=HOLD_STEPS,
        place_offset=PICK_TOTAL,
    )
    expanded, report = instance._expand_trajectory_variants(trajectory, phases)
    assert expanded.shape[0] == num_envs
    assert expanded.shape[2] == TOTAL_JOINTS
    assert report["fallbacks"] == 0
    assert report["distinct_variants"] == num_envs
    assert report["row_variants"][0]["spatial_operator"] == "none"
    assert len({row["variant_ordinal"] for row in report["row_variants"]}) == num_envs

    reference_length = report["row_lengths"][0]
    torch.testing.assert_close(
        expanded[0, :reference_length], trajectory[0], atol=1e-6, rtol=0
    )
    assert max(report["row_lengths"]) > reference_length

    for env_id in range(num_envs):
        row_phases = {
            name: (start, stop) for name, start, stop, _ in report["row_phases"][env_id]
        }
        for name, source in (
            ("grasp_close", (APPROACH_STOP, CLOSE_STOP)),
            ("grasp_hold", (CLOSE_STOP, CLOSE_STOP + HOLD_STEPS)),
            ("place_release", (PICK_TOTAL + APPROACH_STOP, PICK_TOTAL + CLOSE_STOP)),
        ):
            start, stop = row_phases[name]
            torch.testing.assert_close(
                expanded[env_id, start:stop],
                trajectory[env_id, source[0] : source[1]],
                atol=1e-6,
                rtol=0,
            )
        # Every row still ends at its own planned retract pose.
        length = report["row_lengths"][env_id]
        torch.testing.assert_close(
            expanded[env_id, length - 1], trajectory[env_id, -1], atol=1e-6, rtol=0
        )
    assert not torch.allclose(expanded[1, :reference_length], trajectory[1])


def test_shorter_rows_hold_their_final_command() -> None:
    num_envs = 4
    instance = task(num_envs, enabled_settings(enabled=True))
    trajectory = nominal_trajectory(num_envs)
    phases = instance._variant_phases(
        pick_compiled(),
        place_compiled(),
        hold_steps=HOLD_STEPS,
        place_offset=PICK_TOTAL,
    )
    expanded, report = instance._expand_trajectory_variants(trajectory, phases)
    horizon = expanded.shape[1]
    for env_id, length in enumerate(report["row_lengths"]):
        if length == horizon:
            continue
        torch.testing.assert_close(
            expanded[env_id, length:],
            expanded[env_id, length - 1].expand(horizon - length, TOTAL_JOINTS),
        )


def test_reference_jacobians_are_permuted_into_the_task_joint_order() -> None:
    num_envs = 1
    settings = enabled_settings(enabled=True)
    settings["factors"]["ik"] = {"enabled": True, "task_rows": [0, 1, 2]}
    instance = task(num_envs)

    # The solver reverses the arm joints and omits the gripper, which is the
    # column order its Jacobian uses.
    solver_names = [f"joint_{index}" for index in reversed(range(ARM_JOINTS))]

    def get_jacobian(qpos: torch.Tensor) -> torch.Tensor:
        # Encode each column's identity so a wrong permutation is visible.
        samples = qpos.shape[0]
        jacobian = torch.zeros(samples, 6, ARM_JOINTS)
        for column in range(ARM_JOINTS):
            jacobian[:, :, column] = float(column)
        return jacobian

    instance.robot.get_solver = lambda name: SimpleNamespace(
        joint_names=solver_names, get_jacobian=get_jacobian
    )
    instance.trajectory_variants = settings
    instance._initialize_trajectory_variants()

    assert instance._variant_solver_columns == list(reversed(range(ARM_JOINTS)))
    assert instance._variant_jacobian_columns == list(reversed(range(ARM_JOINTS)))
    jacobians = instance._reference_jacobians(nominal_trajectory(num_envs))
    assert len(jacobians) == num_envs
    assert jacobians[0].shape == (TOTAL_SAMPLES, 3, ARM_JOINTS)
    # Column i of the result must carry the solver column for arm joint i.
    torch.testing.assert_close(
        jacobians[0][0, 0],
        torch.tensor([float(ARM_JOINTS - 1 - index) for index in range(ARM_JOINTS)]),
    )


def test_redundancy_expansion_without_a_named_solver_fails_at_setup() -> None:
    settings = enabled_settings(enabled=True)
    settings["factors"]["ik"] = {"enabled": True}
    instance = task(1)
    instance.robot.get_solver = lambda name: SimpleNamespace(
        joint_names=None, get_jacobian=None
    )
    instance.trajectory_variants = settings
    with pytest.raises(RuntimeError, match="declares its joint names"):
        instance._initialize_trajectory_variants()


def test_successive_episodes_advance_through_the_factor_grid() -> None:
    num_envs = 4
    instance = task(num_envs, enabled_settings(enabled=True))
    trajectory = nominal_trajectory(num_envs)
    phases = instance._variant_phases(
        pick_compiled(),
        place_compiled(),
        hold_steps=HOLD_STEPS,
        place_offset=PICK_TOTAL,
    )
    first, first_report = instance._expand_trajectory_variants(trajectory, phases)
    second, second_report = instance._expand_trajectory_variants(trajectory, phases)

    assert first_report["episode_index"] == 0
    assert second_report["episode_index"] == 1
    first_ordinals = [row["variant_ordinal"] for row in first_report["row_variants"]]
    second_ordinals = [row["variant_ordinal"] for row in second_report["row_variants"]]
    assert first_ordinals[0] == second_ordinals[0] == 0
    assert set(first_ordinals[1:]).isdisjoint(second_ordinals[1:])
    # The reference row is replayed unchanged in both episodes.
    reference = first_report["row_lengths"][0]
    torch.testing.assert_close(first[0, :reference], trajectory[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(
        second[0, : second_report["row_lengths"][0]], trajectory[0], atol=1e-6, rtol=0
    )
    # A repeated environment draws a fresh residual rather than replaying one.
    # The approach phase is never retimed, so its indices are comparable.
    assert not torch.allclose(
        first[1, :APPROACH_STOP], second[1, :APPROACH_STOP], atol=1e-6, rtol=0
    )
    assert first_report["row_variants"][1] != second_report["row_variants"][1]
