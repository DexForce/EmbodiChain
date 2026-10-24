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

"""Compatibility contracts for Task Program workspace validation."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.motion.expansion import ValidationResult
from embodichain.lab.sim.scene.expansion import ScenePoseChange
from embodichain.lab.sim.scene.validation import SceneCheck, SceneCheckReport
from embodichain.lab.sim.scene.workspace import (
    RobotSceneWorkspace as SceneWorkspace,
    SceneWorkspaceCandidate as SceneCandidate,
)
from embodichain.lab.task_program.integrations.simulation.workspace import (
    RobotSceneWorkspace,
    SceneWorkspaceCandidate,
)

pytestmark = pytest.mark.no_sim


def test_legacy_workspace_retains_service_and_candidate_contracts() -> None:
    assert issubclass(RobotSceneWorkspace, SceneWorkspace)
    assert SceneWorkspaceCandidate is SceneCandidate


@pytest.mark.parametrize(
    ("status", "expected_status", "accepted"),
    [
        ("passed", "passed", True),
        ("failed", "failed", False),
        ("unverified", "unavailable", False),
    ],
)
def test_adapter_preserves_evidence_and_forwards_pose_arguments(
    monkeypatch: pytest.MonkeyPatch, status: str, expected_status: str, accepted: bool
) -> None:
    # Sampling and IK/FK are covered at the inherited scene-service owner.
    workspace = object.__new__(RobotSceneWorkspace)
    pose = ScenePoseChange("cube", torch.eye(4).tolist())
    grasp = torch.eye(4)
    seed = torch.tensor([0.25, -0.5])
    report = SceneCheckReport(
        (
            SceneCheck(
                "workspace.pose_ik",
                status,
                "measured evidence",
                {"position_error_m": 0.01},
            ),
        )
    )
    calls = []

    def check(self, pose_change, object_grasp_pose, *, joint_seed=None):
        calls.append((self, pose_change, object_grasp_pose, joint_seed))
        return report

    monkeypatch.setattr(SceneWorkspace, "check_object_pose", check)
    result = workspace.check_object_pose(pose, grasp, joint_seed=seed)

    assert len(calls) == 1
    assert all(
        actual is expected
        for actual, expected in zip(calls[0], (workspace, pose, grasp, seed))
    )
    assert isinstance(result, ValidationResult)
    assert result.accepted is accepted
    (converted,) = result.checks
    assert converted.check_id == report.checks[0].name
    assert converted.status == expected_status
    assert converted.detail == report.checks[0].message
    assert converted.metrics == report.checks[0].metrics


def test_adapter_keeps_missing_evidence_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SceneWorkspace,
        "check_object_pose",
        lambda *args, **kwargs: SceneCheckReport(()),
    )
    result = object.__new__(RobotSceneWorkspace).check_object_pose(
        ScenePoseChange("cube", torch.eye(4).tolist()), torch.eye(4)
    )
    assert result.checks == ()
    assert not result.accepted


def test_adapter_propagates_service_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise ValueError("invalid joint seed")

    monkeypatch.setattr(SceneWorkspace, "check_object_pose", fail)
    with pytest.raises(ValueError, match="invalid joint seed"):
        object.__new__(RobotSceneWorkspace).check_object_pose(
            ScenePoseChange("cube", torch.eye(4).tolist()), torch.eye(4)
        )
