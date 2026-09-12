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

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("dexsim.kit.motion_policy.evaluator")

from dexsim.kit.motion_policy import (
    AdapterRequest,
    EvaluationFrame,
    PolicyContext,
    RobotDescription,
    RobotState,
    parse_policy_spec,
)

from embodichain.learning.rl.policy_evaluation import (
    MotionProfileRequest,
    build_motion_profile,
)

_JOINT_NAMES = (
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
)
_DEFAULT_POSITION = np.asarray(
    (0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8),
    dtype=np.float32,
)
_COMMAND = np.asarray((0.4, -0.2, 0.6), dtype=np.float32)


class _CommandPolicy(torch.nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        padding = torch.zeros(
            (observation.shape[0], 9),
            dtype=observation.dtype,
            device=observation.device,
        )
        return torch.cat((observation[:, 9:12], padding), dim=1)


def test_anymal_c_profile_builds_and_runs_torchscript(tmp_path, monkeypatch):
    example_root = (
        Path(__file__).resolve().parents[4] / "examples/learning/policy_evaluation"
    )
    monkeypatch.syspath_prepend(str(example_root))
    from anymal_c.profile import (
        AnymalCVelocityAdapter,
    )
    from anymal_c import register

    checkpoint = tmp_path / "mjw_anymal.pt"
    traced = torch.jit.trace(_CommandPolicy().eval(), torch.zeros((1, 48)))
    torch.jit.save(traced, checkpoint)

    robot_asset = tmp_path / "anybotics_anymal_c/urdf/anymal.urdf"
    robot_asset.parent.mkdir(parents=True)
    robot_asset.write_text("<robot name='anymal_c'/>\n", encoding="utf-8")

    request = MotionProfileRequest(
        checkpoint=checkpoint,
        device=torch.device("cpu"),
        resource_root=tmp_path,
    )
    register()
    profile = build_motion_profile("newton-anymal-c-velocity", request)
    spec = parse_policy_spec(profile.policy_spec)
    assert spec.environment.entrypoint is None
    config = profile.policy_spec["policy"]["adapter"]["config"]
    adapter = AnymalCVelocityAdapter(
        AdapterRequest(
            asset_path=robot_asset,
            models={"actor": checkpoint},
            resources={},
            config=config,
        )
    )
    context = PolicyContext(
        robot=RobotDescription(
            _JOINT_NAMES,
            ("base",),
            "base",
        ),
        physics_dt=0.005,
        sim_steps_per_control=4,
        policy_dt=0.02,
    )
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = 0.76
    frame = EvaluationFrame(
        control_step=0,
        policy_time=0.0,
        simulation_time=0.0,
        simulation_step=0,
        robot_state=RobotState(
            joint_names=_JOINT_NAMES,
            qpos=_DEFAULT_POSITION.copy(),
            qvel=np.zeros(12, dtype=np.float32),
            target_qpos=_DEFAULT_POSITION.copy(),
            target_qvel=np.zeros(12, dtype=np.float32),
            joint_effort=np.zeros(12, dtype=np.float32),
            root_name="base",
            root_pose=pose,
            root_velocity=np.zeros(6, dtype=np.float32),
            link_names=("base",),
            link_poses=pose[None, ...],
            link_velocities=np.zeros((1, 6), dtype=np.float32),
        ),
        controls={"command": _COMMAND},
    )

    adapter.setup(context)
    adapter.reset(frame)
    output = adapter.infer(frame)

    assert output.action.joint_names == _JOINT_NAMES
    expected_position = _DEFAULT_POSITION.copy()
    expected_position[:3] += 0.5 * _COMMAND
    np.testing.assert_allclose(output.action.position, expected_position)
    np.testing.assert_allclose(
        adapter.previous_action,
        np.concatenate((_COMMAND, np.zeros(9, dtype=np.float32))),
    )
    assert adapter.command_enabled
    assert adapter.command_limits == (1.0, 0.5, 1.0)
    assert output.termination_reason is None
    adapter.reset(frame)
    np.testing.assert_array_equal(adapter.previous_action, np.zeros(12))
    adapter.close()


def test_example_script_supplies_default_resource_paths(tmp_path, monkeypatch):
    example_root = (
        Path(__file__).resolve().parents[4] / "examples/learning/policy_evaluation"
    )
    monkeypatch.syspath_prepend(str(example_root))
    monkeypatch.setenv("ANYMAL_C_EXAMPLE_CACHE", str(tmp_path))
    from eval_policy import example_arguments

    arguments = example_arguments(["--control-steps", "5"])

    assert arguments == [
        "--profile",
        "newton-anymal-c-velocity",
        "--checkpoint",
        str(tmp_path / "upstream/anybotics_anymal_c/rl_policies/mjw_anymal.pt"),
        "--resource-root",
        str(tmp_path / "upstream"),
        "--control-steps",
        "5",
    ]
