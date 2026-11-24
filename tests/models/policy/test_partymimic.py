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

from types import SimpleNamespace

import numpy as np
import pytest

from embodichain.extensions.models.policies.partymimic import (
    PartyMimicPolicy,
    PartyMimicState,
    sparse5_window,
)


class _RecordingSession:
    def get_inputs(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(name=name, shape=[size], type="tensor(float)")
            for name, size in (
                ("proprio", 900),
                ("root_quat_hist", 80),
                ("ref_kp", 462),
            )
        ]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="action", shape=[28], type="tensor(float)")]

    def run(self, outputs: list[str], feed: dict) -> list[np.ndarray]:
        assert outputs == ["action"]
        self.feed = feed
        return [np.arange(28, dtype=np.float32) / 10]


@pytest.fixture
def actor() -> tuple[PartyMimicPolicy, _RecordingSession]:
    joints = [f"j{i}" for i in range(28)]
    keypoints = (
        "pelvis",
        "head",
        "left_wrist",
        "right_wrist",
        "left_ankle",
        "right_ankle",
    )
    session = _RecordingSession()
    policy = PartyMimicPolicy(
        session,
        {
            "policy_joint_names": joints,
            "motion": {"body_names": list(keypoints)},
            "interface": {
                "name": "sparse5",
                "goal_frames": list(range(1, 11)),
                "keypoints": dict(zip(keypoints, keypoints, strict=True)),
            },
            "default_joint_pos": dict.fromkeys(joints, 0.12),
            "joint_kp": dict.fromkeys(joints, 12.5),
            "joint_kd": dict.fromkeys(joints, 0.75),
            "action_scale": dict.fromkeys(joints, 0.25),
        },
    )
    return policy, session


def test_partymimic_history_and_onnx_frames(
    actor: tuple[PartyMimicPolicy, _RecordingSession],
) -> None:
    policy, session = actor
    quat = np.sqrt(0.5)
    # A +90 degree roll sends world +Z to body +Y on inverse rotation.
    state = PartyMimicState(
        np.full(28, 0.12), np.ones(28), [quat, 0, 0, quat], [0, 0, 2]
    )
    reference = np.zeros((11, 6, 7), dtype=np.float32)
    reference[..., 5:] = quat  # +90 degree yaw in xyzw
    reference[:, :, 0] = np.arange(11)[:, None]

    with pytest.raises(RuntimeError, match="Reset"):
        policy.step(state, reference)
    policy.reset(state, reference)
    target = policy.step(state, reference)
    row = np.r_[np.zeros(28), np.ones(28), [0, 2, 0], [0, -1, 0], np.zeros(28)]
    np.testing.assert_allclose(
        session.feed["proprio"].reshape(10, 90), np.tile(row, (10, 1)), atol=1e-6
    )
    np.testing.assert_allclose(
        session.feed["root_quat_hist"].reshape(10, 8),
        np.tile([quat, quat, 0, 0, quat, 0, 0, quat], (10, 1)),
    )
    frames = session.feed["ref_kp"].reshape(11, 6, 7)
    np.testing.assert_allclose(frames[-1, 0], [10, 0, 0, quat, 0, 0, quat])
    np.testing.assert_allclose(target, 0.12 + 0.25 * np.arange(28) / 10)

    policy.step(state, reference)
    history = session.feed["proprio"].reshape(10, 90)
    np.testing.assert_array_equal(history[-2, -28:], 0)
    np.testing.assert_allclose(history[-1, -28:], np.arange(28) / 10)
    policy.reset(state, reference)
    policy.step(state, reference)
    np.testing.assert_array_equal(session.feed["proprio"].reshape(10, 90)[:, -28:], 0)


def test_partymimic_joint_indices_follow_policy_order(
    actor: tuple[PartyMimicPolicy, _RecordingSession],
) -> None:
    policy, _ = actor
    np.testing.assert_array_equal(
        policy.joint_indices(policy.joint_names[::-1]), np.arange(27, -1, -1)
    )
    with pytest.raises(ValueError, match="Simulator joints"):
        policy.joint_indices((*policy.joint_names[:-1], "wrong"))


def test_sparse5_window_holds_last_pose() -> None:
    clip = np.arange(3 * 6 * 7).reshape(3, 6, 7)
    window = sparse5_window(clip, 1)
    np.testing.assert_array_equal(window[0], clip[1])
    np.testing.assert_array_equal(window[1:], np.repeat(clip[-1:], 10, axis=0))
    with pytest.raises(IndexError):
        sparse5_window(clip, len(clip))
