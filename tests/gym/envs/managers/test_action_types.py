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

"""Tests for action IO descriptor contracts."""

from __future__ import annotations

import pytest

from embodichain.lab.gym.envs.managers.action_types import (
    ActionDescriptor,
    ActionTermDescriptor,
)


def test_action_term_descriptor_rejects_mismatched_names() -> None:
    """Feature names and units must cover every raw action dimension."""
    with pytest.raises(ValueError, match="feature_names, units, and action_dim"):
        ActionTermDescriptor(
            representation="joint_position",
            action_dim=2,
            feature_names=("joint_a",),
            units=("rad", "rad"),
            normalization=None,
            joint_names=("joint_a", "joint_b"),
            metadata={},
        )


def test_action_descriptor_serializes_bound_slice() -> None:
    """Manager-bound descriptors preserve ordered term semantics as JSON."""
    term = ActionTermDescriptor(
        representation="eef_pose",
        action_dim=6,
        feature_names=("x", "y", "z", "roll", "pitch", "yaw"),
        units=("m", "m", "m", "rad", "rad", "rad"),
        normalization=None,
        joint_names=("joint_1", "joint_2"),
        metadata={"frame": "arena"},
        contract="eef_pose.absolute@1",
    )
    descriptor = ActionDescriptor("arm_action", 0, 6, term)

    serialized = descriptor.to_dict()

    assert serialized["name"] == "arm_action"
    assert serialized["slice"] == [0, 6]
    assert serialized["term"]["representation"] == "eef_pose"
    assert serialized["term"]["metadata"] == {"frame": "arena"}
    assert serialized["term"]["contract"] == "eef_pose.absolute@1"


def test_action_term_descriptor_owns_metadata() -> None:
    """Caller mutation cannot change stored descriptor metadata."""
    metadata = {"frame": "arena", "labels": ["left"]}
    descriptor = ActionTermDescriptor(
        representation="joint_position",
        action_dim=1,
        feature_names=("joint",),
        units=("rad",),
        normalization=None,
        joint_names=("joint",),
        metadata=metadata,
    )

    metadata["frame"] = "world"
    metadata["labels"].append("right")

    assert descriptor.to_dict()["metadata"] == {
        "frame": "arena",
        "labels": ["left"],
    }


@pytest.mark.parametrize(
    ("name", "start", "stop", "message"),
    [
        ("", 0, 1, "name must be non-empty"),
        ("arm", -1, 0, "start must be non-negative"),
        ("arm", 0, 2, "slice width must match"),
    ],
)
def test_action_descriptor_rejects_invalid_binding(
    name: str,
    start: int,
    stop: int,
    message: str,
) -> None:
    """Term bindings require a valid name and exact flat slice width."""
    term = ActionTermDescriptor(
        representation="joint_position",
        action_dim=1,
        feature_names=("joint",),
        units=("rad",),
        normalization=None,
        joint_names=("joint",),
        metadata={},
    )

    with pytest.raises(ValueError, match=message):
        ActionDescriptor(name, start, stop, term)
