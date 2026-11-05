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

"""Carrier observations retain source history checks and unavailable row identity."""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import pytest
import torch
from embodichain.lab.sim.evidence import PhysicalEvidenceRequest
from embodichain.lab.sim.sensors.cargo_envelope import (
    capture_carrier_contents,
    observe_carrier_contents,
)

GOLDEN = json.loads(
    (Path(__file__).parent / "data/cargo_envelope_b0.json").read_text()
)["cases"]


def fixture(dtype=torch.float32, mode="accepted"):
    B, T = 2, 130
    carrier = torch.eye(4, dtype=dtype).repeat(B, 1, 1)
    cargo = carrier.clone()
    cargo[:, 2, 3] = 0.06
    if mode == "unselected_row":
        cargo[1, 0, 3] = 2.0
    loose = carrier.clone()
    loose[:, 0, 3] = 3.0
    poses = {"tray": carrier, "cargo": cargo, "tool": cargo.clone(), "loose": loose}
    geometry = {
        "tray": torch.tensor([[-0.5, -0.4, 0.0], [0.5, 0.4, 0.2]], dtype=dtype),
        **{
            name: torch.tensor([[-0.03, -0.03, -0.03], [0.03, 0.03, 0.03]], dtype=dtype)
            for name in ("cargo", "tool", "loose")
        },
    }
    request = PhysicalEvidenceRequest(0.0, 0, torch.tensor([7, 3]))
    bindings = capture_carrier_contents(
        geometry,
        poses.__getitem__,
        request,
        episode_ids=(2, 5),
        carrier_ids=("tray",),
        excluded_ids=("tool",),
    )
    a = torch.zeros(B, T, 7, dtype=dtype)
    a[:, :, 6] = 1.0
    b = a.clone()
    b[:, :, 2] = 0.06
    if mode.startswith("frame"):
        b[0, int(mode[5:]), 0] = 1.0
    if mode == "terminal_escape":
        poses["cargo"][1, 0, 3] = 1.0
        b[1, -1, 0] = 1.0
    trajectory = {"tray": {"pose": a}, "cargo": {"pose": b}}
    final = PhysicalEvidenceRequest(5.2, 130, request.env_ids)
    return bindings, poses, final, trajectory


@pytest.mark.parametrize("pose_format", ["xyz_xyzw", "matrix"])
@pytest.mark.parametrize("case", GOLDEN, ids=lambda c: c["mode"] + "-" + c["dtype"])
def test_envelope_terminal_and_complete_history_match_frozen_source(case, pose_format):
    bindings, poses, request, trajectory = fixture(
        getattr(torch, case["dtype"]), case["mode"]
    )
    assert len(bindings) == 1
    if pose_format == "matrix":
        from embodichain.utils.math import xyz_quat_to_4x4_matrix

        trajectory = {
            name: {
                "pose": torch.stack(
                    [xyz_quat_to_4x4_matrix(row) for row in value["pose"]]
                )
            }
            for name, value in trajectory.items()
        }
    observation = observe_carrier_contents(
        bindings,
        poses.__getitem__,
        request,
        episode_ids=(2, 5),
        invocation_id="program/final:0",
        trajectory=None if case["mode"] == "terminal" else trajectory,
        step_counts=[130, 130],
        trajectory_pose_format=pose_format,
    )
    key = bindings[0].evidence_id
    batch = observation.frame.get(key)
    expected = case["report"]
    assert batch.valid.tolist() == [True, True]
    assert batch.acquisition_errors == (None, None)
    assert batch.values.tolist() == expected["accepted_mask"]
    assert (
        observation.initial_masks[key].tolist()
        == expected["contents"][0]["initial_mask"]
    )
    assert (
        list(observation.first_failed_frames[key])
        == expected["contents"][0]["first_failed_frames"]
    )
    assert observation.scope == expected["scope"]
    assert observation.frame.env_ids.tolist() == [7, 3]
    assert observation.episode_ids == (2, 5)
    assert observation.initial_timestamp == 0.0 and observation.frame.timestamp == 5.2


@pytest.mark.parametrize(
    "missing",
    [
        "terminal_pose",
        "history_pose",
        "valid_prefix",
        "nonfinite_row",
        "nonfinite_history",
    ],
)
def test_missing_cargo_facts_retain_unavailable_rows(missing):
    bindings, poses, request, trajectory = fixture()
    counts = [130, 130]
    if missing == "terminal_pose":
        poses.pop("cargo")
    elif missing == "history_pose":
        trajectory.pop("cargo")
    elif missing == "valid_prefix":
        counts[1] = 0
    elif missing == "nonfinite_history":
        trajectory["cargo"]["pose"][1, 65, 0] = float("nan")
    elif missing == "nonfinite_row":
        poses["cargo"][1, 0, 3] = float("nan")
    observation = observe_carrier_contents(
        bindings,
        poses.__getitem__,
        request,
        episode_ids=(2, 5),
        invocation_id="program/final:0",
        trajectory=trajectory,
        step_counts=counts,
    )
    batch = observation.frame.get(bindings[0].evidence_id)
    expected = (
        [False, False]
        if missing in {"terminal_pose", "history_pose"}
        else [True, False]
    )
    assert batch.valid.tolist() == expected
    assert all(
        (error is None) == valid
        for error, valid in zip(batch.acquisition_errors, expected)
    )


def test_cargo_binding_owns_initial_geometry_and_rejects_episode_or_row_drift():
    bindings, poses, request, trajectory = fixture()
    for kwargs in (
        {"episode_ids": (3, 5)},
        {"request": PhysicalEvidenceRequest(5.2, 130, torch.tensor([3, 7]))},
    ):
        arguments = dict(
            request=request, episode_ids=(2, 5), invocation_id="program/final:0"
        )
        arguments.update(kwargs)
        with pytest.raises(ValueError, match="rows/episodes"):
            observe_carrier_contents(bindings, poses.__getitem__, **arguments)
    first = observe_carrier_contents(
        bindings,
        poses.__getitem__,
        request,
        episode_ids=(2, 5),
        invocation_id="program/final:0",
    )
    key = bindings[0].evidence_id
    batch = first.frame.get(key)
    batch.values.fill_(False)
    assert first.frame.get(key).values.tolist() == [True, True]
    assert first.first_failed_frames[key] == (None, None)


def test_initial_cargo_geometry_unavailable_cannot_silently_drop_contents():
    pose = torch.eye(4)[None]
    geometry = {
        "tray": torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        "cargo": torch.tensor([[float("nan"), 0.0, 0.0]]),
    }
    with pytest.raises(ValueError, match="initial geometry"):
        capture_carrier_contents(
            geometry,
            lambda name: pose,
            PhysicalEvidenceRequest(0.0, 0, torch.tensor([0])),
            episode_ids=(1,),
            carrier_ids=("tray",),
        )
