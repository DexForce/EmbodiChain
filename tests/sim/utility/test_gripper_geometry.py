# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Registered actual hand geometry and runtime-TCP behavior, CPU only."""

from __future__ import annotations

_GRIPPER_IDS = {
    "franka_panda_hand": "FrankaPandaHand",
    "dh_pgi_140_80": "DHPGI14080",
    "cobotmagic_v100_gripper": "CobotMagicV100Gripper",
    "pika_gripper": "PikaGripper",
    "tianji_marvin_gripper": "TianjiMarvinGripper",
    "robotiq_2f85": "RobotiqArg2F85",
    "robotiq_arg2f_140": "RobotiqArg2F140",
}

from embodichain.robot_registry import resolve_gripper_defaults
from embodichain.lab.sim.utility.gripper_geometry import (
    build_gripper_collision_geometry,
)

import numpy as np
import pytest
import torch


@pytest.mark.parametrize(
    "model,side,root,opened,parts",
    [
        ("franka_panda_hand", None, "fr3_hand", 0.04, 3),
        ("dh_pgi_140_80", None, "gripper_base_link_1", 0.0, 3),
        ("cobotmagic_v100_gripper", None, "gripper_base", 0.05, 3),
        ("pika_gripper", None, "GRIPPER_ROOT_LINK", 0.05, 3),
        ("tianji_marvin_gripper", "left", "left_hand_base_link", 0.0, 3),
        ("tianji_marvin_gripper", "right", "right_hand_base_link", 0.0, 3),
        ("robotiq_2f85", None, "robotiq_arg2f_base_link", 0.0, 11),
        ("robotiq_arg2f_140", None, "robotiq_arg2f_base_link", 0.0, 11),
    ],
)
def test_registered_real_hand_assets_are_complete_and_open(
    model, side, root, opened, parts
):
    geometry = build_gripper_collision_geometry(
        resolve_gripper_defaults(_GRIPPER_IDS.get(model, model), variant=side)
    )
    provenance = geometry.provenance
    assert provenance["root_link"] == root
    assert len(provenance["parts"]) == parts
    assert len(provenance["files_sha256"]) >= 3
    assert set(provenance["joint_positions"].values()) == {opened}
    assert len(geometry.points) > 100
    assert torch.isfinite(geometry.points).all()
    # Full link meshes (not fitted scalar boxes) retain exact mesh identities.
    assert all(part["surface_point_count"] > 0 for part in provenance["parts"])
    if model == "cobotmagic_v100_gripper":
        assert (
            len(provenance["mesh_paths"]) == 6
        )  # Exact collision and visual archive-path corrections.


def test_runtime_tcp_override_changes_output_without_moving_hand_geometry():
    default = build_gripper_collision_geometry(resolve_gripper_defaults("DHPGI14080"))
    root_from_tcp = np.eye(4)
    root_from_tcp[2, 3] = 0.17
    overridden = build_gripper_collision_geometry(
        resolve_gripper_defaults("DHPGI14080"), root_from_tcp=root_from_tcp
    )
    torch.testing.assert_close(overridden.points, default.points, rtol=0, atol=1e-8)
    candidate = torch.eye(4).repeat(2, 1, 1)
    candidate[1, :3, 3] = torch.tensor([0.3, -0.2, 0.4])
    result = overridden.canonical_to_tcp_poses(candidate)
    torch.testing.assert_close(
        result[:, :3, 3], candidate[:, :3, 3] + torch.tensor([0, 0, 0.02])
    )
    assert not torch.equal(result, default.canonical_to_tcp_poses(candidate))


def test_native_panda_tcp_uses_y_closing_once_and_retains_contact_origin():
    geometry = build_gripper_collision_geometry(
        resolve_gripper_defaults("FrankaPandaHand")
    )
    canonical = torch.eye(4).unsqueeze(0)
    native = geometry.canonical_to_tcp_poses(canonical)
    torch.testing.assert_close(native[0, :3, 1], torch.tensor([1.0, 0, 0]))
    torch.testing.assert_close(native[0, :3, 2], torch.tensor([0.0, 0, 1]))
    torch.testing.assert_close(native[0, :3, 3], torch.zeros(3))


@pytest.mark.parametrize(
    "model,side",
    [
        ("tianji_marvin_gripper", None),
        ("tianji_marvin_gripper", "other"),
        ("franka_panda_hand", "left"),
        ("custom", None),
    ],
)
def test_geometry_registry_never_guesses_unknown_hand_or_side(model, side):
    with pytest.raises(ValueError):
        build_gripper_collision_geometry(
            resolve_gripper_defaults(_GRIPPER_IDS.get(model, model), variant=side)
        )


def test_contact_frame_override_preserves_root_surfaces_and_runtime_tcp():
    default = build_gripper_collision_geometry(
        resolve_gripper_defaults("RobotiqArg2F140")
    )
    contact = np.array(
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0.21], [0, 0, 0, 1]], dtype=float
    )
    calibrated = build_gripper_collision_geometry(
        resolve_gripper_defaults("RobotiqArg2F140"),
        root_from_tcp=contact,
        root_from_contact_frame=contact,
    )
    # Moving the contact origin by +1cm re-expresses the same mesh at Z-1cm.
    torch.testing.assert_close(
        calibrated.points,
        default.points - torch.tensor([0, 0, 0.01]),
        rtol=0,
        atol=1e-7,
    )
    candidates = torch.eye(4).repeat(2, 1, 1)
    candidates[1, :3, 3] = torch.tensor([0.3, -0.2, 0.4])
    torch.testing.assert_close(
        calibrated.canonical_to_tcp_poses(candidates), candidates
    )
    assert calibrated.provenance["files_sha256"] == default.provenance["files_sha256"]
