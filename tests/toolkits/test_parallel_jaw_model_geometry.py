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

import numpy as np
import pytest
import torch

from embodichain.toolkits.graspkit.pg_grasp.model_geometry import (
    get_parallel_jaw_collision_geometry,
)


@pytest.mark.parametrize(
    "model,side,root,opened",
    [
        ("franka_panda_hand", None, "fr3_hand", 0.04),
        ("dh_pgi_140_80", None, "gripper_base_link_1", 0.0),
        ("cobotmagic_v100_gripper", None, "gripper_base", 0.05),
        ("pika_gripper", None, "GRIPPER_ROOT_LINK", 0.05),
        ("tianji_marvin_gripper", "left", "left_hand_base_link", 0.0),
        ("tianji_marvin_gripper", "right", "right_hand_base_link", 0.0),
    ],
)
def test_registered_real_hand_assets_are_complete_and_open(model, side, root, opened):
    geometry = get_parallel_jaw_collision_geometry(model, side=side)
    provenance = geometry.provenance
    assert provenance["root_link"] == root
    assert len(provenance["parts"]) == 3
    assert len(provenance["files_sha256"]) >= 3
    assert set(provenance["joint_positions"].values()) == {opened}
    assert len(geometry.points) > 100
    assert torch.isfinite(geometry.points).all()
    # Full link meshes (not fitted scalar boxes) retain exact mesh identities.
    assert all(part["surface_point_count"] > 0 for part in provenance["parts"])
    if model == "cobotmagic_v100_gripper":
        assert len(provenance["mesh_paths"]) == 3


def test_runtime_tcp_override_changes_output_without_moving_hand_geometry():
    default = get_parallel_jaw_collision_geometry("dh_pgi_140_80")
    root_from_tcp = np.eye(4)
    root_from_tcp[2, 3] = 0.17
    overridden = get_parallel_jaw_collision_geometry(
        "dh_pgi_140_80", root_from_tcp=root_from_tcp
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
    geometry = get_parallel_jaw_collision_geometry("franka_panda_hand")
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
        get_parallel_jaw_collision_geometry(model, side=side)
