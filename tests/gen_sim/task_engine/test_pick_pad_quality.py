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

"""Native pad-line quality is additive and stops at first nominal contact."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh
import torch
from types import SimpleNamespace

from embodichain.gen_sim.task_engine._task_program.pick_contact import (
    ClosingContactScreen,
    ContactPart,
    _quality_gravity,
)


def _screen(revision: int = 9, gravity=(0.0, 0.0, -1.0), *, axis=0, late_tilt=False):
    parts = []
    for sign in (-1, 1):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, axis, 3] = sign * np.array([0.15, 0.105, 0.08])
        if late_tilt:
            poses[2, 2, 3] = sign * 0.08
        parts.append(
            ContactPart(
                str(sign),
                (trimesh.creation.box(extents=(0.02, 0.02, 0.02)),),
                poses,
                0.0,
                True,
                0.002,
            )
        )
    return ClosingContactScreen(
        tuple(parts),
        (trimesh.creation.box(extents=(0.2, 0.2, 0.2)),),
        0.0,
        revision=revision,
        target_contact_offset=0.002,
        **({"gravity_in_object": np.asarray(gravity)} if revision == 9 else {}),
    )


@pytest.mark.parametrize("angle, accepted", [(0.0, True), (1.9, True), (2.1, False)])
def test_native_pad_tilt_boundary(angle: float, accepted: bool) -> None:
    screen = _screen()
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("y", angle, degrees=True).as_matrix()
    result = screen.inspect(pose)
    assert result["valid"] is accepted
    assert screen.inspect(pose) == result
    assert len(screen.cache) == 1


def test_actual_pad_axis_not_tcp_x_and_not_postcontact_penetration() -> None:
    screen = _screen(axis=1, late_tilt=True)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("y", 20, degrees=True).as_matrix()
    result = screen.inspect(pose)
    assert result["valid"]
    assert result["checked_samples"] == 2


def test_quality_uses_bound_gravity_and_checks_later_precontact_samples() -> None:
    assert not _screen(gravity=(1.0, 0.0, 0.0)).inspect(np.eye(4))["valid"]
    screen = _screen()
    # Keep the initial pad line level but tilt the second, contact-time sample.
    for part in screen.parts:
        part.tcp_poses[1, 2, 3] = float(part.name) * 0.01
    # Rebuild so the immutable captured geometry contains the changed FK grid.
    screen = ClosingContactScreen(
        screen.parts,
        (trimesh.creation.box(extents=(0.2, 0.2, 0.2)),),
        0.0,
        revision=9,
        target_contact_offset=0.002,
        gravity_in_object=np.array([0.0, 0.0, -1.0]),
    )
    assert not screen.inspect(np.eye(4))["valid"]


@pytest.mark.parametrize("gravity", [None, (0, 0, 0), (0, np.nan, -1), (0, -1)])
def test_missing_or_invalid_quality_frame_is_rejected(gravity) -> None:
    with pytest.raises(ValueError):
        _screen(gravity=gravity)


def test_revision_eight_retains_legacy_candidate_and_report() -> None:
    screen = _screen(revision=8)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("y", 10, degrees=True).as_matrix()
    assert screen.inspect(pose)["valid"]
    assert "pad_line_quality" not in screen.report()


def test_quality_gravity_captures_nonidentity_object_frame(monkeypatch) -> None:
    from embodichain.lab.sim.atomic_actions import goals

    pose = torch.eye(4, dtype=torch.float64)[None]
    pose[0, :3, :3] = torch.from_numpy(
        Rotation.from_euler("z", 90, degrees=True).as_matrix()
    )
    monkeypatch.setattr(goals, "_resolve_object_pose", lambda *args, **kwargs: pose)
    simulation = SimpleNamespace(
        num_envs=1,
        physics_backend="default",
        is_default_backend=True,
        sim_config=SimpleNamespace(
            physics_cfg=SimpleNamespace(gravity=(0.0, -9.81, 0.0))
        ),
    )
    request = SimpleNamespace(goal=SimpleNamespace(semantics=None, object_pose=None))
    result = _quality_gravity(simulation, request, SimpleNamespace(batch_size=1))
    np.testing.assert_allclose(result, (-9.81, 0.0, 0.0), atol=1e-10)
    with pytest.raises(ValueError, match="current B1"):
        _quality_gravity(simulation, request, None)


def test_world_candidates_remain_level_under_rotated_object_frame() -> None:
    object_pose = torch.eye(4, dtype=torch.float64)[None]
    object_pose[0, :3, :3] = torch.from_numpy(
        Rotation.from_euler("y", 90, degrees=True).as_matrix()
    )
    gravity = object_pose[0, :3, :3].numpy().T @ np.array([0.0, 0.0, -1.0])
    screen = _screen(gravity=gravity)
    candidates = torch.eye(4, dtype=torch.float64).repeat(1, 2, 1, 1)
    candidates[0, 1, :3, :3] = torch.from_numpy(
        Rotation.from_euler("y", 10, degrees=True).as_matrix()
    )
    assert screen.mask(candidates, object_pose).tolist() == [[True, False]]
    assert screen.mask(candidates, object_pose).tolist() == [[True, False]]
    assert len(screen.cache) == 2
