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

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = pytest.mark.no_sim

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_TUTORIAL_PATH = (
    _REPOSITORY_ROOT / "scripts/tutorials/atomic_action/pickup_rubiks_cube.py"
)


def _load_tutorial_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "pickup_rubiks_cube_tutorial", _TUTORIAL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeCube:
    """Minimal articulation stand-in recording drive and qpos writes."""

    def __init__(self, num_envs: int = 1, root_pos=(0.0, 0.0, 0.0)) -> None:
        self.uid = "rubiks_cube"
        self.device = torch.device("cpu")
        self.joint_names = ["top_turn"]
        self._num_envs = num_envs
        self._root_pos = root_pos
        self.drive_calls: list[dict] = []
        self.qpos_calls: list[dict] = []
        self.cleared = False

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        pose = torch.eye(4).unsqueeze(0).repeat(self._num_envs, 1, 1)
        pose[:, :3, 3] = torch.tensor(self._root_pos)
        return pose

    def set_joint_drive(self, **kwargs) -> None:
        self.drive_calls.append(kwargs)

    def set_qpos(self, qpos: torch.Tensor, **kwargs) -> None:
        self.qpos_calls.append({"qpos": qpos, **kwargs})

    def clear_dynamics(self) -> None:
        self.cleared = True


class _FakeSimulation:
    """Simulation stand-in capturing the spawned articulation configuration."""

    def __init__(self, cube: _FakeCube) -> None:
        self._cube = cube
        self.captured_cfg = None
        self.prepared = 0
        self.updates: list[int] = []

    def add_articulation(self, cfg):
        self.captured_cfg = cfg
        return self._cube

    def prepare(self) -> None:
        self.prepared += 1

    def update(self, step: int = 1) -> None:
        self.updates.append(step)


def test_spawn_compensates_scene_engine_bottom_center_offset(tmp_path) -> None:
    """The cube spawns resting on the table despite the Y-authored offset.

    The Scene Engine seats articulated USDC assets on their bounding-box
    minimum along Y while stages are Z-up, so the asset carries a
    ``(0, +edge/2, 0)`` root translate. The tutorial must cancel that lateral
    shift and raise the root by half an edge so the cube rests where asked.
    """
    tutorial = _load_tutorial_module()
    asset = tmp_path / "rubiks_cube_001.usdc"
    asset.write_bytes(b"PXR-USDC")
    cube = _FakeCube()
    sim = _FakeSimulation(cube)

    tutorial.create_pick_object(sim, str(asset))

    init_pos = sim.captured_cfg.init_pos
    offset_x, offset_y, _ = tutorial.CUBE_BOTTOM_CENTER_OFFSET
    assert init_pos[0] == pytest.approx(tutorial.OBJECT_XY[0] - offset_x)
    assert init_pos[1] == pytest.approx(tutorial.OBJECT_XY[1] - offset_y)
    assert init_pos[2] == pytest.approx(tutorial.CUBE_EDGE / 2.0)


def test_spawn_requests_a_floating_root(tmp_path) -> None:
    """The cube spawns with a free root so the lift can actually move it.

    ``ArticulationCfg`` fixes roots to the world by default, which suits
    drawers and doors. Left at the default the cube is welded to the table:
    PickUp still reports ``plan_success`` and replays its whole trajectory
    while the object never moves, so only a physical check catches it.
    """
    tutorial = _load_tutorial_module()
    asset = tmp_path / "rubiks_cube_001.usdc"
    asset.write_bytes(b"PXR-USDC")
    sim = _FakeSimulation(_FakeCube())

    tutorial.create_pick_object(sim, str(asset))

    assert sim.captured_cfg.root_props.fixed_base is False


def test_spawn_rejects_missing_asset(tmp_path) -> None:
    """A missing asset fails loudly instead of surfacing as a spawn error."""
    tutorial = _load_tutorial_module()
    sim = _FakeSimulation(_FakeCube())

    with pytest.raises(FileNotFoundError, match="rubiks_cube_001.usdc"):
        tutorial.create_pick_object(sim, str(tmp_path / "absent.usdc"))


def test_lock_turn_joint_pins_top_turn_at_zero() -> None:
    """The turn joint is driven stiffly to zero so the cube grasps as one body.

    The asset ships ``top_turn`` with zero stiffness and zero damping; without
    this drive the top layer swings freely during the lift.
    """
    tutorial = _load_tutorial_module()
    cube = _FakeCube(num_envs=3)

    tutorial.lock_turn_joint(cube)

    assert len(cube.drive_calls) == 1
    drive = cube.drive_calls[0]
    assert drive["joint_ids"] == [0]
    assert drive["target_mode"] == "position"
    assert torch.allclose(
        drive["stiffness"], torch.full((3, 1), tutorial.LOCK_STIFFNESS)
    )
    assert torch.allclose(drive["damping"], torch.full((3, 1), tutorial.LOCK_DAMPING))
    assert len(cube.qpos_calls) == 1
    assert torch.allclose(cube.qpos_calls[0]["qpos"], torch.zeros(3, 1))


def test_cube_center_pose_offsets_root_translation() -> None:
    """Grasp planning targets the cube center, not its offset root frame."""
    tutorial = _load_tutorial_module()
    cube = _FakeCube(num_envs=2, root_pos=(0.1, -0.2, 0.3))

    center = tutorial.cube_center_pose(cube)

    expected = torch.tensor((0.1, -0.2, 0.3)) + torch.tensor(
        tutorial.CUBE_BOTTOM_CENTER_OFFSET
    )
    assert center.shape == (2, 4, 4)
    assert torch.allclose(center[:, :3, 3], expected.expand(2, 3), atol=1e-6)
    assert torch.allclose(center[:, :3, :3], torch.eye(3).expand(2, 3, 3))


def test_cube_edge_matches_locked_single_dof_asset() -> None:
    """The measured constants stay pinned to the shipped asset geometry.

    The authored offset is close to but not exactly ``edge / 2``: the asset's
    local bounds run from ``-0.02878`` to ``+0.02882``, so its geometry is
    roughly 0.02 mm off-center. The offset therefore tracks the measured
    translate rather than a derived half-edge.
    """
    tutorial = _load_tutorial_module()

    assert tutorial.CUBE_EDGE == pytest.approx(0.0576)
    assert tutorial.TURN_JOINT == "top_turn"
    assert tutorial.CUBE_BOTTOM_CENTER_OFFSET[0] == 0.0
    assert tutorial.CUBE_BOTTOM_CENTER_OFFSET[2] == 0.0
    assert tutorial.CUBE_BOTTOM_CENTER_OFFSET[1] == pytest.approx(
        tutorial.CUBE_EDGE / 2.0, abs=1e-4
    )
