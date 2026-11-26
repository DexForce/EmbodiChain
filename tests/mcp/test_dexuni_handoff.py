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

"""Capability checks shared by borrowed observations, attachment and writes."""

from __future__ import annotations

from pathlib import Path

import pytest

from embodichain.lab.sim.dexsim_session.robot_state import _require_joint_solver


@pytest.mark.parametrize(
    "diagnostics",
    [
        {"solver_type": "mujoco_warp"},
        {"solver_type": "dexuni", "features": {"mujoco_solve_enabled": True}},
    ],
)
def test_supported_joint_solver(diagnostics: dict) -> None:
    _require_joint_solver(diagnostics)


@pytest.mark.parametrize(
    "diagnostics",
    [
        {},
        {"solver_type": "auto"},
        {"solver_type": "dexuni"},
        {"solver_type": "dexuni", "features": {"mujoco_solve_enabled": False}},
    ],
)
def test_unsupported_joint_solver(diagnostics: dict) -> None:
    with pytest.raises(ValueError):
        _require_joint_solver(diagnostics)


def test_live_ur5_releases_adapters_registered_after_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import ExitStack
    from unittest.mock import Mock

    from embodichain.lab.sim.dexsim_session import handoff
    from embodichain.lab.sim.dexsim_session.ur5 import _ur5_motion, ur5_pick_place

    deployment = tmp_path / "deployment.yaml"
    deployment.touch()
    monkeypatch.setenv("EMBODICHAIN_UR5_DEPLOYMENT", str(deployment))
    monkeypatch.setenv("EMBODICHAIN_MOTION_OUTPUT", str(tmp_path / "jobs"))
    adapter = Mock()
    episode = Mock()

    def build_episode(context, deployment, video, resources: ExitStack, **binding):
        episode.register_adapter.side_effect = lambda: resources.callback(adapter.close)
        return episode

    episode_factory = Mock(side_effect=build_episode)
    motion = Mock()
    motion.jobs.active = None
    motion.operations.return_value = {}
    monkeypatch.setattr(ur5_pick_place, "_Episode", episode_factory)
    monkeypatch.setattr(_ur5_motion, "Ur5Motion", Mock(return_value=motion))

    binding = {
        "profile": "ur5-pick-place",
        "paths": {"UR5": "robot", "cube": "cube", "platform": "left"},
        "destinations": ["left", "right"],
    }
    world = handoff.BoundWorld(Mock(), b"x" * 32, binding)
    episode.register_adapter()
    adapter.close.assert_not_called()
    world.close()
    adapter.close.assert_called_once_with()
