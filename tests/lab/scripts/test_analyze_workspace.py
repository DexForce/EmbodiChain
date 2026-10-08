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

"""Workspace display keeps its frozen scene without physical integration."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.scripts import analyze_workspace
from embodichain.lab.sim import sim_manager
from embodichain.lab.sim.motion.workspace import analyzer

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize(
    "initial",
    [None, [0.7], [0.7, 0.8]],
    ids=["configured", "requested", "invalid-width"],
)
def test_workspace_display_publishes_without_physics(
    monkeypatch: pytest.MonkeyPatch,
    preview: bool,
    initial: list[float] | None,
) -> None:
    events: list[str] = []
    current = torch.zeros((1, 1))
    target = torch.zeros_like(current)
    expected = torch.tensor([[0.7]]) if initial == [0.7] else torch.zeros_like(current)

    def set_qpos(qpos, joint_ids, target=True):
        output = targets if target else current
        output[:, joint_ids] = qpos

    targets = target
    robot = SimpleNamespace(
        cfg=SimpleNamespace(uid="mock"),
        control_parts={"arm": ["joint"]},
        get_joint_ids=lambda _: [0],
        set_qpos=set_qpos,
    )

    def check_state():
        torch.testing.assert_close(current, expected)
        torch.testing.assert_close(targets, expected)

    class FakeSimulation:
        def __init__(self, cfg):
            self.update = Mock(
                side_effect=AssertionError("Workspace display stepped physics")
            )

        def add_robot(self, **kwargs):
            return robot

        def prepare(self):
            pass

        def sync_render_state(self):
            check_state()

        @contextmanager
        def render_frame(self, **kwargs):
            assert not kwargs.get("force_visualization", False)
            check_state()
            events.append("publish")
            yield
            raise KeyboardInterrupt

        def destroy(self):
            events.append("destroy")
            self.update.assert_not_called()

        @staticmethod
        def flush_cleanup_queue():
            events.append("flush")

    monkeypatch.setattr(sim_manager, "SimulationManager", FakeSimulation)
    monkeypatch.setattr(
        analyze_workspace, "build_robot_cfg", lambda _: (object(), "arm", None)
    )
    monkeypatch.setattr(analyze_workspace, "build_sim_cfg", lambda _: object())
    monkeypatch.setattr(analyze_workspace, "build_analyzer_config", lambda *_: object())
    monkeypatch.setattr(
        analyze_workspace, "_resolve_control_part", lambda _, part: part
    )
    monkeypatch.setattr(analyze_workspace, "_visualization_enabled", lambda _: True)
    monkeypatch.setattr(analyze_workspace, "_viser_enabled", lambda _: False)
    monkeypatch.setattr(analyze_workspace, "_print_summary", lambda *_: None)
    analyze = Mock(side_effect=lambda **_: check_state())

    def construct_analyzer(**kwargs):
        check_state()
        return SimpleNamespace(analyze=analyze)

    monkeypatch.setattr(analyzer, "WorkspaceAnalyzer", construct_analyzer)
    cache_preview = Mock(side_effect=lambda *_: check_state())
    monkeypatch.setattr(analyze_workspace, "preview_cache", cache_preview)
    args = SimpleNamespace(
        robot="mock",
        asset=None,
        init_qpos=initial,
        preview_cache="cache.npz" if preview else None,
        num_samples=1,
        force_recompute=False,
        output=None,
    )
    analyze_workspace.main(args)
    assert events == ["publish", "destroy", "flush"]
    assert cache_preview.call_count == int(preview)
    assert analyze.call_count == int(not preview)
