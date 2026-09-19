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

"""Frame, backend and lifecycle checks for the simulated workspace tutorial."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from scripts.tutorials.sim import visualize_robot_workspace as tutorial

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("backend", ["dexsim", "viser"])
def test_workspace_keeps_score_alignment_and_adds_arena_offset(backend: str) -> None:
    """A rejected sample must not shift scores onto different world positions."""
    sim = MagicMock()
    sim.arena_offsets = torch.tensor([[10.0, -2.0, 1.0]])
    points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    result = {
        "mode": "cartesian_space",
        "all_points": points.copy(),
        "reachable_points": points[[0, 2]].copy(),
        "reachability_mask": np.array([True, False, True]),
        "joint_configurations": np.zeros((2, 6)),
        "manipulability_scores": np.array([0.1, 0.8]),
    }

    tutorial.publish_workspace(sim, result, backend=backend)

    if backend == "dexsim":
        sim.set_visualization_overlays.assert_not_called()
        call = sim.visualize_point_cloud.call_args.kwargs
        actual_points, colors = call["points"], call["colors"]
        assert call["point_size"] == 3.0
    else:
        sim.visualize_point_cloud.assert_not_called()
        overlay = sim.set_visualization_overlays.call_args.args[0].point_clouds[0]
        actual_points, colors = overlay.points, overlay.colors / 255.0
        assert overlay.point_size == 0.006

    np.testing.assert_allclose(actual_points, points + [10.0, -2.0, 1.0])
    np.testing.assert_allclose(colors[1], [0.62, 0.62, 0.62], atol=1 / 255)
    # Low viridis is blue/purple; high viridis is yellow. The gray point
    # between them must not consume a row from the compact score array.
    assert colors[0, 2] > colors[0, 1]
    assert colors[2, 1] > colors[2, 2]
    np.testing.assert_array_equal(result["all_points"], points)


@pytest.mark.parametrize("failure_phase", [None, "analysis", "render"])
def test_prepare_precedes_analysis_and_cleanup_runs_on_failure(
    monkeypatch: pytest.MonkeyPatch, failure_phase: str | None
) -> None:
    """Robot metadata must be materialized, and failed rendering must release sim."""
    sim = MagicMock()
    sim.arena_offsets = torch.zeros((1, 3))
    manager = MagicMock(return_value=sim)
    monkeypatch.setattr(tutorial, "SimulationManager", manager)
    monkeypatch.setattr(tutorial, "URRobotCfg", MagicMock())
    analyzer = MagicMock()

    def create_analyzer(*args: object, **kwargs: object) -> MagicMock:
        sim.prepare.assert_called_once()
        return analyzer

    monkeypatch.setattr(tutorial, "WorkspaceAnalyzer", create_analyzer)
    monkeypatch.setattr(tutorial, "publish_workspace", MagicMock())
    render = MagicMock()
    monkeypatch.setattr(tutorial, "save_scene_image", render)
    if failure_phase == "analysis":
        analyzer.analyze.side_effect = RuntimeError("analysis failed")
    elif failure_phase == "render":
        render.side_effect = RuntimeError("render failed")

    if failure_phase is None:
        tutorial.main(["--headless", "--num-samples", "32"])
        render.assert_called_once()
    else:
        with pytest.raises(RuntimeError, match=f"{failure_phase} failed"):
            tutorial.main(["--headless", "--num-samples", "32"])

    sim.destroy.assert_called_once_with(exit_process=False)
    manager.flush_cleanup_queue.assert_called_once()


def test_viser_rejects_native_screenshot_before_creating_sim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = MagicMock()
    monkeypatch.setattr(tutorial, "SimulationManager", manager)
    with pytest.raises(SystemExit):
        tutorial.main(["--backend", "viser", "--headless"])
    manager.assert_not_called()
