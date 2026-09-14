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
from unittest.mock import MagicMock

import pytest
import torch

from scripts.tutorials.sim import create_robot as tutorial

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("failure_phase", [None, "create", "prepare", "run"])
def test_main_cleans_up_without_suppressing_errors(
    monkeypatch: pytest.MonkeyPatch, failure_phase: str | None
) -> None:
    args = SimpleNamespace(
        max_steps=400,
        device="cpu",
        renderer="hybrid",
        physics="newton",
        num_envs=1,
        headless=True,
    )
    sim = MagicMock()
    robot = SimpleNamespace(dof=16)
    create = MagicMock(return_value=robot)
    run = MagicMock()
    failures = {"create": create, "prepare": sim.prepare, "run": run}
    if failure_phase is not None:
        failures[failure_phase].side_effect = RuntimeError("simulation failed")
    monkeypatch.setattr(tutorial.argparse.ArgumentParser, "parse_args", lambda _: args)
    monkeypatch.setattr(tutorial, "SimulationManager", lambda _: sim)
    monkeypatch.setattr(tutorial, "visualization_cfg_from_args", lambda _: None)
    monkeypatch.setattr(tutorial, "create_robot", create)
    monkeypatch.setattr(tutorial, "run_simulation", run)

    if failure_phase is None:
        tutorial.main()
        run.assert_called_once_with(sim, robot, max_steps=400)
    else:
        with pytest.raises(RuntimeError, match="simulation failed"):
            tutorial.main()

    sim.destroy.assert_called_once_with(exit_process=False)


def test_run_simulation_propagates_step_error_to_scene_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = SimpleNamespace(device="cpu", num_envs=1, update=MagicMock())
    sim.update.side_effect = RuntimeError("physics step failed")
    robot = MagicMock()
    robot.get_joint_ids.return_value = [0]
    robot.body_data.qpos_limits = torch.tensor([[[0.0, 1.0]]])
    monkeypatch.setattr(
        tutorial, "_expand_mimic_targets", lambda *args: torch.zeros((1, 1))
    )

    with pytest.raises(RuntimeError, match="physics step failed"):
        tutorial.run_simulation(sim, robot, max_steps=1)
