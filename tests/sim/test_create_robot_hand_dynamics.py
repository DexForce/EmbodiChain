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

"""Dynamic regression for the SR5/BrainCo tutorial's coupled hand joints."""

from __future__ import annotations

import pytest
import torch

from scripts.tutorials.sim import create_robot as tutorial

pytestmark = [pytest.mark.requires_sim, pytest.mark.gpu]


@pytest.mark.parametrize("backend", ["default", "newton"])
def test_hand_open_close_stays_bounded(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """Check transients and settling over two complete tutorial cycles."""
    run_simulation = tutorial.run_simulation
    samples: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    metadata: dict = {}

    def record(sim, robot, max_steps=None):
        update = sim.update
        metadata.update(
            hand_ids=robot.get_joint_ids("hand"),
            mimic_ids=robot.mimic_ids,
            parents=robot.mimic_parents,
            multipliers=torch.tensor(robot.mimic_multipliers),
            offsets=torch.tensor(robot.mimic_offsets),
            limits=robot.body_data.qpos_limits[0].cpu().clone(),
        )

        def step(*args, **kwargs):
            result = update(*args, **kwargs)
            samples.append(robot.get_qpos()[0].cpu().clone())
            targets.append(robot.get_qpos(target=True)[0].cpu().clone())
            return result

        with monkeypatch.context() as patch:
            patch.setattr(sim, "update", step)
            run_simulation(sim, robot, max_steps=max_steps)

    monkeypatch.setattr(tutorial, "run_simulation", record)
    args = tutorial.build_parser().parse_args(
        ["--physics", backend, "--headless", "--max-steps", "800"]
    )
    try:
        tutorial.main(args)
    finally:
        tutorial.SimulationManager.flush_cleanup_queue()

    qpos = torch.stack(samples)
    target = torch.stack(targets)
    hand_ids = metadata["hand_ids"]
    limits = metadata["limits"][hand_ids]
    hand_qpos = qpos[:, hand_ids]
    assert torch.isfinite(qpos).all()
    # Allow 0.02 rad of compliant limit deflection, including target switches.
    assert torch.all(hand_qpos >= limits[:, 0] - 0.02)
    assert torch.all(hand_qpos <= limits[:, 1] + 0.02)
    mimic_error = qpos[:, metadata["mimic_ids"]] - (
        qpos[:, metadata["parents"]] * metadata["multipliers"] + metadata["offsets"]
    )
    # The coupling is compliant; constrain its transient error to 0.2 rad.
    assert mimic_error.abs().max().item() < 0.2
    # End of each 1-second command phase, including both hand transitions.
    settled_steps = torch.arange(99, 800, 100)
    settled_error = (qpos - target)[settled_steps][:, hand_ids]
    assert settled_error.abs().max().item() < 0.02
