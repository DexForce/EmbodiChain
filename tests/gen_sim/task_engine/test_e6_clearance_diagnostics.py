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

"""Failed E6 sampling distinguishes aperture and table clearance without changing gates."""

from __future__ import annotations

import json

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import e6_clearance as module


@pytest.mark.parametrize("reason", ["opening", "table", "valid"])
def test_failed_filter_records_the_first_geometric_layer(monkeypatch, reason):
    messages = []
    monkeypatch.setattr(module.logger, "log_info", messages.append)
    screen = module.HandClearance(
        torch.tensor([[0.0, 0.0, -0.05]]),
        0.005 if reason == "opening" else 0.02,
        0.03,
        0.0,
    )
    pose = torch.eye(4)[None]
    pose[:, 2, 3] = 0.04 if reason == "table" else 0.1
    widths, costs = torch.tensor([0.01]), torch.tensor([0.0])
    result, output_widths, output_costs = screen.filter(pose, widths, costs)
    assert torch.equal(output_widths, widths) and torch.equal(
        costs, torch.tensor([0.0])
    )
    assert pose[0, 2, 3] == pytest.approx(0.04 if reason == "table" else 0.1)
    if reason == "valid":
        assert torch.isfinite(output_costs).all() and not messages
    else:
        assert not torch.isfinite(output_costs).any()
        record = json.loads(messages[0].split(": ", 1)[1])
        assert record["finite_source_candidates"] == 1
        assert record["opening_pass"] == (0 if reason == "opening" else 1)
        assert record["table_pass"] == 0
        if reason == "table":
            assert record["best_hand_min_z_m"] < record["required_hand_min_z_m"]
