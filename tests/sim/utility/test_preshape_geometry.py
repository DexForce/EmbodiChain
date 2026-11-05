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

"""Exact source hand sweep captures from real URDF/FK data."""

from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import json, hashlib
import pytest
import torch
from ._preshape_hand_fixture import preshape_fixture
from embodichain.lab.sim.utility.preshape_geometry import (
    capture_preshape_hand_clearance,
)

_DATA = json.loads(
    (Path(__file__).parent / "fixtures/preshape_capture_b0.json").read_text()
)


@pytest.mark.parametrize("case", _DATA["cases"], ids=lambda case: str(case["fraction"]))
def test_preshape_capture_preserves_all_source_points_and_fk_inputs(tmp_path, case):
    robot, cfg = preshape_fixture(tmp_path)
    initial = robot.qpos.clone()
    measured = capture_preshape_hand_clearance(
        robot,
        cfg=cfg,
        expected_qpos=initial,
        table_z=0.0,
        preshape_fraction=case["fraction"],
    )
    assert (
        hashlib.sha256(measured.points.numpy().tobytes()).hexdigest()
        == case["point_bytes_sha256"]
    )
    assert (
        measured.aperture == case["aperture"]
        and measured.max_retreat == case["max_retreat"]
    )
    assert robot.fk_calls == case["fk_calls"]
    assert torch.equal(robot.qpos, initial)


def test_preshape_capture_rejects_changed_observation_or_foreign_calibration(tmp_path):
    robot, cfg = preshape_fixture(tmp_path)
    with pytest.raises(ValueError, match="actual B1"):
        capture_preshape_hand_clearance(
            robot, cfg=cfg, expected_qpos=robot.qpos + 1, table_z=0.0
        )
    cfg.hand_root_link = "foreign"
    with pytest.raises(ValueError, match="root"):
        capture_preshape_hand_clearance(
            robot, cfg=cfg, expected_qpos=robot.qpos, table_z=0.0
        )
