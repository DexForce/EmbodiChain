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
"""Verify the standalone RJ45 replay's configured sampling cadence."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize(
    ("samples", "substeps_per_sample", "diagnostic_rate_hz"),
    [(4, 8, 240), (8, 4, 480)],
)
def test_rj45_replay_uses_the_configured_contact_cadence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    samples: int,
    substeps_per_sample: int,
    diagnostic_rate_hz: int,
) -> None:
    scripts = Path(__file__).resolve().parents[2] / (
        "scripts/benchmark/tech_reports/demonstrations/platforms"
    )
    monkeypatch.syspath_prepend(str(scripts))
    import franka_rj45_cycle_inputs as inputs

    config, _, _ = inputs.load_scene()
    deployment = yaml.safe_load(inputs.TASK_CONFIG.read_text())
    component = inputs.TASK_CONFIG.parent / deployment["evaluation"]["component"]
    evaluation = yaml.safe_load(component.read_text())
    evaluation["measurements"]["samples_per_control"] = samples
    (tmp_path / "evaluation.yaml").write_text(yaml.safe_dump(evaluation))
    task_config = tmp_path / "task.yaml"
    task_config.write_text("evaluation: {component: evaluation.yaml}\n")
    monkeypatch.setattr(inputs, "TASK_CONFIG", task_config)

    measurements = inputs.load_measurements(config)

    assert measurements.samples_per_control == samples
    assert measurements.substeps_per_sample == substeps_per_sample
    assert measurements.diagnostic_rate_hz == diagnostic_rate_hz
    assert measurements.evaluate([])["diagnostic_rate_hz"] == diagnostic_rate_hz
