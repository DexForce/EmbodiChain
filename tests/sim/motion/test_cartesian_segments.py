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

"""Public Cartesian segment recovery against the frozen source control flow."""

from __future__ import annotations
import hashlib, json, random
from pathlib import Path
import numpy as np
import pytest
import torch
from embodichain.lab.sim.motion import cartesian
from embodichain.lab.sim.motion.motion_generator import (
    MotionGenOptions,
    MotionGenerator,
)
from _cartesian_segment_fixture import segment_fixture

_REFERENCE = json.loads(
    (Path(__file__).parent / "data/cartesian_segment_reference.json").read_text()
)["cases"]


@pytest.mark.parametrize("reference", _REFERENCE, ids=lambda item: item["case"]["name"])
def test_segment_sampling_retries_preserve_source_counts_paths_and_rng(
    reference, monkeypatch
):
    case = reference["case"]
    generator, targets, calls, continuous, recover = segment_fixture(case)
    monkeypatch.setattr(cartesian, "resolve_continuous_ik_path", recover)
    options = MotionGenOptions(
        strategy="ik_interp",
        interpolation_mode="fixed_grid",
        ik_seed_mode="sequential",
        sample_count=case["count"],
        start_qpos=torch.zeros(1, 1),
        control_part="arm",
        interpolation_dt=0.04,
        segment_sample_count_retries=(260, 320),
        segment_continuous_ik_recovery=True,
    )
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    value = cartesian.generate_cartesian_motion(
        generator, targets, options, defer_recovery=case["defer"]
    )
    result = value.result
    digest = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
    actual = {
        "case": case,
        "calls": calls,
        "continuous": continuous,
        "used_continuity": value.used_continuity,
        "success": result.success.tolist(),
        "positions_sha256": digest(result.positions),
        "dt_sha256": digest(result.dt),
        "after_rng": [
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
    }
    assert actual == reference
    assert value.attempted_counts == tuple(call["count"] for call in calls)
    assert (
        options.segment_sample_count_retries == (260, 320)
        and not options.preserve_cartesian_samples
    )


def test_ordinary_motion_generate_cannot_silently_ignore_segment_policy():
    generator, targets, calls, _, _ = segment_fixture({"outcome": "nominal"})
    options = MotionGenOptions(
        strategy="ik_interp",
        interpolation_mode="fixed_grid",
        ik_seed_mode="sequential",
        sample_count=4,
        start_qpos=torch.zeros(1, 1),
        control_part="arm",
        interpolation_dt=0.04,
        segment_sample_count_retries=(260, 320),
    )
    with pytest.raises(ValueError, match="generate_cartesian_motion"):
        MotionGenerator.generate(generator, targets, options)
    assert not calls
