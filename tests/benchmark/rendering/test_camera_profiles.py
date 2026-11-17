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

from pathlib import Path
import json

import dexsim
import numpy as np
import pytest
import yaml

from embodichain.lab.sim.cfg import RenderCfg
from scripts.benchmark.rendering import camera_profiles
from scripts.benchmark.rendering.camera_profiles import _gain_interval, _settings


@pytest.mark.parametrize(
    ("filename", "gutter"),
    [("fast_rt_high_throughput.yaml", 0), ("fast_rt_quality_preserving.yaml", 16)],
)
def test_fast_rt_profiles_reach_native_settings(filename: str, gutter: int) -> None:
    """Both rendering profiles use the production loader and explicit guard bands."""
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts/benchmark/rendering/settings"
        / filename
    )
    settings = yaml.safe_load(path.read_text())["render_cfg"]
    config = RenderCfg.from_dict(settings)
    native = dexsim.WorldConfig()

    config.apply_to_dexsim_config(native)

    if gutter == 0:
        workload, _, counts, repeats = _settings(path)
        assert workload.resolution == (640, 480)
        assert counts == [2, 4, 8, 16, 32, 64]
        assert repeats == 5
    assert native.renderer == dexsim.types.Renderer.FASTRT
    assert native.raytrace_config.min_bounces == 1
    assert native.raytrace_config.max_bounces == 8
    assert native.dlss_config.dlss_quality == 2
    assert native.dlss_config.tiled_gutter_pixels == gutter
    assert getattr(native.dlss_config, "tiled_overscan_pixels", 0) == 0
    assert native.dlss_config.frame_time_delta_ms == pytest.approx(1000 / 240)
    assert (
        native.postprocess_config.tone_mapping_type == dexsim.types.ToneMappingType.ACES
    )


def test_process_paired_gain_uses_throughput_direction() -> None:
    """Halving each independent process latency doubles throughput, without scatter."""
    gain, lower, upper = _gain_interval([2.0, 4.0, 8.0], [1.0, 2.0, 4.0])

    assert (gain, lower, upper) == pytest.approx((100.0, 100.0, 100.0))


def test_report_compares_all_three_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known process latencies preserve every profile and pairwise gain direction."""
    (tmp_path / "manifest.json").write_text("{}")
    # Independent contract examples: 4ms -> 2ms doubles throughput; 4ms -> 3ms
    # adds one third; restoring the gutter costs one third versus the 2ms case.
    for profile, latency in (
        ("default", 4.0),
        ("high-throughput", 2.0),
        ("quality-preserving", 3.0),
    ):
        for repeat in range(2):
            payload = {
                "render_ms": [latency, latency],
                "complete_ms": [latency + 1, latency + 1],
                "cpu_delta_mb": 1,
                "gpu_delta_mb": 2,
                "peak_gpu_mb": 10,
                "success_rate": 100,
                "sdk": "test",
                "sdk_path": "test",
                "torch": "test",
                "python": "test",
                "cpu_threads": 4,
                "effective_native": {},
            }
            (tmp_path / f"2_{repeat}_{profile}.json").write_text(json.dumps(payload))
            np.savez_compressed(
                tmp_path / f"2_{repeat}_{profile}.npz",
                rgb=np.full((2, 16, 16, 3), 100, dtype=np.uint8),
                depth=np.ones((2, 16, 16), dtype=np.float32),
            )
    monkeypatch.setattr(camera_profiles, "_plot", lambda *_: None)

    report = camera_profiles._report(tmp_path, [2], 2)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert {row["profile"] for row in summary["performance"]} == {
        "default",
        "high-throughput",
        "quality-preserving",
    }
    assert len(summary["leaderboard"]) == 3
    gains = {
        (row["reference"], row["candidate"]): row["gain_pct"]
        for row in summary["gains"]
    }
    assert gains[("default", "high-throughput")] == pytest.approx(100)
    assert gains[("default", "quality-preserving")] == pytest.approx(100 / 3)
    assert gains[("high-throughput", "quality-preserving")] == pytest.approx(-100 / 3)
    assert all(row["depth_max_abs_m"] == 0 for row in summary["quality"])
    assert (
        sum(line.startswith("| ---") for line in report.read_text().splitlines()) == 3
    )
