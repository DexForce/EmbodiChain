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

import math

import numpy as np
import pytest

from scripts.benchmark.rendering.offscreen_denoising import (
    build_leaderboard_rows,
    compute_quality_metrics,
    write_markdown_report,
)


def test_quality_metrics_identical_images_are_exact() -> None:
    """Identical inputs report perfect similarity and zero pixel error."""
    image = np.full((16, 16, 3), 127, dtype=np.uint8)

    metrics = compute_quality_metrics(image, image.copy())

    assert math.isinf(metrics["psnr_db"])
    assert metrics["ssim"] == 1.0
    assert metrics["mae"] == 0.0
    assert metrics["edge_mae"] == 0.0


def test_quality_metrics_detect_local_image_error() -> None:
    """A local luminance change lowers similarity and raises both errors."""
    reference = np.zeros((16, 16, 3), dtype=np.uint8)
    candidate = reference.copy()
    candidate[4:12, 4:12] = 255

    metrics = compute_quality_metrics(reference, candidate)

    assert metrics["psnr_db"] == pytest.approx(6.0206, abs=1e-3)
    assert 0.0 < metrics["ssim"] < 1.0
    assert metrics["mae"] == pytest.approx(0.25)
    assert metrics["edge_mae"] > 0.0


def test_leaderboard_sorts_valid_frame_rate_then_similarity() -> None:
    """Leaderboard prioritizes successful captures before reference similarity."""
    rows = build_leaderboard_rows(
        [
            {"mode": "nrd", "success_rate": 100.0, "ssim": 0.92, "fps": 60.0},
            {"mode": "dlss", "success_rate": 100.0, "ssim": 0.95, "fps": 55.0},
            {"mode": "optix", "success_rate": 90.0, "ssim": 1.0, "fps": 70.0},
        ]
    )

    assert [row["algorithm"] for row in rows] == ["dlss", "nrd", "optix"]
    assert [row["rank"] for row in rows] == [1, 2, 3]


def test_report_contains_exactly_three_markdown_tables(tmp_path) -> None:
    """A benchmark run produces the required three-table report contract."""
    report_path = write_markdown_report(
        output_path=tmp_path / "report.md",
        perf_rows=[
            {
                "mode": "dlss",
                "cost_time_ms": 12.5,
                "cpu_delta_mb": 1.0,
                "gpu_delta_mb": 2.0,
                "peak_gpu_mb": 100.0,
            }
        ],
        metric_rows=[
            {
                "mode": "dlss",
                "success_rate": 100.0,
                "psnr_db": 30.0,
                "ssim": 0.95,
            }
        ],
        leaderboard_rows=[
            {
                "rank": 1,
                "algorithm": "dlss",
                "overall_success_rate": 100.0,
            }
        ],
        notes=["OptiX 1 spp is the quality reference."],
    )

    report = report_path.read_text(encoding="utf-8")
    table_separators = [
        line for line in report.splitlines() if line.startswith("| ---")
    ]
    assert len(table_separators) == 3
    assert "## Time & Memory" in report
    assert "## Success & Other Metrics" in report
    assert "## Leaderboard" in report
