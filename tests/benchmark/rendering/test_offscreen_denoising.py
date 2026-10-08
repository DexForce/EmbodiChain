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
import json
import sys
import tracemalloc
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.benchmark.rendering import offscreen_denoising as benchmark_module
from scripts.benchmark.rendering.offscreen_denoising import (
    BenchmarkCfg,
    build_leaderboard_rows,
    build_parser,
    compute_quality_metrics,
    _aggregate_quality,
    _build_sweep_leaderboard_rows,
    _temporal_delta_error,
    _write_comparison_image,
    run_all_benchmarks,
    run_environment_sweep,
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


def test_quality_metrics_weight_local_defects() -> None:
    """Local-window SSIM distinguishes concentrated defects from background error."""
    reference = np.zeros((64, 64, 3), dtype=np.uint8)
    concentrated = reference.copy()
    concentrated[31:33, 31:33] = 255
    distributed = reference.copy()
    distributed[:2, :2] = 255

    concentrated_metrics = compute_quality_metrics(reference, concentrated)
    distributed_metrics = compute_quality_metrics(reference, distributed)

    assert concentrated_metrics["mae"] == distributed_metrics["mae"]
    assert concentrated_metrics["ssim"] < distributed_metrics["ssim"]


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


def test_benchmark_cfg_carries_rendering_dimensions_and_modes() -> None:
    """External settings configure camera, environment, and denoising scope."""
    cfg = BenchmarkCfg.from_mapping(
        {
            "resolution": "320x200",
            "camera_count": 3,
            "num_envs": 4,
            "denoising": ["off", "nrd"],
            "measure_frames": 4,
            "quality_frames": 2,
        }
    )

    assert cfg.resolution == (320, 200)
    assert cfg.camera_count == 3
    assert cfg.num_envs == 4
    assert cfg.denoising_modes == ("off", "nrd")
    assert cfg.resolved_reference_mode == "off"
    assert cfg.to_dict()["resolution"] == "320x200"


def test_benchmark_cfg_keeps_legacy_positional_order() -> None:
    """The original worker settings remain valid for callers using keywords or positions."""
    cfg = BenchmarkCfg(320, 200, 1, 2, 1, 3, 0, "cpu")

    assert cfg.resolution == (320, 200)
    assert cfg.measure_frames == 2
    assert cfg.device == "cpu"


def test_benchmark_cfg_rejects_invalid_parallel_settings() -> None:
    """Invalid dimensions and duplicate denoising modes fail before startup."""
    with pytest.raises(ValueError, match="camera_count"):
        BenchmarkCfg(camera_count=0)
    with pytest.raises(ValueError, match="must not contain duplicates"):
        BenchmarkCfg(denoising_modes=("optix", "optix"))


def test_aggregate_quality_counts_parallel_images() -> None:
    """Quality aggregation traverses every configured environment and camera."""
    frames = np.zeros((2, 2, 3, 8, 8, 4), dtype=np.uint8)
    metrics = _aggregate_quality(
        "optix", frames, frames.copy(), fps=10.0, expected_frames=2
    )

    assert metrics["success_rate"] == 100.0
    assert metrics["rendered_images"] == 12
    assert metrics["camera_count"] == 3
    assert metrics["num_envs"] == 2


def test_comparison_image_supports_multiple_cameras(tmp_path) -> None:
    """The comparison artifact keeps one row for each first-environment camera."""
    import cv2

    frames = np.zeros((2, 1, 2, 8, 8, 4), dtype=np.uint8)
    frames[:, :, 1, :, :, 0] = 255
    for mode in ("optix", "nrd"):
        np.savez_compressed(tmp_path / f"{mode}_frames.npz", frames=frames)

    path = _write_comparison_image(tmp_path, frame_index=0)
    image = cv2.imread(str(path))
    assert image is not None
    assert image.shape[0] == 2 * 2 * 8


def test_environment_sweep_parser_and_leaderboard() -> None:
    """A sweep keeps all environment counts while ranking modes by averages."""
    args = build_parser().parse_args(["--env-sweep", "1", "4", "8"])
    assert args.env_sweep == [1, 4, 8]

    rows = _build_sweep_leaderboard_rows(
        [
            {"mode": "optix", "success_rate": 100.0, "ssim": 1.0, "fps": 50.0},
            {"mode": "optix", "success_rate": 100.0, "ssim": 1.0, "fps": 30.0},
            {"mode": "nrd", "success_rate": 100.0, "ssim": 0.8, "fps": 70.0},
            {"mode": "nrd", "success_rate": 100.0, "ssim": 0.8, "fps": 60.0},
        ]
    )
    assert [row["algorithm"] for row in rows] == ["optix", "nrd"]


def test_environment_sweep_rejects_non_positive_counts(tmp_path) -> None:
    """Sweep validation happens before any simulation worker is started."""
    with pytest.raises(ValueError, match="positive integers"):
        run_environment_sweep(BenchmarkCfg(), [1, 0], tmp_path)


@pytest.mark.parametrize(
    "shape", [(5, 12, 8, 4), (5, 3, 12, 8, 4), (5, 3, 2, 12, 8, 4)]
)
def test_temporal_error_matches_full_sequence_formula_without_mutating_frames(
    shape: tuple[int, ...],
) -> None:
    """Bounded aggregation preserves the metric for legacy and batched layouts."""
    rng = np.random.default_rng(20261009)
    reference = rng.integers(0, 256, size=shape, dtype=np.uint8)
    candidate = rng.integers(0, 256, size=shape, dtype=np.uint8)
    original_reference = reference.copy()
    original_candidate = candidate.copy()
    reference_delta = np.diff(reference[..., :3].astype(np.float32) / 255.0, axis=0)
    candidate_delta = np.diff(candidate[..., :3].astype(np.float32) / 255.0, axis=0)
    expected = float(np.mean(np.abs(reference_delta - candidate_delta)))

    actual = _temporal_delta_error(reference, candidate)

    assert actual == pytest.approx(expected, abs=1e-7)
    np.testing.assert_array_equal(reference, original_reference)
    np.testing.assert_array_equal(candidate, original_candidate)


def test_temporal_error_bounds_temporary_memory_for_parallel_sequences() -> None:
    """Temporary float arrays stay small as frame and environment counts grow."""
    rng = np.random.default_rng(20261009)
    shape = (12, 8, 1, 64, 64, 4)
    reference = rng.integers(0, 256, size=shape, dtype=np.uint8)
    candidate = rng.integers(0, 256, size=shape, dtype=np.uint8)
    tracemalloc.start()
    try:
        error = _temporal_delta_error(reference, candidate)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert error > 0.0
    # The former whole-sequence formula allocates about 25.5 MiB here.
    assert peak_bytes < 1024**2


@pytest.mark.parametrize("shape", [(1, 8, 8, 4), (1, 2, 3, 8, 8, 4)])
def test_temporal_error_returns_zero_without_adjacent_frames(
    shape: tuple[int, ...],
) -> None:
    """Single-quality-frame runs have no temporal differences to aggregate."""
    reference = np.zeros(shape, dtype=np.uint8)
    candidate = np.full(shape, 255, dtype=np.uint8)

    assert _temporal_delta_error(reference, candidate) == 0.0


def test_report_generation_keeps_at_most_two_loaded_sequences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Candidate frames and comparison views cannot retain all mode sequences."""
    cfg = BenchmarkCfg(
        width=16, height=16, num_envs=2, measure_frames=2, quality_frames=2
    )
    arrays: list[weakref.ReferenceType[np.ndarray]] = []
    peak_live = 0
    real_load = np.load

    class TrackingArchive:
        """Observe array lifetimes without retaining decompressed frame data."""

        def __init__(self, archive: object) -> None:
            self.archive = archive

        def __getitem__(self, key: str) -> np.ndarray:
            nonlocal peak_live
            array = self.archive[key]
            owner = array
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            arrays.append(weakref.ref(owner))
            peak_live = max(peak_live, sum(ref() is not None for ref in arrays))
            return array

        def __enter__(self) -> TrackingArchive:
            return self

        def __exit__(self, *_: object) -> None:
            self.archive.close()

    def tracking_load(*args: object, **kwargs: object) -> TrackingArchive:
        return TrackingArchive(real_load(*args, **kwargs))

    def fake_worker(command: list[str], *, check: bool) -> None:
        """Write deterministic worker artifacts without initializing a renderer."""
        assert check
        mode = command[command.index("--worker-mode") + 1]
        run_dir = Path(command[command.index("--run-dir") + 1])
        value = {"optix": 20, "dlss": 40, "nrd": 60}[mode]
        frames = np.full((2, 2, 1, 16, 16, 4), value, dtype=np.uint8)
        np.savez_compressed(run_dir / f"{mode}_frames.npz", frames=frames)
        performance = {
            "rendered_images": 4,
            "median_ms": 1.0,
            "mean_ms": 1.0,
            "p95_ms": 1.0,
            "fps": 1000.0,
            "cpu_delta_mb": 0.0,
            "gpu_delta_mb": 0.0,
            "peak_gpu_mb": 0.0,
            "torch_peak_gpu_mb": 0.0,
        }
        (run_dir / f"{mode}_performance.json").write_text(json.dumps(performance))

    monkeypatch.setitem(sys.modules, "dexsim", SimpleNamespace(__version__="test"))
    monkeypatch.setattr(benchmark_module.subprocess, "run", fake_worker)
    monkeypatch.setattr(benchmark_module, "_gpu_description", lambda _: "test GPU")
    monkeypatch.setattr(benchmark_module.np, "load", tracking_load)

    report_path = run_all_benchmarks(cfg, tmp_path)

    assert peak_live <= 2
    assert arrays and all(ref() is None for ref in arrays)
    assert (report_path.parent / "comparison.png").is_file()
    summary = json.loads((report_path.parent / "benchmark_summary.json").read_text())
    assert [row["mode"] for row in summary["metrics"]] == ["optix", "dlss", "nrd"]
    assert summary["metrics"][0]["mae"] == 0.0
