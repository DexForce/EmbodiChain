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

"""CPU contract tests for the complete pure-rendering benchmark suite."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_rendering_catalog_contains_all_issue_679_render_tracks() -> None:
    from scripts.benchmark.rendering.suite import EXPERIMENT_IDS, experiment_catalog

    catalog = experiment_catalog()
    assert tuple(catalog) == EXPERIMENT_IDS
    assert set(catalog) == {"R-03", "R-04", "R-05", "R-06", "R-09"}
    assert catalog["R-04"].parameter_matrix["num_envs"] == (1, 4, 16)
    assert catalog["R-05"].parameter_matrix["modalities"] == (
        "rgb",
        "rgb_depth",
        "rgb_normals",
    )


def test_rendering_cases_have_shared_scene_and_explicit_boundaries() -> None:
    from scripts.benchmark.rendering.suite import (
        case_config,
        expand_experiment_cases,
    )

    for experiment_id in ("R-03", "R-04", "R-05", "R-06", "R-09"):
        cases = expand_experiment_cases(experiment_id)
        assert cases
        for case in cases:
            cfg = case_config(experiment_id, case.parameters)
            assert cfg.experiment_id == experiment_id
            assert cfg.width > 0 and cfg.height > 0
            assert cfg.num_envs >= 1
            assert cfg.cameras_per_env >= 1
            assert cfg.modalities
            assert cfg.delivery in {
                "render_only",
                "host_readback",
                "duplicate_readback",
            }
            assert cfg.physics_steps_in_measurement == 0
            assert cfg.scene_version == "three_boxes_v1"


def test_rendering_case_rejects_unsupported_cross_platform_modality() -> None:
    from scripts.benchmark.rendering.suite import RenderCaseCfg

    with pytest.raises(ValueError, match="modalities"):
        RenderCaseCfg(modalities=("rgb", "semantic_segmentation"))
    with pytest.raises(ValueError, match="delivery"):
        RenderCaseCfg(delivery="gpu_magic")


def test_capture_packet_validation_preserves_delivery_evidence() -> None:
    from scripts.benchmark.rendering.suite import (
        CapturePacket,
        RenderCaseCfg,
        validate_capture_packet,
    )

    cfg = RenderCaseCfg(width=4, height=3, measured_frames=1)
    packet = CapturePacket(
        arrays={"rgb": np.zeros((1, 3, 4, 3), dtype=np.uint8)},
        render_calls=1,
        readback_calls=1,
        gpu_sync_calls=1,
        host_bytes=36,
        exposure_count=1,
        delivery="host_readback",
    )
    validate_capture_packet(packet, cfg)
    assert packet.to_metadata()["host_bytes"] == 36
    assert packet.to_metadata()["exposure_count"] == 1


def test_suite_report_groups_cases_and_keeps_comparison_unqualified(
    tmp_path: Path,
) -> None:
    from scripts.benchmark.rendering.suite_report import rebuild_report

    rows = []
    for backend, value in (("embodichain", 100.0), ("isaaclab", 50.0)):
        rows.append(
            {
                "schema_version": 1,
                "experiment_id": "R-04",
                "run_id": f"{backend}-0",
                "run_dir": f"r00_{backend}_case-0000",
                "backend": backend,
                "repeat": 0,
                "case_id": "case-0000",
                "status": "completed",
                "quality_status": "not_qualified",
                "config_sha256": "same",
                "hardware_id": "gpu",
                "metrics": {
                    "camera_frames_per_s": value,
                    "observations_per_s": value,
                    "latency_p50_s": 0.01,
                    "latency_p95_s": 0.02,
                    "host_readback_bytes_per_s": 1000.0,
                },
                "validation": {"freshness_probe_passed": True},
            }
        )
    (tmp_path / "runs.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "definition.json").write_text(
        json.dumps({"experiment_id": "R-04"}), encoding="utf-8"
    )

    report = rebuild_report(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    text = report.read_text()
    assert "R-04" in text
    assert "case-0000" in text
    assert summary["comparison_status"] == "not_qualified"
    assert summary["comparisons"]["case-0000"]["ratio"] is None
    assert (tmp_path / "metrics.json").exists()


def test_suite_list_and_report_only_paths_do_not_import_simulators(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    (tmp_path / "runs.json").write_text("[]", encoding="utf-8")
    (tmp_path / "definition.json").write_text(
        json.dumps({"experiment_id": "R-03"}), encoding="utf-8"
    )
    script = """
import importlib.abc
import sys
class BlockSim(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'dexsim', 'isaaclab', 'isaacsim'}:
            raise AssertionError('Unexpected simulator import: ' + fullname)
sys.meta_path.insert(0, BlockSim())
from scripts.benchmark.__main__ import main
main(['rendering-suite', '--list'])
main(['rendering-suite', '--report-only', sys.argv[1]])
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "R-03" in result.stdout
    assert (tmp_path / "report.md").exists()


def test_suite_worker_validation_rejects_wrong_batch_shape() -> None:
    from scripts.benchmark.rendering.suite import (
        CapturePacket,
        RenderCaseCfg,
        validate_capture_packet,
    )

    cfg = RenderCaseCfg(width=4, height=3, num_envs=2, measured_frames=1)
    packet = CapturePacket(
        arrays={"rgb": np.zeros((1, 3, 4, 3), dtype=np.uint8)},
        render_calls=1,
        readback_calls=1,
        gpu_sync_calls=1,
        host_bytes=36,
        exposure_count=1,
        delivery="host_readback",
    )
    with pytest.raises(ValueError, match="batch"):
        validate_capture_packet(packet, cfg)
