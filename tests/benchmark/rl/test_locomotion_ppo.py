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
"""Locomotion workload, throughput aggregation, and memory/report tests."""

from __future__ import annotations

import pytest

from scripts.benchmark.rl.locomotion_ppo import (
    TASK_DIR,
    _load_config,
    _summarize,
    _write_report,
)
from scripts.benchmark.rl.resource_usage import ProcessResources, _process_gpu_mib


def _result(status: str = "passed") -> dict:
    def row(warming: bool, rollout: float, update: float) -> dict:
        return {
            "warmup": warming,
            **{
                phase: {
                    "wall_s": duration,
                    "cpu_delta_mb": 1,
                    "gpu_delta_mb": 2,
                    "peak_gpu_mb": 3,
                }
                for phase, duration in (("rollout", rollout), ("ppo_update", update))
            },
        }

    return {
        "backend": "default",
        "status": status,
        "num_envs": 32,
        "steps": 24,
        "warmup": 1,
        "measured_updates": 2,
        "create_runtime": row(False, 1000, 0)["rollout"],
        "updates": [row(True, 100, 100), row(False, 1, 1), row(False, 3, 1)],
    }


def test_throughput_uses_total_time_and_excludes_warmup() -> None:
    summary = _summarize(_result())
    assert summary["ppo_sps"] == pytest.approx(32 * 24 * 2 / 6)
    assert summary["rollout_sps"] == pytest.approx(32 * 24 * 2 / 4)
    assert summary["rollout_ms"] == 2000
    assert summary["update_ms"] == 1000
    assert summary["ppo_sps_cv_pct"] == pytest.approx(100 / 3)


@pytest.mark.parametrize("second_vram", [1200, None])
def test_report_separates_process_memory_from_allocator(tmp_path, second_vram) -> None:
    result = _result()
    result["resources"] = {
        "samples": [
            {
                "cpu_rss_mib": 50,
                "cpu_rss_lifetime_peak_mib": 80,
                "gpu_process_mib": 1000,
            },
            {
                "cpu_rss_mib": 70,
                "cpu_rss_lifetime_peak_mib": 65,
                "gpu_process_mib": second_vram,
            },
        ]
    }
    summary = _summarize(result)
    assert summary["cpu_rss_peak_mib"] == 80
    assert summary["gpu_process_sampled_peak_mib"] == second_vram
    _write_report(tmp_path, result)
    report = (tmp_path / "report.md").read_text()
    assert "Process RAM (RSS peak): 80.000 MiB" in report
    assert (
        f"process VRAM (sampled peak): {'n/a' if second_vram is None else '1200.000'} MiB"
        in report
    )
    assert "| create_runtime | 1000000.000 |" in report


@pytest.mark.parametrize("backend", ["default", "newton"])
def test_config_keeps_official_policy_and_scales_minibatches(backend: str) -> None:
    config = _load_config(TASK_DIR, backend, 32, 24)
    assert config["trainer"]["num_envs"] == 32
    assert config["trainer"]["buffer_size"] == 24
    assert config["algorithm"]["cfg"]["batch_size"] == 192
    assert config["policy"]["obs_groups"] == {"actor": ["policy"], "critic": ["critic"]}
    assert config["policy"]["actor_obs_normalization"]
    assert config["policy"]["critic_obs_normalization"]


@pytest.mark.parametrize("envs,steps", [(1, 1), (3, 5)])
def test_config_rejects_incomplete_minibatches(envs: int, steps: int) -> None:
    with pytest.raises(ValueError, match="divisible"):
        _load_config(TASK_DIR, "default", envs, steps)


@pytest.mark.parametrize("status", ["passed", "failed"])
def test_report_does_not_rank_training_as_task_success(tmp_path, status: str) -> None:
    result = _result(status)
    _write_report(tmp_path, result)
    report = (tmp_path / "report.md").read_text()
    assert sum(line.startswith("| --- |") for line in report.splitlines()) == 3
    assert f"Dexsim Default CUDA | {status} | n/a |" in report
    assert "EmbodiChain PPO | Dexsim Default CUDA | n/a |" in report
    if status == "failed":
        assert _summarize(result) == {}
        assert "| rollout | n/a | n/a | n/a | n/a |" in report


def test_memory_sample_matches_device_and_process() -> None:
    output = "11, 100, GPU-a\n11, 900, GPU-b\n12, 700, GPU-a\n"
    assert _process_gpu_mib(output, {11}, "a") == 100
    assert _process_gpu_mib(output, {13}, "a") is None
    assert _process_gpu_mib("11, N/A, GPU-a", {11}, "a") is None


def test_resource_output_omits_process_and_device_identifiers() -> None:
    result = ProcessResources("local-device-id").result()
    assert "gpu_uuid" not in result
    assert "pid_candidates" not in result
    assert "local-device-id" not in str(result)
