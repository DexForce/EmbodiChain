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

"""Real eight-row conditional layout collection and immutable readback."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import pytest


def _probe(output: Path, strategy: str) -> None:
    import json
    import yaml
    import torch
    from embodichain.lab.expansion import (
        ArtifactEpisodeSink,
        load_expansion_episode,
        run_task_program_expansion,
        report_expansion_distribution,
    )

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    fixture = (
        root
        / "embodichain/extensions/tasks/configs/tasks/manipulation/workspace_pose_pick_place/acceptance/scene_first"
    )
    cfg = yaml.safe_load((fixture / "expansion.yaml").read_text())
    cfg["engine"].update(
        workload_count=1,
        batch_size=8,
        episodes_per_operator=8,
        max_attempts=32,
        nominal_seed=7,
    )
    cfg["scene_sampling"].update(strategy=strategy, max_rollouts=32)
    path = output / "expansion.yaml"
    path.write_text(yaml.safe_dump(cfg))
    result = run_task_program_expansion(
        fixture / "task.yaml", path, output / "collection", seed=11
    )
    execution = json.loads((output / "collection/workload-execution.json").read_text())
    reference = json.loads((output / "collection/reference-scene.json").read_text())
    assert (reference["seed"], reference["sampling_seed"]) == (7, 11)
    assert execution["num_envs"] == 8 and execution["world"]["world_count"] == 1
    group = result["operators"]["object_pose"]
    receipts = ArtifactEpisodeSink(group["directory"]).inspect()
    assert len(receipts) == group["committed_episodes"] == 8
    selected = set()
    for receipt in receipts:
        episode = load_expansion_episode(receipt.storage_id)
        assert episode.validation.accepted
        assert len(episode.timestamps) == len(episode.actions) + 1
        selected.add(episode.metadata["source_env_id"])
    assert selected == set(range(8))
    source = Path(group["directory"]) / "scene-first-checks.json"
    checks = json.loads(source.read_text())["phase_checks"]
    assert checks and all(
        any(c["stage"] == "screening.clock" and c["status"] == "passed" for c in rows)
        for rows in checks.values()
    )
    report = report_expansion_distribution(
        [output / "collection"],
        [output / "collection"],
        output / "distribution",
        entity="cube",
        cell_size_m=0.01,
    )
    assert report["before"] == report["after"] == 8
    assert report["comparison_kind"] == "cumulative"


@pytest.mark.parametrize("strategy", ["scene_first", "robot_first", "bidirectional"])
def test_conditional_layout_uses_one_world_and_eight_real_rows(tmp_path, strategy):
    result = subprocess.run(
        [sys.executable, __file__, str(tmp_path), strategy],
        env={**os.environ, "EMBODICHAIN_SIM_EXIT_PROCESS": "0"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _probe(Path(sys.argv[1]), sys.argv[2])
