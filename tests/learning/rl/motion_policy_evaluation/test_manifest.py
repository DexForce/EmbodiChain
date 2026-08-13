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

from embodichain.learning.rl.motion_policy_evaluation import (
    RunManifest,
    write_run_manifest,
)


def test_manifest_snapshots_configs_and_selects_latest_fallback(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "policy.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    train = tmp_path / "train.yaml"
    train.write_text("trainer: {}\n", encoding="utf-8")
    gym = tmp_path / "gym.yaml"
    gym.write_text("id: Example\n", encoding="utf-8")

    write_run_manifest(
        run,
        train_config=train,
        gym_config=gym,
        latest_checkpoint=checkpoint,
        motion_profile="example-motion",
    )
    manifest = RunManifest.load(run)
    selected, path = manifest.select_checkpoint("best")

    assert manifest.motion_profile == "example-motion"
    assert manifest.configs["train"] == run / "configs" / "train.yaml"
    assert manifest.configs["gym"] == run / "configs" / "gym.yaml"
    assert selected == "latest"
    assert path == checkpoint
