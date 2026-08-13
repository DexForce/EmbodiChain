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

from embodichain.learning.rl.motion_policy_evaluation import RunManifest
from embodichain.learning.rl.train import _write_motion_run_manifest


def test_training_summary_writes_minimal_motion_manifest(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "policy.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "train.yaml"
    config.write_text("trainer: {}\n", encoding="utf-8")

    _write_motion_run_manifest(
        run,
        config,
        {"motion_profile": "example-motion"},
        {
            "latest_checkpoint_path": str(checkpoint),
            "best_checkpoint_path": None,
        },
    )

    manifest = RunManifest.load(run)
    assert manifest.motion_profile == "example-motion"
    assert manifest.checkpoints["latest"] == checkpoint
