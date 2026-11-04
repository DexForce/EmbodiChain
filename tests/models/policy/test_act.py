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

"""Portable upstream policies, preprocessing, reset and control contracts."""

from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from embodichain_models.policy.act import load_act_policy, policy_fingerprint

pytestmark = pytest.mark.skipif(
    version("lerobot") != "0.6.1",
    reason="Official ACT integration requires LeRobot 0.6.1",
)


@pytest.fixture
def artifact(act_data, tmp_path: Path):
    from embodichain_models.policy.act import build_act_train_config
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    root, schema = act_data
    cfg = build_act_train_config(
        root,
        tmp_path / "run",
        schema=schema,
        image_key="observation.images.camera",
        device="cpu",
        smoke=True,
        num_workers=0,
    )
    stats = {
        "observation.state": {"mean": torch.zeros(2), "std": torch.ones(2)},
        "observation.images.camera": {
            "mean": torch.full((3, 1, 1), 0.5),
            "std": torch.full((3, 1, 1), 0.25),
        },
        "action": {"mean": torch.tensor([1.0, 2.0]), "std": torch.tensor([2.0, 3.0])},
    }
    policy = ACTPolicy(cfg.policy).eval()
    pre, post = make_act_pre_post_processors(cfg.policy, stats)
    output = tmp_path / "policy"
    policy.save_pretrained(output)
    pre.save_pretrained(output)
    post.save_pretrained(output)
    manifest = {
        "recipe": "lerobot-act-v1",
        "schema": schema,
        "image_key": "observation.images.camera",
        "backbone_initialization": None,
        "policy_sha256": policy_fingerprint(output),
    }
    (output / "embodichain_act.json").write_text(json.dumps(manifest))
    return output, policy, pre, post


def test_import_does_not_initialize_simulation():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from embodichain_models.policy.act import load_act_policy; import sys; assert 'dexsim' not in sys.modules",
        ],
        check=True,
    )


def test_native_reload_matches_upstream_processors_and_reset(artifact):
    root, policy, pre, post = artifact
    rgb = torch.full((1, 3, 32, 32), 128, dtype=torch.uint8)
    state = torch.tensor([[0.2, 0.4]])
    with torch.inference_mode():
        policy.reset()
        expected = post(
            policy.select_action(
                pre(
                    {
                        "observation.images.camera": rgb.float() / 255,
                        "observation.state": state,
                    }
                )
            )
        )
    loaded = load_act_policy(root)
    torch.testing.assert_close(loaded.select_action(rgb, state), expected)
    loaded.select_action(rgb, state + 1)
    loaded.reset()
    torch.testing.assert_close(loaded.select_action(rgb, state), expected)
    assert loaded.select_action(rgb, state).device.type == "cpu"


def test_artifact_mutation_and_legacy_checkpoints_are_rejected(
    artifact, tmp_path: Path
):
    root, *_ = artifact
    with (root / "policy_preprocessor.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="fingerprint"):
        load_act_policy(root)
    legacy = tmp_path / "legacy.pt"
    legacy.write_bytes(b"legacy")
    with pytest.raises(ValueError, match="Historical"):
        load_act_policy(legacy)


def test_validation_uses_frozen_stats_and_valid_element_weighting(artifact, act_data):
    root, *_ = artifact
    data, _ = act_data
    loaded = load_act_policy(root)
    with torch.no_grad():
        loaded.policy.model.action_head.weight.zero_()
        loaded.policy.model.action_head.bias.zero_()
    report = loaded.evaluate_dataset(data, batch_size=2)
    # Two length-three episodes yield 12 valid future frames, each with two
    # action elements. Frozen mean=[1,2], std=[2,3] gives total absolute error 10.
    assert report["valid_action_elements"] == 24
    assert report["action_l1"] == pytest.approx(10 / 24, abs=1e-6)


def test_portable_policy_manifest_prevents_train_and_selection_seed_overlap(artifact):
    from examples.learning.act._environment import task_identity
    from examples.learning.act.evaluate import _validate_policy_identity

    root, *_ = artifact
    manifest = root / "embodichain_act.json"
    metadata = json.loads(manifest.read_text())
    metadata.update(
        task_identity=task_identity(),
        scene_seeds={"train": [100000], "validation": [200000]},
        selection={"panel": "development", "seeds": [300000]},
    )
    manifest.write_text(json.dumps(metadata))
    loaded = load_act_policy(root)
    # No evaluation.json accompanies this directory: isolation must travel with it.
    for seed in [100000, 200000, 300000]:
        with pytest.raises(ValueError, match="overlap"):
            _validate_policy_identity(loaded.metadata, root, panel="test", seeds=[seed])
    _validate_policy_identity(loaded.metadata, root, panel="test", seeds=[400000])
    _validate_policy_identity(loaded.metadata, root, panel="overfit", seeds=[100000])
