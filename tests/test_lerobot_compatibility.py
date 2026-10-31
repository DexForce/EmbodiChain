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

"""Qualify the Python-selected SDK stack and its official visual ACT policy."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import sys

from packaging.version import Version
import torch


def test_python_selects_supported_lerobot_stack() -> None:
    expected = "0.4.4" if sys.version_info < (3, 12) else "0.6.1"
    assert version("lerobot") == expected
    assert Version(version("torch")).release[:2] == (2, 7)
    assert Version(version("torchvision")).release[:2] == (0, 22)
    assert Version(version("torchcodec")).release[:2] == (0, 5)
    hub = Version(version("huggingface-hub"))
    if expected == "0.4.4":
        assert Version("0.35.3") <= hub < Version("0.36")
    else:
        assert Version("1.6") <= hub < Version("2")


def test_official_visual_act_update_reload_and_reset(tmp_path: Path) -> None:
    """Real policy/processor APIs work with no network, simulator or GPU."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = ACTConfig(
            device="cpu",
            input_features={
                "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
                "observation.images.camera": PolicyFeature(
                    FeatureType.VISUAL, (3, 32, 32)
                ),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
            pretrained_backbone_weights=None,
            chunk_size=2,
            n_action_steps=1,
            temporal_ensemble_coeff=0.01,
            dim_model=32,
            n_heads=4,
            dim_feedforward=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
            latent_dim=4,
            n_vae_encoder_layers=1,
            dropout=0.0,
        )
        stats = {
            "observation.state": {"mean": torch.zeros(2), "std": torch.ones(2)},
            "observation.images.camera": {
                "mean": torch.zeros(3, 1, 1),
                "std": torch.ones(3, 1, 1),
            },
            "action": {
                "mean": torch.tensor([0.3, 0.4]),
                "std": torch.tensor([2.0, 3.0]),
            },
        }
        preprocess, postprocess = make_act_pre_post_processors(config, stats)
        batch = preprocess(
            {
                "observation.state": torch.zeros(2, 2),
                "observation.images.camera": torch.zeros(2, 3, 32, 32),
                "action": torch.ones(2, 2, 2),
                "action_is_pad": torch.tensor([[False, False], [False, True]]),
            }
        )
        torch.testing.assert_close(
            postprocess(torch.zeros(2, 2)), torch.tensor([[0.3, 0.4], [0.3, 0.4]])
        )
        policy = ACTPolicy(config)
        optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)
        before = policy.model.action_head.weight.detach().clone()
        loss, metrics = policy(batch)
        assert torch.isfinite(loss)
        assert "l1_loss" in metrics
        loss.backward()
        assert torch.isfinite(policy.model.action_head.weight.grad).all()
        optimizer.step()
        assert not torch.equal(before, policy.model.action_head.weight)

        observation = {
            key: value for key, value in batch.items() if key.startswith("observation.")
        }
        policy.eval()
        expected = policy.select_action(observation)
        policy.save_pretrained(tmp_path / "policy")
        restored = ACTPolicy.from_pretrained(tmp_path / "policy")
        torch.testing.assert_close(restored.select_action(observation), expected)
        restored.select_action({**observation, "observation.state": torch.ones(2, 2)})
        restored.reset()
        torch.testing.assert_close(restored.select_action(observation), expected)
    finally:
        torch.set_num_threads(previous_threads)
