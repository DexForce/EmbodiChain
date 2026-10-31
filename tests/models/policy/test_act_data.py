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

"""Real SDK coverage of the visual/action sequence join."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from embodichain.data_pipeline.datasets import LeRobotDataSource
from embodichain_models.policy.act import ACTSequenceSource, build_act_trainer


def test_rgb_and_action_windows_align_and_resume_checks_content(tmp_path: Path):
    sdk = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    root = tmp_path / "dataset"
    writer = sdk.LeRobotDataset.create(
        repo_id="local/act-test",
        root=root,
        fps=30,
        use_videos=False,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["episode", "frame"],
            },
            "observation.images.camera": {
                "dtype": "image",
                "shape": (8, 8, 3),
                "names": ["height", "width", "channel"],
            },
            "action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["episode", "frame"],
            },
        },
    )
    for episode, length in enumerate((3, 2)):
        for frame in range(length):
            writer.add_frame(
                {
                    "observation.state": np.array([episode, frame], dtype=np.float32),
                    "action": np.array([episode, frame], dtype=np.float32),
                    "observation.images.camera": np.full(
                        (8, 8, 3), episode * 10 + frame, dtype=np.uint8
                    ),
                    "task": "push",
                }
            )
        writer.save_episode()
    writer.finalize()
    schema = {
        "observation": {"names": ["episode", "frame"], "dtype": "float32"},
        "action": {
            "names": ["episode", "frame"],
            "semantics": "absolute",
            "step_dt": 1 / 30,
        },
    }
    numeric = LeRobotDataSource(
        root,
        schema=schema,
        source_id="act-test",
        repo_id="local/act-test",
        memory_limit_bytes=1024 * 1024,
        image_keys=["observation.images.camera"],
        image_cache_bytes=192,
    )
    source = ACTSequenceSource(
        numeric, horizon=4, image_key="observation.images.camera"
    )
    batch = source.sample(40, generator=torch.Generator().manual_seed(32))
    state, actions, valid, rgb = (
        batch.inputs["state"],
        batch.targets,
        batch.masks["valid"],
        batch.inputs["rgb"],
    )
    assert torch.equal(
        rgb[:, 0, 0, 0], (state[:, 0] * 10 + state[:, 1]).to(torch.uint8)
    )
    torch.testing.assert_close(actions[:, 0], state)
    assert torch.all(
        actions[..., 0][valid] == state[:, None, 0].expand_as(valid)[valid]
    )
    assert torch.all(
        actions[..., 1][valid] == (state[:, None, 1] + torch.arange(4))[valid]
    )
    assert (~valid).any()
    saved = rgb.clone()
    source.sample(8, generator=torch.Generator().manual_seed(2)).inputs["rgb"].zero_()
    torch.testing.assert_close(rgb, saved)
    source.validate_resume()

    recipe = {
        "model": {
            "state_dim": 2,
            "action_dim": 2,
            "horizon": 4,
            "hidden_dim": 32,
            "heads": 4,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "feedforward_dim": 64,
            "latent_dim": 4,
            "dropout": 0.0,
            "pretrained_backbone": False,
        },
        "data": {
            "train_path": str(root),
            "schema": schema,
            "image_key": "observation.images.camera",
        },
        "trainer": {
            "total_updates": 2,
            "batch_size": 2,
            "device": "cpu",
            "seed": 12,
            "output_dir": str(tmp_path / "run"),
            "checkpoint_interval": 0,
        },
    }
    first = build_act_trainer(recipe)
    torch.testing.assert_close(first.learner.model.state_mean, torch.tensor([0.4, 0.8]))
    checkpoint = first.fit(stop_after=1)["checkpoint"]
    resumed = build_act_trainer(recipe)
    resumed.resume(checkpoint)
    assert resumed.fit()["num_updates"] == 2

    with (root / "meta" / "info.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="content changed"):
        source.validate_resume()
