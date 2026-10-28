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

"""Behavioral checks for LeWM temporal, training and export contracts."""

from __future__ import annotations

import json
import types

import h5py
import numpy as np
import pytest
import torch
from torchvision.transforms import v2

from embodichain.learning.sl import SupervisedBatch
from embodichain_models.dynamics.lewm import (
    LeWM,
    LeWMCfg,
    LeWMCEMCfg,
    LeWMCEMPlanner,
    LeWMObjective,
    LeWMSequenceSource,
    build_lewm_trainer,
    export_lewm,
    load_lewm,
    preprocess_rgb,
)


def tiny_cfg() -> LeWMCfg:
    return LeWMCfg(
        action_dim=2,
        image_size=28,
        patch_size=14,
        embed_dim=24,
        history_size=3,
        vision_depth=1,
        vision_heads=3,
        predictor_depth=1,
        predictor_heads=2,
        predictor_head_dim=4,
        mlp_dim=48,
        dropout=0.0,
    )


def write_corpus(path, split: str, seed: int) -> None:
    with h5py.File(path, "w") as file:
        file.attrs["format"] = "embodichain.lewm.trajectory.v1"
        file.attrs["split"] = split
        file.attrs["metadata"] = json.dumps(
            {"action_dim": 2, "action_semantics": "relative", "step_dt": 0.04}
        )
        root = file.create_group("episodes")
        for episode in range(2):
            entry = root.create_group(str(episode))
            entry.attrs["seed"] = seed + episode
            pixels = np.stack(
                [np.full((28, 28, 3), 100 * episode + t, np.uint8) for t in range(6)]
            )
            entry.create_dataset("pixels", data=pixels)
            entry.create_dataset(
                "actions", data=np.arange(10, dtype=np.float32).reshape(5, 2)
            )


def test_preprocessing_matches_reference_transform() -> None:
    rng = torch.Generator().manual_seed(5)
    image = torch.randint(0, 256, (2, 27, 33, 4), dtype=torch.uint8, generator=rng)
    reference = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            v2.Resize((28, 28)),
        ]
    )
    expected = torch.stack([reference(x[..., :3].numpy()) for x in image])
    torch.testing.assert_close(
        preprocess_rgb(image, 28), expected, atol=1e-6, rtol=1e-6
    )


def test_rollout_pairs_the_current_frame_with_its_outgoing_action() -> None:
    model = LeWM(tiny_cfg()).eval()

    def predict(self, embeddings, actions):
        addition = torch.nn.functional.pad(actions.cumsum(1), (0, 22))
        return embeddings + addition

    model.predict = types.MethodType(predict, model)
    history = torch.zeros(1, 3, 24)
    past = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
    candidates = torch.tensor([[[[3.0, 0.0], [4.0, 0.0]]]])
    result = model.rollout(history, past, candidates)
    torch.testing.assert_close(result[0, 0, :, 0], torch.tensor([6.0, 15.0]))
    with pytest.raises(ValueError, match="H-1"):
        model.rollout(history, past[:, :1], candidates)


def test_target_encoder_keeps_gradient() -> None:
    torch.manual_seed(4)
    model = LeWM(tiny_cfg()).train()
    captured = []

    def capture(_module, _inputs, output):
        output.retain_grad()
        captured.append(output)

    hook = model.projector.register_forward_hook(capture)
    pixels = torch.randint(0, 255, (2, 4, 28, 28, 3), dtype=torch.uint8)
    result = LeWMObjective(sigreg_weight=0, num_proj=4).compute_loss(
        model,
        SupervisedBatch({"pixels": pixels, "actions": torch.randn(2, 3, 2)}, None),
    )
    result.loss.backward()
    hook.remove()
    assert captured[0].grad[[3, 7]].abs().sum() > 0


def test_windows_stay_inside_episodes_and_statistics_are_train_only(tmp_path) -> None:
    path = tmp_path / "train.h5"
    write_corpus(path, "train", 1)
    source = LeWMSequenceSource(path, "train", 3, 1)
    assert len(source) == 6
    for batch in source.iter_batches(2):
        values = batch.inputs["pixels"][:, :, 0, 0, 0].to(torch.int16)
        assert torch.all(values[:, 1:] - values[:, :-1] == 1)
        assert batch.inputs["actions"].shape[1] == 3
    mean, std = source.action_statistics()
    torch.testing.assert_close(mean, torch.tensor([4.0, 5.0]))
    assert torch.all(std > 0)
    with pytest.raises(ValueError, match="split"):
        LeWMSequenceSource(path, "validation")


def test_nonfinite_actions_and_cross_split_episode_leakage_are_rejected(
    tmp_path,
) -> None:
    train, validation = tmp_path / "train.h5", tmp_path / "validation.h5"
    write_corpus(train, "train", 10)
    write_corpus(validation, "validation", 10)
    config = {
        "model": tiny_cfg().to_dict(),
        "data": {"train_path": str(train), "validation_path": str(validation)},
        "trainer": {"total_updates": 1, "batch_size": 2, "device": "cpu"},
    }
    with pytest.raises(ValueError, match="seeds overlap"):
        build_lewm_trainer(config)
    with h5py.File(train, "a") as file:
        file["episodes/0/actions"][0, 0] = np.nan
    source = LeWMSequenceSource(train, "train")
    with pytest.raises(ValueError, match="finite"):
        next(source.iter_batches(1))
    with h5py.File(validation, "a") as file:
        file["episodes/0"].create_dataset("timestamp", data=np.arange(6) * 0.05)
    with pytest.raises(ValueError, match="timestamps"):
        LeWMSequenceSource(validation, "validation")


def test_training_resume_and_export(tmp_path) -> None:
    train, validation = tmp_path / "train.h5", tmp_path / "validation.h5"
    write_corpus(train, "train", 10)
    write_corpus(validation, "validation", 20)

    def config(output):
        return {
            "model": tiny_cfg().to_dict(),
            "data": {"train_path": str(train), "validation_path": str(validation)},
            "objective": {"num_proj": 4},
            "trainer": {
                "total_updates": 2,
                "batch_size": 2,
                "device": "cpu",
                "output_dir": str(output),
                "checkpoint_interval": 0,
                "seed": 3,
            },
        }

    full = build_lewm_trainer(config(tmp_path / "full"))
    full.fit()
    part = build_lewm_trainer(config(tmp_path / "resume"))
    partial = part.fit(stop_after=1)
    resumed = build_lewm_trainer(config(tmp_path / "resume"))
    resumed.resume(partial["checkpoint"])
    resumed.fit()
    for name, value in full.learner.model.state_dict().items():
        torch.testing.assert_close(
            value, resumed.learner.model.state_dict()[name], atol=0, rtol=0
        )
    export_lewm(tmp_path / "export", resumed.learner.model, {"source": "test"})
    loaded, manifest = load_lewm(tmp_path / "export")
    assert manifest["metadata"]["source"] == "test"
    for name, value in resumed.learner.model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], atol=0, rtol=0)


def test_cem_is_bounded_and_candidate_chunking_preserves_result() -> None:
    torch.manual_seed(3)
    model = LeWM(tiny_cfg()).eval()
    for layer in model.predictor.transformer.layers:
        torch.nn.init.normal_(layer.adaLN_modulation[-1].weight, std=0.02)
    history = torch.randn(1, 3, 24)
    past = torch.randn(1, 2, 2)
    goal = torch.randn(1, 24)
    base = dict(horizon=2, num_samples=8, iterations=2, topk=2)
    one = LeWMCEMPlanner(
        model,
        torch.full((2,), -1.0),
        torch.ones(2),
        LeWMCEMCfg(**base, candidate_batch_size=1),
        seed=5,
    )
    all_at_once = LeWMCEMPlanner(
        model,
        torch.full((2,), -1.0),
        torch.ones(2),
        LeWMCEMCfg(**base, candidate_batch_size=8),
        seed=5,
    )
    a, cost = one.plan(history, past, goal)
    b, other_cost = all_at_once.plan(history, past, goal)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(cost, other_cost, atol=1e-5, rtol=1e-5)
    assert torch.all(a >= -1) and torch.all(a <= 1)
