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

from embodichain.learning.sl import SupervisedBatch, SupervisedLearner
from examples.learning.lewm._finetune import _LatentWindows, _RolloutObjective
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


def write_corpus(path, split: str, seed: int, steps: int = 5) -> None:
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
                [
                    np.full((28, 28, 3), 100 * episode + t, np.uint8)
                    for t in range(steps + 1)
                ]
            )
            entry.create_dataset("pixels", data=pixels)
            entry.create_dataset(
                "actions", data=np.arange(steps * 2, dtype=np.float32).reshape(steps, 2)
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


def test_blocked_windows_keep_every_action_and_raw_episode_boundaries(tmp_path) -> None:
    path = tmp_path / "blocked.h5"
    write_corpus(path, "train", 10, steps=12)
    source = LeWMSequenceSource(path, "train", 2, 2, action_block=3)
    assert len(source) == 8
    assert source.model_action_dim == 6
    assert source.metadata["action_dim"] == 2
    assert source.metadata["action_block"] == 3
    assert source.source_id.endswith(":block=3")
    batch = next(source.iter_batches(8))
    frames = batch.inputs["pixels"][:, :, 0, 0, 0].to(torch.int16)
    torch.testing.assert_close(
        frames[:, 1:] - frames[:, :-1], torch.full((8, 3), 3, dtype=torch.int16)
    )
    assert set(frames[:, 0].tolist()) == {0, 1, 2, 3, 100, 101, 102, 103}
    torch.testing.assert_close(
        batch.inputs["actions"][0], torch.arange(18).float().reshape(3, 6)
    )
    mean, std = source.action_statistics()
    torch.testing.assert_close(mean, torch.tensor([11.0, 12.0] * 3))
    raw = np.tile(np.arange(24, dtype=np.float32).reshape(12, 2), (2, 1))
    torch.testing.assert_close(std, torch.tensor(raw.std(axis=0, ddof=1)).repeat(3))


def test_resident_windows_match_disk_and_own_their_results(tmp_path) -> None:
    path = tmp_path / "cached.h5"
    write_corpus(path, "train", 20, steps=12)
    disk = LeWMSequenceSource(path, "train", 2, 2, action_block=3)
    with pytest.raises(ValueError, match="cache budget"):
        LeWMSequenceSource(
            path, "train", 2, 2, action_block=3, cache_device="cpu", cache_max_bytes=1
        )
    cached = LeWMSequenceSource(
        path, "train", 2, 2, action_block=3, cache_device="cpu", cache_max_bytes=100_000
    )
    assert cached.source_id == disk.source_id
    assert cached.cache_bytes <= 100_000
    left = cached.sample(4, generator=torch.Generator().manual_seed(7))
    right = disk.sample(4, generator=torch.Generator().manual_seed(7))
    for key in left.inputs:
        torch.testing.assert_close(left.inputs[key], right.inputs[key], atol=0, rtol=0)
    left.inputs["pixels"].zero_()
    fresh = cached.sample(4, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(
        fresh.inputs["pixels"], right.inputs["pixels"], atol=0, rtol=0
    )


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


@pytest.mark.parametrize("action_block", [1, 3])
def test_training_resume_and_export(tmp_path, action_block: int) -> None:
    train, validation = tmp_path / "train.h5", tmp_path / "validation.h5"
    write_corpus(train, "train", 10, steps=12)
    write_corpus(validation, "validation", 20, steps=12)

    def config(output):
        return {
            "model": tiny_cfg().to_dict()
            | {"action_dim": 2 * action_block, "action_block": action_block},
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
    cached_config = config(tmp_path / "resume")
    cached_config["data"].update(cache_device="cpu", cache_max_bytes=1_000_000)
    resumed = build_lewm_trainer(cached_config)
    resumed.resume(partial["checkpoint"])
    resumed.fit()
    for name, value in full.learner.model.state_dict().items():
        torch.testing.assert_close(
            value, resumed.learner.model.state_dict()[name], atol=0, rtol=0
        )
    export_lewm(tmp_path / "export", resumed.learner.model, {"source": "test"})
    loaded, manifest = load_lewm(tmp_path / "export")
    assert manifest["metadata"]["source"] == "test"
    assert loaded.cfg.action_block == action_block
    for name, value in resumed.learner.model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize("sampling_distribution", ["bounds", "training", "normalized"])
@pytest.mark.parametrize("action_selection", ["best", "mean"])
def test_cem_is_bounded_and_candidate_chunking_preserves_result(
    sampling_distribution: str, action_selection: str
) -> None:
    torch.manual_seed(3)
    model = LeWM(tiny_cfg()).eval()
    for layer in model.predictor.transformer.layers:
        torch.nn.init.normal_(layer.adaLN_modulation[-1].weight, std=0.02)
    history = torch.randn(1, 3, 24)
    past = torch.randn(1, 2, 2)
    goal = torch.randn(1, 24)
    model.set_action_statistics(torch.tensor([0.2, -0.3]), torch.tensor([0.2, 0.15]))
    base = dict(
        horizon=2,
        num_samples=8,
        iterations=2,
        topk=2,
        sampling_distribution=sampling_distribution,
        action_selection=action_selection,
    )
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
    one.generator.manual_seed(5)
    sequence, sequence_cost = one.plan_sequence(history, past, goal)
    assert sequence.shape == (1, 2, 2)
    torch.testing.assert_close(sequence[:, 0], a)
    torch.testing.assert_close(sequence_cost, cost)
    measured = model.goal_cost(model.rollout(history, past, sequence[:, None]), goal)[
        :, 0
    ]
    torch.testing.assert_close(sequence_cost, measured, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("selection", "expected_action", "expected_cost"),
    [("best", [0.3, -0.2], 0.13), ("mean", [0.35, -0.175], 0.153125)],
)
def test_cem_training_moments_and_selected_sequence_cost(
    monkeypatch, selection: str, expected_action: list[float], expected_cost: float
) -> None:
    model = LeWM(tiny_cfg()).eval()
    model.set_action_statistics(torch.tensor([0.4, -0.2]), torch.tensor([0.1, 0.05]))

    # A one-step additive oracle makes the known raw candidates and their costs
    # independent of learned network weights or action normalization internals.
    def rollout(self, history, past, candidates):
        return torch.nn.functional.pad(candidates, (0, 22))

    model.rollout = types.MethodType(rollout, model)
    noise = torch.tensor([[[[-1.0, 0], [0, 0], [1, 0], [0, 1]]]]).reshape(1, 4, 1, 2)
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: noise.clone())
    planner = LeWMCEMPlanner(
        model,
        -torch.ones(2),
        torch.ones(2),
        LeWMCEMCfg(
            horizon=1,
            num_samples=4,
            iterations=1,
            topk=2,
            sampling_distribution="training",
            action_selection=selection,
        ),
    )
    action, cost = planner.plan(
        torch.zeros(1, 3, 24), torch.zeros(1, 2, 2), torch.zeros(1, 24)
    )
    # The two elite raw actions are (0.3, -0.2) and (0.4, -0.15).
    torch.testing.assert_close(action[0], torch.tensor(expected_action))
    torch.testing.assert_close(cost[0], torch.tensor(expected_cost))


def test_cem_official_normalized_search_preserves_mean_and_zero_variance(
    monkeypatch,
) -> None:
    model = LeWM(tiny_cfg()).eval()
    model.set_action_statistics(torch.tensor([2.0, -0.2]), torch.tensor([0.5, 0.05]))
    sampled = []

    def rollout(self, history, past, candidates):
        sampled.append(candidates.clone())
        return torch.nn.functional.pad(candidates, (0, 22))

    model.rollout = types.MethodType(rollout, model)
    noise = torch.tensor([[[[100.0, 100.0]], [[4.0, 0.0]], [[0.0, 1.0]]]])
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: noise.clone())
    planner = LeWMCEMPlanner(
        model,
        -torch.ones(2),
        torch.ones(2),
        LeWMCEMCfg(
            horizon=1,
            num_samples=3,
            iterations=2,
            topk=1,
            sampling_distribution="normalized",
            action_selection="mean",
            min_std=0.0,
            clip_candidates=False,
            include_mean=True,
        ),
    )
    goal = torch.nn.functional.pad(torch.tensor([[4.0, -0.2]]), (0, 22))
    action, cost = planner.plan(torch.zeros(1, 3, 24), torch.zeros(1, 2, 2), goal)
    # In standardized space the initial candidates are (0,0), (4,0), (0,1).
    # The known optimum is outside [-1,1]. A single elite collapses variance
    # exactly, so the second iteration must contain only that optimum.
    torch.testing.assert_close(
        sampled[0][0, :, 0], torch.tensor([[2.0, -0.2], [4.0, -0.2], [2.0, -0.15]])
    )
    torch.testing.assert_close(sampled[1][0, :, 0], action.expand(3, -1))
    torch.testing.assert_close(action, torch.tensor([[4.0, -0.2]]))
    torch.testing.assert_close(cost, torch.zeros(1))


@pytest.mark.parametrize("correction, spread", [(0, 1.0), (1, 2**0.5)])
def test_cem_elite_deviation_controls_the_next_population(
    monkeypatch, correction: int, spread: float
) -> None:
    model = LeWM(tiny_cfg()).eval()
    sampled = []

    def rollout(self, history, past, candidates):
        sampled.append(candidates.clone())
        return torch.nn.functional.pad(candidates, (0, 22))

    model.rollout = types.MethodType(rollout, model)
    noise = iter([torch.tensor([[[[0.0, 0.0]], [[2.0, 0.0]]]]), torch.ones(1, 2, 1, 2)])
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: next(noise))
    planner = LeWMCEMPlanner(
        model,
        -torch.ones(2),
        torch.ones(2),
        LeWMCEMCfg(
            horizon=1,
            num_samples=2,
            iterations=2,
            topk=2,
            min_std=0,
            clip_candidates=False,
            std_correction=correction,
        ),
    )
    planner.plan(torch.zeros(1, 1, 24), torch.zeros(1, 0, 2), torch.zeros(1, 24))
    torch.testing.assert_close(
        sampled[1][0, :, 0], torch.tensor([[1 + spread, 0.0]]).expand(2, -1)
    )


def test_frozen_latent_windows_preserve_episode_and_outgoing_action_alignment() -> None:
    episodes = [
        {
            "embeddings": (torch.arange(6) + offset).float()[:, None],
            "actions": (torch.arange(5) + offset).float()[:, None],
        }
        for offset in (0, 100)
    ]
    source = _LatentWindows({"identity": {}, "episodes": episodes}, 2, 2, "cpu")
    batch = next(source.iter_batches(6))
    frames, actions = batch.inputs["embeddings"], batch.inputs["actions"]
    assert frames.shape == (6, 4, 1)
    torch.testing.assert_close(frames[:, 1:] - frames[:, :-1], torch.ones(6, 3, 1))
    torch.testing.assert_close(actions, frames[:, :-1])
    assert set(frames[:, 0, 0].tolist()) == {0, 1, 2, 100, 101, 102}


def test_predictor_finetuning_keeps_visual_parameters_and_buffers_fixed() -> None:
    torch.manual_seed(4)
    model = LeWM(tiny_cfg())
    model.encoder.requires_grad_(False)
    model.projector.requires_grad_(False)
    frozen = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if name.startswith(("encoder.", "projector."))
    }
    batch = SupervisedBatch(
        inputs={"embeddings": torch.randn(4, 8, 24), "actions": torch.randn(4, 7, 2)},
        targets=None,
    )
    learner = SupervisedLearner(
        model,
        _RolloutObjective(),
        torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3),
    )
    metrics = learner.update(batch)
    assert metrics["grad_norm"] > 0
    for name, expected in frozen.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
    assert all(p.grad is None for p in model.encoder.parameters())


def test_successful_tail_sampling_selects_complete_successful_windows() -> None:
    episodes = [
        {
            "embeddings": (torch.arange(6) + offset).float()[:, None],
            "actions": (torch.arange(5) + offset).float()[:, None],
            "success": offset == 100,
        }
        for offset in (0, 100)
    ]
    source = _LatentWindows(
        {"identity": {}, "episodes": episodes},
        2,
        2,
        "cpu",
        successful_tail_fraction=1.0,
        tail_steps=2,
    )
    batch = source.sample(8, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(
        batch.inputs["embeddings"][:, 0, 0], torch.full((8,), 102.0)
    )
    torch.testing.assert_close(
        batch.inputs["actions"], batch.inputs["embeddings"][:, :-1]
    )
