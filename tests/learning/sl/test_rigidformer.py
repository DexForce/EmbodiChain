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

"""Compare the SL adapter against the independent optional upstream model."""

from __future__ import annotations

from copy import deepcopy
import subprocess
import sys

import pytest
import torch

from embodichain.learning.models.rigidformer import RigidFormerCfg, RigidFormerModel
from embodichain.learning.sl import SupervisedLearner
from embodichain.learning.sl.rigidformer import (
    RigidFormerObjective,
    build_rigidformer_trainer,
    rigidformer_batch,
)


@pytest.fixture
def geometry():
    generator = torch.Generator().manual_seed(19)
    previous = torch.randn(2, 2, 32, 3, generator=generator)
    return dict(
        object_pos=previous + 0.01,
        object_pos_prev=previous,
        object_first_frame_pos=previous.clone(),
        object_pos_next=previous + 0.025,
        vertex_properties=torch.ones(2, 2, 3),
        delta_times=torch.full((2,), 0.1),
        anchor_indices=torch.tensor([0, 8, 16, 24]).expand(2, 2, 4),
    )


def test_rigidformer_import_is_lazy_without_optional_model_or_simulation():
    code = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'rigidformer', 'dexsim', 'newton', 'warp'}:
            raise ImportError('blocked ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from embodichain.learning.models.rigidformer import RigidFormerModel
from embodichain.learning.sl.rigidformer import RigidFormerObjective
from embodichain.data_pipeline.datasets.rigid_body import RigidBodySequenceSource
try:
    RigidFormerModel()
except ImportError as exc:
    assert 'optional rigidformer dependency' in str(exc)
else:
    raise AssertionError('Missing optional dependency must be explicit')
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_rigidformer_objective_separates_targets_and_preserves_reference_update(
    geometry,
):
    upstream = pytest.importorskip("rigidformer")
    torch.set_num_threads(2)
    cfg = RigidFormerCfg(
        dim=16,
        dim_head=6,
        heads=2,
        num_register_tokens=2,
        object_self_attn_depth=1,
        anchor_cross_attn_depth=2,
        object_hidden_layers=(0, 1),
    )
    candidate = RigidFormerModel(cfg)
    reference_cfg = cfg.to_dict()
    reference_cfg.pop("max_pairwise_bytes")
    reference = upstream.Rigidformer(**reference_cfg)
    reference.load_state_dict(deepcopy(candidate.network.state_dict()))
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=3e-4)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=3e-4)
    batch = rigidformer_batch(geometry, metadata={"source_id": "test"})
    assert "object_pos_next" not in batch.inputs
    assert batch.targets is geometry["object_pos_next"]
    reference_optimizer.zero_grad(set_to_none=True)
    state = torch.get_rng_state()
    loss, _ = reference(**geometry)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    reference_optimizer.step()
    torch.set_rng_state(state)
    metrics = SupervisedLearner(
        candidate, RigidFormerObjective(), optimizer, max_grad_norm=1.0
    ).update(batch)
    assert metrics["loss"] == float(loss)
    assert metrics["weight"] == 4.0  # Two scenes, each with two valid objects.
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(
            candidate.network.state_dict()[name], expected, atol=0, rtol=0
        )


@pytest.mark.parametrize("object_lens", [None, (1, 2)])
def test_rigidformer_split_scene_weighted_loss_and_gradients_match_upstream_batch(
    geometry, object_lens
):
    pytest.importorskip("rigidformer")
    torch.set_num_threads(2)
    torch.manual_seed(31)
    candidate = RigidFormerModel(
        RigidFormerCfg(
            dim=16,
            dim_head=6,
            heads=2,
            num_register_tokens=2,
            object_self_attn_depth=1,
            anchor_cross_attn_depth=2,
            object_hidden_layers=(0, 1),
        )
    ).eval()
    # Fixed valid anchors remove FPS randomness; the scenes have different
    # target accelerations so treating their local means equally is observable.
    geometry["object_pos_next"][0] += torch.tensor([0.04, -0.03, 0.02])
    geometry["object_pos_next"][1] += torch.tensor([-0.02, 0.04, -0.01])
    if object_lens is not None:
        geometry["object_lens"] = torch.tensor(object_lens)
    reference = deepcopy(candidate.network)
    expected_loss, _ = reference(**geometry)
    expected_loss.backward()
    expected_gradients = {
        name: parameter.grad.clone()
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }
    objective = RigidFormerObjective()
    results = [
        objective.compute_loss(
            candidate,
            rigidformer_batch(
                {key: value[row : row + 1] for key, value in geometry.items()}
            ),
        )
        for row in range(2)
    ]
    assert [result.weight for result in results] == (
        [2.0, 2.0] if object_lens is None else [1.0, 2.0]
    )
    actual_loss = sum(result.loss * result.weight for result in results) / sum(
        result.weight for result in results
    )
    torch.testing.assert_close(actual_loss, expected_loss, atol=1e-5, rtol=1e-4)
    actual_loss.backward()
    for name, parameter in candidate.network.named_parameters():
        if name in expected_gradients:
            torch.testing.assert_close(
                parameter.grad, expected_gradients[name], atol=1e-5, rtol=1e-4
            )


def test_rigidformer_reference_geometry_and_distance_budget_are_explicit(geometry):
    pytest.importorskip("rigidformer")
    model = RigidFormerModel(RigidFormerCfg(max_pairwise_bytes=16))
    with pytest.raises(ValueError, match="Pairwise distances"):
        model(**geometry)
    del geometry["object_first_frame_pos"]
    with pytest.raises(ValueError, match="Missing geometry"):
        rigidformer_batch(geometry)


def test_rigidformer_rollout_matches_reference_and_retains_rigidity(geometry):
    upstream = pytest.importorskip("rigidformer")
    torch.set_num_threads(2)
    model = RigidFormerModel(
        RigidFormerCfg(
            dim=16,
            dim_head=6,
            heads=2,
            object_self_attn_depth=1,
            anchor_cross_attn_depth=2,
            object_hidden_layers=(0, 1),
        )
    )
    initial = [geometry["object_pos_prev"], geometry["object_pos"]]
    kwargs = dict(
        object_positions=initial,
        vertex_properties=geometry["vertex_properties"],
        delta_times=geometry["delta_times"],
        anchor_indices=geometry["anchor_indices"],
        num_steps=3,
    )
    with torch.no_grad():
        expected = upstream.RigidformerRolloutWrapper(model.network)(**kwargs)
        actual = model.rollout(**kwargs)
    assert len(actual) == 5
    torch.testing.assert_close(
        torch.stack(actual), torch.stack(expected), atol=0, rtol=0
    )
    # Rigid projection preserves pairwise distances within each object.
    reference_distances = torch.cdist(
        initial[0], initial[0], compute_mode="donot_use_mm_for_euclid_dist"
    )
    torch.testing.assert_close(
        torch.cdist(
            actual[-1], actual[-1], compute_mode="donot_use_mm_for_euclid_dist"
        ),
        reference_distances,
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize("points", [48, 128])
def test_rigidformer_unequal_lengths_ignore_finite_padding_in_objective_and_rollout(
    points,
):
    pytest.importorskip("rigidformer")
    torch.set_num_threads(2)
    torch.manual_seed(31)
    model = RigidFormerModel(
        RigidFormerCfg(
            dim=16,
            dim_head=6,
            heads=2,
            object_self_attn_depth=1,
            anchor_cross_attn_depth=2,
            object_hidden_layers=(0, 1),
        )
    ).eval()
    previous = torch.randn(2, 3, points, 3) * 0.1 + torch.tensor([0.0, 0.0, 0.4])
    values = dict(
        object_pos_prev=previous,
        object_pos=previous + 0.01,
        object_first_frame_pos=previous.clone(),
        object_pos_next=previous + 0.025,
        vertex_properties=torch.randn(2, 3, points, 3),
        delta_times=torch.full((2,), 0.1),
        object_lens=torch.tensor([2, 3]),
        object_point_lens=torch.tensor([[16, 32, points], [32, points, 16]]),
    )
    valid = (torch.arange(3)[None, :, None] < values["object_lens"][:, None, None]) & (
        torch.arange(points)[None, None, :] < values["object_point_lens"][..., None]
    )
    changed = {key: value.clone() for key, value in values.items()}
    for key in (
        "object_pos_prev",
        "object_pos",
        "object_first_frame_pos",
        "object_pos_next",
        "vertex_properties",
    ):
        changed[key][~valid] = torch.randn_like(changed[key][~valid]) * 3
    rng = torch.get_rng_state()
    objective = RigidFormerObjective()
    expected = objective.compute_loss(model, rigidformer_batch(values))
    assert expected.weight == 5.0  # Valid object counts are two and three.
    expected.loss.backward()
    gradients = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(rng)
    actual = objective.compute_loss(model, rigidformer_batch(changed))
    assert actual.weight == expected.weight
    actual.loss.backward()
    torch.testing.assert_close(actual.loss, expected.loss, atol=0, rtol=0)
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], atol=0, rtol=0)
    trajectories = []
    for inputs in (values, changed):
        torch.set_rng_state(rng)
        with torch.no_grad():
            trajectories.append(
                model.rollout(
                    [inputs["object_pos_prev"], inputs["object_pos"]],
                    vertex_properties=inputs["vertex_properties"],
                    delta_times=inputs["delta_times"],
                    object_lens=inputs["object_lens"],
                    object_point_lens=inputs["object_point_lens"],
                    num_steps=3,
                )
            )
    torch.testing.assert_close(
        torch.stack(trajectories[0])[:, valid],
        torch.stack(trajectories[1])[:, valid],
        atol=0,
        rtol=0,
    )


def test_rigidformer_rejects_unsupported_point_masking_and_invalid_anchors(geometry):
    pytest.importorskip("rigidformer")
    model = RigidFormerModel()
    values = {
        key: value.repeat_interleave(5, dim=2) if value.ndim == 4 else value
        for key, value in geometry.items()
    }
    values["object_point_lens"] = torch.tensor([[16, 160], [64, 160]])
    with pytest.raises(ValueError, match="PointNet FPS/KNN leaks finite padding"):
        RigidFormerObjective().compute_loss(model, rigidformer_batch(values))
    with pytest.raises(ValueError, match="PointNet FPS/KNN leaks finite padding"):
        model.rollout(
            [values["object_pos_prev"], values["object_pos"]],
            vertex_properties=values["vertex_properties"],
            delta_times=values["delta_times"],
            object_point_lens=values["object_point_lens"],
            num_steps=1,
        )
    geometry["object_point_lens"] = torch.tensor([[16, 32], [32, 16]])
    with pytest.raises(ValueError, match="anchor_indices must index valid points"):
        model(**geometry)


def test_rigidformer_offline_recipe_restores_the_shared_training_boundary(tmp_path):
    pytest.importorskip("rigidformer")
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(9)
    first = torch.randn(2, 1, 2, 32, 3, generator=generator)
    positions = first + torch.arange(6).reshape(1, 6, 1, 1, 1) * 0.01
    path = tmp_path / "geometry.pt"
    torch.save(
        {"positions": positions, "properties": torch.ones(2, 2, 3), "step_dt": 0.1},
        path,
    )
    config = {
        "data": {"path": str(path)},
        "model": {
            "dim": 16,
            "dim_head": 6,
            "heads": 2,
            "object_self_attn_depth": 1,
            "anchor_cross_attn_depth": 2,
            "object_hidden_layers": (0, 1),
        },
        "trainer": {
            "total_updates": 2,
            "batch_size": 2,
            "output_dir": str(tmp_path / "run"),
        },
    }
    torch.manual_seed(31)
    uninterrupted = build_rigidformer_trainer(config)
    uninterrupted.fit()
    expected = deepcopy(uninterrupted.learner.model.state_dict())
    torch.manual_seed(31)
    config["trainer"]["output_dir"] = str(tmp_path / "interrupted")
    interrupted = build_rigidformer_trainer(config)
    result = interrupted.fit(stop_after=1)
    config["trainer"]["output_dir"] = str(tmp_path / "resumed")
    resumed = build_rigidformer_trainer(config)
    resumed.resume(result["checkpoint"])
    resumed.fit()
    for name, value in expected.items():
        torch.testing.assert_close(
            resumed.learner.model.state_dict()[name], value, atol=0, rtol=0
        )
