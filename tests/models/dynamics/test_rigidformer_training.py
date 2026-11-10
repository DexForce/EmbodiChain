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

import hashlib
import json
import os
import numpy as np
import pytest
import torch
from embodichain_models.dynamics.rigidformer.supervised import (
    RigidFormerObjective,
    rigidformer_batch,
)
from ._rigidformer_fixtures import tiny_cfg, geometry_batch
from ._rigidformer_training import make_packed, source, recovery_config, assert_exact


@pytest.fixture
def packed_data(tmp_path):
    return make_packed(tmp_path / "ToyData")


def test_shifted_objective_matches_independent_anchor_huber_oracle():
    v = geometry_batch()

    class Predictions(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = torch.nn.Parameter(torch.tensor(0.02))

        def forward(self, values):
            raw = values["obj_positions"][:, 1:-1] + self.offset
            return {
                "obj_positions_norm": raw,
                "obj_positions_icp_norm": raw + self.offset,
                "obj_accelerations_no_icp_norm_anchor": self.offset.expand(
                    2, 2, 2, 4, 3
                ),
                "obj_accelerations_norm": self.offset.expand_as(raw) * 2,
            }

    model = Predictions()
    batch = rigidformer_batch(v)
    got = RigidFormerObjective().compute_loss(model, batch)

    def huber(e):
        a = e.abs()
        return torch.where(a < 1, 0.5 * e.square(), a - 0.5).sum(-1)

    expected = 0.0
    terms = []
    for b, objects in enumerate((2, 1)):
        for t in range(2):
            for o in range(objects):
                scale = float(v["skip_step_size"][b]) ** 2
                gt = v["obj_positions"][b, t + 2, o, :4]
                prev = v["obj_positions"][b, t, o, :4]
                curr = v["obj_positions"][b, t + 1, o, :4]
                raw = curr + model.offset
                accel = gt - 2 * curr + prev
                terms.append(
                    10 * huber((raw - gt) / scale)
                    + 10 * huber((raw + model.offset - gt) / scale)
                    + huber((model.offset - accel) / scale)
                    + huber((2 * model.offset - accel) / scale)
                )
    expected = torch.cat(terms).mean()
    assert got.weight == 24.0
    torch.testing.assert_close(got.loss, expected)
    got.loss.backward()
    actual = model.offset.grad.clone()
    model.offset.grad = None
    expected.backward()
    torch.testing.assert_close(model.offset.grad, actual)
    # The final target is absent from conditioning.
    assert batch.inputs["obj_positions"].shape[1] == 3


@pytest.mark.parametrize("random_step", [True, False])
def test_packed_geometry_scales_once_and_replays_owned_windows(
    packed_data, random_step
):
    a = source(packed_data, random_step=random_step)
    generator = torch.Generator().manual_seed(999)
    first = a.sample(2, generator=generator)
    a.on_update(None)
    saved = a.state_dict()
    rng = generator.get_state().clone()
    expected = a.sample(2, generator=generator)
    b = source(packed_data, random_step=random_step)
    b.validate_state_dict(saved)
    b.load_state_dict(saved)
    generator.set_state(rng)
    actual = b.sample(2, generator=generator)
    for key, value in expected.inputs.items():
        torch.testing.assert_close(actual.inputs[key], value, rtol=0, atol=0)
    torch.testing.assert_close(expected.targets, actual.targets, rtol=0, atol=0)
    assert (
        first.inputs["obj_positions"].untyped_storage().data_ptr()
        != actual.inputs["obj_positions"].untyped_storage().data_ptr()
    )
    assert float(
        first.inputs["obj_positions"][0, 0, 0, :, 2].max() - 1.0
    ) == pytest.approx(0.07, abs=1e-6)
    assert float(
        first.inputs["obj_positions"][0, 0, 1, :, 2].max() - 1.0
    ) == pytest.approx(0.14, abs=1e-6)


def test_changed_geometry_rejected_before_sampling_recovery(packed_data):
    a = source(packed_data)
    state = a.state_dict()
    p = packed_data / "scene_0" / "frames.npy"
    values = np.load(p)
    values[0, 0, 0] += 1
    np.save(p, values)
    with pytest.raises(ValueError, match="hash differs"):
        a.validate_state_dict(state)


def test_geometry_missing_from_manifest_is_not_a_verified_source(packed_data):
    path = packed_data / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"].pop("scene_0/frames.npy")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not committed by manifest"):
        source(packed_data)


@pytest.mark.parametrize("missing_cache", ["anchor", "fps_order"])
def test_missing_prepared_cache_does_not_modify_source(packed_data, missing_cache):
    from embodichain_models.dynamics.rigidformer.data import MOViSequenceSource

    original = source(packed_data, random_step=False)
    valid = dict(original.config)
    if missing_cache == "anchor":
        config = {**valid, "anchor_k": 2}
        message = "Prepared anchor cache is missing"
    else:
        path = packed_data / "scene_0" / "anchors" / "obj_0_fpsorder_cf.npy"
        path.unlink()
        manifest_path = packed_data / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"].pop(str(path.relative_to(packed_data)))
        manifest_path.write_text(json.dumps(manifest))
        config = dict(valid)
        valid["fps_order"] = False
        message = "Prepared FPS-order cache is missing"
    before = {p.relative_to(packed_data) for p in packed_data.rglob("*")}
    with pytest.raises(ValueError, match=message):
        MOViSequenceSource(
            packed_data,
            packed_data / "splits.json",
            packed_data / "scales.json",
            data_config=config,
        )
    assert {p.relative_to(packed_data) for p in packed_data.rglob("*")} == before
    MOViSequenceSource(
        packed_data,
        packed_data / "splits.json",
        packed_data / "scales.json",
        data_config=valid,
    )


def test_new_manifest_reloads_replaced_packed_geometry(packed_data):
    first = source(packed_data, random_step=False)
    before = first.sample(1, generator=torch.Generator().manual_seed(17))
    scene = packed_data / "scene_0"
    manifest_path = packed_data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("frames.npy", "vis_traj.npy"):
        path = scene / name
        values = np.load(path).copy()
        if name == "frames.npy":
            values[1:, :, 0] += 0.5
        else:
            values[0, 1:, :, 0] += 0.5
        replacement = scene / "replacement.tmp.npy"
        np.save(replacement, values)
        os.replace(replacement, path)
        manifest["files"][str(path.relative_to(packed_data))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    second = source(packed_data, random_step=False)
    after = second.sample(1, generator=torch.Generator().manual_seed(17))
    assert second.source_id != first.source_id
    torch.testing.assert_close(
        after.inputs["obj_positions"][:, :, :, :, 0],
        before.inputs["obj_positions"][:, :, :, :, 0] + 0.5,
    )


def test_new_manifest_reloads_replaced_per_frame_geometry(tmp_path):
    root = make_packed(tmp_path / "ToyData", frames_count=10)
    scene = root / "scene_0"
    frames = np.load(scene / "frames.npy")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for frame, values in enumerate(frames):
        directory = scene / f"frame_{frame:05d}"
        directory.mkdir()
        for obj in range(2):
            path = directory / f"obj_{obj}_pos.npy"
            np.save(path, values[obj * 4 : (obj + 1) * 4])
            manifest["files"][str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    for name in ("frames.npy", "frames_counts.npy"):
        (scene / name).unlink()
        manifest["files"].pop(f"scene_0/{name}")
    manifest_path.write_text(json.dumps(manifest))
    first = source(root, random_step=False)
    before = first.sample(1, generator=torch.Generator().manual_seed(17))
    path = scene / "frame_00002" / "obj_0_pos.npy"
    changed = np.load(path).copy()
    changed[:, 0] += 0.5
    replacement = path.with_suffix(".tmp.npy")
    np.save(replacement, changed)
    os.replace(replacement, path)
    manifest["files"][str(path.relative_to(root))] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    second = source(root, random_step=False)
    after = second.sample(1, generator=torch.Generator().manual_seed(17))
    assert second.source_id != first.source_id
    assert after.inputs["obj_positions"][0, 0, 0, 0, 0].item() == pytest.approx(
        before.inputs["obj_positions"][0, 0, 0, 0, 0].item() + 0.35
    )


def test_fixed_step_and_stride_are_applied(packed_data):
    from embodichain_models.dynamics.rigidformer.data import MOViSequenceSource

    original = source(packed_data, random_step=False)
    updated = MOViSequenceSource(
        packed_data,
        packed_data / "splits.json",
        packed_data / "scales.json",
        data_config={**original.config, "skip_step_size": 5, "stride": 3},
    )
    batch = updated.sample(1, generator=torch.Generator().manual_seed(17))
    assert batch.inputs["skip_step_size"].tolist() == [5.0]
    assert np.diff(batch.metadata["frame_ids"][0]).tolist() == [5] * 7
    assert len(updated.dataset) == 6


def test_small_dataset_pads_every_distributed_rank(tmp_path, monkeypatch):
    root = make_packed(tmp_path / "ToyData", frames_count=10)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    for rank in range(4):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
        dataset = source(root, random_step=False)
        assert dataset.updates_per_epoch(1) == 1
        batch = dataset.sample(1, generator=torch.Generator().manual_seed(17))
        assert batch.metadata["scenes"] == ["scene_0"]


def test_default_object_padding_counts_toward_memory_limit(packed_data):
    from embodichain_models.dynamics.rigidformer.data import MOViSequenceSource

    original = source(packed_data, random_step=False)
    config = dict(original.config)
    config.pop("max_objects_per_frame")
    config["sequence_length"] = 3
    bounded = MOViSequenceSource(
        packed_data,
        packed_data / "splits.json",
        packed_data / "scales.json",
        data_config=config,
        max_batch_bytes=2000,
    )
    with pytest.raises(ValueError, match="max_batch_bytes"):
        bounded.sample(1, generator=torch.Generator().manual_seed(17))


def test_recipe_rejects_anchor_count_mismatched_with_model(packed_data, tmp_path):
    from embodichain_models.dynamics.rigidformer.sl_recipe import (
        build_rigidformer_trainer,
    )

    config = recovery_config(packed_data, tmp_path / "output", "cpu")
    config["data"]["sampling"]["anchor_k"] = 2
    with pytest.raises(ValueError, match="anchor_k must match"):
        build_rigidformer_trainer(config)


def test_centered_predictions_use_centered_position_targets():
    class CenteredPrediction(torch.nn.Module):
        reference_config = {"data": {"center_scene_by_t0": True}}

        def forward(self, values):
            positions = values["obj_positions"]
            center = positions[:, 0].mean(dim=(1, 2))[:, None, None, None]
            predicted = positions[:, 1:-1] - center
            zero = torch.zeros_like(predicted)
            return {
                "obj_positions_norm": predicted,
                "obj_positions_icp_norm": predicted,
                "obj_accelerations_no_icp_norm_anchor": zero,
                "obj_accelerations_norm": zero,
            }

    positions = torch.tensor([1.0, 2.0, 3.0]).view(1, 1, 1, 1, 3)
    positions = positions.expand(1, 3, 2, 1, 3).clone()
    values = {
        "obj_positions": positions,
        "valid_vtx_mask": torch.ones(1, 3, 2, 1, dtype=torch.bool),
        "valid_obj_mask": torch.ones(1, 2, dtype=torch.bool),
        "anchor_idx": torch.zeros(1, 2, 1, dtype=torch.long),
        "skip_step_size": torch.ones(1),
    }
    result = RigidFormerObjective().compute_loss(
        CenteredPrediction(), rigidformer_batch(values)
    )
    assert result.loss.item() == pytest.approx(0.0)


@pytest.mark.gpu
def test_centered_model_loss_is_translation_invariant():
    pytest.importorskip("roma")
    from embodichain_models.dynamics.rigidformer import RigidFormerModel

    cfg = tiny_cfg()
    cfg.architecture["data"]["center_scene_by_t0"] = True
    model = RigidFormerModel(cfg).cuda().eval()
    values = geometry_batch("cuda")
    values["valid_obj_mask"][:] = True
    moved = {key: value.clone() for key, value in values.items()}
    translation = torch.tensor([0.4, -0.6, 1.2], device="cuda")
    moved["obj_positions"] += translation
    moved["ref_obj_positions"] += translation
    objective = RigidFormerObjective()
    with torch.no_grad():
        original = objective.compute_loss(model, rigidformer_batch(values)).loss
        translated = objective.compute_loss(model, rigidformer_batch(moved)).loss
    torch.testing.assert_close(translated, original, rtol=1e-4, atol=1e-5)


def test_export_accepts_legacy_numpy_rng_pickle(tmp_path, monkeypatch):
    from embodichain_models.dynamics.rigidformer.checkpoint import (
        export_rigidformer_checkpoint,
    )

    try:
        reconstruct = np._core.multiarray._reconstruct
    except AttributeError:
        reconstruct = np.core.multiarray._reconstruct
    checkpoint = tmp_path / "training.pt"
    original_module = reconstruct.__module__
    monkeypatch.setattr(reconstruct, "__module__", "numpy.core.multiarray")
    raw, average = torch.tensor([1.0]), torch.tensor([2.0])
    torch.save(
        {
            "rng": {"numpy": np.random.get_state()},
            "signature": {
                "family": "rigidformer-author-v1",
                "reference_config": {},
                "author_revision": "test",
            },
            "learner": {"model": {"network.weight": raw}, "num_updates": 1},
            "training_states": {
                "ema": {"state": {"shadow": {"network.weight": average}}}
            },
        },
        checkpoint,
    )
    monkeypatch.setattr(reconstruct, "__module__", original_module)
    for selection, expected in (("raw", raw), ("ema", average)):
        output = export_rigidformer_checkpoint(
            checkpoint, tmp_path / f"{selection}.pt", weights=selection
        )
        payload = torch.load(output, weights_only=True)
        torch.testing.assert_close(payload["model_state_dict"]["weight"], expected)


@pytest.mark.parametrize("random_sampling", [False, True])
def test_compact_windows_preserve_scene_stride_and_boundaries(random_sampling):
    from pathlib import Path
    from embodichain_models.dynamics.rigidformer._native.data._index import (
        _SceneWindows,
    )

    windows = _SceneWindows(random_sampling=random_sampling)
    windows.add_scene("a", [Path(f"a/frame_{i:05d}") for i in range(7)], 4, 2)
    windows.add_scene("b", [Path(f"b/frame_{i:05d}") for i in range(8)], 4, 2)
    expected = [
        ("a", [0, 1, 2, 3]),
        ("a", [2, 3, 4, 5]),
        ("b", [0, 1, 2, 3]),
        ("b", [2, 3, 4, 5]),
        ("b", [4, 5, 6, 7]),
    ]
    assert len(windows) == len(expected)
    for i, (scene, frames) in enumerate(expected):
        item = windows[i]
        assert item["dataset"] == scene
        if not random_sampling:
            assert item["frame_ids"] == frames
            assert item["frames"] == [f"{scene}/frame_{j:05d}" for j in frames]
    assert windows[-1] == windows[len(expected) - 1]


@pytest.mark.gpu
def test_recipe_resume_crosses_epoch_and_warmup_with_exact_ema(tmp_path):
    pytest.importorskip("roma")
    pytest.importorskip("einops")
    from embodichain_models.dynamics.rigidformer.sl_recipe import (
        build_rigidformer_trainer,
    )
    from embodichain_models.dynamics.rigidformer import (
        export_rigidformer_checkpoint,
        load_rigidformer_checkpoint,
    )

    root = make_packed(tmp_path / "short-scenes", frames_count=13)
    config = recovery_config(root, tmp_path / "full", "cuda")
    config["trainer"]["batch_size"] = 2
    full = build_rigidformer_trainer(config)
    assert full.signature["steps_per_epoch"] == 2
    full.fit()
    config["trainer"]["output_dir"] = str(tmp_path / "split")
    split = build_rigidformer_trainer(config)
    split.fit(stop_after=3)
    config["trainer"]["output_dir"] = str(tmp_path / "resumed")
    resumed = build_rigidformer_trainer(config)
    resumed.resume(tmp_path / "split/last.pt")
    resumed.fit()
    for key in ("learner", "training_states", "sampler", "rng"):
        a = torch.load(tmp_path / "full/last.pt", weights_only=False)[key]
        b = torch.load(tmp_path / "resumed/last.pt", weights_only=False)[key]
        assert_exact(a, b)
    schedule = resumed.training_states["epoch_schedule"]
    assert schedule.scheduler.last_epoch == 3
    for selection in ("raw", "ema"):
        path = export_rigidformer_checkpoint(
            tmp_path / "resumed/last.pt",
            tmp_path / f"{selection}.pt",
            weights=selection,
        )
        model = load_rigidformer_checkpoint(path)
        expected = (
            resumed.learner.model.state_dict()
            if selection == "raw"
            else resumed.training_states["ema"].shadow
        )
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key].cpu(), rtol=0, atol=0)


@pytest.mark.gpu
def test_fp32_geometry_augmentation_precedes_bf16_model_autocast():
    pytest.importorskip("roma")
    from embodichain_models.dynamics.rigidformer import RigidFormerModel
    from embodichain_models.dynamics.rigidformer._native.data.gpu_augmentation import (
        GPUAugmentation,
    )

    cfg = tiny_cfg()
    architecture = cfg.architecture
    architecture["data"]["augmentation"] = {
        "rotation_augmentation": {
            "enable": True,
            "angle_step_size": 90,
            "prob_use_rotation": 1.0,
        }
    }
    model = RigidFormerModel(cfg).cuda().train()
    values = geometry_batch("cuda")
    values["valid_obj_mask"][:] = True
    objective = RigidFormerObjective(
        augmentation=GPUAugmentation(architecture, torch.device("cuda"))
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = objective.compute_loss(model, rigidformer_batch(values))
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
