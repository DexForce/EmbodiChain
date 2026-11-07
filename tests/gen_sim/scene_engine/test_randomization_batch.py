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

"""Batch budgets, ownership, canonical identities and packed run persistence."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from embodichain.gen_sim.scene_engine.randomization import (
    _batch,
    generation,
    loading,
    persistence,
)
from embodichain.gen_sim.scene_engine.randomization.cfg import load_randomization_config

from .test_randomization_generation import grouped_export, _mesh_aabb


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "unknown"),
        ("device", "mps"),
        ("device", "cuda:bad"),
        ("batch_size", 0),
        ("batch_size", True),
    ],
)
def test_invalid_batch_controls_fail_through_loader(
    tmp_path: Path, field: str, value: object
) -> None:
    import yaml

    policy = {"schema_version": 1, "backend": "batched", field: value}
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(policy))
    with pytest.raises(ValueError):
        load_randomization_config(path, output_root=tmp_path)


def test_slsqp_cannot_silently_ignore_cuda(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text("schema_version: 1\ndevice: cuda\n")
    with pytest.raises(ValueError, match="slsqp"):
        load_randomization_config(path, output_root=tmp_path)


def test_resolved_batch_policy_is_reusable_input(grouped_export: tuple) -> None:
    root, cfg, _ = grouped_export
    original = generation.randomize_scene(
        cfg.replace(backend="batched"), candidate_count=2
    )
    run = persistence.save_layouts(
        original, output_root=root.parent, layout_format="packed"
    )
    restored_cfg = load_randomization_config(
        run / "policy_resolved.yaml", output_root=root.parent
    )
    restored = generation.randomize_scene(restored_cfg)
    assert restored.resolved_parameters == original.resolved_parameters
    assert [c.variant for c in restored.candidates] == [
        c.variant for c in original.candidates
    ]


def test_unavailable_cuda_is_explicit(
    grouped_export: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, cfg, _ = grouped_export
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        generation.randomize_scene(cfg.replace(backend="batched", device="cuda:0"))


def test_missing_cuda_device_is_explicit(
    grouped_export: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, cfg, _ = grouped_export
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="unavailable"):
        generation.randomize_scene(cfg.replace(backend="batched", device="cuda:1"))


def test_batch_partitioning_preserves_logical_proposals(grouped_export: tuple) -> None:
    _, cfg, _ = grouped_export
    results = [
        generation.randomize_scene(
            cfg.replace(backend="batched", batch_size=size),
            seed=2**40 + 7,
            candidate_count=9,
            max_attempts=1000,
        )
        for size in (1, 7, 128)
    ]
    reference = results[0]
    for result in results[1:]:
        assert result.attempts == reference.attempts
        assert [c.variant.variant_id for c in result.candidates] == [
            c.variant.variant_id for c in reference.candidates
        ]
    prefix = generation.randomize_scene(
        cfg.replace(backend="batched", batch_size=11),
        seed=2**40 + 7,
        candidate_count=4,
        max_attempts=1000,
    )
    assert [c.variant for c in prefix.candidates] == [
        c.variant for c in reference.candidates[:4]
    ]
    assert (
        prefix.attempts
        == int(dict(prefix.candidates[-1].variant.metadata)["proposal_ordinal"]) + 1
    )


@pytest.mark.parametrize("layout_format", ["json", "packed"])
@pytest.mark.parametrize("backend", ["slsqp", "batched"])
def test_batch_roundtrip_is_lazy_and_preserves_complete_poses(
    grouped_export: tuple,
    monkeypatch: pytest.MonkeyPatch,
    layout_format: str,
    backend: str,
) -> None:
    root, cfg, entries = grouped_export
    result = generation.randomize_scene(
        cfg.replace(backend=backend, batch_size=32),
        candidate_count=8,
        max_attempts=1000,
    )
    real_getitem = _batch._CandidatePool.__getitem__

    def forbidden(*args: object) -> None:
        pytest.fail("Saving must not materialize per-candidate runtime configurations")

    monkeypatch.setattr(_batch._CandidatePool, "__getitem__", forbidden)
    run = persistence.save_layouts(
        result, output_root=root.parent, layout_format=layout_format
    )
    loaded = loading.load_layouts(run / "layouts")
    monkeypatch.setattr(_batch._CandidatePool, "__getitem__", real_getitem)
    assert len(loaded.variants) == 8
    assert len(list((run / "layouts").glob("*.json"))) == (
        1 if layout_format == "packed" else 8
    )
    for candidate, variant in zip(result.candidates, loaded.variants, strict=True):
        assert variant == candidate.variant
        entities = {e.uid: e for e in candidate.fragment.entities}
        for change in variant.pose_changes:
            np.testing.assert_allclose(
                entities[change.entity_uid].init_local_pose,
                change.arena_pose,
                atol=1e-12,
            )
        boxes = [
            _mesh_aabb(root, entries[change.entity_uid], np.asarray(change.arena_pose))
            for change in variant.pose_changes
        ]
        assert all(np.isfinite(box).all() for box in boxes)
    detached = result.candidates[0].fragment.entities
    detached[0].init_pos = (999, 999, 999)
    assert result.candidates[0].fragment.entities[0].init_pos != (999, 999, 999)
    assert loaded.variants[-1] == loaded.variants[len(loaded.variants) - 1]
    assert tuple(loaded.variants[:2]) == (loaded.variants[0], loaded.variants[1])


def test_exhaustion_retains_exact_accounting_and_empty_manifest(
    grouped_export: tuple,
) -> None:
    root, cfg, _ = grouped_export
    result = generation.randomize_scene(
        cfg.replace(backend="batched", batch_size=8),
        collision_margin_m=100,
        candidate_count=3,
        max_attempts=13,
    )
    assert result.status == "budget_exhausted"
    assert result.attempts == 13 and len(result.candidates) == 0
    assert result.solver_status_counts == {"solved": 0, "no_solution": 13}
    run = persistence.save_layouts(
        result, output_root=root.parent, layout_format="packed"
    )
    assert json.loads((run / "run_result.json").read_text())["generated_count"] == 0
    with pytest.raises(ValueError):
        loading.load_layouts(run / "layouts")


@pytest.mark.parametrize(
    "tamper",
    ["index", "identity", "center", "missing", "seed_type", "ordinal_type", "metadata"],
)
def test_packed_tampering_is_rejected_before_consumption(
    grouped_export: tuple, tamper: str
) -> None:
    root, cfg, _ = grouped_export
    result = generation.randomize_scene(
        cfg.replace(backend="batched"), candidate_count=2
    )
    run = persistence.save_layouts(
        result, output_root=root.parent, layout_format="packed"
    )
    manifest_path = run / "run_result.json"
    manifest = json.loads(manifest_path.read_text())
    path = run / "layouts/candidates.json"
    packed = json.loads(path.read_text())
    if tamper == "index":
        manifest["layouts"][0]["index"] = True
    elif tamper == "identity":
        packed["candidates"][0]["variant_id"] = "scene-variant:invalid"
        manifest["layouts"][0]["variant_id"] = "scene-variant:invalid"
    elif tamper == "center":
        packed["candidates"][0]["groups"][0]["center_xy"][0] += 100
    elif tamper == "missing":
        packed["candidates"].pop()
    elif tamper == "seed_type":
        packed["candidates"][0]["seed"] = False
    elif tamper == "ordinal_type":
        packed["candidates"][0]["variant_ordinal"] = False
    else:
        packed["candidates"][0]["variant_metadata"]["generator"] = "unknown"
    path.write_text(json.dumps(packed))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        loading.load_layouts(run / "layouts")


def test_batch_provenance_cannot_be_replaced_before_save(grouped_export: tuple) -> None:
    root, cfg, _ = grouped_export
    result = generation.randomize_scene(
        cfg.replace(backend="batched"), candidate_count=2
    )
    changed = replace(
        result, resolved_parameters={**result.resolved_parameters, "seed": 99}
    )
    with pytest.raises(ValueError, match="provenance"):
        persistence.save_layouts(
            changed, output_root=root.parent, layout_format="packed"
        )
    assert not (root.parent / "scene_layouts").exists()


def test_numeric_pipeline_defers_typed_pose_allocations(
    grouped_export: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cfg, _ = grouped_export

    def forbidden(*args: object) -> None:
        pytest.fail("Unconsumed batch poses must remain numeric")

    monkeypatch.setattr(_batch, "ScenePoseChange", forbidden)
    result = generation.randomize_scene(
        cfg.replace(backend="batched"), candidate_count=12
    )
    run = persistence.save_layouts(
        result, output_root=root.parent, layout_format="packed"
    )
    loaded = loading.load_layouts(run / "layouts")
    assert len(loaded.variants) == len(result.candidates) == 12


def test_cached_batch_does_not_advance_global_rng(grouped_export: tuple) -> None:
    _, cfg, _ = grouped_export
    # Trimesh baseline extraction has its own sampling; isolate the counter
    # proposal backend from that one-time geometry preparation.
    generation.prepare_scene_baseline(cfg)
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    generation.randomize_scene(cfg.replace(backend="batched"), candidate_count=5)
    assert torch.equal(torch.get_rng_state(), torch_state)
    after = np.random.get_state()
    np.testing.assert_array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]


def test_source_change_during_packed_write_blocks_manifest(
    grouped_export: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cfg, _ = grouped_export
    result = generation.randomize_scene(
        cfg.replace(backend="batched"), candidate_count=2
    )
    original = persistence._write_atomic

    def change_source(path: Path, data: dict) -> None:
        original(path, data)
        if path.name == "candidates.json":
            source = root / "scene_config.json"
            source.write_bytes(source.read_bytes() + b"\n")

    monkeypatch.setattr(persistence, "_write_atomic", change_source)
    with pytest.raises(ValueError, match="Source scene changed"):
        persistence.save_layouts(
            result, output_root=root.parent, layout_format="packed"
        )
    assert not list((root.parent / "scene_layouts").glob("*/run_result.json"))


def test_packed_write_failure_does_not_publish_completion(
    grouped_export: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cfg, _ = grouped_export
    result = generation.randomize_scene(
        cfg.replace(backend="batched"), candidate_count=2
    )
    original = persistence.os.replace

    def fail_container(source: str, destination: Path) -> None:
        if Path(destination).name == "candidates.json":
            raise OSError("Injected packed publication failure")
        original(source, destination)

    monkeypatch.setattr(persistence.os, "replace", fail_container)
    with pytest.raises(OSError, match="Injected"):
        persistence.save_layouts(
            result, output_root=root.parent, layout_format="packed"
        )
    assert not list((root.parent / "scene_layouts").glob("*/run_result.json"))
    assert not list((root.parent / "scene_layouts").rglob("*.tmp"))


@pytest.mark.gpu
def test_cuda_batch_roundtrip_matches_cpu_geometry(grouped_export: tuple) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    root, cfg, _ = grouped_export
    cpu = generation.randomize_scene(
        cfg.replace(backend="batched", batch_size=17), candidate_count=7
    )
    cuda = generation.randomize_scene(
        cfg.replace(backend="batched", device="cuda:0", batch_size=11),
        candidate_count=7,
    )
    assert cpu.attempts == cuda.attempts
    for first, second in zip(cpu.candidates, cuda.candidates, strict=True):
        for a, b in zip(
            first.variant.pose_changes, second.variant.pose_changes, strict=True
        ):
            np.testing.assert_allclose(a.arena_pose, b.arena_pose, atol=1e-12, rtol=0)
    run = persistence.save_layouts(
        cuda, output_root=root.parent, layout_format="packed"
    )
    loaded = loading.load_layouts(run / "layouts")
    assert [v.variant_id for v in loaded.variants] == [
        c.variant.variant_id for c in cuda.candidates
    ]
