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

from dataclasses import replace
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon, box, shape
import torch
import trimesh
import yaml

from embodichain.gen_sim.scene_engine.randomization.baseline import (
    prepare_scene_baseline,
    _entry_pose,
)
from embodichain.gen_sim.scene_engine.randomization.cfg import (
    SceneRandomizationCfg,
    MemberRandomizationCfg,
    IntraGroupRandomizationCfg,
    load_randomization_config,
)
from embodichain.gen_sim.scene_engine.randomization.generation import randomize_scene
from embodichain.gen_sim.scene_engine.randomization.loading import load_layouts
from embodichain.gen_sim.scene_engine.randomization.persistence import save_layouts
from embodichain.gen_sim.scene_engine.randomization._intra_geometry import (
    _prepare_geometry,
)
from .test_randomization_baseline import _export


@pytest.fixture
def intra_export(tmp_path: Path) -> tuple[Path, SceneRandomizationCfg, dict]:
    root, cfg = _export(tmp_path)
    path = root / "scene_config.json"
    config = json.loads(path.read_text())
    entries = {e["uid"]: e for e in config["background"] + config["rigid_object"]}
    entries["tray"].update(init_pos=[0.12, 0.02, 0.06], init_rot=[4, -3, 30])
    entries["cup"]["body_scale"] = [0.8, 1.3, 1.1]
    specs = {
        "tray": ([0.6, 0.04, 0.4], [0, 0, 0]),
        "cup": ([0.08, 0.06, 0.07], [0.015, 0.004, -0.003]),
        "block": ([0.02, 0.02, 0.02], [0, 0, 0]),
        "bottle": ([0.06, 0.08, 0.07], [0.02, 0, 0]),
    }
    for uid, (size, offset) in specs.items():
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(offset)
        mesh.export(root / entries[uid]["shape"]["fpath"])

    def place(uid: str, parent: str, translation: list, angles: list) -> None:
        rel = np.eye(4)
        rel[:3, :3] = Rotation.from_euler("XYZ", angles, degrees=True).as_matrix()
        rel[:3, 3] = translation
        pose = _entry_pose(entries[parent]) @ rel
        entries[uid].update(
            init_pos=pose[:3, 3].tolist(),
            init_rot=Rotation.from_matrix(pose[:3, :3])
            .as_euler("XYZ", degrees=True)
            .tolist(),
        )

    place("cup", "tray", [-0.10, 0, 0.075], [6, 3, -20])
    place("block", "cup", [0, 0, 0.0552], [0, 0, 0])
    entries["bottle"].update(init_pos=[-0.5, -0.05, 0.06], init_rot=[0, 0, 0])
    path.write_text(json.dumps(config))
    cfg = replace(
        cfg,
        backend="batched",
        candidate_count=8,
        max_attempts=1000,
        batch_size=64,
        collision_margin_m=0.01,
        intra_group=IntraGroupRandomizationCfg(
            targets={"cup": MemberRandomizationCfg()}
        ),
    )
    return root, cfg, entries


@pytest.mark.parametrize("groups", [False, True])
@pytest.mark.parametrize("storage", ["json", "packed"])
def test_hierarchical_roundtrip_and_independent_geometry(intra_export, groups, storage):
    root, cfg, entries = intra_export
    cfg.randomize_groups = groups
    result = randomize_scene(cfg)
    assert result.status == "completed"
    path = save_layouts(result, output_root=root.parent, layout_format=storage)
    loaded = load_layouts(path / "layouts")
    original = {uid: _entry_pose(e) for uid, e in entries.items()}
    for index, variant in enumerate(loaded.variants):
        assert variant.variant_id == result.candidates[index].variant.variant_id
        changes = {c.entity_uid: np.array(c.arena_pose) for c in variant.pose_changes}
        assert set(changes) == (
            {"tray", "cup", "block", "bottle"} if groups else {"cup", "block"}
        )
        poses = {**original, **changes}
        np.testing.assert_allclose(
            np.linalg.inv(poses["cup"]) @ poses["block"],
            np.linalg.inv(original["cup"]) @ original["block"],
            atol=1e-12,
        )
        if not groups:
            np.testing.assert_array_equal(poses["tray"], original["tray"])
        # Independent mesh vertices, parent frame and polygon containment.
        projected = []
        for uid in ("cup", "block"):
            mesh = trimesh.load(root / entries[uid]["shape"]["fpath"], force="mesh")
            v = np.asarray(mesh.vertices) * entries[uid]["body_scale"]
            v = v[:, [0, 2, 1]] * np.array([1, -1, 1])
            transform = np.linalg.inv(poses["tray"]) @ poses[uid]
            v = v @ transform[:3, :3].T + transform[:3, 3]
            projected.append(v)
        footprint = Polygon(np.concatenate(projected)[:, :2]).convex_hull
        assert box(-0.298, -0.198, 0.298, 0.198).covers(footprint)
        assert abs(projected[0][:, 2].min() - 0.021) < 1e-8


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_hierarchy_prefix_independent_of_batch_size(intra_export, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    _, cfg, _ = intra_export
    results = [
        randomize_scene(replace(cfg, device=device, batch_size=size, seed=2**40 + 17))
        for size in (1, 7, 128)
    ]
    assert all(r.status == "completed" for r in results)
    assert len({r.attempts for r in results}) == 1
    assert all(
        [c.variant.variant_id for c in r.candidates]
        == [c.variant.variant_id for c in results[0].candidates]
        for r in results
    )


def test_nested_targets_fixed_obstacles_and_cpu_cuda_geometry(intra_export):
    root, cfg, entries = intra_export
    cfg.intra_group.targets["block"] = MemberRandomizationCfg(
        xy_offset_bounds_m=[[-0.006, -0.005], [0.006, 0.005]],
        yaw_delta_range_deg=[-40, 40],
    )
    cpu = randomize_scene(cfg)
    assert cpu.status == "completed"
    if torch.cuda.is_available():
        gpu = randomize_scene(replace(cfg, device="cuda:0"))
        np.testing.assert_array_equal(
            cpu.candidates._ordinals, gpu.candidates._ordinals
        )
        np.testing.assert_allclose(
            cpu.candidates._selected, gpu.candidates._selected, atol=1e-12
        )
    # Add a fixed sibling into the selected parent's support area.
    config = json.loads((root / "scene_config.json").read_text())
    mesh_dir = root / "mesh_assets/obstacle"
    mesh_dir.mkdir()
    shutil.copyfile(
        root / entries["block"]["shape"]["fpath"], mesh_dir / "obstacle.glb"
    )
    obstacle = dict(
        entries["block"],
        uid="obstacle",
        shape={"shape_type": "Mesh", "fpath": "mesh_assets/obstacle/obstacle.glb"},
    )
    pose = _entry_pose(entries["tray"])
    pos = pose[:3, :3] @ np.array([0.12, 0, 0.035]) + pose[:3, 3]
    obstacle.update(init_pos=pos.tolist(), init_rot=entries["tray"]["init_rot"])
    config["rigid_object"].append(obstacle)
    (root / "scene_config.json").write_text(json.dumps(config))
    graph = json.loads((root / "scene_graph.json").read_text())
    graph["nodes"].append(
        dict(graph["nodes"][2], object_id="obstacle", parent_id="tray")
    )
    (root / "scene_graph.json").write_text(json.dumps(graph))
    result = randomize_scene(cfg)
    assert result.status == "completed"
    for candidate in result.candidates:
        poses = {
            c.entity_uid: np.array(c.arena_pose) for c in candidate.variant.pose_changes
        }
        np.testing.assert_allclose(
            np.linalg.inv(poses["tray"]) @ poses["obstacle"],
            np.linalg.inv(_entry_pose(entries["tray"])) @ _entry_pose(obstacle),
            atol=1e-12,
        )


def test_cache_immutable_cold_hot_and_no_asset_work_on_hot_path(
    intra_export, monkeypatch
):
    _, cfg, _ = intra_export
    baseline = prepare_scene_baseline(cfg)
    before = baseline.path.read_bytes()
    path, data, hit = _prepare_geometry(baseline, cfg.intra_group)
    assert not hit
    cache_before = path.read_bytes()
    import embodichain.gen_sim.scene_engine.randomization._intra_geometry as mod

    def forbidden(*args, **kwargs):
        raise AssertionError("Mesh work in warm generation")

    monkeypatch.setattr(mod, "load_glb_mesh", forbidden)
    for _ in range(2):
        result = randomize_scene(cfg)
        assert result.status == "completed"
    assert baseline.path.read_bytes() == before and path.read_bytes() == cache_before
    _, _, hit = _prepare_geometry(baseline, cfg.intra_group)
    assert hit
    other = replace(cfg.intra_group, grid_resolution_m=0.002)
    assert mod._cache_path(baseline, other) != path


@pytest.mark.parametrize(
    "tamper", ["member", "root", "identity", "cache", "source", "metadata"]
)
def test_v3_validates_all_records_before_consumption(intra_export, tamper):
    root, cfg, _ = intra_export
    result = randomize_scene(cfg)
    run = save_layouts(result, output_root=root.parent, layout_format="packed")
    file = run / "layouts/candidates.json"
    packed = json.loads(file.read_text())
    record = packed["candidates"][-1]
    if tamper == "member":
        record["member_poses_in_parent"]["cup"][0][3] += 0.5
    elif tamper == "root":
        record["root_xy_deltas"]["tray"][0] += 2
    elif tamper == "identity":
        record["variant_id"] = "forged"
    elif tamper == "metadata":
        record["variant_metadata"]["support_cache_sha256"] = "forged"
    elif tamper == "cache":
        cache = result.candidates._geometry_path
        data = json.loads(cache.read_text())
        data["nodes"][0]["corners"][0][0] += 0.1
        cache.write_text(json.dumps(data))
    else:
        (root / "mesh_assets/cup/cup.glb").write_bytes(b"changed source")
    file.write_text(json.dumps(packed))
    with pytest.raises(ValueError):
        load_layouts(run / "layouts")


def test_impossible_offsets_budget_and_empty_save(intra_export):
    root, cfg, _ = intra_export
    cfg.intra_group.targets["cup"].xy_offset_bounds_m = [[1, 1], [1, 1]]
    cfg.max_attempts = 19
    result = randomize_scene(cfg)
    assert (
        result.status == "budget_exhausted"
        and result.attempts == 19
        and not result.candidates
    )
    run = save_layouts(result, output_root=root.parent, layout_format="packed")
    assert json.loads((run / "run_result.json").read_text())["generated_count"] == 0


@pytest.mark.parametrize("uid", ["table", "tray", "unknown"])
def test_target_must_be_existing_nonroot(intra_export, uid):
    _, cfg, _ = intra_export
    cfg.intra_group.targets = {uid: MemberRandomizationCfg()}
    with pytest.raises(ValueError, match="non-root"):
        randomize_scene(cfg)


@pytest.mark.parametrize(
    "updates",
    [
        {"randomize_groups": False},
        {"intra_group": {"targets": {"cup": {"unknown": 2}}}},
        {"intra_group": {"targets": {"cup": {"yaw_delta_range_deg": [180, -180]}}}},
        {"intra_group": {"grid_resolution_m": 0}},
        {"intra_group": {"targets": {"cup": {}}}, "backend": "slsqp"},
    ],
)
def test_strict_v3_policy_validation(tmp_path, updates):
    path = tmp_path / "policy.yaml"
    path.write_text(
        yaml.safe_dump({"schema_version": 3, "backend": "batched", **updates})
    )
    with pytest.raises(ValueError):
        load_randomization_config(path, output_root=tmp_path)


def test_resolved_v3_reusable_and_intra_disabled_legacy_identity(intra_export):
    root, cfg, _ = intra_export
    result = randomize_scene(cfg)
    run = save_layouts(result, output_root=root.parent)
    reloaded = load_randomization_config(
        run / "policy_resolved.yaml", output_root=root.parent
    )
    again = randomize_scene(reloaded)
    assert [c.variant.variant_id for c in result.candidates] == [
        c.variant.variant_id for c in again.candidates
    ]
    plain = randomize_scene(replace(cfg, intra_group=None))
    disabled = randomize_scene(replace(cfg, intra_group=IntraGroupRandomizationCfg()))
    assert [c.variant.variant_id for c in plain.candidates] == [
        c.variant.variant_id for c in disabled.candidates
    ]


@pytest.mark.parametrize("yaw", [-180, 180, 270])
def test_degenerate_continuous_yaw_range_and_explicit_origin_bounds(intra_export, yaw):
    root, cfg, _ = intra_export
    cfg.intra_group.targets["cup"] = MemberRandomizationCfg(
        xy_offset_bounds_m=[[0, 0], [0, 0]], yaw_delta_range_deg=[yaw, yaw]
    )
    cfg.randomize_groups = False
    result = randomize_scene(cfg)
    assert result.status == "completed"
    run = save_layouts(result, output_root=root.parent, layout_format="packed")
    assert len(load_layouts(run / "layouts").variants) == cfg.candidate_count


def test_partial_budget_and_manifest_accounting(intra_export):
    root, cfg, _ = intra_export
    cfg.candidate_count = cfg.max_attempts = 30
    result = randomize_scene(cfg)
    assert result.status == "budget_exhausted" and 0 < len(result.candidates) < 30
    assert result.attempts == 30
    run = save_layouts(result, output_root=root.parent, layout_format="packed")
    loaded = load_layouts(run / "layouts")
    assert len(loaded.variants) == len(result.candidates)
    path = run / "run_result.json"
    manifest = json.loads(path.read_text())
    manifest["solver_status_counts"]["solved"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="accounting"):
        load_layouts(run / "layouts")


@pytest.mark.parametrize("value", [None, [], 1])
def test_targets_requires_mapping(tmp_path, value):
    path = tmp_path / "policy.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "backend": "batched",
                "intra_group": {"targets": value},
            }
        )
    )
    with pytest.raises(ValueError, match="mapping"):
        load_randomization_config(path, output_root=tmp_path)


def test_saved_member_rejects_bool_and_text_before_consumption(intra_export):
    root, cfg, _ = intra_export
    result = randomize_scene(cfg)
    run = save_layouts(result, output_root=root.parent, layout_format="packed")
    path = run / "layouts/candidates.json"
    packed = json.loads(path.read_text())
    packed["candidates"][-1]["member_poses_in_parent"]["cup"][3][3] = True
    path.write_text(json.dumps(packed))
    with pytest.raises(ValueError, match="numeric"):
        load_layouts(run / "layouts")


def test_numeric_save_load_is_lazy_and_parameters_are_owned(intra_export, monkeypatch):
    import embodichain.gen_sim.scene_engine.randomization._batch as batch

    root, cfg, _ = intra_export
    result = randomize_scene(cfg)
    cfg.intra_group.targets["cup"].yaw_delta_range_deg[0] = 0
    assert (
        result.resolved_parameters["intra_group"]["targets"]["cup"][
            "yaw_delta_range_deg"
        ][0]
        == -180
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Materialized a typed entity pose before consumption")

    monkeypatch.setattr(batch, "ScenePoseChange", forbidden)
    run = save_layouts(result, output_root=root.parent, layout_format="packed")
    loaded = load_layouts(run / "layouts")
    assert len(loaded.variants) == len(result.candidates)
    with pytest.raises(AssertionError, match="before consumption"):
        loaded.variants[0]


def test_programmatic_override_cache_survives_json_normalization(intra_export):
    from embodichain.gen_sim.scene_engine.randomization.cfg import SupportOverrideCfg
    from shapely.geometry import mapping

    root, cfg, _ = intra_export
    frame = np.eye(4)
    frame[2, 3] = 0.02
    cfg.intra_group.support_overrides = {
        "tray": SupportOverrideCfg(
            parent_from_support=frame.tolist(),
            polygon=mapping(box(-0.28, -0.18, 0.28, 0.18)),
        )
    }
    first = randomize_scene(cfg)
    assert first.status == "completed"
    again = randomize_scene(cfg)
    assert [x.variant.variant_id for x in first.candidates] == [
        x.variant.variant_id for x in again.candidates
    ]
    run = save_layouts(first, output_root=root.parent, layout_format="packed")
    assert len(load_layouts(run / "layouts").variants) == cfg.candidate_count


def test_dynamic_groups_use_final_footprint_not_oversized_initial_proxy(intra_export):
    from embodichain.gen_sim.scene_engine.randomization.cfg import SupportOverrideCfg
    from shapely.geometry import mapping

    root, cfg, _ = intra_export
    path = root / "scene_config.json"
    config = json.loads(path.read_text())
    # An authored overhang can be repaired using an explicit mesh-backed floor.
    for entry in config["rigid_object"]:
        if entry["uid"] in ("cup", "block"):
            entry["init_pos"][0] += 3
    path.write_text(json.dumps(config))
    frame = np.eye(4)
    frame[2, 3] = 0.02
    cfg.intra_group.support_overrides = {
        "tray": SupportOverrideCfg(
            parent_from_support=frame.tolist(),
            polygon=mapping(box(-0.28, -0.18, 0.28, 0.18)),
        )
    }
    result = randomize_scene(cfg)
    assert result.status == "completed"
    assert any(
        group["aabb_2d"]["size_xy"][0] > result.baseline.data["table"]["size_xy"][0]
        for group in result.baseline.data["groups"]
    )
    run = save_layouts(result, output_root=root.parent, layout_format="packed")
    assert len(load_layouts(run / "layouts").variants) == cfg.candidate_count


def test_intra_only_playback_limits_candidate_write_permissions(
    intra_export, monkeypatch
):
    from embodichain.gen_sim.scene_engine.randomization import loading

    root, cfg, _ = intra_export
    cfg.randomize_groups = False
    result = randomize_scene(cfg)
    run = save_layouts(result, output_root=root.parent, layout_format="packed")
    loaded = load_layouts(run / "layouts")
    emitted = []
    monkeypatch.setattr(loading, "SceneView", lambda sim: object())
    monkeypatch.setattr(
        loading, "apply_changes", lambda view, changes: emitted.append(changes)
    )
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(loading.time, "monotonic", lambda: next(clock))
    playback = loading._Playback(loaded, 1.0)
    playback.update(object())
    playback.update(object())
    assert set(emitted[0].movable_uids) == set(loaded.baseline_poses)
    assert set(emitted[1].movable_uids) == {"cup", "block"}


def test_concurrent_cold_support_preparations_publish_one_immutable_cache(
    intra_export, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from embodichain.gen_sim.scene_engine.randomization import persistence

    _, cfg, _ = intra_export
    baseline = prepare_scene_baseline(cfg)
    writer = persistence._write_atomic
    ready = threading.Barrier(2)
    first_done = threading.Event()
    current = threading.local()

    def synchronized_write(path, data):
        writer(path, data)
        ready.wait(timeout=15)
        if current.role == "second":
            assert first_done.wait(timeout=15)

    def prepare(role):
        current.role = role
        try:
            return _prepare_geometry(baseline, cfg.intra_group)
        finally:
            if role == "first":
                first_done.set()

    monkeypatch.setattr(persistence, "_write_atomic", synchronized_write)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(prepare, role) for role in ("first", "second")]
        results = [future.result(timeout=30) for future in futures]
    assert results[0][0] == results[1][0]
    assert results[0][1]["data_sha256"] == results[1][1]["data_sha256"]
    assert list(results[0][0].parent.iterdir()) == [results[0][0]]


@pytest.mark.parametrize("storage", ["json", "packed"])
@pytest.mark.parametrize("stage", ["before_save", "during_write"])
def test_cache_replacement_cannot_publish_a_completed_run(
    intra_export, monkeypatch, storage, stage
):
    from embodichain.gen_sim.scene_engine.randomization import persistence
    from embodichain.gen_sim.scene_engine.randomization._intra_geometry import _digest

    root, cfg, _ = intra_export
    result = randomize_scene(cfg)
    cache = result.candidates._geometry_path

    def replace_cache():
        data = json.loads(cache.read_text())
        data["metadata_note"] = "replaced after the generation snapshot"
        data["data_sha256"] = _digest(
            {key: value for key, value in data.items() if key != "data_sha256"}
        )
        cache.write_text(json.dumps(data))

    if stage == "before_save":
        replace_cache()
    else:
        writer = persistence._write_atomic
        changed = False

        def replace_during_write(path, data):
            nonlocal changed
            writer(path, data)
            if path.parent.name == "layouts" and not changed:
                replace_cache()
                changed = True

        monkeypatch.setattr(persistence, "_write_atomic", replace_during_write)
    with pytest.raises(ValueError, match="generated snapshot"):
        save_layouts(result, output_root=root.parent, layout_format=storage)
    assert not list(root.parent.glob("scene_layouts/*/run_result.json"))


@pytest.mark.parametrize(
    "polygon",
    [
        None,
        [],
        {"type": "Point", "coordinates": [0, 0]},
        {"type": "Polygon", "coordinates": [[[0, 0], [10**400, 0], [0, 1], [0, 0]]]},
        {"type": "Polygon", "coordinates": []},
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 1]]]},
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]]]},
    ],
)
def test_override_polygon_is_validated_before_scene_access(tmp_path, polygon):
    path = tmp_path / "policy.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "backend": "batched",
                "intra_group": {
                    "targets": {"cup": {}},
                    "support_overrides": {
                        "tray": {
                            "parent_from_support": np.eye(4).tolist(),
                            "polygon": polygon,
                        }
                    },
                },
            }
        )
    )
    with pytest.raises(ValueError, match="Support override 'tray'"):
        load_randomization_config(path, output_root=tmp_path)
    assert not (tmp_path / "scene_export").exists()


def test_shallow_parent_patch_cannot_consume_the_support_gap(intra_export):
    root, cfg, entries = intra_export
    tray_path = root / entries["tray"]["shape"]["fpath"]
    tray = trimesh.load(tray_path, force="mesh")
    patch = trimesh.creation.box(extents=[0.16, 0.0018, 0.14])
    patch.apply_translation([-0.10, 0.0209, 0])
    trimesh.util.concatenate([tray, patch]).export(tray_path)
    cfg.randomize_groups = False
    cfg.intra_group.targets["cup"] = MemberRandomizationCfg(
        xy_offset_bounds_m=[[0, 0], [0, 0]], yaw_delta_range_deg=[0, 0]
    )
    result = randomize_scene(cfg)
    assert result.status == "completed"
    parent_pose = _entry_pose(entries["tray"])
    mesh = trimesh.load(root / entries["cup"]["shape"]["fpath"], force="mesh")
    vertices = np.asarray(mesh.vertices) * entries["cup"]["body_scale"]
    vertices = vertices[:, [0, 2, 1]] * [1, -1, 1]
    patch_top = np.asarray(trimesh.load(tray_path, force="mesh").vertices)[:, 1].max()
    for candidate in result.candidates:
        pose = next(
            c.arena_pose
            for c in candidate.variant.pose_changes
            if c.entity_uid == "cup"
        )
        relative = np.linalg.inv(parent_pose) @ np.asarray(pose)
        points = vertices @ relative[:3, :3].T + relative[:3, 3]
        assert points[:, 2].min() >= patch_top + cfg.intra_group.support_gap_m - 1e-9


def test_v3_previous_generator_and_cache_still_replay(intra_export, monkeypatch):
    from embodichain.gen_sim.scene_engine.randomization import _intra, _intra_geometry

    root, cfg, _ = intra_export
    legacy_generator = "scene_engine.hierarchical_batch_rejection/v1"
    monkeypatch.setattr(_intra, "_INTRA_GENERATOR", legacy_generator)
    monkeypatch.setattr(_intra_geometry, "_EXTRACTOR", "parent_support_grid/v1")
    prepare_support = _intra_geometry._prepare_support

    def legacy_support(*args, **kwargs):
        support = prepare_support(*args, **kwargs)
        support.pop("height_ceiling_m")
        return support

    monkeypatch.setattr(_intra_geometry, "_prepare_support", legacy_support)
    result = randomize_scene(cfg)
    run = save_layouts(result, output_root=root.parent, layout_format="packed")
    expected = [c.variant.variant_id for c in result.candidates]
    monkeypatch.undo()
    loaded = load_layouts(run / "layouts")
    assert [v.variant_id for v in loaded.variants] == expected


@pytest.mark.parametrize(
    "field", ["grid_resolution_m", "boundary_margin_m", "support_gap_m"]
)
def test_unrepresentable_geometry_controls_are_configuration_errors(tmp_path, field):
    path = tmp_path / "policy.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "backend": "batched",
                "intra_group": {"targets": {"cup": {}}, field: 10**400},
            }
        )
    )
    with pytest.raises(ValueError, match=field):
        load_randomization_config(path, output_root=tmp_path)


def test_large_group_count_bounds_actual_batch_memory_without_changing_prefix(
    intra_export, monkeypatch
):
    from embodichain.gen_sim.scene_engine.randomization import _intra

    root, cfg, _ = intra_export
    limit = _intra._hierarchy_batch_limit
    assert limit(4096, 200) < limit(4096, 100) < 4096
    assert limit(4096, 4) == 4096
    with pytest.raises(ValueError, match="Group count.*memory budget"):
        limit(4096, 100000)
    expected = randomize_scene(cfg)
    monkeypatch.setattr(_intra, "_hierarchy_batch_limit", lambda requested, count: 3)
    bounded = randomize_scene(cfg)
    assert bounded.resolved_parameters["batch_size"] == 3
    assert expected.attempts == bounded.attempts
    assert [c.variant.variant_id for c in expected.candidates] == [
        c.variant.variant_id for c in bounded.candidates
    ]


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 10**400, "invalid"])
def test_override_transform_numeric_errors_are_configuration_errors(tmp_path, value):
    frame = np.eye(4).tolist()
    frame[0][3] = value
    path = tmp_path / "policy.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "backend": "batched",
                "intra_group": {
                    "targets": {"cup": {}},
                    "support_overrides": {
                        "tray": {
                            "parent_from_support": frame,
                            "polygon": {
                                "type": "Polygon",
                                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                            },
                        }
                    },
                },
            }
        )
    )
    with pytest.raises(ValueError, match="parent_from_support"):
        load_randomization_config(path, output_root=tmp_path)
    assert not (tmp_path / "scene_export").exists()


@pytest.mark.parametrize("slot", [(3, 3), (0, 0)])
def test_override_rejects_nonrigid_matrix_before_preparing_assets(tmp_path, slot):
    frame = np.eye(4).tolist()
    frame[slot[0]][slot[1]] += 5e-6
    path = tmp_path / "policy.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "backend": "batched",
                "intra_group": {
                    "targets": {"cup": {}},
                    "support_overrides": {
                        "tray": {
                            "parent_from_support": frame,
                            "polygon": {
                                "type": "Polygon",
                                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                            },
                        }
                    },
                },
            }
        )
    )
    with pytest.raises(ValueError, match="proper rigid"):
        load_randomization_config(path, output_root=tmp_path)
    assert not (tmp_path / "scene_export").exists()
