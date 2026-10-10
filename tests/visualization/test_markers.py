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

import numpy as np
import pytest

from embodichain.lab.visualization.markers import (
    MarkerGroup,
    MarkerGroupCfg,
    MarkerPrototypeCfg,
)


def make_group(**kwargs):
    return MarkerGroup(
        MarkerGroupCfg(
            name="targets",
            prototypes={
                "red": MarkerPrototypeCfg(shape="box", color=(1, 0, 0, 0.4)),
                "blue": MarkerPrototypeCfg(
                    shape="sphere", scale=(0.2, 0.2, 0.2), color=(0, 0, 1, 1)
                ),
            },
            **kwargs,
        )
    )


def test_update_preserves_other_fields_and_snapshots_own_arrays():
    group = make_group()
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    group.update(translations=positions, prototype_indices=[1, 0])
    positions[:] = 99
    before = group.snapshot()
    group.update(colors=[[0, 1, 0, 0.25], [1, 1, 0, 0.5]])
    after = group.snapshot()
    assert len(after) == group.count == 2
    np.testing.assert_allclose(after[0].position, [1, 2, 3])
    np.testing.assert_allclose(after[0].scale, [0.2, 0.2, 0.2])
    assert after[0].color == (0, 1, 0, 0.25)
    assert before[0].color == (0, 0, 1, 1)
    after[0].position[:] = 9
    np.testing.assert_allclose(group.snapshot()[0].position, [1, 2, 3])


@pytest.mark.parametrize(
    "bad",
    [
        {"translations": [[float("nan"), 0, 0]]},
        {"orientations_xyzw": [[0, 0, 0, 0]]},
        {"scales": [[1, -1, 1]]},
        {"prototype_indices": [2]},
        {"prototype_indices": [0.5]},
        {"colors": [[1, 0, 0, 2]]},
        {"visible": [2]},
        {"orientations_xyzw": [[0, 0, 0, 1], [0, 0, 0, 1]]},
    ],
)
def test_invalid_update_leaves_state_unchanged(bad):
    group = make_group()
    group.update(translations=[[1, 2, 3]])
    before = group.snapshot()[0]
    with pytest.raises(ValueError):
        group.update(**bad)
    after = group.snapshot()[0]
    np.testing.assert_array_equal(after.position, before.position)
    np.testing.assert_array_equal(after.wxyz, before.wxyz)
    assert after.color == before.color


def test_count_changes_reset_omitted_attributes_and_clear_retains_group():
    group = make_group()
    group.update(translations=[[0, 0, 0]], prototype_indices=[1], visible=[False])
    group.update(translations=[[1, 0, 0], [2, 0, 0]])
    assert all(s.visible for s in group.snapshot())
    assert all(s.color == (1, 0, 0, 0.4) for s in group.snapshot())
    group.clear()
    assert group.count == 0 and group.snapshot() == ()
    group.update(translations=[[3, 0, 0]])
    assert group.count == 1


def test_group_visibility_does_not_discard_updates_or_per_instance_visibility():
    group = make_group()
    group.update(translations=[[0, 0, 0], [1, 0, 0]], visible=[True, False])
    group.set_visibility(False)
    group.update(translations=[[2, 0, 0], [3, 0, 0]])
    assert not any(s.visible for s in group.snapshot())
    group.set_visibility(True)
    assert [s.visible for s in group.snapshot()] == [True, False]
    np.testing.assert_allclose(group.snapshot()[0].position, [2, 0, 0])


def test_quaternion_order_and_arena_translation_applied_once():
    group = MarkerGroup(
        MarkerGroupCfg(name="env", prototypes={"box": MarkerPrototypeCfg()}),
        origin=[10, 20, 30],
    )
    quat = np.array([1.0, 2.0, 3.0, 4.0])
    group.update(translations=[[1, 2, 3]], orientations_xyzw=[quat])
    marker = group.snapshot()[0]
    np.testing.assert_allclose(marker.position, [11, 22, 33])
    np.testing.assert_allclose(marker.wxyz, quat[[3, 0, 1, 2]] / np.linalg.norm(quat))
    assert marker.env_id == 0


def test_removed_group_cannot_be_reused_and_callback_releases_once():
    removed = []
    group = MarkerGroup(
        MarkerGroupCfg(name="gone", prototypes={"box": MarkerPrototypeCfg()}),
        on_remove=lambda g: removed.append(g.name),
    )
    group.remove()
    group.remove()
    assert removed == ["gone"]
    with pytest.raises(RuntimeError, match="removed"):
        group.update(translations=[[0, 0, 0]])


@pytest.mark.parametrize(
    "shape", ["box", "sphere", "cylinder", "capsule", "cone", "arrow", "frame"]
)
def test_builtin_geometries_are_finite_valid_triangles(shape):
    group = MarkerGroup(
        MarkerGroupCfg(name=shape, prototypes={shape: MarkerPrototypeCfg(shape=shape)})
    )
    group.update(translations=[[0, 0, 0]])
    meshes = group.snapshot()
    assert len(meshes) == (3 if shape == "frame" else 1)
    for mesh in meshes:
        assert len(mesh.vertices) > 3 and len(mesh.faces) > 0
        assert np.isfinite(mesh.vertices).all()
        assert mesh.faces.max() < len(mesh.vertices)
    if shape == "frame":
        assert [s.color[:3] for s in meshes] == [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def test_custom_mesh_and_config_are_detached_from_caller():
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    cfg = MarkerGroupCfg(
        name="triangle",
        prototypes={
            "mesh": MarkerPrototypeCfg(
                shape="mesh", vertices=vertices, faces=[[0, 1, 2]]
            )
        },
    )
    group = MarkerGroup(cfg)
    vertices[:] = 99
    cfg.prototypes["mesh"].color = (0, 0, 0, 0)
    group.update(translations=[[0, 0, 0]])
    assert group.snapshot()[0].vertices.max() == 1
    assert group.snapshot()[0].color == (1, 0, 0, 1)


def test_failed_publication_rolls_back_logical_state():
    def fail(group):
        raise RuntimeError("renderer rejected update")

    group = MarkerGroup(
        MarkerGroupCfg(name="failure", prototypes={"box": MarkerPrototypeCfg()}),
        on_change=fail,
    )
    with pytest.raises(RuntimeError, match="renderer rejected"):
        group.update(translations=[[0, 0, 0]])
    assert group.count == 0


@pytest.mark.parametrize(
    "values",
    [
        {"translations": [[1e40, 0, 0]]},
        {"scales": [[1e40, 1, 1]]},
        {"orientations_xyzw": [[1e300, 0, 0, 1]]},
    ],
)
def test_update_rejects_values_not_representable_by_render_snapshots(values):
    group = make_group()
    group.update(translations=[[0, 0, 0]])
    with pytest.raises(ValueError):
        group.update(**values)
    np.testing.assert_allclose(group.snapshot()[0].position, [0, 0, 0])


def test_scale_product_is_validated_before_committing_state():
    group = MarkerGroup(
        MarkerGroupCfg(
            name="scaled", prototypes={"box": MarkerPrototypeCfg(scale=(1e30, 1, 1))}
        )
    )
    group.update(translations=[[0, 0, 0]])
    with pytest.raises(ValueError):
        group.update(scales=[[1e30, 1, 1]])
    np.testing.assert_allclose(group.snapshot()[0].scale, [1e30, 1, 1])


def batched_group(**kwargs):
    return MarkerGroup(
        MarkerGroupCfg(name="batch", prototypes={"box": MarkerPrototypeCfg()}),
        num_envs=2,
        origins=[[10, 0, 0], [0, 20, 0]],
        **kwargs,
    )


def test_environment_batches_preserve_unselected_counts_styles_and_origins():
    group = batched_group()
    group.update(
        translations=[[[1, 0, 0]], [[2, 0, 0]]],
        colors=[[[1, 0, 0, 0.2]], [[0, 1, 0, 0.4]]],
    )
    assert group.count == 2 and group.counts == (1, 1)
    original_ids = [m.overlay_id for m in group.snapshot()]
    group.update(env_ids=[1], translations=[[3, 0, 0], [4, 0, 0]])
    assert group.counts == (1, 2)
    meshes = group.snapshot()
    np.testing.assert_allclose(
        [m.position for m in meshes], [[11, 0, 0], [3, 20, 0], [4, 20, 0]]
    )
    assert meshes[0].color == (1, 0, 0, 0.2)
    assert meshes[1].color == (1, 0, 0, 1)
    assert [m.overlay_id for m in meshes[:2]] == original_ids
    group.set_visibility(False, env_ids=[1])
    group.update(env_ids=[1], translations=[[5, 0, 0]])
    assert [m.visible for m in group.snapshot()] == [True, False]
    group.clear(env_ids=[0])
    assert group.counts == (0, 1)
    group.update(env_ids=[0], translations=[[1, 0, 0]])
    assert group.snapshot()[0].visible


@pytest.mark.parametrize("env_ids", [[0, 0], [2], [-1], [0.5], [True], [[0]], ["0"]])
def test_invalid_environment_selection_is_atomic(env_ids):
    group = batched_group()
    group.update(translations=[[[1, 0, 0]], [[2, 0, 0]]])
    with pytest.raises(ValueError):
        group.update(env_ids=env_ids, translations=[[9, 0, 0]])
    np.testing.assert_allclose(
        [m.position for m in group.snapshot()], [[11, 0, 0], [2, 20, 0]]
    )


def test_batch_shape_validation_and_empty_selection():
    group = batched_group()
    with pytest.raises(ValueError):
        group.update(translations=[[1, 0, 0], [2, 0, 0]])
    group.update(env_ids=[], translations=np.empty((0, 3, 3)))
    assert group.counts == (0, 0)
    with pytest.raises(ValueError):
        group.update(env_ids=[], translations=[[1, 0, 0]])
    group.update(translations=[[[0, 0, 0]], [[1, 0, 0]]])
    with pytest.raises(ValueError):
        group.update(
            translations=[[[9, 0, 0]], [[8, 0, 0]]],
            colors=[[[1, 0, 0, 0.5]], [[0, 0, 0, 2]]],
        )
    np.testing.assert_allclose(
        [m.position for m in group.snapshot()], [[10, 0, 0], [1, 20, 0]]
    )


def test_attachment_rotation_live_refresh_and_detach_world_pose():
    angle = np.sqrt(0.5)
    poses = np.array([[2, 0, 0, 0, 0, angle, angle], [0, 3, 0, 0, 0, 0, 1.0]])
    calls = []

    def resolve(parent, link_name, env_ids):
        calls.append((parent, link_name, tuple(env_ids)))
        return poses[env_ids]

    group = batched_group(pose_resolver=resolve)
    group.update(translations=[[[1, 0, 0]], [[2, 0, 0]]])
    group.attach("robot", link_name="tool", env_ids=[0])
    assert calls[-1] == ("robot", "tool", (0,))
    np.testing.assert_allclose(group.snapshot()[0].position, [12, 1, 0], atol=1e-6)
    np.testing.assert_allclose(group.snapshot()[0].wxyz, [angle, 0, 0, angle])
    poses[0, 0] = 4
    group._refresh_attachments()
    np.testing.assert_allclose(group.snapshot()[0].position, [14, 1, 0], atol=1e-6)
    group.detach(env_ids=[0])
    poses[0, 0] = 9
    group._refresh_attachments()
    np.testing.assert_allclose(group.snapshot()[0].position, [14, 1, 0], atol=1e-6)
    np.testing.assert_allclose(group.snapshot()[1].position, [2, 20, 0])


def test_world_scope_single_batch_rejects_attachment():
    group = MarkerGroup(
        MarkerGroupCfg(
            name="world", scope="world", prototypes={"box": MarkerPrototypeCfg()}
        ),
        num_envs=2,
        origins=[[10, 0, 0], [0, 20, 0]],
    )
    group.update(translations=[[1, 2, 3]])
    assert group.counts == (1,)
    assert group.snapshot()[0].env_id is None
    np.testing.assert_allclose(group.snapshot()[0].position, [1, 2, 3])
    with pytest.raises(ValueError, match="world"):
        group.attach("robot")


def test_attachment_and_selected_updates_roll_back_on_publication_failure():
    fail = False

    def publish(group):
        if fail:
            raise RuntimeError("publish failed")

    group = batched_group(
        on_change=publish,
        pose_resolver=lambda parent, link, ids: np.tile(
            [5, 0, 0, 0, 0, 0, 1], (len(ids), 1)
        ),
    )
    group.update(translations=[[[1, 0, 0]], [[2, 0, 0]]])
    fail = True
    for action in (
        lambda: group.attach("parent"),
        lambda: group.set_visibility(False, env_ids=[1]),
        lambda: group.clear(env_ids=[1]),
    ):
        with pytest.raises(RuntimeError, match="publish failed"):
            action()
        assert group.counts == (1, 1)
        np.testing.assert_allclose(
            [m.position for m in group.snapshot()], [[11, 0, 0], [2, 20, 0]]
        )
        assert all(m.visible for m in group.snapshot())


@pytest.mark.parametrize("env_ids", [0, True, [False, 1], [0, True], np.array(0)])
def test_environment_selection_rejects_scalars_and_mixed_booleans(env_ids):
    group = batched_group()
    with pytest.raises(ValueError):
        group.clear(env_ids=env_ids)


def test_detach_without_preserving_world_pose_and_failed_detach_are_atomic():
    fail = False

    def publish(group):
        if fail:
            raise RuntimeError("publish failed")

    group = batched_group(
        on_change=publish,
        pose_resolver=lambda parent, link, ids: np.tile(
            [5, 0, 0, 0, 0, 0, 1], (len(ids), 1)
        ),
    )
    group.update(translations=[[[1, 0, 0]], [[2, 0, 0]]])
    group.attach("parent")
    fail = True
    with pytest.raises(RuntimeError, match="publish failed"):
        group.detach(env_ids=[0])
    np.testing.assert_allclose(group.snapshot()[0].position, [16, 0, 0])
    fail = False
    group.detach(env_ids=[0], keep_world_pose=False)
    np.testing.assert_allclose(group.snapshot()[0].position, [11, 0, 0])
    np.testing.assert_allclose(group.snapshot()[1].position, [7, 20, 0])


def test_snapshot_never_resolves_live_attachments():
    calls = []

    def resolver(parent, link, ids):
        calls.append(tuple(ids))
        return np.tile([5, 0, 0, 0, 0, 0, 1], (len(ids), 1))

    group = batched_group(pose_resolver=resolver)
    group.update(translations=[[[1, 0, 0]], [[2, 0, 0]]])
    group.attach("parent")
    calls.clear()
    group.snapshot()
    group.snapshot()
    assert calls == []


def test_updates_validate_transforms_without_constructing_render_geometry(monkeypatch):
    import embodichain.lab.visualization.markers.group as group_module

    group = batched_group(
        pose_resolver=lambda parent, link, ids: np.tile(
            [5, 0, 0, 0, 0, 0, 1], (len(ids), 1)
        )
    )

    def unexpected_snapshot(**kwargs):
        raise AssertionError(
            "Updates without a publisher must not allocate render geometry"
        )

    with monkeypatch.context() as patch:
        patch.setattr(group_module, "MeshMarkerOverlay", unexpected_snapshot)
        group.update(translations=[[[1, 0, 0]], [[2, 0, 0]]])
        group.attach("parent", env_ids=[1])
        group._refresh_attachments()
        group.detach(env_ids=[1])
    np.testing.assert_allclose(
        [m.position for m in group.snapshot()], [[11, 0, 0], [7, 20, 0]]
    )


def test_composed_attachment_overflow_is_rejected_before_publication():
    published = []
    group = batched_group(
        on_change=lambda group: published.append(group.count),
        pose_resolver=lambda parent, link, ids: np.tile(
            [3e38, 0, 0, 0, 0, 0, 1], (len(ids), 1)
        ),
    )
    group.update(translations=[[[3e38, 0, 0]], [[0, 0, 0]]])
    with pytest.raises(ValueError, match="World positions"):
        group.attach("parent", env_ids=[0])
    assert published == [2]
    np.testing.assert_allclose(group.snapshot()[0].position, [3e38, 0, 0], rtol=1e-6)
