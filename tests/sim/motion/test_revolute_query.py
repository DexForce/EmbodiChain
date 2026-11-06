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

"""Declared revolute path scope and explicit borrowed-owner lifetimes."""

from __future__ import annotations

from types import SimpleNamespace as NS
from pathlib import Path
import json
import ast

import numpy as np
import pytest
import torch

from embodichain.lab.sim.motion.revolute_query import RevolutePathQueryFactory
from embodichain.toolkits.graspkit.pg_grasp.native_contact import NativeQueryScope
from _revolute_query_fixture import revolute_query_inputs


def _factory(data):
    return RevolutePathQueryFactory(
        data.simulation,
        data.robot,
        binding=data.binding,
        geometry=data.geometry,
        source_seal=data.seal,
        hand_geometry_cfg=data.hand,
        table_uid=data.table.uid,
        collision_link_names=data.names,
    )


def _query(factory, data):
    return factory.prepare(
        scope=NativeQueryScope("fixture-query", 0.0, 0),
        robot_root_pose=data.root[0],
        parent_pose=data.parent[0],
        child_pose=data.child[0],
        measured_joint_position=0.0,
    )


_REFERENCE = json.loads(
    (
        Path(__file__).with_name("data") / "revolute_query_source_reference.json"
    ).read_text()
)


@pytest.mark.parametrize(
    "reference", _REFERENCE["cases"], ids=lambda value: value["case"]
)
def test_revolute_queries_match_frozen_source_reports_and_fk_order(tmp_path, reference):
    kind = {
        "convex_refinement": "separated_projection",
        "rotor_sweep_blocked": "blocked_projection",
    }.get(reference["case"], "box")
    height = 0.05 if reference["case"] == "table_skin_overlap" else 0.8
    data = revolute_query_inputs(tmp_path, mesh_kind=kind, root_height=height)
    factory = _factory(data)
    with _query(factory, data) as query:
        report = query.inspect_free_path(data.positions, data.frames)
        assert report.metadata() == reference["table"]
        assert data.trace == reference["table_fk_trace"]
        # Independent geometry checks retain the table/skin boundary meaning.
        if reference["case"] == "ordinary":
            assert report.minimum_clearance_after_skin_m == pytest.approx(
                0.737, abs=1e-6
            )
        elif reference["case"] == "table_skin_overlap":
            assert report.supported and not report.accepted
            assert report.minimum_clearance_after_skin_m == pytest.approx(-0.003)
        data.trace.clear()
        root = query.qualify_root_dependency(
            data.positions,
            grasp_point=(0.0, 0.0, 0.02),
            axis=torch.tensor(data.binding.axis),
            roll=0.0,
        )
        actual = root.metadata()
        if root.qualified:
            assert root.source_sha256 == data.binding.source_sha256
        actual["source_sha256"] = "<qualified-input-digest>" if root.qualified else None
        assert actual == reference["root_dependency"]
        assert data.trace == reference["root_fk_trace"]
    with pytest.raises(RuntimeError, match="closed"):
        query.inspect_free_path(data.positions, data.frames)
    factory.close()
    assert data.simulation.get_robot(data.robot.uid) is data.robot


@pytest.mark.parametrize(
    "change",
    [
        "owner",
        "source",
        "root",
        "shape_count",
        "pad_body",
        "duplicate_frame",
        "fk_coverage",
    ],
)
def test_changed_or_incomplete_native_inputs_never_return_clearance(tmp_path, change):
    data = revolute_query_inputs(tmp_path)
    factory = _factory(data)
    query = _query(factory, data)
    frames = data.frames
    if change == "owner":
        data.robot.body_data.entities[0] = NS()
    elif change == "source":
        path = tmp_path / "robot.urdf"
        path.write_text(path.read_text() + "\n")
    elif change == "root":
        data.root[0, 0, 3] += 0.1
    elif change == "shape_count":
        data.table_body.get_shape_count = lambda: 2
    elif change == "pad_body":
        data.native_bodies["pad_b"].get_shape_count = lambda: 0
    elif change == "fk_coverage":
        original_fk = data.robot.compute_fk
        data.robot.compute_fk = lambda **kwargs: original_fk(**kwargs)[:, :-1]
    else:
        frames = [*frames, frames[0]]
    result = query.inspect_free_path(data.positions, frames)
    assert not result.supported and not result.accepted and result.reason
    query.close()


def test_explicit_table_uid_and_hand_names_do_not_require_legacy_prefixes(tmp_path):
    data = revolute_query_inputs(tmp_path, table_uid="custom_support")
    with _query(_factory(data), data) as query:
        assert query.inspect_free_path(data.positions, data.frames).accepted


def test_partial_collision_scope_cannot_omit_an_arm_collider(tmp_path):
    data = revolute_query_inputs(tmp_path)
    with pytest.raises(ValueError, match="complete.*subtree"):
        RevolutePathQueryFactory(
            data.simulation,
            data.robot,
            binding=data.binding,
            geometry=data.geometry,
            source_seal=data.seal,
            hand_geometry_cfg=data.hand,
            table_uid=data.table.uid,
            collision_link_names=("clamp_root", "pad_a", "pad_b"),
        )


def test_default_collision_scope_uses_explicit_motion_root_in_urdf_order(tmp_path):
    data = revolute_query_inputs(tmp_path)
    factory = RevolutePathQueryFactory(
        data.simulation,
        data.robot,
        binding=data.binding,
        geometry=data.geometry,
        source_seal=data.seal,
        hand_geometry_cfg=data.hand,
        table_uid=data.table.uid,
    )
    with _query(factory, data) as query:
        result = query.inspect_free_path(data.positions, data.frames)
    assert result.accepted and result.moving_links == data.names


def test_solver_frame_drift_invalidates_captured_geometry_before_fk(tmp_path):
    data = revolute_query_inputs(tmp_path)
    factory = _factory(data)
    query = _query(factory, data)
    data.robot.cfg.solver_cfg["manipulator"].tcp[0, 3] = 0.01
    result = query.inspect_free_path(data.positions, data.frames)
    assert not result.supported and not result.accepted
    assert data.trace == []


def test_non_singleton_or_foreign_robot_fails_before_geometry_capture(
    tmp_path, monkeypatch
):
    data = revolute_query_inputs(tmp_path)
    from embodichain.lab.sim.motion import revolute_query

    monkeypatch.setattr(
        revolute_query,
        "_resolve_hand_frame",
        lambda *args: pytest.fail("Unsupported owner must fail before capture."),
    )
    data.simulation.num_envs = 2
    with pytest.raises(ValueError, match="one Default"):
        _factory(data)
    data.simulation.num_envs = 1
    data.simulation.get_robot = lambda uid: object()
    with pytest.raises(ValueError, match="belong"):
        _factory(data)


@pytest.mark.parametrize(
    "module_name, forbidden",
    [
        (
            "embodichain.lab.sim.motion.revolute_query",
            ("atomic_actions", "task_program", "gen_sim"),
        ),
        (
            "embodichain.lab.sim.motion.collision_query",
            ("atomic_actions", "task_program", "gen_sim", "revolute"),
        ),
        (
            "embodichain.lab.sim.scene.robot_collision",
            ("atomic_actions", "task_program", "gen_sim", "motion", "revolute"),
        ),
        (
            "embodichain.toolkits.graspkit.pg_grasp.contact_metrics",
            ("embodichain.lab", "gen_sim"),
        ),
    ],
)
def test_query_owners_do_not_import_higher_layer_contracts(module_name, forbidden):
    from importlib import import_module

    tree = ast.parse(Path(import_module(module_name).__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
    assert not any(part in module for module in imports for part in forbidden)


@pytest.mark.parametrize(
    "reference",
    json.loads(
        (
            Path(__file__).with_name("data") / "revolute_contact_source_reference.json"
        ).read_text()
    )["cases"],
)
def test_contact_metrics_preserve_source_sampling_frames_and_dtype(tmp_path, reference):
    from scipy.spatial.transform import Rotation

    data = revolute_query_inputs(tmp_path, ring_count=reference["ring_count"])
    data.root[0, :3, :3] = torch.tensor(
        Rotation.from_euler("z", 0.7).as_matrix(), dtype=torch.float32
    )
    data.parent[0, :3, :3] = torch.tensor(
        Rotation.from_euler("z", 0.4).as_matrix(), dtype=torch.float32
    )
    data.parent[0, :2, 3] = torch.tensor([0.1, -0.2])
    factory = _factory(data)
    qpos = torch.tensor([[reference["qpos_x"], 0.0, 0.0]])
    with _query(factory, data) as query:
        near, distance, displacement = query.contact_metrics(
            qpos, torch.zeros_like(qpos)
        )
        expected_near, expected_distance, expected_displacement = reference["metrics"]
        assert near == expected_near
        assert displacement == expected_displacement
        # CPU/Torch distance reductions can differ by one float32 ULP across
        # runners. Keep the frozen reference and bound only that final rounding.
        distance_ulp = float(np.spacing(np.float32(expected_distance)))
        assert distance == pytest.approx(expected_distance, rel=0, abs=distance_ulp)
    factory.close()


def test_contact_metrics_require_original_triangle_qualified_pad_inputs(tmp_path):
    import xml.etree.ElementTree as ET
    import trimesh

    data = revolute_query_inputs(tmp_path)
    path = Path(data.robot.cfg.fpath)
    mesh = tmp_path / "degenerate_pad.ply"
    trimesh.Trimesh(
        vertices=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        faces=[[0, 1, 2]],
        process=False,
    ).export(mesh)
    document = ET.parse(path)
    geometry = document.find("link[@name='pad_a']/collision/geometry")
    geometry.remove(geometry.find("box"))
    ET.SubElement(geometry, "mesh", filename=mesh.name)
    document.write(path)
    # Vertex-envelope capture is legal, but the source pad score has the
    # stronger triangle-input qualification and must not silently inherit it.
    factory = _factory(data)
    before = len(data.trace)
    with _query(factory, data) as query:
        with pytest.raises(ValueError, match="degenerate"):
            query.contact_metrics(torch.zeros(1, 3), torch.zeros(1, 3))
    assert len(data.trace) == before
    factory.close()


def test_static_table_missing_body_data_is_deferred_to_actual_path_query(tmp_path):
    data = revolute_query_inputs(tmp_path)
    original = data.table.body_data
    data.table.body_data = None
    factory = _factory(data)
    with _query(factory, data) as query:
        report = query.inspect_free_path(data.positions, data.frames)
        assert not report.supported and not report.accepted
        assert report.reason
    data.table.body_data = original
    with pytest.raises(ValueError, match="native owner was replaced"):
        _query(factory, data)
    factory.close()


@pytest.mark.parametrize(
    "root_height, accepted, clearance", [(0.8, True, 0.737), (0.05, False, -0.003)]
)
def test_shared_support_path_needs_neither_hand_nor_articulated_target(
    tmp_path, root_height, accepted, clearance
):
    from embodichain.lab.sim.motion.collision_query import RobotPathQuery
    from embodichain.lab.sim.scene.robot_collision import (
        RobotCollisionInputs,
        SupportCollisionInputs,
    )

    data = revolute_query_inputs(tmp_path, root_height=root_height)
    del data.simulation.get_articulation
    del data.robot.cfg.control_parts["clamp"]
    geometry = RobotCollisionInputs(
        data.simulation, data.robot, motion_part="manipulator"
    )
    support = SupportCollisionInputs(data.simulation, data.table.uid)
    try:
        with RobotPathQuery(
            geometry,
            scope=NativeQueryScope("generic-support", 0.0, 0),
            robot_root_pose=data.root[0],
        ) as query:
            report = query.inspect_support_path(support, data.positions, data.frames)
        assert report.supported and report.accepted is accepted
        # Separated: 0.8 - 0.01 m link - 0.05 m support - 0.003 m skins.
        # Overlapping boxes have zero separation before subtracting the skins.
        assert report.minimum_clearance_after_skin_m == pytest.approx(
            clearance, abs=1e-6
        )
        with pytest.raises(RuntimeError, match="closed"):
            query.inspect_support_path(support, data.positions, data.frames)
        # Query close borrows rather than consumes the reusable captures.
        geometry.assert_current()
        support.assert_current()
    finally:
        geometry.close()
        support.close()
    assert data.simulation.get_robot(data.robot.uid) is data.robot


@pytest.mark.parametrize("change", ["source", "support_owner"])
def test_shared_support_path_rejects_invalid_inputs_before_fk(tmp_path, change):
    from copy import copy
    from embodichain.lab.sim.motion.collision_query import RobotPathQuery
    from embodichain.lab.sim.scene.robot_collision import (
        RobotCollisionInputs,
        SupportCollisionInputs,
    )

    data = revolute_query_inputs(tmp_path)
    geometry = RobotCollisionInputs(
        data.simulation, data.robot, motion_part="manipulator"
    )
    support = SupportCollisionInputs(data.simulation, data.table.uid)
    query = RobotPathQuery(
        geometry,
        scope=NativeQueryScope("changed-source", 0.0, 0),
        robot_root_pose=data.root[0],
    )
    try:
        if change == "source":
            source = Path(data.robot.cfg.fpath)
            source.write_text(source.read_text() + "\n")
        else:
            support.close()
            support = SupportCollisionInputs(copy(data.simulation), data.table.uid)
        report = query.inspect_support_path(support, data.positions, data.frames)
        assert not report.supported and not report.accepted and report.reason
        assert data.trace == []
    finally:
        query.close()
        geometry.close()
        support.close()
