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

"""Native capture contracts using SDK-shaped ports, without launching simulation."""

from __future__ import annotations
from embodichain.lab.sim.scene import native_hand, _native_hand_capture as capture
from embodichain.lab.sim.scene._newton_queries import _ConvexCollection

from copy import deepcopy
from dataclasses import replace
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import dexsim.engine
import numpy as np
import pytest
import torch
import trimesh

from embodichain.lab.sim.scene import _native_hand_capture as capture
from embodichain.lab.sim.scene.native_hand import (
    NativeOpeningTarget,
    NativeQueryScope,
    build_native_closing_query,
    build_native_rigid_rest_scene,
)
from ._native_capture_fixture import (
    Box,
    Convex,
    Body,
    Robot,
    Rigid,
    setup_case,
    parts_record,
)
from ._native_hand_fixture import decisions, recording_manager, stable_report


@pytest.fixture
def ports(tmp_path, monkeypatch):
    pytest.importorskip(
        "newton",
        reason="Native capture geometry is unqualified without Newton runtime.",
    )
    monkeypatch.setattr(dexsim.engine, "BoxGeometry", Box)
    monkeypatch.setattr(dexsim.engine, "ConvexMeshGeometry", Convex)
    return setup_case(tmp_path)


def _arguments(ports):
    robot, simulation, cfg, opened, closed = ports
    return dict(
        source_uid="source",
        hand_geometry_cfg=cfg,
        open_qpos=opened,
        grasp_qpos=closed,
        observation_qpos=robot.qpos,
        object_pose=simulation.rigids["source"].pose,
        scope=NativeQueryScope("fresh-prepare", 2.5, 4),
        revision=2,
        identity={"object_id": "fixture_object"},
    )


def test_cooked_capture_full_stroke_and_newton_preserve_pinned_source_decisions(
    ports, monkeypatch
):
    expected = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("native_hand_f1_capture.json")
        .read_text()
    )
    robot, simulation, cfg, opened, closed = ports
    before = (robot.qpos.clone(), opened.clone(), closed.clone(), cfg.copy())
    manager, trace = recording_manager(_ConvexCollection)
    monkeypatch.setattr(native_hand, "_ConvexCollection", manager)
    with build_native_closing_query(robot, simulation, **_arguments(ports)) as query:
        assert parts_record(query) == expected["parts"]
        assert robot.calls == expected["fk_calls"]
        assert query.identity == expected["identity"]
        assert decisions(query.inspect(np.eye(4))) == decisions(expected["result"])
        assert decisions(stable_report(query.report())) == decisions(expected["report"])
        assert decisions(trace) == decisions(expected["trace"])
        assert robot.reads == 0 and expected["source_state_reads"] == 1
    torch.testing.assert_close(robot.qpos, before[0])
    torch.testing.assert_close(opened, before[1])
    torch.testing.assert_close(closed, before[2])
    assert cfg.open_qpos == before[3].open_qpos == (99, -99)


def test_opening_scope_scene_owner_and_query_lifetimes_are_shared(ports):
    robot, simulation, *_ = ports
    args = _arguments(ports)
    args["revision"] = 4
    target_pose = simulation.rigids["source"].pose[0].numpy().copy()
    target_pose[2, 3] = 0.2
    opening = NativeOpeningTarget(
        "support",
        simulation.rigids["support"].pose[0].numpy(),
        target_pose,
        args["scope"],
        {"reference_entity_id": "fixture_support"},
    )
    args["opening"] = opening
    with build_native_closing_query(robot, simulation, **args) as query:
        opening_query = query.opening_screen
        assert opening_query is not None and opening_query.scope == query.scope
        rest = build_native_rigid_rest_scene(simulation, query)
        assert rest.scope == query.scope
        assert rest.inspect(simulation.rigids["source"].pose[0].numpy(), 2)["valid"]
        candidates = simulation.rigids["source"].pose[:, None]
        assert query.mask(candidates, simulation.rigids["source"].pose).tolist() == [
            [True]
        ]
        with pytest.raises(RuntimeError, match="already owns"):
            build_native_rigid_rest_scene(simulation, query)
    assert rest.closed and opening_query.closed
    assert (
        query.report()["opening"]["identity"]["reference_entity_id"]
        == "fixture_support"
    )
    with pytest.raises(RuntimeError, match="closed"):
        rest.inspect(np.eye(4), 2)


def test_same_uid_cannot_substitute_robot_or_native_source(ports):
    robot, simulation, *_ = ports
    args = _arguments(ports)
    foreign = Robot(Path(robot.cfg.fpath))
    with pytest.raises(ValueError, match="exact simulation-owned"):
        build_native_closing_query(foreign, simulation, **args)
    with build_native_closing_query(robot, simulation, **args) as query:
        original = simulation.rigids["source"]
        replacement = Rigid("source", original.body, (0.5, 0, 0.1))
        simulation.handles[original.path] = replacement
        with pytest.raises(ValueError, match="replaced"):
            build_native_rigid_rest_scene(simulation, query)


@pytest.mark.parametrize(
    "change", ["batch", "backend", "stroke", "observation", "opening_scope"]
)
def test_unsupported_native_inputs_fail_before_fk(ports, change):
    robot, simulation, *_ = ports
    args = _arguments(ports)
    if change == "batch":
        simulation.num_envs = 2
    elif change == "backend":
        simulation.is_default_backend = False
    elif change == "stroke":
        args["grasp_qpos"] = torch.tensor([[0.1]])
    elif change == "observation":
        args["observation_qpos"] = torch.zeros(2, 3)
    elif change == "opening_scope":
        args["revision"] = 4
        args["opening"] = NativeOpeningTarget(
            "support", np.eye(4), np.eye(4), replace(args["scope"], scene_version=5)
        )
    with pytest.raises(ValueError):
        build_native_closing_query(robot, simulation, **args)
    assert robot.calls == []


def test_missing_newton_is_preflighted_before_native_capture(monkeypatch):
    failure = Mock(side_effect=ImportError("Newton unavailable"))
    monkeypatch.setattr(capture, "_require_newton", failure)
    with pytest.raises(ImportError, match="Newton"):
        build_native_closing_query(
            None,
            None,
            source_uid="source",
            hand_geometry_cfg=None,
            open_qpos=None,
            grasp_qpos=None,
            observation_qpos=None,
            object_pose=None,
            scope=NativeQueryScope("preflight", 0.0, 0),
            revision=2,
        )
    failure.assert_called_once_with()


def test_cooked_convex_scale_and_local_pose_are_applied_once(monkeypatch):
    monkeypatch.setattr(dexsim.engine, "ConvexMeshGeometry", Convex)
    monkeypatch.setattr(dexsim.engine, "BoxGeometry", Box)
    pose = np.array(
        [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float
    )
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    source = Convex(vertices, faces, np.array([2.0, 3.0, 4.0]), pose)
    meshes = capture._native_meshes(Body((source,)))
    np.testing.assert_array_equal(
        meshes[0].vertices, [[1, 2, 3], [1, 4, 3], [-2, 2, 3], [1, 2, 7]]
    )
    np.testing.assert_array_equal(meshes[0].faces, faces)
    vertices[:] = 99
    assert meshes[0].vertices.max() == 7
    assert capture._native_meshes(Body((source,), enabled=False)) == ()
    with pytest.raises(ValueError, match="convex or box"):
        capture._native_meshes(Body((object(),)))


def test_native_scene_queries_do_not_import_task_or_atomic_request_layers():
    directory = Path(__file__).parents[3] / "embodichain/lab/sim/scene"
    forbidden = []
    for filename in ("native_hand.py", "_native_hand_capture.py"):
        tree = ast.parse((directory / filename).read_text())
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    parent = "embodichain.lab.sim.scene".split(".")
                    module = ".".join([*parent[: len(parent) - node.level + 1], module])
                modules = [module]
            forbidden.extend(
                f"{filename}:{module}"
                for module in modules
                if any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in (
                        "embodichain.gen_sim",
                        "embodichain.lab.task_program",
                        "embodichain.lab.sim.atomic_actions",
                    )
                )
            )
    assert forbidden == []
