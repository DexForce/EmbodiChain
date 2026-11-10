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
from embodichain.lab.sim.scene._newton_queries import _ConvexCollection
from embodichain.lab.sim.scene import _newton_queries, _collision_backends

from dataclasses import replace
import ast
import json
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from embodichain.lab.sim.scene import native_hand as capture
from embodichain.lab.sim.objects.backends import collision as shape_decoder
from embodichain.lab.sim.atomic_skills import (
    SkillPlanningServices,
    GraspOpeningTarget,
)
from embodichain.toolkits.graspkit.pg_grasp.native_contact import NativeQueryScope

from ._native_capture_fixture import (
    Box,
    Convex,
    Body,
    Robot,
    Rigid,
    setup_case,
    parts_record,
)
from ._native_hand_fixture import (
    decision_digest,
    recording_manager,
    stable_report,
    source_policy,
)


@pytest.fixture
def ports(tmp_path, monkeypatch):
    pytest.importorskip(
        "newton",
        reason="Native capture geometry is unqualified without Newton runtime.",
    )
    monkeypatch.setattr(shape_decoder, "BoxGeometry", Box)
    monkeypatch.setattr(shape_decoder, "ConvexMeshGeometry", Convex)
    return setup_case(tmp_path)


def _services(robot, cfg):
    return SkillPlanningServices(
        NS(robot=robot, device="cpu"),
        hand_geometry_cfgs={"fixture_grasp": cfg},
    )


def _prepare(ports, **arguments):
    robot, simulation, cfg, *_ = ports
    return _services(robot, cfg).prepare_grasp_query(
        "fixture_grasp", simulation, **arguments
    )


def _arguments(ports):
    robot, simulation, cfg, opened, closed = ports
    return dict(
        source_uid="source",
        open_qpos=opened,
        grasp_qpos=closed,
        observation_qpos=robot.qpos,
        object_pose=simulation.rigids["source"].pose,
        scope=NativeQueryScope("fresh-prepare", 2.5, 4),
        policy=source_policy(2),
        report_metadata={"revision": 2},
        identity={"object_id": "fixture_object"},
    )


def test_cooked_capture_full_stroke_and_newton_preserve_pinned_source_decisions(
    ports, monkeypatch
):
    expected = json.loads(
        Path(__file__).with_name("fixtures").joinpath("native_hand_f1.json").read_text()
    )["capture"]
    robot, simulation, cfg, opened, closed = ports
    before = (robot.qpos.clone(), opened.clone(), closed.clone(), cfg.copy())
    manager, trace = recording_manager(_ConvexCollection)
    monkeypatch.setattr(_newton_queries, "_ConvexCollection", manager)
    with _prepare(ports, **_arguments(ports)) as query:
        assert parts_record(query) == expected["parts"]
        assert robot.calls == expected["fk_calls"]
        assert query.identity == expected["identity"]
        assert decision_digest(query.inspect(np.eye(4))) == expected["result_sha256"]
        assert (
            decision_digest(stable_report(query.report())) == expected["report_sha256"]
        )
        assert decision_digest(trace) == expected["trace_sha256"]
        assert robot.reads == 0 and expected["source_state_reads"] == 1
    torch.testing.assert_close(robot.qpos, before[0])
    torch.testing.assert_close(opened, before[1])
    torch.testing.assert_close(closed, before[2])
    assert cfg.open_qpos == before[3].open_qpos == (99, -99)


def test_opening_scope_scene_owner_and_query_lifetimes_are_shared(ports):
    robot, simulation, *_ = ports
    args = _arguments(ports)
    args["policy"] = source_policy(4)
    target_pose = simulation.rigids["source"].pose[0].numpy().copy()
    target_pose[2, 3] = 0.2
    opening = GraspOpeningTarget(
        "support",
        simulation.rigids["support"].pose[0].numpy(),
        target_pose,
        args["scope"],
        {"reference_entity_id": "fixture_support"},
    )
    args["opening"] = opening
    args["include_rigid_scene"] = True
    with _prepare(ports, **args) as query:
        opening_query = query.opening_screen
        assert opening_query is not None and opening_query.scope == query.scope
        rest = query.rigid_scene_screen
        assert rest is not None
        assert rest.scope == query.scope
        assert rest.inspect(simulation.rigids["source"].pose[0].numpy(), 2)["valid"]
        candidates = simulation.rigids["source"].pose[:, None]
        assert query.mask(candidates, simulation.rigids["source"].pose).tolist() == [
            [True]
        ]
    assert rest.closed and opening_query.closed
    assert (
        query.report()["opening"]["identity"]["reference_entity_id"]
        == "fixture_support"
    )
    with pytest.raises(RuntimeError, match="closed"):
        rest.inspect(np.eye(4), 2)


def test_same_uid_cannot_substitute_robot_or_native_source(ports):
    robot, simulation, cfg, opened, closed = ports
    args = _arguments(ports)
    foreign = Robot(Path(robot.cfg.fpath))
    with pytest.raises(ValueError, match="exact simulation-owned"):
        with _services(foreign, cfg).prepare_grasp_query(
            "fixture_grasp", simulation, **args
        ):
            pass
    geometry = capture.capture_native_hand_geometry(
        robot,
        simulation,
        source_uid="source",
        hand_geometry_cfg=cfg,
        open_qpos=opened,
        grasp_qpos=closed,
        observation_qpos=robot.qpos,
        object_pose=simulation.rigids["source"].pose,
        include_contact_offsets=True,
    )
    original = simulation.rigids["source"]
    simulation.handles[original.path] = Rigid("source", original.body, (0.5, 0, 0.1))
    with pytest.raises(ValueError, match="replaced"):
        capture.capture_native_rigid_obstacles(simulation, geometry)


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
        args["policy"] = source_policy(4)
        args["opening"] = GraspOpeningTarget(
            "support", np.eye(4), np.eye(4), replace(args["scope"], scene_version=5)
        )
    with pytest.raises(ValueError):
        with _prepare(ports, **args):
            pass
    assert robot.calls == []


def test_missing_newton_is_preflighted_before_native_capture(monkeypatch):
    failure = Mock(side_effect=ImportError("Newton unavailable"))
    monkeypatch.setattr(_collision_backends, "_require_newton", failure)
    services = SkillPlanningServices(NS(robot=None, device="cpu"))
    with pytest.raises(ImportError, match="Newton"):
        with services.prepare_grasp_query(
            "missing",
            None,
            source_uid="source",
            open_qpos=None,
            grasp_qpos=None,
            observation_qpos=None,
            object_pose=None,
            scope=NativeQueryScope("preflight", 0.0, 0),
            policy=source_policy(2),
        ):
            pass
    failure.assert_called_once_with()


def test_planning_services_close_query_when_consumer_raises(ports):
    with pytest.raises(RuntimeError, match="consumer cancelled"):
        with _prepare(ports, **_arguments(ports)) as query:
            raise RuntimeError("consumer cancelled")
    assert query.closed


def test_planning_services_require_declared_calibration_before_capture(ports):
    robot, simulation, *_ = ports
    services = SkillPlanningServices(NS(robot=robot, device="cpu"))
    with pytest.raises(KeyError, match="calibration"):
        with services.prepare_grasp_query(
            "undeclared", simulation, **_arguments(ports)
        ):
            pass
    assert robot.calls == []


def test_cooked_convex_scale_and_local_pose_are_applied_once(monkeypatch):
    monkeypatch.setattr(shape_decoder, "ConvexMeshGeometry", Convex)
    monkeypatch.setattr(shape_decoder, "BoxGeometry", Box)
    pose = np.array(
        [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float
    )
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    source = Convex(vertices, faces, np.array([2.0, 3.0, 4.0]), pose)
    snapshot = shape_decoder.capture_physical_collision_geometry(
        Body((source,)),
        source_path="fixture",
        include_contact_offsets=True,
    )
    meshes = snapshot.meshes
    assert snapshot.rest_offset == 0.0002 and snapshot.contact_offset == 0.003
    np.testing.assert_array_equal(
        meshes[0].vertices, [[1, 2, 3], [1, 4, 3], [-2, 2, 3], [1, 2, 7]]
    )
    np.testing.assert_array_equal(meshes[0].faces, faces)
    vertices[:] = 99
    assert meshes[0].vertices.max() == 7
    assert (
        shape_decoder.capture_physical_collision_geometry(
            Body((source,), enabled=False),
            source_path="fixture",
        ).meshes
        == ()
    )
    with pytest.raises(ValueError, match="convex or box"):
        shape_decoder.capture_physical_collision_geometry(
            Body((object(),)), source_path="fixture"
        )
    # Public descriptor decoding may reshape flat SDK arrays. The native source
    # contract rejects them instead of silently changing geometry availability.
    malformed = Convex(vertices.ravel(), faces, np.ones(3), pose)
    with pytest.raises(ValueError, match="topology"):
        shape_decoder.capture_physical_collision_geometry(
            Body((malformed,)), source_path="fixture"
        )


def test_native_capture_and_graspkit_respect_owner_dependencies():
    root = Path(__file__).parents[3] / "embodichain"
    sources = [
        (
            root / "lab/sim/scene/native_hand.py",
            (
                "embodichain.gen_sim",
                "embodichain.lab.task_program",
                "embodichain.lab.sim.atomic_skills",
            ),
        ),
        (
            root / "toolkits/graspkit/pg_grasp/native_contact.py",
            (
                "embodichain.gen_sim",
                "embodichain.lab",
            ),
        ),
    ]
    forbidden = []
    for source, prefixes in sources:
        for node in ast.walk(ast.parse(source.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert (
                    node.level == 0
                ), "Use absolute imports at these owner boundaries."
                modules = [node.module or ""]
            forbidden.extend(
                f"{source.name}:{module}"
                for module in modules
                if any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in prefixes
                )
            )
    assert forbidden == []
