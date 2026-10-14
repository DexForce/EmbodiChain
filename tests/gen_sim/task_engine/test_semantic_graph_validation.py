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

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.task_spec import Predicate
from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
from embodichain.gen_sim.task_engine.semantic_graph_validation import (
    validate_graph_coverage,
    verify_graph_contract,
    write_graph_contract,
)
from embodichain.utils.utility import save_config, load_config
from embodichain.gen_sim.task_engine.orchestration.artifacts import ArtifactTransaction
from .test_task_spec_generation import candidate, selector, step


def _plan(*, count: int = 0):
    source = candidate(step("E2", "cans", object=selector("cans", count=count)))
    bindings = {
        "schema_version": "action_engine_role_bindings_v1",
        "task_id": source["draft"]["task_id"],
        "candidate_id": source["candidate_id"],
        "reference_bindings": {"step_01.object": ["a", "b"] if count else ["a"]},
        "role_bindings": {},
    }
    return SemanticTaskPlanner().plan_spec(
        source,
        bindings,
        [
            {
                "runtime_uid": uid,
                "init_pos": [0.1, -0.2, 0.85],
                "init_rot": [0.0, 0.0, 0.0],
            }
            for uid in ("a", "b")
        ],
    )


def _coverage(plan, **changes):
    arguments = {
        "graph": plan.graph,
        "task_spec": plan.task_spec,
        "candidate": plan.candidate,
        "role_bindings": plan.role_bindings,
        "lineage": plan.lineage,
    }
    arguments.update(changes)
    return validate_graph_coverage(**arguments)


def test_expansion_preserves_source_spec_and_covers_every_set_member() -> None:
    plan = _plan(count=2)
    report = _coverage(plan)
    assert report["status"] == "covered"
    assert len(plan.task_spec.milestones) == 1
    assert len(plan.graph["task_groups"]) == 2
    assert {m["object_uid"] for m in plan.lineage["step_01"]} == {"a", "b"}
    assert len(report["goals"][0]["groups"]) == 2
    assert report["runtime_evidence_status"] == "not_run"


def test_repeated_side_placements_preserve_target_part_for_every_member() -> None:
    source = candidate(
        step(
            "E1",
            "cups",
            object=selector("cups", count=2),
            target=selector("drawer"),
            relation="left_of",
            required_arm="left_arm",
        )
    )
    bindings = {
        "schema_version": "action_engine_role_bindings_v1",
        "task_id": source["draft"]["task_id"],
        "candidate_id": source["candidate_id"],
        "reference_bindings": {
            "step_01.object": ["a", "b"],
            "step_01.target": ["drawer"],
        },
        "role_bindings": {"step_01.target": "part_fixture"},
    }
    plan = SemanticTaskPlanner().plan_spec(
        source,
        bindings,
        [
            {"runtime_uid": uid, "init_pos": [0.1, -0.2, 0.8], "init_rot": [0.0] * 3}
            for uid in ("a", "b", "drawer")
        ],
    )
    calls = [
        n["call"]
        for n in plan.graph["nodes"]
        if n["call"].get("call_id") == "simulation.place_relative"
    ]
    assert len(calls) == 2
    assert all(c["arguments"]["reference_part"] == "part_fixture" for c in calls)
    assert _coverage(plan)["status"] == "covered"


@pytest.mark.parametrize(
    "damage",
    [
        "missing_member",
        "duplicate_member",
        "wrong_uid",
        "wrong_goal",
        "wrong_call",
        "wrong_arm",
    ],
)
def test_declaration_claims_cannot_hide_missing_or_wrong_coverage(damage: str) -> None:
    plan = _plan(count=2)
    changes = {}
    lineage = deepcopy(plan.lineage)
    if damage == "missing_member":
        lineage["step_01"].pop()
    elif damage == "duplicate_member":
        lineage["step_01"][1] = deepcopy(lineage["step_01"][0])
    elif damage == "wrong_uid":
        lineage["step_01"][0]["object_uid"] = "other"
    elif damage == "wrong_goal":
        changes["task_spec"] = replace(
            plan.task_spec,
            goal=(
                *plan.task_spec.goal,
                Predicate("pressed", {"subject": "role_001"}),
            ),
        )
    else:
        graph = deepcopy(plan.graph)
        if damage == "wrong_call":
            for node in graph["nodes"]:
                if node["call"].get("call_id") == "gen_sim.align_held":
                    node["call"]["call_id"] = "simulation.park"
                    node["call"]["arguments"] = {}
        else:
            from embodichain.gen_sim.task_engine.agent import (
                derive_scene_request,
                derive_success_spec,
            )
            from embodichain.gen_sim.task_engine.contracts import canonical_hash
            from embodichain.gen_sim.task_engine.task_spec import (
                task_spec_from_candidate,
            )

            source = deepcopy(plan.candidate)
            source["draft"]["steps"][0]["required_arm"] = "right_arm"
            source["scene_request"] = derive_scene_request(source["draft"])
            source["success_spec"] = derive_success_spec(source["draft"])
            source["semantic_hash"] = canonical_hash(source["draft"]["steps"])
            changes.update(candidate=source, task_spec=task_spec_from_candidate(source))
            for node in graph["nodes"]:
                node["call"].setdefault("resources", {})["primary"] = "left"
        changes["graph"] = graph
    changes["lineage"] = lineage
    assert _coverage(plan, **changes)["status"] == "uncovered"


def _contract(tmp_path: Path):
    plan = _plan()
    for name in (
        "task_program/program.yaml",
        "task_program/integration.yaml",
        "task_program/constraints.json",
        "components/scene.yaml",
        "components/embodiment.yaml",
        "components/execution_policy.yaml",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        save_config(path, {"fixture": name})
    write_graph_contract(
        tmp_path,
        plan.graph,
        plan.task_spec,
        plan.candidate,
        plan.role_bindings,
        plan.lineage,
    )
    return plan


@pytest.mark.parametrize(
    "damage",
    ["spec_metadata", "observation", "program", "scene", "bindings", "coverage"],
)
def test_full_identity_and_recomputation_reject_stale_companions(
    tmp_path: Path, damage: str
) -> None:
    plan = _contract(tmp_path)
    assert (
        verify_graph_contract(tmp_path, plan.graph)["coverage"]["status"] == "covered"
    )
    spec_path = tmp_path / "task_spec.json"
    if damage in {"spec_metadata", "observation"}:
        spec = load_config(spec_path)
        if damage == "spec_metadata":
            spec["metadata"]["candidate_id"] = "other"
        else:
            spec["goal"][0]["observation"] = "other_measurement"
        save_config(spec_path, spec)
    elif damage in {"program", "scene"}:
        save_config(
            tmp_path
            / (
                "task_program/program.yaml"
                if damage == "program"
                else "components/scene.yaml"
            ),
            {"changed": True},
        )
    else:
        path = tmp_path / "semantic_graph_validation.json"
        envelope = load_config(path)
        if damage == "bindings":
            envelope["source"]["role_bindings"]["reference_bindings"][
                "step_01.object"
            ] = ["other"]
        else:
            envelope["coverage"]["goals"][0]["groups"] = []
        save_config(path, envelope)
    with pytest.raises(ValueError, match="identity|coverage|witness"):
        verify_graph_contract(tmp_path, plan.graph)


def test_absent_legacy_companions_do_not_rewrite_old_bundle(tmp_path: Path) -> None:
    assert verify_graph_contract(tmp_path, _plan().graph) is None
    assert list(tmp_path.iterdir()) == []


def _relocatable_contract(root: Path):
    plan = _contract(root)
    save_config(root / "semantic_task_graph.json", plan.graph)
    mesh = root / "mesh_assets/normalized/a.glb"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"publication fixture")
    save_config(
        root / "components/scene.yaml",
        {
            "simulation": {
                "rigid_object": [
                    {
                        "uid": "a",
                        "shape": {"fpath": mesh.as_posix()},
                        "init_pos": [0.0, 0.0, 0.8],
                    }
                ]
            }
        },
    )
    write_graph_contract(
        root,
        plan.graph,
        plan.task_spec,
        plan.candidate,
        plan.role_bindings,
        plan.lineage,
    )
    return plan


def test_nested_publication_keeps_full_contract_and_html_identity(
    tmp_path: Path,
) -> None:
    from .test_semantic_graph_html import _Scripts
    import json

    with ArtifactTransaction(tmp_path / "run") as outer:
        with ArtifactTransaction(
            outer.staging_dir / "attempts/scene_0001/bundle"
        ) as inner:
            plan = _relocatable_contract(inner.staging_dir)
            assert (
                verify_graph_contract(inner.staging_dir, plan.graph)["coverage"][
                    "status"
                ]
                == "covered"
            )
            bundle = inner.commit()
        assert (
            verify_graph_contract(bundle, plan.graph)["coverage"]["status"] == "covered"
        )
        published = outer.commit()
    root = published / "attempts/scene_0001/bundle"
    envelope = verify_graph_contract(root, plan.graph)
    scene = load_config(root / "components/scene.yaml")
    assert (
        scene["simulation"]["rigid_object"][0]["shape"]["fpath"]
        == (root / "mesh_assets/normalized/a.glb").as_posix()
    )
    data = json.loads(
        _Scripts((root / "semantic_graph.html").read_text(encoding="utf-8")).scripts[0][
            1
        ]
    )
    assert data["validation"] == envelope
    assert ".staging-" not in json.dumps(data)
    scene["simulation"]["rigid_object"][0]["init_pos"][2] = 5.0
    save_config(root / "components/scene.yaml", scene)
    with pytest.raises(ValueError, match="identity"):
        verify_graph_contract(root, plan.graph)


@pytest.mark.parametrize("damage", ["scene", "program", "missing_companion"])
def test_publication_cannot_reseal_tampered_contract(
    tmp_path: Path, damage: str
) -> None:
    destination = tmp_path / "published"
    with ArtifactTransaction(destination) as transaction:
        root = transaction.staging_dir
        _relocatable_contract(root)
        if damage == "missing_companion":
            (root / "task_spec.json").unlink()
        else:
            save_config(
                root
                / (
                    "components/scene.yaml"
                    if damage == "scene"
                    else "task_program/program.yaml"
                ),
                {"tampered": True},
            )
        with pytest.raises(ValueError, match="identity|incomplete"):
            transaction.commit()
    assert not destination.exists()


def _inside_plan(kind: str, part: str | None):
    source = candidate(
        step(
            kind,
            "can",
            target=selector("box"),
            relation="inside",
            terminal_behavior="place" if kind == "E4" else "none",
        )
    )
    bindings = {
        "schema_version": "action_engine_role_bindings_v1",
        "task_id": "demo",
        "candidate_id": source["candidate_id"],
        "reference_bindings": {"step_01.object": ["a"], "step_01.target": ["box"]},
        "role_bindings": {},
    }
    if part:
        bindings["role_bindings"]["step_01.target"] = part
    return SemanticTaskPlanner().plan_spec(
        source,
        bindings,
        [
            {
                "runtime_uid": uid,
                "init_pos": [0.1, -0.2, 0.85],
                "init_rot": [0.0, 0.0, 0.0],
            }
            for uid in ("a", "box")
        ],
    )


@pytest.mark.parametrize(
    "kind,part", [("E1", None), ("E4", None), ("E1", "part_0"), ("E4", "part_0")]
)
def test_inside_uses_existing_builtin_affordance_protocol(
    kind: str, part: str | None
) -> None:
    plan = _inside_plan(kind, part)
    inside = next(
        node["call"]["inside"]
        for node in plan.graph["nodes"]
        if "inside" in node["call"]
    )
    expected = "inside__box__a" + (f"__{part}" if part else "")
    assert inside == expected
    assert _coverage(plan)["status"] == "covered"
    graph = deepcopy(plan.graph)
    for node in graph["nodes"]:
        if node["call"].get("inside"):
            node["call"]["inside"] = "inside__wrong__a"
    assert _coverage(plan, graph=graph)["status"] == "uncovered"


def _mock_drawer_parts(monkeypatch: pytest.MonkeyPatch, *parts: str):
    from embodichain.gen_sim.task_engine._task_program import drawer_binding
    from embodichain.gen_sim.task_engine._task_program.articulation_binding import (
        PrismaticBinding,
    )

    bindings = {
        part: PrismaticBinding(
            "box",
            f"slide_{part}",
            f"drawer_{part}",
            "base",
            "/fixture/handle",
            "a" * 64,
            1.0,
            (1.0, 0.0, 0.0),
            (0.0, 0.2),
            part_id=part,
        )
        for part in parts
    }
    monkeypatch.setattr(drawer_binding, "graph_bindings", lambda *_: bindings)
    monkeypatch.setattr(
        drawer_binding, "discover_prismatic_parts", lambda *_: tuple(bindings.values())
    )
    monkeypatch.setattr(
        drawer_binding, "inspect_prismatic_part", lambda _, part: bindings[part]
    )
    return SimpleNamespace(articulations=({"uid": "box"},))


@pytest.mark.parametrize("kind", ["E1", "E4"])
def test_qualified_inside_coverage_survives_drawer_rewrite(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from embodichain.gen_sim.task_engine._task_program.drawer_binding import (
        prepare_drawer_graph,
    )

    plan = _inside_plan(kind, "part_0")
    scene = _mock_drawer_parts(monkeypatch, "part_0")
    graph = prepare_drawer_graph(plan.graph, scene)
    assert any(
        node["call"].get("call_id") == "gen_sim.drawer_transport"
        for node in graph["nodes"]
    )
    assert _coverage(plan, graph=graph)["status"] == "covered"
    assert prepare_drawer_graph(graph, scene) == graph


@pytest.mark.parametrize("kind", ["E1", "E4"])
@pytest.mark.parametrize("damage", ["wrong_part", "missing_part", "unbound_part"])
def test_inside_coverage_rejects_missing_or_foreign_part(
    kind: str, damage: str
) -> None:
    plan = _inside_plan(kind, "part_0")
    graph, bindings = deepcopy(plan.graph), deepcopy(plan.role_bindings)
    if damage == "unbound_part":
        bindings["role_bindings"].pop("step_01.target")
    else:
        inside = "inside__box__a" + ("__part_other" if damage == "wrong_part" else "")
        for node in graph["nodes"]:
            if "inside" in node["call"]:
                node["call"]["inside"] = inside
    report = _coverage(plan, graph=graph, role_bindings=bindings)
    assert {"code": "required_relation_not_covered", "owner": "step_01"} in report[
        "issues"
    ]


@pytest.mark.parametrize("kind", ["E1", "E4"])
def test_unqualified_inside_cannot_default_to_first_drawer(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from embodichain.gen_sim.task_engine._task_program.drawer_binding import (
        prepare_drawer_graph,
    )

    plan = _inside_plan(kind, "part_0")
    graph = deepcopy(plan.graph)
    for node in graph["nodes"]:
        if "inside" in node["call"]:
            node["call"]["inside"] = "inside__box__a"
    scene = _mock_drawer_parts(monkeypatch, "part_0", "part_1")
    with pytest.raises(ValueError, match="one explicit E6 part"):
        prepare_drawer_graph(graph, scene)


def test_pour_target_and_primary_arm_are_not_covered_by_other_calls() -> None:
    source = candidate(
        step(
            "E3",
            "vessel",
            target=selector("cup"),
            relation="above",
            required_arm="right_arm",
        )
    )
    bindings = {
        "schema_version": "action_engine_role_bindings_v1",
        "task_id": "demo",
        "candidate_id": source["candidate_id"],
        "reference_bindings": {"step_01.object": ["a"], "step_01.target": ["cup"]},
        "role_bindings": {},
    }
    plan = SemanticTaskPlanner().plan_spec(
        source,
        bindings,
        [
            {
                "runtime_uid": uid,
                "init_pos": [0.1, -0.2, 0.85],
                "init_rot": [0.0, 0.0, 0.0],
            }
            for uid in ("a", "cup")
        ],
    )
    for damage in ("target", "resource"):
        graph = deepcopy(plan.graph)
        for node in graph["nodes"]:
            if node["call"].get("call_id") == "simulation.move_held_object" and node[
                "call"
            ]["arguments"].get("reference"):
                if damage == "target":
                    node["call"]["arguments"]["reference"] = "other"
                else:
                    node["call"]["resources"]["primary"] = "left"
        assert _coverage(plan, graph=graph)["status"] == "uncovered"


def test_aliased_rigid_roles_cannot_declare_conflicting_final_positions() -> None:
    source = candidate(
        step("E1", "can", target=selector("box"), relation="left_of"),
        step("E1", "red can", target=selector("box"), relation="right_of"),
    )
    bindings = {
        "schema_version": "action_engine_role_bindings_v1",
        "task_id": "demo",
        "candidate_id": source["candidate_id"],
        "reference_bindings": {
            "step_01.object": ["a"],
            "step_02.object": ["a"],
            "step_01.target": ["box"],
            "step_02.target": ["box"],
        },
        "role_bindings": {},
    }
    with pytest.raises(ValueError, match="conflicting_aliased"):
        SemanticTaskPlanner().plan_spec(
            source,
            bindings,
            [
                {
                    "runtime_uid": uid,
                    "init_pos": [0.1, -0.2, 0.85],
                    "init_rot": [0.0, 0.0, 0.0],
                }
                for uid in ("a", "box")
            ],
        )


def test_empty_goal_and_ambiguous_observation_keys_are_not_covered() -> None:
    plan = _plan(count=2)
    with pytest.raises(ValueError, match="goal"):
        _coverage(plan, task_spec=replace(plan.task_spec, goal=()))
    goals = tuple(replace(p, observation="same") for p in plan.task_spec.goal)
    milestones = tuple(
        replace(m, achieve=tuple(replace(p, observation="same") for p in m.achieve))
        for m in plan.task_spec.milestones
    )
    assert (
        _coverage(
            plan, task_spec=replace(plan.task_spec, goal=goals, milestones=milestones)
        )["status"]
        == "uncovered"
    )


def test_hold_cannot_be_covered_by_transport_release() -> None:
    source = candidate(step("E5", "tray", direction="front"))
    bindings = {
        "schema_version": "action_engine_role_bindings_v1",
        "task_id": "demo",
        "candidate_id": source["candidate_id"],
        "reference_bindings": {"step_01.object": ["a"]},
        "role_bindings": {},
    }
    plan = SemanticTaskPlanner().plan_spec(
        source,
        bindings,
        [
            {
                "runtime_uid": "a",
                "init_pos": [0.1, -0.2, 0.85],
                "init_rot": [0.0, 0.0, 0.0],
            }
        ],
    )
    graph = deepcopy(plan.graph)
    for node in graph["nodes"]:
        if node["call"].get("call_id") == "simulation.coordinated_hold":
            node["call"]["call_id"] = "simulation.coordinated_transport"
    assert _coverage(plan, graph=graph)["status"] == "uncovered"
