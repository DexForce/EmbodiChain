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

"""Fixed unrefined semantic inputs; expected refinements come only from B0."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import trimesh

GRAPHS = {
    family: row["graph"]
    for family, row in json.loads(
        (Path(__file__).parent / "data/source_graph_inputs_b0.json").read_text()
    )["graphs"].items()
}

CASES = (
    "upright",
    "upright_support",
    "upright_preserve",
    "upright_reused",
    "fresh_handover_vertical",
    "fresh_handover_horizontal",
    "used_handover",
    "handover_identity_collision",
    "pour_return",
    "pour_without_table",
    "carry_maximum",
    "carry_minimum",
    "carry_middle",
    "carry_explicit",
)


def source_refinement_fixture(root: Path, case: str):
    root.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=[0.08, 0.4, 0.1])
    mesh.apply_translation([0.02, -0.04, 0.03])
    path = root / "source.glb"
    mesh.export(path)
    family = (
        "e3"
        if case.startswith("pour")
        else "e5" if case.startswith("carry") else "e4" if "handover" in case else "e2"
    )
    graph = deepcopy(GRAPHS[family])
    for target in graph["targets"].values():
        for pose in target.get("values", ()):
            pose["position"][2] = 123.0
            pose["quaternion_xyzw"] = [0, 0, 0, 1]
    for node in graph["nodes"]:
        call = node["call"]
        identity = call.get("call_id", "")
        if identity.startswith("gen_sim.pick.step_"):
            call["call_id"] = "simulation.pick"
    if "handover" in case:
        transfer = next(n for n in graph["nodes"] if n["call"]["kind"] == "hand_over")
        index = graph["nodes"].index(transfer)
        pick = graph["nodes"][index - 1]
        pick["call"] = dict(
            kind="pick",
            object=transfer["call"]["object"],
            resources={"primary": transfer["call"]["resources"]["source"]},
        )
        if case != "used_handover":
            pick["depends_on"] = []
            transfer["depends_on"] = [pick["id"]]
            graph["nodes"] = [pick, transfer]
            group = next(
                g
                for g in graph["task_groups"]
                if g["id"] == transfer["task_instance_id"]
            )
            group.update(node_ids=[pick["id"], transfer["id"]], depends_on=[])
            graph["task_groups"] = [group]
        if case == "handover_identity_collision":
            graph["targets"][transfer["id"] + "__staging_target"] = dict(
                kind="cyclic_pose",
                values=[dict(position=[0, 0, 1], quaternion_xyzw=[0, 0, 0, 1])],
            )
    if case.startswith("upright_") and case != "upright_support":
        for node in graph["nodes"]:
            node["task_type"] = "E1"
            if node["call"].get("call_id") == "gen_sim.align_held":
                node["call"]["arguments"]["target"] = "current_object_pose"
        graph["task_groups"][0]["task_type"] = "E1"
        if case == "upright_reused":
            prior = deepcopy(graph["nodes"][0])
            prior.update(id="earlier_pick", depends_on=[])
            prior["call"] = dict(
                kind="pick",
                object=prior["call"]["arguments"]["object"],
                resources=prior["call"]["resources"],
            )
            graph["nodes"][0]["depends_on"] = [prior["id"]]
            graph["nodes"].insert(0, prior)
            graph["task_groups"][0]["node_ids"].insert(0, prior["id"])
    if case == "upright_support":
        release = next(
            n
            for n in graph["nodes"]
            if n["call"].get("call_id") == "simulation.place_relative"
        )
        release["call"]["call_id"] = "gen_sim.place_upright"
        release["call"]["arguments"]["reference"] = "other_support"
    if case.startswith("carry"):
        first = graph["nodes"][0]["call"]
        first["call_id"] = "simulation.coordinated_hold"
        first["arguments"]["world_displacement"] = [
            0.1,
            0.2,
            0.23 if case == "carry_explicit" else 0.0,
        ]
    objects = set()
    for node in graph["nodes"]:
        c = node["call"]
        a = c.get("arguments", {})
        objects.update(
            x
            for x in (c.get("object"), a.get("object"), a.get("reference"))
            if isinstance(x, str)
        )
    objects.discard("table")
    scale_z = (
        0.05 if case == "carry_minimum" else 0.2 if case == "carry_middle" else 1.0
    )
    source = [
        dict(
            runtime_uid=uid,
            uid=uid,
            role="rigid_object",
            shape=dict(shape_type="Mesh", fpath=str(path)),
            body_scale=[1, 1, scale_z],
            init_pos=[0.3, -0.4, 0.81],
            init_rot=([90, 0, 0] if case == "fresh_handover_horizontal" else [0, 0, 0]),
        )
        for uid in sorted(objects)
    ]
    table = dict(
        runtime_uid="table",
        uid="table",
        role="background",
        shape=dict(shape_type="Cube", size=[1.6, 0.8, 0.1]),
        init_pos=[0.1, 0.2, 0.75],
        init_rot=[0, 0, 0],
    )
    scene = SimpleNamespace(
        planner_objects=tuple([*source, table]),
        background=(table,),
        table_top_z=None if case == "pour_without_table" else 0.8,
    )
    return graph, scene
