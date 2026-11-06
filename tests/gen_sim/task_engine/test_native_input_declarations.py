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

"""Native input declarations retain exact occurrence and permission ownership."""

from __future__ import annotations

from copy import deepcopy

import pytest

from embodichain.gen_sim.task_engine.continuation_declarations import (
    resolve_source_continuation_declarations,
    resolve_inside_yaw_declarations,
    resolve_measured_rigid_declarations,
)
from embodichain.gen_sim.task_engine.native_pick_declarations import (
    resolve_native_pick_revisions,
    resolve_native_pick_bindings,
)


def _pairs() -> tuple[dict, dict, dict, dict, dict, dict]:
    nodes = []
    for suffix in ("first", "second"):
        nodes.extend(
            [
                dict(
                    id=f"pick_{suffix}",
                    task_type="E1",
                    call=dict(
                        kind="pick", object="payload", resources={"primary": "arm"}
                    ),
                ),
                dict(
                    id=f"place_{suffix}",
                    task_type="E1",
                    call=dict(
                        kind="place",
                        object="payload",
                        on="on__tray__payload",
                        resources={"primary": "arm"},
                    ),
                ),
            ]
        )
    graph = dict(nodes=nodes)
    occurrences = {row["id"]: f"program/{row['id']}:0" for row in nodes}
    options = {
        row["id"]: (
            dict(adaptive_grasp="ordinary") if row["call"]["kind"] == "pick" else {}
        )
        for row in nodes
    }
    motions = {row["id"]: dict(strategy="ik_interp") for row in nodes}
    recoveries = {row["id"]: dict(kind="pick_bounded") for row in nodes}
    scene = dict(
        rigid_objects=[
            dict(entity_id="payload", affordances=[]),
            dict(
                entity_id="tray",
                affordances=[dict(entity_id="on__tray__payload", kind="support")],
            ),
        ]
    )
    return graph, occurrences, options, motions, recoveries, scene


def test_repeated_pick_continuations_keep_distinct_downstream_opening_occurrences() -> (
    None
):
    graph, occurrences, options, motions, recoveries, scene = _pairs()
    before = deepcopy((graph, occurrences, options, motions, recoveries, scene))
    continuations = resolve_source_continuation_declarations(
        graph,
        program_fingerprint="source",
        occurrences=occurrences,
        options=options,
        motions=motions,
        recoveries=recoveries,
        staging_bindings=(),
    )
    assert continuations == [
        dict(
            program_fingerprint="source",
            source_invocation_id=f"program/pick_{suffix}:0",
            place_invocation_id=f"program/place_{suffix}:0",
        )
        for suffix in ("first", "second")
    ]
    revisions = resolve_native_pick_revisions(graph, 4, None)
    assert revisions == {"pick_first": 4, "pick_second": 4}
    assert resolve_native_pick_bindings(
        graph,
        revisions=revisions,
        occurrences=occurrences,
        options=options,
        continuations=continuations,
        scene_binding=scene,
    ) == [
        dict(
            invocation_id=f"program/pick_{suffix}:0",
            object_id="payload",
            opening_reference_id="tray",
            opening_place_invocation_id=f"program/place_{suffix}:0",
        )
        for suffix in ("first", "second")
    ]
    assert (graph, occurrences, options, motions, recoveries, scene) == before


def test_continuation_and_measured_permissions_reject_unqualified_selections() -> None:
    graph, occurrences, options, motions, recoveries, scene = _pairs()
    motions["pick_first"] = dict(strategy="external_nominal")
    graph["nodes"][-1]["call"]["resources"] = {"primary": "another_arm"}
    assert (
        resolve_source_continuation_declarations(
            graph,
            program_fingerprint="source",
            occurrences=occurrences,
            options=options,
            motions=motions,
            recoveries=recoveries,
            staging_bindings=(),
        )
        == []
    )
    with pytest.raises(ValueError, match="missing source node"):
        resolve_native_pick_revisions(graph, None, {"absent": 4})
    with pytest.raises(ValueError, match="integer"):
        resolve_native_pick_revisions(graph, True, None)
    with pytest.raises(ValueError, match="revision4"):
        resolve_measured_rigid_declarations(
            graph,
            ["pick_first"],
            scene_binding=scene,
            native_revisions={"pick_first": 8},
        )
    graph["nodes"][1]["call"].pop("on")
    graph["nodes"][1]["call"]["inside"] = "inside__tray__payload"
    with pytest.raises(ValueError, match="verified"):
        resolve_inside_yaw_declarations(
            graph,
            ["place_first"],
            scene_binding=scene,
            effect_monitors={"pick": "retain", "place": "release"},
            verified=False,
        )
