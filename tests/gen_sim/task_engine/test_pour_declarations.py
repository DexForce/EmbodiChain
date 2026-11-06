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


def test_explicit_e2_held_target_keeps_source_world_pose_priority_over_reference():
    from embodichain.gen_sim.task_engine.pour_declarations import (
        resolve_pour_lowerer_declarations,
    )

    pose = dict(position=[0.1, 0.2, 0.8], quaternion_xyzw=[0, 0, 0, 1])
    graph = dict(
        nodes=[
            dict(
                task_type="E2",
                call=dict(
                    kind="registered",
                    call_id="simulation.move_held_object",
                    arguments=dict(
                        object="payload", target="upright", reference="unread"
                    ),
                ),
            )
        ],
        targets=dict(upright=dict(kind="cyclic_pose", values=[pose])),
    )
    assert resolve_pour_lowerer_declarations(graph, {}, {}, ()) == [
        dict(
            kind="move_held_object",
            effect_contract="terminal_attachment",
            routes=[
                dict(
                    object_id="payload",
                    target_id="upright",
                    pose={"kind": "pose", **pose},
                )
            ],
        )
    ]
