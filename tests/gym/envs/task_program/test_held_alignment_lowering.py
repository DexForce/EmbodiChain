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

"""Canonical alignment aliases retain observed source targets and effects."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
import torch
from embodichain.lab.sim.atomic_actions import MoveHeldObjectOptions, StateDelta
from embodichain.lab.task_program.integrations.configured import (
    _decode_registered_lowerer,
)
from embodichain.lab.task_program.integrations.simulation.held_alignment import (
    HeldAlignmentRoute,
    HeldAlignmentLowererFactory,
)
from ._held_alignment_lowering_fixture import alignment_inputs, lowering_record

_REFERENCE = json.loads(
    (Path(__file__).parent / "data/held_alignment_lowering_b0.json").read_text()
)


@pytest.mark.parametrize("case", _REFERENCE["cases"])
def test_alignment_public_factory_retains_frozen_source_targets_and_root_reads(case):
    route, call, context, bound, robot, reads = alignment_inputs(case["inputs"])
    factory = HeldAlignmentLowererFactory(
        (route,), verify_retention=case["inputs"]["retain"]
    )
    scene = SimpleNamespace(
        resolve=lambda name, **kwargs: SimpleNamespace(entity_id=name)
    )
    lowerer = factory.create(
        simulation=None,
        robot=robot,
        scene_registry=scene,
        engine=SimpleNamespace(robot=robot),
    )
    before = context.scene.entities["item"].pose.clone()
    result = lowerer.lower(
        call, context=context, bound=bound, option_template=MoveHeldObjectOptions()
    )
    assert lowering_record(result, reads) == case["expected"]
    torch.testing.assert_close(context.scene.entities["item"].pose, before)
    assert lowerer.call_id == "gen_sim.align_held"
    assert lowerer.preserves_symbolic_state is not case["inputs"]["retain"]


def test_alignment_declaration_rejects_unknown_options_and_missing_exclusive_hold():
    cfg = dict(
        kind="align_held",
        verify_retention=True,
        routes=[
            dict(
                object_id="item",
                target_id="destination",
                preserve_yaw=False,
                axis_direction_free=False,
                local_axis=[0.0, 0.0, 1.0],
                position=[0.3, -0.1, 0.4],
            )
        ],
    )
    factory = _decode_registered_lowerer(cfg, path="lowerer")
    assert type(factory) is HeldAlignmentLowererFactory
    cfg["routes"][0]["position"][0] = 999
    assert factory.routes[0].position == (0.3, -0.1, 0.4)
    with pytest.raises(ValueError, match="unsupported fields"):
        _decode_registered_lowerer({**cfg, "sample_count": 260}, path="lowerer")
    with pytest.raises(TypeError, match="verify_retention"):
        _decode_registered_lowerer({**cfg, "verify_retention": 1}, path="lowerer")
    with pytest.raises(ValueError, match="destination disagree"):
        HeldAlignmentRoute(
            "item",
            "current_object_pose",
            False,
            False,
            (0.0, 0.0, 1.0),
            (0.1, 0.2, 0.3),
        )
    case = _REFERENCE["cases"][0]["inputs"]
    route, call, context, bound, robot, reads = alignment_inputs(case)
    lowerer = HeldAlignmentLowererFactory((route,)).create(
        simulation=None,
        robot=robot,
        scene_registry=SimpleNamespace(resolve=lambda *a, **k: None),
        engine=SimpleNamespace(robot=robot),
    )
    task = StateDelta(held_object_updates={"custom_state": None}).apply(
        context.task, torch.tensor([True])
    )
    with pytest.raises(ValueError, match="exclusive held"):
        lowerer.lower(
            call,
            context=replace(context, task=task),
            bound=bound,
            option_template=MoveHeldObjectOptions(),
        )
    assert reads == []
