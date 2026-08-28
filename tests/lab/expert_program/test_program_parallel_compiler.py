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

"""Tests for provider-free parallel program compilation."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.expert_program import (
    BarrierCfg,
    ExpertProgramCfg,
    ExpertProgramCompileError,
    ExpertProgramCompiler,
    ExpertProgramIntegrationCfg,
    InvokeCfg,
    ParallelCfg,
    PickCfg,
    RepeatCfg,
    SegmentCfg,
    SequenceCfg,
)
from embodichain.lab.sim.atomic_actions import EntityState
from embodichain.lab.semantic_skills.calls import Pick
from embodichain.lab.semantic_skills.scene import (
    SceneEntityRegistration,
    SceneObjectRef,
    SceneRegistry,
)


class _NeverObserveProvider:
    """Reject dynamic observation during provider-free compilation."""

    def observe(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> EntityState:
        del timestamp, env_ids
        raise AssertionError("Compilation must not observe the scene.")


def _compiler() -> ExpertProgramCompiler:
    provider = _NeverObserveProvider()
    registry = SceneRegistry(
        tuple(
            SceneEntityRegistration(
                ref=SceneObjectRef(entity_id),
                state_provider=provider,
            )
            for entity_id in ("left_cube", "right_cube")
        )
    )
    return ExpertProgramCompiler.from_scene_registry(registry)


def _integration() -> ExpertProgramIntegrationCfg:
    return ExpertProgramIntegrationCfg(
        robot_profile="dual_arm",
        scene_registry="scene",
        runtime_preset="safe",
    )


def _parallel() -> ParallelCfg:
    return ParallelCfg(
        branches=(
            SequenceCfg(
                items=(
                    InvokeCfg(call=PickCfg(object="left_cube")),
                    InvokeCfg(call=PickCfg(object="left_cube")),
                )
            ),
            RepeatCfg(
                count=2,
                body=InvokeCfg(call=PickCfg(object="right_cube")),
            ),
        ),
        barrier=BarrierCfg(
            name="both_arms_done",
            timeout_steps=240,
            failure_policy="fail_fast",
        ),
    )


def _config(program: ParallelCfg | SegmentCfg | SequenceCfg) -> ExpertProgramCfg:
    return ExpertProgramCfg(
        program_id="parallel_pick",
        integration=_integration(),
        program=program,
        targets={},
    )


def test_parallel_compiles_independent_ordered_lanes_and_explicit_join() -> None:
    segment = tuple(_compiler().compile(_config(_parallel())))[0]

    assert segment.implicit
    assert segment.name == "parallel:both_arms_done"
    assert segment.parallel_block is not None
    block = segment.parallel_block
    assert block.barrier.name == "both_arms_done"
    assert block.barrier.timeout_steps == 240
    assert block.barrier.failure_policy == "fail_fast"
    assert tuple(branch.branch_index for branch in block.branches) == (0, 1)
    assert tuple(len(branch.calls) for branch in block.branches) == (2, 2)
    assert tuple(call.call_index for call in segment.calls) == (0, 1, 2, 3)
    assert tuple(call.segment_call_index for call in segment.calls) == (0, 1, 2, 3)
    assert segment.calls == tuple(
        call for branch in block.branches for call in branch.calls
    )
    assert all(type(call.call) is Pick for call in segment.calls)
    assert tuple(call.call.object.entity_id for call in block.branches[0].calls) == (
        "left_cube",
        "left_cube",
    )
    assert tuple(call.call.object.entity_id for call in block.branches[1].calls) == (
        "right_cube",
        "right_cube",
    )
    assert tuple(
        frame.iteration_index
        for call in block.branches[1].calls
        for frame in call.repeat_frames
    ) == (0, 1)


def test_segment_may_wrap_one_parallel_block() -> None:
    segment = tuple(
        _compiler().compile(_config(SegmentCfg(name="dual_pick", steps=_parallel())))
    )[0]

    assert not segment.implicit
    assert segment.name == "dual_pick"
    assert segment.parallel_block is not None
    assert len(segment.calls) == 4


def test_compiled_analysis_stops_sequential_lookahead_at_parallel_barriers() -> None:
    config = _config(
        SequenceCfg(
            items=(
                InvokeCfg(call=PickCfg(object="left_cube")),
                _parallel(),
                InvokeCfg(call=PickCfg(object="left_cube")),
            )
        )
    )

    program = _compiler().compile(config)
    analyses = program.preflight_analyses()

    assert [analysis.kind for analysis in analyses] == [
        "sequential_stretch",
        "parallel_branch",
        "parallel_branch",
        "sequential_stretch",
    ]
    assert [analysis.segment_indices for analysis in analyses] == [
        (0,),
        (1,),
        (1,),
        (2,),
    ]
    assert program.sequential_execution_analysis(0).segment_indices == (0,)
    assert program.sequential_execution_analysis(2).segment_indices == (2,)
    with pytest.raises(ValueError, match="Parallel segments"):
        program.sequential_execution_analysis(1)


def test_parallel_branch_rejects_segment_owned_lifecycle() -> None:
    invoke = InvokeCfg(call=PickCfg(object="left_cube"))
    parallel = ParallelCfg(
        branches=(
            SegmentCfg(name="branch", steps=invoke),
            invoke,
        ),
        barrier=BarrierCfg(name="join"),
    )

    with pytest.raises(ValueError, match="wrap the Parallel node in one Segment"):
        _config(parallel)


def test_segment_rejects_mixed_sequential_and_parallel_tree() -> None:
    invoke = InvokeCfg(call=PickCfg(object="left_cube"))
    config = _config(
        SegmentCfg(
            name="ambiguous_boundary",
            steps=SequenceCfg(items=(invoke, _parallel())),
        )
    )

    with pytest.raises(
        ExpertProgramCompileError,
        match="either a call-only program or one direct Parallel",
    ):
        _compiler().compile(config)


__all__: list[str] = []
