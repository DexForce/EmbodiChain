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

"""Workload admission and action-schema compatibility boundaries."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock
import hashlib
import json
import pytest
import torch
from embodichain.lab.expansion.gym_workloads import (
    _GymWorkloadSource,
    GymExpansionWorkload,
    run_gym_expansion_workloads,
)
from embodichain.lab.expansion.workloads import (
    ExpansionWorkloadCfg,
    WorkloadExecutionCfg,
    WorkloadAssignment,
    WorkloadProgramExecutor,
)

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("settled", [0, 5])
def test_expired_warming_row_finishes_group_without_disturbing_live_peer(
    monkeypatch, settled
):
    monkeypatch.setattr(
        "embodichain.lab.expansion.workloads.time.monotonic", lambda: 1.0
    )
    executor = object.__new__(WorkloadProgramExecutor)
    executor.cfg = WorkloadExecutionCfg(
        workload_count=2, batch_size=1, settling_steps=5
    )
    executor.workloads = {
        name: ExpansionWorkloadCfg(name, 1, 1, budget)
        for name, budget in (("expired", 0.01), ("peer", 10))
    }
    executor.started_at = {"expired": 0.0, "peer": 0.0}
    executor.counts = {
        name: {"attempts": 1, "rollouts": 0, "committed": 0}
        for name in executor.workloads
    }
    executor.warming = {
        0: {"assignment": WorkloadAssignment("expired", 0, 0, 0), "settled": settled}
    }
    peer = {"assignment": WorkloadAssignment("peer", 0, 1, 0)}
    executor.live = {1: peer}
    executor.lanes, executor.completed = ["expired", "peer"], set()
    executor.on_workload_done = Mock()
    executor.source = SimpleNamespace(activate=Mock(), complete=Mock(return_value=None))
    executor._activate()
    assert executor.warming == {}
    assert executor.live[1] is peer
    executor.source.activate.assert_not_called()
    result = executor.source.complete.call_args.args[0]
    assert result.program.reasons == ("wall_time_exhausted",)
    assert result.world_control_start is result.world_control_end is None
    executor._fill()
    executor.on_workload_done.assert_called_once_with(
        "expired", executor.counts["expired"]
    )
    assert executor.live[1] is peer


def test_native_source_retires_expired_attempt_before_reading_world():
    from embodichain.lab.expansion.session import ExpansionSession
    from embodichain.lab.expansion.session_cfg import ExpansionSessionCfg

    clock = [0.0]
    session = ExpansionSession(
        ExpansionSessionCfg.from_mapping({"collection": {"max_wall_time_s": 0.01}}),
        clock=lambda: clock[0],
    )
    clock[0] = 1.0
    request = object()
    journal = Mock()
    ctx = SimpleNamespace(requests={0: request}, journal=journal)
    source = object.__new__(_GymWorkloadSource)
    source.groups = {
        "expired": {
            "host": SimpleNamespace(_context=ctx),
            "engine": object(),
            "session": session,
        }
    }
    assert source.activate(WorkloadAssignment("expired", 0, 0, 0), None) is None
    journal.update.assert_called_once_with(
        request, status="initial_rejected", reason="wall_time_exhausted"
    )
    assert session.snapshot()["counts"]["proposed"] == 0


@pytest.mark.parametrize("stop", ["wall_time_exhausted", "unexpected"])
def test_admission_expiry_race_only_suppresses_budget_exhaustion(stop):
    session = SimpleNamespace(stop_reason=None, register_case=Mock())

    def propose(*args, **kwargs):
        session.stop_reason = stop
        raise RuntimeError("proposal failed")

    session.propose = propose
    request = SimpleNamespace(
        recipe=SimpleNamespace(
            recipe_id="recipe",
            operations=[SimpleNamespace(name="physical_parameters", payload_json="{}")],
        ),
        attempt_id=0,
    )
    ctx = SimpleNamespace(
        requests={0: request},
        journal=Mock(),
        collision=None,
        report={"semantic_id": "semantic", "recipe_revision": "revision"},
        cfg=SimpleNamespace(seed=0, source="task_program"),
        compatibility=SimpleNamespace(key="compatible"),
        program=SimpleNamespace(program_id="program"),
        limits=object(),
        names=("joint",),
    )
    source = object.__new__(_GymWorkloadSource)
    source.groups = {
        "group": {
            "host": SimpleNamespace(_context=ctx),
            "engine": object(),
            "session": session,
        }
    }
    source.activation_snapshot = (SimpleNamespace(batches={0: ()}), {})
    source.env = SimpleNamespace(robot=SimpleNamespace(uid="robot"))
    assignment = WorkloadAssignment("group", 0, 0, 0)
    if stop == "wall_time_exhausted":
        assert source.activate(assignment, None) is None
        assert ctx.journal.update.call_args.kwargs["reason"] == stop
    else:
        with pytest.raises(RuntimeError, match="proposal failed"):
            source.activate(assignment, None)
        ctx.journal.update.assert_not_called()


@pytest.mark.parametrize(
    "difference", [None, "position_velocity", "use_action_manager", "joint_selection"]
)
def test_workload_action_schema_checked_before_constructing_world(
    tmp_path, monkeypatch, difference
):
    from embodichain.lab.expansion import ExpansionRunCfg
    from embodichain.lab.gym.envs.expert_trajectory import ExpertTrajectoryCfg
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
    import embodichain.lab.gym.utils.gym_utils as utils
    import embodichain.lab.expansion.host as host

    class ReachedWorld(Exception):
        pass

    factory = Mock(side_effect=ReachedWorld)
    monkeypatch.setattr(host, "GymExpansionHost", factory)
    workloads, configs = [], {}
    for i, name in enumerate(("first", "second")):
        deployment = tmp_path / f"{name}.json"
        deployment.write_text(json.dumps({"id": name}))
        expert = ExpertTrajectoryCfg(
            joint_command_mode=(
                "position_velocity"
                if i and difference == "position_velocity"
                else "position"
            ),
            use_action_manager=bool(i and difference == "use_action_manager"),
        )
        configs[name] = SimpleNamespace(
            robot=SimpleNamespace(
                to_dict=lambda: {
                    "joint_names": ["a", "b"],
                    "control_parts": {"arm": [0, 1]},
                }
            ),
            expert_trajectory=expert,
        )
        cls = type(
            f"Env{i}",
            (),
            {
                "control_parts": ["arm"],
                "active_joint_ids": (
                    [1] if i and difference == "joint_selection" else [0]
                ),
            },
        )
        monkeypatch.setitem(REGISTERED_ENVS, name, SimpleNamespace(cls=cls))
        cfg = ExpansionRunCfg(
            deployment=str(deployment),
            output_dir=str(tmp_path / name),
            max_steps=20,
            settling_steps=1,
        )
        workloads.append(
            GymExpansionWorkload(
                name,
                cfg,
                {
                    "input_files": {
                        "deployment": hashlib.sha256(
                            deployment.read_bytes()
                        ).hexdigest()
                    }
                },
                {},
            )
        )
    monkeypatch.setattr(
        utils, "config_to_cfg", lambda raw, **kwargs: configs[raw["id"]]
    )
    execution = WorkloadExecutionCfg(
        workload_count=2, batch_size=1, max_steps=20, settling_steps=1
    )
    if difference is None:
        with pytest.raises(ReachedWorld):
            run_gym_expansion_workloads(
                workloads, execution, output_dir=tmp_path / "run"
            )
        factory.assert_called_once()
    else:
        with pytest.raises(ValueError, match="schema"):
            run_gym_expansion_workloads(
                workloads, execution, output_dir=tmp_path / "run"
            )
        factory.assert_not_called()
