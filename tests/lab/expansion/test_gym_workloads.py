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


def _prepare_cube_scene(cfg, *, x):
    """A valid pre-World hook whose parameters alter the prepared scene."""
    from copy import deepcopy

    prepared = deepcopy(cfg)
    cube = next(obj for obj in prepared.rigid_object if obj.uid == "cube")
    cube.init_pos = (x, *cube.init_pos[1:])
    return prepared


def _prepare_cube_scene_offset(cfg, *, x):
    """A distinct callable declaration with different scene preparation."""
    return _prepare_cube_scene(cfg, x=x + 0.1)


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


@pytest.mark.parametrize("stop", ["wall_time_exhausted", "rollout_budget_exhausted"])
def test_native_source_retires_exhausted_attempt_before_reading_world(stop):
    from embodichain.lab.expansion.contracts import (
        EpisodeCandidate,
        SceneCase,
        ValidationCheck,
        ValidationResult,
    )
    from embodichain.lab.expansion.session import ExpansionSession
    from embodichain.lab.expansion.session_cfg import ExpansionSessionCfg

    clock = [0.0]
    session = ExpansionSession(
        ExpansionSessionCfg.from_mapping(
            {
                "collection": {
                    "max_wall_time_s": 0.01,
                    "target_committed_episodes": 2,
                    "max_proposals": 3,
                    "max_rollout_attempts": 2,
                },
                "execution": {"ready_high_watermark": 3, "ready_low_watermark": 0},
                "persistence": {"pending_episode_limit": 2, "pending_max_bytes": 4096},
            }
        ),
        clock=lambda: clock[0],
    )
    if stop == "wall_time_exhausted":
        clock[0] = 1.0
    else:
        session.register_case(
            SceneCase("case", "initial", "scene", "task", "robot"),
            torch.tensor([[-1.0, 1.0]]),
            joint_names=("joint",),
        )
        passed = ValidationResult(
            (
                ValidationCheck("path_collision", "passed"),
                ValidationCheck("task_success", "passed"),
            )
        )
        identities = []
        for index in range(2):
            identity, _ = session.propose(
                "case",
                "initial",
                source_id="source",
                source_revision="revision",
                template_id=f"recipe-{index}",
                operator_id="episode",
            )
            session.add_prepared_episode(EpisodeCandidate(identity, "{}"), passed)
            assert (
                session.take_ready("case", "initial", episode_byte_budget=1024)
                is not None
            )
            session.mark_rollout_started(identity)
            identities.append(identity)
        session.release(identities[0], reason="rejected")
        assert session.stop_reason is None
        session.release(identities[1], reason="rejected")
    assert session.stop_reason == stop
    proposed = session.snapshot()["counts"]["proposed"]
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
        request, status="initial_rejected", reason=stop
    )
    assert session.snapshot()["counts"]["proposed"] == proposed


@pytest.mark.parametrize(
    "stop", ["wall_time_exhausted", "rollout_budget_exhausted", "unexpected"]
)
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
    if stop != "unexpected":
        assert source.activate(assignment, None) is None
        assert ctx.journal.update.call_args.kwargs["reason"] == stop
    else:
        with pytest.raises(RuntimeError, match="proposal failed"):
            source.activate(assignment, None)
        ctx.journal.update.assert_not_called()


@pytest.mark.parametrize(
    "difference",
    [
        None,
        "position_velocity",
        "use_action_manager",
        "active_joint_ids",
        "joint_order",
        "preparation_params",
        "preparation_func",
        "preparation_same",
    ],
)
def test_workload_action_schema_checked_before_constructing_world(
    tmp_path, monkeypatch, difference
):
    from embodichain.lab.expansion import ExpansionRunCfg
    from embodichain.lab.gym.envs.expert_trajectory import ExpertTrajectoryCfg
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS, EnvSpec
    import embodichain.lab.gym.utils.gym_utils as utils
    import embodichain.lab.expansion.host as host

    class ReachedWorld(Exception):
        pass

    factory = Mock(side_effect=ReachedWorld)
    monkeypatch.setattr(host, "GymExpansionHost", factory)
    from copy import deepcopy
    from pathlib import Path
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.utils.utility import load_config

    deployment = (
        Path(__file__).resolve().parents[3]
        / "embodichain/extensions/tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml"
    )
    base = utils.config_to_cfg(load_config(deployment), source_path=deployment)
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
        config = deepcopy(base)
        config.expert_trajectory = expert
        config.control_parts = ["arm"]
        config.active_joint_ids = []
        if difference in ("active_joint_ids", "joint_order"):
            config.control_parts = None
            config.active_joint_ids = (
                [i] if difference == "active_joint_ids" else ([1, 0] if i else [0, 1])
            )
        if difference in {"preparation_params", "preparation_func", "preparation_same"}:
            from embodichain.lab.gym.envs.managers.cfg import FunctorCfg

            function = (
                "_prepare_cube_scene_offset"
                if i and difference == "preparation_func"
                else "_prepare_cube_scene"
            )
            config.scene_preparation = FunctorCfg(
                func=f"{__name__}:{function}",
                params={"x": 0.8 if i and difference == "preparation_params" else 0.2},
            )
        configs[name] = config
        # Configured tasks share this implementation class. Selection belongs
        # to the resolved configs above, never to artificial per-task classes.
        monkeypatch.setitem(REGISTERED_ENVS, name, EnvSpec(name, EmbodiedEnv))
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
    if difference in (None, "preparation_same"):
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


@pytest.mark.parametrize(
    "field,registered_x,explicit_x,compatible",
    [
        ("control_parts", (None, None), (None, None), False),
        ("scene_preparation", (None, None), (0.2, 0.8), False),
        ("scene_preparation", (0.2, 0.8), (None, None), False),
        ("scene_preparation", (0.2, 0.2), (None, None), True),
        ("scene_preparation", (0.2, 0.8), (0.4, 0.4), True),
        ("scene_preparation", (None, 0.2), (0.2, None), True),
    ],
    ids=[
        "control_parts",
        "explicit_preparation_mismatch",
        "registered_preparation_mismatch",
        "registered_preparation_same",
        "explicit_preparation_overrides_defaults",
        "explicit_matches_registered_preparation",
    ],
)
def test_real_deployments_validate_shared_world_setup(
    tmp_path, monkeypatch, field, registered_x, explicit_x, compatible
):
    from pathlib import Path
    from embodichain.lab.expansion import ExpansionRunCfg
    from embodichain.lab.expansion.task_program_jobs import _prepare_deployment
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import (
        REGISTERED_ENVS,
        register_env_function,
    )
    from embodichain.utils.utility import load_config
    import embodichain.lab.expansion.host as host

    import yaml

    task = (
        Path(__file__).resolve().parents[3]
        / "embodichain/extensions/tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml"
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    base = load_config(_prepare_deployment(task, inputs))
    workloads = []

    def preparation(x):
        return {"func": f"{__name__}:_prepare_cube_scene", "params": {"x": x}}

    for i, name in enumerate(("first", "second")):
        component = load_config(task.parent / "envs/default.yaml")
        if field == "control_parts":
            component["env"][field] = ["arm"] if i == 0 else ["arm", "eef"]
        elif explicit_x[i] is not None:
            component["env"][field] = preparation(explicit_x[i])
        physical = inputs / f"{name}.environment.yaml"
        physical.write_text(yaml.safe_dump(component))
        declaration = {
            **base,
            "id": f"WorkloadControlSchema-{name}-v1",
            "environment": {"component": str(physical)},
        }
        path = inputs / f"{name}.json"
        path.write_text(json.dumps(declaration))
        config = config_to_cfg(declaration, source_path=path)
        if field == "control_parts":
            assert config.control_parts == component["env"][field]
        elif explicit_x[i] is None:
            assert config.scene_preparation is None
        else:
            assert config.scene_preparation.func == preparation(explicit_x[i])["func"]
            assert config.scene_preparation.params == {"x": explicit_x[i]}
        spec = REGISTERED_ENVS[declaration["id"]]
        # Keep the real configured integration and ordinary environment class,
        # but register task-owned defaults through the supported entry point.
        register_env_function(
            EmbodiedEnv,
            declaration["id"],
            override=True,
            max_episode_steps=spec.max_episode_steps,
            task_program_adapter_factory=spec.task_program_adapter_factory,
            **(
                {"scene_preparation": preparation(registered_x[i])}
                if registered_x[i] is not None
                else {}
            ),
        )
        cfg = ExpansionRunCfg(
            deployment=str(path),
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
                        "deployment": hashlib.sha256(path.read_bytes()).hexdigest()
                    }
                },
                {},
            )
        )

    class ReachedWorld(Exception):
        pass

    constructor = Mock(side_effect=ReachedWorld)
    monkeypatch.setattr(host, "GymExpansionHost", constructor)
    with pytest.raises(
        ReachedWorld if compatible else ValueError,
        match=None if compatible else "workloads must share",
    ):
        run_gym_expansion_workloads(
            workloads,
            WorkloadExecutionCfg(
                workload_count=2, batch_size=1, max_steps=20, settling_steps=1
            ),
            output_dir=tmp_path / "run",
        )
    if compatible:
        constructor.assert_called_once()
    else:
        constructor.assert_not_called()
