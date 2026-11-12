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


"""Physical acceptance and installed-path registration for cloth folding."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


@pytest.mark.parametrize("config_first", (True, False))
def test_cloth_fold_registration_order(tmp_path: Path, config_first: bool) -> None:
    package = (
        Path(__file__).resolve().parents[4] / "extensions/embodichain_tasks/configs"
    )
    config = package / "tasks/manipulation/cloth_fold/task.w1.yaml"
    script = """
import sys
from dataclasses import replace
from pathlib import Path

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import (
    REGISTERED_ENVS,
    discover_task_packages,
    get_env_spec,
)
from embodichain.utils.utility import load_config

path = Path(sys.argv[1])
env_id = "TaskProgramClothFold-W1-v1"
assert env_id not in REGISTERED_ENVS
if sys.argv[2] == "True":
    cfg = config_to_cfg(load_config(path), source_path=path)
    assert get_env_spec(env_id).cls is EmbodiedEnv
    discover_task_packages()
else:
    discover_task_packages()
    assert env_id not in REGISTERED_ENVS
    cfg = config_to_cfg(load_config(path), source_path=path)

spec = get_env_spec(env_id)
assert spec.cls is EmbodiedEnv
assert config_to_cfg(load_config(path), source_path=path).final_acceptance == cfg.final_acceptance
assert cfg.robot.uid == "w1"
assert cfg.task_program.program_id == "cloth_single_fold"
assert cfg.kinematic_replay.load()[0].shape == (991, 24)
if sys.argv[2] == "True":
    cfg.extensions = {"task_program": None}
    try:
        spec.make(cfg=cfg)
    except ValueError as exc:
        assert "Extensions cannot override" in str(exc)
    else:
        raise AssertionError("Extensions bypassed configured acceptance.")
    cfg.extensions = None
    surface = cfg.replay_surface
    cfg.replay_surface = replace(surface, mesh_sha256="0" * 64)
    try:
        spec.make(cfg=cfg)
    except ValueError as exc:
        assert "registered replay mesh" in str(exc)
    else:
        raise AssertionError("A different mesh passed construction.")
    cfg.replay_surface = surface
cfg.final_acceptance = replace(cfg.final_acceptance, max_distance_m=0.020)
try:
    spec.make(cfg=cfg)
except ValueError as exc:
    assert "registered final acceptance" in str(exc)
else:
    raise AssertionError("A different acceptance passed construction.")
"""
    robot = tmp_path / "w1_v030" / "robot.urdf"
    robot.parent.mkdir()
    robot.touch()
    subprocess.run(
        [sys.executable, "-c", script, str(config), str(config_first)],
        cwd=tmp_path,
        env={**os.environ, "EMBODICHAIN_DATA_ROOT": str(tmp_path)},
        check=True,
        capture_output=True,
        text=True,
    )


def _measured_fold(*, device="cpu", pairs=(((0,), (1,)), ((2,), (3,))), node_count=4):
    from embodichain.lab.gym.envs import EmbodiedEnv
    from embodichain.lab.gym.envs._final_acceptance import _FinalAcceptance
    from embodichain.lab.gym.envs._physical_replay import _PhysicalReplay
    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.utils.trajectory_state import _evaluation_observation_cfgs
    from embodichain.lab.evaluation import EvaluationSpec, builtin_predicates

    env = object.__new__(EmbodiedEnv)
    env._num_envs = 1
    nodes = torch.zeros((1, node_count, 3), device=device)
    cloth = SimpleNamespace(data=SimpleNamespace(nodal_pos_w=nodes))
    env.sim = SimpleNamespace(
        device=torch.device(device), get_deformable_object=lambda _: cloth
    )
    env.cfg = SimpleNamespace(task_program=object())
    declaration = _FinalAcceptance("cloth", pairs, 0.015, "a" * 64)
    payload = declaration.evaluation()
    graph = builtin_predicates().compile(EvaluationSpec.from_dict(payload["graph"]))
    conditions = {ref: graph.select(ref) for ref in graph.roots}
    registration = SimpleNamespace(
        evaluation_conditions=conditions,
        scene_binding=SimpleNamespace(rigid_objects=(), articulations=()),
    )
    env._final_acceptance_rules = tuple(conditions.items())
    env._evaluation_recording = _evaluation_observation_cfgs(registration)
    env.episode_context = EpisodeContext(1, device)
    env.episode_context._begin_episode([0])
    replay = object.__new__(_PhysicalReplay)
    replay.surface = cloth
    replay.surface_cfg = SimpleNamespace(uid="cloth")
    replay.verified_mesh_sha256 = "a" * 64
    replay._node_shape = None
    replay.factory = SimpleNamespace(steps=2, fps=60)
    env.sim.spawn_result = SimpleNamespace(get_simulation_time=lambda: 2 / 60)
    env._physical_replay = replay
    env._active_task_program_bridge = SimpleNamespace(
        program_completed=False, completion_mask=torch.tensor([True])
    )
    return env, nodes


def test_fold_remeasures_both_corners_after_program_completion() -> None:
    env, nodes = _measured_fold()
    bridge = env._active_task_program_bridge
    nodes[0, 1, 0] = 0.015
    nodes[0, 3, 1] = 0.015
    assert not env.is_task_success().item()
    bridge.program_completed = True
    assert env.is_task_success().item()
    nodes[0, 3, 1] = 0.01501
    assert not env.is_task_success().item()
    nodes[0, 3, 1] = float("nan")
    assert not env.is_task_success().item()


def test_fold_abort_hold_does_not_consume_next_replay_frame() -> None:
    from embodichain.lab.gym.envs import BaseEnv, EmbodiedEnv
    from embodichain.lab.gym.envs._physical_replay import _PhysicalReplay
    from embodichain.lab.gym.envs.types import ControllerAction

    env = object.__new__(EmbodiedEnv)
    replay = object.__new__(_PhysicalReplay)
    env._physical_replay = replay
    replay.factory = SimpleNamespace(fps=60, steps=2)
    replay.commands = torch.tensor([[0.0], [0.1], [0.2]])
    env.sim = SimpleNamespace(
        spawn_result=SimpleNamespace(get_simulation_time=lambda: 1 / 60)
    )
    hold = ControllerAction(
        replay.commands[None, 1],
        metadata={"bridge_action_kind": "runtime_abort_safe_hold"},
    )

    torch.testing.assert_close(env._step_action(hold), hold.value)
    with patch.object(BaseEnv, "_advance_physics") as advance:
        env._advance_physics()
    advance.assert_not_called()


def test_cloth_final_rules_seal_owned_evidence_and_replay_it():
    from embodichain.lab.gym.envs.task_program._final_acceptance import (
        _final_acceptance_rows,
        _export_final_observations,
        _replay_final_acceptance,
    )

    env, nodes = _measured_fold()
    env._active_task_program_bridge.program_completed = True
    accepted = env._finalize_demo_task_success((True,))
    episodes, eligible, completion, measurement = env._pending_final_acceptance
    final = _final_acceptance_rows(
        env._final_acceptance_rules,
        episodes,
        tuple(accepted.tolist()),
        eligible,
        completion,
        measurement,
    )[0]
    assert final["accepted"] is True
    assert set(final["conditions"]) == {"centroid_pair_0", "centroid_pair_1"}
    metadata = [
        {"env_id": 0, "episode_uuid": "sealed-cloth", "final_acceptance": final}
    ]
    raw = _export_final_observations(
        env._final_acceptance_rules, {0: measurement.frame}, metadata
    )
    payload = {
        "meta": {"num_envs": 1, "env_ids": [0], "demo_episodes": metadata},
        "final_acceptance_observations": raw,
    }
    nodes[:, 1, 0] = 0.1
    assert not env.is_task_success().item()
    report = _replay_final_acceptance(env._final_acceptance_rules, payload)
    assert report[0]["accepted"] is True
    assert report[0]["status"] == "recomputed"


@pytest.mark.parametrize("foreign", ["mesh", "group", "episode", "revision", "missing"])
def test_cloth_final_evidence_marks_foreign_or_missing_sources_unavailable(foreign):
    from copy import deepcopy
    from embodichain.lab.gym.envs.task_program._final_acceptance import (
        _final_acceptance_rows,
        _export_final_observations,
        _replay_final_acceptance,
    )

    env, _ = _measured_fold()
    env._active_task_program_bridge.program_completed = True
    accepted = env._finalize_demo_task_success((True,))
    episodes, eligible, completion, measured = env._pending_final_acceptance
    final = _final_acceptance_rows(
        env._final_acceptance_rules, episodes, (True,), eligible, completion, measured
    )[0]
    metadata = [
        {"env_id": 0, "episode_uuid": "sealed-cloth", "final_acceptance": final}
    ]
    raw = _export_final_observations(
        env._final_acceptance_rules, {0: measured.frame}, metadata
    )
    payload = deepcopy(
        {
            "meta": {"num_envs": 1, "env_ids": [0], "demo_episodes": metadata},
            "final_acceptance_observations": raw,
        }
    )
    row = payload["final_acceptance_observations"]["rows"][0]
    if foreign in ("mesh", "group"):
        for field in row["fields"].values():
            identity = field["descriptor"]["entity_uid"]
            field["descriptor"]["entity_uid"] = (
                identity.replace("a" * 64, "b" * 64)
                if foreign == "mesh"
                else identity + ",9"
            )
    elif foreign == "episode":
        row["episode_id"] += 1
    elif foreign == "revision":
        row["observation_revision"] += 1
    else:
        row["fields"] = {}
    report = _replay_final_acceptance(env._final_acceptance_rules, payload)
    assert report[0]["accepted"] is None
    assert report[0]["status"] == "unavailable"


def test_cloth_native_mesh_replacement_and_incomplete_cadence_cannot_pass():
    env, nodes = _measured_fold()
    env._active_task_program_bridge.program_completed = True
    assert env.is_task_success().item()
    env.sim.spawn_result.get_simulation_time = lambda: 1 / 60
    assert not env.is_task_success().item()
    env.sim.spawn_result.get_simulation_time = lambda: 2 / 60
    env.sim.get_deformable_object = lambda uid: SimpleNamespace(
        data=SimpleNamespace(nodal_pos_w=nodes)
    )
    assert not env.is_task_success().item()


def test_one_canonical_cloth_measurement_reads_nodes_once_and_checks_groups():
    from embodichain.lab.gym.utils.trajectory_state import (
        _validate_evaluation_observation_bindings,
    )

    env, nodes = _measured_fold()
    env._active_task_program_bridge.program_completed = True
    reads = []

    class Data:
        @property
        def nodal_pos_w(self):
            reads.append(1)
            return nodes

    env._physical_replay.surface.data = Data()
    assert env.is_task_success().item()
    assert len(reads) == 1
    assert env.is_task_success().item()
    assert len(reads) == 2
    # The constructor rejects a native topology that cannot supply all groups.
    nodes = nodes[:, :3]
    with pytest.raises(ValueError, match="topology changed|particle index"):
        _validate_evaluation_observation_bindings(env, env._evaluation_recording)
    assert not env.is_task_success().item()


@pytest.mark.parametrize("factory", ["missing", "weakened"])
def test_cloth_constructor_requires_matching_registered_final_rules(factory):
    from embodichain.lab.gym.envs import BaseEnv, EmbodiedEnv, EmbodiedEnvCfg
    from embodichain.lab.gym.envs._final_acceptance import _FinalAcceptance
    from embodichain.lab.evaluation import EvaluationSpec, builtin_predicates

    declaration = _FinalAcceptance("cloth", (((0,), (1,)),), 0.015, "a" * 64)
    cfg = EmbodiedEnvCfg()
    cfg.task_program = object()
    cfg.final_acceptance = declaration
    adapter = None
    if factory == "weakened":
        payload = declaration.evaluation()
        payload["graph"]["rules"]["centroid_pair_0"]["params"]["tolerance"] = 0.02
        graph = builtin_predicates().compile(EvaluationSpec.from_dict(payload["graph"]))
        adapter = SimpleNamespace(
            registration=SimpleNamespace(
                evaluation_final_refs=tuple(graph.roots),
                evaluation_conditions={ref: graph.select(ref) for ref in graph.roots},
            )
        )
    with patch.object(
        BaseEnv, "__init__", side_effect=AssertionError("native setup reached")
    ):
        with pytest.raises(ValueError, match="registered final acceptance"):
            EmbodiedEnv(cfg, task_program_adapter_factory=adapter)


def test_malformed_cloth_arena_shape_is_unavailable_without_reacquisition():
    env, nodes = _measured_fold()
    env._active_task_program_bridge.program_completed = True
    assert env.is_task_success().item()
    env._physical_replay.surface.data.nodal_pos_w = nodes.repeat(2, 1, 1)
    assert not env.is_task_success().item()


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_shared_cloth_criteria_measure_four_particle_centroids(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pairs = (((0, 1, 2, 3), (4, 5, 6, 7)), ((8, 9, 10, 11), (12, 13, 14, 15)))
    env, nodes = _measured_fold(device=device, pairs=pairs, node_count=16)
    env._active_task_program_bridge.program_completed = True
    symmetric = nodes.new_tensor(
        [[0.001, 0, 0], [-0.001, 0, 0], [0, 0.001, 0], [0, -0.001, 0]]
    )
    nodes[:, 0:4] = nodes[:, 8:12] = symmetric
    nodes[:, 4:8] = symmetric + nodes.new_tensor([0.014, 0, 0])
    nodes[:, 12:16] = symmetric + nodes.new_tensor([0, 0.014, 0])
    assert env.is_task_success().item()
    nodes[:, 12:16, 1] += 0.002
    assert not env.is_task_success().item()
    nodes[:, 12:16, 1] -= 0.002
    nodes[:, 4, 2] = float("nan")
    assert not env.is_task_success().item()
