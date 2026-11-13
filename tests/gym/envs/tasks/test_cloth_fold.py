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
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import numpy as np


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
assert config_to_cfg(load_config(path), source_path=path).replay == cfg.replay
assert cfg.robot.uid == "w1"
assert cfg.task_program.program_id == "cloth_single_fold"
assert cfg.replay.load().positions.shape == (991, 24)
from embodichain.lab.gym.utils.composition import resolve_gym_components
from embodichain.lab.task_program.integrations.composition import load_configured_task_program_deployment
physical = resolve_gym_components(load_config(path), base_dir=path.parent)
composed = load_configured_task_program_deployment(
    task_program=physical.config["task_program"],
    skill_profile=physical.embodiment_skill_profile,
    evaluation=physical.evaluation,
    base_dir=path.parent,
)
assert composed.integration.registration.evaluation_final_refs == ("centroid_pair_0", "centroid_pair_1")
assert composed.integration.integration_fingerprint == spec.task_program_adapter_factory.integration_fingerprint
# The normal registration fingerprint protects the rules, with no extra schema.
from copy import deepcopy
weakened = deepcopy(physical.evaluation)
weakened["graph"]["rules"]["centroid_pair_0"]["params"]["tolerance"] = 0.02
other = load_configured_task_program_deployment(
    task_program=physical.config["task_program"], skill_profile=physical.embodiment_skill_profile,
    evaluation=weakened, base_dir=path.parent,
)
assert other.integration.integration_fingerprint != composed.integration.integration_fingerprint
if sys.argv[2] == "True":
    cfg.extensions = {"task_program": None}
    try:
        spec.make(cfg=cfg)
    except ValueError as exc:
        assert "Extensions cannot override" in str(exc)
    else:
        raise AssertionError("Extensions bypassed configured acceptance.")
    cfg.extensions = None
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
    from embodichain.lab.evaluation._deformable import _centroid_field

    rules = {
        f"centroid_pair_{i}": {
            "kind": "near_target",
            "bindings": {
                "position": _centroid_field("cloth", "a" * 64, source),
                "target": _centroid_field("cloth", "a" * 64, target),
            },
            "params": {"tolerance": 0.015, "axes": [0, 1], "inclusive": True},
        }
        for i, (source, target) in enumerate(pairs)
    }
    payload = {"graph": {"roots": list(rules), "rules": rules}}

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
    from embodichain.lab.gym.utils._replay import _ReplayCursor

    replay.snapshot = SimpleNamespace(steps=2, source=SimpleNamespace(fps=60))
    replay.cursor = _ReplayCursor(torch.tensor([2]))
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
    from embodichain.lab.gym.utils._replay import _ReplayCommands, _ReplayCursor

    replay.snapshot = SimpleNamespace(steps=2, source=SimpleNamespace(fps=60))
    replay.cursor = _ReplayCursor(torch.tensor([2]))
    replay.commands = _ReplayCommands(torch.tensor([[[0.0], [0.1], [0.2]]]))
    env.sim = SimpleNamespace(
        spawn_result=SimpleNamespace(get_simulation_time=lambda: 1 / 60)
    )
    hold = ControllerAction(
        torch.tensor([[0.1]]),
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
    env.sim.spawn_result.get_simulation_time = lambda: 3 / 60
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


def _native_input(tmp_path):
    from embodichain.lab.gym.utils._replay import _AuthoredJointStream

    path = tmp_path / "commands.npz"
    np.savez(
        path,
        positions=np.array([[0, 10], [1, 11], [2, 12]], dtype=np.float32),
        joint_names=np.array(["b", "a"]),
    )
    source = _AuthoredJointStream(
        "commands", str(path), hashlib.sha256(path.read_bytes()).hexdigest(), 60
    )
    cfg = SimpleNamespace(
        num_envs=1,
        sim_steps_per_control=1,
        target_control_frequency=None,
        sim_cfg=SimpleNamespace(
            physics_dt=1 / 60,
            physics_cfg=SimpleNamespace(use_cuda_graph=False),
        ),
        expert_trajectory=SimpleNamespace(
            joint_command_mode="position", use_action_manager=False
        ),
        replay_surface=SimpleNamespace(
            mesh_sha256="a" * 64,
            prepare=lambda cfg: SimpleNamespace(uid="cloth"),
        ),
        rl_task=None,
        robot=SimpleNamespace(uid="robot"),
        background=[],
    )
    return source, cfg, path


def test_native_replay_uses_one_snapshot_before_and_after_preparation(tmp_path):
    from embodichain.lab.gym.envs._physical_replay import _PhysicalReplay
    from embodichain.lab.task_program.integrations._kinematic_replay import (
        _KinematicReplayFactory,
    )

    source, cfg, path = _native_input(tmp_path)
    prepared = False
    events = []
    clock = [0.0]
    resets = []

    class Robot:
        uid = "robot"
        device = torch.device("cpu")

        @property
        def joint_names(self):
            assert prepared, "Native joint metadata was read before prepare."
            return ["a", "b"]

        def set_qpos(self, value, **kwargs):
            resets.append(value.clone())

        def clear_dynamics(self, **kwargs):
            pass

    robot = Robot()

    def register(uid, positions, **kwargs):
        assert not prepared
        events.append("register")
        assert kwargs["joint_names"] == ("b", "a")
        np.testing.assert_array_equal(positions[0], [[0, 10], [1, 11], [2, 12]])

    sim = SimpleNamespace(
        add_deformable_object=lambda cfg: SimpleNamespace(),
        register_kinematic_joint_trajectory=register,
        spawn_result=SimpleNamespace(
            world=object(), get_simulation_time=lambda: clock[0]
        ),
    )
    env = SimpleNamespace(
        cfg=cfg,
        robot=robot,
        sim=sim,
        add_detached_uids_for_reset=lambda ids: None,
    )
    read_bytes = Path.read_bytes
    with patch.object(
        Path, "read_bytes", autospec=True, side_effect=read_bytes
    ) as read:
        replay = _PhysicalReplay(cfg, source)
        replay.prepare_scene(env)
        events.append("prepare")
        prepared = True
        replay.bind(env)
        env._replay_snapshot = lambda source_id: replay.snapshot
        path.write_bytes(b"changed after initialization")
        factory = _KinematicReplayFactory("commands", (("all", 0, 2),))
        lowerer = factory.create_for_environment(
            environment=env,
            simulation=sim,
            robot=robot,
            scene_registry=None,
            engine=SimpleNamespace(robot=robot),
        )
        lowered = lowerer.lower(
            SimpleNamespace(arguments={"phase": "all"}),
            context=SimpleNamespace(
                batch_size=1,
                require_control_dt=lambda: 1 / 60,
                env_ids=torch.tensor([0]),
            ),
            bound=None,
            option_template=None,
        )
        assert read.call_count == 1
    assert events == ["register", "prepare"]
    torch.testing.assert_close(
        lowered.goal.target.positions, torch.tensor([[[11.0, 1.0], [12.0, 2.0]]])
    )
    replay.consume(env, torch.tensor([[11.0, 1.0]]))
    assert replay.cursor.steps.tolist() == [0]
    clock[0] = 1 / 60
    replay.consume(env, torch.tensor([[12.0, 2.0]]))
    clock[0] = 2 / 60
    with pytest.raises(ValueError, match="differs from native replay"):
        replay.consume(env, torch.tensor([[12.0, 2.0]]))
    backend = SimpleNamespace(set_sim_time=lambda *args: clock.__setitem__(0, 0.0))
    with patch(
        "dexsim.engine.newton_physics.backend_registry.get_newton_backend",
        return_value=backend,
    ):
        replay.reset(env, [0])
    assert replay.cursor.steps.tolist() == [0]
    assert len(resets) == 2
    torch.testing.assert_close(resets[0], torch.tensor([[10.0, 0.0]]))
    replay.consume(env, torch.tensor([[11.0, 1.0]]))
    assert replay.snapshot.metadata()["source_joint_names"] == ["b", "a"]
    assert replay.snapshot.metadata()["joint_names"] == ["a", "b"]
    with pytest.raises(ValueError, match="digest"):
        source.load()


def test_native_replay_resolves_frequency_and_rejects_unsupported_modes(tmp_path):
    from embodichain.lab.gym.envs._physical_replay import _PhysicalReplay

    source, cfg, path = _native_input(tmp_path)
    cfg.sim_steps_per_control = 4
    cfg.target_control_frequency = 60
    _PhysicalReplay(cfg, source)
    cfg.target_control_frequency = 30
    with patch.object(type(source), "load", side_effect=AssertionError("loaded")):
        with pytest.raises(ValueError, match="resolved recorded rate"):
            _PhysicalReplay(cfg, source)
        cfg.target_control_frequency = 60
        cfg.expert_trajectory.joint_command_mode = "position_velocity"
        with pytest.raises(ValueError, match="position-only"):
            _PhysicalReplay(cfg, source)


@pytest.mark.parametrize("extra", ["qvel", "qf"])
@pytest.mark.parametrize("hold", [None, "runtime_abort_safe_hold"])
def test_native_replay_rejects_extra_targets_before_consuming_a_frame(extra, hold):
    from tensordict import TensorDict
    from embodichain.lab.gym.envs.types import ControllerAction
    from embodichain.lab.gym.utils._replay import _ReplayCommands

    env, nodes = _measured_fold()
    replay = env._physical_replay
    replay.hold = False
    replay.commands = _ReplayCommands(torch.tensor([[[0.0], [0.1], [0.2]]]))
    action = ControllerAction(
        TensorDict({"qpos": torch.tensor([[0.1]]), extra: torch.tensor([[6.0]])}, [1]),
        metadata={"bridge_action_kind": hold},
    )
    with pytest.raises(ValueError, match="only qpos"):
        replay.consume(env, action)
    assert not replay.hold
    assert replay.cursor.steps.tolist() == [0]
