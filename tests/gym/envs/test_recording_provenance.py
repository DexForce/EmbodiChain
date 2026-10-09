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

"""Training/replay identity contracts using existing trajectory buffers."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
from embodichain.lab.gym.envs.managers.datasets import LeRobotRecorder
from embodichain.data_pipeline.recording import stable_config_hash
from embodichain.lab.sim.motion.expansion.contracts import (
    CandidateIdentity,
    CandidateSpec,
)


def make_env() -> SimpleNamespace:
    env = SimpleNamespace(
        cfg=SimpleNamespace(seed=12, task_program=None, sim_steps_per_control=2),
        step_dt=0.04,
        physics_dt=0.02,
        control_frequency=25,
        num_envs=1,
        device=torch.device("cpu"),
        active_joint_ids=[0, 1],
        robot=SimpleNamespace(uid="arm", dof=2),
        sim=SimpleNamespace(
            _articulations={}, _rigid_objects={}, physics_backend="default"
        ),
    )
    env._demo_episode_metadata = [EmbodiedEnv._new_demo_episode_metadata(env, 0)]
    env._demo_episode_metadata[0]["length"] = 3
    env._traj_buffer = TensorDict(
        {
            "states": {"robot": {"qpos": torch.arange(6).reshape(1, 3, 2)}},
            "actions": torch.zeros(1, 3, 2),
        },
        batch_size=[1, 3],
    )
    env._traj_steps = torch.tensor([3])
    env.get_demo_episode_metadata = lambda row: copy.deepcopy(
        env._demo_episode_metadata[row]
    )
    env._trajectory_action_metadata = lambda: {"action_kind": "expert"}
    return env


def test_environment_identity_is_unique_per_episode_and_shared_per_run() -> None:
    env = make_env()
    first = env._demo_episode_metadata[0]
    second = EmbodiedEnv._new_demo_episode_metadata(env, 0)

    assert UUID(first["episode_uuid"]) != UUID(second["episode_uuid"])
    assert first["run_uuid"] == second["run_uuid"]
    assert first["config_hash"] == second["config_hash"]


def test_episode_override_program_hash_uses_executed_program() -> None:
    env = make_env()
    env.cfg.task_program = {"program_id": "configured_default"}
    actual = {"program_id": "episode_override", "segments": ["pick", "place"]}
    env._active_task_program_bridge = SimpleNamespace(
        _program=actual, expansion_records=[]
    )
    result = SimpleNamespace(
        lengths=[3],
        completed_by_env=[True],
        terminal_reasons=["success"],
        episode_index=1,
        success=[True],
        terminated=[False],
        truncated=[False],
    )

    EmbodiedEnv._end_demo_episode_recording(env, result)

    recorded_hash = env._demo_episode_metadata[0]["program_hash"]
    assert recorded_hash == stable_config_hash(actual)
    assert recorded_hash != stable_config_hash(env.cfg.task_program)


def test_external_payloads_without_identity_do_not_share_worker_episode_index() -> None:
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder._env = make_env()
    recorder.curr_episode = 0

    first = recorder._ensure_episode_identity(0, None)
    second = recorder._ensure_episode_identity(0, None)

    assert first["episode_uuid"] != second["episode_uuid"]
    assert first["run_uuid"] == second["run_uuid"]


def test_executed_expansion_records_preserve_selected_candidate_lineage() -> None:
    env = make_env()
    selected_identity = CandidateIdentity(
        scene_case_id="scene",
        initial_state_id="initial",
        candidate_id="selected",
        geometry_family_id="family",
        source_id="source",
        source_revision="revision",
        template_id="recipe:place:0:0",
        parent_id="reference",
    )
    unselected_identity = CandidateIdentity(
        scene_case_id="scene",
        initial_state_id="initial",
        candidate_id="other",
        geometry_family_id="other-family",
        source_id="other-source",
        source_revision="revision",
        template_id="other-recipe:place:0:0",
    )
    record = SimpleNamespace(
        env_id=0,
        selected_candidate_id="selected",
        candidates=(
            CandidateSpec(unselected_identity),
            CandidateSpec(selected_identity),
        ),
        to_metadata=lambda: {"selected_candidate_id": "selected"},
    )
    env._active_task_program_bridge = SimpleNamespace(
        _program=None, expansion_records=[record]
    )
    result = SimpleNamespace(
        lengths=[3],
        completed_by_env=[True],
        terminal_reasons=["success"],
        episode_index=1,
        success=[True],
        terminated=[False],
        truncated=[False],
    )

    EmbodiedEnv._end_demo_episode_recording(env, result)

    lineage = env._demo_episode_metadata[0]["expansion"][0][
        "selected_candidate_lineage"
    ]
    assert lineage == asdict(selected_identity)


def test_trajectory_metadata_links_source_identity_and_initial_state(
    tmp_path: Path,
) -> None:
    env = make_env()
    path = tmp_path / "source.pt"

    EmbodiedEnv.save_trajectory(env, str(path), env_ids=[0])

    trajectory = torch.load(path, weights_only=False)
    sidecar = env._demo_episode_metadata[0]
    assert trajectory["meta"]["episode_uuid"] == sidecar["episode_uuid"]
    assert trajectory["meta"]["run_uuid"] == sidecar["run_uuid"]
    assert trajectory["meta"]["config_hash"] == sidecar["config_hash"]
    assert sidecar["replay_artifact"]["initial_state_step"] == 0
    assert sidecar["replay_artifact"]["source_episode_uuid"] == sidecar["episode_uuid"]
    assert sidecar["trajectory_path"] == str(path.resolve())
    torch.testing.assert_close(
        trajectory["states"]["robot"]["qpos"][0, 0], torch.tensor([0, 1])
    )
    assert not list(tmp_path.glob(".source.pt.*"))


def test_fragment_retries_keep_stable_identity_and_source_replay(
    tmp_path: Path,
) -> None:
    env = make_env()
    EmbodiedEnv.save_trajectory(env, str(tmp_path / "source.pt"), env_ids=[0])
    metadata = copy.deepcopy(env._demo_episode_metadata[0])
    metadata.update(
        {
            "output_mode": "segment_fragments",
            "segments": [
                {"segment_id": 1, "start_step": 1, "end_step": 3, "success": True}
            ],
        }
    )
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder._env = env
    observations = TensorDict({"qpos": torch.zeros(3, 2)}, batch_size=[3])
    payload = list(
        recorder._episode_payloads(0, observations, torch.zeros(3, 2), {}, metadata)
    )[0]
    restarted = LeRobotRecorder.__new__(LeRobotRecorder)
    restarted._env = env
    retried = list(
        restarted._episode_payloads(0, observations, torch.zeros(3, 2), {}, metadata)
    )[0]
    fragment = payload[-1]

    assert fragment["episode_uuid"] == retried[-1]["episode_uuid"]
    assert fragment["episode_uuid"] != metadata["episode_uuid"]
    assert fragment["parent_episode_uuid"] == metadata["episode_uuid"]
    assert fragment["source_episode_uuid"] == metadata["source_episode_uuid"]
    assert fragment["source_start_step"] == 1
    assert fragment["replay_artifact"]["initial_state_step"] == 0
    assert fragment["source_trajectory_path"] == metadata["trajectory_path"]
