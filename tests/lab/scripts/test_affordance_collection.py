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

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodichain.lab.gym.envs.augmentation import AffordanceAugmentationCfg
from embodichain.lab.gym.envs.managers.episode_commit import DemoCommitReceipt
from embodichain.lab.scripts import _affordance_collection as collector
from embodichain.lab.scripts import run_env


class _Env:
    num_envs = 4

    def __init__(self, path, *, max_batches=8):
        self.unwrapped = self
        self.policy = AffordanceAugmentationCfg(branches=4, max_batches=max_batches)
        self.cfg = SimpleNamespace(
            seed=7,
            expert_trajectory=SimpleNamespace(affordance_augmentation=self.policy),
        )
        self.events = []
        self.commits = []
        self.receipts = []
        self.program = SimpleNamespace(program_id="pick_place")
        self.task_program_adapter = SimpleNamespace(
            validate_measured_acceptance=lambda program: self.events.append("validate")
        )
        self.dataset_manager = SimpleNamespace(
            episode_commit_path=path,
            episode_commit_receipts=self.receipts,
            finalize=self.finalize,
        )
        self.commit_error = None
        self.finalize_error = None
        self.reset_error = None

    def prepare_affordance_collection(self):
        self.events.append("preflight")
        return self.program

    def reset(self, *, options):
        assert options == {"save_data": False}
        assert getattr(self, "_affordance_sampling_context", None) is None
        self.events.append("discard")
        if self.reset_error:
            raise self.reset_error

    def select_affordance_episode_rows(self, result, program, remaining):
        assert program is self.program
        return result.rows[:remaining]

    def commit_demo_rows(self, rows):
        self.events.append("commit")
        self.commits.append(rows)
        added = []
        for row in rows:
            index = len(self.receipts)
            receipt = DemoCommitReceipt(
                row, f"episode-{index}", f"commit-{index}", index, ("primary",)
            )
            self.receipts.append(receipt)
            added.append(receipt)
            if self.commit_error:
                raise self.commit_error
        return tuple(added)

    def finalize(self):
        self.events.append("finalize")
        if self.finalize_error:
            raise self.finalize_error


def _run(monkeypatch, env, outcomes, *, quota=3, attempts=3):
    calls = []

    def execute(*args, **kwargs):
        assert env._affordance_sampling_context is not None
        assert env._affordance_collection_metadata is not None
        env.events.append("execute")
        calls.append((kwargs["episode_index"], kwargs["attempt_id"]))
        result = next(outcomes)
        if isinstance(result, BaseException):
            raise result
        return SimpleNamespace(
            rows=result,
            terminal_reason="done" if result else "failed",
            to_metadata=lambda: {"rows": list(result)},
        )

    monkeypatch.setattr(collector, "execute_demo_episode", execute)
    collector._collect_affordance_episodes(
        SimpleNamespace(gym_config="task.yaml"),
        env,
        {"id": "test-v0", "max_episodes": quota, "demo_max_attempts": attempts},
    )
    return calls


def _manifest(path):
    paths = list(path.glob("affordance_collection_*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text())


def test_noncontiguous_successes_and_tail_count_receipts(monkeypatch, tmp_path):
    env = _Env(tmp_path)
    calls = _run(monkeypatch, env, iter([(1, 3), (0, 2, 3)]))
    assert env.commits == [(1, 3), (0,)]
    assert calls == [(0, 0), (1, 0)]
    manifest = _manifest(tmp_path)
    assert manifest["committed"] == 3
    assert manifest["status"] == "complete"
    assert env.events[-1] == "finalize"
    assert env._affordance_sampling_context is None


@pytest.mark.parametrize("attempts,max_batches,expected", [(5, 2, 2), (2, 8, 2)])
def test_all_failures_are_bounded(
    monkeypatch, tmp_path, attempts, max_batches, expected
):
    env = _Env(tmp_path, max_batches=max_batches)
    with pytest.raises(RuntimeError, match="(?i)(attempt|batch)"):
        _run(monkeypatch, env, iter([()] * 10), attempts=attempts)
    manifest = _manifest(tmp_path)
    assert manifest["attempts"] == expected
    assert manifest["committed"] == 0
    assert manifest["status"] == "failed"
    assert env.events.count("execute") == expected
    assert env._affordance_collection_metadata is None


def test_zero_quota_never_rolls_out_or_resets(monkeypatch, tmp_path):
    env = _Env(tmp_path)
    assert _run(monkeypatch, env, iter([]), quota=0) == []
    assert env.events == ["preflight", "finalize"]
    assert _manifest(tmp_path)["committed"] == 0


def test_partial_commit_failure_never_reexecutes(monkeypatch, tmp_path):
    env = _Env(tmp_path)
    env.commit_error = OSError("sidecar unavailable")
    with pytest.raises(OSError, match="sidecar unavailable"):
        _run(monkeypatch, env, iter([(1, 3)]))
    assert env.events.count("execute") == 1
    manifest = _manifest(tmp_path)
    assert manifest["committed"] == 0
    assert len(manifest["receipts"]) == 0
    assert len(manifest["dataset_receipts"]) == 1
    assert manifest["status"] == "failed"
    assert env._affordance_sampling_context is None
    assert "discard" in env.events[env.events.index("commit") + 1 :]


def test_finalization_failure_never_reports_complete(monkeypatch, tmp_path, capsys):
    env = _Env(tmp_path)
    env.finalize_error = OSError("drain failed")
    with pytest.raises(OSError, match="drain failed"):
        _run(monkeypatch, env, iter([(1, 2, 3)]))
    assert "Collection complete" not in capsys.readouterr().out
    assert _manifest(tmp_path)["status"] == "failed"


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("rollout failed")])
def test_execution_error_discards_after_context_clears(monkeypatch, tmp_path, error):
    env = _Env(tmp_path)
    with pytest.raises(type(error)):
        _run(monkeypatch, env, iter([error]))
    assert env.events[-2:] == ["discard", "finalize"]
    assert env._affordance_collection_metadata is None
    assert _manifest(tmp_path)["status"] != "complete"


def test_main_routes_before_legacy_reset(monkeypatch, tmp_path):
    env = _Env(tmp_path)
    calls = []
    monkeypatch.setattr(
        collector, "_collect_affordance_episodes", lambda *args: calls.append(args)
    )
    run_env.main(SimpleNamespace(), env, {"max_episodes": 1})
    assert len(calls) == 1
    assert env.events == []


def test_retry_identity_advances_without_committed_counter(monkeypatch, tmp_path):
    env = _Env(tmp_path)
    calls = _run(monkeypatch, env, iter([(), (3,), (1, 2)]))
    assert calls == [(0, 0), (0, 1), (1, 0)]
    assert env.events.count("validate") == 3
    assert _manifest(tmp_path)["attempts"] == 3


def test_preflight_failure_precedes_any_reset(monkeypatch, tmp_path):
    env = _Env(tmp_path)

    def reject():
        raise ValueError("unsupported sink")

    env.prepare_affordance_collection = reject
    with pytest.raises(ValueError, match="unsupported sink"):
        _run(monkeypatch, env, iter([]))
    assert env.events == []


def test_primary_error_survives_discard_finalize_and_manifest_failures(
    monkeypatch, tmp_path
):
    env = _Env(tmp_path)
    original_write = collector._write_manifest

    def fail_report(path, manifest):
        if manifest["status"] == "running":
            original_write(path, manifest)
        else:
            raise OSError("report failed")

    monkeypatch.setattr(collector, "_write_manifest", fail_report)

    def fail_execute(*args, **kwargs):
        env.reset_error = OSError("discard failed")
        raise RuntimeError("primary failure")

    monkeypatch.setattr(collector, "execute_demo_episode", fail_execute)
    env.finalize_error = OSError("finalize failed")
    with pytest.raises(RuntimeError, match="primary failure") as caught:
        collector._collect_affordance_episodes(
            SimpleNamespace(), env, {"max_episodes": 1}
        )
    notes = " ".join(caught.value.__notes__)
    assert "discard failed" in notes
    assert "finalize failed" in notes
    assert "report failed" in notes
    assert env._affordance_sampling_context is None


def test_dataset_success_trajectory_failure_does_not_satisfy_quota(
    monkeypatch, tmp_path
):
    env = _Env(tmp_path)
    original_commit = env.commit_demo_rows

    def fail_trajectory(rows):
        original_commit(rows)
        raise OSError("trajectory failed")

    env.commit_demo_rows = fail_trajectory
    with pytest.raises(OSError, match="trajectory failed"):
        _run(monkeypatch, env, iter([(1, 2, 3)]))
    manifest = _manifest(tmp_path)
    assert manifest["committed"] == 0
    assert len(manifest["dataset_receipts"]) == 3
    assert env.events.count("execute") == 1


def test_duplicate_receipts_are_rejected(monkeypatch, tmp_path):
    env = _Env(tmp_path)
    original_commit = env.commit_demo_rows

    def duplicate_receipts(rows):
        receipts = original_commit(rows)
        return (receipts[0],) * len(rows)

    env.commit_demo_rows = duplicate_receipts
    with pytest.raises(RuntimeError, match="uniquely confirm"):
        _run(monkeypatch, env, iter([(1, 2, 3)]))
    assert _manifest(tmp_path)["committed"] == 0


def test_manifest_and_episode_share_configuration_fingerprint(monkeypatch, tmp_path):
    env = _Env(tmp_path)
    original_select = env.select_affordance_episode_rows
    episode_fingerprints = []

    def select(result, program, remaining):
        episode_fingerprints.append(
            env._affordance_collection_metadata["config_sha256"]
        )
        return original_select(result, program, remaining)

    env.select_affordance_episode_rows = select
    _run(monkeypatch, env, iter([(1, 2, 3)]))
    assert episode_fingerprints == [_manifest(tmp_path)["config_sha256"]]
    assert len(episode_fingerprints[0]) == 64


def _resolved_fingerprint_inputs():
    from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
    from embodichain.lab.task_program import load_task_program
    from embodichain.lab.task_program.integrations._configured_composition import (
        _compose_integration_payload,
        _load_configured_task_program_deployment,
        _resolve_task_program_components,
    )
    from embodichain.utils.utility import load_config

    path = (
        Path(__file__).parents[3]
        / "embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml"
    )
    physical = _resolve_gym_components(load_config(path), base_dir=path.parent)
    deployment = _load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=path.parent,
    )
    _, task, policy = _resolve_task_program_components(
        physical.config["task_program"], base_dir=path.parent
    )
    integration_payload = _compose_integration_payload(
        task=task,
        policy=policy,
        skill_profile=physical.embodiment_skill_profile,
        scene=task["scene_binding"],
    )
    return (
        load_task_program(deployment.program_path, integration=deployment.selection),
        integration_payload,
    )


@pytest.mark.parametrize(
    "change", ["program_geometry", "release_profile", "embodiment", "execution_mode"]
)
def test_fingerprint_changes_for_loaded_content_with_same_source_ids(
    monkeypatch, tmp_path, change
):
    from embodichain.lab.gym.envs.expert_trajectory import ExpertTrajectoryCfg
    from embodichain.lab.task_program.integrations.configured import (
        _decode_configured_task_program_integration,
    )

    program, integration_payload = _resolved_fingerprint_inputs()
    fingerprints = []
    for run in range(3):
        env = _Env(tmp_path / str(run))
        env.cfg.task_program = deepcopy(program)
        env.cfg.expert_trajectory = ExpertTrajectoryCfg(
            affordance_augmentation=env.policy
        )
        env.cfg.robot = {"uid": "same_robot", "asset": "same.urdf", "scale": 1.0}
        payload = deepcopy(integration_payload)
        if run == 2:
            if change == "program_geometry":
                target = next(iter(env.cfg.task_program.targets.values()))
                pose = target.values[0]
                pose.position = (pose.position[0] + 0.1, *pose.position[1:])
            elif change == "release_profile":
                payload["robot_profile"]["presets"][0]["action_options"]["place"][
                    "release_settle_steps"
                ] = 17
            elif change == "embodiment":
                env.cfg.robot["scale"] = 1.5
            else:
                env.cfg.expert_trajectory.joint_command_mode = "position_velocity"
        integration = _decode_configured_task_program_integration(payload)
        env.task_program_adapter._registration = integration.registration
        env.task_program_adapter._configured_integration_fingerprint = (
            integration.integration_fingerprint
        )
        _run(monkeypatch, env, iter([]), quota=0)
        fingerprints.append(_manifest(tmp_path / str(run))["config_sha256"])
    assert fingerprints[0] == fingerprints[1]
    assert fingerprints[0] != fingerprints[2]


def test_actual_resolved_augmentation_config_has_stable_fingerprint(
    monkeypatch, tmp_path
):
    from gymnasium.envs.registration import registry as gym_registry
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
    from embodichain.utils.utility import load_config

    path = (
        Path(__file__).parents[3]
        / "embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.augmentation.yaml"
    )
    payload = load_config(path)
    env_id = "Collector-Fingerprint-Resolved-v0"
    payload["id"] = env_id
    try:
        cfg = config_to_cfg(payload, source_path=path)
        spec = REGISTERED_ENVS[env_id]
        # The Gym spec does not contain the locally injected adapter factory.
        assert "task_program_adapter_factory" not in gym_registry[env_id].kwargs
        fingerprints = []
        for index in range(2):
            env = _Env(tmp_path / str(index))
            env.cfg = deepcopy(cfg)
            env.spec = gym_registry[env_id]
            env.task_program_adapter._registration = spec.task_program_registration
            env.task_program_adapter._configured_integration_fingerprint = (
                spec.task_program_adapter_factory.integration_fingerprint
            )
            env.task_program_adapter.step_dt = 0.02
            _run(monkeypatch, env, iter([]), quota=0)
            fingerprints.append(_manifest(tmp_path / str(index))["config_sha256"])
        assert fingerprints[0] == fingerprints[1]
    finally:
        REGISTERED_ENVS.pop(env_id, None)
        gym_registry.pop(env_id, None)


def test_configured_factory_retains_loaded_fingerprint_on_adapter(monkeypatch):
    from embodichain.lab.task_program.integrations.configured import (
        _decode_configured_task_program_integration,
    )
    from embodichain.lab.task_program.integrations.simulation.environment import (
        SimulationTaskProgramAdapterFactory,
    )

    _, payload = _resolved_fingerprint_inputs()
    integration = _decode_configured_task_program_integration(payload)
    adapter = SimpleNamespace()
    monkeypatch.setattr(
        SimulationTaskProgramAdapterFactory,
        "create_adapter",
        lambda self, environment: adapter,
    )
    returned = integration.adapter_factory.create_adapter(object())
    assert returned is adapter
    assert (
        returned._configured_integration_fingerprint
        == integration.integration_fingerprint
    )
