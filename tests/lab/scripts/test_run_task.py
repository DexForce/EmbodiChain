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
import re
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from embodichain.lab.gym.envs.demo import (
    DemoEpisodeResult,
    DemoExecutionCfg,
    DemoSegment,
    DemoSegmentResult,
)
from embodichain.lab.task_program.language.loader import (
    load_task_program as _load_task_program,
)
from embodichain.lab.gym.utils.gym_utils import merge_args_with_gym_config
from embodichain.lab.scripts import run_task
from embodichain.lab.scripts.run_task import (
    _create_parser,
    _run_replay_control_loop,
    generate_function,
)

GYM_CONFIG_PATH = "task.yaml"
GYM_ID = "Dummy-v0"
EPISODE_INDEX = 3
ACTION_LIST_INDEX = 0
REPLAY_NUM_STEPS = 5
REPLAY_TARGET_STEP = 3
VISER_POLL_INTERVAL = 0.05
REPEATED_PICK_PLACE_SEGMENT_COUNT = 3
REPEATED_PICK_PLACE_SEGMENT_STEP_COUNT = 80


def _task_program_payload() -> dict[str, object]:
    """Return one minimal strict Task Program payload."""
    return {
        "program_id": "cli_pick",
        "integration": {
            "robot_profile": "default_robot",
            "scene_registry": "default_scene",
            "runtime_preset": "default_runtime",
        },
        "targets": {},
        "program": {
            "kind": "invoke",
            "call": {"kind": "pick", "object": "cube"},
        },
    }


class _LegacyProgressEnv:
    num_envs = 1

    @property
    def unwrapped(self):
        return self

    def get_wrapper_attr(self, name: str):
        if name == "create_demo_action_list":
            return lambda **kwargs: [object()]
        raise AttributeError(name)

    def step(self, action):
        return None, None, False, False, {"success": False}

    def is_task_success(self) -> bool:
        return True


class _TaskProgramProgressEnv(_LegacyProgressEnv):
    """Three-segment Task Program stand-in matching repeated pick/place."""

    def create_demo_segments(self, **kwargs):
        del kwargs
        return tuple(
            DemoSegment(
                actions=(
                    object() for _ in range(REPEATED_PICK_PLACE_SEGMENT_STEP_COUNT)
                ),
                name="move_cube",
                metadata={
                    "segment_count": REPEATED_PICK_PLACE_SEGMENT_COUNT,
                },
                progress_total_steps=REPEATED_PICK_PLACE_SEGMENT_STEP_COUNT,
            )
            for _ in range(REPEATED_PICK_PLACE_SEGMENT_COUNT)
        )


def test_legacy_action_list_displays_episode_and_segment_indices(
    monkeypatch,
) -> None:
    """Progress output distinguishes episodes from their local segments."""
    env = _LegacyProgressEnv()
    progress = MagicMock(side_effect=lambda actions, **kwargs: actions)
    monkeypatch.setattr(run_task.tqdm, "tqdm", progress)

    generated = run_task.generate_and_execute_action_list(
        env,
        ACTION_LIST_INDEX,
        debug_mode=False,
        episode_idx=EPISODE_INDEX,
    )

    assert generated
    assert progress.call_args.kwargs["desc"] == (
        f"Ep {EPISODE_INDEX} · Seg {ACTION_LIST_INDEX + 1}/1 · legacy"
    )
    assert len(progress.call_args.args[0]) == 1
    assert progress.call_args.kwargs["file"] is sys.stdout
    assert progress.call_args.kwargs["dynamic_ncols"] is True


def test_task_program_progress_displays_repeated_segment_position(capsys) -> None:
    """Each lazy segment renders its position, percentage, and exact step count."""
    env = _TaskProgramProgressEnv()

    generated = run_task.generate_and_execute_action_list(
        env,
        ACTION_LIST_INDEX,
        debug_mode=False,
        episode_idx=EPISODE_INDEX,
    )

    assert generated
    output = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    for index in range(1, REPEATED_PICK_PLACE_SEGMENT_COUNT + 1):
        assert (
            f"Steps  [fixed]    Ep {EPISODE_INDEX} · Seg {index}/"
            f"{REPEATED_PICK_PLACE_SEGMENT_COUNT} · move_cube 100%"
        ) in output
    total = REPEATED_PICK_PLACE_SEGMENT_STEP_COUNT
    assert output.count(f"{total}/{total}") == REPEATED_PICK_PLACE_SEGMENT_COUNT


def test_progress_wrapper_formats_known_step_total(capsys) -> None:
    """Sized action sequences render a complete step progress bar."""
    actions = (object(), object())

    for _ in run_task._progress_wrapper(actions, "Ep 1 · Seg 1/1 · move_cube"):
        pass

    output = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    assert "Steps  [fixed]    Ep 1 · Seg 1/1 · move_cube" in output
    assert "100%" in output
    assert "2/2" in output
    assert " step/s]" in output


def test_progress_wrapper_formats_unknown_step_total_as_activity(capsys) -> None:
    """Lazy action streams finish an indeterminate bar without a percentage."""
    actions = (object() for _ in range(2))

    for _ in run_task._progress_wrapper(
        actions, "Ep 1 · Seg 1/1 · pour_and_return_bottle"
    ):
        pass

    output = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    assert (
        "Steps  [dynamic]  Ep 1 · Seg 1/1 · pour_and_return_bottle "
        "│━━━━━━━━━━│ ✓ 2 steps"
    ) in output
    assert " step/s]" in output
    assert "0%" not in output


def test_run_task_syncs_viser_images_each_step_by_default() -> None:
    """Run-task uses step-synchronized camera images when no FPS is supplied."""
    args = _create_parser().parse_args(["--gym_config", GYM_CONFIG_PATH, "--viser"])

    merged = merge_args_with_gym_config(args, {"id": GYM_ID, "physics": "default"})

    assert merged["visualization"]["sensor_image_fps"] is None


def test_run_task_accepts_explicit_viser_image_fps() -> None:
    """An explicit image FPS restores wall-clock rate limiting."""
    expected_fps = 6.0
    args = _create_parser().parse_args(
        [
            "--gym_config",
            GYM_CONFIG_PATH,
            "--viser",
            "--viser-image-fps",
            str(expected_fps),
        ]
    )

    merged = merge_args_with_gym_config(args, {"id": GYM_ID, "physics": "default"})

    assert merged["visualization"]["sensor_image_fps"] == expected_fps


def test_run_task_preserves_configured_viser_image_fps() -> None:
    """A file-based rate overrides the run-task step-synchronized default."""
    configured_fps = 4.0
    args = _create_parser().parse_args(["--gym_config", GYM_CONFIG_PATH, "--viser"])

    merged = merge_args_with_gym_config(
        args,
        {
            "id": GYM_ID,
            "physics": "default",
            "visualization": {"sensor_image_fps": configured_fps},
        },
    )

    assert merged["visualization"]["sensor_image_fps"] == configured_fps


def test_run_task_parser_accepts_task_program_path() -> None:
    """The declarative program is an explicit, opt-in CLI input."""
    program_path = "program.yaml"

    args = _create_parser().parse_args(
        ["--gym_config", GYM_CONFIG_PATH, "--task-program", program_path]
    )

    assert args.task_program == program_path


def test_run_task_parser_accepts_debug_trace_mode() -> None:
    """Failed Task Program attempts can expose their structured trace."""
    args = _create_parser().parse_args(
        ["--gym_config", GYM_CONFIG_PATH, "--debug-mode"]
    )

    assert args.debug_mode is True


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_load_task_program_safely_decodes_supported_files(
    tmp_path,
    suffix: str,
) -> None:
    """JSON and safe YAML inputs share the same strict schema decoder."""
    path = tmp_path / f"program{suffix}"
    payload = _task_program_payload()
    if suffix == ".json":
        serialized = json.dumps(payload)
    else:
        import yaml

        serialized = yaml.safe_dump(payload)
    path.write_text(serialized, encoding="utf-8")

    program = _load_task_program(path)

    assert program.program_id == "cli_pick"
    assert program.integration.scene_registry == "default_scene"


@pytest.mark.parametrize(
    ("filename", "serialized", "message"),
    [
        (
            "program.json",
            '{"program_id": "one", "program_id": "two"}',
            "Duplicate JSON key",
        ),
        (
            "program.yaml",
            "program_id: one\nprogram_id: two\n",
            "found duplicate key",
        ),
    ],
)
def test_load_task_program_rejects_duplicate_mapping_keys(
    tmp_path,
    filename: str,
    serialized: str,
    message: str,
) -> None:
    """Ambiguous duplicate keys are rejected before schema decoding."""
    path = tmp_path / filename
    path.write_text(serialized, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_task_program(path)


def test_load_task_program_rejects_unsupported_file_extension(tmp_path) -> None:
    """Only explicit JSON and YAML file formats are accepted."""
    path = tmp_path / "program.toml"
    path.write_text('program_id = "cli_pick"', encoding="utf-8")

    with pytest.raises(ValueError, match=".json, .yaml, or .yml"):
        _load_task_program(path)


def test_replay_restores_wrapper_state_without_closing_caller_env(monkeypatch) -> None:
    """Replay leaves the environment close to its CLI owner."""
    env = MagicMock()
    env.unwrapped = env
    replay_env = MagicMock()
    monkeypatch.setattr(
        run_task,
        "load_trajectory",
        lambda path: {"meta": {"num_envs": 1, "num_steps": 1, "lengths": [1]}},
    )
    monkeypatch.setattr(run_task, "ReplayWrapper", lambda *args, **kwargs: replay_env)
    monkeypatch.setattr(run_task, "replay_auto", lambda *args, **kwargs: None)

    run_task.replay(env, "trajectory.pt")

    replay_env.close.assert_not_called()
    replay_env.env.sim.enable_physics.assert_called_once_with(True)
    assert replay_env.env._replay_no_auto_reset is False


def test_preview_quit_returns_without_zero_exit(monkeypatch) -> None:
    """Preview quit lets CLI cleanup failures determine the process status."""
    env = MagicMock()
    env.reset.return_value = (None, {})
    monkeypatch.setattr("builtins.input", lambda: "q")

    run_task.preview(env)

    env.reset.assert_called_once_with()


class _ResetTrackingEnv:
    def __init__(
        self,
        *,
        num_envs: int = 1,
        save_failed_episodes: bool = False,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.dataset_manager = SimpleNamespace(
            save_failed_episodes=save_failed_episodes
        )
        self.reset_options = []

    def reset(self, options=None):
        self.reset_options.append(options)
        return None, {}


class _LifecycleTrackingEnv(_ResetTrackingEnv):
    def __init__(self) -> None:
        super().__init__()
        self.events = []

    def reset(self, options=None):
        self.events.append(("reset", options))
        return super().reset(options=options)

    def close(self, *, exit_process=None) -> None:
        self.events.append(("close", exit_process))


def _episode_result(
    *,
    success: bool,
    reason: str,
    length: int = 2,
    num_envs: int = 1,
) -> DemoEpisodeResult:
    return DemoEpisodeResult(
        episode_index=0,
        length=length,
        completed=success,
        success=(success,) * num_envs,
        terminated=(False,) * num_envs,
        truncated=(False,) * num_envs,
        terminal_reason=reason,
        lengths=(length,) * num_envs,
    )


def test_generate_function_discards_retry_then_commits_once(monkeypatch) -> None:
    """Failed attempts are discarded and only the complete episode is saved."""
    env = _ResetTrackingEnv()
    results = iter(
        [
            _episode_result(success=False, reason="empty_plan"),
            _episode_result(success=True, reason="success"),
        ]
    )
    monkeypatch.setattr(
        "embodichain.lab.scripts.run_task.execute_demo_episode",
        lambda *args, **kwargs: next(results),
    )

    generated = generate_function(
        env,
        time_id=3,
        max_attempts=2,
        reset_before=False,
    )

    assert generated
    assert env.reset_options == [{"save_data": False}, None]


def test_source_bound_recording_cannot_bypass_real2sim_receipt() -> None:
    env = _ResetTrackingEnv()
    env._task_program_adapter = SimpleNamespace(_configured_nominal_sources=(object(),))
    with pytest.raises(ValueError, match="Real2Sim receipt"):
        generate_function(env, reset_before=False)
    assert env.reset_options == []

    env._task_program_adapter = None
    with pytest.raises(ValueError, match="Real2Sim receipt"):
        generate_function(env, reset_before=False, nominal_source_factory=object())
    assert env.reset_options == []


def test_real2sim_dataset_path_is_owned_by_profile_not_dataset_manager() -> None:
    args = _create_parser().parse_args(
        [
            "--gym-config",
            "task.yaml",
            "--real2sim-profile",
            "profile.yaml",
            "--dataset-dir",
            "/tmp/real2sim-dataset",
        ]
    )
    merged = merge_args_with_gym_config(
        args, {"id": "Dummy-v0", "physics": "default", "env": {"dataset": {}}}
    )
    assert merged["env"]["dataset"] == {}


def test_real2sim_run_task_preserves_failed_write_without_rerunning(
    monkeypatch,
    tmp_path,
) -> None:
    from dataclasses import replace

    from embodichain.lab.gym.envs.managers import real2sim_collection
    from embodichain.lab.gym.envs.managers.real2sim_collection import (
        Real2SimCaptureRequest,
        Real2SimRunTaskOwner,
        load_staged_real2sim_episode,
    )
    from embodichain.lab.gym.envs.managers.real2sim_sink import (
        Real2SimLeRobotEpisodeSink,
    )
    from embodichain.lab.sim.motion.expansion import (
        CandidateIdentity,
        CommitReceipt,
        ExpertEpisode,
        ValidationCheck,
        ValidationResult,
    )

    env = _ResetTrackingEnv()
    sink = object.__new__(Real2SimLeRobotEpisodeSink)
    sink._recorder = SimpleNamespace(_env=env, dataset_full_path=tmp_path)
    identity = CandidateIdentity(
        "scene", "initial", "candidate", "geometry", "source", "1", "unit"
    )
    validation = ValidationResult((ValidationCheck("task", "passed"),))
    request = Real2SimCaptureRequest(
        identity,
        validation,
        "episode",
        "commit",
        {"qpos": ("robot", "qpos")},
        "joint_position",
        {},
    )
    episode = ExpertEpisode(
        identity=identity,
        observations={"qpos": torch.tensor([[0.0], [1.0]])},
        actions=torch.tensor([[1.0]]),
        timestamps=torch.tensor([0.0, 0.1]),
        action_representation="joint_position",
        validation=validation,
        episode_id="episode",
        commit_id="commit",
    )
    failed = CommitReceipt(
        episode_id="episode",
        candidate_id="candidate",
        attempt_id=0,
        storage_id="store",
        commit_id="commit",
        scene_case_id="scene",
        confirmed=False,
        error="injected attachment failure",
        submission_id=0,
    )
    confirmed = replace(failed, confirmed=True, error="", submission_id=1)
    submissions = iter((failed, confirmed))
    monkeypatch.setattr(
        real2sim_collection, "capture_real2sim_episode", lambda *a, **k: episode
    )
    submit = MagicMock(side_effect=lambda *a, **k: next(submissions))
    monkeypatch.setattr(
        real2sim_collection, "submit_and_discard_real2sim_episode", submit
    )
    execute = MagicMock(
        return_value=_episode_result(success=True, reason="success", length=1)
    )
    monkeypatch.setattr(run_task, "execute_demo_episode", execute)
    prepare = MagicMock(return_value=())
    owner = Real2SimRunTaskOwner(
        sink, prepare_attempt=prepare, capture_request=lambda *_: request
    )

    with pytest.raises(RuntimeError, match="unconfirmed"):
        generate_function(
            env, reset_before=False, max_attempts=2, _real2sim_owner=owner
        )
    assert env.reset_options == []
    assert owner.pending_episode is not None
    assert owner.staged_path is not None and owner.staged_path.is_file()
    recovered = load_staged_real2sim_episode(owner.staged_path)
    assert recovered.identity == episode.identity
    assert torch.equal(recovered.actions, episode.actions)
    assert torch.equal(recovered.observations["qpos"], episode.observations["qpos"])
    assert prepare.call_count == execute.call_count == 1
    with pytest.raises(RuntimeError, match="before reset"):
        generate_function(env, max_attempts=1, _real2sim_owner=owner)
    assert env.reset_options == []
    publish = owner._publish_receipt
    monkeypatch.setattr(
        owner, "_publish_receipt", MagicMock(side_effect=OSError("disk failure"))
    )
    with pytest.raises(OSError, match="disk failure"):
        owner.retry_write(env)
    assert owner.pending_episode is not None and owner.staged_path.is_file()
    assert owner.last_receipt.confirmed
    monkeypatch.setattr(owner, "_publish_receipt", publish)
    assert owner.retry_write(env).confirmed
    assert owner.pending_episode is None
    assert owner.staged_path is None
    assert execute.call_count == 1
    assert submit.call_count == 2
    assert submit.call_args_list[1].args[2].commit_id == episode.commit_id


def _real2sim_owner_case(monkeypatch, tmp_path, *, episode=None, session=None):
    """Build a real staging owner with fake capture/write ports, without rollout."""
    from embodichain.lab.gym.envs.managers import real2sim_collection
    from embodichain.lab.gym.envs.managers.real2sim_collection import (
        Real2SimCaptureRequest,
        Real2SimRunTaskOwner,
    )
    from embodichain.lab.gym.envs.managers.real2sim_sink import (
        Real2SimLeRobotEpisodeSink,
    )
    from embodichain.lab.sim.motion.expansion import (
        CandidateIdentity,
        CommitReceipt,
        ExpertEpisode,
        ValidationCheck,
        ValidationResult,
    )

    env = _ResetTrackingEnv()
    sink = object.__new__(Real2SimLeRobotEpisodeSink)
    sink._recorder = SimpleNamespace(_env=env, dataset_full_path=tmp_path)
    if episode is None:
        episode = ExpertEpisode(
            identity=CandidateIdentity(
                "scene", "initial", "candidate", "geometry", "source", "1", "unit"
            ),
            observations={"qpos": torch.tensor([[0.0], [1.0]])},
            actions=torch.tensor([[1.0]]),
            timestamps=torch.tensor([0.0, 0.1]),
            action_representation="joint_position",
            validation=ValidationResult((ValidationCheck("task", "passed"),)),
            episode_id="episode",
            commit_id="commit",
        )
    current = SimpleNamespace(episode=episode)

    def capture_request(*_):
        item = current.episode
        return Real2SimCaptureRequest(
            item.identity,
            item.validation,
            item.episode_id,
            item.commit_id,
            {name: ("robot", name) for name in item.observations},
            item.action_representation,
            {},
        )

    def receipt(item, submission_id=0, confirmed=True):
        return CommitReceipt(
            episode_id=item.episode_id,
            candidate_id=item.identity.candidate_id,
            attempt_id=item.identity.attempt_id,
            storage_id="store",
            commit_id=item.commit_id,
            scene_case_id=item.identity.scene_case_id,
            confirmed=confirmed,
            error="" if confirmed else "injected final write failure",
            submission_id=submission_id,
        )

    capture = MagicMock(side_effect=lambda *a, **k: current.episode)
    submit = MagicMock(side_effect=lambda *a, **k: receipt(a[2], k["submission_id"]))
    monkeypatch.setattr(real2sim_collection, "capture_real2sim_episode", capture)
    monkeypatch.setattr(
        real2sim_collection, "submit_and_discard_real2sim_episode", submit
    )
    prepare = MagicMock(return_value=())
    owner = Real2SimRunTaskOwner(
        sink, prepare_attempt=prepare, capture_request=capture_request, session=session
    )
    return SimpleNamespace(
        owner=owner,
        env=env,
        current=current,
        capture=capture,
        submit=submit,
        prepare=prepare,
        receipt=receipt,
        result=_episode_result(success=True, reason="success", length=1),
    )


def _real2sim_running_session():
    """Use public session transitions to reserve one measured two-joint rollout."""
    from embodichain.lab.sim.motion.expansion import (
        CandidateTrajectoryBatch,
        ExpansionSession,
        ExpertEpisode,
        SceneCase,
        TrajectoryExpansionJobCfg,
        ValidationCheck,
        ValidationResult,
    )

    session = ExpansionSession(TrajectoryExpansionJobCfg())
    names = ("joint_a", "joint_b")
    session.register_case(
        SceneCase("scene", "initial", "v1", "pick", "robot"),
        torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]),
        joint_names=names,
    )
    identity, _ = session.propose(
        "scene",
        "initial",
        source_id="source",
        source_revision="1",
        template_id="unit",
        operator_id="nominal",
        geometry_family_id="geometry",
    )
    validation = ValidationResult(
        (
            ValidationCheck("path_collision", "passed"),
            ValidationCheck("task_success", "passed"),
        )
    )
    session.add_planned(
        CandidateTrajectoryBatch(
            torch.zeros(1, 2, 2),
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([2]),
            (identity,),
            names,
        ),
        validation,
    )
    session.take_ready("scene", "initial", episode_byte_budget=10000)
    session.mark_rollout_started(identity)
    episode_id, commit_id = session.episode_ids(identity)
    episode = ExpertEpisode(
        identity=identity,
        observations={"joint_positions": torch.zeros(2, 2)},
        actions=torch.zeros(1, 2),
        timestamps=torch.tensor([0.0, 1.0]),
        action_representation="qpos",
        validation=validation,
        episode_id=episode_id,
        commit_id=commit_id,
    )
    return session, episode


@pytest.mark.parametrize("after_success", [False, True])
@pytest.mark.parametrize("failure_point", ["stage", "submit"])
def test_real2sim_owner_recovers_receiptless_write_of_the_correct_episode(
    monkeypatch, tmp_path, after_success, failure_point
) -> None:
    from dataclasses import replace

    case = _real2sim_owner_case(monkeypatch, tmp_path)
    if after_success:
        assert case.owner.commit(case.env, case.result).confirmed
        item = case.current.episode
        case.current.episode = replace(
            item,
            identity=replace(item.identity, candidate_id="next-candidate"),
            episode_id="next-episode",
            commit_id="next-commit",
        )
    previous_resets = len(case.env.reset_options)
    capture_count = case.capture.call_count
    target = case.current.episode
    stage = case.owner._stage
    attempts = 0

    def interrupted(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected receiptless failure")
        if failure_point == "stage":
            return stage(*args, **kwargs)
        return case.receipt(args[2], kwargs["submission_id"])

    if failure_point == "stage":
        monkeypatch.setattr(case.owner, "_stage", interrupted)
    else:
        case.submit.side_effect = interrupted
    with pytest.raises(OSError, match="receiptless"):
        case.owner.commit(case.env, case.result)
    assert case.owner.pending_episode.commit_id == target.commit_id
    assert case.owner.last_receipt is None
    assert len(case.env.reset_options) == previous_resets
    receipt = case.owner.retry_write(case.env, submission_id=0)
    assert receipt.confirmed and receipt.commit_id == target.commit_id
    assert case.capture.call_count == capture_count + 1
    case.prepare.assert_not_called()
    assert len(case.env.reset_options) == previous_resets + 1
    assert case.owner.pending_episode is None and case.owner.staged_path is None


@pytest.mark.parametrize("directory", ["pending", "receipts"])
def test_real2sim_owner_retries_directory_sync_after_existing_link(
    monkeypatch, tmp_path, directory
) -> None:
    from embodichain.lab.gym.envs.managers import real2sim_collection

    case = _real2sim_owner_case(monkeypatch, tmp_path)
    sync = real2sim_collection._sync_directory
    calls = 0
    syncs_before_submit = []

    def submit(*args, **kwargs):
        syncs_before_submit.append(calls)
        return case.receipt(args[2], kwargs["submission_id"])

    case.submit.side_effect = submit

    def interrupted(path):
        nonlocal calls
        if path.name == directory:
            calls += 1
            if calls == 1:
                raise OSError("injected directory sync failure")
        return sync(path)

    monkeypatch.setattr(real2sim_collection, "_sync_directory", interrupted)
    with pytest.raises(OSError, match="directory sync"):
        case.owner.commit(case.env, case.result)
    assert len(list((tmp_path / "meta" / "real2sim" / directory).iterdir())) == 1
    assert not case.env.reset_options
    assert case.owner.retry_write(case.env).confirmed
    assert calls >= 2
    assert case.capture.call_count == case.submit.call_count == 1
    assert syncs_before_submit == ([2] if directory == "pending" else [0])


def test_real2sim_owner_session_preserves_outstanding_ids_and_completion_progress(
    monkeypatch, tmp_path
) -> None:
    from embodichain.lab.gym.envs.managers import real2sim_collection

    session, episode = _real2sim_running_session()
    case = _real2sim_owner_case(monkeypatch, tmp_path, episode=episode, session=session)
    accept = MagicMock(wraps=session.accept_episode)
    reserve = MagicMock(wraps=session.retry_write)
    apply = MagicMock(wraps=session.apply_receipt)
    monkeypatch.setattr(session, "accept_episode", accept)
    monkeypatch.setattr(session, "retry_write", reserve)
    monkeypatch.setattr(session, "apply_receipt", apply)
    case.submit.side_effect = [
        OSError("initial submit interrupted"),
        case.receipt(episode, 0, False),
        OSError("retry interrupted"),
        case.receipt(episode, 1),
    ]
    with pytest.raises(OSError, match="initial submit"):
        case.owner.commit(case.env, case.result)
    before = session.snapshot()
    with pytest.raises(ValueError, match="submission ID"):
        case.owner.retry_write(case.env, submission_id=1)
    assert session.snapshot() == before
    assert not case.owner.retry_write(case.env, submission_id=0).confirmed
    before = session.snapshot()
    with pytest.raises(ValueError, match="submission ID"):
        case.owner.retry_write(case.env, submission_id=2)
    assert session.snapshot() == before
    with pytest.raises(OSError, match="retry interrupted"):
        case.owner.retry_write(case.env, submission_id=1)
    reset = case.env.reset
    resets = 0

    def interrupted_reset(*args, **kwargs):
        nonlocal resets
        resets += 1
        if resets == 1:
            raise OSError("reset interrupted")
        return reset(*args, **kwargs)

    monkeypatch.setattr(case.env, "reset", interrupted_reset)
    with pytest.raises(OSError, match="reset interrupted"):
        case.owner.retry_write(case.env, submission_id=1)
    sync = real2sim_collection._sync_directory
    cleanup_failures = 0

    def interrupted_cleanup(path):
        nonlocal cleanup_failures
        if path.name == "pending" and cleanup_failures == 0:
            cleanup_failures += 1
            raise OSError("cleanup interrupted")
        return sync(path)

    monkeypatch.setattr(real2sim_collection, "_sync_directory", interrupted_cleanup)
    with pytest.raises(OSError, match="cleanup interrupted"):
        case.owner.retry_write(case.env)
    assert case.owner.retry_write(case.env).confirmed
    assert [call.kwargs["submission_id"] for call in case.submit.call_args_list] == [
        0,
        0,
        1,
        1,
    ]
    assert case.capture.call_count == accept.call_count == reserve.call_count == 1
    assert apply.call_count == 2  # One final failure and one confirmation.
    assert resets == 2 and len(case.env.reset_options) == 1
    counts = session.snapshot()["counts"]
    assert (
        counts["rollout_attempted"]
        == counts["validated_accepted"]
        == counts["committed"]
        == 1
    )


def test_real2sim_owner_does_not_retry_session_rejection_after_cleanup_error(
    monkeypatch, tmp_path
) -> None:
    from dataclasses import replace
    from embodichain.lab.gym.envs.managers import real2sim_collection
    from embodichain.lab.sim.motion.expansion import ValidationResult

    session, episode = _real2sim_running_session()
    episode = replace(
        episode, validation=ValidationResult((episode.validation.checks[0],))
    )
    case = _real2sim_owner_case(monkeypatch, tmp_path, episode=episode, session=session)
    sync = real2sim_collection._sync_directory
    calls = 0

    def interrupted_cleanup(path):
        nonlocal calls
        if path.name == "pending":
            calls += 1
            if calls == 2:
                raise OSError("rejected payload cleanup interrupted")
        return sync(path)

    monkeypatch.setattr(real2sim_collection, "_sync_directory", interrupted_cleanup)
    with pytest.raises(OSError, match="rejected payload"):
        case.owner.commit(case.env, case.result)
    assert case.owner.pending_episode is None and case.owner.staged_path is None
    case.submit.assert_not_called()
    assert case.owner.prepare(case.env, episode_index=1, attempt_id=0) == ()


def test_close_preserves_staged_real2sim_submission_without_reset() -> None:
    env = _LifecycleTrackingEnv()
    owner = SimpleNamespace(pending_episode=object(), finalize=MagicMock())
    env._real2sim_collection_owner = owner
    run_task._abort_and_close_env(env, exit_process=False)
    assert env.reset_options == []
    assert env.events == [("close", False)]
    owner.finalize.assert_called_once()


def test_generate_function_updates_collection_attempt_and_reset_stats(
    monkeypatch,
) -> None:
    env = _ResetTrackingEnv()
    env._collection_stats = {
        "attempts": 0,
        "discard_reset_count": 0,
        "commit_reset_count": 0,
    }
    results = iter(
        [
            _episode_result(success=False, reason="empty_plan"),
            _episode_result(success=True, reason="success"),
        ]
    )
    monkeypatch.setattr(
        "embodichain.lab.scripts.run_task.execute_demo_episode",
        lambda *args, **kwargs: next(results),
    )

    assert generate_function(env, max_attempts=2, reset_before=False)
    assert env._collection_stats == {
        "attempts": 2,
        "discard_reset_count": 1,
        "commit_reset_count": 1,
    }


def test_generate_function_logs_failed_trace_in_debug_mode(monkeypatch) -> None:
    """Debug retries expose the owned structured episode trace."""
    env = _ResetTrackingEnv()
    result = _episode_result(success=False, reason="segment_validation_failed")
    warnings: list[str] = []
    monkeypatch.setattr(
        "embodichain.lab.scripts.run_task.execute_demo_episode",
        lambda *args, **kwargs: result,
    )
    monkeypatch.setattr(
        "embodichain.lab.scripts.run_task.log_warning",
        warnings.append,
    )

    generated = generate_function(
        env,
        max_attempts=1,
        reset_before=False,
        debug_mode=True,
    )

    assert not generated
    debug_trace = next(
        message for message in warnings if "Failed demo trace" in message
    )
    assert '"terminal_reason":"segment_validation_failed"' in debug_trace


def test_generate_function_commits_failed_episode_when_configured(monkeypatch) -> None:
    """A recorded task failure is a persisted result when explicitly enabled."""
    env = _ResetTrackingEnv(save_failed_episodes=True)
    monkeypatch.setattr(
        "embodichain.lab.scripts.run_task.execute_demo_episode",
        lambda *args, **kwargs: _episode_result(
            success=False,
            reason="task_incomplete",
        ),
    )

    generated = generate_function(
        env,
        time_id=3,
        max_attempts=2,
        reset_before=False,
    )

    assert generated
    assert env.reset_options == [None]


def test_generate_function_commits_accepted_prefix_as_segment_fragment(
    monkeypatch,
) -> None:
    """Fragment mode keeps useful accepted data without claiming episode success."""
    env = _ResetTrackingEnv()
    result = DemoEpisodeResult(
        episode_index=0,
        length=4,
        completed=False,
        success=(False,),
        terminated=(False,),
        truncated=(False,),
        terminal_reason="segment_validation_failed",
        segments=(
            DemoSegmentResult(
                segment_id=0,
                name="pick",
                start_step=0,
                end_step=2,
                success=True,
                active=(True,),
                start_steps=(0,),
                end_steps=(2,),
                successes=(True,),
                failure_reasons=(None,),
            ),
            DemoSegmentResult(
                segment_id=1,
                name="place",
                start_step=2,
                end_step=4,
                success=False,
                failure_reason="segment_validation_failed",
                active=(True,),
                start_steps=(2,),
                end_steps=(4,),
                successes=(False,),
                failure_reasons=("segment_validation_failed",),
            ),
        ),
        lengths=(4,),
    )
    monkeypatch.setattr(
        "embodichain.lab.scripts.run_task.execute_demo_episode",
        lambda *args, **kwargs: result,
    )

    generated = generate_function(
        env,
        execution_cfg=DemoExecutionCfg(mode="segment_fragments"),
        max_attempts=1,
        reset_before=False,
    )

    assert generated
    assert env.reset_options == [None]


def test_generate_function_retries_empty_failure_even_when_saving_failures(
    monkeypatch,
) -> None:
    """An empty plan cannot count as a saved failed episode."""
    env = _ResetTrackingEnv(save_failed_episodes=True)
    results = iter(
        [
            _episode_result(success=False, reason="empty_plan", length=0),
            _episode_result(success=True, reason="success"),
        ]
    )
    monkeypatch.setattr(
        "embodichain.lab.scripts.run_task.execute_demo_episode",
        lambda *args, **kwargs: next(results),
    )

    generated = generate_function(
        env,
        max_attempts=2,
        reset_before=False,
    )

    assert generated
    assert env.reset_options == [{"save_data": False}, None]


def test_generate_function_commits_only_selected_vector_rows(monkeypatch) -> None:
    """The final partial batch commits selected rows in one full-batch reset."""
    num_envs = 3
    env = _ResetTrackingEnv(num_envs=num_envs)
    monkeypatch.setattr(
        "embodichain.lab.scripts.run_task.execute_demo_episode",
        lambda *args, **kwargs: _episode_result(
            success=True,
            reason="success",
            num_envs=num_envs,
        ),
    )

    generated = generate_function(
        env,
        save_env_ids=(0, 1),
        reset_before=False,
    )

    assert generated
    assert len(env.reset_options) == 1
    assert env.reset_options[0]["commit_env_ids"].tolist() == [0, 1]
    assert env.reset_options[0]["save_data"] is False
    assert "reset_ids" not in env.reset_options[0]


def test_main_counts_max_episodes_as_persisted_env_rows(monkeypatch) -> None:
    """Vector collection persists exactly max_episodes, including a partial batch."""
    env = _ResetTrackingEnv(num_envs=3)
    args = SimpleNamespace(
        replay=False,
        preview=False,
        save_path="",
        save_video=False,
        debug_mode=False,
        regenerate=False,
        record_trajectory=False,
    )
    calls: list[tuple[int, tuple[int, ...]]] = []

    def record_batch(*args, **kwargs) -> bool:
        calls.append((kwargs["time_id"], tuple(kwargs["save_env_ids"])))
        return True

    monkeypatch.setattr(run_task, "generate_function", record_batch)

    run_task.main(
        args,
        env,
        {"max_episodes": 5, "demo_max_attempts": 2},
    )

    assert calls == [(0, (0, 1, 2)), (3, (0, 1))]
    assert sum(len(env_ids) for _, env_ids in calls) == 5


def _run_successful_collection(monkeypatch) -> None:
    """Run the collection UI around a deterministic successful generator."""
    env = _ResetTrackingEnv(num_envs=3)
    args = SimpleNamespace(
        replay=False,
        preview=False,
        save_path="",
        save_video=False,
        debug_mode=False,
        regenerate=False,
        record_trajectory=False,
    )
    monkeypatch.setattr(run_task, "generate_function", lambda *args, **kwargs: True)
    run_task.main(
        args,
        env,
        {"id": GYM_ID, "max_episodes": 5, "demo_max_attempts": 2},
    )


def test_main_displays_total_episode_progress_for_vector_batches(
    monkeypatch, capsys
) -> None:
    """Collection progress counts persisted rows rather than vector batches."""
    _run_successful_collection(monkeypatch)

    output = capsys.readouterr().out
    assert "Collecting episodes" in output
    assert "100%" in output
    assert "5/5" in output


def test_main_prints_compact_collection_summary(monkeypatch, capsys) -> None:
    """The run header exposes the settings needed to interpret progress."""
    _run_successful_collection(monkeypatch)

    output = capsys.readouterr().out
    assert "EmbodiChain · Run Task" in output
    assert f"Task        {GYM_ID}" in output
    assert "Episodes    5" in output
    assert "Parallel    3 environments" in output
    assert "Attempts    2 per episode" in output


def test_main_prints_clean_collection_completion(monkeypatch, capsys) -> None:
    """A completed run ends with one concise success line."""
    _run_successful_collection(monkeypatch)

    output = capsys.readouterr().out
    assert "✓ Collection complete · 5 episodes saved in 2 batches" in output


def test_generate_function_rejects_runner_owned_segment_count() -> None:
    """The task, not run-task, defines how many segments an episode contains."""
    with pytest.raises(ValueError, match="create_demo_segments"):
        generate_function(_ResetTrackingEnv(), num_traj=2)


def test_generate_function_discards_partial_episode_on_exception(monkeypatch) -> None:
    """Planner or step exceptions cannot be committed later by finalize()."""
    env = _ResetTrackingEnv()
    monkeypatch.setattr(
        "embodichain.lab.scripts.run_task.execute_demo_episode",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("planner failed")),
    )

    with pytest.raises(RuntimeError, match="planner failed"):
        generate_function(env, reset_before=False)

    assert env.reset_options == [{"save_data": False}]


@pytest.mark.parametrize(
    "exit_type",
    [KeyboardInterrupt, SystemExit, GeneratorExit],
    ids=["keyboard-interrupt", "system-exit", "generator-exit"],
)
def test_generate_function_discards_partial_episode_on_early_exit(
    monkeypatch, exit_type
) -> None:
    """Non-Exception exits abort the pending episode before propagating."""
    env = _ResetTrackingEnv()

    def exit_during_episode(*args, **kwargs):
        raise exit_type()

    monkeypatch.setattr(
        "embodichain.lab.scripts.run_task.execute_demo_episode",
        exit_during_episode,
    )

    with pytest.raises(exit_type):
        generate_function(env, reset_before=False)

    assert env.reset_options == [{"save_data": False}]


def test_cli_aborts_before_closing_environment_once(monkeypatch) -> None:
    """CLI owns finalization and closes only after discarding pending data."""
    env = _LifecycleTrackingEnv()
    args = SimpleNamespace(
        replay=False,
        replay_mode="kinematic",
        preview=False,
        save_path="",
        save_video=False,
        debug_mode=False,
        regenerate=False,
        record_trajectory=False,
    )
    parser = MagicMock()
    parser.parse_args.return_value = args
    gym_config = {
        "id": GYM_ID,
        "max_episodes": 1,
        "demo_max_attempts": 1,
    }

    monkeypatch.setattr(run_task, "_create_parser", lambda: parser)
    monkeypatch.setattr(run_task, "discover_task_packages", lambda: None)
    monkeypatch.setattr(run_task, "execute_init_hooks", lambda: None)
    monkeypatch.setattr(
        run_task,
        "build_env_cfg_from_args",
        lambda parsed_args: (object(), gym_config, {}),
    )
    monkeypatch.setattr(run_task.gymnasium, "make", lambda **kwargs: env)
    monkeypatch.setattr(run_task, "generate_function", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "embodichain.lab.sim.sim_manager.SimulationManager.flush_cleanup_queue",
        lambda: None,
    )

    run_task.cli([])

    abort_event = ("reset", {"save_data": False})
    assert env.events == [abort_event, ("reset", None), abort_event, ("close", None)]


def test_cli_uses_loaded_program_with_scoped_renderer_default(
    monkeypatch,
) -> None:
    """The CLI attaches the strict program config to the environment config."""
    env = _LifecycleTrackingEnv()
    env_cfg = SimpleNamespace(task_program=None)
    decoded_program = object()
    args = SimpleNamespace(
        replay=False,
        replay_mode="kinematic",
        preview=True,
        task_program="program.yaml",
    )
    parser = MagicMock()
    parser.parse_args.return_value = args
    from embodichain.lab.sim import cfg
    from embodichain.lab.sim.utility.render_utils import select_default_renderer

    monkeypatch.setattr(cfg, "DEFAULT_RENDERER", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA RTX 4090")

    def make_task(**kwargs):
        assert select_default_renderer() == "fast-rt"
        return env

    make = MagicMock(side_effect=make_task)

    monkeypatch.setattr(run_task, "_create_parser", lambda: parser)
    monkeypatch.setattr(run_task, "discover_task_packages", lambda: None)
    monkeypatch.setattr(run_task, "execute_init_hooks", lambda: None)

    def build(parsed_args):
        assert parsed_args is args
        env_cfg.task_program = decoded_program
        return env_cfg, {"id": GYM_ID}, {}

    monkeypatch.setattr(run_task, "build_env_cfg_from_args", build)
    monkeypatch.setattr(run_task.gymnasium, "make", make)
    monkeypatch.setattr(run_task, "main", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "embodichain.lab.sim.sim_manager.SimulationManager.flush_cleanup_queue",
        lambda: None,
    )

    run_task.cli([])

    assert env_cfg.task_program is decoded_program
    make.assert_called_once_with(id=GYM_ID, cfg=env_cfg)
    assert select_default_renderer() == "hybrid"


def test_close_durability_failure_is_not_swallowed() -> None:
    """A failed recorder barrier makes the runner fail after aborting pending data."""
    env = _LifecycleTrackingEnv()

    def fail_close(*, exit_process=None) -> None:
        env.events.append(("close", exit_process))
        raise OSError("dataset flush failed")

    env.close = fail_close

    with pytest.raises(OSError, match="dataset flush failed"):
        run_task._abort_and_close_env(env)

    assert env.events == [
        ("reset", {"save_data": False}),
        ("close", None),
    ]


def test_cli_preserves_main_error_and_disables_fast_exit(monkeypatch) -> None:
    """Failure cleanup returns normally so the original CLI error reaches the shell."""
    env = _LifecycleTrackingEnv()
    args = SimpleNamespace(
        replay=False,
        replay_mode="kinematic",
        preview=False,
        save_path="",
        save_video=False,
        debug_mode=False,
        regenerate=False,
        record_trajectory=False,
    )
    parser = MagicMock()
    parser.parse_args.return_value = args
    gym_config = {"id": GYM_ID}

    monkeypatch.setattr(run_task, "_create_parser", lambda: parser)
    monkeypatch.setattr(run_task, "discover_task_packages", lambda: None)
    monkeypatch.setattr(run_task, "execute_init_hooks", lambda: None)
    monkeypatch.setattr(
        run_task,
        "build_env_cfg_from_args",
        lambda parsed_args: (object(), gym_config, {}),
    )
    monkeypatch.setattr(run_task.gymnasium, "make", lambda **kwargs: env)
    monkeypatch.setattr(
        run_task,
        "main",
        MagicMock(side_effect=RuntimeError("generation failed")),
    )
    monkeypatch.setattr(
        "embodichain.lab.sim.sim_manager.SimulationManager.flush_cleanup_queue",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        run_task.cli([])

    assert env.events == [
        ("reset", {"save_data": False}),
        ("close", False),
    ]


def test_zero_system_exit_cannot_hide_cleanup_failure(monkeypatch) -> None:
    """A nominal exit code is replaced by a failed durability barrier."""
    env = _LifecycleTrackingEnv()
    args = SimpleNamespace(
        replay=False,
        replay_mode="kinematic",
        preview=False,
        replay_trajectory=None,
    )
    parser = MagicMock()
    parser.parse_args.return_value = args
    monkeypatch.setattr(run_task, "_create_parser", lambda: parser)
    monkeypatch.setattr(run_task, "discover_task_packages", lambda: None)
    monkeypatch.setattr(run_task, "execute_init_hooks", lambda: None)
    monkeypatch.setattr(
        run_task,
        "build_env_cfg_from_args",
        lambda parsed_args: (object(), {"id": GYM_ID}, {}),
    )
    monkeypatch.setattr(run_task.gymnasium, "make", lambda **kwargs: env)
    monkeypatch.setattr(run_task, "main", MagicMock(side_effect=SystemExit(0)))
    monkeypatch.setattr(
        "embodichain.lab.sim.sim_manager.SimulationManager.flush_cleanup_queue",
        lambda: None,
    )

    def fail_close(*, exit_process=None) -> None:
        raise OSError("dataset flush failed")

    env.close = fail_close

    with pytest.raises(OSError, match="dataset flush failed"):
        run_task.cli([])


def test_control_loop_consumes_viser_seek_while_paused() -> None:
    """A browser slider seek is applied without waiting for terminal input."""

    class FakeControlInput:
        single_key = True

        def __init__(self) -> None:
            self.timeouts: list[float | None] = []

        def read_key(self, timeout: float | None = None) -> str:
            self.timeouts.append(timeout)
            return "q"

    class FakeReplayEnv:
        def __init__(self) -> None:
            self._lengths = torch.tensor([REPLAY_NUM_STEPS])
            self.env = SimpleNamespace(
                sim_cfg=SimpleNamespace(physics_dt=0.01),
                cfg=SimpleNamespace(sim_steps_per_control=4),
            )
            self.visited_steps: list[int] = []

        def go_to_step(self, step: int) -> None:
            self.visited_steps.append(step)

    class FakeVisualizationRuntime:
        def __init__(self) -> None:
            self.target_step: int | None = REPLAY_TARGET_STEP
            self.states: list[tuple[int, int, bool]] = []

        def drain_replay_control_command(self) -> int | None:
            target_step, self.target_step = self.target_step, None
            return target_step

        def publish_replay_control(
            self,
            *,
            step: int,
            max_step: int,
            visible: bool = True,
        ) -> None:
            self.states.append((step, max_step, visible))

    replay_env = FakeReplayEnv()
    control_input = FakeControlInput()
    runtime = FakeVisualizationRuntime()
    _run_replay_control_loop(
        replay_env,
        control_input,
        visualization_runtime=runtime,
    )

    assert replay_env.visited_steps == [0, REPLAY_TARGET_STEP]
    assert control_input.timeouts == [VISER_POLL_INTERVAL]
    assert runtime.states[-1] == (
        REPLAY_TARGET_STEP,
        REPLAY_NUM_STEPS - 1,
        False,
    )
