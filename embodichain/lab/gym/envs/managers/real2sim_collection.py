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

"""Transaction owner for source-qualified Gym and run-task collection."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5
import zipfile

import numpy as np
import torch

from embodichain.lab.gym.envs._json import json_safe_copy
from embodichain.data_pipeline.recording.journal import inspect_recording
from embodichain.lab.sim.motion.expansion import (
    CandidateIdentity,
    CommitReceipt,
    ExpansionSession,
    ExpertEpisode,
    TrajectoryPhase,
    ValidationCheck,
    ValidationResult,
)
from .real2sim_sink import (
    Real2SimLeRobotEpisodeSink,
    _sync_directory,
    capture_real2sim_episode,
    submit_and_discard_real2sim_episode,
)

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.demo import DemoEpisodeResult, DemoSegment

__all__ = [
    "Real2SimCaptureRequest",
    "Real2SimRunTaskOwner",
    "load_staged_real2sim_episode",
]


def load_staged_real2sim_episode(path: str | Path) -> ExpertEpisode:
    """Recover a bounded no-pickle pending payload for write-only repair.

    The sink must revalidate the source bundles, trace, checks and journal on
    submission. Loading a pending payload does not confirm persistence.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("pending episode is missing or exceeds its size limit")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if (
                len({entry.filename for entry in entries}) != len(entries)
                or sum(entry.file_size for entry in entries) > 256 * 1024 * 1024
            ):
                raise ValueError("pending episode exceeds its archive limits")
        with np.load(path, allow_pickle=False) as arrays:
            if "manifest" not in arrays.files or arrays["manifest"].dtype != np.uint8:
                raise ValueError("pending episode manifest is invalid")
            metadata = json.loads(arrays["manifest"].tobytes().decode("utf-8"))
            required = {
                "identity",
                "episode_id",
                "commit_id",
                "action_representation",
                "observations",
                "validation",
                "phases",
                "metadata",
            }
            if not isinstance(metadata, dict) or set(metadata) != required:
                raise ValueError("pending episode manifest fields are invalid")
            names = metadata["observations"]
            if (
                not isinstance(names, list)
                or not names
                or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name for name in names)
            ):
                raise ValueError("pending observation names are invalid")
            expected = {"manifest", "actions", "timestamps"} | {
                f"observation_{index}" for index in range(len(names))
            }
            if set(arrays.files) != expected:
                raise ValueError("pending arrays do not match the manifest")
            observations = {
                name: torch.from_numpy(
                    np.array(arrays[f"observation_{index}"], copy=True)
                )
                for index, name in enumerate(names)
            }
            actions = torch.from_numpy(np.array(arrays["actions"], copy=True))
            timestamps = torch.from_numpy(np.array(arrays["timestamps"], copy=True))
        return ExpertEpisode(
            identity=CandidateIdentity(**metadata["identity"]),
            observations=observations,
            actions=actions,
            timestamps=timestamps,
            action_representation=metadata["action_representation"],
            validation=ValidationResult(
                tuple(ValidationCheck(**item) for item in metadata["validation"])
            ),
            episode_id=metadata["episode_id"],
            commit_id=metadata["commit_id"],
            phases=tuple(TrajectoryPhase(**item) for item in metadata["phases"]),
            metadata=metadata["metadata"],
        )
    except (
        OSError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as error:
        raise ValueError("invalid pending Real2Sim episode") from error


@dataclass(frozen=True)
class Real2SimCaptureRequest:
    """Host-measured acceptance, identity and typed artifact references.

    Observations and actions are intentionally absent: the owner captures them
    from the actual Gym rollout immediately before submitting to the sink.
    """

    identity: CandidateIdentity
    validation: ValidationResult
    episode_id: str
    commit_id: str
    observation_fields: Mapping[str, tuple[str, ...]]
    action_representation: str
    metadata: Mapping[str, object]
    phases: tuple[TrajectoryPhase, ...] = ()


class Real2SimRunTaskOwner:
    """Own one B=1 source-backed attempt and its persistence transaction.

    Trusted host callbacks prepare a traced segment sequence and provide
    measured validation plus source/trace attachments after execution. The
    sink verifies those attachments before confirming a receipt. An unconfirmed
    payload stays owned here for repair or a write-only retry; preparing a new
    attempt while it exists is rejected.
    """

    def __init__(
        self,
        sink: Real2SimLeRobotEpisodeSink,
        *,
        prepare_attempt: Callable[[object, int, int], Iterable[DemoSegment]],
        capture_request: Callable[[object, DemoEpisodeResult], Real2SimCaptureRequest],
        record_failed_attempt: (
            Callable[[object, DemoEpisodeResult, str | None], None] | None
        ) = None,
        session: ExpansionSession | None = None,
    ) -> None:
        if not isinstance(sink, Real2SimLeRobotEpisodeSink):
            raise TypeError("sink must be a Real2SimLeRobotEpisodeSink")
        if not callable(prepare_attempt) or not callable(capture_request):
            raise TypeError(
                "Real2Sim owner needs explicit preparation and validation callbacks"
            )
        if record_failed_attempt is not None and not callable(record_failed_attempt):
            raise TypeError("record_failed_attempt must be callable or None")
        if session is not None and not isinstance(session, ExpansionSession):
            raise TypeError("session must be an ExpansionSession or None")
        self._sink = sink
        self._prepare_attempt = prepare_attempt
        self._capture_request = capture_request
        self._record_failed_attempt = record_failed_attempt
        self._session = session
        self._pending: ExpertEpisode | None = None
        self._last_receipt: CommitReceipt | None = None
        self._staged_path: Path | None = None
        self._confirmed_receipts: list[CommitReceipt] = []

    @property
    def pending_episode(self) -> ExpertEpisode | None:
        """Return the retained payload after an unconfirmed submission."""
        return None if self._pending is None else replace(self._pending)

    @property
    def last_receipt(self) -> CommitReceipt | None:
        """Return the latest finalized submission result."""
        return self._last_receipt

    @property
    def staged_path(self) -> Path | None:
        """Return durable failed-write evidence, absent after confirmation."""
        return self._staged_path

    @property
    def confirmed_receipts(self) -> tuple[CommitReceipt, ...]:
        """Return receipts durably published by this owner."""
        return tuple(self._confirmed_receipts)

    def _stage(self, episode: ExpertEpisode) -> Path:
        root = getattr(self._sink._recorder, "dataset_full_path", None)
        if root is None:
            raise RuntimeError(
                "recorder dataset path is unavailable for pending evidence"
            )
        directory = Path(root) / "meta" / "real2sim" / "pending"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{uuid5(NAMESPACE_URL, episode.commit_id)}.npz"
        names = sorted(episode.observations)
        metadata = {
            "identity": asdict(episode.identity),
            "episode_id": episode.episode_id,
            "commit_id": episode.commit_id,
            "action_representation": episode.action_representation,
            "observations": names,
            "validation": [
                {
                    "check_id": item.check_id,
                    "status": item.status,
                    "detail": item.detail,
                    "metrics": dict(item.metrics),
                }
                for item in episode.validation.checks
            ],
            "phases": [asdict(phase) for phase in episode.phases],
            "metadata": json_safe_copy(episode.metadata, field_name="episode metadata"),
        }
        encoded = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        arrays = {
            f"observation_{index}": episode.observations[name].detach().cpu().numpy()
            for index, name in enumerate(names)
        }
        arrays.update(
            actions=episode.actions.detach().cpu().numpy(),
            timestamps=episode.timestamps.detach().cpu().numpy(),
            manifest=np.frombuffer(encoded, dtype=np.uint8),
        )
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=directory, suffix=".npz", delete=False
            ) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination)
                _sync_directory(destination.parent)
            except FileExistsError:
                with np.load(destination, allow_pickle=False) as existing:
                    if set(existing.files) != set(arrays) or any(
                        not np.array_equal(existing[name], value)
                        for name, value in arrays.items()
                    ):
                        raise ValueError(
                            "pending commit already exists with different payload"
                        )
            return destination
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _target(self, env: object) -> object:
        target = getattr(env, "unwrapped", env)
        if (
            target is not self._sink._recorder._env
            or getattr(target, "num_envs", None) != 1
        ):
            raise ValueError(
                "owner and synchronous recorder must share one Gym environment"
            )
        return target

    def _publish_receipt(self, receipt: CommitReceipt) -> None:
        """Durably publish the matching confirmed receipt without replacement."""
        if not receipt.confirmed or self._pending is None:
            raise ValueError("only a confirmed pending episode has a receipt")
        root = Path(self._sink._recorder.dataset_full_path)
        directory = root / "meta" / "real2sim" / "receipts"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{uuid5(NAMESPACE_URL, receipt.commit_id)}.json"
        content = json.dumps(
            asdict(receipt), sort_keys=True, indent=2, allow_nan=False
        ).encode("utf-8")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination)
                _sync_directory(destination.parent)
            except FileExistsError:
                if destination.read_bytes() != content:
                    raise ValueError("existing receipt differs from this confirmation")
            if destination.read_bytes() != content:
                raise RuntimeError("published receipt could not be verified")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _complete_confirmed(self, target: object, receipt: CommitReceipt) -> None:
        self._publish_receipt(receipt)
        if self._session is not None:
            self._session.apply_receipt(receipt)
        target.reset(options={"save_data": False})
        if self._staged_path is not None:
            self._staged_path.unlink(missing_ok=True)
            _sync_directory(self._staged_path.parent)
            self._staged_path = None
        if receipt not in self._confirmed_receipts:
            self._confirmed_receipts.append(receipt)
        self._pending = None

    def write_collection_report(self, *, expected_episodes: int) -> Path:
        """Verify durable receipts, journal entries and attachments for a run."""
        if type(expected_episodes) is not int or expected_episodes < 0:
            raise ValueError("expected_episodes must be a nonnegative integer")
        if (
            self._pending is not None
            or len(self._confirmed_receipts) != expected_episodes
        ):
            raise ValueError("collection has pending or missing confirmed receipts")
        root = Path(self._sink._recorder.dataset_full_path).resolve()
        journal = inspect_recording(root)
        if journal["errors"]:
            raise RuntimeError("recording journal did not pass verification")
        entries = {item["episode_uuid"]: item for item in journal["commits"]}
        sidecar = root / "meta" / "embodichain_episodes.jsonl"
        rows = (
            [
                json.loads(line)
                for line in sidecar.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if sidecar.is_file()
            else []
        )
        by_commit = {row.get("commit_id"): row for row in rows}
        cases = []
        for receipt in self._confirmed_receipts:
            episode_uuid = receipt.storage_id.rsplit("#", 1)[-1]
            row = by_commit.get(receipt.commit_id)
            receipt_path = (
                root
                / "meta"
                / "real2sim"
                / "receipts"
                / f"{uuid5(NAMESPACE_URL, receipt.commit_id)}.json"
            )
            if (
                not receipt.storage_id.startswith(str(root) + "#")
                or row is None
                or row.get("episode_uuid") != episode_uuid
                or row.get("candidate_id") != receipt.candidate_id
                or entries.get(episode_uuid, {}).get("status") != "complete"
                or not receipt_path.is_file()
                or json.loads(receipt_path.read_text(encoding="utf-8"))
                != asdict(receipt)
            ):
                raise RuntimeError("a recorded episode and its receipt do not match")
            artifacts = [
                item
                for bundle in row["real2sim_artifacts"].values()
                for item in bundle.values()
            ] + [row["real2sim_episode_evidence"], row["real2sim_lineage"]["trace"]]
            observed_geometry = row.get("real2sim_observed_geometry")
            if observed_geometry is not None:
                artifacts.append(observed_geometry)
            elif any(
                item.get("id") == "observed_collision"
                for item in row.get("validation_checks", ())
            ):
                raise ValueError(
                    "observed collision evidence is missing its geometry sidecar"
                )
            for item in artifacts:
                path = (root / item["path"]).resolve()
                if root not in path.parents or not path.is_file():
                    raise ValueError(
                        "recorded artifact is missing or outside the dataset"
                    )
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != item["sha256"]:
                    raise ValueError("recorded artifact digest mismatch")
            checks = row.get("validation_checks", ())
            if not checks or any(item.get("status") != "passed" for item in checks):
                raise ValueError("recorded episode has unverified acceptance checks")
            cases.append(
                {
                    "candidate_id": receipt.candidate_id,
                    "episode_uuid": episode_uuid,
                    "commit_id": receipt.commit_id,
                    "layout": row.get("layout"),
                    "recipe": row.get("recipe"),
                    "receipt": receipt_path.relative_to(root).as_posix(),
                    "checks": checks,
                    "phase_duration_validation": row.get(
                        "phase_duration_validation", []
                    ),
                    "sampled_motion_limits": row.get("sampled_motion_limits", {}),
                    "calls": row["real2sim_lineage"]["calls"],
                }
            )
        report = {
            "expected_episodes": expected_episodes,
            "confirmed_episodes": len(cases),
            "journal_verified": True,
            "attachments_verified": True,
            "cases": cases,
        }
        destination = root / "meta" / "real2sim" / "collection-report.json"
        content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode(
            "utf-8"
        )
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            _sync_directory(destination.parent)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination

    def prepare(
        self, env: object, *, episode_index: int, attempt_id: int
    ) -> Iterable[DemoSegment]:
        """Prepare one source-traced segment stream before any Gym step."""
        target = self._target(env)
        if self._pending is not None:
            raise RuntimeError(
                "unconfirmed Real2Sim submission must be repaired before another rollout"
            )
        segments = self._prepare_attempt(target, episode_index, attempt_id)
        if segments is None or not isinstance(segments, Iterable):
            raise TypeError("Real2Sim preparation must return an iterable of segments")
        return segments

    def commit(self, env: object, result: DemoEpisodeResult) -> CommitReceipt:
        """Capture actual T+1 observations and submit a measured successful run."""
        target = self._target(env)
        if self._pending is not None:
            raise RuntimeError("an earlier Real2Sim submission is still unconfirmed")
        if not result.completed or not result.all_success:
            raise ValueError("only a successfully executed episode can be submitted")
        request = self._capture_request(target, result)
        if not isinstance(request, Real2SimCaptureRequest):
            raise TypeError("capture callback must return Real2SimCaptureRequest")
        if request.identity.attempt_id != result.attempt_id:
            raise ValueError("capture identity does not match the executed attempt")
        if not request.validation.accepted:
            raise ValueError("Real2Sim measured validation did not pass")
        episode = capture_real2sim_episode(
            target,
            identity=request.identity,
            validation=request.validation,
            episode_id=request.episode_id,
            commit_id=request.commit_id,
            observation_fields=request.observation_fields,
            action_representation=request.action_representation,
            metadata=request.metadata,
            phases=request.phases,
        )
        self._pending = episode
        self._staged_path = self._stage(episode)
        if self._session is not None and not self._session.accept_episode(episode):
            self._pending = None
            self._staged_path.unlink(missing_ok=True)
            _sync_directory(self._staged_path.parent)
            self._staged_path = None
            raise ValueError("Real2Sim expansion session rejected the episode")
        receipt = submit_and_discard_real2sim_episode(
            target, self._sink, episode, submission_id=0, reset_on_confirmed=False
        )
        self._last_receipt = receipt
        if receipt.confirmed:
            self._complete_confirmed(target, receipt)
        elif self._session is not None:
            self._session.apply_receipt(receipt)
        return receipt

    def record_failure(
        self, env: object, result: DemoEpisodeResult, *, error: str | None = None
    ) -> None:
        """Retain host failure evidence before the attempt is reset."""
        target = self._target(env)
        if self._record_failed_attempt is not None:
            self._record_failed_attempt(target, result, error)

    def retry_write(
        self, env: object, *, submission_id: int | None = None
    ) -> CommitReceipt:
        """Retry a failed write using the same retained episode, without execution."""
        target = self._target(env)
        if self._pending is None or self._last_receipt is None:
            raise ValueError("there is no failed Real2Sim write to retry")
        if self._last_receipt.confirmed:
            if submission_id is not None:
                raise ValueError(
                    "receipt publication does not create another submission"
                )
            self._complete_confirmed(target, self._last_receipt)
            return self._last_receipt
        if self._session is not None:
            episode, next_id = self._session.retry_write(self._pending.commit_id)
            if submission_id is not None and submission_id != next_id:
                raise ValueError("submission ID must match the session retry")
        else:
            next_id = self._last_receipt.submission_id + 1
            if submission_id is not None and submission_id != next_id:
                raise ValueError("submission ID must advance by one")
            episode = self._pending
        receipt = submit_and_discard_real2sim_episode(
            target, self._sink, episode, submission_id=next_id, reset_on_confirmed=False
        )
        self._last_receipt = receipt
        if receipt.confirmed:
            self._complete_confirmed(target, receipt)
        elif self._session is not None:
            self._session.apply_receipt(receipt)
        return receipt

    def finalize(self) -> None:
        """Finalize the owned synchronous recorder without executing again."""
        self._sink._recorder.finalize()
