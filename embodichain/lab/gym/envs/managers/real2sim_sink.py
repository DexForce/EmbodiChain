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

"""Synchronous durable recording for source-qualified B=1 episodes."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
import torch
from tensordict import TensorDict

from embodichain.data_pipeline.real2sim import TrajectoryTimeMap, load_prototype
from embodichain.data_pipeline.recording.journal import inspect_recording
from embodichain.lab.gym.envs._json import json_safe_copy
from embodichain.lab.sim.atomic_actions import InstantiatedSkillMotion
from embodichain.lab.sim.motion.expansion import (
    CandidateIdentity,
    CommitReceipt,
    ExpertEpisode,
    ExpansionSession,
    TrajectoryPhase,
    ValidationResult,
)

from .datasets import LeRobotDataset, LeRobotRecorder

__all__ = [
    "Real2SimLeRobotEpisodeSink",
    "build_real2sim_execution_trace",
    "capture_real2sim_episode",
    "submit_and_discard_real2sim_episode",
]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    """Persist a newly published directory entry before acknowledging it."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_real2sim_execution_trace(
    consumed_actions: Sequence[Mapping[str, object]],
    motions: Mapping[int, InstantiatedSkillMotion],
    *,
    retime_maps: Mapping[int, TrajectoryTimeMap] | None = None,
) -> dict[str, np.ndarray]:
    """Build a typed action-to-plan sidecar from acknowledged B=1 actions.

    The bridge records only actions consumed by ``env.step``. A ``-1`` plan
    index identifies a runtime or post-policy action without a source-plan
    sample; repeated plan indices retain effect-gate and terminal commands.
    Retime maps describe expanded plan samples in the nominal plan grid.

    Args:
        consumed_actions: Ordered rows from ``TaskProgramDemoBridge.execution_trace``.
        motions: Captured source motions keyed by compiled call occurrence.
        retime_maps: Optional plan-to-nominal maps for retimed occurrences.

    Returns:
        NPZ-compatible, non-object arrays accepted by the Real2Sim sink.

    Raises:
        ValueError: If a step, call, event or source-plan sample is missing.
    """
    if (
        isinstance(consumed_actions, (str, bytes))
        or not isinstance(consumed_actions, Sequence)
        or not consumed_actions
    ):
        raise ValueError("consumed_actions must contain recorded actions")
    if (
        not isinstance(motions, Mapping)
        or not motions
        or any(
            type(index) is not int
            or index < 0
            or not isinstance(motion, InstantiatedSkillMotion)
            for index, motion in motions.items()
        )
    ):
        raise ValueError("motions must map call occurrences to instantiated sources")
    retimed = {} if retime_maps is None else dict(retime_maps)
    if set(retimed) - set(motions) or any(
        not isinstance(mapping, TrajectoryTimeMap) for mapping in retimed.values()
    ):
        raise ValueError("retime maps must belong to recorded source calls")
    calls, samples = [], []
    for step, row in enumerate(consumed_actions):
        if not isinstance(row, Mapping) or row.get("step") != step:
            raise ValueError(
                "consumed actions must have contiguous actual step indices"
            )
        call, sample = row.get("call_index"), row.get("plan_index")
        if (
            type(call) is not int
            or type(sample) is not int
            or call < -1
            or sample < -1
            or (call == -1 and sample != -1)
            or (call >= 0 and call not in motions)
        ):
            raise ValueError("consumed action refers to an unknown call or sample")
        calls.append(call)
        samples.append(sample)
    arrays = {
        "step_call_index": np.array(calls, dtype=np.int64),
        "step_plan_index": np.array(samples, dtype=np.int64),
    }
    if np.any(np.diff(arrays["step_call_index"][arrays["step_call_index"] >= 0]) < 0):
        raise ValueError("consumed calls must be sequential")
    for index, motion in sorted(motions.items()):
        if motion.source_time_map is None:
            raise ValueError("each source motion requires its normalization time map")
        count = motion.trajectory.waypoint_count
        boundaries = {0, motion.source_sample_offset}
        start = 0
        for length in motion.segment_lengths.values():
            boundaries.update((start, start + length - 1))
            start += length
        if index in retimed:
            mapping = retimed[index]
        else:
            event_indices = np.array(sorted(boundaries), dtype=np.int64)
            samples_identity = np.arange(count, dtype=np.int64)
            mapping = TrajectoryTimeMap(
                samples_identity,
                samples_identity,
                np.zeros(count),
                event_indices,
                event_indices,
                0.0,
            )
        if (
            not len(mapping.left)
            or np.any(mapping.right >= count)
            or np.any(mapping.source_event_indices >= count)
            or np.any(mapping.execution_event_indices >= len(mapping.left))
        ):
            raise ValueError("retimed plan map is outside the source motion")
        event_map = dict(
            zip(
                mapping.source_event_indices.tolist(),
                mapping.execution_event_indices.tolist(),
            )
        )
        if not boundaries.issubset(event_map) or any(
            mapping.left[event_map[event]] != event
            or mapping.right[event_map[event]] != event
            or mapping.weight[event_map[event]] != 0
            for event in boundaries
        ):
            raise ValueError("retimed plan map does not preserve hard source events")
        observed = arrays["step_plan_index"][arrays["step_call_index"] == index]
        observed = observed[observed >= 0]
        if np.any(np.diff(observed) < 0) or set(observed.tolist()) != set(
            range(len(mapping.left))
        ):
            raise ValueError("consumed actions do not cover every plan sample in order")
        prefix = f"call_{index}_"
        if motion.refinement_report:
            arrays[prefix + "refinement_original_tcp"] = (
                motion.refinement_original_tcp.detach().cpu().numpy().copy()
            )
            arrays[prefix + "refinement_target_tcp"] = (
                motion.refinement_target_tcp.detach().cpu().numpy().copy()
            )
        for name, values in (
            ("left", mapping.left),
            ("right", mapping.right),
            ("weight", mapping.weight),
            ("source_events", mapping.source_event_indices),
            ("plan_events", mapping.execution_event_indices),
        ):
            arrays[prefix + name] = np.array(values, copy=True)
    return arrays


def capture_real2sim_episode(
    env: object,
    *,
    identity: CandidateIdentity,
    validation: ValidationResult,
    episode_id: str,
    commit_id: str,
    observation_fields: Mapping[str, tuple[str, ...]],
    action_representation: str,
    metadata: Mapping[str, object],
    phases: tuple[TrajectoryPhase, ...] = (),
) -> ExpertEpisode:
    """Capture one actual B=1 Gym rollout before reset or buffer clearing.

    Every returned observation includes the seeded post-action terminal frame.
    A full buffer without space for that observation is rejected instead of
    synthesizing a state. The caller supplies independent validation evidence.
    """
    target = getattr(env, "unwrapped", env)
    if getattr(target, "num_envs", None) != 1:
        raise ValueError("Real2Sim capture currently requires B=1")
    if getattr(target, "rollout_action_semantics", None) != "expert_controller_command":
        raise ValueError("Real2Sim capture requires actual expert controller commands")
    buffer = getattr(target, "rollout_buffer", None)
    steps = getattr(target, "rollout_steps", None)
    if (
        not isinstance(buffer, TensorDict)
        or not isinstance(steps, torch.Tensor)
        or steps.shape != (1,)
    ):
        raise ValueError("Real2Sim capture requires an active Gym rollout buffer")
    length = int(steps[0].item())
    if length <= 0 or buffer.shape[1] <= length:
        raise ValueError("rollout does not retain a terminal observation")
    if not isinstance(observation_fields, Mapping) or not observation_fields:
        raise ValueError("observation fields must be explicitly mapped")
    observations = {}
    for name, path in observation_fields.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(path, tuple)
            or not path
        ):
            raise ValueError("observation fields require names and TensorDict paths")
        value = buffer["obs"].get(path)
        if not isinstance(value, torch.Tensor) or value.shape[:2] != buffer.shape[:2]:
            raise ValueError(f"rollout observation {name!r} is unavailable")
        observations[name] = value[0, : length + 1].detach().clone()
    actions = buffer["actions"][0, :length].detach().clone()
    step_dt = float(getattr(target, "step_dt", 0.0))
    if not np.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("Gym control period must be finite and positive")
    timestamps = (
        torch.arange(length + 1, dtype=torch.float64, device=actions.device) * step_dt
    )
    return ExpertEpisode(
        identity=identity,
        observations=observations,
        actions=actions,
        timestamps=timestamps,
        action_representation=action_representation,
        validation=validation,
        episode_id=episode_id,
        commit_id=commit_id,
        phases=phases,
        metadata=metadata,
    )


def submit_and_discard_real2sim_episode(
    env: object,
    sink: Real2SimLeRobotEpisodeSink,
    episode: ExpertEpisode,
    *,
    submission_id: int,
    session: ExpansionSession | None = None,
    reset_on_confirmed: bool = True,
) -> CommitReceipt:
    """Submit one owned rollout and prevent Gym from committing it twice.

    The buffer is made ineligible for normal reset-time persistence. By
    default a confirmed receipt resets with ``save_data=False``; a host that
    must publish additional confirmation evidence first can set
    ``reset_on_confirmed=False`` and owns the later reset.
    Unconfirmed submissions retain their blocked buffer for diagnosis.
    If a session is supplied, the initial submission reserves measured
    coverage before writing and applies the returned receipt afterwards. For
    a retry, obtain the payload and next submission ID from
    ``session.retry_write``; a retry never re-executes the episode.
    """
    target = getattr(env, "unwrapped", env)
    if (
        not isinstance(sink, Real2SimLeRobotEpisodeSink)
        or sink._recorder._env is not target
    ):
        raise ValueError(
            "sink must own a recorder bound to the supplied Gym environment"
        )
    if session is not None and not isinstance(session, ExpansionSession):
        raise TypeError("session must be an ExpansionSession or None")
    if type(reset_on_confirmed) is not bool:
        raise TypeError("reset_on_confirmed must be a boolean")
    context = getattr(target, "episode_context", None)
    if context is None or not callable(getattr(context, "check", None)):
        raise ValueError("Gym environment must expose its EpisodeContext")
    with context.check("real2sim/sink", [0]) as finish:
        if (
            session is not None
            and submission_id == 0
            and not session.accept_episode(episode)
        ):
            finish(
                accepted=False,
                reasons="Real2Sim validation or coverage rejected the episode",
            )
            raise ValueError("Real2Sim expansion session rejected the episode")
        receipt = sink.submit(episode, submission_id=submission_id)
        if not isinstance(receipt, CommitReceipt) or (
            receipt.episode_id,
            receipt.candidate_id,
            receipt.attempt_id,
            receipt.commit_id,
            receipt.scene_case_id,
            receipt.submission_id,
        ) != (
            episode.episode_id,
            episode.identity.candidate_id,
            episode.identity.attempt_id,
            episode.commit_id,
            episode.identity.scene_case_id,
            submission_id,
        ):
            raise ValueError("Real2Sim receipt does not match the submitted episode")
        if session is not None:
            session.apply_receipt(receipt)
        finish(accepted=False, reasons="recording delegated to Real2Sim sink")
    if receipt.confirmed and reset_on_confirmed:
        target.reset(options={"save_data": False})
    return receipt


class Real2SimLeRobotEpisodeSink:
    """Commit accepted, sensor-free episodes through a synchronous recorder.

    The sink closes the SDK's buffered writers and verifies the existing
    recording journal before confirming each receipt. It reopens the same
    dataset for the next episode. It never steps or resets the environment.
    Source bundles, the terminal observation and an explicit action-to-plan
    trace are copied as typed artifacts. An absent, incomplete or unmappable
    trace prevents a confirmed receipt; runtime holds must be identified
    instead of being attributed to an invented source sample.

    Args:
        recorder: A synchronous B=1 LeRobot recorder without image/video data.
        observation_fields: Explicit source observation keys to TensorDict
            paths expected by the recorder, such as ``("robot", "qpos")``.
        action_representation: Exact recorded action representation.
        required_check_ids: Explicit host-owned motion, collision and task
            evidence that must all be present and passing before submission.
    """

    def __init__(
        self,
        recorder: LeRobotRecorder,
        *,
        observation_fields: Mapping[str, tuple[str, ...]],
        action_representation: str,
        required_check_ids: tuple[str, ...],
    ) -> None:
        if type(recorder) is not LeRobotRecorder:
            raise TypeError("a synchronous LeRobotRecorder is required")
        if (
            recorder._env.num_envs != 1
            or recorder.use_videos
            or recorder._env.has_sensors
        ):
            raise ValueError("the initial sink requires B=1 and no sensors or video")
        if getattr(recorder, "_depth_manager", None) is not None:
            raise ValueError("the initial sink does not support depth sidecars")
        if not isinstance(observation_fields, Mapping) or not observation_fields:
            raise ValueError(
                "observation_fields must explicitly map source observations"
            )
        fields = {key: tuple(path) for key, path in observation_fields.items()}
        if any(
            not isinstance(key, str)
            or not key
            or not path
            or any(not isinstance(part, str) or not part for part in path)
            for key, path in fields.items()
        ):
            raise ValueError(
                "observation mappings must contain nonempty names and paths"
            )
        if len(set(fields.values())) != len(fields):
            raise ValueError("observation destination paths must be unique")
        contract = recorder._action_contract()
        representation = (
            contract["representation"]
            if contract is not None
            else (
                "joint_position_velocity"
                if getattr(
                    getattr(recorder._env, "expert_action_spec", None),
                    "joint_command_mode",
                    "position",
                )
                == "position_velocity"
                else "joint_position"
            )
        )
        if action_representation != representation:
            raise ValueError(
                "action representation does not match the recorder contract"
            )
        check_ids = tuple(required_check_ids)
        if (
            not check_ids
            or len(set(check_ids)) != len(check_ids)
            or any(not isinstance(name, str) or not name.strip() for name in check_ids)
        ):
            raise ValueError("required_check_ids must explicitly name unique checks")
        self._recorder = recorder
        self._fields = fields
        self._representation = representation
        self._required_check_ids = check_ids
        self._broken = False

    def _source_artifacts(
        self, episode: ExpertEpisode, root: Path
    ) -> dict[str, dict[str, dict[str, str]]]:
        metadata = episode.metadata.get("real2sim_source_artifacts")
        required = {
            "prototype_manifest",
            "normalized_arrays",
            "source_manifest",
            "source_arrays",
        }
        if not isinstance(metadata, Mapping) or not metadata:
            raise ValueError("episode must declare Real2Sim source artifacts")
        bundles = {"default": metadata} if set(metadata) == required else dict(metadata)
        if any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(bundle, Mapping)
            or set(bundle) != required
            for name, bundle in bundles.items()
        ):
            raise ValueError(
                "each source bundle must declare all four Real2Sim source artifacts"
            )
        return {
            name: self._copy_source_bundle(bundle, root)
            for name, bundle in bundles.items()
        }

    def _copy_source_bundle(
        self, metadata: Mapping, root: Path
    ) -> dict[str, dict[str, str]]:
        paths: dict[str, Path] = {}
        for role, item in metadata.items():
            if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
                raise ValueError("source artifact needs path and sha256")
            path = Path(item["path"]).resolve()
            digest = item["sha256"]
            if (
                not path.is_file()
                or not isinstance(digest, str)
                or len(digest) != 64
                or _digest(path) != digest
            ):
                raise ValueError(f"source artifact {role!r} is missing or changed")
            paths[role] = path
        prototype = paths["prototype_manifest"]
        if (
            paths["normalized_arrays"] != prototype.parent / "normalized.npz"
            or paths["source_manifest"]
            != prototype.parent / "source" / "trajectory.json"
            or paths["source_arrays"] != prototype.parent / "source" / "trajectory.npz"
        ):
            raise ValueError("source artifacts do not describe one prototype bundle")
        load_prototype(prototype)
        assets = (
            root
            / "meta"
            / "real2sim"
            / "prototypes"
            / metadata["prototype_manifest"]["sha256"]
        )
        relative_names = {
            "prototype_manifest": "prototype.json",
            "normalized_arrays": "normalized.npz",
            "source_manifest": "source/trajectory.json",
            "source_arrays": "source/trajectory.npz",
        }
        copied = {}
        for role, source in paths.items():
            digest = _digest(source)
            destination = assets / relative_names[role]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if _digest(destination) != digest:
                    raise ValueError(
                        "existing source artifact failed digest validation"
                    )
            else:
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=destination.parent, delete=False
                    ) as stream:
                        temporary = Path(stream.name)
                        with source.open("rb") as incoming:
                            shutil.copyfileobj(incoming, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    if _digest(temporary) != digest:
                        raise ValueError("source artifact changed while copying")
                    os.link(temporary, destination)
                    _sync_directory(destination.parent)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            copied[role] = {
                "path": destination.relative_to(root).as_posix(),
                "sha256": digest,
            }
        load_prototype(assets / "prototype.json")
        return copied

    def _episode_artifact(
        self, episode: ExpertEpisode, root: Path, identity: str
    ) -> dict[str, object]:
        directory = root / "meta" / "real2sim" / "episodes"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{identity}.npz"
        names = sorted(episode.observations)
        arrays = {
            f"observation_{index}": episode.observations[name].detach().cpu().numpy()
            for index, name in enumerate(names)
        }
        arrays["actions"] = episode.actions.detach().cpu().numpy()
        arrays["timestamps"] = episode.timestamps.detach().cpu().numpy()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=directory, suffix=".npz", delete=False
            ) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            digest = _digest(temporary)
            if path.exists():
                if _digest(path) != digest:
                    raise ValueError(
                        "existing episode artifact does not match this submission"
                    )
            else:
                os.link(temporary, path)
                _sync_directory(path.parent)
            return {
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "observation_names": names,
            }
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _observed_geometry_artifact(
        self, episode: ExpertEpisode, root: Path, identity: str
    ) -> dict[str, str] | None:
        """Verify and archive measured object poses used by collision checks."""
        entry = episode.metadata.get("real2sim_observed_geometry")
        if entry is None:
            if "observed_collision" in self._required_check_ids:
                raise ValueError(
                    "observed collision requires the measured geometry sidecar"
                )
            return None
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise ValueError("observed geometry requires a path and digest")
        source = Path(entry["path"]).resolve()
        digest = entry["sha256"]
        if (
            not source.is_file()
            or source.stat().st_size > 64 * 1024 * 1024
            or not isinstance(digest, str)
            or len(digest) != 64
            or _digest(source) != digest
        ):
            raise ValueError(
                "observed geometry sidecar is missing, too large, or changed"
            )
        with zipfile.ZipFile(source) as archive:
            entries = archive.infolist()
            if (
                len({item.filename for item in entries}) != len(entries)
                or sum(item.file_size for item in entries) > 128 * 1024 * 1024
            ):
                raise ValueError("observed geometry sidecar exceeds archive limits")
        with np.load(source, allow_pickle=False) as payload:
            if set(payload.files) != {"object_pose", "target_pose", "timestamps"}:
                raise ValueError("observed geometry arrays are incomplete")
            objects = np.asarray(payload["object_pose"])
            target = np.asarray(payload["target_pose"])
            timestamps = np.asarray(payload["timestamps"])
            if (
                objects.shape != (len(episode.actions) + 1, 7)
                or target.shape != (4, 4)
                or timestamps.shape != (len(episode.actions) + 1,)
                or not np.isfinite(objects).all()
                or not np.isfinite(target).all()
                or not np.isfinite(timestamps).all()
                or not np.allclose(
                    np.linalg.norm(objects[:, 3:], axis=1), 1.0, atol=1e-4, rtol=0
                )
                or not np.allclose(target[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
                or not np.allclose(
                    target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-5, rtol=0
                )
                or not np.isclose(np.linalg.det(target[:3, :3]), 1.0, atol=1e-5, rtol=0)
                or not np.allclose(
                    timestamps,
                    episode.timestamps.detach().cpu().numpy(),
                    atol=1e-8,
                    rtol=0,
                )
            ):
                raise ValueError(
                    "observed geometry does not align with the actual episode"
                )
        directory = root / "meta" / "real2sim" / "episodes"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{identity}-observed.npz"
        if destination.exists():
            if _digest(destination) != digest:
                raise ValueError(
                    "existing observed geometry differs from this submission"
                )
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                    temporary = Path(stream.name)
                    with source.open("rb") as incoming:
                        shutil.copyfileobj(incoming, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                if _digest(temporary) != digest:
                    raise ValueError("observed geometry changed while copying")
                os.link(temporary, destination)
                _sync_directory(destination.parent)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return {"path": destination.relative_to(root).as_posix(), "sha256": digest}

    def _execution_trace(
        self,
        episode: ExpertEpisode,
        root: Path,
        identity: str,
        bundles: Mapping[str, Mapping[str, Mapping[str, str]]],
    ) -> dict[str, object]:
        """Validate and archive the layered, per-submitted-action time map."""
        lineage = episode.metadata.get("real2sim_lineage")
        if not isinstance(lineage, Mapping):
            raise ValueError("episode must declare Real2Sim execution lineage")
        revision, fingerprint, trace, calls = (
            lineage.get(name)
            for name in ("scene_revision", "program_fingerprint", "trace", "calls")
        )
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("lineage requires a scene revision")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)
        ):
            raise ValueError("lineage requires a compiled-program SHA-256 fingerprint")
        if not isinstance(trace, Mapping) or set(trace) != {"path", "sha256"}:
            raise ValueError("lineage trace requires path and sha256")
        source = Path(trace["path"]).resolve()
        digest = trace["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not source.is_file()
            or source.stat().st_size > 64 * 1024 * 1024
            or _digest(source) != digest
        ):
            raise ValueError("lineage trace is missing, too large, or changed")
        if not isinstance(calls, (tuple, list)) or not calls:
            raise ValueError("lineage requires call-level source bindings")
        indices: set[int] = set()
        normalized_calls = []
        required_arrays = {"step_call_index", "step_plan_index"}
        refinements: dict[int, Mapping[str, object]] = {}
        for call in calls:
            if not isinstance(call, Mapping):
                raise ValueError("lineage calls must be mappings")
            index, bundle_name, family, offset, operator, prototype_hash = (
                call.get(name)
                for name in (
                    "call_index",
                    "source_bundle",
                    "geometry_family_id",
                    "source_sample_offset",
                    "operator",
                    "prototype_sha256",
                )
            )
            if type(index) is not int or index < 0 or index in indices:
                raise ValueError("lineage call indices must be unique and nonnegative")
            if (
                bundle_name not in bundles
                or family != episode.identity.geometry_family_id
                or type(offset) is not int
                or offset < 0
                or not isinstance(operator, str)
                or not operator.strip()
            ):
                raise ValueError(
                    "lineage call has invalid bundle, geometry, offset, or operator"
                )
            manifest = bundles[bundle_name]["prototype_manifest"]
            if prototype_hash != manifest["sha256"]:
                raise ValueError(
                    "lineage prototype hash does not match its source bundle"
                )
            prototype = load_prototype(root / manifest["path"])
            indices.add(index)
            normalized_calls.append((index, offset, prototype))
            refinement = call.get("refinement")
            if refinement:
                if (
                    prototype.unit.world_tcp is None
                    or not isinstance(refinement, Mapping)
                    or refinement.get("schema") != "kinematic-refinement/v1"
                    or refinement.get("samples") != len(prototype.unit.timestamps)
                    or refinement.get("time_grid_changed") is not False
                ):
                    raise ValueError(
                        "lineage contains an invalid refinement declaration"
                    )
                refinements[index] = refinement
                required_arrays.update(
                    f"call_{index}_refinement_{name}_tcp"
                    for name in ("original", "target")
                )
            required_arrays.update(
                f"call_{index}_{name}"
                for name in ("left", "right", "weight", "source_events", "plan_events")
            )
        with zipfile.ZipFile(source) as archive:
            entries = archive.infolist()
            if len({entry.filename for entry in entries}) != len(entries):
                raise ValueError("lineage trace contains duplicate archive members")
            if sum(item.file_size for item in entries) > 128 * 1024 * 1024:
                raise ValueError("lineage trace expands beyond its memory budget")
        with np.load(source, allow_pickle=False) as payload:
            if set(payload.files) != required_arrays:
                raise ValueError("lineage trace arrays do not match declared calls")
            steps = np.array(payload["step_call_index"], copy=True)
            positions = np.array(payload["step_plan_index"], copy=True)
            if (
                steps.shape != (len(episode.actions),)
                or positions.shape != steps.shape
                or steps.dtype.kind != "i"
                or positions.dtype.kind != "i"
            ):
                raise ValueError(
                    "lineage needs one integer call and plan index per submitted action"
                )
            if (
                np.any(steps < -1)
                or np.any(positions < -1)
                or np.any((steps == -1) & (positions != -1))
                or any(int(index) not in indices for index in steps if index >= 0)
            ):
                raise ValueError("lineage action indices reference undeclared calls")
            active_calls = steps[steps >= 0]
            if np.any(np.diff(active_calls) < 0):
                raise ValueError("lineage calls must remain in execution order")
            for index, offset, prototype in normalized_calls:
                prefix = f"call_{index}_"
                if index in refinements:
                    from embodichain.compute.trajectory.refinement import _poses
                    from scipy.spatial.transform import Rotation

                    declaration = refinements[index]
                    original = _poses(
                        payload[prefix + "refinement_original_tcp"],
                        "refinement original",
                    )
                    target = _poses(
                        payload[prefix + "refinement_target_tcp"], "refinement target"
                    )
                    if (
                        original.shape != (len(prototype.unit.timestamps), 4, 4)
                        or target.shape != original.shape
                    ):
                        raise ValueError(
                            "lineage refinement arrays do not match the source grid"
                        )
                    refinement_digest = hashlib.sha256()
                    refinement_digest.update(
                        np.asarray(original, dtype="<f8").tobytes()
                    )
                    refinement_digest.update(np.asarray(target, dtype="<f8").tobytes())
                    for name in ("profile_sha256", "geometry_id"):
                        value = declaration.get(name)
                        if not isinstance(value, str) or len(value) != 64:
                            raise ValueError("lineage refinement identity is missing")
                        refinement_digest.update(value.encode())
                    if refinement_digest.hexdigest() != declaration.get("baseline_id"):
                        raise ValueError(
                            "lineage refinement baseline digest does not match"
                        )
                    distance = float(
                        np.linalg.norm(
                            target[:, :3, 3] - original[:, :3, 3], axis=1
                        ).max()
                    )
                    angle = float(
                        Rotation.from_matrix(
                            target[:, :3, :3] @ original[:, :3, :3].transpose(0, 2, 1)
                        )
                        .magnitude()
                        .max()
                    )
                    for measured, name, budget in (
                        (distance, "max_translation_m", "translation_budget_m"),
                        (angle, "max_rotation_rad", "rotation_budget_rad"),
                    ):
                        reported, limit = declaration.get(name), declaration.get(budget)
                        if (
                            not isinstance(reported, (int, float))
                            or not isinstance(limit, (int, float))
                            or not np.isfinite([reported, limit]).all()
                            or abs(measured - reported) > 1e-7
                            or measured > limit + 1e-7
                        ):
                            raise ValueError(
                                "lineage refinement correction exceeds its declared budget"
                            )
                mapping = TrajectoryTimeMap(
                    payload[prefix + "left"],
                    payload[prefix + "right"],
                    payload[prefix + "weight"],
                    payload[prefix + "source_events"],
                    payload[prefix + "plan_events"],
                    0.0,
                )
                nominal_count = offset + len(prototype.unit.timestamps)
                if (
                    not len(mapping.left)
                    or np.any(mapping.right >= nominal_count)
                    or np.any(mapping.source_event_indices >= nominal_count)
                    or np.any(mapping.execution_event_indices >= len(mapping.left))
                ):
                    raise ValueError("lineage map refers outside its source or plan")
                if np.any(
                    (mapping.left < offset)
                    & (mapping.right >= offset)
                    & (mapping.weight > 0)
                    & (mapping.weight < 1)
                ):
                    raise ValueError(
                        "lineage map cannot interpolate across a connection boundary"
                    )
                event_map = dict(
                    zip(
                        mapping.source_event_indices.tolist(),
                        mapping.execution_event_indices.tolist(),
                    )
                )
                required_events = {
                    offset + value
                    for phase in prototype.unit.phases
                    for value in (phase.start, phase.stop - 1)
                } | {0, offset}
                if not required_events.issubset(event_map):
                    raise ValueError(
                        "lineage map is missing a hard phase or connection event"
                    )
                for event in required_events:
                    dest = event_map[event]
                    if (
                        mapping.left[dest] != event
                        or mapping.right[dest] != event
                        or mapping.weight[dest] != 0
                    ):
                        raise ValueError(
                            "lineage hard events must map to exact plan samples"
                        )
                selected = positions[(steps == index) & (positions >= 0)]
                if (
                    np.any(selected >= len(mapping.left))
                    or np.any(np.diff(selected) < 0)
                    or set(selected.tolist()) != set(range(len(mapping.left)))
                ):
                    raise ValueError(
                        "lineage does not account for every plan sample in execution order"
                    )
        directory = root / "meta" / "real2sim" / "episodes"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{identity}-trace.npz"
        if destination.exists():
            if _digest(destination) != digest:
                raise ValueError(
                    "existing execution trace differs from this submission"
                )
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                    temporary = Path(stream.name)
                    with source.open("rb") as incoming:
                        shutil.copyfileobj(incoming, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                if _digest(temporary) != digest:
                    raise ValueError("execution trace changed while copying")
                os.link(temporary, destination)
                _sync_directory(destination.parent)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return {"path": destination.relative_to(root).as_posix(), "sha256": digest}

    def submit(self, episode: ExpertEpisode, *, submission_id: int) -> CommitReceipt:
        """Commit one accepted episode and return its disk-verified receipt."""
        if not isinstance(episode, ExpertEpisode):
            raise TypeError("episode must be an ExpertEpisode")
        if type(submission_id) is not int or submission_id < 0:
            raise ValueError("submission_id must be a nonnegative integer")
        if not episode.validation.accepted:
            raise ValueError("only measured and accepted episodes may be recorded")
        if not set(self._required_check_ids).issubset(
            {check.check_id for check in episode.validation.checks}
        ):
            raise ValueError("episode lacks required physical validation checks")
        if episode.action_representation != self._representation:
            raise ValueError(
                "episode action representation does not match the recorder"
            )
        if set(self._fields) - set(episode.observations):
            raise ValueError("episode is missing an explicitly mapped observation")
        if not torch.allclose(
            episode.timestamps[1:] - episode.timestamps[:-1],
            torch.full_like(episode.timestamps[1:], float(self._recorder._env.step_dt)),
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError(
                "episode timestamps do not match the recorder control period"
            )
        root = self._recorder.dataset_full_path
        if root is None:
            raise RuntimeError("recorder dataset is not initialized")
        root = Path(root).resolve()
        episode_uuid = str(
            uuid5(NAMESPACE_URL, "embodichain-commit:" + episode.commit_id)
        )
        identity = episode.identity
        receipt_args = dict(
            episode_id=episode.episode_id,
            candidate_id=identity.candidate_id,
            attempt_id=identity.attempt_id,
            storage_id=f"{root}#{episode_uuid}",
            commit_id=episode.commit_id,
            scene_case_id=identity.scene_case_id,
            submission_id=submission_id,
        )
        with self._recorder._finalize_lock:
            if self._broken or self._recorder._finalized:
                raise RuntimeError(
                    "recorder is unavailable after finalization or a partial commit"
                )
            try:
                artifacts = self._source_artifacts(episode, root)
                trace = self._execution_trace(episode, root, episode_uuid, artifacts)
                evidence = self._episode_artifact(episode, root, episode_uuid)
                observed_geometry = self._observed_geometry_artifact(
                    episode, root, episode_uuid
                )
                metadata = json_safe_copy(
                    episode.metadata, field_name="episode metadata"
                )
                metadata.pop("real2sim_source_artifacts", None)
                metadata["real2sim_lineage"]["trace"] = trace
                if observed_geometry is not None:
                    metadata["real2sim_observed_geometry"] = observed_geometry
                source_uuid = metadata.get("source_episode_uuid") or str(
                    uuid5(
                        NAMESPACE_URL,
                        f"embodichain-source:{identity.source_id}:{identity.source_revision}",
                    )
                )
                source_uuid = str(UUID(str(source_uuid)))
                metadata.update(
                    {
                        "episode_uuid": episode_uuid,
                        "source_episode_uuid": source_uuid,
                        "real2sim_artifacts": artifacts,
                        "real2sim_episode_evidence": evidence,
                        "candidate_id": identity.candidate_id,
                        "attempt_id": identity.attempt_id,
                        "commit_id": episode.commit_id,
                        "scene_case_id": identity.scene_case_id,
                        "initial_state_id": identity.initial_state_id,
                        "geometry_family_id": identity.geometry_family_id,
                        "source_id": identity.source_id,
                        "source_revision": identity.source_revision,
                        "source_unit_id": identity.template_id,
                        "source_phases": [
                            {
                                "name": phase.phase_id,
                                "start": phase.start_index,
                                "stop": phase.stop_index,
                                "kind": phase.kind,
                            }
                            for phase in episode.phases
                        ],
                        "validation_checks": [
                            {
                                "id": check.check_id,
                                "status": check.status,
                                "detail": check.detail,
                                "metrics": dict(check.metrics),
                            }
                            for check in episode.validation.checks
                        ],
                    }
                )
                observations = TensorDict({}, batch_size=[episode.actions.shape[0]])
                for name, destination in self._fields.items():
                    observations.set(
                        destination, episode.observations[name][:-1].detach().cpu()
                    )
                actions = episode.actions.detach().cpu()
                saved = self._recorder._persist_episode_payload(
                    0, observations, actions, episode_metadata=metadata
                )
                if not saved:
                    raise RuntimeError("recorder did not persist the episode")
                previous = self._recorder.dataset
                previous.finalize()
                report = inspect_recording(root)
                entry = next(
                    (
                        item
                        for item in report["commits"]
                        if item["episode_uuid"] == episode_uuid
                    ),
                    None,
                )
                if report["errors"] or entry is None or entry["status"] != "complete":
                    raise RuntimeError(
                        "recording journal or on-disk episode verification failed"
                    )
                required_artifacts = [
                    item for bundle in artifacts.values() for item in bundle.values()
                ]
                additional = () if observed_geometry is None else (observed_geometry,)
                for item in (*required_artifacts, evidence, trace, *additional):
                    if _digest(root / item["path"]) != item["sha256"]:
                        raise RuntimeError(
                            "required source or episode artifact is incomplete"
                        )
                self._recorder.dataset = LeRobotDataset(
                    repo_id=previous.repo_id, root=root, download_videos=False
                )
                return CommitReceipt(**receipt_args, confirmed=True)
            except Exception as exc:
                journal = getattr(self._recorder, "_recording_journal", None)
                state = None if journal is None else journal.get(episode_uuid)
                if state is not None and state["phase"] != "prepared":
                    self._broken = True
                return CommitReceipt(
                    **receipt_args,
                    confirmed=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
