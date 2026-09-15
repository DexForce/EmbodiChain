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

"""Full-batch qpos rollouts with actual controller labels and causal evidence."""

from __future__ import annotations

import math
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any

import torch

from embodichain.lab.sim.motion.expansion import (
    CandidateIdentity,
    CandidateTrajectoryBatch,
    ExpertEpisode,
    ValidationCheck,
    ValidationResult,
)
from .initial_state import FixedSceneHost, PreparedBatch
from .integrations.contact import PickUpMotionValidator

__all__ = ["QposRolloutExecutor"]

_METADATA_RESERVE = 64 * 1024
_EXECUTOR_METADATA_LIMIT = _METADATA_RESERVE // 2


def _failure_text(error: Exception) -> str:
    """Retain the concrete cause hidden by an execution-loop wrapper."""
    seen = {id(error)}
    while error.__cause__ is not None and id(error.__cause__) not in seen:
        error = error.__cause__
        seen.add(id(error))
    return f"{type(error).__name__}: {error}"[:512]


def _metadata_bytes(value: object) -> int:
    def encode(item: object) -> dict:
        if isinstance(item, Mapping):
            return dict(item)
        return {entry.name: getattr(item, entry.name) for entry in fields(item)}

    size = 0
    for chunk in json.JSONEncoder(default=encode, ensure_ascii=False).iterencode(value):
        size += len(chunk.encode())
        if size > _EXECUTOR_METADATA_LIMIT:
            break
    return size


@dataclass
class _Row:
    candidate: CandidateTrajectoryBatch
    steps: int
    observations: list[dict[str, torch.Tensor]] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=lambda: [0.0])
    started: bool = False
    failure: str | None = None
    fixed_world: bool = True


def _flatten_observation(value: Mapping, prefix: str = "") -> dict[str, torch.Tensor]:
    """Preserve every nested tensor when no explicit Gym encoder is supplied."""
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or "." in key:
            raise ValueError(
                "Default Gym observation keys must be nonempty and contain no dots."
            )
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, torch.Tensor):
            result[name] = item
        elif isinstance(item, Mapping):
            result.update(_flatten_observation(item, name))
        else:
            raise ValueError(
                f"Observation field {name!r} requires an explicit encoder."
            )
    return result


class QposRolloutExecutor:
    """Execute one candidate per physical row through Gym or a pure simulator.

    Each candidate contains one initial sample followed by controller targets;
    ``valid_length - 1`` commands are executed on the fixed control clock.
    Missing, finished, and terminated rows receive safe holds while remaining
    rows advance in a single batched physics update. Holds and padding never
    enter returned episodes. Gym execution uses the normal demonstration loop
    and ``ControllerAction``; policy preprocessing is skipped once and normal
    postprocessing remains enabled. Actual targets are captured before physics.

    The executor supports changes only to the host's active joints. Inactive
    or mimic coordinates must remain constant unless a shared PickUp validator
    models their complete geometry and physical motion. Task
    validation runs at the final batch boundary; early trajectory interruption,
    target disagreement, and observation/physics errors always reject evidence.
    A submitted command without a complete observation transition still calls
    ``on_started``, but cannot produce a fabricated training transition.

    Args:
        host: Exclusive fixed-scene host; prepare it before calling execute.
        control_dt: Control period, an integer multiple of profile physics_dt.
        observe: Required pure-simulator observation callback, returning full
            batch tensors. It is never used for Gym execution.
        encode_observation: Optional Gym observation encoder. The default
            flattens every tensor field. Uses prepared/step observations only.
        validator: Trusted final task validator for one candidate and its owned
            ``observations, actions, timestamps``. Must include ``validator_id``.
        validator_id: Required task validation check identifier.
        validation_profile_id: Registered execution validation profile identity.
        max_episode_bytes: Maximum tensor plus metadata payload per row. Before
            commands, reserve 64 KiB for bounded metadata in addition to tensors;
            half remains available for downstream validation and storage schema.
        contact_validator: Optional PickUp validator sharing this simulator and
            robot. Adds physical-substep contact gates and measured object/TCP
            observations; permits its target and observed mimic joints to move.
            Supported only with pure simulation and the matching Runner planner.
    """

    def __init__(
        self,
        host: FixedSceneHost,
        *,
        control_dt: float,
        observe: Callable[[], Mapping[str, torch.Tensor]] | None = None,
        encode_observation: Callable[[Any], Mapping[str, torch.Tensor]] | None = None,
        validator: Callable[
            [
                CandidateTrajectoryBatch,
                Mapping[str, torch.Tensor],
                torch.Tensor,
                torch.Tensor,
            ],
            ValidationResult,
        ],
        validator_id: str = "task_success",
        validation_profile_id: str = "verified_motion",
        max_episode_bytes: int = 256 * 1024 * 1024,
        contact_validator: PickUpMotionValidator | None = None,
    ) -> None:
        if (
            isinstance(control_dt, bool)
            or not isinstance(control_dt, (int, float))
            or not math.isfinite(control_dt)
            or control_dt <= 0
        ):
            raise ValueError("control_dt must be finite and positive.")
        if not callable(validator) or (host.env is None and not callable(observe)):
            raise ValueError(
                "Provide a task validator and a pure-sim observation callback when needed."
            )
        if encode_observation is not None and not callable(encode_observation):
            raise TypeError("encode_observation must be callable.")
        if type(max_episode_bytes) is not int or max_episode_bytes <= 0:
            raise ValueError("max_episode_bytes must be a positive integer.")
        for name, value in (
            ("validator_id", validator_id),
            ("validation_profile_id", validation_profile_id),
        ):
            if not isinstance(value, str) or not value or len(value.encode()) > 1024:
                raise ValueError(f"{name} must be nonempty text of at most 1024 bytes.")
        if validator_id in {"execution_complete", "fixed_collision_world"}:
            raise ValueError("validator_id cannot replace an execution check.")
        ratio = control_dt / host.profile.physics_dt
        if round(ratio) < 1 or not math.isclose(
            ratio, round(ratio), rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError(
                "control_dt must contain an integer number of physics steps."
            )
        if host.env is not None and not math.isclose(
            control_dt, host.env.step_dt, rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError("control_dt must match the Gym control clock.")
        self.host = host
        self.control_dt = float(control_dt)
        self.observe = observe
        self.encode_observation = encode_observation or _flatten_observation
        self.validator = validator
        self.validator_id = validator_id
        self.validation_profile_id = validation_profile_id
        self.max_episode_bytes = max_episode_bytes
        if contact_validator is not None and (
            not isinstance(contact_validator, PickUpMotionValidator)
            or host.env is not None
            or contact_validator.robot is not host.adapter.robot
            or contact_validator.sim is not host.adapter.sim
        ):
            raise ValueError(
                "Contact validation requires the same pure-sim host robot."
            )
        self.contact_validator = contact_validator
        self._substeps = round(ratio)
        self._running = False
        self._last_failures: Mapping[str, str] = MappingProxyType({})

    @property
    def last_failures(self) -> Mapping[str, str]:
        """Read failure reasons from the last batch, including zero-frame rows.

        Cleared when a new serialized ``execute`` call begins. The owned,
        immutable mapping contains at most one entry per candidate, each
        limited to 512 characters. Failed physics/observation transitions retain
        their concrete exception cause even when no episode can be returned.
        """
        return self._last_failures

    def _observations(
        self, raw: Any = None, schema: dict | None = None
    ) -> dict[str, torch.Tensor]:
        values = (
            self.observe() if self.host.env is None else self.encode_observation(raw)
        )
        if not isinstance(values, Mapping) or not values:
            raise ValueError(
                "Observation providers must return a nonempty tensor mapping."
            )
        values = dict(values)
        if self.contact_validator is not None:
            physical = self.contact_validator.observations()
            if any(
                key in values and not torch.equal(values[key], value)
                for key, value in physical.items()
            ):
                raise ValueError(
                    "Contact observations must contain actual simulator poses."
                )
            values.update(physical)
        joints = self.host.adapter.robot.get_qpos()
        if "joint_positions" in values and not torch.equal(
            values["joint_positions"], joints
        ):
            raise ValueError(
                "joint_positions must contain actual full-joint measurements."
            )
        values["joint_positions"] = joints
        batch = self.host.adapter.sim.num_envs
        if joints.shape != (batch, len(self.host.adapter.robot.joint_names)):
            raise ValueError(
                "Measured joint positions do not match the full joint layout."
            )
        for key, value in values.items():
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(value, torch.Tensor)
                or value.ndim < 1
                or value.shape[0] != batch
                or value.is_complex()
            ):
                raise ValueError(f"Invalid full-batch observation field {key!r}.")
        if (
            sum(value.numel() * value.element_size() for value in values.values())
            > batch * self.max_episode_bytes
        ):
            raise ValueError(
                "Observation batch exceeds the payload budget before ownership copy."
            )
        if (
            schema is not None
            and {key: (value.shape, value.dtype) for key, value in values.items()}
            != schema
        ):
            raise ValueError("Observation schema changed during the rollout.")
        if not all(bool(torch.isfinite(value).all()) for value in values.values()):
            raise ValueError("Observations must contain finite real values.")
        return {key: value.detach().cpu().clone() for key, value in values.items()}

    def _active_joints(self) -> tuple[int, ...]:
        source = self.host.env if self.host.env is not None else self.host.adapter.robot
        indices = tuple(int(value) for value in source.active_joint_ids)
        dof = len(self.host.adapter.robot.joint_names)
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(index < 0 or index >= dof for index in indices)
            or not set(indices).issubset(self.host.adapter.robot.active_joint_ids)
        ):
            raise ValueError("The host must declare unique active joint indices.")
        return indices

    def _preflight(
        self,
        binding: PreparedBatch,
        candidates: Sequence[CandidateTrajectoryBatch | None],
        episode_ids: Mapping[str, tuple[str, str]],
        initial: dict[str, torch.Tensor],
        joints: tuple[int, ...],
    ) -> list[_Row | None]:
        batch = self.host.adapter.sim.num_envs
        if len(candidates) != batch:
            raise ValueError(
                "Candidates must cover every physical row, using None for idle rows."
            )
        if not self.host.verify_initial(binding).accepted:
            raise ValueError(
                "The physical batch no longer matches its prepared initial state."
            )
        robot = self.host.adapter.robot
        inactive = [
            index
            for index in range(len(robot.joint_names))
            if index not in joints
            and (
                self.contact_validator is None
                or index not in self.contact_validator.mimic_ids
            )
        ]
        limits = robot.get_qpos_limits().detach().cpu()
        rows = []
        seen = set()
        for index, candidate in enumerate(candidates):
            if candidate is None:
                rows.append(None)
                continue
            if (
                not isinstance(candidate, CandidateTrajectoryBatch)
                or len(candidate.identities) != 1
            ):
                raise ValueError("Each physical row requires a single owned candidate.")
            candidate = candidate.row(0)
            identity = candidate.identities[0]
            if identity.candidate_id in seen:
                raise ValueError("A candidate cannot execute in two rows at once.")
            seen.add(identity.candidate_id)
            if (
                identity.candidate_id not in episode_ids
                or len(episode_ids[identity.candidate_id]) != 2
                or any(
                    not isinstance(value, str) or not value
                    for value in episode_ids[identity.candidate_id]
                )
            ):
                raise ValueError(
                    "Every candidate requires stable episode and commit IDs."
                )
            case = binding.cases[index]
            if (
                identity.scene_case_id != case.scene_case_id
                or identity.initial_state_id != case.initial_state_id
                or candidate.joint_names != tuple(robot.joint_names)
            ):
                raise ValueError(
                    "Candidate case, initial state, or joint order does not match its slot."
                )
            length = int(candidate.valid_length[0])
            if length < 2:
                raise ValueError(
                    "A rollout requires an initial sample and at least one command."
                )
            positions = candidate.positions[0, :length].detach().cpu()
            intervals = candidate.dt[0, 1:length].detach().cpu().to(torch.float64)
            if not torch.allclose(
                intervals,
                torch.full_like(intervals, self.control_dt),
                atol=1e-8,
                rtol=0,
            ):
                raise ValueError(
                    "Candidate timing must use the executor's control clock."
                )
            if not torch.allclose(
                positions[0],
                initial["joint_positions"][index].to(positions.dtype),
                atol=self.host.adapter.atol,
                rtol=0,
            ):
                raise ValueError(
                    "Candidate first sample does not match the actual initial joints."
                )
            if inactive and not torch.equal(
                positions[:, inactive], positions[0, inactive].expand(length, -1)
            ):
                raise ValueError(
                    "Candidates cannot change inactive or mimic joint coordinates."
                )
            if bool(
                (
                    (positions < limits[index, :, 0])
                    | (positions > limits[index, :, 1])
                ).any()
            ):
                raise ValueError(
                    "Candidate joint targets exceed actual controller limits."
                )
            bytes_per_obs = sum(
                value[index].numel() * value.element_size()
                for value in initial.values()
            )
            size = (
                length * (bytes_per_obs + 8)
                + (length - 1)
                * initial["joint_positions"][index].numel()
                * initial["joint_positions"].element_size()
                + _METADATA_RESERVE
            )
            if size > self.max_episode_bytes:
                raise ValueError(
                    "Rollout would exceed max_episode_bytes before its first command."
                )
            if (
                _metadata_bytes(
                    (
                        identity,
                        episode_ids[identity.candidate_id],
                        candidate.phases,
                        tuple(initial),
                    )
                )
                > _EXECUTOR_METADATA_LIMIT
            ):
                raise ValueError(
                    "Candidate lineage or observation keys exceed the metadata budget."
                )
            rows.append(
                _Row(
                    candidate,
                    length - 1,
                    [{key: value[index].clone() for key, value in initial.items()}],
                )
            )
        return rows

    def _hold(self, joints: tuple[int, ...]) -> None:
        robot = self.host.adapter.robot
        measured = robot.get_qpos()[:, joints]
        if not bool(torch.isfinite(measured).all()):
            raise RuntimeError("Cannot safely hold nonfinite measured joint positions.")
        robot.set_qpos(measured, joint_ids=joints, target=True)
        robot.set_qvel(torch.zeros_like(measured), joint_ids=joints, target=True)
        robot.set_qf(torch.zeros_like(measured), joint_ids=joints)

    def _clear_hold_dynamics(
        self, joints: tuple[int, ...], active: tuple[bool, ...]
    ) -> None:
        if all(active):
            return
        robot = self.host.adapter.robot
        velocity = robot.get_qvel(target=True)[:, joints].clone()
        effort = robot.get_qf()[:, joints].clone()
        inactive = torch.tensor([not value for value in active], device=velocity.device)
        velocity[inactive] = 0
        effort[inactive] = 0
        robot.set_qvel(velocity, joint_ids=joints, target=True)
        robot.set_qf(effort, joint_ids=joints)

    def execute(
        self,
        binding: PreparedBatch,
        candidates: Sequence[CandidateTrajectoryBatch | None],
        episode_ids: Mapping[str, tuple[str, str]],
        *,
        on_started: Callable[[CandidateIdentity], None],
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[ExpertEpisode | None, ...]:
        """Execute the full batch and freeze each row at its final observed step.

        Invalid inputs raise before commands. Runtime failures yield rejected
        episodes for rows with complete transitions and ``None`` otherwise.
        ``on_started`` fires after the first submitted command, even when its
        subsequent physics or observation fails. Inability to install the final
        safe hold raises and must stop the caller's collection job.
        """
        if self._running:
            raise RuntimeError("QposRolloutExecutor requires serialized execution.")
        self._last_failures = MappingProxyType({})
        if not callable(on_started) or (
            should_stop is not None and not callable(should_stop)
        ):
            raise TypeError("Execution callbacks must be callable.")
        self.host.assert_current(binding)
        initial = self._observations(self.host.initial_observation(binding))
        joints = self._active_joints()
        rows = self._preflight(binding, candidates, episode_ids, initial, joints)
        if not any(row is not None for row in rows):
            return tuple(None for _ in rows)
        schema = {key: (value.shape, value.dtype) for key, value in initial.items()}
        robot, sim, env = self.host.adapter.robot, self.host.adapter.sim, self.host.env
        snapshots = self.host.snapshots(binding)
        if self.contact_validator is not None:
            self.contact_validator.begin_rollout(candidates, snapshots)
        initial_time = float(sim.simulation_time)
        if not math.isfinite(initial_time):
            raise ValueError("The simulator clock must be finite.")
        last_time = initial_time
        pending: torch.Tensor | None = None
        current_mask = tuple(False for _ in rows)
        expected_command: torch.Tensor | None = None
        self._running = True
        stop_requested = False

        def stop() -> bool:
            nonlocal stop_requested
            requested = should_stop is not None and bool(should_stop())
            stop_requested |= requested
            return requested

        def command_submitted() -> None:
            nonlocal pending, current_mask
            if env is not None:
                current_mask = tuple(bool(value) for value in env._demo_active_mask)
            callback_error = None
            for index, row in enumerate(rows):
                if row is not None and current_mask[index] and not row.started:
                    row.started = True
                    try:
                        on_started(row.candidate.identities[0])
                    except Exception as error:
                        callback_error = error
            pending = robot.get_qpos(target=True).detach().cpu().clone()
            if (
                pending.dtype != initial["joint_positions"].dtype
                or pending.shape != initial["joint_positions"].shape
                or not bool(torch.isfinite(pending).all())
            ):
                raise ValueError(
                    "Actual controller targets have invalid shape or values."
                )
            for index, row in enumerate(rows):
                if (
                    row is not None
                    and current_mask[index]
                    and not torch.allclose(
                        pending[index, joints],
                        expected_command[index, joints].cpu().to(pending.dtype),
                        atol=self.host.adapter.atol,
                        rtol=0,
                    )
                ):
                    row.failure = "actual controller targets differ from the candidate"
            if callback_error is not None:
                raise callback_error

        def transition(raw: Any = None, mask: tuple[bool, ...] | None = None) -> None:
            nonlocal pending, last_time
            self.host.assert_current(binding)
            observed = self._observations(raw, schema)
            now = float(sim.simulation_time)
            if not math.isfinite(now) or not math.isclose(
                now - last_time, self.control_dt, rel_tol=0, abs_tol=1e-8
            ):
                raise ValueError(
                    "Actual simulator time does not match the control clock."
                )
            last_time = now
            if pending is None:
                raise RuntimeError(
                    "A transition has no observed controller submission."
                )
            active = current_mask if mask is None else mask
            root_pose = robot.get_local_pose(to_matrix=True).detach().cpu()
            entity_poses = {
                uid: sim.get_rigid_object(uid)
                .get_local_pose(to_matrix=True)
                .detach()
                .cpu()
                for uid in sim.get_rigid_object_uid_list()
            }
            for index, row in enumerate(rows):
                if row is not None:
                    snapshot = snapshots[index]
                    row.fixed_world &= (
                        set(entity_poses) == set(snapshot.entity_poses)
                        and torch.allclose(
                            root_pose[index],
                            snapshot.root_pose.cpu(),
                            atol=self.host.adapter.atol,
                            rtol=0,
                        )
                        and all(
                            torch.allclose(
                                pose[index],
                                snapshot.entity_poses[uid].cpu(),
                                atol=self.host.adapter.atol,
                                rtol=0,
                            )
                            for uid, pose in entity_poses.items()
                            if self.contact_validator is None
                            or uid not in self.contact_validator.dynamic_entity_ids
                        )
                    )
                if row is not None and active[index] and len(row.actions) < row.steps:
                    inactive = [
                        joint
                        for joint in range(len(robot.joint_names))
                        if joint not in joints
                        and (
                            self.contact_validator is None
                            or joint not in self.contact_validator.mimic_ids
                        )
                    ]
                    if inactive and (
                        not torch.allclose(
                            observed["joint_positions"][index, inactive],
                            initial["joint_positions"][index, inactive],
                            atol=self.host.adapter.atol,
                            rtol=0,
                        )
                        or not torch.allclose(
                            pending[index, inactive],
                            initial["joint_positions"][index, inactive].to(pending),
                            atol=self.host.adapter.atol,
                            rtol=0,
                        )
                    ):
                        row.failure = (
                            "inactive or mimic joints changed during execution"
                        )
                    row.actions.append(pending[index].clone())
                    row.timestamps.append(now - initial_time)
                    row.observations.append(
                        {key: value[index].clone() for key, value in observed.items()}
                    )
            pending = None

        def actions():
            nonlocal expected_command, current_mask
            for index in range(max(row.steps for row in rows if row is not None)):
                self.host.assert_current(binding)
                expected_command = robot.get_qpos().detach().clone()
                current_mask = tuple(
                    row is not None and index < row.steps for row in rows
                )
                hold_mask = (
                    current_mask
                    if env is None
                    else tuple(
                        active and bool(env._demo_active_mask[slot])
                        for slot, active in enumerate(current_mask)
                    )
                )
                self._clear_hold_dynamics(joints, hold_mask)
                for slot, row in enumerate(rows):
                    if row is not None and current_mask[slot]:
                        expected_command[slot] = row.candidate.positions[
                            0, index + 1
                        ].to(expected_command)
                yield expected_command.clone()

        try:
            if env is not None:
                from embodichain.lab.gym.envs.demo import (
                    DemoSegment,
                    execute_demo_episode,
                )
                from embodichain.lab.gym.envs.types import ControllerAction

                def observe_step(result, mask) -> None:
                    transition(result[0], mask)
                    terminated, truncated, info = result[2], result[3], result[4]
                    for index, row in enumerate(rows):
                        if row is None or not mask[index]:
                            continue
                        if bool(truncated[index]) or bool(
                            info.get("fail", torch.zeros(len(rows), dtype=torch.bool))[
                                index
                            ]
                        ):
                            row.failure = "environment failed or truncated"
                        elif bool(terminated[index]) and len(row.actions) < row.steps:
                            row.failure = (
                                "environment terminated before the trajectory ended"
                            )

                with env.observe_generation_commands(self.host, command_submitted):
                    execute_demo_episode(
                        env,
                        segments=DemoSegment(
                            actions=(ControllerAction(value) for value in actions()),
                            name="qpos_candidate",
                            failure_policy="row_independent",
                        ),
                        row_step_limits=tuple(
                            row.steps if row is not None else 0 for row in rows
                        ),
                        step_observer=observe_step,
                        should_stop=stop,
                    )
            else:
                for command in actions():
                    if stop():
                        break
                    robot.set_qpos(command[:, joints], joint_ids=joints, target=True)
                    command_submitted()
                    if self.contact_validator is None:
                        sim.update(self.host.profile.physics_dt, self._substeps)
                    else:
                        sample_index = (
                            max(len(row.actions) for row in rows if row is not None) + 1
                        )
                        for _ in range(self._substeps):
                            sim.update(self.host.profile.physics_dt, 1)
                            self.contact_validator.observe_substep(
                                sample_index,
                                current_mask,
                                physics_dt=self.host.profile.physics_dt,
                            )
                    transition()
        except Exception as error:
            for row in rows:
                if row is not None:
                    row.failure = _failure_text(error)
        finally:
            try:
                self._hold(joints)
            except Exception as error:
                self._last_failures = MappingProxyType(
                    {
                        row.candidate.identities[
                            0
                        ].candidate_id: f"Safe hold failed: {_failure_text(error)}"[
                            :512
                        ]
                        for row in rows
                        if row is not None
                    }
                )
                self._running = False
                raise

        for row in rows:
            if row is not None and row.failure is None and len(row.actions) < row.steps:
                row.failure = (
                    "execution cancelled by should_stop"
                    if stop_requested
                    else "trajectory interrupted before completion"
                )
        self._last_failures = MappingProxyType(
            {
                row.candidate.identities[0].candidate_id: row.failure[:512]
                for row in rows
                if row is not None and row.failure is not None
            }
        )
        try:
            episodes = self._freeze(rows, episode_ids)
            failures = dict(self._last_failures)
            for episode in episodes:
                if episode is not None and not episode.validation.accepted:
                    failure = next(
                        check
                        for check in episode.validation.checks
                        if check.status != "passed"
                    )
                    failures[episode.identity.candidate_id] = (
                        failure.detail or failure.check_id
                    )[:512]
            self._last_failures = MappingProxyType(failures)
            return episodes
        finally:
            self._running = False

    def _freeze(
        self,
        rows: list[_Row | None],
        episode_ids: Mapping[str, tuple[str, str]],
    ) -> tuple[ExpertEpisode | None, ...]:
        episodes = []
        for row_index, row in enumerate(rows):
            if row is None or not row.actions:
                episodes.append(None)
                continue
            observations = {
                key: torch.stack([sample[key] for sample in row.observations])
                for key in row.observations[0]
            }
            commands = torch.stack(row.actions)
            timestamps = torch.tensor(row.timestamps, dtype=torch.float64)
            complete = len(commands) == row.steps and row.failure is None
            checks = [
                ValidationCheck(
                    "execution_complete",
                    "passed" if complete else "failed",
                    row.failure
                    or ("" if complete else "trajectory interrupted before completion"),
                ),
                ValidationCheck(
                    "fixed_collision_world",
                    "passed" if row.fixed_world else "failed",
                    (
                        ""
                        if row.fixed_world
                        else "Robot root or obstacle poses changed during execution."
                    ),
                ),
            ]
            if self.contact_validator is not None:
                checks.extend(
                    self.contact_validator.rollout_validation(row_index).checks
                )
            try:
                validation = self.validator(
                    row.candidate,
                    {key: value.clone() for key, value in observations.items()},
                    commands.clone(),
                    timestamps.clone(),
                )
                if not isinstance(validation, ValidationResult):
                    raise TypeError("Task validator must return ValidationResult.")
                if not any(
                    check.check_id == self.validator_id for check in validation.checks
                ):
                    raise ValueError(
                        f"Task validator omitted required check {self.validator_id!r}."
                    )
                if any(
                    check.check_id in {"execution_complete", "fixed_collision_world"}
                    for check in validation.checks
                ):
                    raise ValueError("Task validator cannot replace execution checks.")
                checks.extend(validation.checks)
            except Exception as error:
                checks.append(
                    ValidationCheck(self.validator_id, "failed", str(error)[:512])
                )
            episode_id, commit_id = episode_ids[
                row.candidate.identities[0].candidate_id
            ]
            phases = tuple(
                replace(phase, stop_index=min(phase.stop_index, len(commands) + 1))
                for phase in row.candidate.phases[0]
                if phase.start_index < len(commands) + 1
            )
            metadata = {
                "validation_profile_id": self.validation_profile_id,
                "timing_source": "simulation_time",
                "phase_annotations": "planned_sample_intervals",
                "joint_names": row.candidate.joint_names,
                "commanded_joint_indices": self._active_joints(),
            }
            if self.contact_validator is not None:
                metadata["contact_profile"] = self.contact_validator.profile.to_dict()
                metadata["collision_geometry"] = (
                    "URDF collision convex hulls; cuboid world; implicit ground"
                )
            if (
                _metadata_bytes(
                    (
                        row.candidate.identities[0],
                        episode_id,
                        commit_id,
                        "qpos",
                        tuple(observations),
                        ValidationResult(tuple(checks)),
                        phases,
                        metadata,
                    )
                )
                > _EXECUTOR_METADATA_LIMIT
            ):
                checks = [
                    ValidationCheck(
                        "execution_complete",
                        "failed",
                        "Execution validation metadata exceeds its reserved budget.",
                    )
                ]
            episode = ExpertEpisode(
                identity=row.candidate.identities[0],
                observations=observations,
                actions=commands,
                timestamps=timestamps,
                action_representation="qpos",
                validation=ValidationResult(tuple(checks)),
                episode_id=episode_id,
                commit_id=commit_id,
                phases=phases,
                metadata=metadata,
            )
            episodes.append(episode)
        return tuple(episodes)
