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

"""Exclusive full-batch preparation for fixed-scene trajectory collection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import MISSING, dataclass
import math
from typing import TYPE_CHECKING
from uuid import uuid4

from embodichain.lab.sim.motion.expansion import (
    MotionSnapshot,
    SceneCase,
    ValidationCheck,
    ValidationResult,
)
from embodichain.utils import configclass

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.sim.types import EnvObs
    from .integrations.sim import SimInitialState, SimInitialStateAdapter

__all__ = ["InitialStateProfile", "PreparedBatch", "FixedSceneHost"]


@configclass
class InitialStateProfile:
    """Trusted preparation callbacks for one task and controller configuration.

    ``prepare`` must reset all task/controller state not owned by the standard
    Gym managers without randomizing the scene. ``signature`` must identify the
    fixed physical, visual, sensor, and control conditions, excluding evolving
    poses and joint positions. ``verify`` checks the provided task initial state
    against every declared case after settling. These callables are supplied
    by trusted Python integration code, never imported from job YAML strings.

    Args:
        profile_id: Stable preparation profile identifier.
        prepare: Deterministic task/controller preparation callback.
        signature: Callback returning a nonempty fixed-condition signature.
        verify: Task-specific initial-state checks for the complete batch.
        physics_dt: Physics step duration in seconds; must match Gym when used.
        settling_steps: Number of preparation physics steps, outside rollouts.
        allowed_interval_events: Gym interval event names explicitly certified
            to preserve the fixed conditions by this profile.
    """

    profile_id: str = MISSING
    prepare: Callable[[], None] = MISSING
    signature: Callable[[], str] = MISSING
    verify: Callable[[tuple[SceneCase, ...]], ValidationResult] = MISSING
    physics_dt: float = 0.01
    settling_steps: int = 0
    allowed_interval_events: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be a nonempty string")
        for name in ("prepare", "signature", "verify"):
            if not callable(getattr(self, name)):
                raise ValueError(f"{name} must be a trusted callback")
        if (
            isinstance(self.physics_dt, bool)
            or not isinstance(self.physics_dt, (int, float))
            or not math.isfinite(self.physics_dt)
            or self.physics_dt <= 0
        ):
            raise ValueError("physics_dt must be positive and finite")
        if type(self.settling_steps) is not int or self.settling_steps < 0:
            raise ValueError("settling_steps must be a nonnegative integer")
        if isinstance(self.allowed_interval_events, str):
            raise ValueError("allowed_interval_events must be a sequence of names")
        self.allowed_interval_events = tuple(self.allowed_interval_events)
        if any(
            not isinstance(name, str) or not name.strip()
            for name in self.allowed_interval_events
        ) or len(set(self.allowed_interval_events)) != len(
            self.allowed_interval_events
        ):
            raise ValueError("allowed_interval_events must contain unique names")


@dataclass(frozen=True)
class PreparedBatch:
    """Identity of a successfully prepared full batch.

    Cases are ordered by physical slot. A token is valid only for its owning
    host and epoch; it does not make an old ActionPlan or runtime reusable.
    """

    host_id: str
    epoch: int
    cases: tuple[SceneCase, ...]


class FixedSceneHost:
    """Own preparation of one complete simulator batch, optionally through Gym.

    Capture consumes a caller-provided scene. Restoring discards pending Gym
    recordings, so the caller must freeze old episode evidence first. Both
    paths run deterministic preparation, settling, and verification before
    publishing a new binding. The owner must remain the sole simulation
    stepper for the lifetime of this object.

    Args:
        adapter: Physical state adapter for the complete simulator batch.
        profile: Trusted task, controller, event, and fixed-condition policy.
        env: Optional unwrapped Gym host using the same simulator and robot.

    Raises:
        ValueError: If the host or preparation policy is incompatible.
        RuntimeError: If another generation host already owns the batch.
    """

    def __init__(
        self,
        adapter: SimInitialStateAdapter,
        profile: InitialStateProfile,
        *,
        env: EmbodiedEnv | None = None,
    ) -> None:
        self.adapter = adapter
        self.profile = profile.copy()
        self.env = env
        self._host_id = uuid4().hex
        self._epoch = 0
        self._binding: PreparedBatch | None = None
        self._initial_state: SimInitialState | None = None
        self._cases: tuple[SceneCase, ...] = ()
        self._condition_signature: str | None = None
        self._prepared_observation: EnvObs | None = None
        self._closed = False
        if env is not None:
            if env.sim is not adapter.sim or env.robot is not adapter.robot:
                raise ValueError("Gym and physical adapter must own the same sim/robot")
            if env.num_envs != adapter.sim.num_envs:
                raise ValueError("generation requires the complete simulator batch")
            if not math.isclose(env.physics_dt, profile.physics_dt, rel_tol=1e-9):
                raise ValueError("profile physics_dt must match the Gym physics clock")
            self._check_interval_events()
        if getattr(adapter.sim, "_trajectory_generation_owner", None) is not None:
            raise RuntimeError("simulator batch already has a generation owner")
        self._structure_signature = adapter.signature()
        if env is not None:
            env.acquire_generation_lease(self)
        adapter.sim._trajectory_generation_owner = self

    def acquire_case(self, cases: Sequence[SceneCase]) -> PreparedBatch:
        """Prepare and capture all caller-provided initial states.

        Args:
            cases: One immutable scene case per simulator row.

        Returns:
            A binding published only after preparation and validation succeed.

        Raises:
            ValueError: If cases do not cover the full batch.
            RuntimeError: If this host already acquired its fixed cases.
        """
        self._assert_owner()
        if self._initial_state is not None:
            raise RuntimeError("cases already acquired; use restore_initial")
        values = tuple(cases)
        if len(values) != self.adapter.sim.num_envs or any(
            not isinstance(case, SceneCase) for case in values
        ):
            raise ValueError("provide one SceneCase for every simulator row")
        self._cases = values
        self._condition_signature = self._signature()
        return self._prepare(capture=True)

    def restore_initial(self) -> PreparedBatch:
        """Restore the owned initial batch and invalidate every earlier binding.

        Returns:
            The new successfully prepared binding.

        Raises:
            RuntimeError: If no initial batch exists or fixed conditions drifted.
        """
        self._assert_owner()
        if self._initial_state is None:
            raise RuntimeError("acquire_case must succeed before restore_initial")
        return self._prepare(capture=False)

    def assert_current(self, batch: PreparedBatch) -> None:
        """Reject a foreign or obsolete execution binding before issuing commands.

        Args:
            batch: The binding attached to the candidate execution.

        Raises:
            RuntimeError: If ownership, epoch, or fixed conditions changed.
        """
        self._assert_owner()
        if batch is not self._binding or batch.epoch != self._current_epoch():
            raise RuntimeError("execution binding belongs to an obsolete host epoch")
        self._check_conditions()

    def verify_initial(self, batch: PreparedBatch) -> ValidationResult:
        """Recheck physical and task initial state for the current binding.

        Args:
            batch: The current prepared binding, before rollout begins.

        Returns:
            Physical restoration and task-specific verification checks.
        """
        self.assert_current(batch)
        return self._verify()

    def snapshots(self, batch: PreparedBatch) -> tuple[MotionSnapshot, ...]:
        """Export owned planning inputs from the captured initial batch.

        Args:
            batch: Current preparation binding for the declared cases.

        Returns:
            One full-joint snapshot per physical row, in local arena coordinates.
        """
        self.assert_current(batch)
        state = self._initial_state
        return tuple(
            MotionSnapshot(
                scene_case=case,
                joint_names=state.joint_names,
                joint_positions=state.robot["qpos"][row],
                joint_velocities=state.robot["qvel"][row],
                root_pose=state.robot["root_pose"][row],
                entity_poses={
                    uid: fields["pose"][row]
                    for uid, fields in state.rigid_objects.items()
                },
            )
            for row, case in enumerate(batch.cases)
        )

    def initial_observation(self, batch: PreparedBatch) -> EnvObs | None:
        """Copy the Gym observation published by preparation without querying again.

        Args:
            batch: Current preparation binding.

        Returns:
            The owned Gym first frame, or ``None`` for a pure simulator host.
        """
        self.assert_current(batch)
        return (
            None
            if self._prepared_observation is None
            else self._prepared_observation.clone()
        )

    def close(self) -> None:
        """Release exclusive ownership and invalidate all execution bindings."""
        if self._closed:
            return
        self._assert_owner()
        if self.env is not None:
            self.env.release_generation_lease(self)
        self.adapter.sim._trajectory_generation_owner = None
        self._binding = None
        self._prepared_observation = None
        self._epoch += 1
        self._closed = True

    def __enter__(self) -> FixedSceneHost:
        self._assert_owner()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _assert_owner(self) -> None:
        if (
            self._closed
            or getattr(self.adapter.sim, "_trajectory_generation_owner", None)
            is not self
        ):
            raise RuntimeError("generation host does not own the simulator batch")

    def _signature(self) -> str:
        value = self.profile.signature()
        if not isinstance(value, str) or not value.strip():
            raise ValueError("fixed-condition signature must be a nonempty string")
        return value

    def _check_interval_events(self) -> None:
        manager = getattr(self.env, "event_manager", None)
        active = manager.active_functors if manager is not None else {}
        unknown = set(active.get("interval", ())) - set(
            self.profile.allowed_interval_events
        )
        if unknown:
            raise ValueError(f"uncertified interval events: {sorted(unknown)}")

    def _check_conditions(self) -> None:
        self._check_interval_events()
        if self.adapter.signature() != self._structure_signature:
            raise RuntimeError("simulator topology or controller identity changed")
        if self._signature() != self._condition_signature:
            raise RuntimeError("fixed scene conditions changed after case acquisition")

    def _current_epoch(self) -> int:
        return self.env.generation_epoch if self.env is not None else self._epoch

    def _settle(self) -> None:
        if self.profile.settling_steps:
            self.adapter.sim.update(
                self.profile.physics_dt, self.profile.settling_steps
            )

    def _verify(self) -> ValidationResult:
        self._check_conditions()
        task_result = self.profile.verify(self._cases)
        if not isinstance(task_result, ValidationResult):
            raise TypeError("profile verify must return ValidationResult")
        if not task_result.checks:
            task_result = ValidationResult(
                (
                    ValidationCheck(
                        "initial_conditions", "failed", "profile rejected initial state"
                    ),
                )
            )
        physical = (
            self.adapter.verify(self._initial_state)
            if self._initial_state is not None
            else ValidationResult(())
        )
        return ValidationResult((*physical.checks, *task_result.checks))

    def _prepare(self, *, capture: bool) -> PreparedBatch:
        self._binding = None
        self._prepared_observation = None
        self._epoch += 1

        def prepare() -> None:
            self._check_conditions()
            self.profile.prepare()

        def restore() -> None:
            if not capture:
                self.adapter.restore(self._initial_state)

        def verify() -> ValidationResult:
            result = self._verify()
            if result.accepted and capture:
                state = self.adapter.capture()
                physical = self.adapter.verify(state)
                result = ValidationResult((*physical.checks, *result.checks))
                if result.accepted:
                    self._initial_state = state
            return result

        try:
            if self.env is not None:
                observation, _ = self.env.prepare_generation_episode(
                    self,
                    prepare=prepare,
                    restore=restore,
                    settle=self._settle,
                    verify=verify,
                )
                self._prepared_observation = observation.clone()
            else:
                prepare()
                restore()
                self._settle()
                result = verify()
                if not result.accepted:
                    raise RuntimeError(
                        f"initial state verification failed: {result.checks}"
                    )
        except Exception:
            if capture:
                self._initial_state = None
            raise
        self._binding = PreparedBatch(self._host_id, self._current_epoch(), self._cases)
        return self._binding
