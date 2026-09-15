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

"""Simulation-thread session that compiles and executes authored sequences.

The session owns an ordered list of :class:`SkillCard` values, converts the
configured cards into atomic-action invocations, compiles them through an
:class:`~embodichain.lab.sim.atomic_actions.AtomicActionEngine`, and replays
the compiled trajectory on the simulation thread. It never touches viser;
browser-facing layers observe it exclusively through immutable
:class:`SequenceSnapshot` values and drive it through
:meth:`AuthoringSession.handle_command`.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Iterator, Mapping, Sequence

import torch

from .protocol import (
    AddCard,
    AuthoringCommand,
    CompileSequence,
    ExecuteSequence,
    ExecutionProgress,
    MoveCard,
    RemoveCard,
    SequenceSnapshot,
    SkillCard,
    SkillCardState,
    SUPPORTED_SKILL_IDS,
    UpdateCard,
    is_finite_number,
)

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.atomic_actions import (
        ActionInvocation,
        ActionPlan,
        AtomicActionEngine,
        CompiledTrajectory,
        SceneSnapshot,
        TimedTrajectory,
    )
    from embodichain.lab.sim.objects import RigidObject, Robot

__all__ = ["AuthoringSession"]

_TOP_DOWN_EEF_ROTATION: tuple[tuple[float, float, float], ...] = (
    (-0.0539, -0.9985, -0.0022),
    (-0.9977, 0.0540, -0.0401),
    (0.0401, 0.0000, -0.9992),
)
_DEFAULT_APPROACH_DIRECTION = (0.0, 0.0, -1.0)
_DEFAULT_SAMPLE_COUNTS: Mapping[str, int] = {
    "move_end_effector": 80,
    "pick_up": 120,
    "place": 120,
}
_DEFAULT_PRE_GRASP_DISTANCE = 0.15
_DEFAULT_PICK_LIFT_HEIGHT = 0.16
_DEFAULT_PLACE_LIFT_HEIGHT = 0.14
_DEFAULT_HAND_INTERP_STEPS = 12
_HOLD_SIM_STEPS = 2
_ALLOWED_PARAM_KEYS: Mapping[str, frozenset[str]] = {
    "move_end_effector": frozenset({"position", "rotation", "sample_count"}),
    "pick_up": frozenset(
        {
            "approach_direction",
            "pre_grasp_distance",
            "lift_height",
            "hand_interp_steps",
            "sample_count",
        }
    ),
    "place": frozenset(
        {
            "position",
            "rotation",
            "lift_height",
            "hand_interp_steps",
            "sample_count",
        }
    ),
}


def _validate_vector3(value: object, name: str) -> tuple[float, float, float]:
    """Validate and convert a three-component real vector parameter."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of three numbers.")
    items = tuple(value)
    if len(items) != 3 or not all(is_finite_number(item) for item in items):
        raise ValueError(f"{name} must contain exactly three finite numbers.")
    return tuple(float(item) for item in items)


def _validate_rotation(value: object, name: str) -> tuple[tuple[float, ...], ...]:
    """Validate and convert a 3x3 rotation-matrix parameter."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a 3x3 nested sequence of numbers.")
    rows = tuple(value)
    if len(rows) != 3:
        raise ValueError(f"{name} must contain exactly three rows.")
    return tuple(_validate_vector3(row, f"{name} row") for row in rows)


def _validate_positive_number(value: object, name: str) -> float:
    """Validate a strictly positive finite scalar parameter."""
    if not is_finite_number(value) or float(value) <= 0.0:
        raise ValueError(f"{name} must be a finite number greater than zero.")
    return float(value)


def _validate_positive_int(value: object, name: str) -> int:
    """Validate a strictly positive integer parameter."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an integer greater than zero.")
    return int(value)


class AuthoringSession:
    """Skill-sequence authoring state machine on the simulation thread.

    Args:
        robot: Robot receiving full-DOF trajectory positions during execution.
        engine: Atomic-action engine that owns planning resources. It must be
            constructed for ``robot`` and, when pick cards are used, provide a
            grasp-pose generator and control profile for the grasp part.
        sim: Simulation manager stepped during execution and used to resolve
            card target entities by UID.
        control_parts: Endpoint-role to robot control-part mapping. The
            ``"motion"`` role is required; the ``"grasp"`` role is required
            once a ``pick_up`` or ``place`` card is compiled.
    """

    def __init__(
        self,
        robot: Robot,
        engine: AtomicActionEngine,
        sim: SimulationManager,
        control_parts: Mapping[str, str],
    ) -> None:
        if not isinstance(control_parts, Mapping):
            raise TypeError("control_parts must be a mapping of role to part name.")
        parts = {str(role): str(part) for role, part in control_parts.items()}
        if not parts.get("motion"):
            raise ValueError("control_parts must define a non-empty 'motion' part.")
        self._robot = robot
        self._engine = engine
        self._sim = sim
        self._control_parts = parts
        self._cards: list[SkillCard] = []
        self._card_counter = 0
        self._compiled: CompiledTrajectory | None = None

    # ------------------------------------------------------------------
    # Sequence management
    # ------------------------------------------------------------------

    @property
    def robot(self) -> Robot:
        """Robot the session commands, exposed for read-only preview layers."""
        return self._robot

    @property
    def cards(self) -> tuple[SkillCard, ...]:
        """Current ordered card snapshots."""
        return tuple(self._cards)

    @property
    def is_compiled(self) -> bool:
        """Whether the sequence holds a fully successful compilation."""
        return self._compiled is not None

    @property
    def compiled_trajectory(self) -> CompiledTrajectory | None:
        """Last fully successful compilation result, if any."""
        return self._compiled

    def add_card(
        self,
        skill_id: str,
        *,
        card_id: str | None = None,
        entity_uid: str | None = None,
        params: Mapping[str, object] | None = None,
        index: int | None = None,
    ) -> SkillCard:
        """Insert one new card and invalidate any previous compilation.

        Args:
            skill_id: One of :data:`SUPPORTED_SKILL_IDS`.
            card_id: Optional explicit unique card identifier.
            entity_uid: Optional target scene-entity UID.
            params: Optional skill parameters; keys are validated per skill.
            index: Optional insertion position; appends when omitted.

        Returns:
            The stored card snapshot.

        Raises:
            ValueError: If the skill is unsupported, the card ID is already
                used, or a parameter is invalid.
            IndexError: If ``index`` is outside ``[0, len(cards)]``.
        """
        if skill_id not in SUPPORTED_SKILL_IDS:
            raise ValueError(
                f"Unsupported skill {skill_id!r}; expected one of "
                f"{SUPPORTED_SKILL_IDS}."
            )
        merged = dict(params or {})
        self._validate_params(skill_id, merged)
        if card_id is None:
            card_id = self._next_card_id()
        elif any(card.card_id == card_id for card in self._cards):
            raise ValueError(f"Card ID {card_id!r} is already in the sequence.")
        if index is None:
            index = len(self._cards)
        elif index < 0 or index > len(self._cards):
            raise IndexError(
                f"index {index} is outside the sequence of {len(self._cards)} cards."
            )
        self._invalidate()
        card = SkillCard(
            card_id=card_id,
            skill_id=skill_id,
            entity_uid=entity_uid,
            params=merged,
            state=self._configured_state(skill_id, entity_uid, merged),
        )
        self._cards.insert(index, card)
        return card

    def remove_card(self, card_id: str) -> None:
        """Remove one card and invalidate any previous compilation."""
        index = self._card_index(card_id)
        self._invalidate()
        del self._cards[index]

    def move_card(self, card_id: str, new_index: int) -> None:
        """Move one card to ``new_index`` and invalidate the compilation."""
        index = self._card_index(card_id)
        if new_index < 0 or new_index >= len(self._cards):
            raise IndexError(
                f"new_index {new_index} is outside the sequence of "
                f"{len(self._cards)} cards."
            )
        self._invalidate()
        card = self._cards.pop(index)
        self._cards.insert(new_index, card)

    def update_card(
        self,
        card_id: str,
        *,
        params: Mapping[str, object] | None = None,
        entity_uid: str | None = None,
        clear_entity_uid: bool = False,
    ) -> SkillCard:
        """Merge parameter updates into one card and refresh its state.

        Args:
            card_id: Identifier of the card to update.
            params: Parameter entries merged over the existing parameters.
            entity_uid: New target entity UID; unchanged when ``None``.
            clear_entity_uid: Remove the target entity UID.

        Returns:
            The updated card snapshot.
        """
        if entity_uid is not None and clear_entity_uid:
            raise ValueError("Cannot both set and clear the entity UID.")
        index = self._card_index(card_id)
        card = self._cards[index]
        merged = dict(card.params)
        if params is not None:
            merged.update(params)
        self._validate_params(card.skill_id, merged)
        new_entity_uid = card.entity_uid
        if clear_entity_uid:
            new_entity_uid = None
        elif entity_uid is not None:
            new_entity_uid = entity_uid
        self._invalidate()
        updated = SkillCard(
            card_id=card.card_id,
            skill_id=card.skill_id,
            entity_uid=new_entity_uid,
            params=merged,
            state=self._configured_state(card.skill_id, new_entity_uid, merged),
        )
        self._cards[index] = updated
        return updated

    # ------------------------------------------------------------------
    # Compilation and preview
    # ------------------------------------------------------------------

    def compile(self) -> bool:
        """Compile every card into one trajectory and back-fill segments.

        Cards are converted in order into grounded
        :class:`~embodichain.lab.sim.atomic_actions.ActionInvocation` values
        and compiled through the engine. On success each card keeps its
        ``READY`` state and receives its half-open waypoint range inside the
        compiled trajectory. A card whose plan failed becomes ``FAILED`` with
        the planner's diagnostic text; cards after the failure remain
        ``READY`` without segment metadata.

        Returns:
            Whether the whole sequence compiled successfully.

        Raises:
            ValueError: If the sequence is empty or contains an
                ``UNCONFIGURED`` card.
        """
        if not self._cards:
            raise ValueError("Cannot compile an empty skill sequence.")
        self._invalidate()
        unconfigured = [
            card.card_id
            for card in self._cards
            if card.state is SkillCardState.UNCONFIGURED
        ]
        if unconfigured:
            raise ValueError(
                f"Cards {unconfigured} are not fully configured for compilation."
            )

        invocations = tuple(self._build_invocation(card) for card in self._cards)
        context = self._engine.initial_context(
            scene=self._build_scene_snapshot(),
            control_dt=float(self._sim.sim_config.physics_dt),
        )
        try:
            compiled = self._engine.compile(invocations, context)
        except Exception as exc:  # noqa: BLE001 - surfaced on the cards.
            message = f"Compilation raised {type(exc).__name__}: {exc}"
            self._cards = [
                replace(card, state=SkillCardState.FAILED, failure_message=message)
                for card in self._cards
            ]
            return False

        cards = list(self._cards)
        for action_index, plan in enumerate(compiled.action_plans):
            card = cards[action_index]
            success = bool(plan.plan_success.all().item())
            waypoint_count = (
                0
                if plan.joint_trajectory is None
                else plan.joint_trajectory.waypoint_count
            )
            if success and waypoint_count > 0:
                offset = compiled.action_waypoint_offset(action_index)
                cards[action_index] = replace(
                    card,
                    state=SkillCardState.READY,
                    segment_start=offset,
                    segment_stop=offset + waypoint_count,
                )
            elif not success:
                cards[action_index] = replace(
                    card,
                    state=SkillCardState.FAILED,
                    failure_message=self._failure_message(plan),
                )
        self._cards = cards

        succeeded = bool(compiled.plan_success.all().item()) and all(
            card.state is SkillCardState.READY for card in cards
        )
        if succeeded:
            self._compiled = compiled
        return succeeded

    @property
    def preview_length(self) -> int:
        """Waypoint count of the compiled trajectory, or zero."""
        if self._compiled is None:
            return 0
        return int(self._compiled.trajectory.waypoint_count)

    def preview_qpos(self, step_index: int) -> torch.Tensor:
        """Return the full-robot joint positions of one compiled waypoint.

        Args:
            step_index: Waypoint index in ``[0, preview_length)``.

        Returns:
            Cloned tensor with shape ``(num_envs, robot_dof)`` for upper
            layers to run preview-model forward kinematics.

        Raises:
            RuntimeError: If the sequence has not been compiled successfully.
            IndexError: If ``step_index`` is out of range.
        """
        if self._compiled is None:
            raise RuntimeError("The sequence has no successful compilation.")
        if step_index < 0 or step_index >= self.preview_length:
            raise IndexError(
                f"step_index {step_index} is outside the compiled trajectory of "
                f"{self.preview_length} waypoints."
            )
        return self._compiled.trajectory.positions[:, step_index, :].clone()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        on_step: Callable[[int, int], None] | None = None,
        *,
        hold_steps: int = 0,
    ) -> bool:
        """Execute the compiled trajectory waypoint by waypoint.

        For every waypoint the robot is commanded with the full-robot joint
        positions and the simulation advances by the number of physics steps
        implied by the trajectory's arrival intervals, mirroring the tutorial
        replay pacing. Card states advance ``READY -> RUNNING -> SUCCEEDED``
        as execution enters and leaves their waypoint ranges.

        Args:
            on_step: Optional callback invoked as ``on_step(step_index,
                total_steps)`` after every simulation update.
            hold_steps: Number of final-pose hold updates after the last
                waypoint.

        Returns:
            Whether the whole sequence executed without raising.

        Raises:
            RuntimeError: If the sequence has not been compiled successfully.
        """
        steps = self._start_execution(hold_steps, on_step)
        while True:
            try:
                next(steps)
            except StopIteration as stop:
                return bool(stop.value)

    def execute_stepwise(
        self,
        on_step: Callable[[int, int], None] | None = None,
        *,
        hold_steps: int = 0,
    ) -> Iterator[ExecutionProgress]:
        """Return an iterator that executes one waypoint per ``next`` call.

        This is the non-blocking counterpart of :meth:`execute`. It performs
        exactly the same work, in the same order, but hands control back to the
        caller after every simulation update so a host loop can keep a browser
        responsive and publish the ``READY -> RUNNING -> SUCCEEDED`` card
        transitions while the trajectory is still running. Draining the
        iterator to exhaustion is equivalent to calling :meth:`execute`.

        The compilation is validated and the card states are reset when this
        method is called, not when the iterator is first advanced, so an
        unusable sequence fails immediately. The iterator never raises from the
        execution itself: a failure marks the owning card ``FAILED`` and stops
        the iteration. Its ``StopIteration`` value carries the same boolean
        :meth:`execute` returns.

        Because every tick steps physics, the host loop must not step the
        simulation itself while the iterator is active.

        Args:
            on_step: Optional callback invoked as ``on_step(step_index,
                total_steps)`` after every trajectory simulation update, before
                the corresponding progress value is produced.
            hold_steps: Number of final-pose hold updates after the last
                waypoint. Hold ticks are reported with ``holding=True``.

        Returns:
            Iterator of :class:`~embodichain.lab.visualization.authoring.protocol.ExecutionProgress`
            values, one per simulation update.

        Raises:
            RuntimeError: If the sequence has not been compiled successfully.
            ValueError: If ``hold_steps`` is negative.
        """
        return self._start_execution(hold_steps, on_step)

    def _start_execution(
        self,
        hold_steps: int,
        on_step: Callable[[int, int], None] | None,
    ) -> Iterator[ExecutionProgress]:
        """Validate the request, reset card states, and build the iterator."""
        if self._compiled is None:
            raise RuntimeError("The sequence has no successful compilation.")
        if hold_steps < 0:
            raise ValueError("hold_steps must be non-negative.")
        self._cards = [
            (
                replace(card, state=SkillCardState.READY, failure_message=None)
                if card.segment_start is not None
                else card
            )
            for card in self._cards
        ]
        return self._execution_steps(self._compiled, int(hold_steps), on_step)

    def _execution_steps(
        self,
        compiled: CompiledTrajectory,
        hold_steps: int,
        on_step: Callable[[int, int], None] | None,
    ) -> Iterator[ExecutionProgress]:
        """Replay one compiled trajectory, yielding after every sim update."""
        trajectory = compiled.trajectory
        positions = trajectory.positions
        total_steps = int(positions.shape[1])
        physics_dt = float(self._sim.sim_config.physics_dt)
        step_index = 0
        try:
            for step_index in range(total_steps):
                self._apply_execution_states(step_index)
                self._robot.set_qpos(positions[:, step_index, :])
                self._sim.update(
                    step=self._waypoint_sim_steps(
                        trajectory, step_index, total_steps, physics_dt
                    )
                )
                if on_step is not None:
                    on_step(step_index, total_steps)
                yield ExecutionProgress(
                    step_index=step_index,
                    total_steps=total_steps,
                )
            final_qpos = positions[:, -1, :]
            for hold_index in range(hold_steps):
                self._robot.set_qpos(final_qpos)
                self._sim.update(step=_HOLD_SIM_STEPS)
                yield ExecutionProgress(
                    step_index=hold_index,
                    total_steps=hold_steps,
                    holding=True,
                )
        except Exception as exc:  # noqa: BLE001 - surfaced on the cards.
            self._mark_execution_failure(step_index, exc)
            return False
        self._cards = [
            (
                replace(card, state=SkillCardState.SUCCEEDED)
                if card.segment_start is not None
                else card
            )
            for card in self._cards
        ]
        return True

    # ------------------------------------------------------------------
    # Snapshots and command dispatch
    # ------------------------------------------------------------------

    def snapshot(self) -> SequenceSnapshot:
        """Return the immutable sequence state for browser-facing layers."""
        return SequenceSnapshot(
            cards=tuple(self._cards),
            compiled=self._compiled is not None,
            trajectory_waypoint_count=self.preview_length,
        )

    def handle_command(self, command: AuthoringCommand) -> SequenceSnapshot:
        """Apply one UI command and return the resulting snapshot.

        This is the single entry point reserved for browser-facing layers.

        Args:
            command: One protocol command value.

        Returns:
            Sequence snapshot taken after the command was applied.

        Raises:
            TypeError: If ``command`` is not a known authoring command.
        """
        if isinstance(command, AddCard):
            self.add_card(
                command.skill_id,
                card_id=command.card_id,
                entity_uid=command.entity_uid,
                params=command.params,
                index=command.index,
            )
        elif isinstance(command, RemoveCard):
            self.remove_card(command.card_id)
        elif isinstance(command, MoveCard):
            self.move_card(command.card_id, command.new_index)
        elif isinstance(command, UpdateCard):
            self.update_card(
                command.card_id,
                params=command.params,
                entity_uid=command.entity_uid,
                clear_entity_uid=command.clear_entity_uid,
            )
        elif isinstance(command, CompileSequence):
            self.compile()
        elif isinstance(command, ExecuteSequence):
            self.execute(hold_steps=command.hold_steps)
        else:
            raise TypeError(f"Unknown authoring command {type(command).__name__}.")
        return self.snapshot()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_card_id(self) -> str:
        """Generate a monotonically unique card identifier."""
        existing = {card.card_id for card in self._cards}
        while True:
            self._card_counter += 1
            card_id = f"card_{self._card_counter}"
            if card_id not in existing:
                return card_id

    def _card_index(self, card_id: str) -> int:
        """Return the position of one card or raise ``KeyError``."""
        for index, card in enumerate(self._cards):
            if card.card_id == card_id:
                return index
        raise KeyError(f"No skill card with ID {card_id!r}.")

    def _invalidate(self) -> None:
        """Discard compilation results and reset transient card states."""
        self._compiled = None
        self._cards = [
            SkillCard(
                card_id=card.card_id,
                skill_id=card.skill_id,
                entity_uid=card.entity_uid,
                params=card.params,
                state=self._configured_state(
                    card.skill_id, card.entity_uid, card.params
                ),
            )
            for card in self._cards
        ]

    def _configured_state(
        self,
        skill_id: str,
        entity_uid: str | None,
        params: Mapping[str, object],
    ) -> SkillCardState:
        """Return ``READY`` or ``UNCONFIGURED`` for a card configuration."""
        if skill_id == "pick_up":
            configured = entity_uid is not None
        else:
            configured = "position" in params
        return SkillCardState.READY if configured else SkillCardState.UNCONFIGURED

    def _validate_params(self, skill_id: str, params: Mapping[str, object]) -> None:
        """Validate parameter names and values for one skill."""
        allowed = _ALLOWED_PARAM_KEYS[skill_id]
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown parameters {unknown} for skill {skill_id!r}; "
                f"expected a subset of {sorted(allowed)}."
            )
        if "position" in params:
            _validate_vector3(params["position"], "position")
        if "rotation" in params:
            _validate_rotation(params["rotation"], "rotation")
        if "approach_direction" in params:
            direction = _validate_vector3(
                params["approach_direction"], "approach_direction"
            )
            if math.hypot(*direction) < 1.0e-6:
                raise ValueError("approach_direction must be non-zero.")
        for key in ("pre_grasp_distance", "lift_height"):
            if key in params:
                _validate_positive_number(params[key], key)
        for key in ("hand_interp_steps", "sample_count"):
            if key in params:
                _validate_positive_int(params[key], key)

    def _pose_from_params(self, params: Mapping[str, object]) -> torch.Tensor:
        """Build a homogeneous end-effector target pose from card parameters."""
        position = _validate_vector3(params["position"], "position")
        rotation = _validate_rotation(
            params.get("rotation", _TOP_DOWN_EEF_ROTATION), "rotation"
        )
        pose = torch.eye(4, dtype=torch.float32, device=self._engine.device)
        pose[:3, :3] = torch.tensor(
            rotation, dtype=torch.float32, device=self._engine.device
        )
        pose[:3, 3] = torch.tensor(
            position, dtype=torch.float32, device=self._engine.device
        )
        return pose

    def _resolve_entity(self, card: SkillCard) -> RigidObject:
        """Resolve one card's target entity from the simulation."""
        entity = self._sim.get_rigid_object(card.entity_uid)
        if entity is None:
            raise ValueError(
                f"Card {card.card_id!r} references unknown rigid object "
                f"{card.entity_uid!r}."
            )
        return entity

    def _grasp_part(self, card: SkillCard) -> str:
        """Return the grasp control part required by pick and place cards."""
        grasp = self._control_parts.get("grasp")
        if not grasp:
            raise ValueError(
                f"Card {card.card_id!r} ({card.skill_id}) requires a 'grasp' "
                "entry in control_parts."
            )
        return grasp

    def _sample_count(self, card: SkillCard) -> int:
        """Return the motion-policy sample count for one card."""
        value = card.params.get("sample_count")
        if value is None:
            return _DEFAULT_SAMPLE_COUNTS[card.skill_id]
        return int(value)

    def _build_invocation(self, card: SkillCard) -> ActionInvocation:
        """Convert one configured card into a grounded action invocation."""
        from embodichain.lab.sim.atomic_actions import (
            ActionInvocation,
            AntipodalAffordance,
            EndEffectorPoseGoal,
            GraspGoal,
            MotionPolicy,
            ObjectSemantics,
            PickUpOptions,
            PlaceGoal,
            PlaceOptions,
        )

        motion_part = self._control_parts["motion"]
        motion_policy = MotionPolicy(sample_count=self._sample_count(card))
        if card.skill_id == "move_end_effector":
            return ActionInvocation(
                skill_id="move_end_effector",
                goal=EndEffectorPoseGoal(xpos=self._pose_from_params(card.params)),
                binding=self._engine.bind_control_parts(
                    "move_end_effector",
                    {"primary": {"motion": motion_part}},
                ),
                motion_policy=motion_policy,
                invocation_id=card.card_id,
            )
        if card.skill_id == "pick_up":
            entity = self._resolve_entity(card)
            vertices = entity.get_vertices(env_ids=[0], scale=True)[0]
            triangles = entity.get_triangles(env_ids=[0])[0]
            semantics = ObjectSemantics(
                label=card.entity_uid,
                geometry={},
                affordance=AntipodalAffordance(
                    mesh_vertices=vertices,
                    mesh_triangles=triangles,
                ),
                entity_id=entity.uid,
            )
            direction = torch.tensor(
                _validate_vector3(
                    card.params.get("approach_direction", _DEFAULT_APPROACH_DIRECTION),
                    "approach_direction",
                ),
                dtype=torch.float32,
                device=self._engine.device,
            )
            direction = direction / torch.linalg.norm(direction)
            return ActionInvocation(
                skill_id="pick_up",
                goal=GraspGoal(semantics=semantics),
                binding=self._engine.bind_control_parts(
                    "pick_up",
                    {
                        "primary": {
                            "motion": motion_part,
                            "grasp": self._grasp_part(card),
                        }
                    },
                ),
                motion_policy=motion_policy,
                skill_options=PickUpOptions(
                    approach_direction=direction,
                    pre_grasp_distance=float(
                        card.params.get(
                            "pre_grasp_distance", _DEFAULT_PRE_GRASP_DISTANCE
                        )
                    ),
                    lift_height=float(
                        card.params.get("lift_height", _DEFAULT_PICK_LIFT_HEIGHT)
                    ),
                    hand_interp_steps=int(
                        card.params.get("hand_interp_steps", _DEFAULT_HAND_INTERP_STEPS)
                    ),
                ),
                invocation_id=card.card_id,
            )
        if card.skill_id == "place":
            return ActionInvocation(
                skill_id="place",
                goal=PlaceGoal(xpos=self._pose_from_params(card.params)),
                binding=self._engine.bind_control_parts(
                    "place",
                    {
                        "primary": {
                            "motion": motion_part,
                            "grasp": self._grasp_part(card),
                        }
                    },
                ),
                motion_policy=motion_policy,
                skill_options=PlaceOptions(
                    lift_height=float(
                        card.params.get("lift_height", _DEFAULT_PLACE_LIFT_HEIGHT)
                    ),
                    hand_interp_steps=int(
                        card.params.get("hand_interp_steps", _DEFAULT_HAND_INTERP_STEPS)
                    ),
                ),
                invocation_id=card.card_id,
            )
        raise ValueError(f"Unsupported skill {card.skill_id!r}.")

    def _build_scene_snapshot(self) -> SceneSnapshot | None:
        """Snapshot every referenced entity pose for grounded planning."""
        from embodichain.lab.sim.atomic_actions import EntityState, SceneSnapshot

        entities = {}
        for card in self._cards:
            if card.entity_uid is None or card.entity_uid in entities:
                continue
            entity = self._resolve_entity(card)
            entities[entity.uid] = EntityState(entity.get_local_pose(to_matrix=True))
        if not entities:
            return None
        return SceneSnapshot(timestamp=0.0, version=0, entities=entities)

    @staticmethod
    def _failure_message(plan: ActionPlan) -> str:
        """Format one failed plan's diagnostics into a card message."""
        diagnostics = plan.diagnostics
        parts: list[str] = []
        if diagnostics.failure is not None:
            parts.append(diagnostics.failure.code)
        parts.extend(message for message in diagnostics.messages if message)
        return "; ".join(parts) if parts else "Planning failed without diagnostics."

    @staticmethod
    def _waypoint_sim_steps(
        trajectory: TimedTrajectory,
        step_index: int,
        total_steps: int,
        physics_dt: float,
    ) -> int:
        """Convert one waypoint arrival interval into physics-step counts."""
        next_index = min(step_index + 1, total_steps - 1)
        duration = float(trajectory.dt[:, next_index].max().item())
        step_ratio = duration / physics_dt
        nearest_step_count = round(step_ratio)
        if math.isclose(step_ratio, nearest_step_count, rel_tol=1.0e-6, abs_tol=1.0e-9):
            return max(1, nearest_step_count)
        return max(1, math.ceil(step_ratio))

    def _apply_execution_states(self, step_index: int) -> None:
        """Advance card states for the waypoint about to be executed."""
        cards = []
        for card in self._cards:
            if card.segment_start is None:
                cards.append(card)
            elif card.segment_stop <= step_index:
                cards.append(replace(card, state=SkillCardState.SUCCEEDED))
            elif card.segment_start <= step_index:
                cards.append(replace(card, state=SkillCardState.RUNNING))
            else:
                cards.append(card)
        self._cards = cards

    def _mark_execution_failure(self, step_index: int, error: Exception) -> None:
        """Mark the card owning ``step_index`` as failed after an exception."""
        message = f"Execution raised {type(error).__name__}: {error}"
        cards = []
        for card in self._cards:
            owns_step = (
                card.segment_start is not None
                and card.segment_start <= step_index < card.segment_stop
            )
            if owns_step or card.state is SkillCardState.RUNNING:
                cards.append(
                    replace(
                        card,
                        state=SkillCardState.FAILED,
                        failure_message=message,
                    )
                )
            else:
                cards.append(card)
        self._cards = cards
