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

"""Simulation-thread bridge between a browser panel and an authoring session.

:class:`AuthoringBridge` closes the loop opened by
:class:`~embodichain.lab.visualization.authoring.panel.SkillSequencePanel`.
Once per simulation iteration it drains the browser click-picks and the panel
commands queued by the visualization runtime, applies them through
:meth:`~embodichain.lab.visualization.authoring.session.AuthoringSession.handle_command`
or on the preview driver, and publishes the resulting immutable
:class:`~embodichain.lab.visualization.authoring.panel.PanelViewState` back to
the panel.

The bridge is the only object allowed to mutate the session on behalf of the
browser. It converts every failure into a status string instead of raising, so
one invalid browser request can never abort a simulation loop. Browser input
stamped with a stale ``run_id`` or ``scene_revision`` is dropped, so a click
queued before a scene refresh can never be applied to a new topology.

The click-pick queue is shared with
:meth:`~embodichain.lab.sim.SimulationManager.process_pick_commands`, which
:meth:`~embodichain.lab.sim.SimulationManager.update` runs on every step. A host
loop whose simulation manager also has Gizmo picking enabled must therefore call
:meth:`AuthoringBridge.drain_picks` **before** ``sim.update()``; otherwise the
simulation manager consumes every pick and the panel's selection stays empty.

By default an :class:`~embodichain.lab.visualization.authoring.protocol.ExecuteSequence`
command is applied through the session's blocking
:meth:`~embodichain.lab.visualization.authoring.session.AuthoringSession.execute`,
which keeps the browser frozen until the run finishes. Enabling
``stepwise_execution`` instead starts a
:class:`~embodichain.lab.visualization.authoring.execution.StepwiseExecution`
that the host loop advances a few ticks per :meth:`AuthoringBridge.update`, so
the card states reach the panel while the trajectory is still running.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .execution import StepwiseExecution
from .panel import (
    PanelViewState,
    SeekPreview,
    SkillSequencePanel,
    StepPreview,
    TogglePreviewPlayback,
)
from .protocol import (
    AddCard,
    CompileSequence,
    ExecuteSequence,
    MoveCard,
    RemoveCard,
    SequenceSnapshot,
    UpdateCard,
)

if TYPE_CHECKING:
    from ..runtime import VisualizationRuntime
    from .preview import SequencePreview
    from .session import AuthoringSession

__all__ = ["AuthoringBridge"]

_AUTHORING_COMMANDS = (
    AddCard,
    RemoveCard,
    MoveCard,
    UpdateCard,
    CompileSequence,
    ExecuteSequence,
)
_PREVIEW_COMMANDS = (TogglePreviewPlayback, SeekPreview, StepPreview)
_PICKABLE_KINDS = frozenset({"rigid"})


class AuthoringBridge:
    """Drive an authoring session from one browser panel.

    Args:
        session: Authoring session owning the sequence and its compilation.
        runtime: Visualization runtime whose queues carry browser commands.
        panel: Panel registered on the runtime. Only commands carrying its
            panel identifier are applied.
        preview: Optional preview driver receiving playback commands.
        process_picks: Whether the bridge drains browser click-picks to track
            the selected entity. Disable it when another consumer, for example
            :meth:`~embodichain.lab.sim.SimulationManager.process_pick_commands`,
            already drains that queue. When both consume picks, the host loop
            must call :meth:`drain_picks` before ``sim.update()``, which drains
            the same queue through its Gizmo processing.
        pickable_kinds: Asset kinds accepted as skill targets.
        stepwise_execution: Whether an ``ExecuteSequence`` command starts a
            host-driven :class:`StepwiseExecution` instead of blocking inside
            :meth:`update`. Enable it whenever a browser is attached, and have
            the host loop skip its own simulation stepping while
            :attr:`execution_active` is true.
        execution_steps_per_update: Simulation updates performed by each
            :meth:`update` call while a stepwise execution is active. Larger
            values replay faster at a coarser browser frame rate.
        execution_on_step: Optional callback forwarded to the stepwise
            execution as ``on_step(step_index, total_steps)``. It runs on the
            simulation thread after every trajectory update, which is where a
            host can freeze an attached object's dynamics.
    """

    def __init__(
        self,
        session: AuthoringSession,
        runtime: VisualizationRuntime,
        panel: SkillSequencePanel,
        preview: SequencePreview | None = None,
        *,
        process_picks: bool = True,
        pickable_kinds: frozenset[str] = _PICKABLE_KINDS,
        stepwise_execution: bool = False,
        execution_steps_per_update: int = 1,
        execution_on_step: Callable[[int, int], None] | None = None,
    ) -> None:
        if execution_steps_per_update < 1:
            raise ValueError("execution_steps_per_update must be at least one.")
        self._session = session
        self._runtime = runtime
        self._panel = panel
        self._preview = preview
        self._process_picks = bool(process_picks)
        self._pickable_kinds = frozenset(pickable_kinds)
        self._stepwise_execution = bool(stepwise_execution)
        self._execution_steps_per_update = int(execution_steps_per_update)
        self._execution_on_step = execution_on_step
        self._execution: StepwiseExecution | None = None
        self._selected_entity_uid: str | None = None
        self._status = ""
        self._view = PanelViewState(snapshot=SequenceSnapshot(cards=(), compiled=False))
        self._published = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def panel_id(self) -> str:
        """Identifier of the panel this bridge serves."""
        return self._panel.panel_id

    @property
    def view_state(self) -> PanelViewState:
        """Most recently built view state."""
        return self._view

    @property
    def selected_entity_uid(self) -> str | None:
        """Entity UID of the newest browser click-pick, if any."""
        return self._selected_entity_uid

    @property
    def status(self) -> str:
        """Status or error text shown in the panel header."""
        return self._status

    @property
    def execution_active(self) -> bool:
        """Whether a host-driven execution still owns the simulation.

        A host loop must not step the simulation itself while this is true:
        every :meth:`update` call already advances the run, which steps physics
        for the waypoints it replays.
        """
        return self._execution is not None

    def register(self) -> None:
        """Register the panel on the runtime and publish the first state."""
        self._runtime.register_panel(self._panel.spec)
        self.publish(force=True)

    def unregister(self) -> None:
        """Remove the panel from the runtime and abandon a running execution."""
        self.cancel_execution()
        self._runtime.unregister_panel(self._panel.panel_id)
        self._published = False

    def cancel_execution(self) -> None:
        """Abandon a host-driven execution without advancing it further."""
        execution = self._execution
        if execution is None:
            return
        self._execution = None
        execution.close()
        self._status = "Execution cancelled."

    # ------------------------------------------------------------------
    # Per-iteration update
    # ------------------------------------------------------------------

    def update(self) -> PanelViewState:
        """Apply queued browser input and publish the resulting state.

        Call once per simulation iteration, before capturing a frame. When a
        host-driven execution is active the call also advances it by
        ``execution_steps_per_update`` simulation updates.

        Commands stamped with a stale ``run_id`` or ``scene_revision`` are
        discarded, so a click queued before a
        :meth:`~embodichain.lab.visualization.VisualizationRuntime.refresh_scene`
        cannot be applied against a different scene topology.

        The browser pick queue is shared with
        :meth:`~embodichain.lab.sim.SimulationManager.process_pick_commands`,
        which :meth:`~embodichain.lab.sim.SimulationManager.update` calls on
        every step. When the simulation manager also processes picks, call
        :meth:`drain_picks` before ``sim.update()`` or the browser selection
        never reaches this bridge.

        Returns:
            The view state published to the panel.
        """
        self.drain_picks()
        applied = False
        exporter = self._runtime.exporter
        for command in self._runtime.drain_panel_commands(self._panel.panel_id):
            if command.panel_id != self._panel.panel_id:
                continue
            if (
                command.run_id != exporter.run_id
                or command.scene_revision != exporter.scene_revision
            ):
                continue
            self._apply(command.value)
            applied = True
        self._advance_execution(keep_status=applied)
        return self.publish()

    def drain_picks(self) -> str | None:
        """Consume queued browser click-picks and track the newest selection.

        :meth:`update` already calls this. Call it separately, **before**
        :meth:`~embodichain.lab.sim.SimulationManager.update`, whenever the
        simulation manager processes picks too: its per-step
        :meth:`~embodichain.lab.sim.SimulationManager.process_pick_commands`
        drains the same runtime queue, so a bridge running after it would never
        observe a browser selection. The call is cheap and idempotent when the
        queue is empty, and does nothing when ``process_picks`` is disabled.

        Picks stamped with a stale ``run_id`` or ``scene_revision`` are
        discarded, and picks resolving to a kind outside ``pickable_kinds``
        only update the status line.

        Returns:
            The selected entity UID after the drain, or ``None`` when nothing
            is selected.
        """
        if not self._process_picks:
            return self._selected_entity_uid
        exporter = self._runtime.exporter
        for command in self._runtime.drain_pick_commands():
            if (
                command.run_id != exporter.run_id
                or command.scene_revision != exporter.scene_revision
            ):
                continue
            if command.node_id is None:
                self._selected_entity_uid = None
                continue
            resolved = exporter.resolve_node_target(command.node_id)
            if resolved is None:
                continue
            uid, kind = resolved
            if kind not in self._pickable_kinds:
                self._status = f"Picked {kind} {uid!r} cannot be a skill target."
                continue
            self._selected_entity_uid = uid
            self._status = f"Picked entity {uid!r}."
        return self._selected_entity_uid

    def publish(self, *, force: bool = False) -> PanelViewState:
        """Rebuild the view state and publish it when it changed.

        Args:
            force: Publish even when the state is unchanged.

        Returns:
            The current view state.
        """
        view = self._build_view()
        changed = force or not self._published or view != self._view
        self._view = view
        if changed:
            self._runtime.publish_panel_state(self._panel.panel_id, view)
            self._published = True
        return view

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_view(self) -> PanelViewState:
        """Snapshot the session, the preview, and the selected entity."""
        playback = None if self._preview is None else self._preview.state()
        return PanelViewState(
            snapshot=self._session.snapshot(),
            playback=playback,
            selected_entity_uid=self._selected_entity_uid,
            status=self._status,
        )

    def _apply(self, value: object) -> None:
        """Apply one panel command value on the simulation thread."""
        if isinstance(value, _AUTHORING_COMMANDS):
            self._apply_authoring(value)
        elif isinstance(value, _PREVIEW_COMMANDS):
            self._apply_preview(value)
        else:
            self._status = f"Ignored unknown panel command {type(value).__name__}."

    def _apply_authoring(self, command: object) -> None:
        """Route one authoring command through the session's single entry."""
        if self._execution is not None:
            self._status = (
                f"Ignored {type(command).__name__}; an execution is in progress."
            )
            return
        if self._stepwise_execution and isinstance(command, ExecuteSequence):
            self._start_execution(command)
            return
        try:
            self._session.handle_command(command)
        except Exception as error:  # noqa: BLE001 - surfaced in the panel.
            self._status = f"{type(command).__name__} failed: {error}"
            return
        self._status = self._success_status(command)

    def _start_execution(self, command: ExecuteSequence) -> None:
        """Begin a host-driven execution instead of blocking on the session."""
        try:
            execution = StepwiseExecution(
                self._session,
                self._execution_on_step,
                hold_steps=command.hold_steps,
            )
        except Exception as error:  # noqa: BLE001 - surfaced in the panel.
            self._status = f"ExecuteSequence failed: {error}"
            return
        self._execution = execution
        self._status = f"Executing {self._session.preview_length} waypoints."

    def _advance_execution(self, *, keep_status: bool = False) -> None:
        """Advance a host-driven execution by one bounded slice of ticks.

        Args:
            keep_status: Whether a status produced by a command applied in the
                same iteration must survive. A progress line is noise next to
                the reason one browser request was refused, but a terminal
                execution result always replaces it.
        """
        execution = self._execution
        if execution is None:
            return
        progress = execution.advance(self._execution_steps_per_update)
        if execution.is_finished:
            self._execution = None
            self._status = (
                "Execution finished."
                if execution.succeeded
                else "Execution failed; see the failed card below."
            )
            return
        if progress is None or keep_status:
            return
        phase = "holding" if progress.holding else "waypoint"
        self._status = (
            f"Executing: {phase} {progress.step_index + 1}/{progress.total_steps}."
        )

    def _success_status(self, command: object) -> str:
        """Describe one successfully applied authoring command."""
        if isinstance(command, CompileSequence):
            if self._session.is_compiled:
                return (
                    f"Compiled {self._session.preview_length} waypoints from "
                    f"{len(self._session.cards)} cards."
                )
            return "Compilation failed; see the failed card below."
        if isinstance(command, ExecuteSequence):
            return "Execution finished."
        return f"Applied {type(command).__name__}."

    def _apply_preview(self, command: object) -> None:
        """Apply one playback command on the optional preview driver."""
        preview = self._preview
        if preview is None:
            self._status = "Preview playback is not available."
            return
        try:
            if isinstance(command, TogglePreviewPlayback):
                preview.toggle()
            elif isinstance(command, SeekPreview):
                preview.seek(command.index)
            elif isinstance(command, StepPreview):
                preview.step(command.delta)
        except Exception as error:  # noqa: BLE001 - surfaced in the panel.
            self._status = f"{type(command).__name__} failed: {error}"

    def clear_status(self) -> None:
        """Clear the status line shown in the panel header.

        The cleared text reaches the browser through the next
        :meth:`publish` or :meth:`update`, which observes the changed state.
        """
        self._status = ""
