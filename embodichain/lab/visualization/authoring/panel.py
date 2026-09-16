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

"""Browser side panel for authoring and previewing a skill sequence.

The panel is a pure UI layer living on the visualization worker thread. It
renders an immutable :class:`PanelViewState` into browser controls and turns
every browser interaction into an immutable command value, which the backend
forwards to the simulation thread as a
:class:`~embodichain.lab.visualization.protocol.PanelCommand`. The panel never
calls a mutating method on
:class:`~embodichain.lab.visualization.authoring.session.AuthoringSession`;
:class:`~embodichain.lab.visualization.authoring.bridge.AuthoringBridge` owns
that side of the boundary.

Sequence edits are emitted as
:data:`~embodichain.lab.visualization.authoring.protocol.AuthoringCommand`
values. Preview playback is not part of the authoring protocol, so the panel
emits the small :data:`PreviewCommand` family defined here instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, Sequence, Union

from embodichain.utils import configclass

from .protocol import (
    AddCard,
    CompileSequence,
    ExecuteSequence,
    MoveCard,
    RemoveCard,
    SUPPORTED_SKILL_IDS,
    SequenceSnapshot,
    SkillCard,
    SkillCardState,
    UpdateCard,
)

if TYPE_CHECKING:
    from ..panels import PanelBuildContext, PanelEventSink, PanelSpec
    from .preview import PreviewPlaybackState

__all__ = [
    "CardRow",
    "PanelViewState",
    "PreviewCommand",
    "SeekPreview",
    "SkillSequencePanel",
    "SkillSequencePanelCfg",
    "StepPreview",
    "TogglePreviewPlayback",
    "card_rows",
    "render_cards_markdown",
    "render_preview_markdown",
    "render_summary_markdown",
]

NO_CARD_OPTION = "—"
"""Dropdown placeholder shown while the sequence holds no card."""

_STATE_STYLE: Mapping[SkillCardState, tuple[str, str]] = {
    SkillCardState.UNCONFIGURED: ("🟡", "#c9a227"),
    SkillCardState.READY: ("⚪", "#c9ccd1"),
    SkillCardState.RUNNING: ("🔵", "#3b9dff"),
    SkillCardState.SUCCEEDED: ("🟢", "#2fb344"),
    SkillCardState.FAILED: ("🔴", "#e5484d"),
}
"""Five-state marker glyph and RGB hex used to color one card row."""


# ----------------------------------------------------------------------------
# Preview playback commands
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class TogglePreviewPlayback:
    """Flip translucent preview playback between playing and paused."""


@dataclass(frozen=True)
class SeekPreview:
    """Move the preview cursor to one compiled waypoint."""

    index: int

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError("SeekPreview.index must be an integer.")
        if self.index < 0:
            raise ValueError("SeekPreview.index must be non-negative.")


@dataclass(frozen=True)
class StepPreview:
    """Nudge the preview cursor by a signed number of waypoints."""

    delta: int

    def __post_init__(self) -> None:
        if isinstance(self.delta, bool) or not isinstance(self.delta, int):
            raise TypeError("StepPreview.delta must be an integer.")
        if self.delta == 0:
            raise ValueError("StepPreview.delta must not be zero.")


PreviewCommand = Union[TogglePreviewPlayback, SeekPreview, StepPreview]
"""Union of every preview playback command the panel can emit."""


# ----------------------------------------------------------------------------
# Display model
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelViewState:
    """Immutable panel input published by the simulation thread."""

    snapshot: SequenceSnapshot
    playback: PreviewPlaybackState | None = None
    selected_entity_uid: str | None = None
    status: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, SequenceSnapshot):
            raise TypeError("PanelViewState.snapshot must be a SequenceSnapshot.")
        if self.selected_entity_uid is not None and not self.selected_entity_uid:
            raise ValueError("selected_entity_uid must be None or non-empty.")


@dataclass(frozen=True)
class CardRow:
    """One card rendered as a backend-independent display row."""

    index: int
    card_id: str
    skill_id: str
    entity_uid: str | None
    state: SkillCardState
    marker: str
    color: str
    headline: str
    detail: str
    selected: bool = False
    active: bool = False


def _card_detail(card: SkillCard) -> str:
    """Return the secondary line describing one card's state."""
    if card.state is SkillCardState.FAILED:
        return card.failure_message or "Failed without a diagnostic message."
    if card.state is SkillCardState.UNCONFIGURED:
        if card.skill_id == "pick_up":
            return "Needs a target entity."
        return "Needs a target position."
    if card.segment_start is not None and card.segment_stop is not None:
        return (
            f"waypoints {card.segment_start}–{card.segment_stop} "
            f"({card.segment_waypoint_count})"
        )
    return ""


def card_rows(
    view: PanelViewState,
    selected_card_id: str | None = None,
) -> tuple[CardRow, ...]:
    """Map one panel view state onto ordered display rows.

    Args:
        view: Immutable state published by the simulation thread.
        selected_card_id: Card currently selected in the panel, if any.

    Returns:
        One row per card, in sequence order.
    """
    active_card_id = None if view.playback is None else view.playback.active_card_id
    rows: list[CardRow] = []
    for index, card in enumerate(view.snapshot.cards):
        marker, color = _STATE_STYLE[card.state]
        target = "no target" if card.entity_uid is None else card.entity_uid
        rows.append(
            CardRow(
                index=index,
                card_id=card.card_id,
                skill_id=card.skill_id,
                entity_uid=card.entity_uid,
                state=card.state,
                marker=marker,
                color=color,
                headline=f"{index + 1}. {card.skill_id} · {target}",
                detail=_card_detail(card),
                selected=card.card_id == selected_card_id,
                active=card.card_id == active_card_id,
            )
        )
    return tuple(rows)


def render_cards_markdown(rows: Sequence[CardRow]) -> str:
    """Render display rows as markdown for a backend without rich text.

    Args:
        rows: Rows produced by :func:`card_rows`.

    Returns:
        Markdown text where the five card states are distinguished by a
        colored marker glyph, the selected card by a caret, and the card being
        previewed by a trailing marker.
    """
    if not rows:
        return "_No skill cards yet._"
    lines: list[str] = []
    for row in rows:
        caret = "▸ " if row.selected else "　"
        headline = f"**{row.headline}**" if row.active else row.headline
        suffix = "  ◀ preview" if row.active else ""
        lines.append(f"{caret}{row.marker} {headline}{suffix}")
        if row.detail:
            lines.append(f"　　`{row.state.value}` {row.detail}")
    return "  \n".join(lines)


def render_summary_markdown(view: PanelViewState) -> str:
    """Render the sequence headline shown above the card list."""
    snapshot = view.snapshot
    compiled = "yes" if snapshot.compiled else "no"
    summary = (
        f"**Cards:** {len(snapshot.cards)} · **Compiled:** {compiled} · "
        f"**Waypoints:** {snapshot.trajectory_waypoint_count}"
    )
    if view.status:
        summary = f"{summary}  \n{view.status}"
    return summary


def render_preview_markdown(view: PanelViewState) -> str:
    """Render the preview playback status line."""
    playback = view.playback
    if playback is None:
        return "_Preview playback is not available._"
    if playback.length == 0:
        return "_Compile the sequence to preview it._"
    mode = "playing" if playback.playing else "paused"
    active = playback.active_card_id or "—"
    return (
        f"**{mode}** · frame {playback.cursor + 1}/{playback.length} · "
        f"card `{active}`"
    )


# ----------------------------------------------------------------------------
# Panel
# ----------------------------------------------------------------------------


@configclass
class SkillSequencePanelCfg:
    """Configure the skill-sequence side panel.

    Args:
        panel_id: Identifier used to register the panel and to match the
            commands it emits.
        title: Folder title of the panel container in the browser sidebar.
        skill_ids: Skills offered by the panel's skill dropdown.
        position_step: Step of the target-position numeric inputs, in meters.
        position_limit: Absolute bound of the target-position inputs, in
            meters.
        execute_hold_steps: ``hold_steps`` carried by emitted
            :class:`~embodichain.lab.visualization.authoring.protocol.ExecuteSequence`
            commands.
        show_preview_controls: Whether playback controls are built.
    """

    panel_id: str = "skill_sequence"
    title: str = "Skill sequence"
    skill_ids: tuple[str, ...] = SUPPORTED_SKILL_IDS
    position_step: float = 0.005
    position_limit: float = 5.0
    execute_hold_steps: int = 0
    show_preview_controls: bool = True

    def __post_init__(self) -> None:
        """Validate the panel identity and numeric-input ranges."""
        if not self.panel_id:
            raise ValueError("panel_id must not be empty.")
        if not self.title:
            raise ValueError("title must not be empty.")
        if not self.skill_ids:
            raise ValueError("skill_ids must not be empty.")
        if self.position_step <= 0.0:
            raise ValueError("position_step must be greater than zero.")
        if self.position_limit <= 0.0:
            raise ValueError("position_limit must be greater than zero.")
        if self.execute_hold_steps < 0:
            raise ValueError("execute_hold_steps must be non-negative.")


class SkillSequencePanel:
    """Skill-sequence authoring panel built on the visualization thread.

    The panel owns browser handles only. :meth:`build` and :meth:`apply_state`
    run on the visualization worker thread, which is the only thread allowed to
    create or mutate them. Browser callbacks run on the backend's callback
    thread, where they read control values and the last applied view state,
    emit an immutable command value, and at most rebind the selected card
    identifier. Both cross-thread attributes hold immutable values, so a reader
    never observes a partially written sequence.

    Args:
        cfg: Panel identity and input ranges. Defaults are used when omitted.
    """

    def __init__(self, cfg: SkillSequencePanelCfg | None = None) -> None:
        self.cfg = cfg if cfg is not None else SkillSequencePanelCfg()
        self._gui: object | None = None
        self._emit: PanelEventSink | None = None
        self._client_id: Callable[[object], str | None] | None = None
        self._allow_commands = False
        # Written on the visualization thread, read by browser callbacks. The
        # value is always a frozen snapshot, so readers never observe a
        # half-updated sequence.
        self._view: PanelViewState | None = None
        self._selected_card_id: str | None = None
        self._summary_handle: object | None = None
        self._cards_handle: object | None = None
        self._card_dropdown: object | None = None
        self._skill_dropdown: object | None = None
        self._entity_handle: object | None = None
        self._position_handles: tuple[object, object, object] | None = None
        self._preview_folder: object | None = None
        self._preview_handle: object | None = None
        self._frame_slider: object | None = None
        self._frame_slider_max: int | None = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @property
    def panel_id(self) -> str:
        """Identifier this panel registers under."""
        return self.cfg.panel_id

    @property
    def spec(self) -> PanelSpec:
        """Backend-neutral registration for this panel."""
        from ..panels import PanelSpec

        return PanelSpec(
            panel_id=self.cfg.panel_id,
            build=self.build,
            apply_state=self.apply_state,
            title=self.cfg.title,
        )

    @property
    def view(self) -> PanelViewState | None:
        """Last applied view state, or ``None`` before the first update."""
        return self._view

    @property
    def selected_card_id(self) -> str | None:
        """Card currently selected in the panel, if any."""
        return self._selected_card_id

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, context: PanelBuildContext) -> None:
        """Create every browser control of the panel.

        Args:
            context: Backend-provided GUI namespace and event sink.
        """
        self._gui = context.gui
        self._emit = context.emit
        self._client_id = context.client_id
        self._allow_commands = bool(context.allow_commands)
        self._frame_slider = None
        self._frame_slider_max = None
        gui = context.gui
        self._summary_handle = gui.add_markdown(self._summary_text())
        with gui.add_folder("Cards", expand_by_default=True):
            self._cards_handle = gui.add_markdown(self._cards_text())
            self._card_dropdown = gui.add_dropdown(
                "Selected card",
                options=self._card_options(),
                initial_value=self._selected_card_id or NO_CARD_OPTION,
            )

            @self._card_dropdown.on_update
            def _(event: object) -> None:
                value = str(event.target.value)
                self._selected_card_id = None if value == NO_CARD_OPTION else value

        with gui.add_folder("Edit", expand_by_default=True):
            self._skill_dropdown = gui.add_dropdown(
                "Skill",
                options=list(self.cfg.skill_ids),
                initial_value=self.cfg.skill_ids[0],
            )
            self._add_button(gui, "Add card", self._on_add_card)
            self._add_button(gui, "Remove card", self._on_remove_card)
            self._add_button(gui, "Move up", lambda event: self._on_move(event, -1))
            self._add_button(gui, "Move down", lambda event: self._on_move(event, 1))
            self._entity_handle = gui.add_markdown(self._entity_text())
            self._add_button(gui, "Bind selected entity", self._on_bind_entity)
            self._position_handles = tuple(
                gui.add_number(
                    f"Target {axis} (m)",
                    initial_value=0.0,
                    min=-self.cfg.position_limit,
                    max=self.cfg.position_limit,
                    step=self.cfg.position_step,
                    disabled=not self._allow_commands,
                    hint=f"{axis} component of the card's target position.",
                )
                for axis in ("x", "y", "z")
            )
            self._add_button(gui, "Apply target position", self._on_apply_position)

        with gui.add_folder("Run", expand_by_default=True):
            self._add_button(gui, "Compile sequence", self._on_compile)
            self._add_button(gui, "Execute sequence", self._on_execute)

        if self.cfg.show_preview_controls:
            self._preview_folder = gui.add_folder("Preview", expand_by_default=True)
            with self._preview_folder:
                self._preview_handle = gui.add_markdown(self._preview_text())
                self._add_button(gui, "Play / Pause", self._on_toggle_playback)
                self._add_button(
                    gui, "Step backward", lambda event: self._on_step(event, -1)
                )
                self._add_button(
                    gui, "Step forward", lambda event: self._on_step(event, 1)
                )
            self._sync_frame_slider()

    def _add_button(self, gui: object, label: str, handler: object) -> object:
        """Add one command button wired to ``handler``."""
        button = gui.add_button(label, disabled=not self._allow_commands)
        button.on_click(handler)
        return button

    # ------------------------------------------------------------------
    # State application
    # ------------------------------------------------------------------

    def apply_state(self, state: object) -> None:
        """Render one published view state into the panel's controls.

        Args:
            state: A :class:`PanelViewState` published by the bridge. Other
                payloads are ignored so an unrelated publisher cannot corrupt
                the panel.
        """
        if not isinstance(state, PanelViewState):
            return
        self._view = state
        card_ids = [card.card_id for card in state.snapshot.cards]
        if self._selected_card_id not in card_ids:
            self._selected_card_id = card_ids[0] if card_ids else None
        if self._summary_handle is not None:
            self._summary_handle.content = self._summary_text()
        if self._cards_handle is not None:
            self._cards_handle.content = self._cards_text()
        if self._card_dropdown is not None:
            options = self._card_options()
            if list(self._card_dropdown.options) != options:
                self._card_dropdown.options = options
            value = self._selected_card_id or NO_CARD_OPTION
            if self._card_dropdown.value != value:
                self._card_dropdown.value = value
        if self._entity_handle is not None:
            self._entity_handle.content = self._entity_text()
        if self._preview_handle is not None:
            self._preview_handle.content = self._preview_text()
        if self.cfg.show_preview_controls:
            self._sync_frame_slider()

    def _sync_frame_slider(self) -> None:
        """Create, resize, or move the preview seek slider.

        Viser sliders expose no mutable range, so a changed trajectory length
        rebuilds the control inside the preview folder. The slider is the last
        control of that folder, which keeps its position stable across rebuilds.
        """
        playback = None if self._view is None else self._view.playback
        length = 0 if playback is None else playback.length
        cursor = 0 if playback is None else playback.cursor
        slider_max = max(1, length - 1)
        if self._frame_slider is None or self._frame_slider_max != slider_max:
            if self._frame_slider is not None:
                self._frame_slider.remove()
                self._frame_slider = None
            if self._gui is None or self._preview_folder is None:
                return
            with self._preview_folder:
                slider = self._gui.add_slider(
                    "Frame",
                    min=0,
                    max=slider_max,
                    step=1,
                    initial_value=min(cursor, slider_max),
                    marks=(),
                    disabled=not self._allow_commands or length == 0,
                    hint="Seek the translucent preview to a compiled waypoint.",
                )
            slider.on_update(self._on_seek)
            self._frame_slider = slider
            self._frame_slider_max = slider_max
            return
        target = min(cursor, slider_max)
        if self._frame_slider.value != target:
            self._frame_slider.value = target

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _empty_view(self) -> PanelViewState:
        """Return the placeholder state used before the first publication."""
        return PanelViewState(snapshot=SequenceSnapshot(cards=(), compiled=False))

    def _summary_text(self) -> str:
        return render_summary_markdown(self._view or self._empty_view())

    def _cards_text(self) -> str:
        view = self._view or self._empty_view()
        return render_cards_markdown(card_rows(view, self._selected_card_id))

    def _preview_text(self) -> str:
        return render_preview_markdown(self._view or self._empty_view())

    def _entity_text(self) -> str:
        uid = None if self._view is None else self._view.selected_entity_uid
        if uid is None:
            return "**Picked entity:** _none_"
        return f"**Picked entity:** `{uid}`"

    def _card_options(self) -> list[str]:
        view = self._view
        if view is None or not view.snapshot.cards:
            return [NO_CARD_OPTION]
        return [card.card_id for card in view.snapshot.cards]

    # ------------------------------------------------------------------
    # Browser callbacks
    # ------------------------------------------------------------------

    def _publish(self, value: object, event: object | None = None) -> None:
        """Emit one immutable command value to the simulation thread."""
        if self._emit is None:
            return
        self._emit(value, event=event)

    def _current_card_id(self) -> str | None:
        """Return the selected card identifier, preferring the live control."""
        dropdown = self._card_dropdown
        if dropdown is not None:
            value = str(dropdown.value)
            if value != NO_CARD_OPTION:
                return value
            return None
        return self._selected_card_id

    def _card_index(self, card_id: str) -> int | None:
        """Return the sequence position of ``card_id`` in the last view."""
        view = self._view
        if view is None:
            return None
        for index, card in enumerate(view.snapshot.cards):
            if card.card_id == card_id:
                return index
        return None

    def _on_add_card(self, event: object) -> None:
        skill_id = str(self._skill_dropdown.value)
        self._publish(AddCard(skill_id=skill_id), event)

    def _on_remove_card(self, event: object) -> None:
        card_id = self._current_card_id()
        if card_id is not None:
            self._publish(RemoveCard(card_id=card_id), event)

    def _on_move(self, event: object, delta: int) -> None:
        card_id = self._current_card_id()
        if card_id is None:
            return
        index = self._card_index(card_id)
        if index is None:
            return
        new_index = index + delta
        view = self._view
        if new_index < 0 or view is None or new_index >= len(view.snapshot.cards):
            return
        self._publish(MoveCard(card_id=card_id, new_index=new_index), event)

    def _on_bind_entity(self, event: object) -> None:
        card_id = self._current_card_id()
        view = self._view
        if card_id is None or view is None or view.selected_entity_uid is None:
            return
        self._publish(
            UpdateCard(card_id=card_id, entity_uid=view.selected_entity_uid),
            event,
        )

    def _on_apply_position(self, event: object) -> None:
        card_id = self._current_card_id()
        if card_id is None or self._position_handles is None:
            return
        position = tuple(float(handle.value) for handle in self._position_handles)
        self._publish(
            UpdateCard(card_id=card_id, params={"position": position}),
            event,
        )

    def _on_compile(self, event: object) -> None:
        self._publish(CompileSequence(), event)

    def _on_execute(self, event: object) -> None:
        self._publish(ExecuteSequence(hold_steps=self.cfg.execute_hold_steps), event)

    def _on_toggle_playback(self, event: object) -> None:
        self._publish(TogglePreviewPlayback(), event)

    def _on_step(self, event: object, delta: int) -> None:
        self._publish(StepPreview(delta=delta), event)

    def _on_seek(self, event: object) -> None:
        # Writing ``value`` from the visualization thread synchronously invokes
        # this callback without an originating client. Ignoring those events
        # keeps a played-back cursor from being seeked back to a stale frame.
        if self._client_id is None or self._client_id(event) is None:
            return
        self._publish(SeekPreview(index=int(round(event.target.value))), event)
