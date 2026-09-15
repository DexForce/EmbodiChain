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

"""Tests for the custom panel registry and the skill-sequence panel."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from embodichain.lab.visualization import (
    PanelBuildContext,
    PanelCommand,
    PanelCommandQueue,
    PanelSpec,
    SceneManifest,
    VisualizationCfg,
    VisualizationRuntime,
    ViserServerCfg,
)
from embodichain.lab.visualization.backends.base import VisualizationBackend
from embodichain.lab.visualization.authoring import (
    AddCard,
    AuthoringBridge,
    AuthoringSession,
    CompileSequence,
    ExecuteSequence,
    MoveCard,
    PanelViewState,
    PreviewPlaybackState,
    RemoveCard,
    SeekPreview,
    SequenceSnapshot,
    SkillCard,
    SkillCardState,
    SkillSequencePanel,
    SkillSequencePanelCfg,
    StepPreview,
    TogglePreviewPlayback,
    UpdateCard,
    card_rows,
    render_cards_markdown,
    render_preview_markdown,
    render_summary_markdown,
)
from embodichain.lab.visualization.backends.viser import ViserBackend

TARGET_POSITION = (-0.42, -0.08, 0.36)
PREVIEW_LENGTH = 24
PREVIEW_CURSOR = 7


def _queued_panel_command(panel_id: str, sequence: int) -> PanelCommand:
    """Build one panel command carrying its arrival index as the value."""
    return PanelCommand(
        run_id="run",
        scene_revision=0,
        sequence=sequence,
        panel_id=panel_id,
        client_id="client",
        value=sequence,
    )


# ----------------------------------------------------------------------------
# Browser-free fakes
# ----------------------------------------------------------------------------


class _Control:
    """Minimal stand-in for a Viser input handle."""

    def __init__(self, label: str, **kwargs: object) -> None:
        self.label = label
        self.removed = False
        self.callback: object | None = None
        self.__dict__.update(kwargs)

    def on_update(self, callback: object) -> object:
        self.callback = callback
        return callback

    def on_click(self, callback: object) -> object:
        self.callback = callback
        return callback

    def remove(self) -> None:
        self.removed = True

    def fire(self, value: object = None, client_id: str | None = "client-1") -> None:
        """Invoke the registered callback like a browser interaction would."""
        assert self.callback is not None
        if value is not None:
            self.value = value
        self.callback(SimpleNamespace(client_id=client_id, target=self))


class _Folder(_Control):
    def __enter__(self) -> _Folder:
        self.gui.container.append(self.label)
        return self

    def __exit__(self, *args: object) -> None:
        assert self.gui.container.pop() == self.label


class _Gui:
    """Records every control a panel creates, keyed by label."""

    def __init__(self) -> None:
        self.controls: dict[str, _Control] = {}
        self.folders: dict[str, _Folder] = {}
        self.markdowns: list[_Control] = []
        self.container: list[str] = []
        self.creation_order: list[tuple[str, str]] = []

    def _record(self, kind: str, control: _Control) -> _Control:
        self.controls[control.label] = control
        self.creation_order.append((kind, control.label))
        control.container = self.container[-1] if self.container else None
        return control

    def reset(self) -> None:
        self.controls.clear()
        self.folders.clear()
        self.markdowns.clear()
        self.container.clear()
        self.creation_order.clear()

    def add_markdown(self, content: str) -> _Control:
        control = _Control(f"markdown_{len(self.markdowns)}", content=content)
        self.markdowns.append(control)
        return self._record("markdown", control)

    def add_folder(self, label: str, **kwargs: object) -> _Folder:
        folder = _Folder(label, gui=self, **kwargs)
        self.folders[label] = folder
        self._record("folder", folder)
        return folder

    def add_dropdown(self, label: str, options: list[str], initial_value: str):
        return self._record(
            "dropdown", _Control(label, options=list(options), value=initial_value)
        )

    def add_button(self, label: str, **kwargs: object) -> _Control:
        return self._record("button", _Control(label, value=False, **kwargs))

    def add_number(self, label: str, *, initial_value: float, **kwargs: object):
        return self._record("number", _Control(label, value=initial_value, **kwargs))

    def add_slider(self, label: str, *, initial_value: float, **kwargs: object):
        return self._record("slider", _Control(label, value=initial_value, **kwargs))

    def add_checkbox(self, label: str, initial_value: bool) -> _Control:
        return self._record("checkbox", _Control(label, value=initial_value))

    def add_image(self, image: object, **kwargs: object) -> _Control:
        return self._record("image", _Control("image"))


def _client_id(event: object) -> str | None:
    return getattr(event, "client_id", None)


def _build_context(gui: _Gui, emitted: list[object]) -> PanelBuildContext:
    def emit(value: object, *, event: object | None = None) -> None:
        emitted.append(value)

    return PanelBuildContext(
        gui=gui,
        emit=emit,
        client_id=_client_id,
        run_id="run",
        scene_revision=1,
        allow_commands=True,
    )


def _card(
    card_id: str,
    skill_id: str = "move_end_effector",
    **kwargs: object,
) -> SkillCard:
    return SkillCard(card_id=card_id, skill_id=skill_id, **kwargs)


def _view(
    cards: tuple[SkillCard, ...],
    *,
    compiled: bool = False,
    waypoints: int = 0,
    playback: PreviewPlaybackState | None = None,
    selected_entity_uid: str | None = None,
    status: str = "",
) -> PanelViewState:
    return PanelViewState(
        snapshot=SequenceSnapshot(
            cards=cards,
            compiled=compiled,
            trajectory_waypoint_count=waypoints,
        ),
        playback=playback,
        selected_entity_uid=selected_entity_uid,
        status=status,
    )


def _built_panel(
    cfg: SkillSequencePanelCfg | None = None,
) -> tuple[SkillSequencePanel, _Gui, list[object]]:
    panel = SkillSequencePanel(cfg)
    gui = _Gui()
    emitted: list[object] = []
    panel.build(_build_context(gui, emitted))
    return panel, gui, emitted


# ----------------------------------------------------------------------------
# Display model
# ----------------------------------------------------------------------------


class TestPanelDisplayModel:
    """Pure snapshot-to-display-row mapping."""

    def test_rows_expose_all_five_card_states(self) -> None:
        cards = (
            _card("a", "pick_up"),
            _card(
                "b",
                params={"position": TARGET_POSITION},
                state=SkillCardState.READY,
            ),
            _card(
                "c",
                params={"position": TARGET_POSITION},
                state=SkillCardState.RUNNING,
                segment_start=0,
                segment_stop=8,
            ),
            _card(
                "d",
                params={"position": TARGET_POSITION},
                state=SkillCardState.SUCCEEDED,
            ),
            _card(
                "e",
                state=SkillCardState.FAILED,
                failure_message="IK_FAILED; no solution",
            ),
        )
        rows = card_rows(_view(cards))

        assert [row.state for row in rows] == [
            SkillCardState.UNCONFIGURED,
            SkillCardState.READY,
            SkillCardState.RUNNING,
            SkillCardState.SUCCEEDED,
            SkillCardState.FAILED,
        ]
        assert len({row.marker for row in rows}) == 5
        assert len({row.color for row in rows}) == 5
        assert rows[0].detail == "Needs a target entity."
        assert rows[2].detail == "waypoints 0–8 (8)"
        assert rows[4].detail == "IK_FAILED; no solution"
        assert [row.index for row in rows] == [0, 1, 2, 3, 4]

    def test_rows_mark_the_selected_and_previewed_cards(self) -> None:
        cards = (_card("a"), _card("b"))
        playback = PreviewPlaybackState(
            group_id="preview",
            length=PREVIEW_LENGTH,
            cursor=PREVIEW_CURSOR,
            playing=True,
            loop=True,
            step_stride=1,
            active_card_id="b",
        )

        rows = card_rows(_view(cards, playback=playback), selected_card_id="a")

        assert (rows[0].selected, rows[0].active) == (True, False)
        assert (rows[1].selected, rows[1].active) == (False, True)

    def test_markdown_renders_states_without_rich_text(self) -> None:
        cards = (
            _card("a", "pick_up", entity_uid="cube", state=SkillCardState.READY),
            _card(
                "b",
                state=SkillCardState.FAILED,
                failure_message="planning failed",
            ),
        )
        rows = card_rows(_view(cards), selected_card_id="a")

        text = render_cards_markdown(rows)

        assert "1. pick_up · cube" in text
        assert "2. move_end_effector · no target" in text
        assert "planning failed" in text
        assert text.startswith("▸ ")
        assert render_cards_markdown(()) == "_No skill cards yet._"

    def test_summary_and_preview_lines_report_compilation(self) -> None:
        view = _view(
            (_card("a"),),
            compiled=True,
            waypoints=PREVIEW_LENGTH,
            status="Compiled 24 waypoints from 1 cards.",
            playback=PreviewPlaybackState(
                group_id="preview",
                length=PREVIEW_LENGTH,
                cursor=PREVIEW_CURSOR,
                playing=False,
                loop=True,
                step_stride=1,
                active_card_id="a",
            ),
        )

        summary = render_summary_markdown(view)
        preview = render_preview_markdown(view)

        assert "**Compiled:** yes" in summary
        assert f"**Waypoints:** {PREVIEW_LENGTH}" in summary
        assert "Compiled 24 waypoints" in summary
        assert "paused" in preview
        assert f"frame {PREVIEW_CURSOR + 1}/{PREVIEW_LENGTH}" in preview
        assert render_preview_markdown(_view(())) == (
            "_Preview playback is not available._"
        )


# ----------------------------------------------------------------------------
# Command emission
# ----------------------------------------------------------------------------


class TestPanelCommandEmission:
    """Every browser interaction becomes an immutable command value."""

    def test_add_uses_the_skill_dropdown(self) -> None:
        panel, gui, emitted = _built_panel()

        gui.controls["Skill"].value = "place"
        gui.controls["Add card"].fire()

        assert emitted == [AddCard(skill_id="place")]

    def test_remove_and_reorder_target_the_selected_card(self) -> None:
        panel, gui, emitted = _built_panel()
        panel.apply_state(_view((_card("a"), _card("b"), _card("c"))))

        gui.controls["Selected card"].fire(value="b")
        gui.controls["Remove card"].fire()
        gui.controls["Move up"].fire()
        gui.controls["Move down"].fire()

        assert emitted == [
            RemoveCard(card_id="b"),
            MoveCard(card_id="b", new_index=0),
            MoveCard(card_id="b", new_index=2),
        ]

    def test_reorder_is_silent_at_the_sequence_bounds(self) -> None:
        panel, gui, emitted = _built_panel()
        panel.apply_state(_view((_card("a"), _card("b"))))

        gui.controls["Selected card"].fire(value="a")
        gui.controls["Move up"].fire()
        gui.controls["Selected card"].fire(value="b")
        gui.controls["Move down"].fire()

        assert emitted == []

    def test_edit_commands_require_a_card(self) -> None:
        panel, gui, emitted = _built_panel()
        panel.apply_state(_view((), selected_entity_uid="cube"))

        gui.controls["Remove card"].fire()
        gui.controls["Bind selected entity"].fire()
        gui.controls["Apply target position"].fire()

        assert emitted == []

    def test_bind_entity_and_position_emit_update_commands(self) -> None:
        panel, gui, emitted = _built_panel()
        panel.apply_state(_view((_card("a", "pick_up"),), selected_entity_uid="cube"))

        gui.controls["Bind selected entity"].fire()
        for axis, value in zip("xyz", TARGET_POSITION):
            gui.controls[f"Target {axis} (m)"].value = value
        gui.controls["Apply target position"].fire()

        assert emitted[0] == UpdateCard(card_id="a", entity_uid="cube")
        assert emitted[1] == UpdateCard(
            card_id="a", params={"position": TARGET_POSITION}
        )

    def test_run_buttons_emit_compile_and_execute(self) -> None:
        panel, gui, emitted = _built_panel(SkillSequencePanelCfg(execute_hold_steps=4))

        gui.controls["Compile sequence"].fire()
        gui.controls["Execute sequence"].fire()

        assert emitted == [CompileSequence(), ExecuteSequence(hold_steps=4)]

    def test_preview_controls_emit_playback_commands(self) -> None:
        panel, gui, emitted = _built_panel()
        panel.apply_state(
            _view(
                (_card("a"),),
                compiled=True,
                waypoints=PREVIEW_LENGTH,
                playback=PreviewPlaybackState(
                    group_id="preview",
                    length=PREVIEW_LENGTH,
                    cursor=0,
                    playing=False,
                    loop=True,
                    step_stride=1,
                ),
            )
        )

        gui.controls["Play / Pause"].fire()
        gui.controls["Step backward"].fire()
        gui.controls["Step forward"].fire()
        gui.controls["Frame"].fire(value=PREVIEW_CURSOR)

        assert emitted == [
            TogglePreviewPlayback(),
            StepPreview(delta=-1),
            StepPreview(delta=1),
            SeekPreview(index=PREVIEW_CURSOR),
        ]

    def test_seek_ignores_server_side_synchronization(self) -> None:
        panel, gui, emitted = _built_panel()

        gui.controls["Frame"].fire(value=3, client_id=None)

        assert emitted == []


# ----------------------------------------------------------------------------
# State application
# ----------------------------------------------------------------------------


class TestPanelStateApplication:
    """Published states are rendered into the panel's existing handles."""

    def test_state_updates_markdown_and_card_options(self) -> None:
        panel, gui, _ = _built_panel()

        panel.apply_state(
            _view(
                (_card("a", "pick_up", entity_uid="cube"), _card("b")),
                selected_entity_uid="cube",
                status="Picked entity 'cube'.",
            )
        )

        assert gui.controls["Selected card"].options == ["a", "b"]
        assert gui.controls["Selected card"].value == "a"
        assert panel.selected_card_id == "a"
        assert "1. pick_up · cube" in gui.markdowns[1].content
        assert "Picked entity" in gui.markdowns[0].content
        assert "`cube`" in gui.markdowns[2].content

    def test_selection_falls_back_when_the_card_disappears(self) -> None:
        panel, gui, _ = _built_panel()
        panel.apply_state(_view((_card("a"), _card("b"))))
        gui.controls["Selected card"].fire(value="b")

        panel.apply_state(_view((_card("a"),)))

        assert panel.selected_card_id == "a"
        assert gui.controls["Selected card"].value == "a"
        panel.apply_state(_view(()))
        assert panel.selected_card_id is None
        assert gui.controls["Selected card"].options == ["—"]

    def test_frame_slider_is_rebuilt_only_when_the_length_changes(self) -> None:
        panel, gui, emitted = _built_panel()
        first = gui.controls["Frame"]

        def publish(length: int, cursor: int) -> None:
            panel.apply_state(
                _view(
                    (_card("a"),),
                    compiled=length > 0,
                    waypoints=length,
                    playback=PreviewPlaybackState(
                        group_id="preview",
                        length=length,
                        cursor=cursor,
                        playing=True,
                        loop=True,
                        step_stride=1,
                    ),
                )
            )

        publish(PREVIEW_LENGTH, 0)
        rebuilt = gui.controls["Frame"]
        publish(PREVIEW_LENGTH, PREVIEW_CURSOR)

        assert first.removed
        assert rebuilt is not first
        assert rebuilt.max == PREVIEW_LENGTH - 1
        assert gui.controls["Frame"] is rebuilt
        assert rebuilt.value == PREVIEW_CURSOR
        assert emitted == []
        assert gui.creation_order[-1] == ("slider", "Frame")

    def test_unknown_states_are_ignored(self) -> None:
        panel, gui, _ = _built_panel()
        panel.apply_state(_view((_card("a"),)))

        panel.apply_state("not a view state")

        assert panel.view is not None
        assert panel.view.snapshot.cards[0].card_id == "a"

    def test_preview_controls_can_be_disabled(self) -> None:
        panel, gui, _ = _built_panel(SkillSequencePanelCfg(show_preview_controls=False))

        assert "Frame" not in gui.controls
        assert "Play / Pause" not in gui.controls

    def test_read_only_sessions_disable_mutating_controls(self) -> None:
        panel = SkillSequencePanel()
        gui = _Gui()
        emitted: list[object] = []
        context = PanelBuildContext(
            gui=gui,
            emit=lambda value, *, event=None: emitted.append(value),
            client_id=_client_id,
            run_id="run",
            scene_revision=0,
            allow_commands=False,
        )

        panel.build(context)

        assert gui.controls["Add card"].disabled
        assert gui.controls["Target x (m)"].disabled
        assert gui.controls["Frame"].disabled


# ----------------------------------------------------------------------------
# Generic backend panel registry
# ----------------------------------------------------------------------------


class _Scene:
    def set_up_direction(self, direction: str) -> None:
        self.up_direction = direction

    def add_frame(self, name: str, **kwargs: object) -> _Control:
        return _Control(name, **kwargs)

    def add_grid(self, name: str, **kwargs: object) -> _Control:
        return _Control(name, **kwargs)

    def on_pointer_event(self, event_type: str):
        def decorator(callback: object) -> object:
            return callback

        return decorator


class _Server:
    def __init__(self, **kwargs: object) -> None:
        self.scene = _Scene()
        self.gui = _Gui()
        self.stopped = False

    def get_port(self) -> int:
        return 8765

    def get_clients(self) -> dict[str, object]:
        return {}

    def flush(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True


def _started_backend(**kwargs: object) -> tuple[ViserBackend, _Server]:
    server = _Server()
    backend = ViserBackend(
        ViserServerCfg(port=8765),
        server_factory=lambda **_: server,
        **kwargs,
    )
    backend.start()
    return backend, server


class TestBackendPanelRegistry:
    """The Viser backend hosts panels without knowing what they mean."""

    def test_no_panel_is_registered_by_default(self) -> None:
        backend, server = _started_backend()

        backend.publish_manifest(SceneManifest("run", 1, (), ()))

        assert server.gui.folders.keys() == {"Environments", "Overlays"}

    def test_registered_panel_is_built_and_rebuilt_per_manifest(self) -> None:
        backend, server = _started_backend()
        builds: list[PanelBuildContext] = []
        spec = PanelSpec(
            panel_id="demo",
            build=builds.append,
            title="Demo panel",
        )

        backend.publish_manifest(SceneManifest("run", 1, (), ()))
        backend.register_panel(spec)
        backend.publish_manifest(SceneManifest("run", 2, (), ()))

        assert len(builds) == 2
        assert builds[0].run_id == "run"
        assert [context.scene_revision for context in builds] == [1, 2]
        assert "Demo panel" in server.gui.folders

    def test_panel_emission_reaches_the_command_sink(self) -> None:
        backend, server = _started_backend(allow_commands=True)
        commands: list[PanelCommand] = []
        backend.set_panel_command_sink(commands.append)
        emitters: list[object] = []

        def build(context: PanelBuildContext) -> None:
            emitters.append(context.emit)

        backend.register_panel(PanelSpec(panel_id="demo", build=build))
        backend.publish_manifest(SceneManifest("run", 3, (), ()))
        emitters[-1](("hello", 1), event=SimpleNamespace(client_id="browser-7"))
        emitters[-1]("plain")
        backend.poll()

        assert [command.value for command in commands] == [("hello", 1), "plain"]
        assert commands[0].panel_id == "demo"
        assert commands[0].client_id == "browser-7"
        assert commands[1].client_id == "unknown"
        assert commands[0].scene_revision == 3
        assert [command.sequence for command in commands] == [1, 2]

    def test_panel_state_is_replayed_after_a_rebuild(self) -> None:
        backend, server = _started_backend()
        states: list[object] = []
        backend.register_panel(
            PanelSpec(
                panel_id="demo",
                build=lambda context: None,
                apply_state=states.append,
                title="Demo panel",
            )
        )
        backend.publish_manifest(SceneManifest("run", 1, (), ()))

        backend.publish_panel_state("demo", "first")
        backend.publish_manifest(SceneManifest("run", 2, (), ()))
        backend.publish_panel_state("unknown", "ignored")

        assert states == ["first", "first"]

    def test_unregistering_removes_the_owned_container(self) -> None:
        backend, server = _started_backend()
        backend.publish_manifest(SceneManifest("run", 1, (), ()))
        backend.register_panel(
            PanelSpec(panel_id="demo", build=lambda context: None, title="Demo panel")
        )
        folder = server.gui.folders["Demo panel"]

        backend.unregister_panel("demo")
        backend.publish_manifest(SceneManifest("run", 2, (), ()))

        assert folder.removed
        assert "Demo panel" not in server.gui.folders

    def test_panel_command_queue_keeps_arrival_order(self) -> None:
        queue = PanelCommandQueue(maxsize=2)
        for index in range(3):
            queue.put(
                PanelCommand(
                    run_id="run",
                    scene_revision=0,
                    sequence=index,
                    panel_id="demo",
                    client_id="client",
                    value=index,
                )
            )

        drained = queue.drain()

        assert [command.value for command in drained] == [1, 2]
        assert queue.drain() == ()

    def test_panel_command_queue_drains_one_panel_without_swallowing_others(
        self,
    ) -> None:
        queue = PanelCommandQueue()
        for index, panel_id in enumerate(("left", "right", "left", "right")):
            queue.put(_queued_panel_command(panel_id, index))

        assert [command.value for command in queue.drain("left")] == [0, 2]
        assert queue.drain("left") == ()
        assert [command.value for command in queue.drain("right")] == [1, 3]

    def test_panel_command_queue_drains_every_panel_in_arrival_order(self) -> None:
        queue = PanelCommandQueue()
        for index, panel_id in enumerate(("left", "right", "left", "right")):
            queue.put(_queued_panel_command(panel_id, index))

        assert [command.value for command in queue.drain()] == [0, 1, 2, 3]
        assert queue.drain() == ()
        assert queue.drain("left") == ()

    def test_panel_command_rejects_mutable_payloads(self) -> None:
        with pytest.raises(TypeError, match="immutable"):
            PanelCommand(
                run_id="run",
                scene_revision=0,
                sequence=0,
                panel_id="demo",
                client_id="client",
                value=["mutable"],
            )
        with pytest.raises(ValueError, match="panel_id"):
            PanelCommand(
                run_id="run",
                scene_revision=0,
                sequence=0,
                panel_id="",
                client_id="client",
                value=None,
            )

    def test_panel_spec_validates_its_callbacks(self) -> None:
        with pytest.raises(ValueError, match="panel_id"):
            PanelSpec(panel_id="", build=lambda context: None)
        with pytest.raises(TypeError, match="build"):
            PanelSpec(panel_id="demo", build="not callable")
        with pytest.raises(TypeError, match="apply_state"):
            PanelSpec(panel_id="demo", build=lambda c: None, apply_state=3)


# ----------------------------------------------------------------------------
# Runtime plumbing
# ----------------------------------------------------------------------------


class _ManifestExporter:
    """Scene exporter stub producing empty manifests."""

    scene_revision = 0
    run_id = "run"

    @property
    def has_cameras(self) -> bool:
        return False

    @property
    def has_deformables(self) -> bool:
        return False

    def build_manifest(self) -> SceneManifest:
        self.scene_revision += 1
        return SceneManifest("run", self.scene_revision, (), ())


class _RecordingBackend(VisualizationBackend):
    """Backend stub recording panel calls and the thread that made them."""

    def __init__(self) -> None:
        self.registered: list[tuple[str, int]] = []
        self.unregistered: list[str] = []
        self.states: list[tuple[str, object]] = []
        self.applied = threading.Event()

    @property
    def endpoint(self) -> str:
        return "http://localhost:1234"

    @property
    def client_count(self) -> int:
        return 0

    def start(self) -> None:
        pass

    def publish_manifest(self, manifest: SceneManifest) -> None:
        pass

    def publish_frame(self, frame: object) -> bool:
        return True

    def publish_camera_images(self, frame: object) -> bool:
        return True

    def register_panel(self, spec: PanelSpec) -> None:
        self.registered.append((spec.panel_id, threading.get_ident()))
        self.applied.set()

    def unregister_panel(self, panel_id: str) -> None:
        self.unregistered.append(panel_id)
        self.applied.set()

    def publish_panel_state(self, panel_id: str, state: object) -> None:
        self.states.append((panel_id, state))
        self.applied.set()

    def poll(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _wait(backend: _RecordingBackend) -> None:
    assert backend.applied.wait(timeout=5.0)
    backend.applied.clear()


class TestRuntimePanelChannel:
    """Panel registration, state, and commands cross the worker thread."""

    def test_registration_and_state_reach_the_backend(self) -> None:
        backend = _RecordingBackend()
        runtime = VisualizationRuntime(
            _ManifestExporter(),
            VisualizationCfg(backend="viser", allow_commands=True),
            backend=backend,
        )
        spec = PanelSpec(panel_id="demo", build=lambda context: None)

        runtime.register_panel(spec)
        runtime.start()
        _wait(backend)
        runtime.publish_panel_state("demo", "state-1")
        _wait(backend)
        runtime.unregister_panel("demo")
        _wait(backend)
        runtime.stop()

        assert [panel_id for panel_id, _ in backend.registered] == ["demo"]
        assert backend.registered[0][1] != threading.get_ident()
        assert backend.states == [("demo", "state-1")]
        assert backend.unregistered == ["demo"]

    def test_commands_are_drained_only_when_commands_are_allowed(self) -> None:
        for allow_commands, expected in ((True, 1), (False, 0)):
            backend = _RecordingBackend()
            runtime = VisualizationRuntime(
                _ManifestExporter(),
                VisualizationCfg(backend="viser", allow_commands=allow_commands),
                backend=backend,
            )
            backend._panel_command_sink(
                PanelCommand(
                    run_id="run",
                    scene_revision=1,
                    sequence=1,
                    panel_id="demo",
                    client_id="client",
                    value="clicked",
                )
            )

            assert len(runtime.drain_panel_commands()) == expected


# ----------------------------------------------------------------------------
# Session bridge
# ----------------------------------------------------------------------------


class _Exporter:
    run_id = "run"
    scene_revision = 1

    def __init__(self, targets: dict[str, tuple[str, str]] | None = None) -> None:
        self._targets = targets or {}

    def resolve_node_target(self, node_id: str) -> tuple[str, str] | None:
        return self._targets.get(node_id)


class _Runtime:
    """Visualization runtime stub exposing only the bridge's dependencies."""

    def __init__(self, exporter: _Exporter | None = None) -> None:
        self.exporter = exporter or _Exporter()
        self.panel_commands: list[PanelCommand] = []
        self.pick_commands: list[object] = []
        self.published: list[tuple[str, object]] = []
        self.registered: list[PanelSpec] = []
        self.unregistered: list[str] = []

    def register_panel(self, spec: PanelSpec) -> None:
        self.registered.append(spec)

    def unregister_panel(self, panel_id: str) -> None:
        self.unregistered.append(panel_id)

    def publish_panel_state(self, panel_id: str, state: object) -> None:
        self.published.append((panel_id, state))

    def drain_panel_commands(
        self,
        panel_id: str | None = None,
    ) -> tuple[PanelCommand, ...]:
        if panel_id is None:
            commands = tuple(self.panel_commands)
            self.panel_commands.clear()
            return commands
        commands = tuple(
            command for command in self.panel_commands if command.panel_id == panel_id
        )
        self.panel_commands = [
            command for command in self.panel_commands if command.panel_id != panel_id
        ]
        return commands

    def drain_pick_commands(self) -> tuple[object, ...]:
        commands = tuple(self.pick_commands)
        self.pick_commands.clear()
        return commands

    def sim_update(self) -> None:
        """Mimic ``SimulationManager.update`` draining the shared pick queue."""
        self.pick_commands.clear()

    def queue(
        self,
        value: object,
        panel_id: str = "skill_sequence",
        *,
        run_id: str = "run",
        scene_revision: int = 1,
    ) -> None:
        self.panel_commands.append(
            PanelCommand(
                run_id=run_id,
                scene_revision=scene_revision,
                sequence=len(self.panel_commands) + 1,
                panel_id=panel_id,
                client_id="client",
                value=value,
            )
        )


class _Preview:
    """Preview driver stub recording the playback calls it receives."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.playback = PreviewPlaybackState(
            group_id="preview",
            length=PREVIEW_LENGTH,
            cursor=0,
            playing=False,
            loop=True,
            step_stride=1,
        )

    def state(self) -> PreviewPlaybackState:
        return self.playback

    def toggle(self) -> bool:
        self.calls.append(("toggle", 0))
        return True

    def seek(self, index: int) -> int:
        self.calls.append(("seek", index))
        return index

    def step(self, delta: int) -> int:
        self.calls.append(("step", delta))
        return delta


def _pick(node_id: str | None, revision: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        run_id="run",
        scene_revision=revision,
        client_id="client",
        node_id=node_id,
    )


def _bridge(
    runtime: _Runtime | None = None,
    preview: _Preview | None = None,
) -> tuple[AuthoringBridge, AuthoringSession, _Runtime]:
    session = AuthoringSession(
        robot=SimpleNamespace(),
        engine=SimpleNamespace(),
        sim=SimpleNamespace(),
        control_parts={"motion": "arm", "grasp": "hand"},
    )
    runtime = runtime or _Runtime()
    bridge = AuthoringBridge(session, runtime, SkillSequencePanel(), preview)
    return bridge, session, runtime


class TestAuthoringBridge:
    """Panel commands reach the session and new snapshots reach the panel."""

    def test_register_publishes_the_initial_state(self) -> None:
        bridge, _, runtime = _bridge()

        bridge.register()

        assert runtime.registered[0].panel_id == "skill_sequence"
        panel_id, state = runtime.published[0]
        assert panel_id == "skill_sequence"
        assert state.snapshot == SequenceSnapshot(cards=(), compiled=False)

    def test_commands_reach_the_session_and_produce_a_new_snapshot(self) -> None:
        bridge, session, runtime = _bridge()
        bridge.register()

        runtime.queue(AddCard(skill_id="pick_up", card_id="a"))
        runtime.queue(AddCard(skill_id="move_end_effector", card_id="b"))
        runtime.queue(UpdateCard(card_id="a", entity_uid="cube"))
        runtime.queue(MoveCard(card_id="b", new_index=0))
        view = bridge.update()

        assert [card.card_id for card in session.cards] == ["b", "a"]
        assert view.snapshot.cards[1].entity_uid == "cube"
        assert view.snapshot.cards[1].state is SkillCardState.READY
        assert runtime.published[-1][1] == view

    def test_states_are_published_only_when_they_change(self) -> None:
        bridge, _, runtime = _bridge()
        bridge.register()

        bridge.update()
        published_after_noop = len(runtime.published)
        runtime.queue(AddCard(skill_id="place", card_id="a"))
        bridge.update()

        assert published_after_noop == 1
        assert len(runtime.published) == 2

    def test_failures_become_status_text_instead_of_exceptions(self) -> None:
        bridge, _, runtime = _bridge()
        bridge.register()

        runtime.queue(RemoveCard(card_id="missing"))
        view = bridge.update()

        assert "RemoveCard failed" in view.status
        assert "missing" in view.status

    def test_clearing_the_status_reaches_the_panel(self) -> None:
        bridge, _, runtime = _bridge()
        bridge.register()
        runtime.queue(RemoveCard(card_id="missing"))
        bridge.update()

        bridge.clear_status()
        view = bridge.update()

        assert view.status == ""
        assert runtime.published[-1][1].status == ""

    def test_compile_failure_is_reported_without_raising(self) -> None:
        bridge, _, runtime = _bridge()
        bridge.register()

        runtime.queue(CompileSequence())
        view = bridge.update()

        assert "CompileSequence failed" in view.status
        assert view.snapshot.compiled is False

    def test_commands_for_other_panels_are_ignored(self) -> None:
        bridge, session, runtime = _bridge()
        bridge.register()

        runtime.queue(AddCard(skill_id="place"), panel_id="other")
        bridge.update()

        assert session.cards == ()

    def test_stale_panel_commands_are_dropped(self) -> None:
        bridge, session, runtime = _bridge()
        bridge.register()

        runtime.queue(AddCard(skill_id="place", card_id="old"), scene_revision=99)
        runtime.queue(AddCard(skill_id="place", card_id="other_run"), run_id="stale")
        bridge.update()

        assert session.cards == ()

        runtime.queue(AddCard(skill_id="place", card_id="fresh"))
        bridge.update()

        assert [card.card_id for card in session.cards] == ["fresh"]

    def test_unknown_command_values_are_reported(self) -> None:
        bridge, _, runtime = _bridge()
        bridge.register()

        runtime.queue("nonsense")
        view = bridge.update()

        assert "Ignored unknown panel command" in view.status

    def test_preview_commands_drive_the_preview_driver(self) -> None:
        preview = _Preview()
        bridge, _, runtime = _bridge(preview=preview)
        bridge.register()

        runtime.queue(TogglePreviewPlayback())
        runtime.queue(SeekPreview(index=PREVIEW_CURSOR))
        runtime.queue(StepPreview(delta=-1))
        view = bridge.update()

        assert preview.calls == [
            ("toggle", 0),
            ("seek", PREVIEW_CURSOR),
            ("step", -1),
        ]
        assert view.playback == preview.playback

    def test_preview_commands_without_a_preview_are_reported(self) -> None:
        bridge, _, runtime = _bridge()
        bridge.register()

        runtime.queue(TogglePreviewPlayback())
        view = bridge.update()

        assert view.playback is None
        assert "not available" in view.status

    def test_picks_select_rigid_entities_only(self) -> None:
        exporter = _Exporter(
            {
                "env:0/rigid:cube": ("cube", "rigid"),
                "env:0/robot:arm": ("arm", "robot"),
            }
        )
        runtime = _Runtime(exporter)
        bridge, _, _ = _bridge(runtime)
        bridge.register()

        runtime.pick_commands.append(_pick("env:0/rigid:cube"))
        assert bridge.update().selected_entity_uid == "cube"

        runtime.pick_commands.append(_pick("env:0/robot:arm"))
        view = bridge.update()
        assert view.selected_entity_uid == "cube"
        assert "cannot be a skill target" in view.status

        runtime.pick_commands.append(_pick(None))
        assert bridge.update().selected_entity_uid is None

    def test_stale_picks_are_dropped(self) -> None:
        exporter = _Exporter({"env:0/rigid:cube": ("cube", "rigid")})
        runtime = _Runtime(exporter)
        bridge, _, _ = _bridge(runtime)
        bridge.register()

        runtime.pick_commands.append(_pick("env:0/rigid:cube", revision=99))
        view = bridge.update()

        assert view.selected_entity_uid is None

    def test_picks_reach_the_bridge_when_drained_before_sim_update(self) -> None:
        """A host loop stepping a simulation must drain picks first.

        ``SimulationManager.update`` drains the same runtime pick queue through
        its Gizmo processing, so a bridge running after it observes nothing.
        """
        exporter = _Exporter({"env:0/rigid:cube": ("cube", "rigid")})
        runtime = _Runtime(exporter)
        bridge, _, _ = _bridge(runtime)
        bridge.register()

        # Tutorial ordering: drain_picks() -> sim.update() -> bridge.update().
        runtime.pick_commands.append(_pick("env:0/rigid:cube"))
        bridge.drain_picks()
        runtime.sim_update()

        assert bridge.update().selected_entity_uid == "cube"

        # The reversed ordering is exactly what the tutorial must not do.
        late_runtime = _Runtime(_Exporter({"env:0/rigid:cube": ("cube", "rigid")}))
        late_bridge, _, _ = _bridge(late_runtime)
        late_bridge.register()

        late_runtime.pick_commands.append(_pick("env:0/rigid:cube"))
        late_runtime.sim_update()

        assert late_bridge.update().selected_entity_uid is None

    def test_bridges_sharing_a_runtime_keep_their_own_commands(self) -> None:
        backend = _RecordingBackend()
        runtime = VisualizationRuntime(
            _ManifestExporter(),
            VisualizationCfg(backend="viser", allow_commands=True),
            backend=backend,
        )
        bridges: dict[str, tuple[AuthoringBridge, AuthoringSession]] = {}
        for panel_id in ("left", "right"):
            session = AuthoringSession(
                robot=SimpleNamespace(),
                engine=SimpleNamespace(),
                sim=SimpleNamespace(),
                control_parts={"motion": "arm", "grasp": "hand"},
            )
            bridge = AuthoringBridge(
                session,
                runtime,
                SkillSequencePanel(SkillSequencePanelCfg(panel_id=panel_id)),
            )
            bridge.register()
            bridges[panel_id] = (bridge, session)
        for sequence, (panel_id, card_id) in enumerate((("left", "a"), ("right", "b"))):
            backend._panel_command_sink(
                PanelCommand(
                    run_id="run",
                    scene_revision=0,
                    sequence=sequence,
                    panel_id=panel_id,
                    client_id="client",
                    value=AddCard(skill_id="pick_up", card_id=card_id),
                )
            )

        left_bridge, left_session = bridges["left"]
        right_bridge, right_session = bridges["right"]
        left_bridge.update()

        assert [card.card_id for card in left_session.cards] == ["a"]
        assert right_session.cards == ()

        right_bridge.update()

        assert [card.card_id for card in right_session.cards] == ["b"]
        assert [card.card_id for card in left_session.cards] == ["a"]

    def test_picks_can_be_left_to_another_consumer(self) -> None:
        exporter = _Exporter({"env:0/rigid:cube": ("cube", "rigid")})
        runtime = _Runtime(exporter)
        bridge = AuthoringBridge(
            AuthoringSession(
                robot=SimpleNamespace(),
                engine=SimpleNamespace(),
                sim=SimpleNamespace(),
                control_parts={"motion": "arm"},
            ),
            runtime,
            SkillSequencePanel(),
            process_picks=False,
        )
        bridge.register()

        runtime.pick_commands.append(_pick("env:0/rigid:cube"))
        view = bridge.update()

        assert view.selected_entity_uid is None
        assert len(runtime.pick_commands) == 1

    def test_unregister_removes_the_panel(self) -> None:
        bridge, _, runtime = _bridge()
        bridge.register()

        bridge.unregister()

        assert runtime.unregistered == ["skill_sequence"]


class TestPanelBridgeRoundTrip:
    """A browser click ends up rendered in the panel it came from."""

    def test_button_click_updates_the_rendered_card_list(self) -> None:
        runtime = _Runtime()
        session = AuthoringSession(
            robot=SimpleNamespace(),
            engine=SimpleNamespace(),
            sim=SimpleNamespace(),
            control_parts={"motion": "arm", "grasp": "hand"},
        )
        panel = SkillSequencePanel()
        bridge = AuthoringBridge(session, runtime, panel)
        gui = _Gui()
        emitted: list[object] = []
        panel.build(_build_context(gui, emitted))
        bridge.register()
        panel.apply_state(runtime.published[-1][1])

        gui.controls["Skill"].value = "pick_up"
        gui.controls["Add card"].fire()
        for value in emitted:
            runtime.queue(value)
        bridge.update()
        panel.apply_state(runtime.published[-1][1])

        assert len(session.cards) == 1
        assert "1. pick_up · no target" in gui.markdowns[1].content
        assert "🟡" in gui.markdowns[1].content
        assert gui.controls["Selected card"].options == [session.cards[0].card_id]
