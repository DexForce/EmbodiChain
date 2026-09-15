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

"""Backend-neutral registration contract for custom browser side panels.

A custom panel is described by two callbacks. :attr:`PanelSpec.build` creates
the panel's browser controls, and :attr:`PanelSpec.apply_state` pushes one
immutable state object into the controls it created. Both run on the
visualization worker thread, which is the only thread allowed to create or
mutate backend GUI handles.

The backend knows nothing about what a panel means. It calls the two callbacks
at the right times and forwards whatever value the panel emits to the
simulation thread as a
:class:`~embodichain.lab.visualization.protocol.PanelCommand`. Panels therefore
never reach into simulation state directly: they emit immutable values, and a
simulation-thread owner interprets them and publishes new panel states back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

__all__ = [
    "PanelBuildContext",
    "PanelEventSink",
    "PanelSpec",
]


class PanelEventSink(Protocol):
    """Thread-safe callable a panel uses to publish one interaction value."""

    def __call__(self, value: object, *, event: object | None = None) -> None:
        """Queue ``value`` for delivery to the simulation thread.

        Args:
            value: Immutable payload defined by the panel and its owner.
            event: Optional backend GUI event that triggered the interaction.
                The backend uses it to attribute the command to a browser
                client; panels never need to parse it.
        """
        ...


@dataclass(frozen=True)
class PanelBuildContext:
    """Everything a panel needs while building its browser controls.

    Args:
        gui: Backend GUI namespace the panel adds controls to. For the Viser
            backend this is ``viser.ViserServer.gui``; controls added while the
            context is active are placed inside the panel's own container.
        emit: Sink used by control callbacks to publish interaction values.
        client_id: Resolver returning the browser client that produced a GUI
            event, or ``None`` for server-side synchronization callbacks. A
            panel that writes control values from the worker thread uses it to
            ignore the echo of its own writes.
        run_id: Run identifier of the manifest the panel was built for.
        scene_revision: Scene revision the panel was built for.
        allow_commands: Whether browser interaction may mutate the simulation.
            Panels should disable or hide mutating controls when this is false.
    """

    gui: object
    emit: PanelEventSink
    client_id: Callable[[object], str | None]
    run_id: str
    scene_revision: int
    allow_commands: bool = False

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("PanelBuildContext.run_id must not be empty.")
        if self.scene_revision < 0:
            raise ValueError("PanelBuildContext.scene_revision must be non-negative.")


@dataclass(frozen=True)
class PanelSpec:
    """One custom side panel registered on a visualization backend.

    Args:
        panel_id: Unique panel identifier. Re-registering the same identifier
            replaces the previous panel.
        build: Callback invoked on the visualization worker thread to create
            the panel's controls. It runs once per published scene manifest,
            because publishing a manifest resets the browser GUI.
        apply_state: Optional callback invoked on the visualization worker
            thread with the newest immutable panel state. It runs after every
            :meth:`~embodichain.lab.visualization.runtime.VisualizationRuntime.publish_panel_state`
            and once directly after ``build`` when a state is already known.
        title: Optional folder title. When set, the backend wraps the panel in
            its own collapsible container and removes that container when the
            panel is unregistered. When ``None`` the panel builds directly into
            the sidebar and its controls persist until the next manifest.
    """

    panel_id: str
    build: Callable[[PanelBuildContext], None]
    apply_state: Callable[[object], None] | None = None
    title: str | None = None

    def __post_init__(self) -> None:
        if not self.panel_id:
            raise ValueError("PanelSpec.panel_id must not be empty.")
        if not callable(self.build):
            raise TypeError("PanelSpec.build must be callable.")
        if self.apply_state is not None and not callable(self.apply_state):
            raise TypeError("PanelSpec.apply_state must be callable or None.")
        if self.title is not None and not self.title:
            raise ValueError("PanelSpec.title must be None or a non-empty string.")
