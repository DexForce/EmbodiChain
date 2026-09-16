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

"""Browser skill-sequence authoring: protocol, session, preview, and panel.

The protocol layer defines immutable card, snapshot, and command values that
are safe to share between the simulation thread and browser-facing UI code.
:class:`AuthoringSession` lives on the simulation thread, compiles configured
cards into atomic-action trajectories, and executes them.
:class:`SequencePreview` replays a compiled trajectory on a translucent preview
robot without mutating simulation state. :class:`SkillSequencePanel` renders the
sequence in the browser sidebar and emits commands, and :class:`AuthoringBridge`
applies those commands on the simulation thread and publishes new panel states.
:class:`StepwiseExecution` replays a compiled sequence under host-loop control
so a browser keeps receiving card-state updates while the robot is moving.
"""

from __future__ import annotations

from .bridge import AuthoringBridge
from .execution import StepwiseExecution
from .panel import (
    CardRow,
    PanelViewState,
    PreviewCommand,
    SeekPreview,
    SkillSequencePanel,
    SkillSequencePanelCfg,
    StepPreview,
    TogglePreviewPlayback,
    card_rows,
    render_cards_markdown,
    render_preview_markdown,
    render_summary_markdown,
)
from .preview import PreviewPlaybackCfg, PreviewPlaybackState, SequencePreview
from .protocol import (
    AddCard,
    AuthoringCommand,
    CompileSequence,
    ExecuteSequence,
    ExecutionProgress,
    MoveCard,
    RemoveCard,
    SUPPORTED_SKILL_IDS,
    SequenceSnapshot,
    SkillCard,
    SkillCardState,
    UpdateCard,
    freeze_params,
)
from .session import AuthoringSession

__all__ = [
    "AddCard",
    "AuthoringBridge",
    "AuthoringCommand",
    "AuthoringSession",
    "CardRow",
    "CompileSequence",
    "ExecuteSequence",
    "ExecutionProgress",
    "MoveCard",
    "PanelViewState",
    "PreviewCommand",
    "PreviewPlaybackCfg",
    "PreviewPlaybackState",
    "RemoveCard",
    "SUPPORTED_SKILL_IDS",
    "SeekPreview",
    "SequencePreview",
    "SequenceSnapshot",
    "SkillCard",
    "SkillCardState",
    "SkillSequencePanel",
    "SkillSequencePanelCfg",
    "StepPreview",
    "StepwiseExecution",
    "TogglePreviewPlayback",
    "UpdateCard",
    "card_rows",
    "freeze_params",
    "render_cards_markdown",
    "render_preview_markdown",
    "render_summary_markdown",
]
