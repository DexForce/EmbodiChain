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
"""Compose restricted model inference with prepared Real2Sim evidence."""

from __future__ import annotations

import json
from pathlib import Path

from embodichain.data_pipeline.real2sim.adaptation.evidence import (
    prepare_trajectory_evidence,
)
from embodichain.gen_sim.services.contracts import (
    ChatRequest,
    ChatMessage,
    TextPart,
    ImagePart,
)
from .annotation import bound_trajectory_phases
from ._artifacts import _json_default, write_json
from ._codex import CodexChat

__all__ = ["CodexTrajectoryAgent"]


class CodexTrajectoryAgent:
    """Return semantics and bounded JSON edits; no shell or trajectory-file output."""

    def __init__(self, model: str, reasoning_effort: str) -> None:
        self.profile = dict(
            model=model,
            timeout_s=240,
            options=dict(reasoning_effort=reasoning_effort, disable_shell=True),
        )

    def __call__(self, stage: str, context: dict, output: Path) -> dict:
        if stage not in ("annotate", "review"):
            raise ValueError(f"Unknown agent stage: {stage}")
        output.mkdir(parents=True, exist_ok=True)
        annotation = (
            json.loads(Path(context["annotation"]).read_text())
            if stage == "review"
            else None
        )
        evidence, images = prepare_trajectory_evidence(
            stage,
            context,
            output,
            annotation=annotation,
            phases=bound_trajectory_phases(annotation) if annotation else None,
        )
        common = """You are a restricted semantic reviewer of real robot demonstrations and native physics replay.
Use only the supplied evidence. Do not use tools, write code, output joint positions or trajectory paths.
Task completion requires the requested actions and final object relationships, not pixel matching.
Scene/assets are fixed. Distinguish startup object motion from robot-caused failure. Return one JSON object.
"""
        if stage == "annotate":
            prompt = (
                common
                + """Annotate the REAL source using the existing Task Program JSON language.
Return {"task_program":{"program_id":string,"instruction":string,"targets":{},
"program":{"kind":"sequence","items":[{"kind":"segment","name":string,
"instruction":string,"steps":{"kind":"invoke","call":SEMANTIC_CALL}}]}},
"trajectory_bindings":[{"name":string,"segment":string,"start":integer,"stop":integer,
"kind":"free|contact|hold","reference":"object|target|gravity","acting_arm":string,"instruction":string}]}.
SEMANTIC_CALL is {"kind":"pick","object":ID}, {"kind":"hand_over","object":ID},
or {"kind":"place","object":ID,"on":ID} / {"kind":"place","object":ID,"inside":ID}.
Use observed scene IDs as logical references; on/inside describes the observed destination,
not an executable affordance. Runtime grounding is host-owned. Do not invent scene IDs,
resource bindings, integration selections, controllers or motion commands.
Use one semantic call per uniquely named segment. Bind detailed approach/close/lift/carry/
release/withdrawal phases to the appropriate segment instead of inventing new call kinds.
Binding name is a unique phase ID; segment is an existing program segment name. Bounds are
HALF-OPEN trajectory sample indices [start,stop): cover [0,sample_count) exactly once in
contiguous order. Every segment must be bound, in program order. Use recorded arm names.
One segment may own multiple phases, including different arms during handover.
Separate observed initial/final waiting from active approach or withdrawal. Use motion and
video evidence for phase boundaries; joint-limit diagnostics expose source/model mismatch.
"""
            )
        else:
            prompt = (
                common
                + """Review the actual replay, locate failed semantic phases and request only small residual edits.
Return {"success":boolean,"failure_phase_ids":[string],"explanation":string,"evidence":[string],
"corrections":[{"phase_ids":[string],"side":string,"anchor_sample":integer,
"translation_m":[x,y,z],"rotation_vector_rad":[x,y,z],"gripper_width_delta_m":number,"gripper_shift_s":number}]}.
If successful or revisions_remaining=0, corrections must be []. Otherwise choose up to max_edits
from correction_limits. Each edit selects CONTIGUOUS semantic phases and one interior sample.
The host spreads that world TCP translation/rotation residual over the recorded time grid,
then calls Real2Sim's approximate source-near IK at each sample. Target errors and URDF
limit excess are diagnostics; a finite IK result does not establish successful execution.
phase_ids and failure_phase_ids reference the supplied binding/phase names. Each phase's
segment and call identify its canonical Task Program meaning; do not change the program.
The residual is ZERO at both endpoints and peaks at the anchor.
Original motion and timestamps remain; IK residuals do not certify collision freedom.
Include adjacent phases when an offset must persist across a grasp/carry boundary. Same-arm edit
intervals cannot overlap. Gripper width delta uses the same envelope (negative closes more);
positive gripper_shift_s delays the measured opening profile, negative advances it. Limits apply
to each local edit; cumulative source deviation is reported, not a task-success criterion.
Preserve the recorded motion shape; do not request a replacement motion.
Check source/commanded_joint_limits as source/model diagnostics. Small source limit excess
does not by itself reject a correction. Simulator limits remain unchanged. Keep unrelated
waiting phases out of grasp/carry corrections.
If correction_failure is present, the previous edits were rejected before execution.
Keep success=false and revise parameters once using the reported error and binding ranges.
Use metric evidence for residuals, preserve successful phases. If these limited parameters cannot
resolve the failure, explain that limitation and return []. Do not claim success before replay.
Use contact_geometry to distinguish a finger tip pressing an object from opposing fingers
enclosing it. A downward contact force alone does not determine which translation will help.
Compare commanded and observed arm geometry when a non-hand link contacts the environment.
Do not infer that a blocked TCP needs more motion in the same direction. Coordinated edits on
both arms may preserve a handoff relationship while moving it to an unobstructed location.
"""
            )
        content = [
            TextPart(prompt),
            TextPart(json.dumps(evidence, default=_json_default)),
        ]
        content.extend(ImagePart(path) for path in images)
        response = CodexChat(self.profile, output).complete(
            ChatRequest(messages=(ChatMessage("user", tuple(content)),))
        )
        result = dict(response.structured)
        write_json(output / "result.json", result)
        return result
