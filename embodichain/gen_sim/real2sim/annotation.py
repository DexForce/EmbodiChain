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
"""Source-time annotation and existing Task Engine graph composition."""

from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from embodichain.data_pipeline.real2sim.adaptation.inputs import (
    ReconstructionInput,
    RegisteredScene,
    RefinedTrajectory,
)
from embodichain.data_pipeline.real2sim.adaptation.video_fusion import (
    decode_video_frames,
)
from embodichain.gen_sim.services import GenSimServices
from embodichain.gen_sim.services.contracts import (
    ChatService,
    ChatRequest,
    ChatMessage,
    TextPart,
    ImagePart,
)
from ._artifacts import write_json, digest

__all__ = [
    "ground_gripper_events",
    "annotate_trajectory",
    "bind_graph_intervals",
    "generate_annotated_graph",
    "validate_program_annotation",
    "bound_trajectory_phases",
]


def validate_program_annotation(
    value: dict, count: int, objects: set[str], arms: set[str]
) -> dict:
    """Decode observed Task Program semantics and their separate sample bindings.

    Args:
        value: Agent response containing task_program and trajectory_bindings.
        count: Number of original trajectory samples.
        objects: Known logical scene references, currently using source object IDs.
        arms: Recorded arm names permitted in observation bindings.

    Returns:
        Canonical Task Program JSON plus validated half-open trajectory bindings.
        This is observation annotation, not a compiled deployment. The integration
        and destination affordances must be grounded before Task Program execution.
    """
    from embodichain.agents.mllm.task_program import decode_mllm_task_program
    from embodichain.data_pipeline.real2sim.contracts import PhaseAnnotation
    from embodichain.lab.task_program.language.schema import (
        InvokeCfg,
        SceneRelativePoseTargetCfg,
        SegmentCfg,
        SequenceCfg,
        TaskProgramIntegrationCfg,
    )

    if type(value) is not dict or set(value) != {
        "task_program",
        "trajectory_bindings",
    }:
        raise ValueError(
            "Annotation requires task_program and trajectory_bindings; "
            "legacy phase-only annotations require a fresh annotation run"
        )
    # As in Task Engine's provider-free graph validation, these host-owned
    # references are decoder context only. They are never serialized or claimed
    # to select a real runtime, and no integration/compiler is constructed here.
    config = decode_mllm_task_program(
        json.dumps(value["task_program"], allow_nan=False),
        integration=TaskProgramIntegrationCfg(
            robot_profile="real2sim_observation_only",
            scene_registry="real2sim_observation_only",
            runtime_preset="real2sim_observation_only",
        ),
    )
    if not config.instruction or type(config.program) is not SequenceCfg:
        raise ValueError(
            "Observed program requires an instruction and a segment sequence"
        )
    segments = {}
    for node in config.program.items:
        if type(node) is not SegmentCfg or type(node.steps) is not InvokeCfg:
            raise ValueError(
                "Each observed segment must contain one canonical semantic call"
            )
        if node.name in segments:
            raise ValueError("Observed segment names must be unique")
        segments[node.name] = node.steps.call
        call = node.steps.call
        references = [call.object]
        references.extend(
            reference
            for key in ("on", "inside", "grasp")
            if (reference := getattr(call, key, None)) is not None
        )
        if not set(references) <= objects:
            raise ValueError(
                "Task Program annotation references an unknown scene object"
            )
    for target in config.targets.values():
        if type(target) is SceneRelativePoseTargetCfg and target.entity not in objects:
            raise ValueError("Task Program target references an unknown scene object")
    bindings = value["trajectory_bindings"]
    if type(bindings) is not list or not bindings:
        raise ValueError("Trajectory bindings must be a nonempty list")
    keys = {
        "name",
        "segment",
        "start",
        "stop",
        "kind",
        "reference",
        "acting_arm",
        "instruction",
    }
    seen, bound = set(), set()
    stop, ordinal = 0, -1
    order = {name: index for index, name in enumerate(segments)}
    for binding in bindings:
        if type(binding) is not dict or set(binding) != keys:
            raise ValueError("Unexpected trajectory binding fields")
        phase = PhaseAnnotation(
            **{
                key: binding[key]
                for key in ("name", "start", "stop", "kind", "reference")
            }
        )
        if phase.name in seen or phase.start != stop or phase.stop > count:
            raise ValueError(
                "Bindings must cover the source exactly once with unique names"
            )
        segment = binding["segment"]
        if segment not in segments or order[segment] < ordinal:
            raise ValueError(
                "Binding must reference an existing segment in program order"
            )
        if binding["acting_arm"] not in arms:
            raise ValueError("Binding references an unknown recorded arm")
        if (
            type(binding["instruction"]) is not str
            or not binding["instruction"].strip()
        ):
            raise ValueError("Binding requires an observed subphase instruction")
        seen.add(phase.name)
        bound.add(segment)
        stop, ordinal = phase.stop, order[segment]
    if stop != count or bound != set(segments):
        raise ValueError(
            "Bindings must cover the full recording and every program segment"
        )
    canonical = asdict(config)
    canonical.pop("integration")
    return json.loads(
        json.dumps(
            dict(task_program=canonical, trajectory_bindings=bindings), allow_nan=False
        )
    )


def bound_trajectory_phases(annotation: dict) -> list[dict]:
    """Project validated bindings to the numerical module's inclusive intervals.

    Args:
        annotation: Result of validate_program_annotation.

    Returns:
        Internal phase views retaining their canonical Task Program segment IDs.
        Persisted bindings continue to use the shared half-open sample convention.
    """
    segments = {
        node["name"]: node["steps"]["call"]
        for node in annotation["task_program"]["program"]["items"]
    }
    return [
        dict(
            id=item["name"],
            segment=item["segment"],
            call=segments[item["segment"]],
            start_frame=item["start"],
            end_frame=item["stop"] - 1,
            acting_arm=item["acting_arm"],
            object_ids=list(
                dict.fromkeys(
                    reference
                    for key in ("object", "on", "inside")
                    if (reference := segments[item["segment"]].get(key)) is not None
                )
            ),
            action_zh=item["instruction"],
            kind=item["kind"],
            reference=item["reference"],
        )
        for item in annotation["trajectory_bindings"]
    ]


def _json_response(
    chat: ChatService, prompt: str, evidence: dict, image: Path | None = None
) -> dict:
    parts = (TextPart(json.dumps(evidence, ensure_ascii=False)),)
    if image is not None:
        parts += (ImagePart(image),)
    result = chat.complete(
        ChatRequest(
            (ChatMessage("system", (TextPart(prompt),)), ChatMessage("user", parts))
        )
    )
    value = (
        result.structured if result.structured is not None else json.loads(result.text)
    )
    if not isinstance(value, dict):
        raise ValueError("Expected a structured annotation object")
    return value


def ground_gripper_events(
    inputs: ReconstructionInput,
    scene: RegisteredScene,
    output: Path,
    chat: ChatService,
) -> dict:
    """Associate source jaw events with observed transported objects before fitting.

    Args:
        inputs: Real observations with explicit sample-to-video-frame mapping.
        scene: Generated identities and optional SAM masks of the same image view.
        output: New directory for evidence and validated associations.
        chat: Caller-provided multimodal service; no simulation is submitted.

    Returns:
        One association per detected event, including null for idle/failed grabs.
        A missing manipulated object or invalid model response raises ValueError.
    """
    from embodichain.data_pipeline.real2sim.adaptation.refinement import (
        _demonstration_events,
    )

    events = _demonstration_events(inputs)
    output.mkdir(parents=True, exist_ok=False)
    camera = inputs.cameras[inputs.primary_camera]
    indices = {0, len(inputs.demonstration.qpos) - 1}
    for event in events:
        g, r = event["grasp"], event["release"]
        indices.update([max(0, g - 30), g, (g + r) // 2, r, event["open_end"]])
    if len(indices) + len(scene.description["objects"]) > 64:
        raise ValueError(
            "Source grounding exceeds 64 evidence cards; split the recording"
        )
    frames = decode_video_frames(
        camera.video, [int(camera.sample_frames[i]) for i in sorted(indices)]
    )
    first = Image.fromarray(frames[int(camera.sample_frames[0])])
    cards = [(first, "REAL first frame")]
    mask_mapping = {}
    for obj in scene.description["objects"]:
        if obj["id"] == "table" or not obj.get("source_mask_path"):
            continue
        with Image.open(obj["source_mask_path"]) as mask:
            original_size = mask.size
            if mask.size != first.size:
                if not np.isclose(
                    mask.width / mask.height,
                    first.width / first.height,
                    rtol=0.005,
                ):
                    raise ValueError(
                        "Object identity mask requires inverse image preprocessing"
                    )
                # Match RGB-D registration's nearest-neighbor inverse resize.
                # Image-edit padding has already been removed by its owner.
                mask = Image.fromarray(
                    cv2.resize(
                        np.asarray(mask.convert("L")),
                        first.size,
                        interpolation=cv2.INTER_NEAREST,
                    )
                )
            mask_path = output / "source_masks" / (obj["id"] + ".png")
            mask_path.parent.mkdir(exist_ok=True)
            mask.convert("L").save(mask_path)
            mask_mapping[obj["id"]] = dict(
                reconstruction_mask=obj["source_mask_path"],
                reconstruction_size=list(original_size),
                source_size=list(first.size),
                source_mask=str(mask_path.resolve()),
            )
            card = first.point(lambda x: int(x * 0.25))
            card.paste(first, mask=mask.convert("L"))
        cards.append((card, "IDENTITY: " + obj["id"]))
    for sample in sorted(indices):
        frame = int(camera.sample_frames[sample])
        cards.append(
            (Image.fromarray(frames[frame]), f"REAL sample {sample}, frame {frame}")
        )
    sheet = Image.new("RGB", (1920, ((len(cards) + 3) // 4) * 390), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (card, label) in enumerate(cards):
        x, y = index % 4 * 480, index // 4 * 390
        sheet.paste(card.resize((480, 360)), (x, y + 25))
        draw.text((x + 5, y + 5), label, fill="black")
    image = output / "evidence.jpg"
    sheet.save(image, quality=93)
    prompt = """Ground detected gripper events in REAL ROBOT observations. Identify the object actually transported by each event, preserving identity over time. Use object_id=null for empty idle closures or failed attempts with no observed transport. Do not choose a support sheet/container merely because fingers approach its contents. Before/grasp/carry/release/open frames are provided; sample IDs use the trajectory clock, not decoded frame IDs. Return strict JSON: events:[{side:int,grasp:int,object_id:string|null,receiver_object_id:string|null,receiver_relation:"inside"|"on"|null,internal_articulation_motion_visible:boolean,reason_zh:string}], missing_manipulated_objects_zh:[string]. Include every supplied event exactly once; use only supplied object IDs. receiver_object_id is the observed receiving support/container, or null for table/uncertain. receiver_relation describes the demonstrated final relation: inside an opening/cavity or on a support surface. Infer it from the actual video/task, not the asset category spelling. Set it to null only when receiver_object_id is null. If a transported object is missing from the scene, report it in missing_manipulated_objects_zh. No new trajectories, poses or simulated-success claims."""
    evidence = dict(
        instruction=inputs.instruction,
        events=events,
        objects=[
            {k: o[k] for k in ("id", "name", "category")}
            for o in scene.description["objects"]
        ],
    )
    value = (
        _json_response(chat, prompt, evidence, image)
        if events
        else dict(events=[], missing_manipulated_objects_zh=[])
    )
    expected = {(e["side"], e["grasp"]) for e in events}
    actual = [(e["side"], e["grasp"]) for e in value["events"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Source grounding omitted or duplicated an original jaw event")
    known = {o["id"] for o in scene.description["objects"] if o["id"] != "table"}
    for event in value["events"]:
        uid, receiver = event["object_id"], event["receiver_object_id"]
        if (
            uid not in known | {None}
            or receiver not in known | {None}
            or (receiver is not None and receiver == uid)
        ):
            raise ValueError(
                "Source grounding supplied an invalid object identity or receiver"
            )
        if (
            not isinstance(event.get("reason_zh"), str)
            or type(event.get("internal_articulation_motion_visible")) is not bool
        ):
            raise ValueError(
                "Source grounding must provide observed evidence and articulation scope"
            )
        relation = event.get("receiver_relation")
        if (
            "receiver_relation" not in event
            or relation not in {"inside", "on", None}
            or (receiver is None) != (relation is None)
        ):
            raise ValueError(
                "Source grounding must specify the observed receiver relation"
            )
    if value.get("missing_manipulated_objects_zh"):
        write_json(output / "grounding.json", value)
        raise ValueError("Source grounding reports a missing manipulated object")
    value["provenance"] = dict(
        source_recording_sha256=inputs.demonstration.recording_sha256,
        evidence_sha256=digest(image),
        source_sample_to_video_frame={
            str(i): int(camera.sample_frames[i]) for i in sorted(indices)
        },
        mask_mapping=mask_mapping,
        scope="real observations only; null events retain jaw timing and may receive local collision clearance",
    )
    write_json(output / "grounding.json", value)
    return value


def _validate_phases(value: dict, count: int, objects: set[str]) -> None:
    phases = value.get("phases", [])
    if (
        not phases
        or not isinstance(value.get("observed_task_instruction"), str)
        or not value["observed_task_instruction"].strip()
    ):
        raise ValueError("Missing source phases or observed instruction")
    stop = 0
    seen = set()
    for phase in phases:
        start, end = phase["start_frame"], phase["end_frame"]
        if (
            type(start) is not int
            or type(end) is not int
            or start != stop
            or not start <= end < count
        ):
            raise ValueError(
                "Source phases must cover the entire record once, in order"
            )
        if not isinstance(phase["id"], str) or not phase["id"] or phase["id"] in seen:
            raise ValueError("Source phase IDs must be unique")
        if not set(phase.get("object_ids", [])) <= objects:
            raise ValueError("Unknown annotated scene object")
        seen.add(phase["id"])
        stop = end + 1
    if stop != count:
        raise ValueError("Source annotation has a missing final interval")


def annotate_trajectory(
    inputs: ReconstructionInput,
    scene: RegisteredScene,
    refined: RefinedTrajectory,
    output: Path,
    chat: ChatService,
) -> dict:
    """Annotate real observations and refined anchors without simulation input.

    Args:
        inputs: Recording, task instruction and explicit video-frame mapping.
        scene: Registered scene object identities.
        refined: Nominal reference and grasp/release diagnostics.
        output: New directory for observation evidence and annotation.
        chat: Caller-provided multimodal chat service.

    Returns:
        Validated contiguous source phases, instruction and provenance.
    """
    output.mkdir(parents=True, exist_ok=False)
    times = inputs.demonstration.timestamps - inputs.demonstration.timestamps[0]
    if inputs.demonstration.time_unit == "ms":
        times = times / 1000
    indices = sorted(
        set(
            np.linspace(0, len(times) - 1, 18).round().astype(int).tolist()
            + [
                e[k]
                for e in refined.report["events"]
                for k in ("close_start", "grasp", "release", "open_end")
            ]
        )
    )
    if len(indices) > 64:
        raise ValueError(
            "Annotation exceeds 64 selected source samples; split the recording explicitly"
        )
    camera = inputs.cameras[inputs.primary_camera]
    frame_ids = [int(camera.sample_frames[i]) for i in indices]
    frames = decode_video_frames(camera.video, frame_ids)
    sheet = Image.new("RGB", (1280, ((len(indices) + 3) // 4) * 270), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for column, (sample, frame) in enumerate(zip(indices, frame_ids)):
        row, col = divmod(column, 4)
        sheet.paste(
            Image.fromarray(frames[frame]).resize((320, 240)),
            (col * 320, row * 270 + 25),
        )
        draw.text(
            (col * 320 + 5, row * 270 + 5),
            f"REAL sample {sample}, frame {frame}, {times[sample]:.2f}s",
            fill="black",
        )
    image = output / "evidence.jpg"
    sheet.save(image, quality=90)
    prompt = """Annotate this real robot demonstration and scene-refined trajectory anchors. All images are REAL observations. Return JSON with observed_task_instruction, phases:[{start_frame,end_frame,acting_arm,object_ids,action_zh,evidence_frames}], limitations_zh. The caller assigns phase IDs in time order. Frame bounds refer to zero-based TRAJECTORY SAMPLE IDs shown on the sheet; end_frame is inclusive. Cover every sample exactly once with ordered contiguous phases. Use supplied scene object IDs. Describe observed actions including handovers, not an alternative planner solution. Refined anchors express intent, not simulated success. Simulation is not an input or prerequisite. Do not claim physical validation."""
    evidence = dict(
        instruction=inputs.instruction,
        sample_count=len(times),
        objects=[
            {k: o[k] for k in ("id", "name", "category")}
            for o in scene.description["objects"]
        ],
        refinement=refined.report,
    )
    value = _json_response(chat, prompt, evidence, image)
    # Repeated actions can have identical model-supplied labels. Identity belongs
    # to the ordered recording intervals, not the wording chosen by the model.
    for index, phase in enumerate(value.get("phases", [])):
        phase["id"] = f"phase_{index:03d}"
    _validate_phases(value, len(times), {o["id"] for o in scene.description["objects"]})
    value["provenance"] = dict(
        source_recording_sha256=inputs.demonstration.recording_sha256,
        evidence_sha256=digest(image),
        source_sample_to_video_frame={str(i): f for i, f in zip(indices, frame_ids)},
        scope="real source observations and nominal correction; no simulated success claim",
    )
    write_json(output / "annotation.json", value)
    return value


def bind_graph_intervals(graph: dict, annotation: dict, chat: ChatService) -> dict:
    """Associate graph nodes with source intervals without changing nodes.

    Args:
        graph: Existing semantic task graph.
        annotation: Validated recorded phases.
        chat: Caller-provided chat service for semantic association.

    Returns:
        Source intervals for every graph node, explicitly non-executable.
    """
    prompt = """Map every existing semantic graph node exactly once to matching recorded phase IDs. Return JSON bindings:[{node_id,phase_ids,reason}]. Use [] when no observed phase matches. Do not alter nodes or phases, plan new actions or claim an executable Task Program binding."""
    value = _json_response(
        chat, prompt, {"graph": graph, "phases": annotation["phases"]}
    )
    bindings = value["bindings"]
    if sorted(b["node_id"] for b in bindings) != sorted(
        n["id"] for n in graph["nodes"]
    ):
        raise ValueError("Graph annotation must include each node exactly once")
    phases = {p["id"]: p for p in annotation["phases"]}
    for binding in bindings:
        ids = binding["phase_ids"]
        if len(set(ids)) != len(ids) or not set(ids) <= phases.keys():
            raise ValueError("Graph annotation refers to unknown/duplicate phases")
        binding["source_frame_intervals"] = [
            [phases[p]["start_frame"], phases[p]["end_frame"]] for p in ids
        ]
    return dict(
        schema="real2sim_graph_annotation/v1",
        bindings=bindings,
        execution_ready=False,
        scope="source-time annotation only; no runtime binding or augmentation",
    )


def generate_annotated_graph(
    image: Path,
    annotation: dict,
    output: Path,
    services: GenSimServices,
    chat: ChatService,
    *,
    task_id: str,
) -> dict:
    """Call the native graph generator and annotate its source intervals.

    Args:
        image: Scene reconstruction input image.
        annotation: Source phases and observed task instruction.
        output: New directory for native graph artifacts and interval bindings.
        services: Existing Task Engine service injection.
        chat: Chat service for graph-to-observation association.
        task_id: Caller-owned task identifier.

    Returns:
        Non-executable source-interval bindings for the generated graph.
    """
    from embodichain.gen_sim.task_engine.image_planning import (
        generate_semantic_task_graph_from_image,
    )

    output.mkdir(parents=True, exist_ok=False)
    artifacts = generate_semantic_task_graph_from_image(
        image,
        annotation["observed_task_instruction"],
        task_id=task_id,
        services=services,
    )
    for name, value in artifacts.items():
        write_json(output / (name + ".json"), value)
    graph = artifacts["semantic_task_graph"]
    binding = bind_graph_intervals(graph, annotation, chat)
    binding["graph_sha256"] = digest(output / "semantic_task_graph.json")
    write_json(output / "trajectory_binding.json", binding)
    return binding
