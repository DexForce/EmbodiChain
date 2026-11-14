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
"""Provider-free validation of Real2Sim service and graph boundaries."""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import shutil
import numpy as np
import pytest
from PIL import Image
from embodichain.gen_sim.real2sim.image_edit import edit_scene_image
from embodichain.gen_sim.real2sim.annotation import (
    bind_graph_intervals,
    _validate_phases,
)
from embodichain.gen_sim.real2sim.pipeline import run_real2sim


class _Chat:
    def __init__(self, value):
        self.value = value

    def complete(self, request):
        return SimpleNamespace(structured=self.value, text="")


def test_image_edit_preserves_provider_resolution_without_padding(tmp_path):
    source = tmp_path / "source.png"
    rgb = np.zeros((73, 101, 3), np.uint8)
    rgb[:] = [40, 80, 120]
    rgb[20:45, 30:60] = [255, 0, 0]
    Image.fromarray(rgb).save(source)
    raw = source.read_bytes()

    class Editor:
        def edit(self, image, prompt, output):
            with Image.open(image) as opened:
                assert opened.size == (768, 576)
            assert prompt == "Use diffuse lighting."
            shutil.copyfile(image, output)

    target = edit_scene_image(
        source, tmp_path / "image_edit", Editor(), prompt="Use diffuse lighting."
    )
    with Image.open(target) as opened:
        assert opened.size == (768, 555)
        restored = np.asarray(opened.resize((101, 73), Image.Resampling.LANCZOS))
        assert restored.shape == rgb.shape
        # Resampling may alter boundary colors, but must preserve object pixels'
        # locations rather than shifting them by the service padding.
        np.testing.assert_array_equal(restored[..., 0] > 200, rgb[..., 0] > 200)
    assert source.read_bytes() == raw


@pytest.mark.parametrize("refs,reveal", [((1,), True), ((1,), False), ((), False)])
def test_multiframe_edit_reaches_native_engine_without_foreground_gate(
    tmp_path, monkeypatch, refs, reveal
):
    from embodichain.gen_sim.real2sim.scene import (
        prepare_scene_input,
        reconstruct_scene,
    )
    from embodichain.gen_sim.scene_engine.pipeline import generate

    background = np.random.default_rng(7).integers(30, 180, (96, 128, 3), np.uint8)
    background[35:65, 45:80] = [220, 30, 40]
    first = background.copy()
    first[40:60, 50:75] = 0
    source = tmp_path / "first.png"
    Image.fromarray(first).save(source)
    original_bytes = source.read_bytes()
    frames = {0: first, 1: background, 2: background, 3: background}
    mask = np.zeros(first.shape[:2], np.uint8)
    mask[40:60, 50:75] = 1
    flat = mask.ravel()
    boundaries = np.r_[0, np.flatnonzero(np.diff(flat)) + 1, flat.size]
    rle = dict(
        size=list(mask.shape), counts=np.diff(boundaries).tolist(), starts_with=0
    )

    class Segmenter:
        def segment(self, request):
            # Preprocessing may identify robot occlusion, never enumerate task objects.
            assert request.prompt in {"robot arm", "robot gripper"}
            return SimpleNamespace(
                masks=[rle] if request.image_path == source or not reveal else []
            )

    def get_service(role):
        assert role == "image_segmentation"
        return Segmenter()

    services = SimpleNamespace(get=get_service)
    monkeypatch.setattr(
        "embodichain.gen_sim.real2sim.scene.decode_video_frames",
        lambda video, indices: frames,
    )

    class Editor:
        def edit(self, image, prompt, output):
            with Image.open(image) as opened:
                # The editor receives the object recovered from donor frames.
                if reveal:
                    np.testing.assert_allclose(
                        np.asarray(opened)[300, 360], [220, 30, 40], atol=2
                    )
                Image.new("RGB", opened.size, (20, 160, 70)).save(output)

    edited = prepare_scene_input(
        source,
        tmp_path / "scene_input",
        services,
        Editor(),
        video=Path("video.mp4"),
        reference_frames=refs,
    )
    expected = np.full((576, 768, 3), [20, 160, 70], dtype=np.uint8)
    np.testing.assert_array_equal(np.asarray(Image.open(edited)), expected)

    def native_generate(image, output, *, services):
        assert services is native_services
        np.testing.assert_array_equal(np.asarray(Image.open(image)), expected)

    native_services = services
    monkeypatch.setattr(generate, "generate_scene_from_image", native_generate)
    reconstruct_scene(edited, tmp_path / "scene_engine", services)
    assert source.read_bytes() == original_bytes


def test_graph_binding_uses_real_intervals_and_cannot_claim_execution():
    graph = {"nodes": [{"id": "pick"}, {"id": "place"}]}
    annotation = {
        "phases": [
            {"id": "a", "start_frame": 0, "end_frame": 9},
            {"id": "b", "start_frame": 10, "end_frame": 19},
        ]
    }
    result = bind_graph_intervals(
        graph,
        annotation,
        _Chat(
            {
                "bindings": [
                    {"node_id": "pick", "phase_ids": ["a"]},
                    {"node_id": "place", "phase_ids": ["b"]},
                ]
            }
        ),
    )
    assert result["bindings"][1]["source_frame_intervals"] == [[10, 19]]
    assert result["execution_ready"] is False
    with pytest.raises(ValueError, match="each node"):
        bind_graph_intervals(
            graph,
            annotation,
            _Chat({"bindings": [{"node_id": "pick", "phase_ids": ["a"]}]}),
        )
    with pytest.raises(ValueError, match="unknown/duplicate"):
        bind_graph_intervals(
            graph,
            annotation,
            _Chat(
                {
                    "bindings": [
                        {"node_id": "pick", "phase_ids": ["missing"]},
                        {"node_id": "place", "phase_ids": []},
                    ]
                }
            ),
        )


def test_annotation_rejects_gaps_in_source_coverage():
    good = {
        "observed_task_instruction": "Move cup",
        "phases": [
            {"id": "a", "start_frame": 0, "end_frame": 9, "object_ids": ["cup"]},
            {"id": "b", "start_frame": 10, "end_frame": 19, "object_ids": ["cup"]},
        ],
    }
    _validate_phases(good, 20, {"cup"})
    good["phases"][1]["start_frame"] = 11
    with pytest.raises(ValueError, match="entire record"):
        _validate_phases(good, 20, {"cup"})


def test_annotation_assigns_distinct_ids_to_repeated_actions(tmp_path, monkeypatch):
    from embodichain.gen_sim.real2sim import annotation
    from embodichain.data_pipeline.real2sim.adaptation.inputs import RegisteredScene

    inputs = SimpleNamespace(
        demonstration=SimpleNamespace(
            timestamps=np.arange(12) / 30, time_unit="s", recording_sha256="a" * 64
        ),
        cameras={
            "front": SimpleNamespace(
                video=Path("recording.mp4"), sample_frames=np.arange(12)
            )
        },
        primary_camera="front",
        instruction="Pick, retry, place",
    )
    scene = RegisteredScene(
        {"objects": [dict(id="block", name="block", category="block")]}, {}
    )
    monkeypatch.setattr(
        annotation,
        "decode_video_frames",
        lambda path, ids: {i: np.zeros((24, 32, 3), np.uint8) for i in ids},
    )
    phases = [
        dict(
            id="approach",
            start_frame=0,
            end_frame=5,
            object_ids=["block"],
            action_zh="approach",
        ),
        dict(
            id="approach",
            start_frame=6,
            end_frame=11,
            object_ids=["block"],
            action_zh="approach again",
        ),
    ]
    result = annotation.annotate_trajectory(
        inputs,
        scene,
        SimpleNamespace(report={"events": []}),
        tmp_path / "annotation",
        _Chat({"observed_task_instruction": "Pick and place", "phases": phases}),
    )
    assert [p["id"] for p in result["phases"]] == ["phase_000", "phase_001"]
    assert [
        (p["start_frame"], p["end_frame"], p["action_zh"]) for p in result["phases"]
    ] == [(0, 5, "approach"), (6, 11, "approach again")]


@pytest.mark.parametrize("mask_scale", [1, 3])
def test_source_grounding_uses_video_mapping_and_rejects_invented_or_missing_events(
    tmp_path, monkeypatch, mask_scale
):
    from copy import deepcopy
    from embodichain.gen_sim.real2sim import annotation
    from embodichain.data_pipeline.real2sim.adaptation.inputs import RegisteredScene

    q = np.zeros((300, 8))
    q[:, 6] = q[:, 7] = np.interp(
        np.arange(300),
        [0, 30, 50, 100, 130, 210, 240, 299],
        [0.01, 0.01, 0.05, 0.05, 0.015, 0.015, 0.05, 0.05],
    )
    original = q.copy()
    mapping = np.arange(300) * 2
    inputs = SimpleNamespace(
        demonstration=SimpleNamespace(qpos=q, recording_sha256="a" * 64),
        robot=SimpleNamespace(gripper_calibrations=()),
        primary_camera="front",
        cameras={
            "front": SimpleNamespace(video=Path("recording.mp4"), sample_frames=mapping)
        },
        instruction="Move block",
    )
    mask = tmp_path / "mask.png"
    source_mask = np.zeros((24, 32), np.uint8)
    source_mask[4:20, 8:24] = 255
    Image.fromarray(
        source_mask.repeat(mask_scale, axis=0).repeat(mask_scale, axis=1)
    ).save(mask)
    scene = RegisteredScene(
        {
            "objects": [
                dict(
                    id="block",
                    name="wooden block",
                    category="block",
                    source_mask_path=str(mask),
                )
            ]
        },
        {},
    )
    requested = []

    def decode(path, ids):
        assert path == Path("recording.mp4")
        requested.extend(ids)
        return {i: np.full((24, 32, 3), i % 255, np.uint8) for i in ids}

    monkeypatch.setattr(annotation, "decode_video_frames", decode)

    class Chat:
        value = None

        def complete(self, request):
            import json

            evidence = json.loads(request.messages[1].content[0].text)
            assert len(evidence["events"]) == 1
            event = evidence["events"][0]
            assert 115 < event["grasp"] < 140
            result = {
                "events": [
                    dict(
                        side=0,
                        grasp=event["grasp"],
                        object_id=None,
                        receiver_object_id=None,
                        receiver_relation=None,
                        internal_articulation_motion_visible=False,
                        reason_zh="Empty attempt",
                    )
                ],
                "missing_manipulated_objects_zh": [],
            }
            return SimpleNamespace(
                structured=deepcopy(self.value if self.value is not None else result),
                text="",
            )

    chat = Chat()
    result = annotation.ground_gripper_events(inputs, scene, tmp_path / "valid", chat)
    np.testing.assert_array_equal(
        np.asarray(Image.open(tmp_path / "valid/source_masks/block.png")), source_mask
    )
    assert result["events"][0]["object_id"] is None
    assert all(
        frame == int(sample) * 2
        for sample, frame in result["provenance"][
            "source_sample_to_video_frame"
        ].items()
    )
    assert all(i % 2 == 0 for i in requested) and 598 in requested
    np.testing.assert_array_equal(q, original)
    chat.value = deepcopy(result)
    chat.value["events"][0]["object_id"] = "invented"
    with pytest.raises(ValueError, match="invalid object identity"):
        annotation.ground_gripper_events(inputs, scene, tmp_path / "invented", chat)
    chat.value = deepcopy(result)
    chat.value["events"] = []
    with pytest.raises(ValueError, match="omitted or duplicated"):
        annotation.ground_gripper_events(inputs, scene, tmp_path / "omitted", chat)
    chat.value = deepcopy(result)
    del chat.value["events"][0]["receiver_relation"]
    with pytest.raises(ValueError, match="receiver relation"):
        annotation.ground_gripper_events(
            inputs, scene, tmp_path / "missing_relation", chat
        )


def test_pipeline_preflight_does_not_launch_services_or_create_output(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="injected chat"):
        run_real2sim(None, output)
    assert not output.exists()


def test_pipeline_refines_registered_scene_without_physics_prerequisite(
    tmp_path, monkeypatch
):
    import json
    from embodichain.gen_sim.real2sim import pipeline
    from embodichain.data_pipeline.real2sim.adaptation.inputs import RegisteredScene

    source = tmp_path / "input.png"
    Image.new("RGB", (32, 32)).save(source)
    initial = np.eye(4)
    initial[2, 3] = 0.03
    scene = RegisteredScene(
        {
            "objects": [{"id": "cup", "transform": initial.tolist()}],
            "robot_base_pose_world": np.eye(4).tolist(),
        },
        {"cup": np.array([[0, 0, 0.03], [0.01, 0.01, 0.04]])},
    )
    inputs = SimpleNamespace(
        demonstration=SimpleNamespace(
            source_id="source", unit_id="unit", recording_sha256="a" * 64
        ),
        instruction="Move cup",
        primary_camera="front",
        cameras={},
    )
    output = tmp_path / "run"

    grounded = (
        {"side": 0, "grasp": 10, "object_id": None, "reason_zh": "Empty closure"},
    )

    def ground(inputs, registered, folder, chat):
        assert registered.vertices["cup"][0, 2] == 0.03
        return {"events": list(grounded)}

    def refine(inputs, registered, folder, *, grounding, approach_clearance_m):
        assert registered.description["objects"][0]["transform"][2][3] == 0.03
        assert registered.vertices["cup"][0, 2] == 0.03
        assert grounding == grounded
        assert approach_clearance_m == 0.007
        return SimpleNamespace(report={})

    monkeypatch.setattr(pipeline, "refine_recording", refine)
    monkeypatch.setattr(
        "embodichain.gen_sim.real2sim.annotation.ground_gripper_events", ground
    )
    result = run_real2sim(
        inputs,
        output,
        prepared_image=source,
        registered_scene=scene,
        annotate=False,
        approach_clearance_m=0.007,
        chat=_Chat({}),
    )
    assert result.scene.description["objects"][0]["transform"][2][3] == 0.03
    exported = json.loads((output / "scene.json").read_text())
    assert exported["objects"][0]["transform"][2][3] == 0.03
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "geometry_and_trajectory_completed"
    assert status["completed_stages"] == [
        "scene_input",
        "registered_scene",
        "source_grounding",
        "refinement",
    ]


@pytest.mark.parametrize("fail", [False, True])
def test_video_scene_stage_keeps_input_and_reports_native_failure(
    tmp_path, monkeypatch, fail
):
    """An actual decoded video reaches native generation and preserves failure stage."""
    import av
    import json
    from embodichain.gen_sim.real2sim import scene

    video = tmp_path / "source.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(3):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), 60 + i, np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    original = video.read_bytes()

    class Segmenter:
        def segment(self, request):
            return SimpleNamespace(masks=[])

    class Editor:
        def edit(self, image, prompt, output):
            shutil.copyfile(image, output)

    def generate(image, output, services):
        assert image.is_file()
        if fail:
            raise RuntimeError("geometry service unavailable")
        (output / "scene_export").mkdir(parents=True)
        return output

    monkeypatch.setattr(scene, "reconstruct_scene", generate)
    output = tmp_path / "run"
    kwargs = dict(
        services=SimpleNamespace(get=lambda role: Segmenter()), editor=Editor()
    )
    if fail:
        with pytest.raises(RuntimeError, match="geometry service unavailable"):
            scene.reconstruct_video(video, output, **kwargs)
    else:
        result = scene.reconstruct_video(video, output, **kwargs)
        assert (result / "scene_export").is_dir()
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == ("failed" if fail else "completed")
    assert status["stage"] == ("scene_engine" if fail else "completed")
    assert (output / "scene_input/image_edit/scene_input.png").is_file()
    assert video.read_bytes() == original
    with pytest.raises(FileExistsError):
        scene.reconstruct_video(video, output, **kwargs)
