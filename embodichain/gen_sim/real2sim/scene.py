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
"""Native Scene Engine invocation and export adaptation for Real2Sim."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import json
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from embodichain.data_pipeline.real2sim.adaptation.inputs import (
        ReconstructionInput,
        RegisteredScene,
    )
from embodichain.data_pipeline.real2sim.adaptation.video_fusion import (
    decode_video_frames,
    recover_scene_image,
    RecoveryCfg,
)
from embodichain.gen_sim.services import GenSimServices
from embodichain.gen_sim.services.contracts import (
    SegmentationRequest,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.image_segmentation_utils import (
    decode_rle_mask,
)
from .image_edit import ImageEditor, edit_scene_image
from ._artifacts import digest, write_json

__all__ = [
    "prepare_scene_input",
    "reconstruct_scene",
    "reconstruct_video",
    "register_scene_export",
]


def _robot_mask(image: Path, services: GenSimServices) -> np.ndarray:
    with Image.open(image) as opened:
        mask = np.zeros((opened.height, opened.width), dtype=bool)
    client = services.get("image_segmentation")
    for prompt in ("robot arm", "robot gripper"):
        result = client.segment(SegmentationRequest(image, prompt))
        for rle in result.masks:
            decoded = np.asarray(decode_rle_mask(rle)) > 0
            if decoded.shape != mask.shape:
                raise ValueError("Segmentation resolution differs from source")
            mask |= decoded
    return mask


def prepare_scene_input(
    image: Path,
    output: Path,
    services: GenSimServices,
    editor: ImageEditor,
    *,
    video: Path | None = None,
    reference_frames: tuple[int, ...] = (),
    recovery_cfg: RecoveryCfg | None = None,
) -> Path:
    """Compose robot detection, observed video recovery and model image cleanup.

    Reference IDs refer to task-relevant video frames from the same camera.
    The current recovery implementation anchors video frame zero explicitly.

    Args:
        image: Original first image; must be decoded frame zero when using references.
        output: New directory for reconstruction-only image artifacts.
        services: Segmentation service for robot occlusion masks.
        editor: Image-edit service for robot, background and lighting cleanup.
        video: Same-camera source video, required when reference frames are supplied.
        reference_frames: Frames exposing task objects for same-view pixel recovery.
        recovery_cfg: Optional observed-pixel recovery tolerances and memory budget.

    Returns:
        Prepared high-resolution image path with the source framing preserved.
    """
    if reference_frames and video is None:
        raise ValueError("Reference frames require a source video")
    output.mkdir(parents=True, exist_ok=False)
    mask = _robot_mask(image, services)
    Image.fromarray(mask.astype(np.uint8) * 255).save(output / "robot_mask.png")
    if mask.any() and reference_frames:
        images = decode_video_frames(video, [0, *reference_frames])
        with Image.open(image) as first:
            if not np.array_equal(np.asarray(first.convert("RGB")), images[0]):
                raise ValueError(
                    "first_image must be decoded frame zero for observed recovery"
                )
        masks = {0: mask}
        for index in reference_frames:
            path = output / f"reference_{index}.png"
            Image.fromarray(images[index]).save(path)
            masks[index] = _robot_mask(path, services)
        recovered = recover_scene_image(
            images, masks, list(reference_frames), cfg=recovery_cfg
        )
        image = output / "observed_recovery.png"
        Image.fromarray(recovered.rgb).save(image)
        Image.fromarray(recovered.unresolved.astype(np.uint8) * 255).save(
            output / "unresolved_mask.png"
        )
        np.save(output / "donor_frame.npy", recovered.donor_frames)
        write_json(output / "recovery.json", recovered.report)
    else:
        write_json(
            output / "recovery.json",
            {"status": "image_edit_only", "robot_detected": bool(mask.any())},
        )
    return edit_scene_image(image, output / "image_edit", editor)


def reconstruct_scene(
    image: Path,
    output: Path,
    services: GenSimServices,
) -> Path:
    """Call the existing complete Scene Engine.

    Args:
        image: Reconstruction input after optional observation cleanup.
        output: Destination for native scene-generation artifacts.
        services: Existing scene-engine capability injection.

    Returns:
        Native scene-generation directory, including its scene export.
    """
    from embodichain.gen_sim.scene_engine.pipeline.generate import (
        generate_scene_from_image,
    )

    generate_scene_from_image(image, output, services=services)
    return output


def reconstruct_video(
    video: Path,
    output: Path,
    *,
    services: GenSimServices,
    editor: ImageEditor,
    reference_frames: tuple[int, ...] = (),
) -> Path:
    """Prepare a recorded scene and run the complete native Scene Engine.

    This stage needs only video observations, not robot models or calibration.
    Reference frames should expose the task objects near contact events. They
    aid image preparation; moving objects are not replayed into the scene.

    Args:
        video: Source recording from one fixed camera.
        output: Fresh directory for inputs, intermediates and native exports.
        services: Repository model-service runtime, owned by the caller.
        editor: Image-edit service for reconstruction input cleanup.
        reference_frames: Task-relevant decoded frame indices.

    Returns:
        Native Scene Engine output directory, including ``scene_export``.
    """
    video, output = video.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "stage": "decode"}
    try:
        write_json(output / "status.json", status)
        image = output / "first_frame.png"
        Image.fromarray(decode_video_frames(video, [0])[0]).save(image)
        write_json(
            output / "input.json",
            {
                "video": str(video),
                "video_sha256": digest(video),
                "reference_frames": list(reference_frames),
            },
        )
        status["stage"] = "scene_input"
        write_json(output / "status.json", status)
        prepared = prepare_scene_input(
            image,
            output / "scene_input",
            services,
            editor,
            video=video,
            reference_frames=reference_frames,
        )
        status["stage"] = "scene_engine"
        write_json(output / "status.json", status)
        result = reconstruct_scene(prepared, output / "scene_engine", services)
        status.update(status="completed", stage="completed")
        return result
    except Exception as error:
        status.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        write_json(output / "status.json", status)


def register_scene_export(
    inputs: ReconstructionInput, native_run: Path
) -> RegisteredScene:
    """Adapt the native scene export to dataset-neutral RGB-D fitting.

    Args:
        inputs: Original calibrated observations and robot demonstration.
        native_run: Complete Scene Engine generation directory.

    Returns:
        Generated geometry registered to the original robot-base frame.
    """
    from embodichain.data_pipeline.real2sim.adaptation.registration import (
        register_scene,
    )

    export = native_run / "scene_export"
    cfg = json.loads((export / "scene_config.json").read_text())
    understanding = json.loads(
        (native_run / "scene_understanding/scene.json").read_text()
    )
    assets = [
        dict(x)
        for key in ("background", "rigid_object", "articulation")
        for x in cfg.get(key, [])
    ]
    for obj in assets:
        path = Path(obj.get("shape", {}).get("fpath", obj.get("proxy_glb_fpath", "")))
        obj["mesh_path"] = str(
            path if path.is_absolute() else (export / path).resolve()
        )
    masks, mask_paths = {}, {}
    for obj in understanding["objects"]:
        path = Path(obj["mask_path"])
        if not path.is_absolute():
            path = native_run / path
        with Image.open(path) as mask:
            masks[obj["id"]] = np.asarray(mask) > 0
        mask_paths[obj["id"]] = str(path.resolve())
    table = next(x for x in assets if x["uid"] == "table")
    registered = register_scene(
        inputs.cameras[inputs.primary_camera],
        assets,
        masks,
        table["support_surface_z"],
        auxiliary_depth={
            k: v for k, v in inputs.cameras.items() if k != inputs.primary_camera
        },
    )
    for obj in registered.description["objects"]:
        obj["source_mask_path"] = mask_paths[obj["id"]]
    return registered
