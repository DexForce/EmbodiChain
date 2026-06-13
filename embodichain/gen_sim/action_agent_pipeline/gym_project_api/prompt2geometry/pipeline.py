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

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .dimensions import estimate_real_dimensions
    from .llm_client import OpenAICompatibleClient
    from .mesh_scaling import scale_mesh_to_real_dimensions
    from .sam3_client import SAM3Client
    from .sam3d_client import SAM3DClient
    from .schemas import SelectedBox
    from .segmentation_outputs import save_segmentation_outputs
    from .zimage_client import ZImageClient
except ImportError:
    from dimensions import estimate_real_dimensions
    from llm_client import OpenAICompatibleClient
    from mesh_scaling import scale_mesh_to_real_dimensions
    from sam3_client import SAM3Client
    from sam3d_client import SAM3DClient
    from schemas import SelectedBox
    from segmentation_outputs import save_segmentation_outputs
    from zimage_client import ZImageClient

__all__ = ["Prompt2GeometryRequest", "run_prompt2geometry"]


@dataclass(frozen=True)
class Prompt2GeometryRequest:
    """Request for prompt-to-single-asset geometry generation."""

    prompt: str
    output_root: Path
    target_id: str = "asset_0"
    request_id: str = "prompt2geometry_asset_0"
    output_name: str | None = None
    zimage_base_url: str = "http://192.168.3.23:5013"
    zimage_width: int = 1024
    zimage_height: int = 1024
    zimage_seed: int = 42
    zimage_num_inference_steps: int = 8
    zimage_prompt_suffix: str = "a complete single object, with pure-black background"
    sam3_base_url: str = "http://192.168.3.23:5015"
    sam3d_base_url: str = "http://192.168.3.23:5016"
    sam3d_seed: int = 42
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_timeout_s: float = 120.0
    verbose: bool = True


def run_prompt2geometry(request: Prompt2GeometryRequest) -> dict[str, Any]:
    """Run z-image, SAM3 segmentation, and SAM3D generation."""
    output_root = request.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_glb_path: Path | None = None
    success = False
    try:
        _log_status(request, "start", f"output_root={output_root}")
        _write_json(
            output_root / "prompt2geometry_request.json",
            _request_manifest(request),
        )

        _log_status(request, "z-image", "generating source image")
        image_path, zimage_manifest = _generate_image(request, output_root)
        _log_status(request, "segmentation", "segmenting generated image")
        raw_mask_path, segmentation_manifest = _segment_image(
            request,
            image_path,
            output_root,
        )
        _log_status(request, "mask", "checking mask orientation with center prior")
        corrected_mask_path = _correct_mask_with_center_prior(
            image_path=image_path,
            raw_mask_path=raw_mask_path,
            output_dir=output_root / "mask_correction",
        )
        _log_status(request, "3D-generation", "generating raw mesh")
        generation_manifest = _generate_geometry(
            request=request,
            image_path=image_path,
            mask_path=corrected_mask_path,
            output_root=output_root,
        )
        _log_status(request, "dimensions", "estimating real-world dimensions")
        dimension_manifest = _estimate_dimensions(request, output_root)
        _log_status(request, "naming", "resolving final GLB file name")
        final_glb_path = _final_scaled_glb_path(request, output_root)
        _log_status(request, "scale", f"writing final mesh to {final_glb_path.name}")
        scaling_manifest = _scale_generated_mesh(
            mesh_path=Path(str(generation_manifest["local_glb_path"])),
            dimensions_m=dimension_manifest,
            output_path=final_glb_path,
            output_root=output_root,
        )
        manifest = {
            "prompt": request.prompt,
            "zimage_prompt": _zimage_prompt(request),
            "output_root": str(output_root),
            "image_path": str(image_path),
            "raw_mask_path": str(raw_mask_path),
            "corrected_mask_path": str(corrected_mask_path),
            "zimage": zimage_manifest,
            "sam3_segmentation": segmentation_manifest,
            "sam3d_generation": generation_manifest,
            "dimension_estimation": dimension_manifest,
            "mesh_scaling": scaling_manifest,
            "mesh_path": generation_manifest.get("local_glb_path"),
            "scaled_mesh_path": scaling_manifest.get("scaled_mesh_path"),
            "transform_metadata_path": generation_manifest.get(
                "local_transform_metadata_path"
            ),
        }
        _write_json(output_root / "prompt2geometry_result.json", manifest)
        success = True
        _log_status(request, "done", f"final_glb={final_glb_path}")
        return manifest
    finally:
        _cleanup_output_root(output_root, keep_path=final_glb_path if success else None)


def _generate_image(
    request: Prompt2GeometryRequest,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    image_path = output_root / "zimage" / "zimage.png"
    client = ZImageClient(base_url=request.zimage_base_url)
    manifest = client.generate_png(
        prompt=_zimage_prompt(request),
        output_path=image_path,
        width=request.zimage_width,
        height=request.zimage_height,
        seed=request.zimage_seed,
        num_inference_steps=request.zimage_num_inference_steps,
    )
    _write_json(output_root / "zimage" / "zimage_result.json", manifest)
    return image_path, manifest


def _segment_image(
    request: Prompt2GeometryRequest,
    image_path: Path,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    width, height = _image_size(image_path)
    full_image_box = SelectedBox(
        target_id=request.target_id,
        target_kind="asset",
        phrase=request.target_id,
        bbox_xyxy=[0.0, 0.0, float(width), float(height)],
        source_candidate_ids=["full_image_bbox"],
        selection_reason="Use the full generated image as a bbox prompt.",
    )
    sam3_client = SAM3Client(
        base_url=os.getenv("PROMPT2GEOMETRY_SAM3_BASE_URL") or request.sam3_base_url,
    )
    health = sam3_client.health()
    _write_json(output_root / "sam3_health.json", health)

    sam3_request = {
        "image": str(image_path),
        "request_id": f"{request.request_id}_sam3_box",
        "selected_boxes": [full_image_box.to_manifest()],
        "save_visualizations": False,
    }
    _write_json(output_root / "sam3_box_segmentation_request.json", sam3_request)
    result = sam3_client.segment_boxes_image(
        image_path,
        selected_boxes=[full_image_box],
        request_id=f"{request.request_id}_sam3_box",
        save_visualizations=False,
        progress_path=output_root / "sam3_progress.jsonl",
        verbose=request.verbose,
    )
    _write_json(output_root / "sam3_segmentation_result.json", result)

    local_outputs = save_segmentation_outputs(
        image_path=image_path,
        segmentation_result=result,
        output_dir=output_root / "segment_box",
    )
    _write_json(output_root / "sam3_local_outputs.json", local_outputs)
    segmentations = local_outputs.get("segmentations", [])
    if not isinstance(segmentations, list) or not segmentations:
        raise RuntimeError("SAM3 box segmentation produced no local masks.")
    first = segmentations[0]
    mask_path = first.get("local_mask_path")
    if not isinstance(mask_path, str) or not mask_path:
        raise RuntimeError("SAM3 local segmentation output missing local_mask_path.")
    return Path(mask_path).expanduser().resolve(), local_outputs


def _generate_geometry(
    *,
    request: Prompt2GeometryRequest,
    image_path: Path,
    mask_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_name = request.output_name or f"{request.request_id}.glb"
    local_glb_path = output_root / "sam3d" / output_name
    local_metadata_path = (
        output_root / "sam3d" / f"{Path(output_name).stem}_transform.json"
    )

    client = SAM3DClient(
        base_url=os.getenv("PROMPT2GEOMETRY_SAM3D_BASE_URL") or request.sam3d_base_url,
    )
    health = client.health()
    _write_json(output_root / "sam3d_health.json", health)
    generation_request = {
        "image": str(image_path),
        "mask": str(mask_path),
        "request_id": request.request_id,
        "output_name": output_name,
        "prompt": request.prompt,
        "seed": request.sam3d_seed,
        "local_glb_path": str(local_glb_path),
        "local_transform_metadata_path": str(local_metadata_path),
    }
    _write_json(output_root / "sam3d_generation_request.json", generation_request)
    result = client.generate_asset(
        image_path=image_path,
        mask_path=mask_path,
        request_id=request.request_id,
        output_name=output_name,
        prompt=request.prompt,
        seed=request.sam3d_seed,
        output_path=local_glb_path,
        metadata_path=local_metadata_path,
        progress_path=output_root / "sam3d_progress.jsonl",
        verbose=request.verbose,
    )
    _write_json(output_root / "sam3d_generation_result.json", result)
    return result


def _estimate_dimensions(
    request: Prompt2GeometryRequest,
    output_root: Path,
) -> dict[str, Any]:
    client = _llm_client_from_request(request, purpose="dimension estimation")
    dimensions = estimate_real_dimensions(
        object_prompt=request.prompt,
        client=client,
    )
    _write_json(output_root / "dimension_estimation.json", dimensions)
    return dimensions


def _scale_generated_mesh(
    *,
    mesh_path: Path,
    dimensions_m: dict[str, Any],
    output_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    report_path = output_root / "mesh_scaling_report.json"
    return scale_mesh_to_real_dimensions(
        mesh_path=mesh_path,
        output_path=output_path,
        dimensions_m=dimensions_m,
        report_path=report_path,
    )


def _final_scaled_glb_path(
    request: Prompt2GeometryRequest,
    output_root: Path,
) -> Path:
    if request.output_name:
        stem = _safe_glb_stem(Path(request.output_name).stem)
    else:
        client = _llm_client_from_request(request, purpose="GLB file naming")
        stem = _extract_glb_stem_from_prompt(request.prompt, client)
    return output_root / f"{stem}.glb"


def _extract_glb_stem_from_prompt(
    prompt: str,
    client: OpenAICompatibleClient,
) -> str:
    system_prompt = """
<role>
You extract a concise object file name from a prompt.
</role>

<task>
Return a JSON object with one field, object_name, containing a short ASCII
snake_case name for the single main object described by the user.
</task>

<rules>
- Output JSON only.
- Required schema: {"object_name": "red_ceramic_mug"}
- object_name must be non-empty.
- Do not include a file extension.
- Use only lowercase English letters, numbers, and underscores.
- Prefer the concrete object noun with one or two useful modifiers.
</rules>
""".strip()
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Prompt:\n" f"{prompt.strip()}\n\n" "Return the object_name JSON only."
            ),
        },
    ]
    while True:
        try:
            raw = client.chat_json(messages=messages)
            return _validate_glb_stem_output(raw)
        except Exception:
            time.sleep(1.0)
            continue


def _validate_glb_stem_output(raw: dict[str, Any]) -> str:
    value = raw.get("object_name")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("object_name must be a non-empty string.")
    return _safe_glb_stem(value)


def _safe_glb_stem(value: str) -> str:
    stem = value.strip().lower()
    if stem.endswith(".glb"):
        stem = stem[:-4]
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem:
        raise ValueError("GLB file name stem is empty after sanitization.")
    return stem


def _llm_client_from_request(
    request: Prompt2GeometryRequest,
    *,
    purpose: str,
) -> OpenAICompatibleClient:
    api_key = os.getenv("PROMPT2GEOMETRY_LLM_API_KEY") or request.llm_api_key
    model = os.getenv("PROMPT2GEOMETRY_LLM_MODEL") or request.llm_model
    base_url = os.getenv("PROMPT2GEOMETRY_LLM_BASE_URL") or request.llm_base_url
    missing = [
        name
        for name, value in {
            "PROMPT2GEOMETRY_LLM_API_KEY or --llm-api-key": api_key,
            "PROMPT2GEOMETRY_LLM_MODEL or --llm-model": model,
            "PROMPT2GEOMETRY_LLM_BASE_URL or --llm-base-url": base_url,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required LLM config for {purpose}: {missing}")
    return OpenAICompatibleClient(
        api_key=str(api_key),
        model=str(model),
        base_url=str(base_url),
        timeout_s=request.llm_timeout_s,
        usage_stage=f"prompt2geometry.{purpose}",
    )


def _cleanup_output_root(output_root: Path, *, keep_path: Path | None) -> None:
    output_root = output_root.expanduser().resolve()
    keep_path = keep_path.expanduser().resolve() if keep_path is not None else None
    if keep_path is not None and not keep_path.is_file():
        keep_path = None
    for child in output_root.iterdir():
        if keep_path is not None and child.resolve() == keep_path:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _correct_mask_with_center_prior(
    *,
    image_path: Path,
    raw_mask_path: Path,
    output_dir: Path,
) -> Path:
    cv2 = _require_cv2()
    np = _require_numpy()

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image for mask correction: {image_path}")
    raw_mask = cv2.imread(str(raw_mask_path), cv2.IMREAD_GRAYSCALE)
    if raw_mask is None:
        raise ValueError(f"Failed to read raw mask for correction: {raw_mask_path}")
    height, width = image.shape[:2]
    if raw_mask.shape[:2] != (height, width):
        raise ValueError(
            "Raw mask shape does not match image shape: "
            f"{raw_mask.shape[:2]} vs {(height, width)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_bool = raw_mask > 0
    inverted_bool = ~raw_bool
    center_bool, edge_bool = _center_prior_regions(height, width)
    normal_score = _center_prior_score(raw_bool, center_bool, edge_bool)
    inverted_score = _center_prior_score(inverted_bool, center_bool, edge_bool)
    used_inverted = inverted_score["score"] > normal_score["score"]
    corrected_bool = inverted_bool if used_inverted else raw_bool

    raw_output = output_dir / "sam3_raw_mask.png"
    center_prior_output = output_dir / "center_prior_reference_mask.png"
    edge_prior_output = output_dir / "edge_prior_reference_mask.png"
    corrected_output = output_dir / "sam3_corrected_mask.png"
    cv2.imwrite(str(raw_output), raw_bool.astype("uint8") * 255)
    cv2.imwrite(str(center_prior_output), center_bool.astype("uint8") * 255)
    cv2.imwrite(str(edge_prior_output), edge_bool.astype("uint8") * 255)
    cv2.imwrite(str(corrected_output), corrected_bool.astype("uint8") * 255)
    _write_json(
        output_dir / "mask_correction_report.json",
        {
            "image_path": str(image_path),
            "raw_mask_path": str(raw_mask_path),
            "raw_mask_copy_path": str(raw_output),
            "center_prior_reference_mask_path": str(center_prior_output),
            "edge_prior_reference_mask_path": str(edge_prior_output),
            "corrected_mask_path": str(corrected_output),
            "normal_center_prior_score": normal_score,
            "inverted_center_prior_score": inverted_score,
            "used_inverted_mask": used_inverted,
            "raw_mask_area_ratio": float(raw_bool.mean()),
            "corrected_mask_area_ratio": float(corrected_bool.mean()),
            "foreground_rule": (
                "prefer masks with high center foreground density and low edge "
                "foreground density"
            ),
        },
    )
    return corrected_output


def _request_manifest(request: Prompt2GeometryRequest) -> dict[str, Any]:
    return {
        "prompt": request.prompt,
        "output_root": str(request.output_root.expanduser().resolve()),
        "target_id": request.target_id,
        "request_id": request.request_id,
        "output_name": request.output_name,
        "zimage_base_url": request.zimage_base_url,
        "zimage_width": request.zimage_width,
        "zimage_height": request.zimage_height,
        "zimage_seed": request.zimage_seed,
        "zimage_num_inference_steps": request.zimage_num_inference_steps,
        "zimage_prompt_suffix": request.zimage_prompt_suffix,
        "sam3_base_url": request.sam3_base_url,
        "sam3d_base_url": request.sam3d_base_url,
        "sam3d_seed": request.sam3d_seed,
        "llm_model": request.llm_model,
        "llm_base_url": request.llm_base_url,
        "has_llm_api_key": bool(request.llm_api_key),
        "llm_timeout_s": request.llm_timeout_s,
        "verbose": request.verbose,
    }


def _zimage_prompt(request: Prompt2GeometryRequest) -> str:
    prompt = request.prompt.strip()
    suffix = request.zimage_prompt_suffix.strip()
    if not suffix:
        return prompt
    lowered = prompt.lower()
    additions = []
    if "single object" not in lowered and "one object" not in lowered:
        additions.append("a complete single object")
    if "background" not in lowered:
        additions.append(_normalize_background_suffix(suffix))
    if not additions:
        return prompt
    return f"{prompt}, {', '.join(additions)}"


def _normalize_background_suffix(suffix: str) -> str:
    lowered = suffix.lower()
    if "black background" in lowered or "pure-black background" in lowered:
        return "with pure-black background"
    if "white background" in lowered or "pure-white background" in lowered:
        return "with pure-white background"
    return suffix


def _image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to read generated image size.") from exc
    with Image.open(image_path) as image:
        return image.size


def _center_prior_regions(height: int, width: int) -> tuple[Any, Any]:
    np = _require_numpy()
    center_x1 = int(width * 0.2)
    center_x2 = int(width * 0.8)
    center_y1 = int(height * 0.2)
    center_y2 = int(height * 0.8)
    center_bool = np.zeros((height, width), dtype=bool)
    center_bool[center_y1:center_y2, center_x1:center_x2] = True

    edge_x = max(1, int(width * 0.08))
    edge_y = max(1, int(height * 0.08))
    edge_bool = np.zeros((height, width), dtype=bool)
    edge_bool[:edge_y, :] = True
    edge_bool[-edge_y:, :] = True
    edge_bool[:, :edge_x] = True
    edge_bool[:, -edge_x:] = True
    return center_bool, edge_bool


def _center_prior_score(
    mask_bool: Any,
    center_bool: Any,
    edge_bool: Any,
) -> dict[str, float]:
    center_density = _masked_mean(mask_bool, center_bool)
    edge_density = _masked_mean(mask_bool, edge_bool)
    return {
        "score": center_density - edge_density,
        "center_foreground_density": center_density,
        "edge_foreground_density": edge_density,
    }


def _masked_mean(mask_bool: Any, region_bool: Any) -> float:
    if not region_bool.any():
        return 0.0
    return float(mask_bool[region_bool].mean())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _log_status(request: Prompt2GeometryRequest, stage: str, message: str) -> None:
    if request.verbose:
        print(f"[prompt2geometry:{stage}] {message}", flush=True)


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for mask correction.") from exc
    return cv2


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for mask correction.") from exc
    return np
