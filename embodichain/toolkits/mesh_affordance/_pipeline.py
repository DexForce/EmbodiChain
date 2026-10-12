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

"""Prepare visual evidence, invoke Codex, and export source-indexed scores."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys

import numpy as np
from numpy.typing import NDArray
import trimesh

from embodichain.compute.geometry.surface import (
    partition_surface,
    face_scores_to_vertices,
)
from embodichain.utils import configclass

from ._codex import run_codex, run_process, validate_response
from ._mesh import load_mesh, patch_table, score_colors
from ._providers import resolve_provider

_LOGGER = logging.getLogger(__name__)


@configclass
class MeshAffordanceCfg:
    """Settings for mesh contact-affordance inference.

    Attributes:
        provider: Model provider used by the Codex harness: openai or deepseek.
        model: Exact model ID, or None for the provider default (gpt-6-astra for
            OpenAI; local config model or deepseek-flash for DeepSeek).
        provider_config: Local DeepSeek JSON credentials file path. Secrets are
            loaded only for provider setup and are never stored in this config.
        patch_count: Maximum geodesic surface regions evaluated by the model.
        render_resolution: Square resolution of each Blender camera view.
        threshold: Minimum semantic score for the exported selection mask.
        min_confidence: Minimum subjective model confidence for selection.
        target_part: Optional part to segment independently of grasp suitability.
        part_threshold: Minimum target-part membership for segmentation.
        codex_executable: Codex CLI command or executable path.
        blender_python: Python executable with bpy; defaults to this interpreter.
        timeout_seconds: Timeout for each Blender or Codex subprocess.
    """

    provider: str = "openai"
    model: str | None = None
    provider_config: str | None = None
    patch_count: int = 64
    render_resolution: int = 512
    threshold: float = 0.7
    min_confidence: float = 0.5
    target_part: str | None = None
    part_threshold: float = 0.5
    codex_executable: str = "codex"
    blender_python: str = sys.executable
    timeout_seconds: float = 600.0


@dataclass(frozen=True)
class MeshAffordanceResult:
    """Source-ordered vertex scores and the directory containing their evidence.

    Attributes:
        scores: Semantic suitability, shape ``(N,)``, in ``[0, 1]``.
        confidence: Subjective model confidence, shape ``(N,)``.
        graspable_mask (numpy.ndarray): Vertices passing score, confidence and reference checks.
        output_dir: Directory with NPZ, PLY, heatmap, evidence and Codex logs.
        part_scores: Optional target-part membership values in source vertex order.
        part_mask (numpy.ndarray | None): Optional target-part selection mask, independent of graspability.
    """

    scores: NDArray[np.float32]
    confidence: NDArray[np.float32]
    graspable_mask: NDArray[np.bool_]
    output_dir: Path
    part_scores: NDArray[np.float32] | None = None
    part_mask: NDArray[np.bool_] | None = None


def _validate_config(cfg: MeshAffordanceCfg) -> None:
    cfg.validate()
    if cfg.provider_config is not None:
        if not isinstance(cfg.provider_config, str) or not cfg.provider_config.strip():
            raise ValueError("provider_config must be None or a nonempty file path")
        cfg.provider_config = str(Path(cfg.provider_config).expanduser().resolve())
    resolve_provider(cfg.provider, cfg.model, cfg.provider_config)
    for name, minimum in (("patch_count", 1), ("render_resolution", 256)):
        value = getattr(cfg, name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name in ("threshold", "min_confidence", "part_threshold"):
        value = getattr(cfg, name)
        if (
            type(value) not in (float, int)
            or not np.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if not np.isfinite(cfg.timeout_seconds) or cfg.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if cfg.target_part is not None and (
        not isinstance(cfg.target_part, str) or not cfg.target_part.strip()
    ):
        raise ValueError("target_part must be None or nonempty text")
    for name in ("codex_executable", "blender_python"):
        if not isinstance(getattr(cfg, name), str) or not getattr(cfg, name).strip():
            raise ValueError(f"{name} must be nonempty text")


def _render(directory: Path, cfg: MeshAffordanceCfg, mode: str) -> None:
    python = shutil.which(cfg.blender_python)
    if python is None:
        raise FileNotFoundError(
            f"Blender Python executable not found: {cfg.blender_python}"
        )
    run_process(
        [python, str(Path(__file__).with_name("_render.py")), str(directory), mode],
        directory,
        f"blender_{mode}",
        cfg.timeout_seconds,
    )


def prepare_mesh_affordance(
    mesh_path: str | Path,
    object_description: str,
    task_description: str,
    output_dir: str | Path,
    cfg: MeshAffordanceCfg | None = None,
) -> Path:
    """Prepare indexed geometry and multiview evidence without calling a model.

    Args:
        mesh_path: Local mesh path or a ``get_data_path`` asset key.
        object_description: Object type and relevant known properties.
        task_description: Task to perform with this object.
        output_dir: New or empty output directory; existing runs are preserved.
        cfg: Partition, rendering and inference settings.

    Returns:
        Absolute evidence directory, ready for :func:`score_mesh_affordance`.

    Raises:
        ValueError: Geometry, descriptions or settings are invalid.
        FileExistsError: The output directory is not empty.
    """
    cfg = cfg or MeshAffordanceCfg()
    _validate_config(cfg)
    if not isinstance(object_description, str) or not object_description.strip():
        raise ValueError("object_description must be nonempty text")
    if not isinstance(task_description, str) or not task_description.strip():
        raise ValueError("task_description must be nonempty text")
    source = Path(mesh_path).expanduser()
    if not source.is_file():
        from embodichain.data import get_data_path

        source = Path(get_data_path(str(mesh_path)))
    source = source.resolve(strict=True)
    directory = Path(output_dir).expanduser().resolve()
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError(f"Use an empty output directory: {directory}")
    _LOGGER.info("Loading and partitioning %s", source)
    vertices, faces, index_contract = load_mesh(source)
    patches = partition_surface(vertices, faces, cfg.patch_count)
    surface_vertices = vertices[np.unique(faces)]
    center = (surface_vertices.min(axis=0) + surface_vertices.max(axis=0)) / 2
    scale = float(np.ptp(surface_vertices, axis=0).max())
    normalized = (vertices - center) / scale
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "geometry.npz",
        vertices=vertices,
        faces=faces,
        normalized_vertices=normalized,
        face_patch_ids=patches,
    )
    manifest = {
        "schema_version": "mesh-affordance/v1",
        "mesh_path": str(source),
        "mesh_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "geometry_sha256": hashlib.sha256(
            (directory / "geometry.npz").read_bytes()
        ).hexdigest(),
        "object_description": object_description,
        "task_description": task_description,
        "target_part": cfg.target_part,
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "vertex_index_contract": index_contract,
        "coordinate_system": "original mesh frame; units unknown; no assumed up axis",
        "render_normalization": {
            "center": center.tolist(),
            "scale": scale,
            "formula": "(original - center) / scale",
        },
        "patch_coordinates": "normalized mesh frame, same as renders",
        "patches": patch_table(normalized, faces, patches),
        "render_resolution": cfg.render_resolution,
        "config": cfg.to_dict(),
        "images": [],
    }
    (directory / "evidence.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _LOGGER.info("Rendering eight Blender views in %s", directory)
    _render(directory, cfg, "evidence")
    return directory


def score_mesh_affordance(
    output_dir: str | Path,
    cfg: MeshAffordanceCfg | None = None,
) -> MeshAffordanceResult:
    """Run Codex against prepared evidence and export vertex contact scores.

    Args:
        output_dir: Directory produced by :func:`prepare_mesh_affordance`.
        cfg: Optional inference overrides; otherwise uses the saved settings.

    Returns:
        Vertex scores, confidence, selection mask and evidence directory.

    Raises:
        ValueError: Geometry has changed or the model response is invalid.
        RuntimeError: Blender or Codex fails; detailed logs remain in output_dir.
    """
    directory = Path(output_dir).expanduser().resolve(strict=True)
    manifest = json.loads((directory / "evidence.json").read_text(encoding="utf-8"))
    cfg = cfg or MeshAffordanceCfg(**manifest["config"])
    _validate_config(cfg)
    if (directory / "affordance.npz").exists():
        raise FileExistsError(
            "This run already has scores; prepare a new output directory"
        )
    if (
        hashlib.sha256((directory / "geometry.npz").read_bytes()).hexdigest()
        != manifest["geometry_sha256"]
    ):
        raise ValueError(
            "Prepared geometry changed; render new evidence before scoring"
        )
    if not manifest["images"] or any(
        not (directory / name).is_file() for name in manifest["images"]
    ):
        raise ValueError("Rendering is incomplete; prepare evidence before scoring")
    executable = shutil.which(cfg.codex_executable)
    if executable is None:
        raise FileNotFoundError("Codex CLI not found; install it and run `codex login`")
    manifest["target_part"] = cfg.target_part
    manifest["config"] = cfg.to_dict()
    (directory / "evidence.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model, _ = resolve_provider(cfg.provider, cfg.model, cfg.provider_config)
    _LOGGER.info(
        "Scoring with Codex harness, provider %s, model %s", cfg.provider, model
    )
    if cfg.provider == "deepseek":
        payload = run_codex(
            directory,
            model,
            executable,
            cfg.timeout_seconds,
            provider=cfg.provider,
            provider_config=cfg.provider_config,
        )
    else:
        payload = run_codex(directory, model, executable, cfg.timeout_seconds)
    return _export_result(directory, cfg, model, payload)


def _export_result(
    directory: Path, cfg: MeshAffordanceCfg, model: str, payload: dict
) -> MeshAffordanceResult:
    """Validate and export an obtained provider response without another API call."""
    manifest = json.loads((directory / "evidence.json").read_text(encoding="utf-8"))
    values = validate_response(payload, len(manifest["patches"]), bool(cfg.target_part))
    patch_scores, patch_confidence = values[:2]
    with np.load(directory / "geometry.npz", allow_pickle=False) as geometry:
        vertices, faces, patches = (
            geometry["vertices"],
            geometry["faces"],
            geometry["face_patch_ids"],
        )
    face_scores = patch_scores[patches]
    scores, valid = face_scores_to_vertices(vertices, faces, face_scores)
    confidence, _ = face_scores_to_vertices(vertices, faces, patch_confidence[patches])
    mask = valid & (scores >= cfg.threshold) & (confidence >= cfg.min_confidence)
    colors = score_colors(scores)
    colors[~valid] = [128, 128, 128]
    np.save(directory / "vertex_colors.npy", colors)
    _LOGGER.info("Rendering the score heatmap")
    _render(directory, cfg, "scores")
    part_scores, part_mask = None, None
    part_arrays = {}
    if cfg.target_part:
        patch_part_scores, patch_part_confidence = values[2:]
        part_scores, _ = face_scores_to_vertices(
            vertices, faces, patch_part_scores[patches]
        )
        part_confidence, _ = face_scores_to_vertices(
            vertices, faces, patch_part_confidence[patches]
        )
        part_mask = (
            valid
            & (part_scores >= cfg.part_threshold)
            & (part_confidence >= cfg.min_confidence)
        )
        part_face_mask = (patch_part_scores[patches] >= cfg.part_threshold) & (
            patch_part_confidence[patches] >= cfg.min_confidence
        )
        part_arrays = dict(
            part_scores=part_scores,
            part_confidence=part_confidence,
            part_mask=part_mask,
            part_face_mask=part_face_mask,
            patch_part_scores=patch_part_scores,
            patch_part_confidence=patch_part_confidence,
        )
        segmentation_colors = np.where(
            part_mask[:, None], np.array([235, 65, 40]), np.array([165, 175, 190])
        ).astype(np.uint8)
        np.save(directory / "segmentation_colors.npy", segmentation_colors)
        _LOGGER.info("Rendering target-part segmentation: %s", cfg.target_part)
        _render(directory, cfg, "part")
        trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=segmentation_colors,
            process=False,
        ).export(directory / "segmentation.ply")
        source_face_ids = np.flatnonzero(part_face_mask)
        source_vertex_ids, remapped = np.unique(
            faces[part_face_mask], return_inverse=True
        )
        part_vertices = vertices[source_vertex_ids]
        part_faces = remapped.reshape(-1, 3)
        np.savez_compressed(
            directory / "target_part.npz",
            vertices=part_vertices,
            faces=part_faces,
            source_vertex_ids=source_vertex_ids,
            source_face_ids=source_face_ids,
        )
        # Empty segmentation is valid: its NPZ contains empty arrays; no invalid PLY is emitted.
        if len(part_faces):
            trimesh.Trimesh(
                vertices=part_vertices, faces=part_faces, process=False
            ).export(directory / "target_part.ply")
    # Publish result artifacts only after inference, validation and visualization succeed.
    trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_colors=colors, process=False
    ).export(directory / "affordance.ply")
    report = {
        **payload,
        "model": model,
        "provider": cfg.provider,
        "harness": "codex",
        "mesh_sha256": manifest["mesh_sha256"],
        "vertex_index_contract": manifest["vertex_index_contract"],
        "threshold": cfg.threshold,
        "min_confidence": cfg.min_confidence,
        "target_part": cfg.target_part,
        "part_threshold": cfg.part_threshold,
        "part_vertex_count": int(part_mask.sum()) if part_mask is not None else None,
        "part_face_count": (
            int(part_arrays["part_face_mask"].sum()) if part_arrays else None
        ),
        "config": cfg.to_dict(),
        "selected_vertex_count": int(mask.sum()),
        "vertex_count": len(vertices),
        "score_semantics": "task-conditioned contact preference, not grasp success probability",
        "limitations": [
            "No gripper/collision/force-closure/reachability verification",
            "Piecewise patch estimates; vertex values are incident-face area averages",
            "Model confidence is subjective and uncalibrated",
        ],
    }
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        directory / "affordance.npz",
        vertices=vertices,
        faces=faces,
        vertex_ids=np.arange(len(vertices)),
        scores=scores,
        confidence=confidence,
        valid_mask=valid,
        graspable_mask=mask,
        face_patch_ids=patches,
        face_scores=face_scores,
        patch_scores=patch_scores,
        patch_confidence=patch_confidence,
        **part_arrays,
    )
    return MeshAffordanceResult(
        scores, confidence, mask, directory, part_scores, part_mask
    )


def analyze_mesh_affordance(
    mesh_path: str | Path,
    object_description: str,
    task_description: str,
    output_dir: str | Path,
    cfg: MeshAffordanceCfg | None = None,
) -> MeshAffordanceResult:
    """Prepare evidence and score a mesh with the Codex harness.

    Args:
        mesh_path: Local mesh path or EmbodiChain asset key.
        object_description: Object category and known properties.
        task_description: Intended manipulation task.
        output_dir: New or empty directory for evidence and result artifacts.
        cfg: Optional settings; defaults to ``gpt-6-astra`` and this Python's bpy.

    Returns:
        Per-vertex scores, confidence and thresholded contact-region selection.
    """
    directory = prepare_mesh_affordance(
        mesh_path, object_description, task_description, output_dir, cfg
    )
    return score_mesh_affordance(directory, cfg)
