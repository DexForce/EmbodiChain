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

"""Independent Mitsuba CPU references for frozen custom rendering scenes.

Mitsuba is optional and never imported by report-only paths. RGB is genuine
path-traced radiance; depth and instance truth use separate centre-pixel native
ray intersections, without antialiasing or denoising. The physical lighting
policy is explicit and is not a claim that either tested renderer is calibrated.
These custom scenes have no official Isaac Lab scene provenance.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

from scripts.benchmark.core.artifacts import stable_hash, write_json
from scripts.benchmark.rendering.scenes import (
    camera_intrinsics,
    load_scene,
    scene_fingerprint,
)

__all__ = [
    "DEPTH_SEMANTICS",
    "SEGMENTATION_SEMANTICS",
    "camera_metadata",
    "load_reference_frame",
    "reference_dependencies",
    "render_reference",
]

DEPTH_SEMANTICS = "optical_axis_metres"
SEGMENTATION_SEMANTICS = "canonical_instance_ids_zero_background"
_MAX_WAVEFRONT_SAMPLES = 1 << 24


def _samples_per_pass(spp: int, width: int, height: int) -> int:
    """Bound live sample lanes without dropping any samples from a repeat.

    Mitsuba's LLVM integrator seeds once and advances the sampler between
    passes. Choose a power-of-two divisor of the requested sample count so
    both independent repeats keep their original seeds and exact total spp.
    A film larger than the lane budget still needs one sample per pixel.
    """
    limit = max(1, _MAX_WAVEFRONT_SAMPLES // (width * height))
    return math.gcd(spp, 1 << (limit.bit_length() - 1))


def reference_dependencies() -> dict[str, Any]:
    """Describe optional reference dependencies without loading a renderer.

    Returns:
        Installed versions and explicit missing reasons.
    """
    versions = {}
    missing = []
    for package in ("mitsuba", "drjit", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
            missing.append(f"Optional package {package} is not installed")
    return {"available": not missing, "versions": versions, "missing_reasons": missing}


def camera_metadata(
    scene: dict[str, Any], camera: dict[str, Any], width: int, height: int
) -> dict[str, Any]:
    """Resolve a look-at view to the shared OpenCV pinhole convention.

    Args:
        scene: Frozen scene providing intrinsics and clipping planes.
        camera: Look-at vectors; ``frame_index`` is optional.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Intrinsics and world-from-camera matrix (right, down, forward axes).
        Pixel centres have coordinates ``(column + .5, row + .5)``.
    """
    import numpy as np

    eye = np.asarray(camera["eye"], dtype=float)
    forward = np.asarray(camera["target"], dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(camera["up"], dtype=float))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, forward])
    pose[:3, 3] = eye
    if not np.isfinite(pose).all():
        raise ValueError("Camera look-at vectors are degenerate")
    return {
        "name": camera["name"],
        "width": width,
        "height": height,
        "intrinsics": camera_intrinsics(scene, width, height),
        "world_from_camera": pose.tolist(),
        "axis_convention": "opencv_right_down_forward",
        "convention": "opencv",
        "pixel_centres": "half_integer",
        "near_m": scene["camera_model"]["near_m"],
        "far_m": scene["camera_model"]["far_m"],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mesh(mi: Any, obj: dict[str, Any], scene_dir: Path) -> Any:
    import numpy as np

    geometry = obj["geometry"]
    material = obj["material"]
    bsdf = mi.load_dict(
        {
            "type": "principled",
            "base_color": {"type": "rgb", "value": material["base_color_linear"]},
            "roughness": material["roughness"],
            "metallic": material["metallic"],
        }
    )
    transform = np.asarray(obj["world_from_object"], dtype=float)
    if geometry["kind"] == "box":
        scale = np.diag([*(np.asarray(geometry["size"]) / 2), 1.0])
        shape = mi.load_dict(
            {
                "type": "cube",
                "to_world": mi.ScalarTransform4f(transform @ scale),
                "bsdf": bsdf,
            }
        )
    elif geometry["kind"] == "mesh":
        with np.load(scene_dir / geometry["path"], allow_pickle=False) as archive:
            vertices = np.asarray(archive["vertices"], dtype=np.float64)
            faces = np.asarray(archive["triangles"], dtype=np.uint32)
            normals = np.asarray(archive["normals"], dtype=np.float64)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or normals.shape != vertices.shape
            or faces.ndim != 2
            or faces.shape[1] != 3
            or not np.isfinite(vertices).all()
            or not np.isfinite(normals).all()
            or faces.max() >= len(vertices)
        ):
            raise ValueError(f"Invalid canonical mesh: {obj['uid']}")
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
        normals = normals @ np.linalg.inv(transform[:3, :3])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        props = mi.Properties()
        props["bsdf"] = bsdf
        shape = mi.Mesh(
            obj["uid"], len(vertices), len(faces), props, has_vertex_normals=True
        )
        parameters = mi.traverse(shape)
        parameters["vertex_positions"] = vertices.astype(np.float32).ravel()
        parameters["vertex_normals"] = normals.astype(np.float32).ravel()
        parameters["faces"] = faces.ravel()
        parameters.update()
    else:
        raise ValueError(f"Unsupported canonical geometry: {geometry['kind']}")
    shape.set_id(obj["uid"])
    return shape


def _geometry_buffers(
    mi: Any, dr: Any, scene: Any, camera: dict[str, Any], objects: list[dict[str, Any]]
) -> tuple[Any, Any]:
    import numpy as np

    width, height = camera["width"], camera["height"]
    k = np.asarray(camera["intrinsics"])
    pose = np.asarray(camera["world_from_camera"])
    rows, columns = np.indices((height, width))
    directions = np.stack(
        [
            (columns + 0.5 - k[0, 2]) / k[0, 0],
            (rows + 0.5 - k[1, 2]) / k[1, 1],
            np.ones_like(rows),
        ],
        axis=-1,
    ).reshape(-1, 3)
    world_directions = directions @ pose[:3, :3].T
    # Unnormalised direction has camera Z=1, so native ray t is optical depth.
    near = float(camera["near_m"])
    ray = mi.Ray3f(
        mi.Point3f((pose[:3, 3] + near * world_directions).T),
        mi.Vector3f(world_directions.T),
    )
    ray.maxt = float(camera["far_m"]) - near
    hit = scene.ray_intersect(ray)
    valid = hit.is_valid()
    depth = np.asarray(dr.select(valid, hit.t + near, dr.inf)).reshape(height, width)
    labels = dr.zeros(mi.UInt32, width * height)
    by_uid = {f"object_{obj['instance_id']}": obj["instance_id"] for obj in objects}
    for shape in scene.shapes():
        labels = dr.select(valid & (hit.shape == shape), by_uid[shape.id()], labels)
    return depth.astype(np.float32), np.asarray(labels).reshape(height, width).astype(
        np.int32
    )


def render_reference(
    scene_path: Path,
    camera_specs: list[dict[str, Any]] | None,
    width: int,
    height: int,
    output_dir: Path,
    *,
    spp: int = 256,
    seed: int = 0,
    threads: int = 4,
    lighting: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render independent CPU RGB, optical-depth, and canonical-ID references.

    Args:
        scene_path: Approved custom manifest with hash-verified canonical assets.
        camera_specs: Look-at views, or ``None`` for all manifest views. Sequence
            views may include ``frame_index`` and ``object_transforms`` keyed by
            canonical object UID; all poses remain kinematic.
        width: Output width.
        height: Output height.
        output_dir: New directory; existing reference evidence is never replaced.
        spp: Path-tracing samples per pixel, recorded in the reference manifest.
        seed: Deterministic sampler seed; consecutive views use consecutive seeds.
        threads: CPU worker limit for Mitsuba/Dr.Jit.
        lighting: Explicit physical values ``directional_irradiance`` and
            ``environment_radiance``. Defaults (3 and .15) are benchmark policy,
            not estimates of the tested backends' native units.

    Returns:
        Manifest with immutable frame paths, checksums, and reference provenance.

    Raises:
        RuntimeError: Optional Mitsuba dependencies are unavailable.

    .. attention::
        Repeat-seed noise is measured and may fail its frozen noise budget.
        Generating a reference does not qualify photometric calibration.
    """
    import copy

    if min(width, height, spp, threads) < 1 or seed < 0:
        raise ValueError(
            "Dimensions, spp and threads must be positive; seed nonnegative"
        )
    availability = reference_dependencies()
    if not availability["available"]:
        raise RuntimeError("; ".join(availability["missing_reasons"]))
    import numpy as np

    scene_path = Path(scene_path).resolve()
    frozen = load_scene(scene_path)
    cameras = camera_specs if camera_specs is not None else frozen["cameras"]
    if not cameras:
        raise ValueError("At least one camera is required")
    lighting = dict(
        lighting or {"directional_irradiance": 3.0, "environment_radiance": 0.15}
    )
    for key in ("directional_irradiance", "environment_radiance"):
        if not math.isfinite(lighting[key]) or lighting[key] < 0:
            raise ValueError(f"Invalid physical lighting value {key}")

    import mitsuba as mi
    import drjit as dr

    mi.set_variant("llvm_ad_rgb")
    dr.set_thread_count(threads)
    samples_per_pass = _samples_per_pass(spp, width, height)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema_version": 1,
        "kind": "independent_offline",
        "renderer": "mitsuba",
        "renderer_version": mi.__version__,
        "variant": "llvm_ad_rgb",
        "device": "cpu",
        "source_scene_sha256": scene_fingerprint(frozen),
        "source_manifest_file_sha256": _sha256(scene_path),
        "implementation_sha256": _sha256(Path(__file__)),
        "assets_verified": True,
        "source_assets": [
            {
                "uid": obj["uid"],
                "path": obj["geometry"]["path"],
                "sha256": obj["geometry"]["sha256"],
            }
            for obj in frozen["objects"]
            if obj["geometry"]["kind"] == "mesh"
        ],
        "scene_origin": "EmbodiChain custom approved scene; not an official Isaac Lab scene",
        "integrator": {
            "type": "path",
            "max_depth": 8,
            "spp": spp,
            "seed": seed,
            "independent_repeat_count": 2,
            "sampling_policy": "bounded_wavefront/v1",
            "samples_per_pass": samples_per_pass,
            "passes_per_repeat": spp // samples_per_pass,
            "maximum_wavefront_samples": _MAX_WAVEFRONT_SAMPLES,
        },
        "geometry_sampling": "native ray intersections at pixel centres, unfiltered",
        "material_model": "Mitsuba principled; canonical linear base_color/roughness/metallic",
        "lighting": {
            **lighting,
            "direction": frozen["lighting"]["direction"],
            "color": frozen["lighting"]["color"],
        },
        "photometric_calibration": "not_qualified",
        "noise_floor_policy": {
            "metric": "independent_seed_display_srgb_rmse",
            "maximum": 0.01,
            "threshold_origin": "prespecified reference noise budget",
        },
        "versions": availability["versions"],
        "threads": threads,
        "width": width,
        "height": height,
        "camera_specs_sha256": stable_hash(cameras),
    }
    provenance["reference_id"] = stable_hash(provenance)
    manifest = {"schema_version": 1, "provenance": provenance, "frames": []}
    moving_uids = {
        obj["uid"]
        for obj in frozen["objects"]
        if any(
            not np.allclose(
                spec.get("object_transforms", {}).get(
                    obj["uid"], obj["world_from_object"]
                ),
                obj["world_from_object"],
                atol=1e-8,
                rtol=0,
            )
            for spec in cameras
        )
    }
    for index, specification in enumerate(cameras):
        camera = camera_metadata(frozen, specification, width, height)
        objects = copy.deepcopy(frozen["objects"])
        transforms = specification.get("object_transforms", {})
        known = {obj["uid"] for obj in objects}
        if set(transforms) - known:
            raise ValueError("Object transform references an unknown UID")
        for obj in objects:
            if obj["uid"] in transforms:
                obj["world_from_object"] = transforms[obj["uid"]]
        scene_dict = {
            "type": "scene",
            "integrator": {
                "type": "path",
                "max_depth": 8,
                "samples_per_pass": samples_per_pass,
            },
            "sensor": {
                "type": "perspective",
                "fov": frozen["camera_model"]["horizontal_fov_deg"],
                "fov_axis": "x",
                "near_clip": camera["near_m"],
                "far_clip": camera["far_m"],
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=specification["eye"],
                    target=specification["target"],
                    up=specification["up"],
                ),
                "sampler": {"type": "independent", "sample_count": spp},
                "film": {
                    "type": "hdrfilm",
                    "width": width,
                    "height": height,
                    "pixel_format": "rgb",
                    "component_format": "float32",
                    "rfilter": {"type": "box"},
                },
            },
            "sun": {
                "type": "directional",
                "direction": frozen["lighting"]["direction"],
                "irradiance": {
                    "type": "rgb",
                    "value": (
                        np.asarray(frozen["lighting"]["color"])
                        * lighting["directional_irradiance"]
                    ).tolist(),
                },
            },
            "environment": {
                "type": "constant",
                "radiance": {"type": "rgb", "value": lighting["environment_radiance"]},
            },
        }
        for obj in objects:
            scene_dict["object_" + str(obj["instance_id"])] = _mesh(
                mi, obj, scene_path.parent
            )
        rendered_scene = mi.load_dict(scene_dict)
        rgb = np.asarray(
            mi.render(rendered_scene, seed=seed + 2 * index, spp=spp)
        ).copy()
        rgb_repeat = np.asarray(
            mi.render(rendered_scene, seed=seed + 2 * index + 1, spp=spp)
        ).copy()
        display = []
        for array in (rgb, rgb_repeat):
            clipped = np.clip(array, 0, 1)
            display.append(
                np.where(
                    clipped <= 0.0031308,
                    12.92 * clipped,
                    1.055 * clipped ** (1 / 2.4) - 0.055,
                )
            )
        noise_rmse = float(np.sqrt(np.mean((display[0] - display[1]) ** 2)))
        reference_validation = {
            "metric": "independent_seed_display_srgb_rmse",
            "value": noise_rmse,
            "threshold": provenance["noise_floor_policy"]["maximum"],
            "passed": noise_rmse <= provenance["noise_floor_policy"]["maximum"],
            "seeds": [seed + 2 * index, seed + 2 * index + 1],
            "samples_per_pixel_per_repeat": spp,
            "denominator": width * height * 3,
        }
        depth, seg = _geometry_buffers(mi, dr, rendered_scene, camera, objects)
        stem = f"frame_{index:04d}"
        arrays_path = destination / f"{stem}.npz"
        np.savez_compressed(
            arrays_path, rgb=rgb, depth=depth, seg=seg, rgb_repeat=rgb_repeat
        )
        metadata = {
            "backend": "mitsuba",
            "scene_sha256": scene_fingerprint(frozen),
            "camera": camera,
            "rgb_encoding": "linear_rec709",
            "depth_semantics": DEPTH_SEMANTICS,
            "segmentation_semantics": SEGMENTATION_SEMANTICS,
            "instance_ids": [obj["instance_id"] for obj in objects],
            "world_from_instances": {
                str(obj["instance_id"]): obj["world_from_object"] for obj in objects
            },
            "moving_instance_ids": [
                obj["instance_id"] for obj in objects if obj["uid"] in moving_uids
            ],
            "frame_index": specification.get(
                "frame_index", specification.get("frame_id", index)
            ),
            "frame_id": specification.get(
                "frame_id", specification.get("frame_index", index)
            ),
            "time_s": specification.get("time_s", index / 60),
            "view_sha256": stable_hash(camera),
            "profile_sha256": stable_hash(provenance["lighting"]),
            "photometric_calibration": "not_qualified",
        }
        frame = {
            "schema_version": 1,
            "metadata": metadata,
            "reference_provenance": provenance,
            "reference_validation": reference_validation,
            "arrays": {"path": arrays_path.name, "sha256": _sha256(arrays_path)},
        }
        metadata_path = destination / f"{stem}.json"
        write_json(metadata_path, frame)
        manifest["frames"].append(
            {
                "path": metadata_path.name,
                "sha256": _sha256(metadata_path),
                "camera_name": camera["name"],
                "frame_index": metadata["frame_index"],
                "frame_id": metadata["frame_id"],
                "time_s": metadata["time_s"],
            }
        )
    write_json(destination / "reference.json", manifest)
    return manifest


def load_reference_frame(path: Path) -> dict[str, Any]:
    """Load a persisted reference frame after verifying its arrays' checksum.

    Args:
        path: Frame metadata JSON produced by :func:`render_reference`.

    Returns:
        Arrays, metadata, provenance, and checksummed evidence for evaluation.
    """
    import numpy as np

    path = Path(path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    array_path = (path.parent / record["arrays"]["path"]).resolve()
    if not array_path.is_relative_to(path.parent):
        raise ValueError("Reference arrays must remain inside their evidence directory")
    if _sha256(array_path) != record["arrays"]["sha256"]:
        raise ValueError("Reference arrays checksum mismatch")
    manifest_path = path.parent / "reference.json"
    if not manifest_path.is_file():
        raise ValueError("Independent reference manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matched_frame = next(
        (item for item in manifest["frames"] if item["path"] == path.name), None
    )
    if matched_frame is None or matched_frame["sha256"] != _sha256(path):
        raise ValueError("Reference frame metadata checksum mismatch")
    if record["reference_provenance"] != manifest["provenance"]:
        raise ValueError("Reference frame provenance differs from its manifest")
    with np.load(array_path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in ("rgb", "depth", "seg")}
    return {
        **arrays,
        "metadata": record["metadata"],
        "reference_provenance": record["reference_provenance"],
        "reference_validation": record.get("reference_validation", {}),
        "artifacts": [
            {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
                "kind": "reference_manifest",
            },
            {"path": str(path), "sha256": _sha256(path), "kind": "reference_metadata"},
            {
                "path": str(array_path),
                "sha256": _sha256(array_path),
                "kind": "reference_arrays",
            },
        ],
    }
