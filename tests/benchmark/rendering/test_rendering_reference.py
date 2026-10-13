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

"""Independent CPU renderer, canonical geometry and persisted reference evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark.core.artifacts import stable_hash, write_json
from scripts.benchmark.rendering import reference
from scripts.benchmark.rendering import reference_precision
from scripts.benchmark.rendering.reference import (
    camera_metadata,
    load_reference_frame,
    render_reference,
)
from scripts.benchmark.rendering.scenes import load_scene, scene_fingerprint


def _scene(directory: Path) -> Path:
    directory.mkdir()
    objects = []
    for uid, instance, x, colour in (
        ("red_left", 7, -0.45, [0.6, 0.02, 0.02]),
        ("green_right", 19, 0.45, [0.02, 0.6, 0.02]),
    ):
        transform = np.eye(4)
        transform[:3, 3] = [x, 0, 2]
        objects.append(
            {
                "uid": uid,
                "instance_id": instance,
                "geometry": {"kind": "box", "size": [0.6, 0.8, 0.8]},
                "world_from_object": transform.tolist(),
                "material": {
                    "base_color_linear": colour,
                    "roughness": 0.8,
                    "metallic": 0,
                },
            }
        )
    scene = {
        "schema_version": 1,
        "scene_id": "reference-test",
        "units": "metres",
        "up_axis": "Z",
        "color_space": "linear_rec709",
        "physics_steps": 0,
        "objects": objects,
        "cameras": [
            {"name": "view", "eye": [0, 0, 0], "target": [0, 0, 1], "up": [0, -1, 0]}
        ],
        "camera_model": {"horizontal_fov_deg": 55, "near_m": 0.01, "far_m": 8},
        "lighting": {
            "direction": [0, 0, 1],
            "color": [1, 1, 1],
            "photometric_calibration": "not_qualified",
        },
    }
    path = directory / "scene.json"
    write_json(path, scene)
    return path


def test_camera_metadata_has_pixel_centre_cv_axes(tmp_path: Path) -> None:
    scene = load_scene(_scene(tmp_path / "scene"))
    camera = camera_metadata(scene, scene["cameras"][0], 32, 24)
    np.testing.assert_allclose(camera["world_from_camera"], np.eye(4))
    assert camera["intrinsics"][0][2] == 16
    assert camera["intrinsics"][1][2] == 12
    assert camera["pixel_centres"] == "half_integer"
    assert camera["convention"] == "opencv"


@pytest.mark.parametrize("spp", [3, 256, 4095, 4096, 16384])
@pytest.mark.parametrize("shape", [(128, 128), (512, 512), (1920, 1080)])
def test_reference_pass_budget_preserves_all_requested_samples(
    spp: int, shape: tuple[int, int]
) -> None:
    width, height = shape
    count = reference._samples_per_pass(spp, width, height)
    assert count >= 1
    assert spp % count == 0
    assert count * width * height <= reference._MAX_WAVEFRONT_SAMPLES
    assert (spp // count) * count == spp


def test_real_bounded_passes_keep_independent_seeds_and_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mitsuba")
    monkeypatch.setattr(reference, "_MAX_WAVEFRONT_SAMPLES", 32 * 32 * 2)
    manifest = render_reference(
        _scene(tmp_path / "scene"),
        None,
        32,
        32,
        tmp_path / "reference",
        spp=8,
        threads=2,
    )
    integrator = manifest["provenance"]["integrator"]
    assert integrator["spp"] == 8
    assert integrator["samples_per_pass"] == 2
    assert integrator["passes_per_repeat"] == 4
    frame = load_reference_frame(tmp_path / "reference/frame_0000.json")
    assert frame["reference_validation"]["seeds"] == [0, 1]
    assert frame["reference_validation"]["samples_per_pixel_per_repeat"] == 8
    assert set(np.unique(frame["seg"])) == {0, 7, 19}
    with np.load(tmp_path / "reference/frame_0000.npz") as arrays:
        assert not np.array_equal(arrays["rgb"], arrays["rgb_repeat"])


def test_real_bounded_generator_matches_both_precision_envelopes(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mitsuba")
    scene_path = _scene(tmp_path / "scene")
    scene = load_scene(scene_path)
    signatures = []
    for spp in (4096, 16384):
        identity = {
            "settings": {"spp": spp, "seed": 0, "threads": 8},
            "scene_sha256": scene_fingerprint(scene),
            "reference_implementation_sha256": hashlib.sha256(
                Path(reference.__file__).read_bytes()
            ).hexdigest(),
            "width": 8,
            "height": 8,
        }
        directory = tmp_path / "references" / stable_hash(identity)
        directory.parent.mkdir(exist_ok=True)
        render_reference(scene_path, None, 8, 8, directory, spp=spp, threads=8, seed=0)
        binding = {
            "identity": identity,
            "frame": (directory / "frame_0000.json").relative_to(tmp_path).as_posix(),
            "artifacts": [
                {
                    "path": (directory / name).relative_to(tmp_path).as_posix(),
                    "sha256": hashlib.sha256(
                        (directory / name).read_bytes()
                    ).hexdigest(),
                }
                for name in ("reference.json", "frame_0000.json", "frame_0000.npz")
            ],
        }
        cache = write_json(directory / "cache_binding.json", binding)
        verified = reference_precision._noise_evidence(tmp_path, cache, spp)
        signatures.append(verified["physical_signature_sha256"])
    assert signatures[0] == signatures[1]


def test_optional_missing_renderer_is_explicit_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        reference,
        "reference_dependencies",
        lambda: {
            "available": False,
            "versions": {},
            "missing_reasons": ["Optional package mitsuba is not installed"],
        },
    )
    with pytest.raises(RuntimeError, match="mitsuba is not installed"):
        render_reference(
            _scene(tmp_path / "scene"), None, 16, 16, tmp_path / "reference"
        )
    assert not (tmp_path / "reference").exists()


def _add_mesh(path: Path) -> None:
    scene = load_scene(path)
    asset = path.parent / "triangle.npz"
    np.savez_compressed(
        asset,
        vertices=np.array(
            [[-0.5, -0.5, 1.2], [0, 0.5, 1.2], [0.5, -0.5, 1.2]], dtype=np.float32
        ),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
        normals=np.array([[0, 0, -1]] * 3, dtype=np.float32),
    )
    scene["objects"].append(
        {
            "uid": "canonical_triangle",
            "instance_id": 31,
            "geometry": {
                "kind": "mesh",
                "path": asset.name,
                "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            },
            "world_from_object": np.eye(4).tolist(),
            "material": {
                "base_color_linear": [0.02, 0.02, 0.6],
                "roughness": 0.8,
                "metallic": 0,
            },
        }
    )
    write_json(path, scene)


def test_changed_source_asset_is_rejected_before_renderer_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _scene(tmp_path / "scene")
    _add_mesh(path)
    (path.parent / "triangle.npz").write_bytes(b"modified")
    monkeypatch.setattr(
        reference, "reference_dependencies", lambda: {"available": True, "versions": {}}
    )
    with pytest.raises(ValueError, match="Geometry hash mismatch"):
        render_reference(path, None, 16, 16, tmp_path / "reference")


def test_real_cpu_path_tracer_keeps_colour_geometry_and_asset_provenance(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mitsuba")
    path = _scene(tmp_path / "scene")
    _add_mesh(path)
    manifest = render_reference(
        path, None, 32, 32, tmp_path / "reference", spp=8, threads=2
    )
    frame_path = tmp_path / "reference/frame_0000.json"
    frame = load_reference_frame(frame_path)
    seg, rgb, depth = frame["seg"], frame["rgb"], frame["depth"]
    assert set(np.unique(seg)) == {0, 7, 19, 31}
    assert np.isposinf(depth[seg == 0]).all()
    # Front triangle is a plane at optical Z=1.2, including off-axis pixels.
    np.testing.assert_allclose(depth[seg == 31], 1.2, atol=1e-6)
    assert np.std(rgb) > 0.05
    # RGB and unfiltered geometry share image orientation and colour space.
    assert rgb[seg == 7, 0].mean() > rgb[seg == 7, 1].mean()
    assert rgb[seg == 19, 1].mean() > rgb[seg == 19, 0].mean()
    provenance = manifest["provenance"]
    assert provenance["renderer"] == "mitsuba"
    assert provenance["device"] == "cpu"
    assert provenance["source_scene_sha256"] == scene_fingerprint(load_scene(path))
    assert (
        provenance["source_assets"][0]["sha256"]
        == hashlib.sha256((path.parent / "triangle.npz").read_bytes()).hexdigest()
    )
    assert provenance["reference_id"] == stable_hash(
        {k: v for k, v in provenance.items() if k != "reference_id"}
    )
    assert "not an official Isaac Lab scene" in provenance["scene_origin"]
    assert frame["reference_validation"]["seeds"] == [0, 1]
    assert frame["metadata"]["photometric_calibration"] == "not_qualified"
    with np.load(tmp_path / "reference/frame_0000.npz") as arrays:
        assert "rgb_repeat" in arrays
    assert (
        hashlib.sha256(frame_path.read_bytes()).hexdigest()
        == manifest["frames"][0]["sha256"]
    )


def test_reference_overrides_move_only_changed_instances(tmp_path: Path) -> None:
    pytest.importorskip("mitsuba")
    path = _scene(tmp_path / "scene")
    scene = load_scene(path)
    camera = dict(scene["cameras"][0])
    transforms = {obj["uid"]: obj["world_from_object"] for obj in scene["objects"]}
    transforms["green_right"][0][3] += 0.1
    camera.update(object_transforms=transforms, frame_id=4, time_s=4 / 60)
    render_reference(path, [camera], 16, 16, tmp_path / "reference", spp=2, threads=1)
    frame = load_reference_frame(tmp_path / "reference/frame_0000.json")
    assert frame["metadata"]["moving_instance_ids"] == [19]
    assert frame["metadata"]["frame_id"] == 4


def test_modified_reference_arrays_are_rejected(tmp_path: Path) -> None:
    data = tmp_path / "frame.npz"
    data.write_bytes(b"original")
    record = {
        "arrays": {
            "path": data.name,
            "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        }
    }
    metadata = tmp_path / "frame.json"
    write_json(metadata, record)
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_reference_frame(metadata)


def test_reference_array_path_cannot_escape_evidence_directory(tmp_path: Path) -> None:
    metadata = tmp_path / "frame.json"
    write_json(metadata, {"arrays": {"path": "../external.npz", "sha256": "anything"}})
    with pytest.raises(ValueError, match="inside their evidence directory"):
        load_reference_frame(metadata)


def test_reference_metadata_cannot_change_after_manifest_freeze(tmp_path: Path) -> None:
    arrays = tmp_path / "frame.npz"
    np.savez_compressed(
        arrays, rgb=np.zeros((2, 2, 3)), depth=np.ones((2, 2)), seg=np.ones((2, 2))
    )
    metadata = tmp_path / "frame.json"
    record = {
        "arrays": {
            "path": arrays.name,
            "sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
        },
        "metadata": {"scene_sha256": "original"},
        "reference_provenance": {},
    }
    write_json(metadata, record)
    write_json(
        tmp_path / "reference.json",
        {
            "frames": [
                {
                    "path": metadata.name,
                    "sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                }
            ],
            "provenance": {},
        },
    )
    record["metadata"]["scene_sha256"] = "modified"
    write_json(metadata, record)
    with pytest.raises(ValueError, match="metadata checksum mismatch"):
        load_reference_frame(metadata)
