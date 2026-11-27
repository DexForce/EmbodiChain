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

"""Native rollout appearance and verification of archived visual variants."""

from __future__ import annotations
from contextlib import contextmanager, ExitStack
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from embodichain.data_pipeline.recording.media import (
    _digest,
    verify_media_manifest,
)
from .persistence import ArtifactEpisodeSink
from .artifacts import resolve_artifact_path

__all__ = [
    "load_visual_profiles",
    "inspect_observation_variants",
]


def load_visual_profiles(path: str | Path) -> dict[str, dict]:
    """Load closed native material, camera and light profile declarations."""
    import yaml

    value = yaml.safe_load(Path(path).read_text())
    if (
        not isinstance(value, dict)
        or set(value) != {"profiles"}
        or not isinstance(value["profiles"], dict)
    ):
        raise ValueError("visual registry must contain a profiles mapping")
    result = value["profiles"]
    for name, profile in result.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(profile, dict)
            or set(profile) - {"materials", "cameras", "lights"}
        ):
            raise ValueError("invalid visual profile")
        for kind, fields in (
            ("materials", {"base_color", "roughness", "metallic", "ior"}),
            ("cameras", {"position_delta", "focal_scale"}),
            ("lights", {"color", "intensity"}),
        ):
            entries = profile.get(kind, {})
            if not isinstance(entries, dict):
                raise ValueError(f"{kind} must be a mapping")
            for uid, params in entries.items():
                if (
                    not isinstance(uid, str)
                    or not uid
                    or not isinstance(params, dict)
                    or set(params) - fields
                ):
                    raise ValueError(f"invalid {kind} parameters")
                for key, item in params.items():
                    values = item if isinstance(item, (list, tuple)) else [item]
                    if any(
                        type(v) not in (float, int) or not math.isfinite(v)
                        for v in values
                    ):
                        raise ValueError("visual values must be finite numbers")
                    if key in {"base_color", "color", "position_delta"}:
                        sizes = (3, 4) if key == "base_color" else (3,)
                        if len(values) not in sizes:
                            raise ValueError("invalid visual vector length")
                    elif isinstance(item, (list, tuple)):
                        raise ValueError("expected scalar visual value")
                    if key in {"base_color", "color", "roughness", "metallic"} and any(
                        not 0 <= v <= 1 for v in values
                    ):
                        raise ValueError("visual color/PBR values must be in [0,1]")
                    if key in {"focal_scale", "ior"} and item <= 0:
                        raise ValueError("visual scale must be positive")
                    if key == "intensity" and item < 0:
                        raise ValueError("light intensity must be nonnegative")
    return result


def _calibration_batch(env):
    """Read each camera's full calibration once at the current control boundary."""
    rows = [{} for _ in range(env.num_envs)]
    for uid in env.sim.get_sensor_uid_list():
        camera = env.sim.get_sensor(uid)
        if not callable(getattr(camera, "get_intrinsics", None)):
            continue
        intrinsic = camera.get_intrinsics()
        if isinstance(intrinsic, tuple):
            poses = camera.get_left_right_arena_pose()
            values = {
                "intrinsics_left": intrinsic[0].detach().cpu(),
                "intrinsics_right": intrinsic[1].detach().cpu(),
                "arena_pose_left": poses[0].detach().cpu(),
                "arena_pose_right": poses[1].detach().cpu(),
            }
        else:
            values = {
                "intrinsics": intrinsic.detach().cpu(),
                "arena_pose": camera.get_arena_pose(to_matrix=True).detach().cpu(),
            }
        for row in range(env.num_envs):
            rows[row][uid] = {key: value[row].tolist() for key, value in values.items()}
            rows[row][uid].update(
                width=camera.cfg.width,
                height=camera.cfg.height,
                camera_axes="opengl",
                pose_convention="camera_to_arena_z_up",
            )
    return tuple(rows)


def _camera_calibration(env, row):
    return _calibration_batch(env)[row]


def _calibrated_frames(env, observation, row, *, calibration=None):
    from ._gym_utils import _images

    result = _images(observation, row)
    for uid, values in (
        calibration if calibration is not None else _camera_calibration(env, row)
    ).items():
        for name, value in values.items():
            if isinstance(value, list):
                result[f"calibration/{uid}/{name}"] = np.asarray(
                    value, dtype=np.float32
                )
    return result


def _profile_revision(profile):
    return hashlib.sha256(
        json.dumps(
            profile, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _variant_key(parent_commit_id, profile_id, revision):
    return hashlib.sha256(
        json.dumps((parent_commit_id, profile_id, revision)).encode()
    ).hexdigest()


def inspect_observation_variants(directory: str | Path) -> tuple[dict, ...]:
    """Verify archived observation children against their physical parents."""
    root = Path(directory)
    markers = sorted((root / "observations").glob("*.json"))
    if not markers:
        return ()
    parents = {
        receipt.commit_id: receipt for receipt in ArtifactEpisodeSink(root).inspect()
    }
    results = []
    for path in markers:
        value = json.loads(path.read_text())
        value["media_manifest"] = str(
            resolve_artifact_path(root, value["media_manifest"], legacy_absolute=True)
        )
        expected_key = _variant_key(
            value["parent_commit_id"], value["profile_id"], value["profile_revision"]
        )
        if value["variant_id"] != expected_key or path.stem != expected_key:
            raise ValueError("observation variant identity changed")
        if value.get("schema_version") == 2 and value[
            "profile_revision"
        ] != _profile_revision(value["profile"]):
            raise ValueError("observation profile content changed")
        parent = parents.get(value["parent_commit_id"])
        if (
            parent is None
            or parent.episode_id != value["parent_episode_id"]
            or value["physics_steps"] != 0
        ):
            raise ValueError("observation variant has no verified physical parent")
        if (
            _digest(Path(parent.storage_id)) != value["source_sha256"]
            or _digest(Path(value["media_manifest"])) != value["media_sha256"]
        ):
            raise ValueError("observation variant content changed")
        with np.load(parent.storage_id, allow_pickle=False) as data:
            times = data["timestamps"]
        verify_media_manifest(
            value["media_manifest"],
            expected_frames=len(times),
            parent_episode_id=parent.episode_id,
            expected_timestamps=times,
        )
        results.append(value)
    return tuple(results)


@contextmanager
def _visual_profile(env, profile):
    from embodichain.lab.sim import VisualMaterialCfg

    materials, cameras, lights = {}, {}, {}
    for uid, values in profile.get("materials", {}).items():
        obj = env.sim.get_rigid_object(uid)
        if obj is None or any(item is None for item in obj.get_visual_material_inst()):
            raise ValueError(f"visual material target {uid!r} is unavailable")
        params = dict(values)
        if "base_color" in params and len(params["base_color"]) == 3:
            params["base_color"] = [*params["base_color"], 1.0]
        key = (
            "expansion_"
            + hashlib.sha256(
                json.dumps((uid, params), sort_keys=True).encode()
            ).hexdigest()[:20]
        )
        material = env.sim.get_visual_material(key)
        if material is None:
            material = env.sim.create_visual_material(
                VisualMaterialCfg(uid=key, **params)
            )
        materials[uid] = (obj, material)
    for uid in profile.get("cameras", {}):
        camera = env.sim.get_sensor(uid)
        if camera is None or not callable(getattr(camera, "get_intrinsics", None)):
            raise ValueError(f"camera {uid!r} is unavailable")
        intrinsic = camera.get_intrinsics()
        cameras[uid] = (
            camera,
            camera.get_local_pose(to_matrix=True).clone(),
            (
                tuple(v.clone() for v in intrinsic)
                if isinstance(intrinsic, tuple)
                else intrinsic.clone()
            ),
        )
    for uid in profile.get("lights", {}):
        light = env.sim.get_light(uid)
        if light is None:
            raise ValueError(f"light {uid!r} is unavailable")
        lights[uid] = (light, light.get_color(), light.get_intensity())
    with ExitStack() as stack:
        for obj, material in materials.values():
            stack.enter_context(obj.preserve_visual_material())
            obj.set_visual_material(material, shared=False, update_default=False)
        try:
            for uid, (camera, pose, intrinsic) in cameras.items():
                values = profile["cameras"][uid]
                changed = pose.clone()
                changed[:, :3, 3] += torch.tensor(
                    values.get("position_delta", [0.0, 0.0, 0.0]), device=pose.device
                )
                camera.set_local_pose(changed)

                def scaled(value):
                    value = value.clone()
                    value[:, 0, 0] *= values.get("focal_scale", 1.0)
                    value[:, 1, 1] *= values.get("focal_scale", 1.0)
                    return value

                if isinstance(intrinsic, tuple):
                    camera.set_intrinsics(*(scaled(v) for v in intrinsic))
                else:
                    camera.set_intrinsics(scaled(intrinsic))
            for uid, (light, _, _) in lights.items():
                values = profile["lights"][uid]
                if "color" in values:
                    light.set_color(torch.tensor(values["color"]))
                if "intensity" in values:
                    light.set_intensity(torch.tensor(values["intensity"]))
            yield
        finally:
            for camera, pose, intrinsic in cameras.values():
                camera.set_local_pose(pose)
                if isinstance(intrinsic, tuple):
                    camera.set_intrinsics(*intrinsic)
                else:
                    camera.set_intrinsics(intrinsic)
            for light, color, intensity in lights.values():
                light.set_color(color)
                light.set_intensity(intensity)
