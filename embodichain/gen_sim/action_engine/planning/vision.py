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

"""Auditable multi-view observation and VLM fact extraction."""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
import json
import math
import os
from typing import Any

import torch

from embodichain.gen_sim.action_engine.domain import public_task_spec

__all__ = [
    "CameraObservation",
    "SceneObservation",
    "analyze_visual_scene",
    "collect_scene_observation",
    "validate_visual_facts",
]

StructuredCaller = Callable[..., Mapping[str, Any]]

_VISUAL_ENTITY_KEYS = frozenset(
    {
        "uid",
        "camera_uid",
        "bbox",
        "keypoints",
        "visible",
        "confidence",
    }
)
_VISUAL_RELATION_KEYS = frozenset({"type", "uids", "confidence"})


@dataclass(frozen=True)
class CameraObservation:
    """One live camera sample with calibration for one vectorized env row."""

    uid: str
    rgb: torch.Tensor
    depth: torch.Tensor | None
    intrinsics: torch.Tensor | None
    extrinsics: torch.Tensor | None


@dataclass(frozen=True)
class SceneObservation:
    """Multi-view evidence and stable simulator entity IDs for online planning."""

    cameras: tuple[CameraObservation, ...]
    entities: tuple[dict[str, Any], ...]
    env_id: int = 0


_VISUAL_FACTS_SCHEMA = {
    "title": "ActionEngineVisualFacts",
    "type": "object",
    "additionalProperties": False,
    "required": ["entities", "relations", "confidence"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["uid", "camera_uid", "confidence"],
                "properties": {
                    "uid": {"type": "string"},
                    "camera_uid": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "number"},
                    },
                    "keypoints": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "number"},
                        },
                    },
                    "visible": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "uids", "confidence"],
                "properties": {
                    "type": {"type": "string"},
                    "uids": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

# Visual facts are deliberately a much smaller contract than a simulator
# snapshot.  In particular, accepting arbitrary nested ``attributes`` would
# let a caller smuggle poses/qpos into the online planner while still passing
# the top-level schema.  Keep the deny-list here (rather than relying only on
# the SeedGraph validator) because visual facts are persisted and may be
# consumed by an independent planner implementation.
_FORBIDDEN_LIVE_KEYS = frozenset(
    {
        "absolute_position",
        "coordinates",
        "extrinsics",
        "grasp_pose",
        "joint_positions",
        "live_pose",
        "live_transform",
        "object_pose",
        "oracle",
        "pose",
        "positions",
        "qpos",
        "target_pose",
        "trajectory",
        "transform",
        "waypoints",
        "xpos",
    }
)


def collect_scene_observation(
    env: Any,
    *,
    camera_uids: Sequence[str] | None = None,
    env_id: int = 0,
) -> SceneObservation:
    """Capture current RGB/depth/calibration and a simulator entity inventory."""
    if env_id < 0 or env_id >= int(env.num_envs):
        raise ValueError("env_id is outside the vectorized environment range.")
    sim = env.sim
    uids = (
        list(camera_uids)
        if camera_uids is not None
        else list(sim.get_sensor_uid_list())
    )
    cameras = []
    for uid in uids:
        sensor = sim.get_sensor(str(uid))
        if sensor is None:
            raise ValueError(f"Unknown camera UID {uid!r}.")
        update = getattr(sensor, "update", None)
        if callable(update):
            update()
        data = sensor.get_data()
        if not isinstance(data, Mapping):
            raise TypeError(f"Camera {uid!r} returned non-mapping sensor data.")
        rgb_data = data.get("color", data.get("rgb"))
        if rgb_data is None:
            raise ValueError(f"Camera {uid!r} does not provide RGB data.")
        rgb = (
            _env_row(
                rgb_data,
                env_id,
                num_envs=int(env.num_envs),
                unbatched_ndim=3,
            )
            .detach()
            .cpu()
        )
        depth = (
            _env_row(
                data["depth"],
                env_id,
                num_envs=int(env.num_envs),
                unbatched_ndim=2,
            )
            .detach()
            .cpu()
            if data.get("depth") is not None
            else None
        )
        intrinsics = _optional_call(
            sensor, "get_intrinsics", env_id, num_envs=int(env.num_envs)
        )
        extrinsics = _optional_call(
            sensor,
            "get_arena_pose",
            env_id,
            num_envs=int(env.num_envs),
            to_matrix=True,
        )
        cameras.append(
            CameraObservation(
                uid=str(uid),
                rgb=rgb,
                depth=depth,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
            )
        )
    if not cameras:
        raise ValueError("Online visual planning requires at least one camera.")

    entity_uids = list(sim.get_rigid_object_uid_list())
    articulation_uids = getattr(sim, "get_articulation_uid_list", lambda: [])()
    entities = []
    for uid in [*entity_uids, *articulation_uids]:
        item: dict[str, Any] = {"uid": str(uid)}
        # Do not expose live simulator transforms to the online planner.  The
        # VLM receives RGB/depth evidence and stable UIDs only; JIT grounding
        # resolves world-space targets inside the runtime immediately before
        # each action.  This also prevents an accidental pose oracle through
        # the entity inventory prompt.
        entities.append(item)
    return SceneObservation(tuple(cameras), tuple(entities), env_id=env_id)


def analyze_visual_scene(
    observation: SceneObservation,
    task_spec: Mapping[str, Any],
    *,
    model: str | None = None,
    caller: StructuredCaller | None = None,
    call_counter: list[int] | None = None,
) -> dict[str, Any]:
    """Ask a VLM for auditable facts, never hidden reasoning or an action plan."""
    _reject_live_fields(observation.entities, "SceneObservation.entities")
    public = public_task_spec(task_spec)
    _reject_live_fields(public, "PublicTaskSpec")
    camera_manifest, images = _camera_evidence(observation)
    prompt = (
        "Inspect every supplied camera view. Return only observable facts needed "
        "for the task. Refer to simulator entities only by the supplied UID. "
        "Use normalized [0,1] bbox/keypoint values, state uncertainty explicitly, "
        "and do not provide reasoning or actions. The image blocks appear in the "
        "camera_evidence order: each RGB image is followed by that camera's "
        "normalized depth image when depth_image_index is present. Camera "
        "calibration is input evidence only; never reproduce it in the facts.\n\n"
        f"TaskSpec:\n{json.dumps(public, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Entity inventory:\n{json.dumps(observation.entities, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Camera evidence:\n{json.dumps(camera_manifest, ensure_ascii=False, sort_keys=True)}"
    )
    invoke = caller or _default_structured_caller
    # Test/mocked callers own their transport and may intentionally receive no
    # configured model.  The production caller must resolve strictly through
    # the visual-model priority rather than falling back to a text-only model.
    selected_model = model if caller is not None else _vlm_model(model)
    first_error: Exception | None = None
    for attempt in range(2):
        current_prompt = prompt
        if first_error is not None:
            current_prompt += (
                "\n\nThe previous visual-facts JSON was invalid. Return corrected "
                f"JSON only. Validation error: {first_error}"
            )
        try:
            if call_counter is not None:
                call_counter[0] += 1
            response = invoke(
                prompt=current_prompt,
                images=images,
                schema=_VISUAL_FACTS_SCHEMA,
                model=selected_model,
            )
            facts = validate_visual_facts(
                response,
                known_uids={str(item["uid"]) for item in observation.entities},
                camera_uids={camera.uid for camera in observation.cameras},
            )
            if facts["confidence"] < 0.5:
                raise ValueError(
                    "VLM visual facts confidence is below the required 0.5 threshold."
                )
            if not any(
                item.get("visible", True) and item["confidence"] >= 0.5
                for item in facts["entities"]
            ):
                raise ValueError("VLM visual facts contain no reliable visible entity.")
            return facts
        except (TypeError, ValueError) as error:
            if attempt:
                raise ValueError(
                    "VLM visual facts failed validation after one repair: " f"{error}"
                ) from error
            first_error = error
    raise AssertionError("unreachable")


def validate_visual_facts(
    value: Mapping[str, Any],
    *,
    known_uids: set[str],
    camera_uids: set[str],
) -> dict[str, Any]:
    """Validate entity identity and normalized image-space evidence."""
    if not isinstance(value, Mapping):
        raise TypeError("VLM visual facts must be a mapping.")
    unknown = set(value) - {"entities", "relations", "confidence"}
    if unknown:
        raise ValueError(
            f"VLM visual facts contain unsupported fields: {sorted(unknown)}."
        )
    confidence = _confidence(value.get("confidence"), "confidence")
    entities = value.get("entities")
    relations = value.get("relations")
    if not isinstance(entities, Sequence) or isinstance(entities, (str, bytes)):
        raise ValueError("VLM visual facts entities must be a list.")
    if not isinstance(relations, Sequence) or isinstance(relations, (str, bytes)):
        raise ValueError("VLM visual facts relations must be a list.")
    normalized_entities = []
    for index, item in enumerate(entities):
        if not isinstance(item, Mapping):
            raise ValueError(f"visual entities[{index}] must be a mapping.")
        unsupported = set(item) - _VISUAL_ENTITY_KEYS
        if unsupported:
            raise ValueError(
                f"visual entities[{index}] contains unsupported fields "
                f"{sorted(unsupported)}."
            )
        uid = item.get("uid")
        camera_uid = item.get("camera_uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError(
                f"visual entities[{index}].uid must be a non-empty string."
            )
        if not isinstance(camera_uid, str) or not camera_uid:
            raise ValueError(
                f"visual entities[{index}].camera_uid must be a non-empty string."
            )
        if uid not in known_uids:
            raise ValueError(
                f"visual entities[{index}] references unknown UID {uid!r}."
            )
        if camera_uid not in camera_uids:
            raise ValueError(
                f"visual entities[{index}] references unknown camera {camera_uid!r}."
            )
        normalized = dict(item)
        _reject_live_fields(normalized, f"visual entities[{index}]")
        if "visible" in normalized and not isinstance(normalized["visible"], bool):
            raise ValueError(f"visual entities[{index}].visible must be a boolean.")
        if "bbox" in normalized:
            normalized["bbox"] = _normalized_vector(
                normalized["bbox"], 4, f"visual entities[{index}].bbox"
            )
            x_min, y_min, x_max, y_max = normalized["bbox"]
            if x_min >= x_max or y_min >= y_max:
                raise ValueError(
                    f"visual entities[{index}].bbox must have non-zero ordered bounds."
                )
        keypoints = normalized.get("keypoints", {})
        if not isinstance(keypoints, Mapping):
            raise ValueError(f"visual entities[{index}].keypoints must be a mapping.")
        normalized["keypoints"] = {
            str(name): _normalized_vector(point, 2, f"keypoint {name!r}")
            for name, point in keypoints.items()
        }
        if (
            normalized.get("visible", True)
            and "bbox" not in normalized
            and not normalized["keypoints"]
        ):
            raise ValueError(
                f"visual entities[{index}] must include a bbox or keypoint evidence."
            )
        normalized["confidence"] = _confidence(
            normalized.get("confidence"), f"visual entities[{index}].confidence"
        )
        normalized_entities.append(normalized)
    normalized_relations = []
    for index, relation in enumerate(relations):
        if not isinstance(relation, Mapping):
            raise ValueError(f"visual relations[{index}] must be a mapping.")
        unsupported = set(relation) - _VISUAL_RELATION_KEYS
        if unsupported:
            raise ValueError(
                f"visual relations[{index}] contains unsupported fields "
                f"{sorted(unsupported)}."
            )
        relation_type = relation.get("type")
        if not isinstance(relation_type, str) or not relation_type:
            raise ValueError(f"visual relations[{index}].type must be non-empty.")
        participants = relation.get("uids", [])
        if not isinstance(participants, Sequence) or isinstance(
            participants, (str, bytes)
        ):
            raise ValueError(f"visual relations[{index}].uids must be a list.")
        if any(not isinstance(uid, str) or not uid for uid in participants):
            raise ValueError(
                f"visual relations[{index}].uids must contain non-empty strings."
            )
        invalid = set(participants) - known_uids
        if invalid:
            raise ValueError(
                f"visual relations[{index}] has unknown UIDs {sorted(invalid)}."
            )
        normalized = dict(relation)
        _reject_live_fields(normalized, f"visual relations[{index}]")
        normalized["confidence"] = _confidence(
            normalized.get("confidence"), f"visual relations[{index}].confidence"
        )
        normalized_relations.append(normalized)
    _reject_live_fields(
        {"entities": normalized_entities, "relations": normalized_relations},
        "VLM visual facts",
    )
    return {
        "entities": normalized_entities,
        "relations": normalized_relations,
        "confidence": confidence,
    }


def _default_structured_caller(
    *,
    prompt: str,
    images: Sequence[str],
    schema: Mapping[str, Any],
    model: str | None,
) -> Mapping[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from .planner import (
        _coerce_model_response,
        _is_mimo_compatible,
        _load_llm_settings,
        _structured_output_runnable,
    )

    settings = _load_llm_settings(model=model)
    kwargs: dict[str, Any] = {
        "api_key": settings["api_key"],
        "model": settings["model"],
        "temperature": 0,
    }
    for key in ("base_url", "default_query"):
        if settings[key]:
            kwargs[key] = settings[key]
    if _is_mimo_compatible(settings):
        kwargs.update(
            {
                "max_completion_tokens": 4096,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
        )
    client = ChatOpenAI(**kwargs)
    structured = _structured_output_runnable(
        client,
        schema,
        settings=settings,
    )
    schema_prompt = (
        f"{prompt}\n\nReturn one JSON object conforming exactly to this JSON "
        f"Schema:\n{json.dumps(schema, ensure_ascii=False, sort_keys=True)}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content[0]["text"] = schema_prompt
    content.extend(
        {"type": "image_url", "image_url": {"url": image}} for image in images
    )
    response = structured.invoke(
        [
            SystemMessage(
                content="Report visual facts only. Never reveal chain-of-thought."
            ),
            HumanMessage(content=content),
        ]
    )
    return _coerce_model_response(response)


def _vlm_model(explicit: str | None) -> str:
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    from .planner import _GEN_SIM_ENV_PATH, _load_env_file

    local_env = _load_env_file(_GEN_SIM_ENV_PATH)
    # A VLM-specific choice wins over the generic OpenAI default regardless of
    # whether it comes from the shell or the project dotenv. Within each name,
    # process variables retain their normal override behavior.
    for key in ("ACTION_ENGINE_VLM_MODEL", "OPENAI_MODEL"):
        for source in (os.environ, local_env):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError(
        "A VLM model is required through --vlm-model, agent_config.vlm_model, "
        "ACTION_ENGINE_VLM_MODEL, or OPENAI_MODEL."
    )


def _rgb_data_url(value: torch.Tensor) -> str:
    from PIL import Image

    image = value
    if image.ndim != 3 or image.shape[-1] not in {3, 4}:
        raise ValueError("Camera RGB must have shape (H, W, 3|4).")
    if image.dtype != torch.uint8:
        image = image.float()
        if float(image.max()) <= 1.0:
            image = image * 255.0
        image = image.clamp(0, 255).to(torch.uint8)
    stream = BytesIO()
    Image.fromarray(image.numpy()).convert("RGB").save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode(
        "ascii"
    )


def _camera_evidence(
    observation: SceneObservation,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Package calibrated RGB/depth evidence in a stable camera order."""
    manifest: list[dict[str, Any]] = []
    images: list[str] = []
    for camera in observation.cameras:
        rgb_index = len(images)
        images.append(_rgb_data_url(camera.rgb))
        item: dict[str, Any] = {
            "uid": camera.uid,
            "rgb_image_index": rgb_index,
            "depth_available": camera.depth is not None,
            "intrinsics": _calibration_list(camera.intrinsics),
            "extrinsics": _calibration_list(camera.extrinsics),
        }
        if camera.depth is not None:
            item["depth_image_index"] = len(images)
            images.append(_depth_data_url(camera.depth))
        manifest.append(item)
    return manifest, images


def _calibration_list(value: torch.Tensor | None) -> list[Any] | None:
    """Serialize finite calibration tensors for the transient VLM prompt."""
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().cpu()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Camera calibration contains non-finite values.")
    return tensor.tolist()


def _depth_data_url(value: torch.Tensor) -> str:
    """Render one depth frame as a normalized grayscale VLM evidence image."""
    from PIL import Image

    depth = torch.as_tensor(value).detach().cpu().float()
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError("Camera depth must have shape (H, W) or a singleton channel.")
    finite = torch.isfinite(depth)
    if not bool(finite.any()):
        raise ValueError("Camera depth contains no finite values.")
    minimum = depth[finite].min()
    maximum = depth[finite].max()
    normalized = torch.zeros_like(depth)
    if float(maximum - minimum) > 0.0:
        normalized[finite] = (depth[finite] - minimum) / (maximum - minimum)
    image = (normalized.clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
    stream = BytesIO()
    Image.fromarray(image, mode="L").save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode(
        "ascii"
    )


def _env_row(
    value: Any,
    env_id: int,
    *,
    num_envs: int | None = None,
    unbatched_ndim: int | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Select one vectorized environment row without slicing image dimensions.

    Sensor APIs return either ``(num_envs, ...)`` or an unbatched ``(...)``
    tensor.  The old ``shape[0] > env_id`` heuristic sliced the first image row
    for an unbatched ``(H, W, C)`` RGB tensor and similarly corrupted 4x4 poses.
    Prefer the known environment count and only use the legacy heuristic when
    no count is available.
    """
    tensor = torch.as_tensor(value)
    if unbatched_ndim is not None:
        allowed_ndim = (
            (unbatched_ndim,)
            if isinstance(unbatched_ndim, int)
            else tuple(unbatched_ndim)
        )
        if tensor.ndim in allowed_ndim:
            return tensor
    if tensor.ndim and num_envs is not None and tensor.shape[0] == int(num_envs):
        if env_id >= tensor.shape[0]:
            raise ValueError("env_id is outside the sensor batch dimension.")
        return tensor[env_id]
    if num_envs is None and tensor.ndim and tensor.shape[0] > env_id:
        return tensor[env_id]
    return tensor


def _optional_call(
    sensor: Any, name: str, env_id: int, *, num_envs: int | None = None, **kwargs: Any
) -> torch.Tensor | None:
    method = getattr(sensor, name, None)
    if not callable(method):
        return None
    try:
        value = method(env_id=env_id, **kwargs)
    except TypeError:
        try:
            value = method(env_id, **kwargs)
        except TypeError:
            value = method(**kwargs)
    value = torch.as_tensor(value)
    # Calibration methods commonly return an unbatched matrix even for a
    # vectorized simulator.  Select a leading environment row only when the
    # shape cannot itself be a canonical calibration matrix.  This preserves
    # 3x3/4x4 matrices while correctly handling batched compact vectors such as
    # ``(num_envs, 4)``.
    unbatched_matrix = value.ndim == 2 and tuple(value.shape) in {
        (3, 3),
        (4, 4),
    }
    if (
        num_envs is not None
        and value.ndim >= 1
        and value.shape[0] == int(num_envs)
        and not unbatched_matrix
    ):
        value = value[env_id]
    return value.detach().cpu()


def _reject_live_fields(value: Any, context: str) -> None:
    """Reject nested simulator state/geometry fields in VLM facts."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_LIVE_KEYS:
                raise ValueError(
                    f"{context} contains forbidden live-state field {key!r}."
                )
            _reject_live_fields(child, f"{context}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_live_fields(child, f"{context}[{index}]")


def _normalized_vector(value: Any, size: int, context: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != size
    ):
        raise ValueError(f"{context} must contain {size} normalized values.")
    if any(
        not isinstance(item, (int, float)) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{context} values must be numeric.")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in result):
        raise ValueError(f"{context} values must lie in [0, 1].")
    return result


def _confidence(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{context} must be a number in [0, 1].")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{context} must lie in [0, 1].")
    return result
