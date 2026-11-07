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

"""Freeze native mesh/FK and bounded image streams for simulator-free browser replay."""

from __future__ import annotations
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from ._cases import dump

__all__: list[str] = []


def plain(value):
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def javascript(path: Path, kind: str, key: str, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        plain(data), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).replace("<", "\\u003c")
    path.write_text(
        f"window.showcaseRegister({json.dumps(kind)}, {json.dumps(key)}, {payload});\n"
    )


def poses(values):
    values = np.asarray(values)
    output = np.tile(np.eye(4, dtype=np.float32), (len(values), 1, 1))
    output[:, :3, :3] = Rotation.from_quat(values[:, 3:]).as_matrix()
    output[:, :3, 3] = values[:, :3]
    return output


def environment(path: Path):
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
    from embodichain.utils.utility import load_config

    raw = load_config(path)
    cfg = config_to_cfg(raw, source_path=path)
    cfg.num_envs = cfg.sim_cfg.num_envs = 1
    cfg.sim_cfg.device = "cpu"
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.render_cfg.renderer = "no-render"
    cfg.enable_sensor = False
    cfg.dataset = None
    cfg.init_rollout_buffer = cfg.record_trajectory = cfg.trajectory_auto_save = False
    cfg.light.direct, cfg.light.indirect = [], None
    return REGISTERED_ENVS[raw["id"]].make(cfg=cfg)


def _asset_model(env: Any) -> tuple[list[dict], list[str]]:
    from embodichain.lab.visualization.cfg import VisualizationCfg
    from embodichain.lab.visualization.scene_exporter import SceneExporter
    from urllib.parse import unquote

    exporter = SceneExporter(env.sim, VisualizationCfg(env_ids=[0]))
    manifest = exporter.build_manifest()
    geometries = {g.geometry_id: g for g in manifest.geometries}
    meshes, links = [], []
    for node in manifest.nodes:
        target = exporter.resolve_node_target(node.node_id)
        if target is None:
            continue
        uid, kind = target
        if uid == "default_plane":
            continue  # The browser draws a reference floor/grid instead.
        if kind not in {"robot", "rigid"} or node.dynamic_geometry:
            raise ValueError("Preview supports recorded rigid scenes and one robot")
        geometry = geometries[node.geometry_id]
        vertices = np.asarray(geometry.vertices, dtype=np.float32)
        faces = np.asarray(geometry.faces, dtype=np.int32)
        normals = np.zeros_like(vertices)
        a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        face_normals = np.cross(b - a, c - a)
        for column in range(3):
            np.add.at(normals, faces[:, column], face_normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        link = unquote(node.node_id.rsplit("/link:", 1)[1]) if kind == "robot" else None
        if link is not None:
            links.append(link)
        meshes.append(
            {
                "uid": uid,
                "kind": kind,
                "link": link,
                "vertices": vertices.reshape(-1),
                "normals": normals.reshape(-1),
                "faces": faces.reshape(-1),
            }
        )
    if not links or not meshes:
        raise ValueError("Native asset export produced no preview geometry")
    return meshes, links


def _matrices(
    env: Any, meshes: list[dict], links: list[str], q: np.ndarray, entities: dict
) -> np.ndarray:
    import torch

    robot = env.robot
    names = robot.pk_chain.get_joint_parameter_names()
    indices = [robot.joint_names.index(name) for name in names]
    with torch.no_grad():
        fk = (
            robot.compute_fk(
                torch.as_tensor(
                    q[:, indices], dtype=torch.float32, device=robot.device
                ),
                link_names=links,
                qpos_joint_names=names,
            )
            .detach()
            .cpu()
            .numpy()
        )
    lookup = {link: index for index, link in enumerate(links)}
    result = np.empty((len(q), len(meshes), 4, 4), dtype=np.float32)
    for index, mesh in enumerate(meshes):
        if mesh["kind"] == "robot":
            result[:, index] = entities[mesh["uid"]] @ fk[:, lookup[mesh["link"]]]
        else:
            result[:, index] = entities[mesh["uid"]]
    # WebGL consumes column-major matrices. These are FK of measured joints,
    # never the requested controller targets or an interpolated success animation.
    return result.swapaxes(-1, -2).reshape(len(q), len(meshes), 16)


def media_preview(manifest: str, destination: Path, key: str) -> dict:
    from embodichain.data_pipeline.recording.media import MediaFrameReader
    from PIL import Image
    import imageio.v2 as imageio

    value = json.loads(Path(manifest).read_text())
    features = value["features"]
    rgb = next(
        (
            k
            for k in features
            if k.startswith("sensor/") and k.rsplit("/", 1)[-1] in {"rgb", "color"}
        ),
        None,
    )
    depth = next(
        (k for k in features if k.startswith("sensor/") and k.endswith("/depth")), None
    )
    if rgb is None:
        return {}
    fields = [rgb] + ([depth] if depth else [])
    destination.mkdir(parents=True, exist_ok=True)
    result = {"calibration": value.get("calibration"), "depth_range_m": [0.0, 2.0]}
    with MediaFrameReader(manifest, fields=fields) as reader:
        count = value["frame_count"]
        dt = reader.read(1)[0] - reader.read(0)[0] if count > 1 else 1 / 30
        stride = max(1, int(np.ceil(1 / (30 * dt))))
        fps = 1 / (dt * stride)
        rgb_path = destination / (key + ".rgb.mp4")
        depth_path = destination / (key + ".depth.mp4")
        writers = {
            "rgb": imageio.get_writer(
                rgb_path,
                fps=fps,
                codec="libx264",
                quality=7,
                macro_block_size=1,
                ffmpeg_params=["-movflags", "+faststart"],
            )
        }
        if depth:
            writers["depth"] = imageio.get_writer(
                depth_path,
                fps=fps,
                codec="libx264",
                quality=7,
                macro_block_size=1,
                ffmpeg_params=["-movflags", "+faststart"],
            )
        try:
            for index in range(0, count, stride):
                _, frame = reader.read(index)
                image = np.asarray(frame[rgb])[..., :3]
                if image.dtype != np.uint8:
                    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                writers["rgb"].append_data(image)
                if index == 0:
                    Image.fromarray(image).save(destination / (key + ".jpg"))
                if depth:
                    z = np.asarray(frame[depth]).squeeze()
                    t = np.clip(np.nan_to_num(z, nan=0, posinf=0) / 2, 0, 1)
                    color = np.stack(
                        (
                            np.clip(1.5 - t * 2, 0, 1),
                            np.sin(t * np.pi),
                            np.clip(t * 2 - 0.3, 0, 1),
                        ),
                        axis=-1,
                    )
                    color[(z <= 0) | ~np.isfinite(z)] = 0
                    writers["depth"].append_data((color * 255).astype(np.uint8))
        finally:
            for writer in writers.values():
                writer.close()
    result.update(
        rgb=str(rgb_path),
        thumbnail=str(destination / (key + ".jpg")),
        depth=str(depth_path) if depth else None,
        preview_stride=stride,
        preview_fps=fps,
    )
    return result


def source_clock(root: Path, episode, selected: np.ndarray) -> dict:
    """Compose execution-to-prototype-to-acquisition-to-video mappings explicitly."""
    from embodichain.data_pipeline.real2sim import load_prototype
    from embodichain.data_pipeline.recording.media import _digest
    from embodichain.data import get_data_path
    from ._cases import ENGINE
    import shutil

    lineage = episode.metadata.get("real2sim_lineage")
    if not lineage:
        return {}
    evidence = json.loads((root / "inputs/carton/qualification.json").read_text())
    clock = evidence["frame_clock_map"]
    original = ENGINE.parent / "trajectory/real2sim_carton"
    annotation_path = original / "annotations/staples.json"
    if _digest(annotation_path) != evidence["annotation_sha256"]:
        raise ValueError(
            "source-video annotation differs from the pinned qualification"
        )
    annotation = json.loads(annotation_path.read_text())
    calls = {c["call_index"]: c for c in lineage["calls"]}
    prototypes = {
        i: load_prototype(
            episode.metadata["real2sim_source_artifacts"][c["source_bundle"]][
                "prototype_manifest"
            ]["path"]
        )
        for i, c in calls.items()
    }
    output = []
    phase = []
    with np.load(lineage["trace"]["path"], allow_pickle=False) as trace:
        for index in selected:
            step = min(int(index), len(episode.actions) - 1)
            call_index = int(trace["step_call_index"][step])
            plan_index = int(trace["step_plan_index"][step])
            mapped = None
            label = "等待 / 验收"
            if call_index >= 0 and plan_index >= 0:
                call = calls[call_index]
                prototype = prototypes[call_index]
                left, right = (
                    int(trace[f"call_{call_index}_{k}"][plan_index])
                    - call["source_sample_offset"]
                    for k in ("left", "right")
                )
                weight = float(trace[f"call_{call_index}_weight"][plan_index])
                if left >= 0:
                    mapping = prototype.unit.report.time_map

                    def source_seconds(i):
                        return (
                            1 - mapping.weight[i]
                        ) * prototype.unit.source.timestamps[
                            mapping.left[i]
                        ] + mapping.weight[
                            i
                        ] * prototype.unit.source.timestamps[
                            mapping.right[i]
                        ]

                    mapped = float(
                        (1 - weight) * source_seconds(left)
                        + weight * source_seconds(right)
                    )
                    mapped += clock["source_seconds"][
                        annotation["phases"][call_index]["start"]
                    ]
                    label = next(
                        p.name
                        for p in prototype.unit.phases
                        if p.start <= left < p.stop
                    )
                else:
                    label = "入口连接 · 无源帧"
            output.append(
                float(
                    np.interp(
                        mapped, clock["source_seconds"], clock["encoding_seconds"]
                    )
                )
                if mapped is not None
                else None
            )
            phase.append(label)
    raw = Path(get_data_path("Real2SimCarton/raw"))
    videos = []
    for entry in json.loads((original / "source.lock.json").read_text())["files"]:
        if not entry["path"].endswith(".mp4"):
            continue
        source = raw / entry["path"]
        if _digest(source) != entry["sha256"]:
            raise ValueError("Source video differs from the pinned recording")
        target = root / "media/source" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or _digest(target) != entry["sha256"]:
            shutil.copyfile(source, target)
        videos.append(
            {
                "path": str(target.relative_to(root)),
                "sha256": entry["sha256"],
                "label": source.stem,
            }
        )
    return {
        "video_seconds": output,
        "source_phase": phase,
        "source_videos": videos,
        "source_license": "DROID / CC BY 4.0",
        "source_id": evidence["source_id"],
        "mapping": "execution → prototype → source acquisition → video encoding",
    }


def preview_artifacts(root: Path, index_path: Path) -> dict[str, str]:
    """Attest the index and each generated file required by its playback entries."""
    from embodichain.data_pipeline.recording.media import _digest
    from embodichain.lab.expansion.artifacts import resolve_artifact_path

    records = json.loads(index_path.read_text())
    paths = {index_path.relative_to(root).as_posix()}
    for record in records:
        paths.update((record["model_path"], record["replay"]))
        paths.update(
            value
            for key, value in record.get("media", {}).items()
            if key in {"rgb", "depth", "thumbnail"} and value
        )
    paths.update(
        p.relative_to(root).as_posix() for p in (root / "media/source").glob("*.mp4")
    )
    return {name: _digest(resolve_artifact_path(root, name)) for name in sorted(paths)}


def preview_cache_valid(
    root: Path, index_path: Path, signature_path: Path, source: dict
) -> bool:
    """Reuse only when source pins and all generated playback files still match."""
    try:
        saved = json.loads(signature_path.read_text())
        if (
            not isinstance(saved, dict)
            or saved.get("source") != source
            or not saved.get("artifacts")
        ):
            return False
        return saved["artifacts"] == preview_artifacts(root, index_path)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def export_replays(root: Path, job: Path) -> dict:
    from embodichain.lab.expansion import (
        load_expansion_job,
        ArtifactEpisodeSink,
        load_expansion_episode,
    )
    from embodichain.lab.expansion.visual import inspect_observation_variants
    from embodichain.lab.sim import SimulationManager

    cfg = load_expansion_job(job)
    directory = Path(cfg.output_dir)
    receipts = ArtifactEpisodeSink(directory).inspect()
    variants = inspect_observation_variants(directory)
    if not receipts:
        return {
            "ok": True,
            "episodes": [],
            "reason": "No confirmed physical episodes; no success animation generated.",
        }
    case = next(
        c
        for c in json.loads((root / "cases.json").read_text())["cases"]
        if root / c["job"] == job
    )
    from embodichain.data_pipeline.recording.media import _digest

    signature = {
        "parents": {r.commit_id: _digest(Path(r.storage_id)) for r in receipts},
        "children": {v["variant_id"]: v["media_sha256"] for v in variants},
        "replay_version": 2,
    }
    signature_path = root / "replay" / (case["id"] + ".signature.json")
    if preview_cache_valid(
        root, root / "replay" / (case["id"] + ".json"), signature_path, signature
    ):
        return {
            "ok": True,
            "case": case["id"],
            "reused": True,
            "episodes": len(receipts) + len(variants),
        }
    owned = environment(Path(cfg.deployment))
    env = getattr(owned, "unwrapped", owned)
    outputs = []
    try:
        meshes, links = _asset_model(env)
        asset = case["source"] + "-" + case["backend"]
        model_path = root / "replay" / (asset + ".model.js")
        javascript(model_path, "model", asset, {"meshes": meshes})
        q0 = env.robot.get_qpos().cpu().numpy()
        native_root = env.robot.get_local_pose(to_matrix=True)[0].cpu().numpy()
        native_entities = {
            uid: env.sim.get_rigid_object(uid)
            .get_local_pose(to_matrix=True)[0]
            .cpu()
            .numpy()[None]
            for uid in env.sim.get_rigid_object_uid_list()
        }
        native_entities[env.robot.uid] = native_root[None]
        checked = _matrices(env, meshes, links, q0, native_entities)[0]
        for index, mesh in enumerate(meshes):
            if mesh["link"]:
                native = (
                    env.robot.get_link_pose(mesh["link"], to_matrix=True)[0]
                    .cpu()
                    .numpy()
                )
                if not np.allclose(
                    checked[index].reshape(4, 4).T, native, atol=2e-4, rtol=0
                ):
                    raise ValueError("Native/FK frames disagree")
        for receipt in receipts:
            episode = load_expansion_episode(receipt.storage_id)
            key = episode.episode_id
            replay_path = root / "replay" / (key + ".js")
            times = episode.timestamps.cpu().numpy()
            indices = np.unique(
                np.r_[
                    np.arange(
                        0, len(times), max(1, int(np.ceil((len(times) - 1) / 240)))
                    ),
                    len(times) - 1,
                ]
            )
            q = episode.observations["full_joint_positions"].cpu().numpy()[indices]
            entities = {
                f["entity_uid"]: poses(
                    episode.observations[f["key"]].cpu().numpy()[indices]
                )
                for f in episode.metadata["scene_state_fields"]
                if f["field"] == "pose"
            }
            matrices = _matrices(env, meshes, links, q, entities)
            joint_ids = env.robot.get_joint_ids("arm")
            tcp = (
                env.robot.compute_batch_fk(
                    torch.tensor(q[:, joint_ids], device=env.robot.device)[None],
                    name="arm",
                    env_ids=[0],
                    to_matrix=True,
                )[0]
                .cpu()
                .numpy()
            )
            tcp = entities[env.robot.uid] @ np.linalg.inv(native_root) @ tcp
            metadata = plain(episode.metadata)
            data = {
                "times": times[indices],
                "matrices": matrices,
                "tcp": tcp[:, :3, 3],
                "qpos": q,
                "joint_names": env.robot.joint_names,
                "joint_types": [
                    env.robot.get_joint_type(name) for name in env.robot.joint_names
                ],
                "segments": metadata.get("segments", []),
                "timestamps_origin": "measured",
                "tcp_origin": "FK of measured qpos",
                "model": asset,
            }
            data.update(source_clock(root, episode, indices))
            javascript(replay_path, "episode", key, data)
            media = (
                media_preview(metadata["media_manifest"], root / "media", key)
                if metadata.get("media_manifest")
                else {}
            )
            for name in ("rgb", "depth", "thumbnail"):
                if media.get(name):
                    media[name] = str(Path(media[name]).relative_to(root))
            record = {
                "id": key,
                "case": case["id"],
                "model": asset,
                "model_path": str(model_path.relative_to(root)),
                "replay": str(replay_path.relative_to(root)),
                "media": media,
                "duration": float(times[-1]),
                "actions": len(episode.actions),
                "states": len(times),
                "recipe_id": metadata.get("recipe_id"),
                "attempt": metadata.get("attempt_number", 0),
                "commit_id": receipt.commit_id,
                "metadata": metadata,
                "validation": plain([vars(c) for c in episode.validation.checks]),
                "artifact": str(Path(receipt.storage_id).relative_to(root)),
                "kind": "physical",
                "joint_types": data["joint_types"],
            }
            outputs.append(record)
            for child in (
                v for v in variants if v["parent_episode_id"] == episode.episode_id
            ):
                childkey = "observation_" + child["variant_id"]
                media = media_preview(child["media_manifest"], root / "media", childkey)
                for name in ("rgb", "depth", "thumbnail"):
                    if media.get(name):
                        media[name] = str(Path(media[name]).relative_to(root))
                outputs.append(
                    {
                        **record,
                        "id": childkey,
                        "kind": "visual",
                        "parent": key,
                        "replay_id": key,
                        "profile": child["profile_id"],
                        "media": media,
                        "child_evidence": plain(child),
                    }
                )
        dump(root / "replay" / (case["id"] + ".json"), outputs)
        dump(
            signature_path,
            {
                "source": signature,
                "artifacts": preview_artifacts(
                    root, root / "replay" / (case["id"] + ".json")
                ),
            },
        )
        return {
            "ok": True,
            "case": case["id"],
            "episodes": len(outputs),
            "native_fk_agreement": True,
            "physics_steps": 0,
        }
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
