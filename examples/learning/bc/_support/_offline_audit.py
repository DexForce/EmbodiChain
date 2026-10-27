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

"""Read-only, preregistered teacher-forced BC error and coverage audit."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import h5py
import torch

from embodichain_models.policy.bc import ChunkBCModel, ChunkBCModelCfg
from embodichain.utils.math import axis_angle_from_quat, quat_from_matrix

from ._pick_place import (
    CAMERAS,
    TARGET_SEMANTICS,
    decode_request,
    pose_targets,
    rotation_from_6d,
)

__all__: list[str] = []

_PHASES = ("approach", "grasp", "transport", "release", "withdraw")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _phases(geometry: torch.Tensor) -> torch.Tensor:
    """Label measured geometry for reporting, without expert state or success."""
    cube, tcp, goal = geometry[:, 27:30], geometry[:, 18:21], geometry[:, 36:39]
    lifted = torch.cummax(cube[:, 2], dim=0).values >= cube[0, 2] + 0.05
    near_goal = (cube - goal).norm(dim=-1) <= 0.03
    distance = (cube - tcp).norm(dim=-1)
    result = torch.zeros(len(geometry), dtype=torch.int64)
    result[distance <= 0.08] = 1
    result[lifted] = 2
    result[near_goal] = 3
    result[near_goal & (distance >= 0.10)] = 4
    return result


def _describe(value: torch.Tensor) -> dict[str, float | int | None]:
    value = value.float().flatten()
    if not value.numel():
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": value.numel(),
        "mean": float(value.mean()),
        "p50": float(value.median()),
        "p95": float(torch.quantile(value, 0.95)),
        "max": float(value.max()),
    }


def _errors(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    geometry: torch.Tensor,
    actions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compare chunk positions against their aligned recorded teacher states."""
    request = decode_request(predicted, geometry)
    label_request = decode_request(targets, geometry)
    translation = predicted[..., :3] - geometry[..., 18:21]
    expected_translation = actions[..., :3] * 0.02
    magnitude = expected_translation.norm(dim=-1)
    predicted_magnitude = translation.norm(dim=-1)
    relative_rotation = rotation_from_6d(predicted[..., 3:9]) @ rotation_from_6d(
        targets[..., 3:9]
    ).transpose(-1, -2)
    skew = torch.stack(
        (
            relative_rotation[..., 2, 1] - relative_rotation[..., 1, 2],
            relative_rotation[..., 0, 2] - relative_rotation[..., 2, 0],
            relative_rotation[..., 1, 0] - relative_rotation[..., 0, 1],
        ),
        dim=-1,
    )
    cosine = (relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    position = (predicted[..., :3] - targets[..., :3]).norm(dim=-1)
    rotation_request = (
        axis_angle_from_quat(
            quat_from_matrix(
                rotation_from_6d(predicted[..., 3:9])
                @ rotation_from_6d(geometry[..., 21:27]).transpose(-1, -2)
            )
        )
        / 0.1
    )
    return {
        "position_error_m": position,
        "rotation_error_rad": torch.atan2(skew.norm(dim=-1) / 2, cosine),
        "gripper_abs_error": (predicted[..., 9] - targets[..., 9]).abs(),
        "gripper_wrong_sign": (predicted[..., 9] > 0) != (actions[..., 6] > 0),
        "translation_request_l2_error": (request[..., :3] - actions[..., :3]).norm(
            dim=-1
        ),
        "rotation_request_l2_error": (request[..., 3:6] - actions[..., 3:6]).norm(
            dim=-1
        ),
        "request_signal_m": magnitude,
        "error_over_request_signal": position / magnitude.clamp_min(1e-6),
        "direction_cosine": (translation * expected_translation).sum(-1)
        / (magnitude * predicted_magnitude).clamp_min(1e-12),
        "direction_valid": (magnitude > 1e-6) & (predicted_magnitude > 1e-6),
        "translation_clip_fraction": (translation.abs() > 0.02).float().mean(-1),
        "rotation_clip_fraction": (rotation_request.abs() > 1).float().mean(-1),
        "gripper_clip": predicted[..., 9].abs() > 1,
        "label_decode_max_abs_error": (label_request - actions).abs().amax(-1),
    }


def _summarize(
    metrics: dict[str, torch.Tensor], selector: torch.Tensor
) -> dict[str, Any]:
    output = {}
    for name, value in metrics.items():
        selection = selector
        if name == "direction_cosine":
            selection = selection & metrics["direction_valid"]
        elif name == "error_over_request_signal":
            selection = selection & (metrics["request_signal_m"] > 1e-6)
        output[name] = _describe(value[selection])
    return output


def _switches(predicted: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    expected, actual = targets[..., 9] > 0, predicted[..., 9] > 0
    records = []
    for row in range(len(expected)):
        reference = (expected[row, 1:] != expected[row, :-1]).nonzero().flatten() + 1
        candidate = (actual[row, 1:] != actual[row, :-1]).nonzero().flatten() + 1
        matches = []
        for offset in reference.tolist():
            same_direction = [
                index
                for index in candidate.tolist()
                if bool(actual[row, index]) == bool(expected[row, offset])
            ]
            nearest = (
                min(same_direction, key=lambda index: (abs(index - offset), index))
                if same_direction
                else None
            )
            matches.append(
                {
                    "target_offset": offset,
                    "opens": bool(expected[row, offset]),
                    "nearest_predicted_offset": nearest,
                    "offset_error_steps": None if nearest is None else nearest - offset,
                }
            )
        records.append(
            {
                "anchor_row": row,
                "target_switches": matches,
                "predicted_switch_offsets": candidate.tolist(),
            }
        )
    return {
        "scope": "Teacher-forced chunk sign transitions; nearest same-direction association is not closed-loop latency and can reuse a predicted transition.",
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", action="append", required=True, metavar="NAME=PATH"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--anchors", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.episodes, args.anchors, args.timeout_seconds) < 1:
        parser.error("Batch, episode, anchor and timeout limits must be positive")
    torch.set_num_threads(2)
    started = time.monotonic()
    checkpoints = {}
    for item in args.checkpoint:
        name, separator, value = item.partition("=")
        if not separator or not name or name in checkpoints or Path(name).name != name:
            parser.error(
                "Checkpoint names must be distinct simple names followed by =PATH"
            )
        checkpoints[name] = Path(value).resolve()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("complete") is not True or tuple(manifest["cameras"]) != CAMERAS:
        raise ValueError("Audit requires a complete dual-camera PickPlace corpus")
    if args.episodes > len(manifest["episodes"]):
        raise ValueError("Episode panel exceeds corpus")
    episodes = manifest["episodes"][: args.episodes]
    all_anchors = [
        (episode, anchor)
        for episode, record in enumerate(episodes)
        for anchor in range(1, record["length"] - 31)
    ]
    if args.anchors > len(all_anchors):
        raise ValueError("Requested panel has more anchors than eligible windows")
    anchors = [
        all_anchors[(2 * index + 1) * len(all_anchors) // (2 * args.anchors)]
        for index in range(args.anchors)
    ]
    source_root = Path(__file__).resolve().parents[4]
    protocol = {
        "format": "bc-offline-audit-v1",
        "training_updates": 0,
        "scope": "First training episodes only; teacher-forced complete H2/F32 windows. Future geometry/actions are supervision and reporting only. No closed-loop or formal acceptance claim.",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "audit_source_sha256": _sha(Path(__file__)),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha(manifest_path),
        "episodes": [
            {"ordinal": index, "path": record["path"], "sha256": record["sha256"]}
            for index, record in enumerate(episodes)
        ],
        "checkpoints": {
            name: {"path": str(path), "sha256": _sha(path)}
            for name, path in checkpoints.items()
        },
        "anchor_rule": "Midpoint stratification over episode-ordered complete anchors, floor((2*i+1)*N/(2*K)); no RNG.",
        "anchors": anchors,
        "phase_names": _PHASES,
        "phase_rule": "Measured geometry only: approach; TCP-cube<=0.08m grasp; prior cube z>=initial+0.05m transport; goal error<=0.03m release; near-goal and TCP-cube>=0.10m withdraw. Later rules take precedence; none establishes task success.",
        "device": args.device,
        "batch_size": args.batch_size,
        "timeout_seconds": args.timeout_seconds,
        "metric_rules": {
            "rotation": "atan2(norm(skew(Rpred*Rtarget.T))/2,(trace-1)/2)",
            "request_signal": "norm(recorded translation request)*0.02m; ratios exclude <=1e-6m in summaries",
            "gripper": "positive=open, zero/nonpositive=close; continuous errors also reported",
            "decode": "Each future prediction is decoded against its aligned recorded future TCP, not rolled-out student state",
            "padding": "All selected windows are complete; every target is valid",
            "inference": "One ordered pass per checkpoint; no checkpoint choice or training",
        },
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError(
                "Existing preregistration differs; use a fresh output directory"
            )
    else:
        _write(protocol_path, protocol)
    _write(output / "protocol-sha256.json", {"protocol_sha256": _sha(protocol_path)})
    for record in episodes:
        if _sha(manifest_path.parent / record["path"]) != record["sha256"]:
            raise ValueError("An input episode differs from its frozen manifest")
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "prepared": str(protocol_path),
                    "anchors": len(anchors),
                    "training_updates": 0,
                }
            )
        )
        return
    if (output / "summary.json").exists() or any(output.glob("*-records.pt")):
        raise ValueError(
            "Audit outputs already exist; refusing to overwrite an inference pass"
        )
    with ExitStack() as stack:
        handles = [
            stack.enter_context(h5py.File(manifest_path.parent / record["path"], "r"))
            for record in episodes
        ]
        numeric = [
            {
                key: torch.from_numpy(handle[key][:]).float()
                for key in ("geometry", "actions")
            }
            for handle in handles
        ]
        phases = [_phases(value["geometry"]) for value in numeric]
        coverage = {}
        for phase_index, phase_name in enumerate(_PHASES):
            coverage[phase_name] = {
                "frames": sum(int((value == phase_index).sum()) for value in phases),
                "complete_anchors": sum(
                    int((value[1 : record["length"] - 31] == phase_index).sum())
                    for value, record in zip(phases, episodes)
                ),
                "partial_anchors": sum(
                    int((value[1:] == phase_index).sum()) for value in phases
                ),
                "selected_anchors": sum(
                    int(phases[episode][anchor]) == phase_index
                    for episode, anchor in anchors
                ),
            }
        geometry = torch.stack(
            [
                numeric[episode]["geometry"][anchor : anchor + 32]
                for episode, anchor in anchors
            ]
        )
        actions = torch.stack(
            [
                numeric[episode]["actions"][anchor : anchor + 32]
                for episode, anchor in anchors
            ]
        )
        target_phases = torch.stack(
            [phases[episode][anchor : anchor + 32] for episode, anchor in anchors]
        )
        targets = pose_targets(geometry, actions)
        torch.save(
            {
                "anchors": anchors,
                "geometry": geometry,
                "actions": actions,
                "targets": targets,
                "target_phases": target_phases,
            },
            output / "labels.pt",
        )
        summaries = {}
        for name, checkpoint in checkpoints.items():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            weights = state["learner"]["model"]
            model = ChunkBCModel(ChunkBCModelCfg(**weights["_extra_state"]["config"]))
            if (
                model.cfg.history_length != 2
                or model.cfg.prediction_horizon != 32
                or model.cfg.observation_dim != 27
                or model.cfg.target_dim != 10
                or model.cfg.cameras != CAMERAS
                or model.cfg.target_semantics != TARGET_SEMANTICS
            ):
                raise ValueError(
                    "Checkpoint is not the frozen visual H2/F32 PickPlace model"
                )
            model.load_state_dict(weights)
            model.to(args.device).eval()
            predictions, input_clips = [], []
            checkpoint_started = time.monotonic()
            with torch.inference_mode():
                for start in range(0, len(anchors), args.batch_size):
                    if time.monotonic() - started >= args.timeout_seconds:
                        raise TimeoutError("Frozen offline audit budget exhausted")
                    selected = anchors[start : start + args.batch_size]
                    history = torch.stack(
                        [
                            numeric[episode]["geometry"][anchor - 1 : anchor + 1, :27]
                            for episode, anchor in selected
                        ]
                    ).to(args.device)
                    images = {
                        camera: torch.stack(
                            [
                                torch.from_numpy(
                                    handles[episode]["images"][camera][
                                        anchor - 1 : anchor + 1
                                    ]
                                ).permute(0, 3, 1, 2)
                                for episode, anchor in selected
                            ]
                        )
                        .to(device=args.device, dtype=torch.float32)
                        .div_(255)
                        for camera in CAMERAS
                    }
                    predictions.append(model.predict_targets(history, images).cpu())
                    input_clips.append(
                        (
                            (
                                (history - model.observation_mean)
                                / model.observation_std
                            ).abs()
                            > model.cfg.normalization_clip
                        )
                        .float()
                        .mean((1, 2))
                        .cpu()
                    )
            predicted = torch.cat(predictions)
            metrics = _errors(predicted, targets, geometry, actions)
            switches = _switches(predicted, targets)
            _write(output / f"{name}-switches.json", switches)
            all_valid = torch.ones(targets.shape[:2], dtype=torch.bool)
            summaries[name] = {
                "checkpoint_sha256": protocol["checkpoints"][name]["sha256"],
                "num_updates": state["learner"]["num_updates"],
                "anchors": len(anchors),
                "valid_targets": int(all_valid.sum()),
                "all": _summarize(metrics, all_valid),
                "by_prediction_offset": {
                    str(offset): _summarize(
                        metrics,
                        torch.arange(32)[None, :].expand(len(anchors), -1) == offset,
                    )
                    for offset in range(32)
                },
                "by_target_phase": {
                    phase_name: _summarize(metrics, target_phases == phase_index)
                    for phase_index, phase_name in enumerate(_PHASES)
                },
                "by_anchor_phase": {
                    phase_name: _summarize(
                        metrics, (target_phases[:, :1] == phase_index).expand(-1, 32)
                    )
                    for phase_index, phase_name in enumerate(_PHASES)
                },
                "numeric_input_clip_fraction": _describe(torch.cat(input_clips)),
                "wall_seconds": time.monotonic() - checkpoint_started,
            }
            torch.save(
                {
                    "predicted_targets": predicted,
                    "decoded_requests": decode_request(predicted, geometry),
                    "metrics": metrics,
                    "numeric_input_clip_fraction": torch.cat(input_clips),
                },
                output / f"{name}-records.pt",
            )
            _write(output / f"{name}-summary.json", summaries[name])
            print(
                json.dumps(
                    {
                        "checkpoint": name,
                        "complete": True,
                        "anchors": len(anchors),
                        "wall_seconds": summaries[name]["wall_seconds"],
                    }
                ),
                flush=True,
            )
            del model, state, weights
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
        summary = {
            "protocol_sha256": _sha(protocol_path),
            "coverage": coverage,
            "results": summaries,
            "training_updates": 0,
            "formal_test_opened": False,
            "wall_seconds": time.monotonic() - started,
        }
        if _sha(manifest_path) != protocol["manifest_sha256"] or any(
            _sha(checkpoints[name]) != identity["sha256"]
            for name, identity in protocol["checkpoints"].items()
        ):
            raise ValueError("A bound manifest/checkpoint changed during the audit")
        for record in episodes:
            if _sha(manifest_path.parent / record["path"]) != record["sha256"]:
                raise ValueError("An episode changed during the audit")
        summary["all_bound_inputs_unchanged"] = True
        _write(output / "summary.json", summary)
        _write(
            output / "artifact-sha256.json",
            {
                path.name: _sha(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact-sha256.json"
            },
        )


if __name__ == "__main__":
    main()
