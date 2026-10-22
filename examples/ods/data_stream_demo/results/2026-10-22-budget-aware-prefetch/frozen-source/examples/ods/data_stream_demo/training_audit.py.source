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

"""Post-timing CPU qualification of saved visual training and standard videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .lerobot_io import (
    TrainingLeRobotReader,
    audit_video_quality,
    read_lerobot_manifest,
)
from .metrics import write_json
from .payload import numeric_episode_digest

__all__ = ["audit_run"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(values: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(memoryview(value.numpy()))
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def _clean_shutdown(result: dict[str, Any]) -> None:
    _require(
        result.get("workers_joined") is True
        and result.get("forced_termination") is False
        and bool(result.get("worker_exitcodes"))
        and all(code == 0 for code in result["worker_exitcodes"]),
        f"Decoded SDK workers did not exit normally: {result}",
    )


def _audit(output: Path, result: dict[str, Any]) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    summary = json.loads((output / "summary.json").read_text())
    config = json.loads((output / "config.json").read_text())
    trace = json.loads((output / "training-trace.json").read_text())
    contract = json.loads((output / "initial-contract.json").read_text())
    schema = json.loads((output / "payload_schema.json").read_text())
    _require(
        summary.get("status") == "complete"
        and summary.get("qualification_passed") is True,
        "Training worker did not complete qualification",
    )
    _require(
        trace.get("format") == "ods-visual-training-trace-v1",
        "Unknown training trace format",
    )
    _require(
        all(config.get(key) == value for key, value in trace["recipe"].items()),
        "Training trace recipe differs from the saved configuration",
    )
    metrics, registry = summary["training_metrics"], trace["episodes"]
    root = output / "lerobot_dataset"
    manifest = read_lerobot_manifest(root)
    episodes, payload = manifest["episodes"], manifest["payload"]
    writer = metrics["writer"]
    _require(
        len(episodes)
        == len(registry)
        == writer["written_episodes"]
        == writer["submitted_episodes"]
        == metrics["generated_episodes"],
        "Accepted, recorded and registered episode counts differ",
    )
    _require(
        writer.get("finalized") is True and writer.get("worker_joined") is True,
        "Recording did not finalize and join",
    )
    _require(
        manifest.get("accepted_episode_count", len(episodes)) == len(episodes),
        "Finalized manifest count differs",
    )
    _require(
        payload == schema and list(payload["cameras"]) == contract["camera_order"],
        "Saved payload/camera order differs from the training contract",
    )
    cameras = list(payload["cameras"])
    _require(
        len(cameras) == (1 if config["vision"] == "single" else 2),
        "Saved camera count differs from the configured visual condition",
    )
    _require(
        config["history_length"] == 2, "Visual audit requires two causal image frames"
    )
    _require(
        math.isclose(payload["step_dt"], 1 / manifest["fps"], rel_tol=0, abs_tol=1e-12),
        "Saved step_dt and observation FPS differ",
    )
    _require(
        payload["action_schema"]["semantics"] == "policy_request"
        and math.isclose(
            payload["action_schema"]["step_dt"],
            payload["step_dt"],
            rel_tol=0,
            abs_tol=1e-12,
        ),
        "Saved action semantics/control period differ",
    )
    dataset = LeRobotDataset(
        "local/ods-data-stream", root=root, video_backend="torchcodec", tolerance_s=1e-4
    )
    info = dataset.meta.info
    _require(
        info["total_episodes"] == len(episodes)
        and info["total_frames"] == sum(row["length"] for row in episodes)
        and info["total_frames"] == writer["valid_frames"]
        and info["fps"] == manifest["fps"],
        "Official SDK counts/FPS differ from the finalized sidecar",
    )
    features = info["features"]
    _require(
        list(features["observation.state"]["shape"]) == [payload["state_width"]]
        and list(features["action"]["shape"]) == [payload["action_width"]]
        and features["observation.state"]["dtype"] == "float32"
        and features["action"]["dtype"] == "float32",
        "Official SDK numeric feature widths differ",
    )
    names = [
        f"{'/'.join(feature['path'])}/{index}"
        for feature in payload["state_features"]
        for index in range(feature["width"])
    ]
    _require(
        features["observation.state"]["names"] == names
        and features["action"]["names"]
        == [f"action/{i}" for i in range(payload["action_width"])],
        "Official SDK numeric feature order differs",
    )
    saved_cameras = [
        key.removeprefix("observation.images.")
        for key, value in features.items()
        if value["dtype"] == "video"
    ]
    _require(saved_cameras == cameras, "Official SDK camera order differs")
    for camera in cameras:
        feature = features[f"observation.images.{camera}"]
        _require(
            list(feature["shape"]) == [*payload["cameras"][camera]["shape"][:2], 3]
            and feature["info"]["video.codec"] == "av1"
            and feature["info"]["video.fps"] == manifest["fps"],
            "Recorded camera shape/codec/FPS differs",
        )
    numeric = dataset.hf_dataset.select_columns(
        ["observation.state", "action", "ods.provenance", "episode_index"]
    ).with_format("numpy")
    offset = 0
    source_by_ordinal = {}
    for ordinal, (saved, registered) in enumerate(zip(episodes, registry)):
        _require(
            saved["index"] == registered["ordinal"] == ordinal
            and saved["offset"] == offset,
            "Saved episode order/offset differs from registration",
        )
        for key in ("length", "source_id", "generation", "numeric_sha256"):
            _require(
                saved[key] == registered[key],
                f"Saved {key} differs for episode {ordinal}",
            )
        values = numeric[offset : offset + saved["length"]]
        state = torch.as_tensor(values["observation.state"], dtype=torch.float32)
        actions = torch.as_tensor(values["action"], dtype=torch.float32)
        provenance = torch.as_tensor(values["ods.provenance"], dtype=torch.int64)
        digest = numeric_episode_digest(
            {"state": state, "actions": actions, "continuity_id": provenance[:, 4]}
        )
        _require(
            digest == saved["numeric_sha256"],
            f"Actual Parquet numeric payload differs for episode {ordinal}",
        )
        _require(
            torch.as_tensor(values["episode_index"]).eq(ordinal).all().item()
            and provenance[:, :2]
            .eq(torch.tensor(saved["source_id"], dtype=torch.int64))
            .all()
            .item()
            and provenance[:, 2].eq(torch.arange(saved["length"])).all().item(),
            "Actual saved episode/source/frame order differs",
        )
        _require(
            provenance[:, 3].eq(saved["generation"]).all().item()
            and provenance[:, 5].eq(registered["published_ns"]).all().item(),
            "Actual saved generation/publication differs",
        )
        source_by_ordinal[ordinal] = ":".join(map(str, saved["source_id"]))
        offset += saved["length"]
    result["recording"] = {
        "episodes": len(episodes),
        "frames": offset,
        "cameras": cameras,
        "all_numeric_sha256_verified": True,
        "sdk_metadata_verified": True,
    }
    updates, batch_size, future = (
        config["updates"],
        config["batch_size"],
        config["prediction_horizon"],
    )
    _require(
        len(trace["updates"]) == updates == metrics["completed_updates"],
        "Optimizer update budget is incomplete",
    )
    _require(
        metrics["trained_chunks"] == updates * batch_size
        and metrics["valid_target_occurrences"] == updates * batch_size * future,
        "Actual training chunks/targets differ from the fixed budget",
    )
    expected_counts, expected_anchors = Counter(), {}
    for number, update in enumerate(trace["updates"], 1):
        _require(
            update["update"] == number and len(update["pairs"]) == batch_size,
            "Training trace update/batch order differs",
        )
        for ordinal, anchor in update["pairs"]:
            _require(
                type(ordinal) is int
                and ordinal in source_by_ordinal
                and type(anchor) is int
                and 1 <= anchor <= episodes[ordinal]["length"] - future,
                "Training trace has an illegal source/anchor",
            )
            _require(
                update["generation"] in (0, episodes[ordinal]["generation"]),
                "Training trace generation differs",
            )
            source = source_by_ordinal[ordinal]
            expected_counts[source] += 1
            expected_anchors.setdefault(source, Counter())[anchor] += 1
    rows = {row["source_id"]: row for row in metrics["episodes"]}
    _require(
        set(rows) == set(source_by_ordinal.values())
        and len(rows) == len(metrics["episodes"]),
        "Training metrics omitted or duplicated accepted sources",
    )
    for ordinal, saved in enumerate(episodes):
        source, length = source_by_ordinal[ordinal], saved["length"]
        row = rows[source]
        _require(
            row["ordinal"] == ordinal
            and row["frame_count"] == length
            and row["trained_chunks"] == expected_counts[source],
            "Per-source trained counts differ from the trace",
        )
        anchors = expected_anchors.get(source, Counter())
        history, targets = Counter(), Counter()
        for anchor, count in anchors.items():
            history.update({frame: count for frame in range(anchor - 1, anchor + 1)})
            targets.update({frame: count for frame in range(anchor, anchor + future)})
        for key, counts in (
            ("anchor_counts", anchors),
            ("history_frame_counts", history),
            ("target_frame_counts", targets),
        ):
            _require(
                row[key] == [counts[frame] for frame in range(length)],
                f"Actual {key} differs from completed training trace",
            )
    ledger = metrics.get("ledger", {}).get("episodes", [])
    seen = set()
    for row in ledger:
        source = ":".join(map(str, row["source_id"]))
        _require(
            source not in seen
            and source in rows
            and row["reserved"] == 0
            and row["delivered"] == row["trained"] == expected_counts[source],
            "Retired delivery/training ledger differs from the trace",
        )
        seen.add(source)
    if config["arm"] != "offline":
        _require(
            all(source in seen for source, count in expected_counts.items() if count),
            "Protected source ledger omitted trained sources",
        )
        _require(
            all(count <= config["episode_budget"] for count in expected_counts.values())
            and all(row["delivered"] <= config["episode_budget"] for row in ledger),
            "Protected delivery/training cap exceeded",
        )
    checkpoint = torch.load(output / "final.pt", map_location="cpu", weights_only=False)
    _require(
        checkpoint["step"] == updates and list(checkpoint["cameras"]) == cameras,
        "Final checkpoint step/cameras differ",
    )
    _require(
        _digest(checkpoint["model"])
        == metrics["final_model_sha256"]
        != metrics["initial_model_sha256"],
        "Final model digest differs or parameters never updated",
    )
    _require(
        metrics["initial_model_sha256"]
        == trace["initial_model_sha256"]
        == contract["model_sha256"],
        "Initial model contract differs from the training trace",
    )
    _require(
        _digest(checkpoint["normalization"])
        == metrics["normalization_sha256"]
        == trace["normalization_sha256"]
        == contract["normalization_sha256"],
        "Frozen normalization differs",
    )
    optimizer_steps = [
        int(state["step"]) for state in checkpoint["optimizer"]["state"].values()
    ]
    _require(
        bool(optimizer_steps) and all(step == updates for step in optimizer_steps),
        "Final optimizer state does not contain the completed update budget",
    )
    del checkpoint
    result["training"] = {
        "updates": updates,
        "chunks": updates * batch_size,
        "valid_targets": updates * batch_size * future,
        "trained_sources": len(expected_counts),
        "untrained_saved_sources": len(episodes) - len(expected_counts),
        "ledger_sources": len(ledger),
        "checkpoint_verified": True,
    }
    plan = [[(ordinal, 1)] for ordinal in range(len(episodes))]
    reader = TrainingLeRobotReader(
        root,
        plan,
        history_length=2,
        prediction_horizon=future,
        num_workers=4,
        prefetch_factor=1,
    )
    decoded = 0
    try:
        for ordinal in range(len(episodes)):
            batch = reader.next_batch()
            _require(
                batch["valid"].all().item()
                and batch["frame_ids"][0, 1, :2].tolist()
                == episodes[ordinal]["source_id"],
                "Representative decode returned a different source",
            )
            for image in batch["images"].values():
                _require(
                    tuple(image.shape) == (1, 2, 3, 256, 256)
                    and image.dtype == torch.float32
                    and torch.isfinite(image).all().item()
                    and image.min().item() >= 0
                    and image.max().item() <= 1,
                    "Representative decoded RGB history differs from the training contract",
                )
            decoded += 1
        try:
            reader.next_batch()
        except StopIteration:
            pass
        else:
            raise ValueError("Finite audit decode plan did not exhaust")
        result["decode"] = {"episodes": decoded, "stats": reader.stats}
    finally:
        result["decode_shutdown"] = reader.close()
    _clean_shutdown(result["decode_shutdown"])
    quality = audit_video_quality(root)
    result["video_quality"] = quality
    _clean_shutdown(quality["worker_cleanup"])
    _require(
        len(quality["samples"]) == 3 * len(cameras)
        and all(
            math.isfinite(row["mae_uint8"])
            and (row["psnr_db"] is None or math.isfinite(row["psnr_db"]))
            for row in quality["samples"]
        ),
        "Raw-reference video quality audit is incomplete or nonfinite",
    )


def audit_run(output: Path, *, receipt: Path | None = None) -> dict[str, Any]:
    """Audit a run after timing; preserve a failed JSON result without using GPUs.

    Args:
        output: Completed run directory, containing its own standard corpus,
            training trace, summary and checkpoint.
        receipt: Fresh result path. Defaults to ``output/post-audit.json``;
            explicitly choose another path to preserve an earlier audit.

    Returns:
        Qualification, source-window SDK checks and full video-frame integrity.

    Raises:
        FileExistsError: The receipt or its full-video artifacts already exist.
    """
    from . import video_integrity

    output = Path(output).resolve()
    receipt = output / "post-audit.json" if receipt is None else Path(receipt).resolve()
    video_receipt = receipt.with_name(receipt.stem + "-video-integrity.json")
    video_artifacts = video_receipt.parent / (video_receipt.stem + "-artifacts")
    if any(path.exists() for path in (receipt, video_receipt, video_artifacts)):
        raise FileExistsError("Use a fresh audit receipt and artifact path")
    corpus = (output / "lerobot_dataset").resolve()
    if receipt == corpus or corpus in receipt.parents:
        raise ValueError("Audit receipts must stay outside the immutable corpus")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump({"passed": False, "status": "pending"}, stream)
        stream.write("\n")
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema": "ods-visual-training-post-audit-v2",
        "passed": False,
        "qualification_passed": False,
        "scope": "post-timing numeric/checkpoint, source-window SDK and all-frame video integrity audit; no duration contributes to pipeline time",
        "audit_source_sha256": {
            "training_audit": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "video_integrity": hashlib.sha256(
                Path(video_integrity.__file__).read_bytes()
            ).hexdigest(),
        },
    }
    try:
        _audit(output, result)
        result["full_video_integrity"] = video_integrity.verify_dataset(
            output / "lerobot_dataset", video_receipt
        )
        _require(
            result["full_video_integrity"].get("passed") is True,
            "Full recorded video-frame integrity failed",
        )
        result.update(passed=True, qualification_passed=True)
    except Exception as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
    result["seconds_outside_pipeline"] = time.monotonic() - started
    write_json(receipt, result)
    return result


def main() -> None:
    """Run CPU qualification and exit unsuccessfully for incomplete evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    torch.set_num_threads(1)
    result = audit_run(args.run_dir, receipt=args.output)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "error": result.get("error"),
                "seconds_outside_pipeline": result["seconds_outside_pipeline"],
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
