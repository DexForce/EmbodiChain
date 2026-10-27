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

"""Bounded first-eight-episode learnability check using the shared SL trainer."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Iterator

import h5py
import torch

from embodichain_models.policy.bc.sl_recipe import build_supervised_trainer

from ._offline_audit import _PHASES, _errors, _phases, _sha, _summarize, _write
from ._pick_place import CAMERAS, TARGET_SEMANTICS, encode_targets, pose_targets, source
from ._train import _require_single_process

__all__: list[str] = []


def _once(path: Path, value: Any) -> None:
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(value)):
            raise ValueError(f"Existing frozen input differs: {path}")
    else:
        _write(path, value)


@contextmanager
def _training_deadline(seconds: int) -> Iterator[None]:
    def exhausted(_signum: int, _frame: Any) -> None:
        raise TimeoutError("The frozen learnability training deadline expired")

    previous = signal.signal(signal.SIGALRM, exhausted)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _build(config: dict[str, Any], output: Path, *, device: str) -> Any:
    initialization, data = config["initialization"], config["data"]
    checkpoint = Path(initialization["checkpoint"])
    if _sha(checkpoint) != initialization["checkpoint_sha256"]:
        raise ValueError("Initialization checkpoint differs from the protocol")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["learner"]["num_updates"] != 20000:
        raise ValueError("Learnability starts from the frozen 20k weights")
    weights = state["learner"]["model"]
    original = source(data["manifest"])
    try:
        statistics = torch.load(
            data["normalization"], map_location="cpu", weights_only=True
        )
        if statistics["source_id"] != original.source_id:
            raise ValueError("Statistics do not belong to the original corpus view")
        statistics = statistics["statistics"]
    finally:
        original.close()
    if any(not torch.equal(value, weights[key]) for key, value in statistics.items()):
        raise ValueError("Frozen normalization differs from initialization")
    statistics_path = output / "normalization-statistics.pt"
    if statistics_path.exists():
        saved = torch.load(statistics_path, map_location="cpu", weights_only=True)
        if saved.keys() != statistics.keys() or any(
            not torch.equal(saved[key], value) for key, value in statistics.items()
        ):
            raise ValueError("Refusing to replace frozen normalization")
    else:
        with statistics_path.open("xb") as stream:
            torch.save(statistics, stream)
    numerics = state["signature"]["numerics"]
    torch.backends.cudnn.allow_tf32 = numerics["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = numerics["matmul_allow_tf32"]
    trainer = build_supervised_trainer(
        {
            "model": weights["_extra_state"]["config"],
            "data": {
                "manifest": str(output / "eight-episode-manifest.json"),
                "normalization": str(statistics_path),
                "prefetch_batches": 1,
                "reader": {
                    "observation_key": "geometry",
                    "observation_slice": [0, 27],
                    "target_encoder": encode_targets,
                    "target_fields": ("geometry", "actions"),
                    "target_encoder_id": TARGET_SEMANTICS
                    + ":relative-scales-0.02m-0.1rad@1",
                    "include_partial_targets": True,
                    "max_open_files": 8,
                },
            },
            "optimizer": config["optimizer"],
            "objective": {"type": "masked_normalized_target_mse"},
            "trainer": {
                **config["trainer"],
                "device": device,
                "output_dir": str(output / "training"),
            },
        }
    )
    trainer.learner.model.load_state_dict(weights)
    trainer.signature.update(
        research_recipe="bc-first-eight-learnability-v1",
        initialization_sha256=initialization["checkpoint_sha256"],
        original_manifest_sha256=data["manifest_sha256"],
        initialization="weights only; fresh AdamW, update zero, no inherited scheduler",
        numerics=numerics,
        camera_profile_version="pick-place-v2",
        environment=state["signature"]["environment"],
        action_horizon=32,
    )
    return trainer


def _evaluate(
    model: Any, protocol: dict[str, Any], output: Path, name: str
) -> dict[str, Any]:
    model.eval()
    anchors = protocol["anchors"]
    prediction, target, geometry, action, phase, offsets = [], [], [], [], [], []
    with ExitStack() as stack:
        handles = [
            stack.enter_context(h5py.File(record["path"], "r"))
            for record in protocol["episodes"]
        ]
        numeric = [
            {
                key: torch.from_numpy(handle[key][:]).float()
                for key in ("geometry", "actions")
            }
            for handle in handles
        ]
        phases = [_phases(value["geometry"]) for value in numeric]
        with torch.inference_mode():
            for start in range(0, len(anchors), 8):
                selected = anchors[start : start + 8]
                history = torch.stack(
                    [numeric[e]["geometry"][a - 1 : a + 1, :27] for e, a in selected]
                ).to(model.observation_mean.device)
                images = {
                    camera: torch.stack(
                        [
                            torch.from_numpy(
                                handles[e]["images"][camera][a - 1 : a + 1]
                            ).permute(0, 3, 1, 2)
                            for e, a in selected
                        ]
                    )
                    .to(device=model.observation_mean.device, dtype=torch.float32)
                    .div_(255)
                    for camera in CAMERAS
                }
                predicted = model.predict_targets(history, images).cpu()
                for row, (episode, anchor) in enumerate(selected):
                    stop = min(anchor + 32, len(numeric[episode]["geometry"]))
                    measured = numeric[episode]["geometry"][anchor:stop]
                    requested = numeric[episode]["actions"][anchor:stop]
                    count = stop - anchor
                    # Encoding and metrics receive real targets only, without padding.
                    prediction.append(predicted[row, :count])
                    target.append(pose_targets(measured, requested))
                    geometry.append(measured)
                    action.append(requested)
                    phase.append(phases[episode][anchor:stop])
                    offsets.append(torch.arange(count))
    predicted, targets, measured, requested, labels, positions = [
        torch.cat(values)
        for values in (prediction, target, geometry, action, phase, offsets)
    ]
    metrics = _errors(predicted, targets, measured, requested)
    normalized_error = (
        ((predicted - targets) / model.target_std.detach().cpu()).square().mean(-1)
    )
    summary = {
        "anchors": len(anchors),
        "valid_target_positions": len(targets),
        "normalized_mse": float(normalized_error.mean()),
        "all": _summarize(metrics, torch.ones(len(targets), dtype=torch.bool)),
        "by_prediction_offset": {
            str(index): _summarize(metrics, positions == index) for index in range(32)
        },
        "by_target_phase": {},
    }
    for index, name_phase in enumerate(_PHASES):
        selected = labels == index
        strong = selected & (requested[:, 6].abs() >= 0.5)
        strong_count = int(strong.sum())
        summary["by_target_phase"][name_phase] = {
            "valid_target_positions": int(selected.sum()),
            "strong_gripper_positions": strong_count,
            "strong_gripper_sign_agreement": (
                float((~metrics["gripper_wrong_sign"][strong]).float().mean())
                if strong_count
                else None
            ),
            **_summarize(metrics, selected),
        }
    torch.save(
        {
            "anchors": anchors,
            "prediction": predicted,
            "targets": targets,
            "geometry": measured,
            "actions": requested,
            "phase": labels,
            "prediction_offset": positions,
            "metrics": metrics,
        },
        output / f"{name}-records.pt",
    )
    _write(output / f"{name}.json", summary)
    return summary


def _gate(summary: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    stages = {}
    for name, value in summary["by_target_phase"].items():
        has_samples = (
            value["valid_target_positions"] > 0
            and value["strong_gripper_positions"] > 0
        )
        stages[name] = {
            "evaluable": has_samples,
            "passed": has_samples
            and (
                value["position_error_m"]["p95"] <= thresholds["tcp_p95_m"]
                and value["rotation_error_rad"]["p95"]
                <= thresholds["rotation_p95_degrees"] * torch.pi / 180
                and value["gripper_abs_error"]["mean"] <= thresholds["gripper_mae"]
                and value["strong_gripper_sign_agreement"]
                >= thresholds["strong_gripper_sign_agreement"]
            ),
        }
    return {
        "passed": all(value["passed"] for value in stages.values()),
        "stages": stages,
        "thresholds": thresholds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--prepare-only", action="store_true")
    operation.add_argument("--train", action="store_true")
    args = parser.parse_args()
    import yaml

    _require_single_process()
    torch.set_num_threads(1)
    config = yaml.safe_load(args.config.read_text())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists()
        for name in ("before.json", "after.json", "summary.json")
    ):
        raise FileExistsError("Refusing to overwrite an executed learnability check")
    parent = Path(config["data"]["manifest"])
    if _sha(parent) != config["data"]["manifest_sha256"]:
        raise ValueError("Parent manifest differs from the protocol")
    corpus = json.loads(parent.read_text())
    if corpus.get("complete") is not True or len(corpus["episodes"]) != 1024:
        raise ValueError("Expected the original complete 1024-episode corpus")
    derived = deepcopy(corpus)
    derived["episodes"] = [
        {**record, "path": str((parent.parent / record["path"]).resolve())}
        for record in corpus["episodes"][:8]
    ]
    derived.update(
        requested_episodes=8,
        parent_manifest={
            "path": str(parent),
            "sha256": _sha(parent),
            "selection": "first-eight-complete-episodes@1",
        },
    )
    _once(output / "eight-episode-manifest.json", derived)
    eligible = [
        (e, a)
        for e, record in enumerate(derived["episodes"])
        for a in range(1, record["length"])
    ]
    anchors = {eligible[(2 * index + 1) * len(eligible) // 512] for index in range(256)}
    anchors.update(
        (e, a)
        for e, record in enumerate(derived["episodes"])
        for a in range(record["length"] - 31, record["length"])
    )
    for record in derived["episodes"]:
        if _sha(Path(record["path"])) != record["sha256"]:
            raise ValueError("A selected episode differs from its frozen content hash")
    trainer = _build(
        config,
        output,
        device="cpu" if args.prepare_only else config["trainer"]["device"],
    )
    try:
        root = Path(__file__).resolve().parents[4]
        if subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip():
            raise ValueError(
                "Freeze a clean source commit before preparing or training"
            )
        files = [
            Path(__file__),
            root / "examples/learning/bc/_support/_offline_audit.py",
            root / "examples/learning/bc/_support/_pick_place.py",
            *list(
                (
                    root / "extensions/embodichain_models/embodichain_models/policy/bc"
                ).glob("*.py")
            ),
            *list((root / "embodichain/learning/sl").glob("*.py")),
        ]
        protocol = {
            "format": "bc-first-eight-learnability-v1",
            "scope": "Training-distribution learnability only; no physical evaluation, formal development, generalization or test claim",
            "source_commit": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_sha256": {
                str(path.relative_to(root)): _sha(path) for path in sorted(files)
            },
            "config": config,
            "config_sha256": _sha(args.config),
            "episodes": derived["episodes"],
            "derived_manifest_sha256": _sha(output / "eight-episode-manifest.json"),
            "normalization_sha256": _sha(Path(config["data"]["normalization"])),
            "normalization_statistics_sha256": _sha(
                output / "normalization-statistics.pt"
            ),
            "anchors": sorted(anchors),
            "anchor_rule": "256 midpoint-stratified anchors from all complete-history partial/full anchors, plus all final31 per episode; union sorted with duplicates removed",
            "phase_names": _PHASES,
            "phase_rule": "Reused measured-geometry audit phases over each full recorded episode; phases are reporting only and never policy inputs",
            "gate": config["gate"],
            "signature": trainer.signature,
            "expected_updates": 2000,
            "training_timeout_seconds": 900,
            "preparation_and_evaluation_budget_seconds": 300,
            "checkpoint_selection": "last only; fixed before/after anchors",
            "formal_test_opened": False,
        }
        _once(output / "protocol.json", protocol)
        _once(
            output / "protocol-sha256.json", {"sha256": _sha(output / "protocol.json")}
        )
        if args.prepare_only:
            _once(
                output / "prepared.json",
                {
                    "training_updates": 0,
                    "source_id": trainer.source.source_id,
                    "anchors": len(anchors),
                    "protocol_sha256": _sha(output / "protocol.json"),
                },
            )
            print(
                json.dumps(
                    {
                        "prepared": True,
                        "anchors": len(anchors),
                        "source_id": trainer.source.source_id,
                    }
                ),
                flush=True,
            )
            return
        if not (output / "prepared.json").exists():
            raise ValueError("Prepare and freeze the CPU protocol before training")
        started = time.monotonic()
        _write(output / "progress.json", {"stage": "before", "started": started})
        before = _evaluate(trainer.learner.model, protocol, output, "before")
        initial_buffers = {
            name: value.detach().cpu().clone()
            for name, value in trainer.learner.model.named_buffers()
        }
        trainer.learner.model.train()
        training_started = time.monotonic()
        _write(
            output / "progress.json", {"stage": "training", "started": training_started}
        )
        with _training_deadline(900):
            fit = trainer.fit()
        training_seconds = time.monotonic() - training_started
        trainer.source.close()
        _write(output / "fit.json", fit)
        _write(
            output / "progress.json", {"stage": "after", "started": time.monotonic()}
        )
        after = _evaluate(trainer.learner.model, protocol, output, "after")
        gates = _gate(after, config["gate"])
        normalization_unchanged = all(
            torch.equal(value.detach().cpu(), initial_buffers[name])
            for name, value in trainer.learner.model.named_buffers()
        )
        result = {
            "source_commit": protocol["source_commit"],
            "protocol_sha256": _sha(output / "protocol.json"),
            "num_updates": fit["num_updates"],
            "initial_policy_updates": 20000,
            "training_seconds": training_seconds,
            "worker_seconds": time.monotonic() - started,
            "before_normalized_mse": before["normalized_mse"],
            "after_normalized_mse": after["normalized_mse"],
            "relative_mse_reduction": 1
            - after["normalized_mse"] / before["normalized_mse"],
            "gate": gates,
            "normalization_unchanged": normalization_unchanged,
            "checkpoint_sha256": _sha(Path(fit["checkpoint"])),
            "formal_test_opened": False,
            "next_action": (
                "Eligible for separately frozen paired sampling comparison"
                if gates["passed"]
                else "Stop; no paired or full-data training under this protocol"
            ),
        }
        _write(output / "summary.json", result)
        _write(
            output / "progress.json",
            {
                "stage": "complete",
                "num_updates": fit["num_updates"],
                "gate_passed": gates["passed"],
            },
        )
        print(json.dumps(result), flush=True)
    finally:
        trainer.source.close()


if __name__ == "__main__":
    main()
