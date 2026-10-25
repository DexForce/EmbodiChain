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

"""Prepare a separately declared window-mode experiment without formal tuning."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from embodichain.learning.sl import SupervisedTrainer
from embodichain_models.policy.bc.sl_recipe import build_supervised_trainer

from ._pick_place import TARGET_SEMANTICS, encode_targets, source
from ._train import _require_single_process

__all__: list[str] = []


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_trainer(config: dict[str, Any]) -> SupervisedTrainer:
    """Build a weight-only warm start through the installable generic factory.

    Args:
        config: Frozen candidate mapping with corpus/checkpoint checksums,
            original statistics, shared optimizer/budget and auxiliary panel.

    Returns:
        Common SL trainer at update zero with a fresh optimizer. The original
        formal run, optimizer, normalization and acceptance driver are untouched.
    """
    _require_single_process()
    data, initialization = config["data"], config["initialization"]
    if _sha(data["manifest"]) != data["manifest_sha256"]:
        raise ValueError("Candidate corpus differs from the frozen manifest")
    if _sha(initialization["checkpoint"]) != initialization["checkpoint_sha256"]:
        raise ValueError("Candidate initialization checkpoint checksum differs")
    state = torch.load(
        initialization["checkpoint"], map_location="cpu", weights_only=False
    )
    if state["learner"]["num_updates"] != initialization["policy_step"]:
        raise ValueError("Candidate initialization policy step differs")
    original = source(data["manifest"])
    try:
        if len(original.episodes) != 1024:
            raise ValueError("Candidate requires the frozen 1024-episode corpus")
        statistics = torch.load(
            data["normalization"], map_location="cpu", weights_only=True
        )
        if statistics["source_id"] != original.source_id:
            raise ValueError("Candidate statistics differ from the frozen corpus view")
        statistics = statistics["statistics"]
    finally:
        original.close()
    weights = state["learner"]["model"]
    if any(not torch.equal(weights[name], value) for name, value in statistics.items()):
        raise ValueError("Candidate normalization differs from checkpoint buffers")
    statistics_path = Path(data["prepared_normalization"])
    if statistics_path.exists():
        saved = torch.load(statistics_path, map_location="cpu", weights_only=True)
        if saved.keys() != statistics.keys() or any(
            not torch.equal(saved[name], value) for name, value in statistics.items()
        ):
            raise ValueError("Refusing to replace prior candidate statistics")
    else:
        statistics_path.parent.mkdir(parents=True, exist_ok=True)
        with statistics_path.open("xb") as handle:
            torch.save(statistics, handle)
    numerics = state["signature"]["numerics"]
    torch.backends.cudnn.allow_tf32 = numerics["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = numerics["matmul_allow_tf32"]
    generic = {
        "model": weights["_extra_state"]["config"],
        "data": {
            "manifest": data["manifest"],
            "normalization": str(statistics_path),
            "prefetch_batches": data["prefetch_batches"],
            "reader": {
                "observation_key": "geometry",
                "observation_slice": [0, 27],
                "target_encoder": encode_targets,
                "target_fields": ("geometry", "actions"),
                "target_encoder_id": TARGET_SEMANTICS
                + ":relative-scales-0.02m-0.1rad@1",
                "include_partial_targets": data["include_partial_targets"],
            },
        },
        "objective": config["objective"],
        "optimizer": config["optimizer"],
        "trainer": config["trainer"],
    }
    trainer = build_supervised_trainer(generic)
    trainer.learner.model.load_state_dict(weights)
    trainer.signature.update(
        research_recipe="bc-tail-conditioning-candidate-v1",
        initialization={
            "checkpoint_sha256": initialization["checkpoint_sha256"],
            "source_policy_step": initialization["policy_step"],
            "weights_only": True,
            "optimizer": "fresh AdamW; no inherited schedule or update count",
        },
        recipe_source_sha256=_sha(__file__),
        numerics=numerics,
        environment=state["signature"]["environment"],
        camera_profile_version="pick-place-v2",
        action_horizon=config["evaluation"]["action_horizon"],
        auxiliary_evaluation=config["evaluation"],
    )
    return trainer


def main() -> None:
    """Prepare or explicitly train a candidate through the shared SL trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--include-partial-targets",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--prepare-only", action="store_true")
    operation.add_argument("--train", action="store_true")
    args = parser.parse_args()
    import yaml

    config = yaml.safe_load(args.config.read_text())
    config["data"]["include_partial_targets"] = args.include_partial_targets
    config["trainer"]["output_dir"] = str(args.output_dir.resolve())
    planned = deepcopy(config)
    if args.prepare_only:
        config["trainer"]["device"] = "cpu"
    if args.report.exists():
        raise FileExistsError(f"Refusing to replace candidate report {args.report}")
    torch.set_num_threads(1)
    trainer = build_trainer(config)
    try:
        result = {"num_updates": 0} if args.prepare_only else trainer.fit()
        report = {
            "prepared_only": args.prepare_only,
            "result": result,
            "planned_config": planned,
            "preparation_device": "cpu" if args.prepare_only else None,
            "source_id": trainer.source.source_id,
            "signature": trainer.signature,
            "learning_acceptance": "pending; no physical evaluation performed",
            "formal_test_opened": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        print(
            json.dumps(
                {
                    "prepared_only": args.prepare_only,
                    "num_updates": result["num_updates"],
                    "source_id": trainer.source.source_id,
                }
            ),
            flush=True,
        )
    finally:
        trainer.source.close()


if __name__ == "__main__":
    main()
