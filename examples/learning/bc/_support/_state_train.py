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

"""Train causal PickPlace state BC recipes with the common SL trainer."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDictBase

from embodichain.data_pipeline.datasets.sequence import BatchSource, SequenceSpecCfg
from embodichain.learning.sl import (
    SequenceBatchSource,
    SupervisedBatch,
    SupervisedLearner,
    SupervisedTrainer,
    SupervisedTrainerCfg,
)
from embodichain_models.policy.bc import ChunkBCModel, ChunkBCModelCfg, ChunkBCObjective

from ._pick_place import TARGET_SEMANTICS, pose_targets
from ._state_data import (
    ACTION_CONTRACTS,
    ACTION_SEMANTICS,
    OBSERVATION_SCHEMA,
    STEP_DT,
    frame_source,
    resident_source,
    sha256,
    state_schema,
    validate_environment,
)

__all__: list[str] = []


def _check_context(context: TensorDictBase, history: int, horizon: int) -> None:
    expected = (context.batch_size[0], history + horizon - 1)
    if (
        tuple(context.batch_size) != expected
        or context["obs"].shape != (*expected, 51)
        or context["actions"].shape != (*expected, 7)
        or context["obs"].dtype != torch.float32
        or context["actions"].dtype != torch.float32
        or context["valid"].dtype != torch.bool
        or context["loss_mask"].dtype != torch.bool
        or not context["valid"][:, history - 1].all()
        or (context["loss_mask"] & ~context["valid"]).any()
    ):
        raise ValueError("State BC context must contain aligned causal numeric frames")


def encode_action(context: TensorDictBase) -> SupervisedBatch:
    """Supervise the recorded request at the current pre-action observation."""
    _check_context(context, 1, 1)
    mask = context["loss_mask"][:, :1]
    if not mask.all():
        raise ValueError("Every current state must have a valid action target")
    return SupervisedBatch(
        {"history": context["obs"][:, :1]},
        context["actions"][:, :1],
        {"target_valid": mask},
    )


def encode_chunk(context: TensorDictBase) -> SupervisedBatch:
    """Encode 64 causal commands and copy current state into missing left history."""
    _check_context(context, 2, 64)
    observation, request = context["obs"], context["actions"]
    history = observation[:, :2].clone()
    history[:, 0] = torch.where(context["valid"][:, :1], history[:, 0], history[:, 1])
    mask = context["loss_mask"][:, 1:]
    if not mask.any():
        raise ValueError("At least one future action must be valid")
    # Never encode synthetic right-padding: a zero rotation is not a real pose.
    target = observation.new_zeros(len(observation), 64, 10)
    target[mask] = pose_targets(observation[:, 1:][mask], request[:, 1:][mask])
    return SupervisedBatch({"history": history}, target, {"target_valid": mask})


def sequence_source(
    source: BatchSource, *, recipe: str, immutable: bool = False
) -> SequenceBatchSource:
    """Bind a resident or caller-owned live ODS source to the same BC encoders.

    Mutable/live sources can train and save snapshots; the shared adapter rejects
    exact resume unless the caller explicitly provides an immutable resident
    source with no delivery quota. Producer start, waits and refill remain owned
    by the host. Source-native sampling does not claim frame-uniform draws.
    """
    action = source.schema.get("action", {})
    terms = tuple(
        entry.get("term", {}).get("contract") for entry in action.get("terms", ())
    )
    if (
        action.get("semantics") != "policy_request"
        or action.get("step_dt") != STEP_DT
        or terms != ACTION_CONTRACTS
        or source.schema.get("version") != OBSERVATION_SCHEMA
    ):
        raise ValueError("Data source differs from the state PickPlace action schema")
    if recipe == "maniskill_action":
        spec, encoder = (
            SequenceSpecCfg(history_length=1, prediction_horizon=1),
            encode_action,
        )
    elif recipe == "legacy_chunk":
        spec, encoder = (
            SequenceSpecCfg(history_length=2, prediction_horizon=64),
            encode_chunk,
        )
    else:
        raise ValueError("Unknown state BC recipe")
    return SequenceBatchSource(
        source,
        spec,
        encoder=encoder,
        encoder_version=f"pick-place-{recipe}-preaction-relative-v1",
        immutable=immutable,
    )


def build_trainer(
    config: dict[str, Any],
    *,
    source: BatchSource | None = None,
    immutable: bool = False,
) -> SupervisedTrainer:
    """Compose one fixed-corpus or caller-owned ODS recipe with shared SL."""
    recipe = config["recipe"]
    if recipe not in {"maniskill_action", "legacy_chunk"}:
        raise ValueError("State BC recipe must be maniskill_action or legacy_chunk")
    data = config["data"]
    view = Path(data["view"]).resolve() if data.get("view") else None
    if view is None:
        if source is None or not data.get("normalization"):
            raise ValueError(
                "Provide an offline state view or a source with frozen normalization"
            )
        statistics_path = Path(data["normalization"]).resolve()
    else:
        prepared = json.loads((view.parent / "prepared.json").read_text())
        if sha256(view) != prepared["view_sha256"]:
            raise ValueError("Frozen state view changed")
        statistics_path = Path(
            data.get("normalization") or prepared["normalization"]
        ).resolve()
        if (
            not data.get("normalization")
            and sha256(statistics_path) != prepared["normalization_sha256"]
        ):
            raise ValueError("Frozen state normalization changed")
    environment_path = Path(config["environment"]).resolve()
    environment = json.loads(environment_path.read_text())
    validate_environment(environment)
    cfg = SupervisedTrainerCfg(**config["trainer"])
    cfg.validate()
    torch.manual_seed(cfg.seed)
    if recipe == "maniskill_action":
        model_cfg = ChunkBCModelCfg(
            observation_dim=51,
            target_dim=7,
            history_length=1,
            prediction_horizon=1,
            hidden_dims=(256, 256),
            cameras=(),
            target_semantics=ACTION_SEMANTICS,
            normalization_clip=1e6,
        )
        action_horizon = 1
    else:
        model_cfg = ChunkBCModelCfg(
            observation_dim=51,
            target_dim=10,
            history_length=2,
            prediction_horizon=64,
            hidden_dims=(512, 512),
            cameras=(),
            target_semantics=TARGET_SEMANTICS,
        )
        action_horizon = 16
    model = ChunkBCModel(model_cfg)
    statistics = torch.load(statistics_path, map_location="cpu", weights_only=True)
    if recipe in statistics:
        statistics = statistics[recipe]
    model.set_normalization(**statistics)
    model.to(cfg.device)
    if config["objective"].get("type") != "masked_normalized_target_mse" or set(
        config["objective"]
    ) - {"type", "observation_noise"}:
        raise ValueError("Unknown state BC objective")
    noise = float(config["objective"].get("observation_noise", 0.0))
    objective = (
        ChunkBCObjective() if noise == 0 else ChunkBCObjective(observation_noise=noise)
    )
    optimizer_cfg = config["optimizer"]
    rate = float(optimizer_cfg["learning_rate"])
    decay = float(optimizer_cfg["weight_decay"])
    maximum = float(optimizer_cfg["max_grad_norm"])
    if not (
        0 < rate < float("inf")
        and 0 <= decay < float("inf")
        and 0 <= maximum < float("inf")
    ):
        raise ValueError("Invalid state BC optimizer values")
    if optimizer_cfg["type"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=rate, weight_decay=decay)
    elif optimizer_cfg["type"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=decay)
    else:
        raise ValueError("State BC optimizer must be Adam or AdamW")
    if source is None and recipe == "maniskill_action":
        assert view is not None
        batch_source = frame_source(view)
        sampling = "uniform-valid-action-frame-with-replacement@1"
    else:
        assert view is not None or source is not None
        actual_source = resident_source(view) if source is None else source
        batch_source = sequence_source(
            actual_source,
            recipe=recipe,
            immutable=True if source is None else immutable,
        )
        sampling = (
            "resident-episode-uniform-context@1"
            if source is None
            else "caller-source-context@1"
        )
    learner = SupervisedLearner(model, objective, optimizer, max_grad_norm=maximum)
    signature = {
        "recipe": "pick-place-state-bc-v1",
        "state_recipe": recipe,
        "model": model.get_extra_state(),
        "objective": {
            "type": "masked_normalized_target_mse",
            "observation_noise": noise,
        },
        "optimizer": {
            "type": optimizer_cfg["type"],
            "learning_rate": rate,
            "weight_decay": decay,
            "max_grad_norm": maximum,
        },
        "data_source": batch_source.source_id,
        "sampling": sampling,
        "view_sha256": sha256(view) if view is not None else None,
        "normalization_sha256": sha256(statistics_path),
        "environment": environment,
        "environment_sha256": sha256(environment_path),
        "action_contracts": list(ACTION_CONTRACTS),
        "action_horizon": action_horizon,
        "step_dt": STEP_DT,
        "observation_schema": OBSERVATION_SCHEMA,
        "schema": state_schema(),
        "numerics": {
            "dtype": "float32",
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "action_units": {
            "translation_m": 0.02,
            "rotation_rad": 0.1,
            "request_clip": [-1, 1],
        },
    }
    return SupervisedTrainer(learner, batch_source, cfg, signature=signature)


def _write(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite state BC artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    recipe_option = parser.add_mutually_exclusive_group(required=True)
    recipe_option.add_argument("--config", type=Path)
    recipe_option.add_argument("--recipe", choices=("maniskill_action", "legacy_chunk"))
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--prepare-only", action="store_true")
    operation.add_argument("--train", action="store_true")
    args = parser.parse_args()
    import yaml

    torch.set_num_threads(1)
    config_path = args.config
    if config_path is None:
        name = (
            "state-maniskill.yaml"
            if args.recipe == "maniskill_action"
            else "state-chunk.yaml"
        )
        config_path = Path(__file__).resolve().parents[1] / "configs" / name
    config = yaml.safe_load(config_path.read_text())
    config["data"]["view"] = str(args.view.resolve())
    config["environment"] = str(args.environment.resolve())
    config["trainer"].update(
        seed=args.seed,
        device=args.device,
        output_dir=str(args.output_dir.resolve() / "training"),
    )
    planned = deepcopy(config)
    if args.prepare_only:
        config["trainer"]["device"] = "cpu"
    trainer = build_trainer(config)
    output = args.output_dir.resolve()
    prepared_path = output / "prepared.json"
    prepared = {
        "recipe": planned,
        "signature": trainer.signature,
        "source_id": trainer.source.source_id,
        "config_sha256": sha256(config_path),
        "training_updates": 0,
        "sampling_reference": (
            "Offline single-step uses uniform real frames with replacement; "
            "ManiSkill's RandomSampler shuffles frames without replacement per epoch. "
            "The marginal distribution matches, but random sequences differ."
            if planned["recipe"] == "maniskill_action"
            else "Resident sequence sampling selects an episode and a valid context anchor."
        ),
    }
    if args.prepare_only:
        _write(prepared_path, prepared)
        print(
            json.dumps({"prepared": True, "source_id": trainer.source.source_id}),
            flush=True,
        )
        return
    if not prepared_path.is_file() or json.loads(prepared_path.read_text()) != prepared:
        raise ValueError(
            "Prepare and freeze the identical state recipe before training"
        )
    if args.resume is not None:
        trainer.resume(args.resume)
    if args.stop_after is not None and args.stop_after < trainer.learner.num_updates:
        raise ValueError("stop_after precedes the restored update cursor")
    checkpoints = [int(value) for value in config["checkpoint_steps"]]
    if (
        checkpoints != sorted(set(checkpoints))
        or not checkpoints
        or checkpoints[-1] != trainer.cfg.total_updates
    ):
        raise ValueError(
            "Checkpoint boundaries must be ordered and end at total_updates"
        )
    if args.stop_after is not None:
        checkpoints = [
            min(args.stop_after, step)
            for step in checkpoints
            if step > trainer.learner.num_updates
        ]
        checkpoints = sorted(set(checkpoints))
    results = []
    for boundary in checkpoints:
        if boundary <= trainer.learner.num_updates:
            continue
        result = trainer.fit(stop_after=boundary)
        destination = Path(trainer.cfg.output_dir) / f"step_{boundary}.pt"
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite fixed checkpoint {destination}"
            )
        trainer.save(destination)
        results.append({**result, "fixed_checkpoint": str(destination)})
    report_path = output / f"completed-{trainer.learner.num_updates:06d}.json"
    _write(
        report_path,
        {
            "result": results,
            "signature": trainer.signature,
            "formal_test_opened": False,
        },
    )
    print(
        json.dumps(
            {
                "num_updates": trainer.learner.num_updates,
                "checkpoints": [item["fixed_checkpoint"] for item in results],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
