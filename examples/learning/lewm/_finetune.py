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

"""Multi-step predictor ablation with frozen, content-bound visual embeddings."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator

import h5py
import torch

from embodichain.learning.sl import (
    LossResult,
    SupervisedBatch,
    SupervisedLearner,
    SupervisedTrainer,
    SupervisedTrainerCfg,
)
from embodichain_models.dynamics.lewm import export_lewm, load_lewm, preprocess_rgb


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def _cache_embeddings(
    model: Any,
    manifest: dict[str, Any],
    corpus: str,
    split: str,
    destination: Path,
    device: str,
    max_bytes: int,
) -> dict[str, Any]:
    """Encode bounded image batches once and bind them to the exact base export."""
    source_hash = _sha256(Path(corpus))
    source_id = f"lewm-hdf5:{source_hash}:{split}:{model.cfg.history_size}:1"
    source_key = "training_source" if split == "train" else "validation_source"
    if manifest["metadata"][source_key] != source_id:
        raise ValueError("fine-tuning corpus differs from the pretrained model source")
    identity = {
        "source_id": source_id,
        "weights_sha256": manifest["weights_sha256"],
        "encoding": "FP32 eval, at most 64 frames per encode call",
    }
    if destination.exists():
        payload = torch.load(destination, map_location="cpu", weights_only=True)
        if payload["identity"] != identity:
            raise ValueError("latent cache has a different corpus or encoder identity")
        with h5py.File(corpus, "r") as file:
            success = {
                int(e.attrs["seed"]): bool(e.attrs["success"])
                for e in file["episodes"].values()
            }
        for episode in payload["episodes"]:
            episode["success"] = success[episode["seed"]]
        return payload
    model.eval()
    episodes = []
    resident_bytes = 0
    with h5py.File(corpus, "r") as file:
        if file.attrs["split"] != split:
            raise ValueError("fine-tuning corpus has a different split")
        total_frames = sum(len(e["pixels"]) for e in file["episodes"].values())
        estimated = total_frames * (model.cfg.embed_dim + model.cfg.action_dim) * 4
        if estimated > max_bytes:
            raise ValueError(
                "latent corpus exceeds the explicit resident-memory budget"
            )
        for index, episode in enumerate(file["episodes"].values()):
            chunks = []
            for start in range(0, len(episode["pixels"]), 64):
                pixels = torch.from_numpy(episode["pixels"][start : start + 64]).to(
                    device
                )[None]
                chunks.append(
                    model.encode(preprocess_rgb(pixels, model.cfg.image_size))[0].cpu()
                )
            embeddings = torch.cat(chunks)
            actions = torch.from_numpy(episode["actions"][:]).float()
            if len(embeddings) != len(actions) + 1:
                raise ValueError("latent cache action/frame alignment differs")
            resident_bytes += (embeddings.numel() + actions.numel()) * 4
            episodes.append(
                {
                    "seed": int(episode.attrs["seed"]),
                    "embeddings": embeddings,
                    "actions": actions,
                    "success": bool(episode.attrs["success"]),
                }
            )
            if index % 64 == 0:
                print(f"cache {split}: {index + 1}/{len(file['episodes'])}", flush=True)
    payload = {
        "identity": identity,
        "resident_bytes": resident_bytes,
        "episodes": episodes,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return payload


class _LatentWindows:
    """Own bounded resident embeddings; gather windows without crossing episodes."""

    def __init__(
        self,
        payload: dict[str, Any],
        history: int,
        horizon: int,
        device: str,
        *,
        successful_tail_fraction: float = 0.0,
        tail_steps: int = 10,
    ) -> None:
        if (
            history < 1
            or horizon < 1
            or not 0 <= successful_tail_fraction <= 1
            or tail_steps < 1
        ):
            raise ValueError("latent windows need positive history and horizon")
        self.history, self.horizon = history, horizon
        self.successful_tail_fraction = successful_tail_fraction
        frames, actions, starts = [], [], []
        priority = []
        window_cursor = 0
        offset = 0
        for episode in payload["episodes"]:
            latent, control = episode["embeddings"], episode["actions"]
            if len(latent) != len(control) + 1:
                raise ValueError("latent windows require T+1 frames and T actions")
            count = len(control) - history - horizon + 2
            if count > 0:
                starts.append(torch.arange(offset, offset + count))
                if episode.get("success", False):
                    first = max(0, len(control) - tail_steps - history + 1)
                    if first < count:
                        priority.append(
                            torch.arange(window_cursor + first, window_cursor + count)
                        )
                window_cursor += count
            frames.append(latent)
            actions.append(torch.cat((control, torch.zeros_like(control[:1]))))
            offset += len(latent)
        if not starts:
            raise ValueError("no complete multi-step latent windows")
        self.frames = torch.cat(frames).to(device)
        self.actions = torch.cat(actions).to(device)
        self.starts = torch.cat(starts).to(device)
        self.offsets = torch.arange(history + horizon, device=device)
        self.priority = (
            torch.cat(priority) if priority else torch.empty(0, dtype=torch.long)
        )
        if successful_tail_fraction and not len(self.priority):
            raise ValueError(
                "successful-tail sampling has no complete priority windows"
            )
        self.source_id = (
            "lewm-frozen-latents:"
            + json.dumps(payload["identity"], sort_keys=True)
            + f":{history}:{horizon}:successful_tail={successful_tail_fraction}:{tail_steps}"
        )

    def _batch(self, selected: torch.Tensor) -> SupervisedBatch:
        indices = (
            self.starts[selected.to(self.starts.device)][:, None] + self.offsets[None]
        )
        return SupervisedBatch(
            inputs={
                "embeddings": self.frames[indices],
                "actions": self.actions[indices[:, :-1]],
            },
            targets=None,
            metadata={"source_id": self.source_id},
        )

    def sample(self, batch_size: int, *, generator: torch.Generator) -> SupervisedBatch:
        prioritized = round(batch_size * self.successful_tail_fraction)
        uniform = torch.randint(
            len(self.starts), (batch_size - prioritized,), generator=generator
        )
        if prioritized:
            priority = self.priority[
                torch.randint(len(self.priority), (prioritized,), generator=generator)
            ]
            uniform = torch.cat((uniform, priority))
        return self._batch(uniform)

    def iter_batches(self, batch_size: int) -> Iterator[SupervisedBatch]:
        for start in range(0, len(self.starts), batch_size):
            yield self._batch(
                torch.arange(start, min(start + batch_size, len(self.starts)))
            )


class _RolloutObjective:
    """Train only predictor modules against fixed observed future embeddings."""

    def compute_loss(self, model: Any, batch: SupervisedBatch) -> LossResult:
        latent, actions = batch.inputs["embeddings"], batch.inputs["actions"]
        h = model.cfg.history_size
        teacher = model.predict(latent[:, :h], actions[:, :h])
        teacher_loss = (teacher - latent[:, 1 : h + 1]).square().mean()
        predicted = model.rollout(
            latent[:, :h], actions[:, : h - 1], actions[:, None, h - 1 :]
        )[:, 0]
        rollout_loss = (predicted - latent[:, h:]).square().mean()
        return LossResult(
            teacher_loss + rollout_loss,
            {
                "prediction_loss": teacher_loss.detach(),
                "rollout_loss": rollout_loss.detach(),
            },
            float(len(latent)),
        )


def finetune_predictor(
    export: str,
    train_data: str,
    validation_data: str,
    output: str,
    device: str,
    *,
    updates: int = 2000,
    horizon: int = 5,
    batch_size: int = 128,
    interval: int = 200,
    seed: int = 0,
    successful_tail_fraction: float = 0.0,
) -> dict[str, Any]:
    """Run a source-bound frozen-encoder ablation through the common SL trainer."""
    if min(updates, horizon, interval) < 1 or batch_size < 2:
        raise ValueError("invalid predictor fine-tuning budget")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    model, manifest = load_lewm(export, device)
    if model.cfg.action_block != 1:
        raise ValueError(
            "frozen-predictor ablations currently require single-step models"
        )
    if set(manifest["metadata"]["train_seeds"]) & set(
        manifest["metadata"]["validation_seeds"]
    ):
        raise ValueError("fine-tuning source seeds overlap")
    if manifest["metadata"]["data_contract"].get("render") is None:
        raise ValueError("fine-tuning requires a known source rendering contract")
    started = time.perf_counter()
    limit = 512 * 1024 * 1024
    cache_root = destination.parent / f"latents-{manifest['weights_sha256'][:12]}"
    train_cache = _cache_embeddings(
        model, manifest, train_data, "train", cache_root / "train.pt", device, limit
    )
    validation_cache = _cache_embeddings(
        model,
        manifest,
        validation_data,
        "validation",
        cache_root / "validation.pt",
        device,
        limit,
    )
    cache_seconds = time.perf_counter() - started
    train = _LatentWindows(
        train_cache,
        model.cfg.history_size,
        horizon,
        device,
        successful_tail_fraction=successful_tail_fraction,
    )
    validation = _LatentWindows(
        validation_cache, model.cfg.history_size, horizon, device
    )
    model.encoder.requires_grad_(False)
    model.projector.requires_grad_(False)
    objective = _RolloutObjective()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=5e-5, weight_decay=1e-3
    )
    learner = SupervisedLearner(model, objective, optimizer, max_grad_norm=1.0)

    def evaluate(current: SupervisedLearner) -> dict[str, float]:
        totals: dict[str, float] = {}
        count = 0
        current.model.eval()
        with torch.no_grad():
            for batch in validation.iter_batches(batch_size):
                result = objective.compute_loss(current.model, batch)
                for name, value in result.metrics.items():
                    totals[name] = totals.get(name, 0.0) + float(value) * result.weight
                count += int(result.weight)
        metrics = {name: value / count for name, value in totals.items()}
        print(f"validation {current.num_updates}: {metrics}", flush=True)
        return metrics

    baseline_metrics = evaluate(learner)
    config = SupervisedTrainerCfg(
        total_updates=updates,
        batch_size=batch_size,
        device=device,
        seed=seed,
        output_dir=str(destination),
        checkpoint_interval=min(interval, updates),
        validation_interval=min(interval, updates),
    )
    torch.manual_seed(seed)
    trainer = SupervisedTrainer(
        learner,
        train,
        config,
        signature={
            "recipe": "lewm-frozen-visual-multistep-v1",
            "base_weights": manifest["weights_sha256"],
            "horizon": horizon,
            "successful_tail_fraction": successful_tail_fraction,
            "validation_source": validation.source_id,
        },
        evaluate=evaluate,
    )
    trained_at = time.perf_counter()
    result = trainer.fit()
    training_seconds = time.perf_counter() - trained_at
    checkpoints = [(baseline_metrics["rollout_loss"], 0, None)]
    for line in (destination / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "validation/rollout_loss" in row:
            checkpoints.append(
                (
                    row["validation/rollout_loss"],
                    row["num_updates"],
                    destination / f"step_{row['num_updates']}.pt",
                )
            )
    loss, step, checkpoint = min(checkpoints, key=lambda item: item[0])
    if checkpoint is None:
        selected, _ = load_lewm(export, "cpu")
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        selected, _ = load_lewm(export, "cpu")
        selected.load_state_dict(state["learner"]["model"], strict=True)
    metadata = dict(manifest["metadata"])
    metadata["predictor_finetuning"] = {
        "base_weights_sha256": manifest["weights_sha256"],
        "horizon": horizon,
        "selected_update": step,
        "validation_rollout_loss": loss,
        "baseline_validation": baseline_metrics,
        "seed": seed,
        "frozen": ["encoder", "projector"],
        "successful_tail_fraction": successful_tail_fraction,
        "objective": "teacher-forced next-frame MSE plus mean recursive rollout MSE; visual representation fixed",
    }
    export_lewm(destination / "export", selected, metadata)
    return {
        "training": result,
        "selected_update": step,
        "validation_rollout_loss": loss,
        "baseline_validation": baseline_metrics,
        "cache_seconds": cache_seconds,
        "training_seconds": training_seconds,
        "cache_tensor_bytes": train_cache["resident_bytes"]
        + validation_cache["resident_bytes"],
        "export": str(destination / "export"),
    }
