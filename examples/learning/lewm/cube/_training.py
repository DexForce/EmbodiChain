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

"""Train portable Cube checkpoints using native RGB."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import h5py
import hdf5plugin  # noqa: F401 -- registers the source-corpus compression filter.
import numpy as np
import torch
import torch.nn.functional as F

from embodichain.extensions.models.dynamics.lewm import (
    LeWM,
    LeWMCfg,
    LeWMObjective,
    LeWMSequenceSource,
    export_lewm,
    preprocess_rgb,
)
from embodichain.learning.sl import SupervisedLearner

from .._official_native import _NATIVE_RENDER, _OFFICIAL_WEIGHTS_SHA256, _query_metadata
from .._reference import _sha
from ._resources import _CubeResources


def _official_model(resources: _CubeResources, device: str) -> LeWM:
    if _sha(resources.weights) != _OFFICIAL_WEIGHTS_SHA256:
        raise ValueError("Official Cube checkpoint checksum mismatch")
    query = _query_metadata(resources.queries)
    model = LeWM(LeWMCfg(action_dim=25, action_block=5)).to(device).eval()
    model.load_reference_weights(
        torch.load(resources.weights, map_location="cpu", weights_only=True)
    )
    model.set_action_statistics(
        torch.tensor(query["action_mean"]).repeat(5),
        torch.tensor(query["action_std"]).repeat(5),
    )
    return model


@torch.no_grad()
def _pairs(
    model: LeWM, pair_root: Path, source: h5py.File, ids: list[int], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    images, targets = [], []
    for episode in ids:
        with h5py.File(pair_root / f"{episode}.h5", "r") as file:
            rows, native = file["rows"][:], file["pixels"][:]
        offset = int(source["ep_offset"][episode])
        np.testing.assert_array_equal(rows, offset + np.arange(episode % 5, 200, 5))
        if native.shape != (40, 224, 224, 3) or native.dtype != np.uint8:
            raise ValueError(f"Unexpected paired image layout in episode {episode}")
        official = torch.as_tensor(source["pixels"][rows], device=device)
        images.append(torch.as_tensor(native, device=device))
        targets.append(model.encode(preprocess_rgb(official[:, None]))[:, 0])
    return torch.stack(images), torch.stack(targets)


@torch.no_grad()
def _visual_metrics(
    model: LeWM, images: torch.Tensor, target: torch.Tensor, indices: torch.Tensor
) -> dict[str, float]:
    encoded = torch.cat(
        [
            model.encode(preprocess_rgb(images.flatten(0, 1)[index, None]))[:, 0]
            for index in indices.split(64)
        ]
    )
    goal = target.flatten(0, 1)[indices]
    return {
        "mse": float(F.mse_loss(encoded, goal)),
        "cosine": float(F.cosine_similarity(encoded, goal).mean()),
    }


def _snapshot(model: LeWM) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def train_visual(
    resources: _CubeResources,
    data: Path,
    output: Path,
    *,
    updates: int,
    device: str,
) -> dict[str, Any]:
    """Fit the visual representation while preserving the official dynamics."""
    meta = json.loads((data / "dataset.json").read_text())
    queries = _query_metadata(resources.queries)
    train_ids, val_ids = (meta["episodes"][split] for split in ("train", "validation"))
    heldout = set(meta["eval_episodes_excluded"])
    if (
        meta["kind"] != "cube-native-paired-poses"
        or meta["render"] != _NATIVE_RENDER
        or meta["source_sha256"] != queries["source_sha256"]
        or meta["queries_sha256"] != queries["queries_sha256"]
        or not set(queries["episode_ids"]).issubset(heldout)
        or not heldout.isdisjoint(train_ids + val_ids)
        or not set(train_ids).isdisjoint(val_ids)
        or min(len(train_ids), len(val_ids), updates) < 1
    ):
        raise ValueError("Incompatible Cube paired dataset or episode split")
    output.mkdir(parents=True)
    start = time.monotonic()
    torch.manual_seed(3072)
    generator = torch.Generator(device=device).manual_seed(3072)
    model = _official_model(resources, device)
    with h5py.File(resources.data, "r") as source:
        train_x, train_y = _pairs(model, data / "pairs", source, train_ids, device)
        val_x, val_y = _pairs(model, data / "pairs", source, val_ids, device)
    model.requires_grad_(False)
    model.encoder.requires_grad_(True)
    model.projector.requires_grad_(True)
    indices = torch.linspace(0, val_y.shape[0] * val_y.shape[1] - 1, 600, device=device)
    indices = indices.round().long()
    params = list(model.encoder.parameters()) + list(model.projector.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 1.5e-5},
            {"params": model.projector.parameters(), "lr": 3e-5},
        ],
        weight_decay=1e-3,
    )
    baseline = _visual_metrics(model, val_x, val_y, indices)
    best, history = float("inf"), []
    for update in range(1, updates + 1):
        episodes = torch.randint(
            len(train_x), (32,), generator=generator, device=device
        )
        offset = torch.randint(
            train_x.shape[1] - 5, (32,), generator=generator, device=device
        )
        delta = torch.randint(1, 6, (32,), generator=generator, device=device)
        frames = torch.stack((offset, offset + delta), dim=1)
        desired = train_y[episodes[:, None], frames]
        batch = preprocess_rgb(train_x[episodes[:, None], frames])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = model.encode(batch)
        actual = encoded.float()
        loss = F.mse_loss(actual, desired) + 0.25 * F.mse_loss(
            actual[:, 1] - actual[:, 0], desired[:, 1] - desired[:, 0]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if update % 200 == 0 or update == updates:
            validation = {
                "update": update,
                **_visual_metrics(model, val_x, val_y, indices),
            }
            history.append(validation)
            print(validation, flush=True)
            if validation["mse"] < best:
                best = validation["mse"]
                selected, weights = validation, _snapshot(model)
    model.to("cpu").load_state_dict(weights)
    export = output / "export"
    export_lewm(
        export,
        model,
        meta
        | {
            "official_checkpoint_sha256": _OFFICIAL_WEIGHTS_SHA256,
            "method": "matched poses, native RGB and published visual targets",
            "validation": selected,
        },
    )
    return {
        "baseline": baseline,
        "history": history,
        "selected": selected,
        "export": str(export),
        "seconds": time.monotonic() - start,
    }


@torch.no_grad()
def _expert_metrics(
    model: LeWM, objective: LeWMObjective, source: LeWMSequenceSource, device: str
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator().manual_seed(16481)
    result: dict[str, float] = {}
    with torch.random.fork_rng(devices=[torch.device(device).index]):
        torch.manual_seed(16481)
        for _ in range(8):
            batch = source.sample(128, generator=generator).to(device)
            loss = objective.compute_loss(model, batch)
            for key, value in {"objective": loss.loss, **loss.metrics}.items():
                result[key] = result.get(key, 0.0) + float(value) / 8
    return result


def train_expert(
    resources: _CubeResources,
    data: Path,
    output: Path,
    *,
    updates: int,
    device: str,
    cache_mib: int,
) -> dict[str, Any]:
    """Warm-start joint LeWM prediction and SIGReg on successful native episodes."""
    if updates < 2 or cache_mib < 0:
        raise ValueError(
            "Expert training requires at least two updates and a valid cache"
        )
    output.mkdir(parents=True)
    start = time.monotonic()
    torch.manual_seed(3072)
    generator = torch.Generator().manual_seed(3072)
    options = {"action_block": 5, "cache_max_bytes": cache_mib * 1024**2}
    if cache_mib:
        options["cache_device"] = device
    train = LeWMSequenceSource(data / "train.h5", "train", **options)
    val = LeWMSequenceSource(data / "validation.h5", "validation", **options)
    heldout = set(train.metadata["eval_episodes_excluded"])
    if (
        train.metadata != val.metadata
        or train.metadata["render"] != _NATIVE_RENDER
        or not train.seeds.isdisjoint(val.seeds)
        or not (train.seeds | val.seeds).isdisjoint(heldout)
        or not set(_query_metadata(resources.queries)["episode_ids"]).issubset(heldout)
    ):
        raise ValueError("Incompatible Cube expert datasets or episode split")
    model = _official_model(resources, device)
    objective = LeWMObjective(sigreg_weight=0.09, knots=17, num_proj=1024)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    warmup = max(1, updates // 100)
    schedule = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=updates - warmup
            ),
        ],
        milestones=[warmup],
    )
    learner = SupervisedLearner(
        model, objective, optimizer, max_grad_norm=1.0, amp_dtype=torch.bfloat16
    )
    baseline = _expert_metrics(model, objective, val, device)
    best, history = float("inf"), []
    for update in range(1, updates + 1):
        learner.update(train.sample(128, generator=generator).to(device))
        schedule.step()
        if update % 150 == 0 or update == updates:
            validation = {
                "update": update,
                **_expert_metrics(model, objective, val, device),
            }
            history.append(validation)
            print(validation, flush=True)
            if validation["objective"] < best:
                best = validation["objective"]
                selected, weights = validation, _snapshot(model)
    model.to("cpu").load_state_dict(weights)
    export = output / "export"
    export_lewm(
        export,
        model,
        train.metadata
        | {
            "official_checkpoint_sha256": _OFFICIAL_WEIGHTS_SHA256,
            "method": "native expert warm start with prediction and SIGReg",
            "train_episodes": sorted(train.seeds),
            "validation_episodes": sorted(val.seeds),
            "validation": selected,
        },
    )
    return {
        "baseline": baseline,
        "history": history,
        "selected": selected,
        "export": str(export),
        "seconds": time.monotonic() - start,
    }
