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
"""Checkpoint-compatible WorldParticle inference, independent of live state."""

from __future__ import annotations
from typing import Any
from pathlib import Path
import math
import warnings

import numpy as np

SUPPORTED_TASKS = ("fancy_fluid", "cloth_on_ball", "duck_drop", "fluid")


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Read one of the four supported official model configurations."""
    import yaml

    if path is None or not Path(path).is_file():
        raise ValueError("WorldParticle config must name an existing local file")
    with Path(path).open() as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict) or cfg.get("data") not in SUPPORTED_TASKS:
        raise ValueError(f"WorldParticle supported data tasks are {SUPPORTED_TASKS}")
    dt = float(cfg.get("timestep", 0.0))
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Checkpoint timestep must be positive and finite")
    if cfg.get("verbose", False):
        raise ValueError("WorldParticle verbose/debug mode must be disabled")
    gravity = np.asarray(cfg.get("gravity"), dtype=np.float32)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("Checkpoint gravity must be a finite 3-vector")
    return dict(cfg, timestep=dt)


def normalize_features(task: str, raw: np.ndarray) -> np.ndarray:
    """Preserve each official dataset's feature transform and input precision."""
    channels = {"fancy_fluid": 3, "cloth_on_ball": 7, "duck_drop": 1, "fluid": 2}
    if raw.ndim != 2 or raw.shape[1] != channels[task] or not len(raw):
        raise ValueError(f"{task} raw feats must have shape (N, {channels[task]})")
    if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
        raise ValueError("WorldParticle features must be finite numeric values")
    if task == "fancy_fluid":
        feature = raw[:, 2:3]
        feature = (feature - feature.mean(axis=0)) / (feature.std(axis=0) + 1e-8)
    elif task == "cloth_on_ball":
        mass, young, density = raw[:, 0], raw[:, -2], raw[:, -1]
        if np.any(young <= 0):
            raise ValueError("Cloth Young modulus feature must be positive")
        young = (np.log10(young) - np.log10(5e3)) / (np.log10(3e5) - np.log10(5e3))
        density = (density - 50.0) / (500.0 - 50.0)
        mass = (mass - mass.mean(axis=0)) / (mass.std(axis=0) + 1e-8)
        feature = np.concatenate(
            [mass[:, None], young[:, None], density[:, None]], axis=1
        )
    else:
        feature = raw
    feature = np.asarray(feature, dtype=np.float32)
    if not np.isfinite(feature).all():
        raise ValueError("Normalized WorldParticle features must be finite")
    return feature


def load_assets(options: dict[str, Any]) -> tuple:
    """Load static NPZ features/boundaries/topology; never trajectory state.

    Config task selects raw feature semantics. Topology-enabled checkpoints
    require the original CSR neighbors, not a newly inferred physical graph.
    """
    from ._network._data import (
        uses_topology_neighbors,
        get_default_topology_search_radius,
    )

    unknown = set(options) - {"checkpoint", "config", "inputs", "model_to_world"}
    if unknown:
        raise ValueError(f"Unknown WorldParticle options: {sorted(unknown)}")
    for key in ("checkpoint", "config", "inputs"):
        if key not in options or not Path(options[key]).is_file():
            raise ValueError(f"WorldParticle {key} must name an existing local file")
    cfg = load_config(options["config"])
    task = cfg["data"]
    use_topology = cfg.get("use_topology_conv")
    if use_topology is None:
        use_topology = uses_topology_neighbors(task)
    topology_radius = cfg.get("topology_search_radius")
    if topology_radius is None and use_topology:
        topology_radius = get_default_topology_search_radius(task)
    rotation = np.asarray(options.get("model_to_world", np.eye(3)), dtype=np.float32)
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ValueError("model_to_world must be a proper orthonormal 3x3 rotation")
    with np.load(options["inputs"], allow_pickle=False) as data:
        raw = data["feats"]
        feature = normalize_features(task, raw)
        box = np.asarray(data["box"], dtype=np.float32)
        box_feats = np.asarray(data["box_feats"], dtype=np.float32)
        topology = ()
        if use_topology:
            if not {"neighbors_index", "neighbors_row_splits"}.issubset(data.files):
                raise ValueError(
                    "Topology-enabled checkpoint requires CSR neighbors_index/row_splits"
                )
            index = data["neighbors_index"]
            rows = data["neighbors_row_splits"]
            if (
                index.ndim != 1
                or rows.ndim != 1
                or len(rows) != len(feature) + 1
                or not np.issubdtype(index.dtype, np.integer)
                or not np.issubdtype(rows.dtype, np.integer)
                or rows[0] != 0
                or rows[-1] != len(index)
                or np.any(np.diff(rows) < 0)
                or np.any(index < 0)
                or np.any(index >= len(feature))
            ):
                raise ValueError("Invalid particle topology CSR layout or index range")
            topology = (index.astype(np.int64), rows.astype(np.int64))
    if (
        box.ndim != 2
        or box.shape[1] != 3
        or not len(box)
        or box_feats.shape != box.shape
    ):
        raise ValueError("box and box_feats must have nonempty matching shapes (M, 3)")
    if not all(np.isfinite(x).all() for x in (box, box_feats)):
        raise ValueError("WorldParticle static inputs must be finite")
    cfg.update(
        resolved_other_feats_channels=feature.shape[1],
        resolved_obstacle_feats_channels=3,
        resolved_use_topology_conv=bool(use_topology),
        resolved_topology_search_radius=topology_radius,
    )
    return cfg, (feature, box, box_feats) + topology, rotation


def load_checkpoint_state(
    network: Any, state: dict[str, Any], *, legacy_fluid: bool = False
) -> None:
    """Strictly load weights, migrating only one known unused legacy buffer."""
    state = dict(state)
    missing = set(network.state_dict()) - set(state)
    legacy = "local_feature_extractor.topology_filter_extent_tensor"
    if legacy_fluid and missing == {legacy}:
        state[legacy] = network.state_dict()[legacy]
        warnings.warn(
            "Legacy Deep Fluid checkpoint omits the unused topology_filter_extent_tensor buffer; using its configured default. All trainable weights remain strict.",
            UserWarning,
            stacklevel=2,
        )
    network.load_state_dict(state, strict=True)


def load_network(cfg: dict[str, Any], checkpoint: str | Path, device: str) -> Any:
    """Construct the original architecture and strictly load its model tensors."""
    import sys
    import platform
    import torch
    from importlib.metadata import version

    if (
        sys.platform != "linux"
        or platform.machine() != "x86_64"
        or sys.version_info[:2] != (3, 12)
    ):
        raise RuntimeError(
            "WorldParticle currently requires Linux x86_64 / Python 3.12"
        )
    if (
        torch.__version__ != "2.7.1+cu128"
        or not torch._C._GLIBCXX_USE_CXX11_ABI
        or torch.cuda.get_device_capability(device) != (12, 0)
    ):
        raise RuntimeError(
            "WorldParticle requires the validated Torch 2.7.1+cu128 / CXX11 ABI / SM120 runtime"
        )
    if (
        version("open3d") != "0.19.0+1e7b174"
        or version("flash-attn").split("+")[0] != "2.8.3"
    ):
        raise RuntimeError(
            "Install EmbodiChain's validated worldparticle extra; the Open3D/FlashAttention operators do not match"
        )
    from ._network.particle_network_cross_attn_feat import (
        ParticleNetworkCrossAttnLocalFeature,
    )

    names = (
        "kernel_size",
        "cconv_embedding_dim",
        "radius_scale",
        "particle_radius",
        "slot_compression_ratio",
        "slot_attn_iters",
        "slot_attn_heads",
        "particle_position_rope_dim",
        "num_decoder_layers",
        "decoder_attn_heads",
        "decoder_attn_ffn_hidden_dim",
        "decoder_attn_dropout",
        "output_hidden_dim",
        "output_layers",
        "velhead",
        "topology_blend_alpha",
    )
    kwargs = {key: cfg[key] for key in names if key in cfg}
    kwargs.update(
        data=cfg["data"],
        gravity=cfg["gravity"],
        timestep=cfg["timestep"],
        other_feats_channels=cfg["resolved_other_feats_channels"],
        obstacle_feats_channels=cfg["resolved_obstacle_feats_channels"],
        verbose=False,
        use_topology_conv=cfg["resolved_use_topology_conv"],
        topology_search_radius=cfg["resolved_topology_search_radius"],
    )
    network = ParticleNetworkCrossAttnLocalFeature(**kwargs)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint_data["state_dict"]
    prefix = "model."
    if not state or any(not key.startswith(prefix) for key in state):
        raise ValueError(
            "Expected a WorldParticle Lightning checkpoint with model.* tensors"
        )
    load_checkpoint_state(
        network,
        {key[len(prefix) :]: value for key, value in state.items()},
        legacy_fluid=cfg["data"] == "fluid" and not cfg["resolved_use_topology_conv"],
    )
    return network.to(device).eval().requires_grad_(False)
