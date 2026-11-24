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

# Portions of the waypoint actor follow Neural Motion Generator.
# MIT License
#
# Copyright (c) 2026 Chen Yang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tensor-only inference for NMG actor checkpoints."""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ._nmg_observation import _WaypointObservationLayout

_CHECKPOINT_FORMAT = "nmg.actor.v1"
_CONSTRAINT_TYPES = 4


class _NativeActor(nn.Module):
    """Four-token waypoint actor with optional future-waypoint attention."""

    def __init__(
        self,
        layout: _WaypointObservationLayout,
        *,
        hidden_dim: int,
        nhead: int,
        num_layers: int,
        ff_dim: int | None,
        future_context: bool,
        disable_future_residual: bool,
    ) -> None:
        super().__init__()
        self._slots = layout.num_waypoints
        self._joints = layout.num_controlled_joints
        self._abs_width = 7 + self._joints
        self._relative = layout.use_relative_obs
        self._periodic = layout.joint_periodic_features
        self._slices = layout.slices
        self._use_future = future_context and not disable_future_residual

        state_width = 7 + 2 * self._joints
        if self._periodic:
            state_width += 2 * self._joints
        target_width = self._abs_width * (1 + int(self._relative))
        self._descriptor_width = target_width + 5
        self.action_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.token_type_embedding = nn.Embedding(4, hidden_dim)
        self.waypoint_index_embedding = nn.Parameter(
            torch.empty(1, self._slots, hidden_dim)
        )
        self.waypoint_modality_embedding = nn.Embedding(_CONSTRAINT_TYPES, hidden_dim)
        self.state_proj = nn.Linear(state_width, hidden_dim)
        self.active_goal_proj = nn.Linear(target_width + 4, hidden_dim)
        self.waypoint_proj = nn.Linear(self._descriptor_width, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ff_dim or hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.actor_head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self._joints),
        )
        if future_context:
            self.future_descriptor_proj = nn.Linear(self._descriptor_width, hidden_dim)
            self.future_offset_proj = nn.Linear(1, hidden_dim, bias=False)
            self.future_memory_norm = nn.LayerNorm(hidden_dim)
            self.future_query = nn.Sequential(
                nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim)
            )
            self.future_attention = nn.MultiheadAttention(
                hidden_dim, nhead, dropout=0.0, batch_first=True
            )
            self.future_residual = nn.Linear(hidden_dim, 2 * hidden_dim)
            self.future_gate = nn.Linear(2 * hidden_dim, 1)

    def _future_update(
        self,
        readout: torch.Tensor,
        sequence: torch.Tensor,
        modality: torch.Tensor,
        valid: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        offsets = torch.arange(self._slots, device=sequence.device) - active[:, None]
        future = valid & (offsets > 0)
        memory = self.future_memory_norm(
            self.future_descriptor_proj(sequence.masked_fill(~future[..., None], 0))
            + modality @ self.waypoint_modality_embedding.weight
            + self.future_offset_proj(offsets.to(sequence.dtype).unsqueeze(-1))
        )
        has_future = future.any(dim=1, keepdim=True)
        memory = torch.cat([memory, torch.zeros_like(memory[:, :1])], dim=1)
        attended, _ = self.future_attention(
            self.future_query(readout).unsqueeze(1),
            memory,
            memory,
            key_padding_mask=torch.cat([~future, has_future], dim=1),
            need_weights=False,
        )
        update = self.future_residual(attended.squeeze(1))
        update = update * torch.sigmoid(self.future_gate(readout))
        return readout + torch.where(has_future, update, torch.zeros_like(update))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        b, n, j = obs.shape[0], self._slots, self._joints
        o = {name: obs[:, field] for name, field in self._slices.items()}
        masks = torch.stack([o["pos_mask"], o["rot_mask"], o["joint_mask"]], dim=-1)
        pos, rot, joint = (masks > 0.5).unbind(-1)
        valid = o["valid_mask"].bool()
        semantic_valid = o["valid_mask"] > 0.5
        modes = torch.full_like(o["valid_mask"], _CONSTRAINT_TYPES, dtype=torch.long)
        modes = torch.where(semantic_valid & pos & rot & ~joint, 0, modes)
        modes = torch.where(semantic_valid & pos & ~rot & ~joint, 1, modes)
        modes = torch.where(semantic_valid & ~pos & rot & ~joint, 2, modes)
        modes = torch.where(semantic_valid & ~pos & ~rot & joint, 3, modes)
        modality = F.one_hot(torch.where(valid, modes, 0), _CONSTRAINT_TYPES).to(
            obs.dtype
        )

        fields = ("waypoint_pos", "waypoint_quat", "waypoint_joint")
        widths = (3, 4, j)
        features = [
            o[name].reshape(b, n, width).masked_fill(~masks[..., i : i + 1].bool(), 0)
            for i, (name, width) in enumerate(zip(fields, widths, strict=True))
        ]
        if self._relative:
            relative = ("waypoint_rel_pos", "waypoint_rel_quat", "waypoint_joint_err")
            features.extend(
                o[name].reshape(b, n, width) * masks[..., i : i + 1]
                for i, (name, width) in enumerate(zip(relative, widths, strict=True))
            )
        sequence = torch.cat(
            [
                *features,
                masks,
                o["active_onehot"][..., None],
                o["valid_mask"][..., None],
            ],
            dim=-1,
        )

        active = torch.minimum(
            o["active_onehot"].argmax(dim=-1), o["valid_mask"].sum(dim=-1).long() - 1
        ).clamp_min(0)
        slot = active[:, None]
        active_descriptor = sequence.gather(
            1, slot[..., None].expand(-1, 1, self._descriptor_width)
        ).squeeze(1)
        mask_start = self._descriptor_width - 5
        goal = [
            active_descriptor[:, : self._abs_width],
            active_descriptor[:, mask_start : mask_start + 3],
        ]
        if self._relative:
            goal.append(active_descriptor[:, self._abs_width : mask_start])
        progress = torch.arange(n, dtype=obs.dtype, device=obs.device) / max(n - 1, 1)
        goal.append(progress.index_select(0, active)[:, None])

        padding = ~valid.gather(1, slot)
        current = torch.where(padding, 0, slot).squeeze(1)
        current_modality = modality.gather(
            1, current[:, None, None].expand(-1, 1, _CONSTRAINT_TYPES)
        ).squeeze(1)
        role = self.token_type_embedding.weight
        state = [o["joint"], o["eef"], o["last_action"]]
        if self._periodic:
            state.extend((o["joint_sin"], o["joint_cos"]))
        tokens = torch.stack(
            [
                self.action_token.expand(b, -1, -1).squeeze(1) + role[0],
                self.state_proj(torch.cat(state, dim=-1)) + role[1],
                self.active_goal_proj(torch.cat(goal, dim=-1)) + role[2],
                self.waypoint_proj(active_descriptor.masked_fill(padding, 0))
                + role[3]
                + self.waypoint_index_embedding[0].index_select(0, current)
                + current_modality @ self.waypoint_modality_embedding.weight,
            ],
            dim=1,
        )
        mask = torch.cat([torch.zeros_like(padding).expand(-1, 3), padding], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=mask)
        readout = torch.cat([encoded[:, 0], encoded[:, 2]], dim=-1)
        if self._use_future:
            readout = self._future_update(readout, sequence, modality, valid, active)
        return self.actor_head(readout)


class _NormalizedActor(nn.Module):
    def __init__(self, actor: _NativeActor, obs_dim: int) -> None:
        super().__init__()
        self.actor = actor
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_var", torch.ones(obs_dim))
        self.register_buffer(
            "obs_normalize_mask", torch.ones(obs_dim, dtype=torch.bool)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = torch.where(
            self.obs_normalize_mask,
            (obs - self.obs_mean) / (self.obs_var.sqrt() + 1e-8),
            obs,
        )
        return torch.tanh(self.actor(normalized))


def _load_native_actor(
    path: Path,
) -> tuple[_NormalizedActor, _WaypointObservationLayout, dict[str, str]]:
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            f"Expected a tensor-only {_CHECKPOINT_FORMAT} deployment checkpoint."
        ) from exc
    if not isinstance(bundle, dict) or bundle.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError(f"Expected a {_CHECKPOINT_FORMAT} deployment checkpoint.")
    metadata = bundle.get("metadata")
    config = bundle.get("actor_config")
    state = bundle.get("state_dict")
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValueError("NMG checkpoint metadata must contain string keys and values.")
    try:
        layout = _WaypointObservationLayout(
            int(metadata["nmg.num_waypoints"]),
            int(metadata["nmg.num_controlled_joints"]),
            metadata["nmg.use_relative_obs"] == "true",
            metadata["nmg.canonicalize_quat_obs"] == "true",
            "nmg.joint_periodic_encoding" in metadata,
        )
    except (KeyError, ValueError) as exc:
        raise ValueError("NMG checkpoint has an invalid observation layout.") from exc
    if any(metadata.get(key) != value for key, value in layout.onnx_metadata.items()):
        raise ValueError("NMG checkpoint observation fingerprint or layout mismatch.")

    fields = {
        "hidden_dim",
        "nhead",
        "num_layers",
        "ff_dim",
        "future_context",
        "disable_future_residual",
    }
    if not isinstance(config, dict) or config.keys() != fields:
        raise ValueError(
            "NMG checkpoint must contain the versioned actor configuration."
        )
    if any(
        type(config[k]) is not int or config[k] <= 0
        for k in ("hidden_dim", "nhead", "num_layers")
    ):
        raise ValueError(
            "NMG actor dimensions and layer counts must be positive integers."
        )
    if config["hidden_dim"] % config["nhead"] != 0:
        raise ValueError(
            "NMG actor width must be divisible by the attention head count."
        )
    ff_dim = config["ff_dim"]
    if ff_dim is not None and (type(ff_dim) is not int or ff_dim < 0):
        raise ValueError(
            "NMG feed-forward width must be a nonnegative integer or None."
        )
    for name in ("future_context", "disable_future_residual"):
        if (
            type(config[name]) is not bool
            or metadata.get(f"nmg.{name}") != str(config[name]).lower()
        ):
            raise ValueError(f"NMG checkpoint {name} must match its metadata.")
    if config["disable_future_residual"] and not config["future_context"]:
        raise ValueError("NMG future residual control requires future context.")
    if not isinstance(state, dict):
        raise ValueError("NMG checkpoint must contain a model state dict.")
    for name, dtype in (
        ("obs_mean", torch.float32),
        ("obs_var", torch.float32),
        ("obs_normalize_mask", torch.bool),
    ):
        tensor = state.get(name)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != (layout.dim,)
            or tensor.dtype != dtype
        ):
            raise ValueError(f"NMG checkpoint has invalid {name} shape or dtype.")

    model = _NormalizedActor(_NativeActor(layout, **config), layout.dim)
    model.load_state_dict(state, strict=True)
    return model, layout, dict(metadata)
