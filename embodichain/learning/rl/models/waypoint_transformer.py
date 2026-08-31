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

"""Transformer models for ordered Cartesian and joint waypoint constraints."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = [
    "WaypointTransformerActor",
    "WaypointTransformerCritic",
    "WaypointTransformerEncoder",
    "parse_waypoint_observation",
    "waypoint_observation_dim",
    "waypoint_observation_normalize_mask",
]

_TOKEN_ACTION = 0
_TOKEN_STATE = 1
_TOKEN_ACTIVE_GOAL = 2
_TOKEN_WAYPOINT = 3
_NUM_TOKEN_TYPES = 4
_NUM_WAYPOINT_TYPES = 3


def _layer_init(
    layer: nn.Linear,
    std: float = math.sqrt(2.0),
    bias_const: float = 0.0,
) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def waypoint_observation_dim(
    num_waypoints: int,
    use_relative_observations: bool,
    *,
    joint_dim: int = 7,
) -> int:
    """Return the flat dimension of the unified waypoint observation layout.

    Args:
        num_waypoints: Maximum ordered waypoint count.
        use_relative_observations: Whether relative pose/joint fields are present.
        joint_dim: Controlled joint dimension.

    Returns:
        Required flat observation dimension.

    Raises:
        ValueError: If a dimension is not positive.
    """
    if num_waypoints <= 0:
        raise ValueError("num_waypoints must be positive.")
    if joint_dim <= 0:
        raise ValueError("joint_dim must be positive.")
    num_waypoints = int(num_waypoints)
    joint_dim = int(joint_dim)
    dimension = (
        joint_dim
        + 7
        + num_waypoints * (3 + 4 + joint_dim)
        + 5 * num_waypoints
        + joint_dim
        + num_waypoints
    )
    if use_relative_observations:
        dimension += 7 + num_waypoints * (3 + 4 + joint_dim)
    return dimension


def waypoint_observation_normalize_mask(
    num_waypoints: int,
    use_relative_observations: bool,
    *,
    joint_dim: int = 7,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a mask that preserves waypoint semantics during normalization.

    Args:
        num_waypoints: Maximum ordered waypoint count.
        use_relative_observations: Whether relative pose/joint fields are present.
        joint_dim: Controlled joint dimension.
        device: Optional output device.

    Returns:
        Boolean mask where ``True`` selects continuous normalized fields.
    """
    num_waypoints = int(num_waypoints)
    joint_dim = int(joint_dim)
    dimension = waypoint_observation_dim(
        num_waypoints,
        use_relative_observations,
        joint_dim=joint_dim,
    )
    mask = torch.ones(dimension, dtype=torch.bool, device=device)
    cursor = joint_dim + 7 + num_waypoints * (3 + 4 + joint_dim)
    mask[cursor : cursor + 5 * num_waypoints] = False
    cursor += 5 * num_waypoints + joint_dim
    if use_relative_observations:
        cursor += 7 + num_waypoints * (3 + 4)
        mask[cursor : cursor + num_waypoints * joint_dim] = False
        cursor += num_waypoints * joint_dim
    mask[cursor : cursor + num_waypoints] = False
    return mask


def parse_waypoint_observation(
    observation: torch.Tensor,
    num_waypoints: int,
    use_relative_observations: bool,
    *,
    joint_dim: int = 7,
) -> dict[str, torch.Tensor | None]:
    """Slice a flat unified waypoint observation into semantic fields.

    The layout is ``joint, eef, waypoint pose, waypoint joint, active, valid,
    position/rotation/joint masks, last action, optional relative fields, type``.

    Args:
        observation: Flat tensor shaped ``[batch, features]``.
        num_waypoints: Maximum ordered waypoint count.
        use_relative_observations: Whether relative fields are present.
        joint_dim: Controlled joint dimension.

    Returns:
        Mapping from semantic field names to tensor views.

    Raises:
        ValueError: If the observation rank or feature dimension is invalid.
    """
    if observation.ndim != 2:
        raise ValueError("observation must have shape [batch, features].")
    expected_dim = waypoint_observation_dim(
        num_waypoints,
        use_relative_observations,
        joint_dim=joint_dim,
    )
    if observation.shape[1] != expected_dim:
        raise ValueError(
            f"Expected waypoint observation dimension {expected_dim}, "
            f"got {observation.shape[1]}."
        )

    n = int(num_waypoints)
    d = int(joint_dim)
    cursor = 0
    joint = observation[:, cursor : cursor + d]
    cursor += d
    end_effector = observation[:, cursor : cursor + 7]
    cursor += 7
    waypoint_position = observation[:, cursor : cursor + 3 * n].reshape(-1, n, 3)
    cursor += 3 * n
    waypoint_quaternion = observation[:, cursor : cursor + 4 * n].reshape(-1, n, 4)
    cursor += 4 * n
    waypoint_joint = observation[:, cursor : cursor + d * n].reshape(-1, n, d)
    cursor += d * n
    active_onehot = observation[:, cursor : cursor + n]
    cursor += n
    valid_mask = observation[:, cursor : cursor + n]
    cursor += n
    position_mask = observation[:, cursor : cursor + n]
    cursor += n
    rotation_mask = observation[:, cursor : cursor + n]
    cursor += n
    joint_mask = observation[:, cursor : cursor + n]
    cursor += n
    last_action = observation[:, cursor : cursor + d]
    cursor += d

    active_relative_pose = None
    waypoint_relative_position = None
    waypoint_relative_quaternion = None
    waypoint_joint_error = None
    if use_relative_observations:
        active_relative_pose = observation[:, cursor : cursor + 7]
        cursor += 7
        waypoint_relative_position = observation[:, cursor : cursor + 3 * n].reshape(
            -1, n, 3
        )
        cursor += 3 * n
        waypoint_relative_quaternion = observation[:, cursor : cursor + 4 * n].reshape(
            -1, n, 4
        )
        cursor += 4 * n
        waypoint_joint_error = observation[:, cursor : cursor + d * n].reshape(-1, n, d)
        cursor += d * n
    waypoint_type = observation[:, cursor : cursor + n]

    active_id = active_onehot.argmax(dim=-1).long()
    valid_count = valid_mask.sum(dim=-1).long()
    active_id = torch.clamp(torch.minimum(active_id, valid_count - 1), min=0)
    return {
        "joint": joint,
        "end_effector": end_effector,
        "last_action": last_action,
        "active_relative_pose": active_relative_pose,
        "waypoint_relative_position": waypoint_relative_position,
        "waypoint_relative_quaternion": waypoint_relative_quaternion,
        "waypoint_joint_error": waypoint_joint_error,
        "active_onehot": active_onehot,
        "valid_mask": valid_mask,
        "position_mask": position_mask,
        "rotation_mask": rotation_mask,
        "joint_mask": joint_mask,
        "active_id": active_id,
        "waypoint_position": waypoint_position,
        "waypoint_quaternion": waypoint_quaternion,
        "waypoint_joint": waypoint_joint,
        "waypoint_type": waypoint_type,
    }


def _token_head(
    hidden_dim: int,
    output_dim: int,
    std: float,
    *,
    input_dim: int | None = None,
) -> nn.Sequential:
    input_dim = hidden_dim if input_dim is None else int(input_dim)
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        _layer_init(nn.Linear(input_dim, hidden_dim)),
        nn.Tanh(),
        _layer_init(nn.Linear(hidden_dim, output_dim), std=std),
    )


class WaypointTransformerEncoder(nn.Module):
    """Encode state, active-goal, and all waypoint tokens bidirectionally.

    Args:
        observation_dim: Flat unified observation dimension.
        num_waypoints: Maximum ordered waypoint count.
        joint_dim: Controlled joint dimension.
        use_relative_observations: Whether relative fields are present.
        hidden_dim: Transformer embedding dimension.
        num_attention_heads: Attention-head count.
        num_layers: Encoder-layer count.
        feedforward_dim: Optional feed-forward dimension; defaults to four
            times ``hidden_dim``.
    """

    def __init__(
        self,
        observation_dim: int,
        num_waypoints: int,
        *,
        joint_dim: int = 7,
        use_relative_observations: bool = True,
        hidden_dim: int = 256,
        num_attention_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_waypoints = int(num_waypoints)
        self.joint_dim = int(joint_dim)
        self.use_relative_observations = bool(use_relative_observations)
        expected_dim = waypoint_observation_dim(
            self.num_waypoints,
            self.use_relative_observations,
            joint_dim=self.joint_dim,
        )
        if int(observation_dim) != expected_dim:
            raise ValueError(
                f"WaypointTransformerEncoder expected observation_dim "
                f"{expected_dim}, got {observation_dim}."
            )
        if hidden_dim % num_attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_attention_heads.")

        self.state_dim = self.joint_dim + 7 + self.joint_dim
        self.active_goal_dim = 7 + self.joint_dim + 3 + 1
        if self.use_relative_observations:
            self.active_goal_dim += 7 + self.joint_dim
        self.waypoint_token_dim = 3 + 4 + self.joint_dim + 3 + 1 + 1
        if self.use_relative_observations:
            self.waypoint_token_dim += 7 + self.joint_dim

        self.action_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.action_token, std=0.02)
        self.token_type_embedding = nn.Embedding(_NUM_TOKEN_TYPES, hidden_dim)
        nn.init.normal_(self.token_type_embedding.weight, std=0.02)
        self.state_proj = _layer_init(nn.Linear(self.state_dim, hidden_dim))
        self.active_goal_proj = _layer_init(nn.Linear(self.active_goal_dim, hidden_dim))
        self.waypoint_proj = _layer_init(nn.Linear(self.waypoint_token_dim, hidden_dim))
        self.waypoint_index_embedding = nn.Parameter(
            torch.empty(1, self.num_waypoints, hidden_dim)
        )
        nn.init.normal_(self.waypoint_index_embedding, std=0.02)
        self.waypoint_modality_embedding = nn.Embedding(
            _NUM_WAYPOINT_TYPES,
            hidden_dim,
        )
        nn.init.normal_(self.waypoint_modality_embedding.weight, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_dim or hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def _type_embedding(
        self,
        token_type: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        indices = torch.full(
            (batch_size, 1),
            token_type,
            dtype=torch.long,
            device=device,
        )
        return self.token_type_embedding(indices)

    def encode_tokens(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode ``[ACTION, STATE, ACTIVE_GOAL, WP_1..WP_K]`` tokens.

        Args:
            observation: Unified flat waypoint observation batch.

        Returns:
            Encoded token tensor shaped ``[batch, 3 + K, hidden_dim]``.
        """
        fields = parse_waypoint_observation(
            observation,
            self.num_waypoints,
            self.use_relative_observations,
            joint_dim=self.joint_dim,
        )
        joint = fields["joint"]
        assert isinstance(joint, torch.Tensor)
        batch_size = observation.shape[0]
        device = observation.device
        action_tokens = self.action_token.expand(
            batch_size, -1, -1
        ) + self._type_embedding(
            _TOKEN_ACTION,
            batch_size,
            device,
        )
        state_features = torch.cat(
            [joint, fields["end_effector"], fields["last_action"]],
            dim=-1,
        )
        state_tokens = self.state_proj(state_features).unsqueeze(
            1
        ) + self._type_embedding(_TOKEN_STATE, batch_size, device)

        batch_indices = torch.arange(batch_size, device=device)
        active_id = fields["active_id"]
        assert isinstance(active_id, torch.Tensor)
        active_parts = [
            fields["waypoint_position"][batch_indices, active_id],
            fields["waypoint_quaternion"][batch_indices, active_id],
            fields["waypoint_joint"][batch_indices, active_id],
            fields["position_mask"][batch_indices, active_id].unsqueeze(-1),
            fields["rotation_mask"][batch_indices, active_id].unsqueeze(-1),
            fields["joint_mask"][batch_indices, active_id].unsqueeze(-1),
        ]
        if self.use_relative_observations:
            active_parts.extend(
                [
                    fields["active_relative_pose"],
                    fields["waypoint_joint_error"][batch_indices, active_id],
                ]
            )
        progress = active_id.float().unsqueeze(-1) / max(self.num_waypoints - 1, 1)
        active_parts.append(progress)
        active_goal = torch.cat(active_parts, dim=-1)
        active_goal_tokens = self.active_goal_proj(active_goal).unsqueeze(
            1
        ) + self._type_embedding(_TOKEN_ACTIVE_GOAL, batch_size, device)

        waypoint_features = [
            fields["waypoint_position"],
            fields["waypoint_quaternion"],
            fields["waypoint_joint"],
        ]
        if self.use_relative_observations:
            waypoint_features.extend(
                [
                    fields["waypoint_relative_position"],
                    fields["waypoint_relative_quaternion"],
                    fields["waypoint_joint_error"],
                ]
            )
        waypoint_features.extend(
            [
                fields["position_mask"].unsqueeze(-1),
                fields["rotation_mask"].unsqueeze(-1),
                fields["joint_mask"].unsqueeze(-1),
                fields["active_onehot"].unsqueeze(-1),
                fields["valid_mask"].unsqueeze(-1),
            ]
        )
        waypoint_token_types = self.token_type_embedding(
            torch.full(
                (batch_size, self.num_waypoints),
                _TOKEN_WAYPOINT,
                dtype=torch.long,
                device=device,
            )
        )
        waypoint_type = (
            fields["waypoint_type"]
            .long()
            .clamp(
                0,
                _NUM_WAYPOINT_TYPES - 1,
            )
        )
        waypoint_tokens = (
            self.waypoint_proj(torch.cat(waypoint_features, dim=-1))
            + waypoint_token_types
            + self.waypoint_index_embedding
            + self.waypoint_modality_embedding(waypoint_type)
        )
        tokens = torch.cat(
            [action_tokens, state_tokens, active_goal_tokens, waypoint_tokens],
            dim=1,
        )
        context_padding = torch.zeros(
            batch_size,
            3,
            dtype=torch.bool,
            device=device,
        )
        waypoint_padding = fields["valid_mask"] < 0.5
        padding_mask = torch.cat([context_padding, waypoint_padding], dim=1)
        return self.encoder(tokens, src_key_padding_mask=padding_mask)


class WaypointTransformerActor(WaypointTransformerEncoder):
    """Predict one joint action from fused action and active-goal tokens.

    Args:
        observation_dim: Flat unified observation dimension.
        action_dim: Output joint-action dimension.
        num_waypoints: Maximum ordered waypoint count.
        **kwargs: Encoder dimensions accepted by :class:`WaypointTransformerEncoder`.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        num_waypoints: int,
        **kwargs: object,
    ) -> None:
        super().__init__(
            observation_dim=observation_dim,
            num_waypoints=num_waypoints,
            **kwargs,
        )
        hidden_dim = int(kwargs.get("hidden_dim", 256))
        self.actor_head = _token_head(
            hidden_dim,
            action_dim,
            std=0.01,
            input_dim=2 * hidden_dim,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return deterministic mean actions.

        Args:
            observation: Unified flat waypoint observation batch.

        Returns:
            Mean action tensor shaped ``[batch, action_dim]``.
        """
        encoded = self.encode_tokens(observation)
        readout = torch.cat([encoded[:, 0], encoded[:, 2]], dim=-1)
        return self.actor_head(readout)


class WaypointTransformerCritic(WaypointTransformerEncoder):
    """Predict values from the full-context action token.

    Args:
        observation_dim: Flat unified observation dimension.
        num_waypoints: Maximum ordered waypoint count.
        **kwargs: Encoder dimensions accepted by :class:`WaypointTransformerEncoder`.
    """

    def __init__(
        self,
        observation_dim: int,
        num_waypoints: int,
        **kwargs: object,
    ) -> None:
        super().__init__(
            observation_dim=observation_dim,
            num_waypoints=num_waypoints,
            **kwargs,
        )
        hidden_dim = int(kwargs.get("hidden_dim", 256))
        self.value_head = _token_head(hidden_dim, 1, std=1.0)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return one value per observation.

        Args:
            observation: Unified flat waypoint observation batch.

        Returns:
            Value tensor shaped ``[batch, 1]``.
        """
        return self.value_head(self.encode_tokens(observation)[:, 0])
