# Derived from Zhiyang Dou's RigidFormer, revision 160fe0852c51cffae7b7440452b58d0b49db6df6.
# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Adaptations: package-relative imports; framework-owned public interfaces.
# The complete MIT license is in ../THIRD_PARTY_LICENSES.

"""
State representation module inspired by CUT3R.
Maintains persistent state across time steps for multi-body dynamics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class StateRepresentation(nn.Module):
    """
    Persistent state representation for multi-body dynamics, following
    CUT3R's state management:
    - Register tokens stored in an nn.Embedding, as in CUT3R
    - Tokens laid out on a 3D grid whose coordinates drive RoPE
    - Rotary-only position encoding (no additive position embeddings)
    """

    def __init__(
        self,
        num_tokens: int = 216,  # 6x6x6 = 216 tokens on a 3D grid
        token_dim: int = 768,
        position_encoding: str = "learnable_3d",  # rope-only: use 3D grid positions
        rope_dim: Optional[int] = None,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.position_encoding = position_encoding

        # Register tokens (no additive position embeddings; rope-only)
        self.register_tokens = nn.Embedding(num_tokens, token_dim)

        # Small random initialization, following CUT3R
        nn.init.normal_(self.register_tokens.weight, mean=0.0, std=0.02)

        # 3D grid positions for RoPE (rope-only)
        if position_encoding != "learnable_3d":
            raise ValueError(
                "StateRepresentation now supports only 'learnable_3d' (rope-only)"
            )
        if num_tokens == 216:
            self.grid_3d = (6, 6, 6)
        else:
            self.grid_3d = self._compute_3d_grid(num_tokens)

        # Layer normalization
        self.norm = nn.LayerNorm(token_dim)

    def _compute_3d_grid(self, n):
        """Compute 3D grid dimensions for n tokens."""
        # Try to make roughly cubic
        cube_root = int(round(n ** (1 / 3)))
        for x in range(cube_root, 0, -1):
            if n % x == 0:
                remaining = n // x
                sqrt_rem = int(round(remaining**0.5))
                for y in range(sqrt_rem, 0, -1):
                    if remaining % y == 0:
                        z = remaining // y
                        return (x, y, z)
        return (n, 1, 1)

    def get_state_and_positions(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get initial state tokens and 3D grid positions for RoPE (rope-only).

        Returns:
            state_tokens: (B, num_tokens, token_dim)
            positions: (B, num_tokens, 3) in [-1, 1]
        """
        # Get state tokens using embedding
        indices = torch.arange(self.num_tokens, dtype=torch.long, device=device)
        state_tokens = self.register_tokens(indices)  # (num_tokens, token_dim)
        state_tokens = state_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Normalize tokens
        state_tokens = self.norm(state_tokens)

        # Build 3D grid positions normalized to [-1, 1]
        gx, gy, gz = self.grid_3d
        xs = torch.linspace(-1, 1, steps=gx, device=device, dtype=state_tokens.dtype)
        ys = torch.linspace(-1, 1, steps=gy, device=device, dtype=state_tokens.dtype)
        zs = torch.linspace(-1, 1, steps=gz, device=device, dtype=state_tokens.dtype)
        grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1).view(
            -1, 3
        )  # (num_tokens,3)
        if grid.shape[0] != self.num_tokens:
            # Tile or truncate to match the token count
            if grid.shape[0] < self.num_tokens:
                reps = (self.num_tokens + grid.shape[0] - 1) // grid.shape[0]
                grid = grid.repeat(reps, 1)[: self.num_tokens]
            else:
                grid = grid[: self.num_tokens]
        positions = grid.unsqueeze(0).expand(batch_size, -1, -1)  # (B,N,3)

        return state_tokens, positions

    def get_initial_state(self, batch_size: int) -> torch.Tensor:
        """
        Return the initial state tokens only (grid positions discarded).
        """
        device = self.register_tokens.weight.device
        state, _ = self.get_state_and_positions(batch_size, device)
        return state

    def forward(
        self, state: Optional[torch.Tensor] = None, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get state representation (rope-only).

        Args:
            state: Current state if available (B, num_tokens, token_dim)
            batch_size: Batch size if state is None

        Returns:
            State tokens (normalized)
        """
        if state is None:
            if batch_size is None:
                raise ValueError("Either state or batch_size must be provided")
            return self.get_initial_state(batch_size)

        # Normalize existing state
        state = self.norm(state)

        return state


class StateUpdateModule(nn.Module):
    """
    Module for updating state based on new observations.
    Implements the State-Update mechanism from CUT3R.
    """

    def __init__(
        self,
        state_dim: int = 512,
        observation_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.observation_dim = observation_dim

        # Project observations to state dimension if needed
        if observation_dim != state_dim:
            self.obs_projection = nn.Linear(observation_dim, state_dim)
        else:
            self.obs_projection = nn.Identity()

        # Multi-head attention for state update
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=state_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Gating mechanism for state update
        self.gate = nn.Sequential(nn.Linear(state_dim * 2, state_dim), nn.Sigmoid())

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(state_dim, state_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim * 4, state_dim),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(state_dim)
        self.norm2 = nn.LayerNorm(state_dim)

    def forward(self, state: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        """
        Update state based on new observation.

        Args:
            state: Current state (B, num_tokens, state_dim)
            observation: New observation features (B, obs_len, obs_dim)

        Returns:
            Updated state (B, num_tokens, state_dim)
        """
        # Project observation to state dimension
        obs_proj = self.obs_projection(observation)

        # Cross-attention: state attends to observation
        attn_output, _ = self.cross_attention(query=state, key=obs_proj, value=obs_proj)

        # Gated update mechanism
        gate_input = torch.cat([state, attn_output], dim=-1)
        gate = self.gate(gate_input)

        # Apply gated update
        state = state + gate * self.norm1(attn_output)

        # Feedforward network with residual connection
        state = state + self.ffn(self.norm2(state))

        return state


class StateReadoutModule(nn.Module):
    """
    Module for reading information from state.
    Implements the State-Readout mechanism from CUT3R.
    """

    def __init__(
        self,
        state_dim: int = 512,
        query_dim: int = 512,
        output_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.query_dim = query_dim
        self.output_dim = output_dim

        # Project query to state dimension if needed
        if query_dim != state_dim:
            self.query_projection = nn.Linear(query_dim, state_dim)
        else:
            self.query_projection = nn.Identity()

        # Multi-head attention for state readout
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=state_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(state_dim, output_dim), nn.LayerNorm(output_dim)
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(state_dim, state_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim * 4, state_dim),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(state_dim)
        self.norm2 = nn.LayerNorm(state_dim)

    def forward(self, state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Read information from state based on query.

        Args:
            state: Current state (B, num_tokens, state_dim)
            query: Query features (B, query_len, query_dim)

        Returns:
            Readout features (B, query_len, output_dim)
        """
        # Project query to state dimension
        query_proj = self.query_projection(query)

        # Cross-attention: query attends to state
        attn_output, attention_weights = self.cross_attention(
            query=query_proj, key=state, value=state
        )

        # Residual connection and normalization
        readout = query_proj + self.norm1(attn_output)

        # Feedforward network with residual connection
        readout = readout + self.ffn(self.norm2(readout))

        # Project to output dimension
        output = self.output_projection(readout)

        return output


class PersistentStateManager(nn.Module):
    """
    Complete state management system combining initialization,
    update, and readout mechanisms.
    """

    def __init__(
        self,
        num_tokens: int = 256,
        token_dim: int = 512,
        observation_dim: int = 512,
        output_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        learnable_init: bool = True,
    ):
        super().__init__()

        # State representation
        self.state_repr = StateRepresentation(
            num_tokens=num_tokens, token_dim=token_dim
        )

        # State update module
        self.state_update = StateUpdateModule(
            state_dim=token_dim,
            observation_dim=observation_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # State readout module
        self.state_readout = StateReadoutModule(
            state_dim=token_dim,
            query_dim=observation_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Current state storage (for sequential processing)
        self.register_buffer("current_state", None)

    def reset_state(self, batch_size: int):
        """Reset state to initial values."""
        self.current_state = self.state_repr.get_initial_state(batch_size)

    def forward(
        self,
        observation: torch.Tensor,
        query: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        return_updated_state: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process observation and update/read from state.

        Args:
            observation: New observation features (B, obs_len, obs_dim)
            query: Query for readout (B, query_len, obs_dim).
                   If None, use observation as query.
            state: External state if provided. If None, use internal state.
            return_updated_state: Whether to return the updated state

        Returns:
            readout: Features read from state (B, query_len, output_dim)
            updated_state: Updated state if return_updated_state=True
        """
        batch_size = observation.shape[0]

        # Get current state
        if state is not None:
            current_state = state
        elif self.current_state is not None:
            current_state = self.current_state
        else:
            current_state = self.state_repr.get_initial_state(batch_size)

        # Update state with new observation
        updated_state = self.state_update(current_state, observation)

        # Store updated state internally
        self.current_state = updated_state

        # Read from updated state
        if query is None:
            query = observation
        readout = self.state_readout(updated_state, query)

        if return_updated_state:
            return readout, updated_state
        else:
            return readout


if __name__ == "__main__":
    """Test the state management modules."""

    print("Testing StateRepresentation...")
    state_repr = StateRepresentation(num_tokens=256, token_dim=512)
    initial_state = state_repr.get_initial_state(batch_size=4)
    print(f"Initial state shape: {initial_state.shape}")  # (4, 256, 512)

    print("\nTesting StateUpdateModule...")
    state_update = StateUpdateModule(state_dim=512, observation_dim=256)
    observation = torch.randn(4, 10, 256)  # 10 observation tokens
    updated_state = state_update(initial_state, observation)
    print(f"Updated state shape: {updated_state.shape}")  # (4, 256, 512)

    print("\nTesting StateReadoutModule...")
    state_readout = StateReadoutModule(state_dim=512, query_dim=256, output_dim=512)
    query = torch.randn(4, 5, 256)  # 5 query tokens
    readout = state_readout(updated_state, query)
    print(f"Readout shape: {readout.shape}")  # (4, 5, 512)

    print("\nTesting PersistentStateManager...")
    manager = PersistentStateManager(
        num_tokens=256, token_dim=512, observation_dim=256, output_dim=512
    )

    # Process sequence
    for t in range(3):
        obs = torch.randn(4, 10, 256)
        readout, state = manager(obs, return_updated_state=True)
        print(f"Time {t}: Readout shape {readout.shape}, State shape {state.shape}")

    print("\nState management test successful!")
