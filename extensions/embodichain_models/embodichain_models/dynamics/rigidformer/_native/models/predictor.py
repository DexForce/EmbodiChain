# Derived from Zhiyang Dou's RigidFormer, revision 160fe0852c51cffae7b7440452b58d0b49db6df6.
# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Adaptations: package-relative imports; framework-owned public interfaces.
# The complete MIT license is in ../THIRD_PARTY_LICENSES.

"""
Prediction heads for the dynamics model: plain MLP, multi-scale, and
residual variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DynamicsPredictor(nn.Module):
    """
    MLP predictor that takes state features and query points to predict dynamics.
    """

    def __init__(
        self,
        state_dim: int = 512,
        query_dim: int = 3,  # 3D position
        hidden_dims: List[int] = [256, 128, 64],
        output_dim: int = 3,  # 3D acceleration/displacement
        activation: str = "relu",
        use_batch_norm: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.query_dim = query_dim
        self.output_dim = output_dim

        # Build MLP layers
        layers = []
        input_dim = state_dim + query_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if activation == "relu":
                layers.append(nn.SiLU())
            elif activation == "gelu":
                layers.append(nn.SiLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Tanh())  # bounded output
        self.output_scale = nn.Parameter(torch.ones(1) * 1.0)

        self.mlp = nn.Sequential(*layers)

    def forward(
        self, state_features: torch.Tensor, query_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict dynamics for query points given state features.

        Args:
            state_features: Enhanced state features (B, N_state, D_state) or (B, D_state)
            query_points: Query vertex positions (B, N_query, 3)

        Returns:
            Predicted accelerations/displacements (B, N_query, 3)
        """
        B = query_points.shape[0]
        N_query = query_points.shape[1]

        # If state_features has sequence dimension, pool it
        if state_features.dim() == 3:
            # Global max pooling over state tokens
            state_features = torch.max(state_features, dim=1)[0]  # (B, D_state)

        # Expand state features to match number of query points
        state_features_expanded = state_features.unsqueeze(1).expand(B, N_query, -1)

        # Concatenate state features with query points
        combined = torch.cat([state_features_expanded, query_points], dim=-1)

        # Reshape for batch processing through MLP
        combined_flat = combined.view(B * N_query, -1)

        # Apply MLP
        output_flat = self.mlp(combined_flat) * self.output_scale

        # Reshape back
        output = output_flat.view(B, N_query, self.output_dim)

        return output


class MultiScalePredictor(nn.Module):
    """
    Multi-scale predictor that processes features at different resolutions.
    """

    def __init__(
        self,
        state_dim: int = 512,
        query_dim: int = 3,
        scales: List[int] = [64, 128, 256],
        output_dim: int = 3,
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.scales = scales

        # Create predictors for each scale
        self.scale_predictors = nn.ModuleList(
            [
                DynamicsPredictor(
                    state_dim=state_dim,
                    query_dim=query_dim,
                    hidden_dims=[scale, scale // 2],
                    output_dim=output_dim,
                    activation=activation,
                    dropout=dropout,
                )
                for scale in scales
            ]
        )

        # Fusion layer
        self.fusion = nn.Linear(output_dim * len(scales), output_dim)

    def forward(
        self, state_features: torch.Tensor, query_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Multi-scale prediction with fusion.
        """
        predictions = []

        for predictor in self.scale_predictors:
            pred = predictor(state_features, query_points)
            predictions.append(pred)

        # Concatenate all scale predictions
        combined = torch.cat(predictions, dim=-1)

        # Fuse predictions
        output = self.fusion(combined)

        return output


class ResidualPredictor(nn.Module):
    """
    Predictor with residual connections for better gradient flow.
    """

    def __init__(
        self,
        state_dim: int = 512,
        query_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 3,
        output_dim: int = 3,
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_projection = nn.Linear(state_dim + query_dim, hidden_dim)

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, activation, dropout) for _ in range(num_layers)]
        )

        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, state_features: torch.Tensor, query_points: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with residual connections."""
        B = query_points.shape[0]
        N_query = query_points.shape[1]

        # Pool state features if needed
        if state_features.dim() == 3:
            state_features = torch.max(state_features, dim=1)[0]

        # Expand and concatenate
        state_expanded = state_features.unsqueeze(1).expand(B, N_query, -1)
        combined = torch.cat([state_expanded, query_points], dim=-1)

        # Initial projection
        x = self.input_projection(combined)

        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Output projection
        output = self.output_projection(x)

        return output


class ResidualBlock(nn.Module):
    """Single residual block."""

    def __init__(self, hidden_dim: int, activation: str = "relu", dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x

        x = self.norm1(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.norm2(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x + residual


if __name__ == "__main__":
    """Test predictor modules."""

    print("Testing DynamicsPredictor...")
    predictor = DynamicsPredictor(
        state_dim=512, query_dim=3, hidden_dims=[256, 128, 64], output_dim=3
    )

    # Test inputs
    batch_size = 4
    state_features = torch.randn(batch_size, 10, 512)  # With sequence dimension
    query_points = torch.randn(batch_size, 8, 3)  # 8 vertices

    output = predictor(state_features, query_points)
    print(f"Output shape: {output.shape}")  # Should be (4, 8, 3)

    print("\nTesting MultiScalePredictor...")
    multi_predictor = MultiScalePredictor(state_dim=512, query_dim=3)
    output_multi = multi_predictor(state_features, query_points)
    print(f"Multi-scale output shape: {output_multi.shape}")  # (4, 8, 3)

    print("\nTesting ResidualPredictor...")
    res_predictor = ResidualPredictor(state_dim=512, query_dim=3)
    output_res = res_predictor(state_features, query_points)
    print(f"Residual output shape: {output_res.shape}")  # (4, 8, 3)

    print("\nPredictor tests successful!")
