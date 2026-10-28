# Derived from Zhiyang Dou's RigidFormer, revision 160fe0852c51cffae7b7440452b58d0b49db6df6.
# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Adaptations: package-relative imports; framework-owned public interfaces.
# The complete MIT license is in ../THIRD_PARTY_LICENSES.

"""
Bidirectional Transformer decoder inspired by CUT3R.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..attention import attention as flash_attn_func
from typing import Optional, Tuple, List
import math
import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


from embodichain_models.dynamics.rigidformer._native.models.physformer_rope import (
    PhysFormerRoPE,
    apply_rotary_emb,
)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    More efficient than LayerNorm, used in modern transformers like LLaMA.

    Reference: Zhang and Sennrich 2019 - "Root Mean Square Layer Normalization"
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization: x / RMS(x) * weight
        # RMS(x) = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return self.weight * x_normed


def get_norm_layer(norm_type: str, dim: int) -> nn.Module:
    """
    Factory function to create normalization layer.

    Args:
        norm_type: 'layer_norm' or 'rms_norm'
        dim: Dimension to normalize

    Returns:
        Normalization module
    """
    if norm_type == "rms_norm":
        return RMSNorm(dim)
    elif norm_type == "layer_norm":
        return nn.LayerNorm(dim)
    else:
        raise ValueError(
            f"Unknown norm_type: {norm_type}. Choose 'layer_norm' or 'rms_norm'."
        )


class BidirectionalDecoder(nn.Module):
    """
    Bidirectional decoder following CUT3R's architecture.

    - Returns all layer outputs for multi-scale fusion
    - Bidirectional cross-attention between state and object tokens
    - RMSNorm and QK-normalization for training stability (configurable)
    - Optional elementwise gated attention (Qwen3 style)
    """

    def __init__(
        self,
        state_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        feedforward_dim: int = None,  # None: 2.5x state_dim (SwiGLU)
        dropout: float = 0.1,
        activation: str = "swiglu",
        return_all_layers: bool = True,  # For multi-scale fusion
        use_learnable_pe: bool = False,
        use_rope: bool = True,
        max_state_tokens: int = 1024,
        max_obs_tokens: int = 256,
        use_self_attention: bool = False,  # Optional state self-attention
        norm_type: str = "rms_norm",  # 'layer_norm' or 'rms_norm'
        qk_norm: bool = True,  # QK-normalization for training stability
        num_register_tokens: int = 16,  # Learnable register tokens (RenderFormer)
        center_rope_positions: bool = False,  # Subtract the valid-object anchor centroid before RoPE (registers sit at the centroid)
        rope_set_aggregation: bool = False,  # Set aggregation for multi-point RoPE
        rope_dim: int = None,  # Custom RoPE dimension (None = auto-compute as head_dim//6*6)
        dim_per_coord_3d: int = None,  # Dim per coordinate for single-point (3D) RoPE
        dim_per_coord_multi: int = None,  # Dim per coordinate for multi-point RoPE
        layer_indices_to_save: set = None,  # Save only these layer outputs to reduce memory
        elementwise_gate: bool = False,  # Elementwise gated attention (Qwen3 style)
        state_update_enabled: bool = True,  # False: state is fixed learnable embeddings
        objects_self_attention: bool = False,  # Objects self-attention for object-object interaction
        use_se3: bool = False,  # SE(3) pairwise bias for object self-attention
        # Block Attention Residuals (Kimi 2026, arXiv:2603.15031)
        block_attn_res: bool = False,  # Enable Block AttnRes for inter-layer residuals
        block_attn_res_block_size: int = 4,  # Layers per block (S in the paper)
        block_attn_res_final_aggregate: bool = False,  # Return the final depth aggregation as the last layer's features
        film_conditioning: bool = False,  # Per-layer FiLM conditioning on step size
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers

        # Optionally save only selected layer outputs to reduce memory
        if layer_indices_to_save is not None:
            for idx in layer_indices_to_save:
                assert (
                    0 <= idx <= num_layers
                ), f"Invalid layer index {idx} to save. Valid range: [0, {num_layers}]"
            self.layer_indices_to_save = layer_indices_to_save
        else:
            # Default: save all layers if return_all_layers is True
            self.layer_indices_to_save = (
                set(range(num_layers + 1)) if return_all_layers else set()
            )
        self.use_rope = bool(use_rope)
        self.rope_set_aggregation = rope_set_aggregation
        # If RoPE is enabled, disable learnable positional embeddings in decoder inputs
        self.use_learnable_pe = bool(use_learnable_pe and (not self.use_rope))
        assert self.use_rope, "RoPE must be enabled for decoder"
        if feedforward_dim is None:
            feedforward_dim = int(2.5 * state_dim)

        # Register tokens (RenderFormer): learnable tokens that provide scratch
        # space for global information and reduce embedding noise.
        self.num_register_tokens = num_register_tokens
        self.center_rope_positions = bool(center_rope_positions)
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, num_register_tokens, state_dim) * 0.02
            )
            # Register positions are created in forward() to match the dimensionality
            # of obj_anchor_pos (3D single-point or Nq*3 D multi-point RoPE).
        else:
            self.register_tokens = None

        # Transformer decoder layers
        self.state_update_enabled = state_update_enabled
        self.objects_self_attention = objects_self_attention
        if not state_update_enabled:
            print(
                f"[Decoder] State update disabled: state acts as learnable embeddings (objects attend to state only)"
            )
        if objects_self_attention:
            print(
                f"[Decoder] Objects self-attention enabled: direct object-to-object interaction"
            )
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    state_dim,
                    num_heads,
                    feedforward_dim,
                    dropout,
                    activation,
                    use_self_attention=use_self_attention,
                    use_rope=self.use_rope,
                    norm_type=norm_type,
                    qk_norm=qk_norm,
                    rope_set_aggregation=self.rope_set_aggregation,
                    rope_dim=rope_dim,
                    dim_per_coord_3d=dim_per_coord_3d,
                    dim_per_coord_multi=dim_per_coord_multi,
                    elementwise_gate=elementwise_gate,
                    state_update_enabled=state_update_enabled,
                    objects_self_attention=objects_self_attention,
                    use_se3=use_se3,
                    film_conditioning=film_conditioning,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer normalization
        self.state_norm = get_norm_layer(norm_type, state_dim)
        self.obs_norm = get_norm_layer(norm_type, state_dim)

        # Block Attention Residuals (Kimi 2026, arXiv:2603.15031)
        # Replaces standard inter-layer residuals with softmax attention over block representations
        # h_l = sum_i alpha_{i->l} * v_i (attention over depth) instead of h_l = h_{l-1} + f_l(h_{l-1})
        self.block_attn_res = block_attn_res
        self.block_attn_res_block_size = block_attn_res_block_size
        # With return_all_layers the last returned features are the last layer's own output;
        # block_attn_res_final_aggregate returns the final depth aggregation instead (off by default).
        self.block_attn_res_final_aggregate = bool(block_attn_res_final_aggregate)
        if self.block_attn_res_final_aggregate and not block_attn_res:
            raise ValueError("block_attn_res_final_aggregate requires block_attn_res")
        if self.block_attn_res_final_aggregate:
            print(
                "[Decoder] Block AttnRes final aggregation returned as the last layer's features"
            )
        if block_attn_res:
            N_blocks = max(1, num_layers // block_attn_res_block_size)
            print(
                f"[Decoder] Block AttnRes enabled: {num_layers} layers, block_size={block_attn_res_block_size}, N_blocks={N_blocks}"
            )
            # Per-layer pseudo-query w_l (d-dimensional learnable vector) for depth attention
            self.attn_res_queries = nn.ParameterList(
                [nn.Parameter(torch.zeros(state_dim)) for _ in range(num_layers)]
            )
            # Shared RMSNorm for keys in depth attention (prevents magnitude bias)
            self.attn_res_key_norm = RMSNorm(state_dim)

    def forward(
        self,
        state: Optional[torch.Tensor],
        obj_features: torch.Tensor,
        state_pos: Optional[torch.Tensor] = None,
        obj_anchor_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        se3_bias: Optional[
            torch.Tensor
        ] = None,  # (B, H, Nobj, Nobj) pairwise SE(3) bias
        time_features: Optional[
            torch.Tensor
        ] = None,  # (B, 2) [s, s^2] for per-layer FiLM
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Process state and per-object features through bidirectional interaction.
        Register tokens (RenderFormer) are prepended to object features.

        When state=None (no_global_state mode):
        - Skips all state cross-attention
        - Runs objects self-attention + objects FFN only
        - Returns dummy state entries

        Args:
            state: State tokens (B, N_state, D) or None for no-state mode
            obj_features: Per-object encoder features (B, N_obj, D)
            state_pos: State positions for RoPE (B, N_state, 3) or None
            obj_anchor_pos: Object anchor positions for multi-point RoPE (B, N_obj, Nq*3)
            mask: Optional attention mask
            time_features: (B, 2) [s, s^2] for per-layer FiLM conditioning

        Returns:
            If return_all_layers: (all_state_layers, all_obj_layers)
            Else: (final_state, final_obj_features)
            Register tokens are stripped from obj_features before returning.
        """
        no_state = state is None

        # Store selected layer outputs for memory efficiency
        all_states = [] if self.return_all_layers else None
        all_obj_features = [] if self.return_all_layers else None

        # Rope-only: expect explicit obj_anchor_pos; state_pos only needed when state is used
        if obj_anchor_pos is None:
            raise ValueError(
                "decoder.forward expects obj_anchor_pos for rope-only mode"
            )
        if not no_state and state_pos is None:
            raise ValueError("decoder.forward expects state_pos when state is provided")

        if self.center_rope_positions:
            # Rotary angles of register tokens are zero, so register<->object attention would otherwise
            # depend on absolute object positions. Expressing every position relative to the centroid of
            # the valid objects' anchors makes the whole decoder exactly translation invariant.
            Bc, Nc, Pc = obj_anchor_pos.shape
            pos = obj_anchor_pos.view(Bc, Nc, Pc // 3, 3)
            if mask is not None:
                valid = (mask.reshape(Bc, -1)[:, -Nc:] == 0).to(
                    pos.dtype
                )  # (B, Nobj): additive mask 0 = valid
            else:
                valid = torch.ones(Bc, Nc, dtype=pos.dtype, device=pos.device)
            w = valid.view(Bc, Nc, 1, 1)
            centroid = (pos * w).sum(dim=(1, 2)) / (
                w.sum(dim=(1, 2)) * pos.shape[2]
            ).clamp_min(
                1.0
            )  # (B, 3)
            obj_anchor_pos = (pos - centroid.view(Bc, 1, 1, 3)).reshape(Bc, Nc, Pc)
            if not getattr(self, "_center_logged", False):
                self._center_logged = True
                print(
                    f"[Decoder] RoPE positions centred at the valid-object anchor centroid ACTIVE "
                    f"(|centroid| max={centroid.abs().max().item():.2f} m; registers at the centroid)",
                    flush=True,
                )
        # Prepend register tokens to object features (RenderFormer)
        if self.num_register_tokens > 0:
            B = obj_features.shape[0]
            register_tokens_expanded = self.register_tokens.expand(
                B, -1, -1
            )  # (B, N_reg, D)
            obj_features_with_reg = torch.cat(
                [register_tokens_expanded, obj_features], dim=1
            )

            # Register positions match the dimensionality of obj_anchor_pos (3 or Nq*3)
            pos_dim = obj_anchor_pos.shape[-1]  # Either 3 or Nq*3
            register_positions = torch.zeros(
                B,
                self.num_register_tokens,
                pos_dim,
                device=obj_anchor_pos.device,
                dtype=obj_anchor_pos.dtype,
            )
            obj_anchor_pos_with_reg = torch.cat(
                [register_positions, obj_anchor_pos], dim=1
            )

            # Extend mask to cover register tokens (always valid)
            if mask is not None:
                # mask: (B, 1, 1, Nobj) SDPA additive -> (B, 1, 1, Nreg+Nobj);
                # register entries are 0.0 (attend normally)
                register_mask = torch.zeros(
                    B,
                    1,
                    1,
                    self.num_register_tokens,
                    device=mask.device,
                    dtype=mask.dtype,
                )
                mask_with_reg = torch.cat([register_mask, mask], dim=-1)
            else:
                mask_with_reg = None

            # Extend SE(3) bias to include register tokens (zero bias = attend freely)
            if se3_bias is not None and self.num_register_tokens > 0:
                N_reg = self.num_register_tokens
                N_obj = se3_bias.shape[-1]  # original Nobj
                H = se3_bias.shape[1]
                # Pad: (B, H, Nobj, Nobj) -> (B, H, Nreg+Nobj, Nreg+Nobj)
                # Register tokens have zero bias (vanilla attention)
                se3_bias = F.pad(se3_bias, (N_reg, 0, N_reg, 0), value=0.0)
        else:
            obj_features_with_reg = obj_features
            obj_anchor_pos_with_reg = obj_anchor_pos
            mask_with_reg = mask

        # Process through each layer with register tokens
        state_current = state
        obj_current = obj_features_with_reg

        # Create a dummy state placeholder for no-state mode (used in returned lists)
        if no_state:
            B = obj_features.shape[0]
            _dummy_state = torch.zeros(
                B,
                1,
                self.state_dim,
                device=obj_features.device,
                dtype=obj_features.dtype,
            )

        # Add initial layer (layer 0) if needed
        if self.return_all_layers and 0 in self.layer_indices_to_save:
            all_states.append(_dummy_state if no_state else state)
            all_obj_features.append(
                obj_features
            )  # Store original without register tokens

        # Block Attention Residuals path
        if self.block_attn_res:
            # Block AttnRes (Kimi 2026): replace inter-layer residuals with
            # softmax attention over completed block sums + partial block sum
            S = self.block_attn_res_block_size
            completed_blocks = [obj_current]  # b_0 = initial obj features (embedding)
            partial_block = None  # Running intra-block sum
            layer_in_block = 0

            for layer_idx, layer in enumerate(self.layers):
                # Block boundary: commit partial sum
                if layer_in_block == S:
                    completed_blocks.append(partial_block)
                    partial_block = None
                    layer_in_block = 0

                # Compute input via depth attention over block representations
                obj_current = self._block_attn_res_aggregate(
                    completed_blocks, partial_block, self.attn_res_queries[layer_idx]
                )

                # Run layer (layer uses internal residuals for sublayers)
                state_current, obj_output = layer(
                    state_current,
                    obj_current,
                    state_pos,
                    obj_anchor_pos_with_reg,
                    mask_with_reg,
                    se3_bias=se3_bias,
                    time_features=time_features,
                )

                # Compute layer delta: f_l(h_l) = output - input
                # The layer output includes internal residuals, so
                # delta = obj_output - obj_current is the "function output" for AttnRes
                layer_delta = obj_output - obj_current

                # Accumulate delta into partial block sum
                partial_block = (
                    layer_delta
                    if partial_block is None
                    else partial_block + layer_delta
                )
                layer_in_block += 1

                # Save layer outputs if needed
                if (
                    self.return_all_layers
                    and (layer_idx + 1) in self.layer_indices_to_save
                ):
                    # For output, use the full obj_output (input + delta)
                    if self.num_register_tokens > 0:
                        obj_out_no_reg = obj_output[:, self.num_register_tokens :, :]
                    else:
                        obj_out_no_reg = obj_output
                    all_states.append(_dummy_state if no_state else state_current)
                    all_obj_features.append(obj_out_no_reg)

            # Final aggregation: attend over all blocks (including last partial)
            if partial_block is not None:
                completed_blocks.append(partial_block)
            # Reuse the last layer's query for the final aggregation
            obj_current = self._block_attn_res_aggregate(
                completed_blocks, None, self.attn_res_queries[-1]
            )
            if self.return_all_layers and self.block_attn_res_final_aggregate:
                if self.num_layers not in self.layer_indices_to_save:
                    raise ValueError(
                        "block_attn_res_final_aggregate needs the last layer among the saved layers"
                    )
                all_obj_features[-1] = (
                    obj_current[:, self.num_register_tokens :, :]
                    if self.num_register_tokens > 0
                    else obj_current
                )

        else:
            # Standard residual path
            for layer_idx, layer in enumerate(self.layers):
                state_current, obj_current = layer(
                    state_current,
                    obj_current,
                    state_pos,
                    obj_anchor_pos_with_reg,
                    mask_with_reg,
                    se3_bias=se3_bias,
                    time_features=time_features,
                )

                if (
                    self.return_all_layers
                    and (layer_idx + 1) in self.layer_indices_to_save
                ):
                    if self.num_register_tokens > 0:
                        obj_current_no_reg = obj_current[
                            :, self.num_register_tokens :, :
                        ]
                    else:
                        obj_current_no_reg = obj_current
                    all_states.append(_dummy_state if no_state else state_current)
                    all_obj_features.append(obj_current_no_reg)

        # Apply final normalization
        if self.return_all_layers:
            if not no_state:
                all_states[-1] = self.state_norm(all_states[-1])
            all_obj_features[-1] = self.obs_norm(all_obj_features[-1])
            return all_states, all_obj_features
        else:
            if self.num_register_tokens > 0:
                obj_current_no_reg = obj_current[:, self.num_register_tokens :, :]
            else:
                obj_current_no_reg = obj_current
            state_final = (
                self.state_norm(state_current) if not no_state else _dummy_state
            )
            obj_final = self.obs_norm(obj_current_no_reg)
            return state_final, obj_final

    def _block_attn_res_aggregate(
        self,
        completed_blocks: list,
        partial_block: Optional[torch.Tensor],
        w_l: nn.Parameter,
    ) -> torch.Tensor:
        """
        Block AttnRes aggregation (Eq. 6 from Kimi 2026).

        Computes softmax attention over completed block representations + optional
        partial block sum, using per-layer pseudo-query w_l.

        phi(q, k) = exp(q^T RMSNorm(k))
        alpha_{i->l} = phi(w_l, k_i) / sum_j phi(w_l, k_j)
        h_l = sum_i alpha_{i->l} * v_i

        Args:
            completed_blocks: List of block representations [b_0, b_1, ...]
            partial_block: Current intra-block partial sum (None if at block boundary)
            w_l: Learnable pseudo-query vector (d,)

        Returns:
            Aggregated input for next layer (B, N_tokens, D)
        """
        if partial_block is not None:
            sources = completed_blocks + [partial_block]
        else:
            sources = completed_blocks

        # Stack: (num_sources, B, N_tokens, D)
        V = torch.stack(sources, dim=0)
        # RMSNorm on keys prevents magnitude bias between complete blocks and partial sums
        K = self.attn_res_key_norm(V)
        # Compute attention logits: w_l^T RMSNorm(k_i) for each source
        # w_l: (D,), K: (num_sources, B, N_tokens, D) -> logits: (num_sources, B, N_tokens)
        logits = torch.einsum("d, n b t d -> n b t", w_l, K)
        alpha = logits.softmax(dim=0)
        # Weighted sum: (num_sources, B, N_tokens) x (num_sources, B, N_tokens, D) -> (B, N_tokens, D)
        return torch.einsum("n b t, n b t d -> b t d", alpha, V)


class TransformerDecoderLayer(nn.Module):
    """
    Single decoder layer of bidirectional cross-attention, following CUT3R and
    RenderFormer (RMSNorm, QK-normalization, optional gated attention).

    When state_update_enabled=False:
    - State is treated as fixed learnable embeddings (never updated)
    - Only the Objects <- State cross-attention runs
    - State self-attention and State <- Objects are skipped, saving roughly
      half the computation

    When objects_self_attention=True:
    - Objects self-attention runs before Objects <- State, enabling direct
      object-to-object interaction
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
        activation: str = "swiglu",
        use_self_attention: bool = False,
        drop_path_prob: float = 0.1,
        use_rope: bool = True,
        norm_type: str = "rms_norm",  # 'layer_norm' or 'rms_norm'
        qk_norm: bool = True,
        rope_set_aggregation: bool = False,  # Set aggregation for multi-point RoPE
        rope_dim: int = None,  # Custom RoPE dimension
        dim_per_coord_3d: int = None,  # Dim per coordinate for single-point (3D) RoPE
        dim_per_coord_multi: int = None,  # Dim per coordinate for multi-point RoPE
        elementwise_gate: bool = False,  # Elementwise gated attention (Qwen3 style)
        state_update_enabled: bool = True,  # False: state is fixed learnable embeddings
        objects_self_attention: bool = False,  # Objects self-attention for object-object interaction
        use_se3: bool = False,  # SE(3) pairwise bias replaces RoPE in object self-attention
        film_conditioning: bool = False,  # Per-layer FiLM conditioning on step size
    ):
        super().__init__()

        self.use_self_attention = use_self_attention
        self.use_rope = bool(use_rope)
        self.state_update_enabled = state_update_enabled
        self.objects_self_attention = objects_self_attention
        self._objects_self_attention_logged = False  # One-time log flag

        # Cross-attention with optional RoPE and QK-normalization
        self.state_cross_attn = CrossAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=self.use_rope,
            qk_norm=qk_norm,
            rope_set_aggregation=rope_set_aggregation,
            rope_dim=rope_dim,
            dim_per_coord_3d=dim_per_coord_3d,
            dim_per_coord_multi=dim_per_coord_multi,
            elementwise_gate=elementwise_gate,
        )
        self.obs_cross_attn = CrossAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=self.use_rope,
            qk_norm=qk_norm,
            rope_set_aggregation=rope_set_aggregation,
            rope_dim=rope_dim,
            dim_per_coord_3d=dim_per_coord_3d,
            dim_per_coord_multi=dim_per_coord_multi,
            elementwise_gate=elementwise_gate,
        )

        # Optional state self-attention (disabled by default)
        if use_self_attention:
            self.state_self_attention = SelfAttentionWithRoPE(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                use_rope=self.use_rope,
                qk_norm=qk_norm,
                rope_set_aggregation=rope_set_aggregation,
                rope_dim=rope_dim,
                dim_per_coord_3d=dim_per_coord_3d,
                dim_per_coord_multi=dim_per_coord_multi,
                elementwise_gate=elementwise_gate,
            )
            self.state_norm0 = get_norm_layer(
                norm_type, d_model
            )  # Norm for state self-attention

        # Objects self-attention for object-to-object interaction. Spatial encoding
        # uses multi-point RoPE over anchor positions, or an SE(3) pairwise bias
        # when use_se3=True.
        if objects_self_attention:
            if use_se3:
                from embodichain_models.dynamics.rigidformer._native.models.se3_attention import (
                    SelfAttentionWithSE3Bias,
                )

                self.obj_self_attn = SelfAttentionWithSE3Bias(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    elementwise_gate=elementwise_gate,
                )
                logger.info(
                    f"[Decoder Init] obj_self_attn = SelfAttentionWithSE3Bias (no RoPE, pairwise bias)"
                )
            else:
                self.obj_self_attn = SelfAttentionWithRoPE(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_rope=self.use_rope,
                    qk_norm=qk_norm,
                    rope_set_aggregation=rope_set_aggregation,
                    rope_dim=rope_dim,
                    dim_per_coord_3d=dim_per_coord_3d,
                    dim_per_coord_multi=dim_per_coord_multi,
                    elementwise_gate=elementwise_gate,
                )
            self.obj_norm0 = get_norm_layer(
                norm_type, d_model
            )  # Norm for objects self-attention

        # Feedforward networks
        self.state_ffn = FeedForward(d_model, feedforward_dim, dropout, activation)
        self.obs_ffn = FeedForward(d_model, feedforward_dim, dropout, activation)

        # Layer normalization (pre-norm)
        self.state_norm1 = get_norm_layer(norm_type, d_model)
        self.state_norm2 = get_norm_layer(norm_type, d_model)
        self.obs_norm1 = get_norm_layer(norm_type, d_model)
        self.obs_norm2 = get_norm_layer(norm_type, d_model)

        # Per-layer FiLM conditioning on step size
        self.layer_film = None
        if film_conditioning:
            from embodichain_models.dynamics.rigidformer._native.models.encoder import (
                StepSizeFiLM,
            )

            self.layer_film = StepSizeFiLM(time_dim=2, feature_dim=d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.drop_path = (
            DropPath(drop_path_prob)
            if drop_path_prob and drop_path_prob > 0.0
            else nn.Identity()
        )

    def forward(
        self,
        state: torch.Tensor,
        obj_features: torch.Tensor,
        state_pos: Optional[torch.Tensor] = None,
        obj_anchor_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        se3_bias: Optional[
            torch.Tensor
        ] = None,  # (B, H, Nobj, Nobj) pairwise SE(3) bias
        time_features: Optional[
            torch.Tensor
        ] = None,  # (B, 2) [s, s^2] for per-layer FiLM
    ) -> tuple:
        """
        Bidirectional or unidirectional cross-attention.

        When state_update_enabled=True (default):
            - Full bidirectional interaction: state and objects attend to each other
            - Optional state self-attention
            - Both state and objects are updated

        When state_update_enabled=False:
            - Unidirectional: Objects <- State only
            - State remains fixed learnable embeddings

        Args:
            state: State tokens (B, N_state, D)
            obj_features: Per-object encoder features (B, N_obj, D)
            state_pos: State 3D positions for RoPE (B, N_state, 3)
            obj_anchor_pos: Object anchor positions for RoPE (B, N_obj, Nq*3)
            mask: Optional attention mask

        Returns:
            Updated state and obj_features
        """
        no_state = state is None

        if self.state_update_enabled and not no_state:
            # Full bidirectional mode
            # Optional state self-attention with RoPE. The object mask does not
            # apply to state tokens; pass None.
            if self.use_self_attention:
                state_sa = self.state_norm0(state)
                state_sa = self.state_self_attention(
                    query=state_sa,
                    key_value=state_sa,
                    pos_q=state_pos,  # Use 3D grid positions for RoPE
                    pos_k=state_pos,
                    attn_mask=None,  # Object mask does not apply to state tokens
                )
                state = state + self.drop_path(self.dropout(state_sa))

            # Bidirectional cross-attention
            # State <- Objects: state queries per-object features
            state_normed = self.state_norm1(state)
            obj_normed = self.obs_norm1(obj_features)
            state2 = self.state_cross_attn(
                query=state_normed,
                key_value=obj_normed,
                pos_q=state_pos,
                pos_k=obj_anchor_pos,
                attn_mask=mask,
            )
            state = state + self.drop_path(self.dropout(state2))

            # Objects <- State: per-object features query state
            obj2 = self.obs_cross_attn(
                query=obj_normed,
                key_value=state_normed,
                pos_q=obj_anchor_pos,
                pos_k=state_pos,
                attn_mask=None,  # State keys are all valid, no need to mask
            )
            obj_features = obj_features + self.drop_path(self.dropout(obj2))

            # Per-layer FiLM conditioning on step size
            if self.layer_film is not None and time_features is not None:
                obj_features = self.layer_film(obj_features, time_features)

            # Feedforward networks with residual connections
            state = state + self.drop_path(self.state_ffn(self.state_norm2(state)))
            obj_features = obj_features + self.drop_path(
                self.obs_ffn(self.obs_norm2(obj_features))
            )

        else:
            # Unidirectional / no-state mode
            # no_state=True: objects self-attention + objects FFN only
            # no_state=False: objects self-attention + Objects <- State + objects FFN

            # Objects self-attention runs before Objects <- State so objects
            # exchange information first
            if self.objects_self_attention:
                # One-time runtime confirmation log
                if not self._objects_self_attention_logged:
                    mode_str = "NO-STATE" if no_state else "UNIDIRECTIONAL"
                    print(
                        f"[Decoder Runtime] Objects self-attention active ({mode_str}): shape={obj_features.shape}, se3_bias={'yes' if se3_bias is not None else 'no'}"
                    )
                    self._objects_self_attention_logged = True
                obj_sa = self.obj_norm0(obj_features)
                if se3_bias is not None:
                    # SE(3) path: use pairwise bias instead of RoPE
                    obj_sa = self.obj_self_attn(
                        query=obj_sa,
                        key_value=obj_sa,
                        se3_bias=se3_bias,
                        attn_mask=mask,
                    )
                else:
                    # ARoPE/OBB/PCA path: use RoPE with anchor positions
                    obj_sa = self.obj_self_attn(
                        query=obj_sa,
                        key_value=obj_sa,
                        pos_q=obj_anchor_pos,  # Multi-point anchor positions for RoPE
                        pos_k=obj_anchor_pos,
                        attn_mask=mask,  # Mask padding objects
                    )
                obj_features = obj_features + self.drop_path(self.dropout(obj_sa))

            # Objects <- State cross-attention (skipped when no_state=True)
            if not no_state:
                state_normed = self.state_norm1(state)
                obj_normed = self.obs_norm1(obj_features)

                # Objects <- State: per-object features query state (learnable context)
                obj2 = self.obs_cross_attn(
                    query=obj_normed,
                    key_value=state_normed,
                    pos_q=obj_anchor_pos,
                    pos_k=state_pos,
                    attn_mask=None,  # State keys are all valid, no need to mask
                )
                obj_features = obj_features + self.drop_path(self.dropout(obj2))

            # Per-layer FiLM conditioning on step size
            # Applied after attention, before the FFN. Padding objects may receive
            # non-zero features from the FiLM beta bias; the SDPA -inf mask keeps
            # valid objects from attending to them.
            if self.layer_film is not None and time_features is not None:
                obj_features = self.layer_film(obj_features, time_features)

            # Objects FFN
            obj_features = obj_features + self.drop_path(
                self.obs_ffn(self.obs_norm2(obj_features))
            )

        return state, obj_features


class SelfAttentionWithRoPE(nn.Module):
    """
    Self-attention that applies RoPE using 3D positions.

    QK-normalization (RenderFormer) improves training stability; elementwise
    gated attention (Qwen3) adds non-linearity and sparsity.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        qk_norm: bool = True,  # QK-normalization (RenderFormer)
        rope_set_aggregation: bool = False,
        rope_dim: int = None,  # Custom RoPE dimension (None = auto-compute)
        dim_per_coord_3d: int = None,  # Dim per coordinate for single-point (3D) RoPE
        dim_per_coord_multi: int = None,  # Dim per coordinate for multi-point RoPE
        elementwise_gate: bool = False,  # Elementwise gated attention (Qwen3 style)
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.use_rope = bool(use_rope)
        self.qk_norm = qk_norm
        self.elementwise_gate = elementwise_gate

        # Projections; with elementwise_gate, q_proj outputs 2x dims (Q and gate scores)
        if self.elementwise_gate:
            self.q_proj = nn.Linear(d_model, d_model * 2, bias=True)
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self._attn_dispatch_logged = False  # One-time attention dispatch log

        # QK-normalization: per-head RMSNorm (RenderFormer)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # 3D rotary embedding; only the first rope_dim dims are rotated
        if rope_dim is None:
            # Auto-compute: multiple of 6, capped at head_dim
            self.rope_dim = max(6, (self.head_dim // 6) * 6)
            self.rope_dim = min(self.rope_dim, self.head_dim)
        else:
            # Custom rope_dim (must be a multiple of 6 for 3D RoPE)
            assert (
                rope_dim % 6 == 0
            ), f"rope_dim must be multiple of 6 for 3D RoPE, got {rope_dim}"
            assert (
                rope_dim <= self.head_dim
            ), f"rope_dim {rope_dim} exceeds head_dim {self.head_dim}"
            self.rope_dim = rope_dim

        self.rope3d = PhysFormerRoPE(
            dim=self.rope_dim,
            max_freq=10.0,
            use_3d=True,
            learnable_freqs=False,
            use_set_aggregation=rope_set_aggregation,
            dim_per_coord_3d=dim_per_coord_3d,  # Pass to PhysFormerRoPE
            dim_per_coord_multi=dim_per_coord_multi,  # Pass to PhysFormerRoPE
        )

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (B,H,N,Hd)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, Hd = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, N, H * Hd)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        pos_q: torch.Tensor,
        pos_k: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Self-attention using Flash Attention, or SDPA when a mask is provided.

        Args:
            query: Input tokens (B, N, D)
            key_value: Same as query for self-attention (B, N, D)
            pos_q: 3D positions (B, N, 3) or multi-point positions (B, N, Nq*3)
            pos_k: Same as pos_q for self-attention
            attn_mask: Optional additive mask; when present, attention runs
                through SDPA since Flash does not support masks

        Returns:
            Output tokens (B, N, D)
        """
        B, N, _ = query.shape

        # Q projection with optional elementwise gate
        if self.elementwise_gate:
            q_gate = self.q_proj(query)  # (B, N, d_model * 2)
            q_states, gate_score = torch.split(q_gate, self.d_model, dim=-1)
            q = q_states.view(
                B, N, self.num_heads, self.head_dim
            )  # (B, N, H, D) - Flash expects (B, S, H, D)
            gate_score = gate_score.view(
                B, N, self.num_heads, self.head_dim
            )  # (B, N, H, D)
        else:
            q = self.q_proj(query).view(
                B, N, self.num_heads, self.head_dim
            )  # (B, N, H, D)
            gate_score = None

        k = self.k_proj(key_value).view(
            B, N, self.num_heads, self.head_dim
        )  # (B, N, H, D)
        v = self.v_proj(key_value).view(
            B, N, self.num_heads, self.head_dim
        )  # (B, N, H, D)

        # QK-normalization (RenderFormer)
        if self.qk_norm:
            # RMSNorm normalizes the last (per-head) dimension
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE (single-point 3D or multi-point)
        if pos_q is None or pos_k is None:
            raise ValueError("SelfAttentionWithRoPE expects pos_q and pos_k")

        # Single-point (3D) vs multi-point positions
        if pos_q.shape[-1] == 3:
            rope_q = self.rope3d.forward_3d(pos_q)  # (B, N, rope_dim)
        else:
            rope_q = self.rope3d.forward_sa_rope(
                pos_q
            )  # SA-RoPE: symmetric anchor pooling

        if pos_k is pos_q:
            rope_k = rope_q  # identical positions -> identical angles (computed once)
        elif pos_k.shape[-1] == 3:
            rope_k = self.rope3d.forward_3d(pos_k)  # (B, N, rope_dim)
        else:
            rope_k = self.rope3d.forward_sa_rope(
                pos_k
            )  # SA-RoPE: symmetric anchor pooling

        # The angles stay at (B, 1, N, rope_dim) and broadcast over heads inside apply_rotary_emb. cos/sin
        # are elementwise, so this equals expanding to (B, H, N, rope_dim) first, at 1/num_heads of the work.
        rope_q_T = rope_q.unsqueeze(1)
        rope_k_T = rope_k.unsqueeze(1) if rope_k is not rope_q else rope_q_T

        # Apply rotation
        q_rot = q[..., : self.rope_dim]
        k_rot = k[..., : self.rope_dim]
        q_pass = q[..., self.rope_dim :]
        k_pass = k[..., self.rope_dim :]

        # apply_rotary_emb operates on (B, H, N, D): transpose for the rotation,
        # then transpose back to the (B, N, H, D) layout Flash expects.

        q_rot = q_rot.transpose(1, 2)  # (B, H, N, D)
        k_rot = k_rot.transpose(1, 2)

        q_rot = apply_rotary_emb(rope_q_T, q_rot)
        k_rot = apply_rotary_emb(rope_k_T, k_rot)

        # Permute back to (B, N, H, D)
        q_rot = q_rot.transpose(1, 2)
        k_rot = k_rot.transpose(1, 2)

        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)

        # flash_attn_func does not support additive masks: dispatch to SDPA when a
        # mask is provided, Flash Attention otherwise.
        if attn_mask is not None:
            if not self._attn_dispatch_logged:
                print(
                    f"[SelfAttentionWithRoPE] SDPA path (mask active): Q={q.shape}, K={k.shape}, mask={attn_mask.shape}, "
                    f"mask_min={attn_mask.min().item():.0f}, n_masked={(attn_mask == float('-inf')).sum().item()}"
                )
                self._attn_dispatch_logged = True
            # SDPA expects (B, H, N, D) layout
            q_sdpa = q.transpose(1, 2)  # (B, H, N, D)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=self.scale,
            )  # (B, H, N, D)
            out = out.transpose(1, 2)  # Back to (B, N, H, D)
        else:
            if not self._attn_dispatch_logged:
                print(f"[SelfAttentionWithRoPE] Flash path (no mask): Q={q.shape}")
                self._attn_dispatch_logged = True
            # Flash Attention runs in bf16/fp16; the output is cast back to the
            # original dtype immediately.
            original_dtype = query.dtype
            attn_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            q_half = q.to(attn_dtype)
            k_half = k.to(attn_dtype)
            v_half = v.to(attn_dtype)
            out = flash_attn_func(
                q_half,
                k_half,
                v_half,
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=self.scale,
                backend=getattr(self, "attention_backend", "sdpa"),
            )
            out = out.to(original_dtype)

        if self.elementwise_gate:
            out = out * torch.sigmoid(gate_score)

        # Merge heads: (B, N, H, D) -> (B, N, H*D)
        out = out.reshape(B, N, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        return out


class CrossAttentionWithRoPE(nn.Module):
    """
    Cross-attention that applies RoPE to queries and keys.

    QK-normalization (RenderFormer) improves training stability; elementwise
    gated attention (Qwen3) adds non-linearity and sparsity.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        qk_norm: bool = True,  # QK-normalization (RenderFormer)
        rope_set_aggregation: bool = False,
        rope_dim: int = None,  # Custom RoPE dimension (None = auto-compute)
        dim_per_coord_3d: int = None,  # Dim per coordinate for single-point (3D) RoPE
        dim_per_coord_multi: int = None,  # Dim per coordinate for multi-point RoPE
        elementwise_gate: bool = False,  # Elementwise gated attention (Qwen3 style)
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.use_rope = bool(use_rope)
        self.qk_norm = qk_norm
        self.elementwise_gate = elementwise_gate

        # Projections; with elementwise_gate, q_proj outputs 2x dims (Q and gate scores)
        if self.elementwise_gate:
            self.q_proj = nn.Linear(d_model, d_model * 2, bias=True)
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self._attn_dispatch_logged = False  # One-time attention dispatch log

        # QK-normalization: per-head RMSNorm (RenderFormer)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # 3D rotary embedding; only the first rope_dim dims are rotated
        if rope_dim is None:
            # Auto-compute: multiple of 6, capped at head_dim
            self.rope_dim = max(6, (self.head_dim // 6) * 6)
            self.rope_dim = min(self.rope_dim, self.head_dim)
        else:
            # Custom rope_dim (must be a multiple of 6 for 3D RoPE)
            assert (
                rope_dim % 6 == 0
            ), f"rope_dim must be multiple of 6 for 3D RoPE, got {rope_dim}"
            assert (
                rope_dim <= self.head_dim
            ), f"rope_dim {rope_dim} exceeds head_dim {self.head_dim}"
            self.rope_dim = rope_dim

        self.rope3d = PhysFormerRoPE(
            dim=self.rope_dim,
            max_freq=10.0,
            use_3d=True,
            learnable_freqs=False,
            use_set_aggregation=rope_set_aggregation,
            dim_per_coord_3d=dim_per_coord_3d,  # Pass to PhysFormerRoPE
            dim_per_coord_multi=dim_per_coord_multi,  # Pass to PhysFormerRoPE
        )

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (B,H,N,Hd)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, Hd = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, N, H * Hd)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        pos_q: torch.Tensor,
        pos_k: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention using Flash Attention v2.
        """
        B, Nq, _ = query.shape
        _, Nk, _ = key_value.shape

        # Projections => (B, N, H, D) for Flash
        if self.elementwise_gate:
            q_gate = self.q_proj(query)
            q_states, gate_score = torch.split(q_gate, self.d_model, dim=-1)
            q = q_states.view(B, Nq, self.num_heads, self.head_dim)
            gate_score = gate_score.view(B, Nq, self.num_heads, self.head_dim)
        else:
            q = self.q_proj(query).view(B, Nq, self.num_heads, self.head_dim)
            gate_score = None

        k = self.k_proj(key_value).view(B, Nk, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).view(B, Nk, self.num_heads, self.head_dim)

        # QK Norm
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE
        if pos_q is None or pos_k is None:
            raise ValueError("CrossAttentionWithRoPE expects pos_q and pos_k")

        # Single-point (3D) vs multi-point positions
        if pos_q.shape[-1] == 3:
            rope_q = self.rope3d.forward_3d(pos_q)
        else:
            rope_q = self.rope3d.forward_sa_rope(
                pos_q
            )  # SA-RoPE: symmetric anchor pooling

        if pos_k is pos_q:
            rope_k = rope_q  # identical positions -> identical angles (computed once)
        elif pos_k.shape[-1] == 3:
            rope_k = self.rope3d.forward_3d(pos_k)
        else:
            rope_k = self.rope3d.forward_sa_rope(
                pos_k
            )  # SA-RoPE: symmetric anchor pooling

        # (B, 1, N, rope_dim) angles broadcast over heads (see SelfAttentionWithRoPE)
        rope_q_T = rope_q.unsqueeze(1)
        rope_k_T = rope_k.unsqueeze(1) if rope_k is not rope_q else rope_q_T

        # apply_rotary_emb operates on (B, H, N, D): transpose for the rotation and back
        q_rot = q[..., : self.rope_dim].transpose(1, 2)
        k_rot = k[..., : self.rope_dim].transpose(1, 2)

        q_rot = apply_rotary_emb(rope_q_T, q_rot)
        k_rot = apply_rotary_emb(rope_k_T, k_rot)

        q_rot = q_rot.transpose(1, 2)
        k_rot = k_rot.transpose(1, 2)

        q_pass = q[..., self.rope_dim :]
        k_pass = k[..., self.rope_dim :]
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)

        # flash_attn_func does not support additive masks: dispatch to SDPA when a
        # mask is provided, Flash Attention otherwise.
        if attn_mask is not None:
            if not self._attn_dispatch_logged:
                n_inf = (attn_mask == float("-inf")).sum().item()
                n_total = attn_mask.numel()
                print(
                    f"[CrossAttentionWithRoPE] SDPA path (mask active): Q={q.shape}, K={k.shape}, "
                    f"mask={attn_mask.shape}, masked={n_inf}/{n_total} ({100*n_inf/max(n_total,1):.1f}%)"
                )
                self._attn_dispatch_logged = True
            # SDPA expects (B, H, N, D) layout
            q_sdpa = q.transpose(1, 2)  # (B, H, Nq, D)
            k_sdpa = k.transpose(1, 2)  # (B, H, Nk, D)
            v_sdpa = v.transpose(1, 2)  # (B, H, Nk, D)
            out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=self.scale,
            )  # (B, H, Nq, D)
            out = out.transpose(1, 2)  # Back to (B, Nq, H, D)
        else:
            if not self._attn_dispatch_logged:
                print(
                    f"[CrossAttentionWithRoPE] Flash path (no mask): Q={q.shape}, K={k.shape}"
                )
                self._attn_dispatch_logged = True
            # Flash Attention runs in bf16/fp16; the output is cast back to the
            # original dtype immediately.
            original_dtype = query.dtype
            attn_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            q_half = q.to(attn_dtype)
            k_half = k.to(attn_dtype)
            v_half = v.to(attn_dtype)
            out = flash_attn_func(
                q_half,
                k_half,
                v_half,
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=self.scale,
                backend=getattr(self, "attention_backend", "sdpa"),
            )
            out = out.to(original_dtype)

        if self.elementwise_gate:
            out = out * torch.sigmoid(gate_score)

        out = out.reshape(B, Nq, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        return out


class FeedForward(nn.Module):
    """Feedforward network with activation and dropout.
    Supports GELU/RELU and SwiGLU (default). For SwiGLU, hidden_dim is the inner dimension.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = "swiglu",
    ):
        super().__init__()
        self.activation_name = activation
        self.dropout = nn.Dropout(dropout)
        if activation == "swiglu":
            self.fc1 = nn.Linear(d_model, hidden_dim)
            self.fc1_gate = nn.Linear(d_model, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, d_model)
        else:
            self.fc1 = nn.Linear(d_model, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, d_model)
            if activation == "gelu":
                self.act = nn.GELU()
            elif activation == "relu":
                self.act = nn.ReLU()
            elif activation == "silu":
                self.act = nn.SiLU()
            else:
                raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        if self.activation_name == "swiglu":
            a = self.fc1(x)
            b = torch.nn.functional.silu(self.fc1_gate(x))
            y = a * b
            y = self.dropout(y)
            y = self.fc2(y)
            y = self.dropout(y)
            return y
        else:
            y = self.fc1(x)
            y = self.act(y)
            y = self.dropout(y)
            y = self.fc2(y)
            y = self.dropout(y)
            return y


class DropPath(nn.Module):
    """Stochastic Depth per sample (DropPath)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


if __name__ == "__main__":
    """Test the decoder module."""

    print("Testing BidirectionalDecoder...")
    decoder = BidirectionalDecoder(
        state_dim=512, num_layers=4, num_heads=8, feedforward_dim=2048
    )

    # Create dummy inputs
    batch_size = 2
    state = torch.randn(batch_size, 256, 512)  # 256 state tokens
    observation = torch.randn(batch_size, 10, 512)  # 10 observation tokens

    # Forward pass
    output = decoder(state, observation)
    print(f"Output shape: {output.shape}")  # Should be (2, 10, 512)

    print("\nDecoder test successful!")
