# Derived from Zhiyang Dou's RigidFormer, revision 160fe0852c51cffae7b7440452b58d0b49db6df6.
# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Adaptations: package-relative imports; framework-owned public interfaces.
# The complete MIT license is in ../THIRD_PARTY_LICENSES.

"""
Multi-scale prediction head with cross-attention to decoder features.
Query-based predictor using RoPE-aware attention for position encoding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# Import RoPE modules for position-aware attention
from embodichain_models.dynamics.rigidformer._native.models.decoder import (
    CrossAttentionWithRoPE,
    SelfAttentionWithRoPE,
)


class QueryBasedMultiScalePredictor(nn.Module):
    """
    Query-based predictor with cross-attention to multi-scale features.
    Anchors attend to multi-scale decoder features instead of pooling them.
    """

    def __init__(
        self,
        state_dim: int = 768,
        query_dim: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 3,
        layer_indices: List[int] = [0, 6, 9, 12],
        num_heads: int = 8,
        dropout: float = 0.1,
        use_ref_pose: bool = False,  # Whether to use reference pose
        norm_type: str = "layer_norm",  # 'layer_norm' | 'rms_norm'
        qk_norm: bool = False,  # normalize q/k before attention
        fusion_activation: str = "swiglu",  # 'swiglu' | 'gelu' | 'silu'
        use_rope: bool = True,  # Position-aware attention with RoPE
        rope_dim: int = None,  # Custom RoPE dimension
        dim_per_coord_3d: int = None,  # Dim per coordinate for single-point (3D) RoPE
        dim_per_coord_multi: int = None,  # Dim per coordinate for multi-point RoPE
        layer_index_mapping: dict = None,  # Maps original layer index to saved position
        elementwise_gate: bool = False,  # Elementwise gated attention
        cross_object_attention: bool = False,  # Cross-object anchor attention
        cross_object_kv: bool = False,  # Cross-object key-value: anchors query all objects
        no_global_state: bool = False,  # Skip state modules entirely when True
    ):
        super().__init__()
        self.cross_object_attention = cross_object_attention
        self.cross_object_kv = cross_object_kv
        self.no_global_state = no_global_state
        self._cross_object_attention_logged = False  # One-time log flag
        self._cross_object_kv_logged = False  # One-time log flag
        self._p0_mask_logged = False  # One-time padding mask log
        self._p1_no_state_logged = False  # One-time no-state log
        if cross_object_attention:
            print(
                f"[Predictor] Cross-object anchor attention enabled: anchors from different objects can interact"
            )
        if cross_object_kv:
            print(
                f"[Predictor] Cross-object key-value enabled: anchors query all objects' features"
            )
        if no_global_state:
            print(
                f"[Predictor] no_global_state=True: state modules omitted, fusion uses object features only"
            )

        self.layer_indices = layer_indices
        self.num_scales = len(layer_indices)

        # Maps original layer index to its position in the saved feature list.
        # If None, all layers are assumed saved.
        if layer_index_mapping is not None:
            self.layer_index_mapping = layer_index_mapping
        else:
            # Default: assume sequential saving (all layers saved)
            self.layer_index_mapping = {idx: idx for idx in layer_indices}
        self.use_ref_pose = use_ref_pose
        self.use_rope = use_rope
        self.hidden_dim = hidden_dim

        # Scale-specific projections for object features
        self.scale_projections = nn.ModuleList(
            [nn.Linear(state_dim, hidden_dim) for _ in range(self.num_scales)]
        )

        # Scale-specific projections for state features (skipped when no_global_state)
        if not no_global_state:
            self.state_scale_projections = nn.ModuleList(
                [nn.Linear(state_dim, hidden_dim) for _ in range(self.num_scales)]
            )

        # Query projection; query_dim already includes the ref-pose concatenation
        self.query_proj = nn.Linear(query_dim, hidden_dim)

        # Cross-attention to object features for each scale with RoPE
        if use_rope:
            self.cross_attentions = nn.ModuleList(
                [
                    CrossAttentionWithRoPE(
                        d_model=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        use_rope=True,
                        rope_dim=rope_dim,
                        dim_per_coord_3d=dim_per_coord_3d,
                        dim_per_coord_multi=dim_per_coord_multi,
                        elementwise_gate=elementwise_gate,
                    )
                    for _ in range(self.num_scales)
                ]
            )
            # Cross-attention to state features (skipped when no_global_state)
            if not no_global_state:
                self.state_cross_attentions = nn.ModuleList(
                    [
                        CrossAttentionWithRoPE(
                            d_model=hidden_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            use_rope=True,
                            rope_dim=rope_dim,
                            dim_per_coord_3d=dim_per_coord_3d,
                            dim_per_coord_multi=dim_per_coord_multi,
                            elementwise_gate=elementwise_gate,
                        )
                        for _ in range(self.num_scales)
                    ]
                )
            # Self-attention for anchors within each object (with RoPE)
            self.self_attentions = nn.ModuleList(
                [
                    SelfAttentionWithRoPE(
                        d_model=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        use_rope=True,
                        rope_dim=rope_dim,
                        dim_per_coord_3d=dim_per_coord_3d,
                        dim_per_coord_multi=dim_per_coord_multi,
                        elementwise_gate=elementwise_gate,
                    )
                    for _ in range(self.num_scales)
                ]
            )
        else:
            # Standard attention
            self.cross_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        hidden_dim, num_heads, dropout=dropout, batch_first=True
                    )
                    for _ in range(self.num_scales)
                ]
            )
            if not no_global_state:
                self.state_cross_attentions = nn.ModuleList(
                    [
                        nn.MultiheadAttention(
                            hidden_dim, num_heads, dropout=dropout, batch_first=True
                        )
                        for _ in range(self.num_scales)
                    ]
                )
            # Self-attention for anchors (standard attention)
            self.self_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        hidden_dim, num_heads, dropout=dropout, batch_first=True
                    )
                    for _ in range(self.num_scales)
                ]
            )

        # Norm modules (configurable type) for norm-first and qk-norm
        Norm = (
            nn.LayerNorm
            if norm_type == "layer_norm"
            else getattr(nn, "RMSNorm", nn.LayerNorm)
        )
        # For object attention
        self.query_in_norms = nn.ModuleList(
            [Norm(hidden_dim) for _ in range(self.num_scales)]
        )
        self.kv_in_norms = nn.ModuleList(
            [Norm(hidden_dim) for _ in range(self.num_scales)]
        )
        self.post_attn_norms = nn.ModuleList(
            [Norm(hidden_dim) for _ in range(self.num_scales)]
        )
        # For state attention (skipped when no_global_state)
        if not no_global_state:
            self.state_kv_in_norms = nn.ModuleList(
                [Norm(hidden_dim) for _ in range(self.num_scales)]
            )
            self.state_post_attn_norms = nn.ModuleList(
                [Norm(hidden_dim) for _ in range(self.num_scales)]
            )
        # For self-attention (within same object)
        self.self_attn_in_norms = nn.ModuleList(
            [Norm(hidden_dim) for _ in range(self.num_scales)]
        )
        self.self_attn_post_norms = nn.ModuleList(
            [Norm(hidden_dim) for _ in range(self.num_scales)]
        )
        self._use_qk_norm = qk_norm

        # Cross-object anchor self-attention
        # Allows anchors from different objects to interact directly
        if cross_object_attention:
            if use_rope:
                self.cross_obj_self_attentions = nn.ModuleList(
                    [
                        SelfAttentionWithRoPE(
                            d_model=hidden_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            use_rope=True,
                            rope_dim=rope_dim,
                            dim_per_coord_3d=dim_per_coord_3d,
                            dim_per_coord_multi=dim_per_coord_multi,
                            elementwise_gate=elementwise_gate,
                        )
                        for _ in range(self.num_scales)
                    ]
                )
            else:
                self.cross_obj_self_attentions = nn.ModuleList(
                    [
                        nn.MultiheadAttention(
                            hidden_dim, num_heads, dropout=dropout, batch_first=True
                        )
                        for _ in range(self.num_scales)
                    ]
                )
            self.cross_obj_in_norms = nn.ModuleList(
                [Norm(hidden_dim) for _ in range(self.num_scales)]
            )
            self.cross_obj_post_norms = nn.ModuleList(
                [Norm(hidden_dim) for _ in range(self.num_scales)]
            )

        # Fusion and prediction
        # With no_global_state, fusion uses object features only (no dead zero channels)
        if no_global_state:
            fusion_input_dim = hidden_dim * self.num_scales  # Only object features
            print(
                f"[Predictor] fusion_input_dim = {fusion_input_dim} (object features only)"
            )
        else:
            fusion_input_dim = (
                hidden_dim * self.num_scales * 2
            )  # × 2 for object + state
        if use_ref_pose:
            # Add extra projection for reference pose in fusion
            # Reference pose is always 3D (positions)
            self.ref_pose_proj = nn.Linear(3, hidden_dim)
            fusion_input_dim += hidden_dim
            print(f"[Predictor] fusion_input_dim with ref_pose = {fusion_input_dim}")

        # Fusion head: norm-first + SwiGLU (default)
        def _fusion_block(in_dim: int, out_dim: int) -> nn.Sequential:
            if fusion_activation == "swiglu":
                return nn.Sequential(
                    (
                        nn.LayerNorm(in_dim)
                        if norm_type == "layer_norm"
                        else getattr(nn, "RMSNorm", nn.LayerNorm)(in_dim)
                    ),
                    nn.Linear(in_dim, out_dim),
                    nn.Linear(in_dim, out_dim),  # gate
                )
            else:
                act = nn.GELU() if fusion_activation == "gelu" else nn.SiLU()
                return nn.Sequential(
                    (
                        nn.LayerNorm(in_dim)
                        if norm_type == "layer_norm"
                        else getattr(nn, "RMSNorm", nn.LayerNorm)(in_dim)
                    ),
                    nn.Linear(in_dim, out_dim),
                    act,
                )

        # Build two-stage fusion: [fusion_dim -> hidden_dim*2 -> hidden_dim] then linear to output
        self.fusion_dim1 = hidden_dim * 2
        self.fusion_norm1 = (
            nn.LayerNorm(fusion_input_dim)
            if norm_type == "layer_norm"
            else getattr(nn, "RMSNorm", nn.LayerNorm)(fusion_input_dim)
        )
        self.fusion_w1 = nn.Linear(fusion_input_dim, self.fusion_dim1)
        self.fusion_g1 = nn.Linear(fusion_input_dim, self.fusion_dim1)
        self.fusion_drop = nn.Dropout(dropout)
        self.fusion_norm2 = (
            nn.LayerNorm(self.fusion_dim1)
            if norm_type == "layer_norm"
            else getattr(nn, "RMSNorm", nn.LayerNorm)(self.fusion_dim1)
        )
        self.fusion_w2 = nn.Linear(self.fusion_dim1, hidden_dim)
        self.fusion_g2 = nn.Linear(self.fusion_dim1, hidden_dim)
        self.fusion_out = nn.Linear(hidden_dim, output_dim)

        # Lightweight penetration head for vertex-level sign prediction: +1 (outside)
        # vs -1 (inside) for all vertices, computed from the last-layer object
        # feature, a global state feature, and the vertex position.
        # Two 32-d projections + a 16-d position projection -> 1-d tanh score.
        self._pen_obj_proj = nn.Linear(state_dim, 32)
        self._pen_state_proj = nn.Linear(state_dim, 32)
        self._pen_pos_proj = nn.Linear(3, 16)
        self._pen_head_norm = (
            nn.LayerNorm(32 + 32 + 16)
            if norm_type == "layer_norm"
            else getattr(nn, "RMSNorm", nn.LayerNorm)(32 + 32 + 16)
        )
        self._pen_head = nn.Linear(32 + 32 + 16, 1)

    # torch.nn.RMSNorm built without an explicit eps uses torch.finfo(input.dtype).eps: 0.0078125 for the bf16 inputs
    # four module groups receive under autocast in training, 1.19e-7 for fp32 inputs, and the evaluation rollouts run
    # without autocast. pin_rmsnorm_eps() gives every module the eps of the dtype it receives in bf16-autocast
    # training (input dtypes measured in bf16-autocast training), so evaluation computes the trained function;
    # bf16-autocast training itself is unchanged.
    _RMSNORM_BF16_TRAINING_INPUT = (
        "query_in_norms",
        "kv_in_norms",
        "post_attn_norms",
        "fusion_norm2",
    )
    _RMSNORM_FP32_TRAINING_INPUT = (
        "self_attn_in_norms",
        "self_attn_post_norms",
        "cross_obj_in_norms",
        "cross_obj_post_norms",
        "fusion_norm1",
        "_pen_head_norm",
    )

    def pin_rmsnorm_eps(self) -> int:
        bf16, fp32 = torch.finfo(torch.bfloat16).eps, torch.finfo(torch.float32).eps
        n = 0
        for name, m in self.named_modules():
            if not isinstance(m, nn.RMSNorm):
                continue
            top = name.split(".")[0]
            if top in self._RMSNORM_BF16_TRAINING_INPUT:
                m.eps = bf16
            elif top in self._RMSNORM_FP32_TRAINING_INPUT:
                m.eps = fp32
            else:
                raise ValueError(
                    f"pin_rmsnorm_eps: training input dtype of RMSNorm '{name}' was not measured"
                )
            n += 1
        return n

    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        query: torch.Tensor,
        ref_pose: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,  # Anchor 3D positions for RoPE
        key_pos: Optional[torch.Tensor] = None,  # Object anchor positions for RoPE
        multi_scale_state_features: Optional[
            List[torch.Tensor]
        ] = None,  # State features from decoder
        state_pos: Optional[torch.Tensor] = None,  # State 3D grid positions for RoPE
        obj_mask: Optional[
            torch.Tensor
        ] = None,  # (B, Nobj) bool mask for valid objects
    ) -> torch.Tensor:
        """
        Query-based prediction using cross-attention to multi-scale object and state features.

        Supports two input formats:
        1. Grouped: query (B, Nobj, Nq, feature_dim) - preferred for per-object attention
        2. Flat: query (B, M, feature_dim) where M=Nobj*Nq

        Args:
            multi_scale_features: List of object features from decoder layers (B, Nobj, D)
            query: Query features
                - Grouped: (B, Nobj, Nq, 9/12) - dist_vec + velocity + rel_pos
                - Flat: (B, Nobj*Nq, 9/12)
            ref_pose: Reference pose
                - Grouped: (B, Nobj, Nq, 3)
                - Flat: (B, Nobj*Nq, 3)
            query_pos: Anchor 3D positions for RoPE
                - Grouped: (B, Nobj, Nq, 3)
                - Flat: (B, Nobj*Nq, 3)
            key_pos: Object anchor positions (B, Nobj, Nq*3) for multi-point RoPE
                     (RenderFormer-style: flattened multi-point representation)
            multi_scale_state_features: List of state features from decoder layers (B, N_state, D)
            state_pos: State 3D grid positions (B, N_state, 3) for RoPE

        Returns:
            Predictions matching input format:
                - Grouped input -> (B, Nobj, Nq, output_dim)
                - Flat input -> (B, Nobj*Nq, output_dim)
        """
        # Detect input format
        is_grouped = query.dim() == 4  # (B, Nobj, Nq, D) vs (B, M, D)

        if is_grouped:
            B, Nobj, Nq, query_dim = query.shape
            M = Nobj * Nq
        else:
            B, M, query_dim = query.shape
            # Infer Nobj from key_pos or from the first available feature layer
            if key_pos is not None:
                Nobj = key_pos.shape[1]
            else:
                # Safely access the first available layer in the saved features
                # (layer 0 might not be saved with memory optimization)
                Nobj = (
                    multi_scale_features[0].shape[1]
                    if len(multi_scale_features) > 0
                    else 1
                )
            Nq = M // Nobj
            assert M == Nobj * Nq, f"M={M} must be divisible by Nobj={Nobj}"

        # Select object features from specified layers using the mapping
        # The mapping converts from original layer index to position in saved list
        selected_features = [
            multi_scale_features[self.layer_index_mapping[i]]
            for i in self.layer_indices
        ]

        # Select state features from specified layers (if provided)
        selected_state_features = None
        if multi_scale_state_features is not None:
            selected_state_features = [
                multi_scale_state_features[self.layer_index_mapping[i]]
                for i in self.layer_indices
            ]

        # Convert to grouped format for processing
        if is_grouped:
            # Already grouped
            query_grouped = query  # (B, Nobj, Nq, query_dim)
            ref_pose_grouped = ref_pose  # (B, Nobj, Nq, 3)
            query_pos_grouped = (
                query_pos if query_pos is not None else None
            )  # (B, Nobj, Nq, 3)
        else:
            # Reshape flat to grouped
            query_grouped = query.view(
                B, Nobj, Nq, query_dim
            )  # (B, Nobj, Nq, query_dim)
            if ref_pose is not None:
                ref_pose_grouped = ref_pose.view(B, Nobj, Nq, 3)
            if query_pos is not None:
                query_pos_grouped = query_pos.view(B, Nobj, Nq, 3)

        # Prepare query with reference pose
        assert ref_pose_grouped is not None
        query_input_grouped = torch.cat(
            [query_grouped, ref_pose_grouped], dim=-1
        )  # (B, Nobj, Nq, query_dim+3)

        # Reshape for batched per-object attention: (B*Nobj, Nq, query_dim+3)
        query_input = query_input_grouped.view(B * Nobj, Nq, -1)

        # Validate query dimensionality against the projection layer
        import logging

        logger = logging.getLogger(__name__)
        expected_dim = self.query_proj.in_features
        actual_dim = query_input.shape[-1]
        if expected_dim != actual_dim:
            logger.error(
                f"Dimension mismatch: query_proj expects {expected_dim}D input, got {actual_dim}D"
            )
            logger.error(
                f"  query shape: {query.shape}, ref_pose shape: {ref_pose.shape if ref_pose is not None else None}"
            )
            raise RuntimeError(
                f"Predictor input dimension mismatch: expected {expected_dim}, got {actual_dim}"
            )

        # Project query: (B*Nobj, Nq, query_dim+3) -> (B*Nobj, Nq, hidden_dim)
        query_proj = self.query_proj(query_input)  # (B*Nobj, Nq, hidden_dim)

        # Process each scale with cross-attention (with RoPE if enabled)
        # For batched per-object attention, we reshape features to (B*Nobj, 1, hidden_dim)
        # so each object's features can be attended by its Nq anchors
        # When cross_object_kv=True, each anchor can attend to ALL objects' features (B*Nobj, Nobj, hidden_dim)
        attended_features = []
        attended_state_features = []

        # Build attention masks for cross-object attention from obj_mask
        kv_mask_for_cross_obj = None  # For cross-object KV attention
        anchor_mask_for_global = None  # For global anchor self-attention
        if obj_mask is not None and (
            self.cross_object_kv or self.cross_object_attention
        ):
            # kv mask for cross-object KV: each anchor group may only attend to
            # valid objects' features; obj_mask (B, Nobj) expands to all anchor groups
            kv_mask_expanded = obj_mask.unsqueeze(1).expand(
                B, Nobj, Nobj
            )  # (B, Nobj, Nobj)
            kv_mask_expanded = kv_mask_expanded.reshape(
                B * Nobj, Nobj
            )  # (B*Nobj, Nobj)
            # SDPA additive format: (B*Nobj, 1, Nq, Nobj); True=attend becomes 0.0
            kv_mask_for_cross_obj = (
                kv_mask_expanded.unsqueeze(1).unsqueeze(1).expand(-1, 1, Nq, -1)
            )  # (B*Nobj, 1, Nq, Nobj)
            kv_mask_for_cross_obj = torch.where(
                kv_mask_for_cross_obj, 0.0, float("-inf")
            )  # SDPA additive mask

            # Mask for global anchor self-attention: expand obj_mask to per-anchor
            # (B, Nobj) -> (B, Nobj*Nq)
            anchor_mask_flat = (
                obj_mask.unsqueeze(-1).expand(B, Nobj, Nq).reshape(B, Nobj * Nq)
            )  # (B, Nobj*Nq)
            # SDPA mask: (B, 1, Nobj*Nq, Nobj*Nq); only valid anchors attend to valid anchors
            anchor_mask_2d = anchor_mask_flat.unsqueeze(1) & anchor_mask_flat.unsqueeze(
                2
            )  # (B, Nobj*Nq, Nobj*Nq)
            anchor_mask_for_global = torch.where(
                anchor_mask_2d.unsqueeze(1), 0.0, float("-inf")
            )  # (B, 1, Nobj*Nq, Nobj*Nq)

            if not self._p0_mask_logged:
                n_valid = obj_mask.sum(dim=1).float().mean().item()
                n_pad = Nobj - n_valid
                print(
                    f"[Predictor] Padding mask active: {n_valid:.0f} valid / {n_pad:.0f} padding objects, "
                    f"{n_pad*Nq:.0f} padded anchors masked in attention"
                )
                self._p0_mask_logged = True

        for i, feat in enumerate(selected_features):
            # Project features: (B, Nobj, hidden_dim)
            feat_proj = self.scale_projections[i](feat)  # (B, Nobj, hidden_dim)

            if self.cross_object_kv:
                # === Cross-Object Key-Value Mode ===
                # Each anchor can attend to ALL objects' features
                # Expand feat_proj for all objects: (B, Nobj, D) -> (B*Nobj, Nobj, D)
                # Each of the B*Nobj anchor-groups sees all Nobj object features
                kv_batched = (
                    feat_proj.unsqueeze(1)
                    .expand(B, Nobj, Nobj, -1)
                    .reshape(B * Nobj, Nobj, -1)
                )
                # kv_batched: (B*Nobj, Nobj, hidden_dim)

                # One-time runtime confirmation log
                if not self._cross_object_kv_logged:
                    print(
                        f"[Predictor Runtime] Cross-object KV active: each anchor queries {Nobj} objects' features"
                    )
                    self._cross_object_kv_logged = True
            else:
                # === Standard Per-Object Mode ===
                # Each object has 1 feature vector that Nq anchors will attend to
                kv_batched = feat_proj.view(B * Nobj, 1, -1)  # (B*Nobj, 1, hidden_dim)

            # Norm-first with optional qk-norm
            q_in = self.query_in_norms[i](query_proj)  # (B*Nobj, Nq, hidden_dim)
            kv_in = self.kv_in_norms[i](kv_batched) if self._use_qk_norm else kv_batched

            # Per-object attention: Nq anchors attend to object feature(s)
            if self.use_rope:
                # Use RoPE-aware cross-attention
                if query_pos_grouped is None or key_pos is None:
                    raise ValueError(
                        "QueryBasedMultiScalePredictor with use_rope=True requires "
                        "query_pos (anchor positions) and key_pos (object positions)"
                    )
                # Reshape positions for batched per-object
                # query_pos: individual anchor positions (B, Nobj, Nq, 3) -> (B*Nobj, Nq, 3)
                query_pos_batched = query_pos_grouped.view(
                    B * Nobj, Nq, 3
                )  # (B*Nobj, Nq, 3)

                if self.cross_object_kv:
                    # Key positions for ALL objects: (B, Nobj, Nq*3) -> (B*Nobj, Nobj, Nq*3)
                    # Each anchor-group sees all objects' anchor positions
                    key_pos_batched = (
                        key_pos.unsqueeze(1)
                        .expand(B, Nobj, Nobj, -1)
                        .reshape(B * Nobj, Nobj, -1)
                    )
                    # key_pos_batched: (B*Nobj, Nobj, Nq*3)
                else:
                    # key_pos: flattened multi-point anchors (B, Nobj, Nq*3) -> (B*Nobj, 1, Nq*3)
                    key_pos_batched = key_pos.view(
                        B * Nobj, 1, -1
                    )  # (B*Nobj, 1, Nq*3) for multi-point RoPE

                attn_out = self.cross_attentions[i](
                    query=q_in,  # (B*Nobj, Nq, hidden_dim)
                    key_value=kv_in,  # (B*Nobj, 1 or Nobj, hidden_dim)
                    pos_q=query_pos_batched,  # (B*Nobj, Nq, 3) - individual anchor positions
                    pos_k=key_pos_batched,  # (B*Nobj, 1 or Nobj, Nq*3) - multi-point anchors
                    attn_mask=kv_mask_for_cross_obj if self.cross_object_kv else None,
                )  # (B*Nobj, Nq, hidden_dim)
            else:
                # Standard attention
                attn_out, _ = self.cross_attentions[i](
                    query=q_in, key=kv_in, value=kv_in
                )  # (B*Nobj, Nq, hidden_dim)

            # Residual + post norm
            attn_out = self.post_attn_norms[i](
                attn_out + query_proj
            )  # (B*Nobj, Nq, hidden_dim)
            attended_features.append(attn_out)

            # === Cross-attention to state features (skipped when no_global_state) ===
            if not self.no_global_state and selected_state_features is not None:
                state_feat = selected_state_features[i]  # (B, N_state, D)
                state_feat_proj = self.state_scale_projections[i](
                    state_feat
                )  # (B, N_state, hidden_dim)

                # Expand state features for all anchors across all objects
                # state_feat_proj: (B, N_state, hidden_dim) -> (B*Nobj, N_state, hidden_dim)
                state_feat_expanded = (
                    state_feat_proj.unsqueeze(1)
                    .expand(B, Nobj, -1, -1)
                    .reshape(B * Nobj, -1, self.hidden_dim)
                )

                # Norm
                state_kv_in = (
                    self.state_kv_in_norms[i](state_feat_expanded)
                    if self._use_qk_norm
                    else state_feat_expanded
                )

                # Cross-attention: anchors attend to state
                if self.use_rope:
                    if state_pos is None:
                        raise ValueError(
                            "state_pos is required when using RoPE for state attention"
                        )

                    # Expand state_pos for all objects: (B, N_state, 3) -> (B*Nobj, N_state, 3)
                    state_pos_expanded = (
                        state_pos.unsqueeze(1)
                        .expand(B, Nobj, -1, -1)
                        .reshape(B * Nobj, -1, 3)
                    )

                    state_attn_out = self.state_cross_attentions[i](
                        query=q_in,  # (B*Nobj, Nq, hidden_dim)
                        key_value=state_kv_in,  # (B*Nobj, N_state, hidden_dim)
                        pos_q=query_pos_batched,  # (B*Nobj, Nq, 3) - anchor positions
                        pos_k=state_pos_expanded,  # (B*Nobj, N_state, 3) - state grid positions
                        attn_mask=None,
                    )  # (B*Nobj, Nq, hidden_dim)
                else:
                    state_attn_out, _ = self.state_cross_attentions[i](
                        query=q_in, key=state_kv_in, value=state_kv_in
                    )

                # Residual + post norm
                state_attn_out = self.state_post_attn_norms[i](
                    state_attn_out + query_proj
                )
                attended_state_features.append(state_attn_out)
            else:
                if not self.no_global_state:
                    # Without a global state input: append zeros to keep fusion dimensions consistent
                    attended_state_features.append(
                        torch.zeros_like(attended_features[-1])
                    )
                # With no_global_state, attended_state_features stays empty

            # === Self-attention among anchors within each object ===
            # With no_global_state, only object features feed the self-attention
            if self.no_global_state or len(attended_state_features) <= i:
                sa_input = attended_features[
                    -1
                ]  # (B*Nobj, Nq, hidden_dim), object features only
            else:
                sa_input = (
                    attended_features[-1] + attended_state_features[-1]
                )  # (B*Nobj, Nq, hidden_dim)

            # Norm-first
            sa_input_normed = self.self_attn_in_norms[i](
                sa_input
            )  # (B*Nobj, Nq, hidden_dim)

            # Self-attention: anchors attend to each other within the same object
            if self.use_rope:
                sa_out = self.self_attentions[i](
                    query=sa_input_normed,  # (B*Nobj, Nq, hidden_dim)
                    key_value=sa_input_normed,  # (B*Nobj, Nq, hidden_dim)
                    pos_q=query_pos_batched,  # (B*Nobj, Nq, 3) - anchor positions
                    pos_k=query_pos_batched,  # (B*Nobj, Nq, 3) - same positions for self-attn
                    attn_mask=None,
                )  # (B*Nobj, Nq, hidden_dim)
            else:
                sa_out, _ = self.self_attentions[i](
                    query=sa_input_normed, key=sa_input_normed, value=sa_input_normed
                )  # (B*Nobj, Nq, hidden_dim)

            # Residual + post norm
            sa_out = self.self_attn_post_norms[i](
                sa_out + sa_input
            )  # (B*Nobj, Nq, hidden_dim)

            # === Cross-object anchor self-attention ===
            # Allows anchors from different objects to interact directly
            if self.cross_object_attention:
                # One-time runtime confirmation log
                if not self._cross_object_attention_logged:
                    print(
                        f"[Predictor Runtime] Cross-object anchor attention active: B={B}, Nobj={Nobj}, Nq={Nq}, total_anchors={Nobj*Nq}"
                    )
                    self._cross_object_attention_logged = True
                # Reshape from per-object to global: (B*Nobj, Nq, D) -> (B, Nobj*Nq, D)
                sa_out_global = sa_out.view(
                    B, Nobj * Nq, -1
                )  # (B, Nobj*Nq, hidden_dim)

                # Reshape anchor positions: (B, Nobj, Nq, 3) -> (B, Nobj*Nq, 3)
                query_pos_global = query_pos_grouped.view(
                    B, Nobj * Nq, 3
                )  # (B, Nobj*Nq, 3)

                # Norm-first
                cross_obj_input = self.cross_obj_in_norms[i](
                    sa_out_global
                )  # (B, Nobj*Nq, hidden_dim)

                # Cross-object self-attention: all anchors attend to all anchors
                if self.use_rope:
                    cross_obj_out = self.cross_obj_self_attentions[i](
                        query=cross_obj_input,  # (B, Nobj*Nq, hidden_dim)
                        key_value=cross_obj_input,  # (B, Nobj*Nq, hidden_dim)
                        pos_q=query_pos_global,  # (B, Nobj*Nq, 3) - all anchor positions
                        pos_k=query_pos_global,  # (B, Nobj*Nq, 3) - same for self-attn
                        attn_mask=anchor_mask_for_global,  # Mask padding anchors
                    )  # (B, Nobj*Nq, hidden_dim)
                else:
                    cross_obj_out, _ = self.cross_obj_self_attentions[i](
                        query=cross_obj_input,
                        key=cross_obj_input,
                        value=cross_obj_input,
                    )  # (B, Nobj*Nq, hidden_dim)

                # Residual + post norm
                cross_obj_out = self.cross_obj_post_norms[i](
                    cross_obj_out + sa_out_global
                )  # (B, Nobj*Nq, hidden_dim)

                # Reshape back to per-object: (B, Nobj*Nq, D) -> (B*Nobj, Nq, D)
                sa_out = cross_obj_out.view(
                    B * Nobj, Nq, -1
                )  # (B*Nobj, Nq, hidden_dim)

            # Replace this scale's object features with the enhanced version
            attended_features[-1] = sa_out

        # Concatenate multi-scale features for fusion
        # With no_global_state, only object features are concatenated
        if self.no_global_state:
            multi_scale_output = torch.cat(attended_features, dim=-1)
            # (B*Nobj, Nq, hidden_dim * num_scales)
            if not self._p1_no_state_logged:
                print(
                    f"[Predictor] Fusion uses object features only: {multi_scale_output.shape[-1]}D"
                )
                self._p1_no_state_logged = True
        else:
            multi_scale_output = torch.cat(
                attended_features + attended_state_features, dim=-1
            )
            # (B*Nobj, Nq, hidden_dim * num_scales * 2)

        # Add reference pose to fusion if enabled
        if self.use_ref_pose:
            # ref_pose_grouped: (B, Nobj, Nq, 3) -> (B*Nobj, Nq, 3)
            ref_pose_batched = ref_pose_grouped.view(B * Nobj, Nq, 3)
            ref_pose_features = self.ref_pose_proj(
                ref_pose_batched
            )  # (B*Nobj, Nq, hidden_dim)
            multi_scale_output = torch.cat(
                [multi_scale_output, ref_pose_features], dim=-1
            )
            # (B*Nobj, Nq, hidden_dim * num_scales + hidden_dim)

        # Final fusion and prediction (SwiGLU blocks)
        # All operations are per-anchor: (B*Nobj, Nq, dim)
        # Block 1
        x = self.fusion_norm1(multi_scale_output)
        if hasattr(torch.nn.functional, "silu"):
            x = self.fusion_drop(
                torch.nn.functional.silu(self.fusion_g1(x)) * self.fusion_w1(x)
            )
        else:
            x = self.fusion_drop(torch.sigmoid(self.fusion_g1(x)) * self.fusion_w1(x))
        # Block 2
        x = self.fusion_norm2(x)
        if hasattr(torch.nn.functional, "silu"):
            x = self.fusion_drop(
                torch.nn.functional.silu(self.fusion_g2(x)) * self.fusion_w2(x)
            )
        else:
            x = self.fusion_drop(torch.sigmoid(self.fusion_g2(x)) * self.fusion_w2(x))
        if getattr(self, "fp32_output", False):
            # model.predictor.fp32_output: the acceleration read-out in fp32 also under bf16 autocast (training), as
            # in the fp32 evaluation rollout
            with torch.autocast(device_type=x.device.type, enabled=False):
                predictions = self.fusion_out(x.float())  # (B*Nobj, Nq, output_dim)
        else:
            predictions = self.fusion_out(x)  # (B*Nobj, Nq, output_dim)

        # Reshape output to match input format
        if is_grouped:
            # Return grouped format: (B, Nobj, Nq, output_dim)
            predictions = predictions.view(B, Nobj, Nq, -1)
        else:
            # Return flat format: (B, Nobj*Nq, output_dim)
            predictions = predictions.view(B, M, -1)

        return predictions

    def predict_penetration_vertices(
        self,
        multi_scale_features: List[torch.Tensor],
        multi_scale_state_features: Optional[List[torch.Tensor]],
        pos_full: torch.Tensor,  # (B, Nobj, Nv, 3)
    ) -> torch.Tensor:
        """
        Predict penetration sign for ALL vertices (vertex-level supervision).

        Returns penetration score in [-1, 1] where:
        - +1 indicates vertex is outside other objects (no penetration)
        - -1 indicates vertex is inside other objects (penetration)

        Args:
            multi_scale_features: List of object features from decoder layers (B, Nobj, D)
            multi_scale_state_features: List of state features from decoder layers (B, N_state, D) or None
            pos_full: Vertex positions (B, Nobj, Nv, 3) - AFTER Kabsch transform

        Returns:
            Penetration scores (B, Nobj, Nv, 1) with tanh activation, requires_grad=True for training
        """
        # Use layer_index_mapping to get the correct position in the saved features list
        last_layer_idx = self.layer_indices[-1]
        mapped_idx = self.layer_index_mapping[last_layer_idx]

        obj_last = multi_scale_features[mapped_idx]  # (B, Nobj, D)
        B, Nobj, _ = obj_last.shape
        obj_proj = self._pen_obj_proj(obj_last)  # (B, Nobj, 32)

        if (
            multi_scale_state_features is not None
            and len(multi_scale_state_features) > 0
        ):
            state_last = multi_scale_state_features[mapped_idx]  # (B, N_state, D)
            state_global = state_last.mean(dim=1)  # (B, D)
        else:
            state_global = obj_last.mean(dim=1) * 0.0  # (B, D)
        state_proj = self._pen_state_proj(state_global)  # (B, 32)
        state_proj = state_proj.unsqueeze(1).expand(-1, Nobj, -1)  # (B, Nobj, 32)

        Nv = pos_full.shape[2]
        pos_proj = self._pen_pos_proj(pos_full)  # (B, Nobj, Nv, 16)
        obj_proj_v = obj_proj.unsqueeze(2).expand(-1, -1, Nv, -1)  # (B, Nobj, Nv, 32)
        state_proj_v = state_proj.unsqueeze(2).expand(
            -1, -1, Nv, -1
        )  # (B, Nobj, Nv, 32)

        pen_feat = torch.cat(
            [obj_proj_v, state_proj_v, pos_proj], dim=-1
        )  # (B, Nobj, Nv, 80)
        pen_feat = self._pen_head_norm(pen_feat)
        pen = torch.tanh(self._pen_head(pen_feat))  # (B, Nobj, Nv, 1)
        return pen
