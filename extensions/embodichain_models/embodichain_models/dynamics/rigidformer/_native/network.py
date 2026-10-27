# Derived from Zhiyang Dou's RigidFormer, revision 160fe0852c51cffae7b7440452b58d0b49db6df6.
# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Adaptations: package-relative imports; framework-owned public interfaces.
# The complete MIT license is in ../THIRD_PARTY_LICENSES.

"""Author model composition, extracted from src/train.py; no training CLI imports."""

from __future__ import annotations

import os
import math
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .models.encoder import RigidBodyEncoder, EncoderPack
from .models.state import PersistentStateManager
from .models.decoder import BidirectionalDecoder
from .models.predictor import DynamicsPredictor
from .models.transform_layer import create_transform_layer
from .models.physformer_rope import rope_memo_reset
from .utils.intersection_handler import handle_intersections
from .utils.cuda_knn import (
    compute_nearest_distance_to_list_cuda,
    compute_knn_all_objects_vectorized,
    knn_pack,
    compute_knn_packed,
)

try:
    import roma

    ROMA_AVAILABLE = True
except ImportError:
    roma = None
    ROMA_AVAILABLE = False
logger = logging.getLogger(__name__)
DEBUG_NAN = os.environ.get("DEBUG_NAN", "0") == "1"


def distance_2_plane(points, plane_point, plane_axis, body_dist_vec):
    """
    Check if points are closer to a plane than to another body, and update distance vectors.

    Args:
        points: Tensor of shape (B, M, 3) containing point positions
        plane_point: List [x, y, z] defining a point on the plane (e.g., [0,0,0])
        plane_axis: String "x", "y", or "z" defining the plane normal direction
        body_dist_vec: Tensor of shape (B, M, 3) containing vectors to nearest points on other body

    Returns:
        Updated distance vectors (B, M, 3) where vectors to plane replace body vectors when closer
    """
    B, M, _ = points.shape
    device = points.device

    # Get the axis index
    axis_map = {"x": 0, "y": 1, "z": 2}
    if plane_axis not in axis_map:
        raise ValueError(f"Invalid plane_axis: {plane_axis}. Must be 'x', 'y', or 'z'")
    axis_idx = axis_map[plane_axis]

    # Get plane position on the specified axis
    plane_pos = plane_point[axis_idx]

    # Calculate distance to plane for each point
    # Distance is just the absolute difference in the specified axis
    point_axis_coords = points[:, :, axis_idx]  # (B, M)
    dist_to_plane = torch.abs(point_axis_coords - plane_pos)  # (B, M)

    # Calculate distance from body vectors (norm of the vectors)
    dist_to_body = torch.norm(body_dist_vec, dim=2)  # (B, M)

    # Create vectors from points to plane
    # The vector points from the point to its projection on the plane
    plane_vectors = torch.zeros_like(points)  # (B, M, 3)
    plane_vectors[:, :, axis_idx] = (
        plane_pos - point_axis_coords
    )  # Only non-zero in the plane normal direction

    # Create mask for points closer to plane than to body
    closer_to_plane = dist_to_plane < dist_to_body  # (B, M)

    # Expand mask to match vector dimensions
    closer_to_plane_expanded = closer_to_plane.unsqueeze(-1).expand_as(
        body_dist_vec
    )  # (B, M, 3)

    # Update vectors: use plane vectors where closer to plane, otherwise keep body vectors
    updated_vectors = torch.where(
        closer_to_plane_expanded, plane_vectors, body_dist_vec
    )

    return updated_vectors


def _object_centered(ref_pose: torch.Tensor, vmask: torch.Tensor) -> torch.Tensor:
    """ref_pose (B, Nobj, Nv, 3) minus each object's x/y centroid over valid vertices vmask (B, Nobj, Nv).

    Only x and y are centred: horizontal translation is a symmetry of the dynamics, height above
    the ground plane (z = 0) is not, so z is kept absolute."""
    m = vmask.to(ref_pose.dtype).unsqueeze(-1)
    c = (ref_pose * m).sum(dim=2, keepdim=True) / m.sum(dim=2, keepdim=True).clamp_min(
        1.0
    )
    c = torch.cat([c[..., :2], torch.zeros_like(c[..., 2:3])], dim=-1)
    return ref_pose - c


class CUT3RDynamicsModel(nn.Module):
    """
    Complete model combining encoder, state management, decoder, and predictor.
    """

    def __init__(self, config):
        super().__init__()

        # Store config for later use
        self.config = config

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Physics-parameter and time-step flags (read before submodule construction)
        # Generalization arm: anchor reference poses expressed relative to each object's
        # frame-0 centroid (removes absolute translation from the predictor query), and
        # rotary pairing that matches rotate_half so attention depends on displacement only.
        self._ref_pose_object_centered = bool(
            config["model"].get("ref_pose_object_centered", False)
        )
        from embodichain_models.dynamics.rigidformer._native.models.physformer_rope import (
            PhysFormerRoPE as _RopeCls,
        )

        _RopeCls.INTERLEAVED_PAIRING = bool(
            config["model"].get("rope_interleaved_pairing", False)
        )

        self._if_phys_param = config.get("model", {}).get("if_phys_param", False)
        self._if_time_step = config.get("model", {}).get("if_time_step", False)
        self._film_conditioning = config.get("model", {}).get(
            "film_conditioning", False
        )
        if self._if_phys_param:
            print(
                f"[Model] Physics parameters enabled: mass, friction, restitution (+3D to encoder/predictor)"
            )
        if self._if_time_step:
            if self._film_conditioning:
                print(
                    f"[Model] Time step enabled via FiLM only (NOT concatenated to encoder input)"
                )
            else:
                print(
                    f"[Model] Time step enabled: skip_step_size (s and s², +2D to encoder/predictor)"
                )

        # Rollout state reset: reset state tokens at each step to avoid train-test mismatch
        # When training with short sequences (e.g., 4 frames), state tokens only see 2 updates.
        # During long rollouts (e.g., 400 frames), state drift causes OOD behavior.
        # Setting this to True resets state tokens at each rollout step to match training distribution.
        self._rollout_state_reset = config.get("model", {}).get(
            "rollout_state_reset", False
        )
        if self._rollout_state_reset:
            print(
                f"[Model] Rollout state reset ENABLED: state tokens will be reset at each rollout step"
            )

        # KNN acceleration: GPU-parallel grouped processing
        self._knn_num_groups = config.get("data", {}).get("knn_num_groups", 1)
        # Valid-vertex packing (model.knn_packed, model.encoder.packed): the distance features and the encoder's
        # pointwise layers run over the valid vertices only instead of every padded (object slot x vertex) entry.
        # Equal to the padded paths up to floating-point rounding; the packs are built once per batch / rollout.
        self._knn_packed = bool(config.get("model", {}).get("knn_packed", False))
        self._encoder_packed = bool(
            config.get("model", {}).get("encoder", {}).get("packed", False)
        )
        if self._knn_packed or self._encoder_packed:
            print(
                f"[Model] valid-vertex packing: knn_packed={self._knn_packed}, encoder.packed={self._encoder_packed}"
            )
        if self._knn_packed and self._knn_num_groups > 1:
            raise ValueError(
                "model.knn_packed replaces data.knn_num_groups; set knn_num_groups to 1"
            )
        # model.encoder.packed_pooling: 'segment' pools the packed rows directly (segment max / mean) instead of
        # scattering every level back to the padded layout; same values up to fp32 summation order in the mean.
        if (
            config.get("model", {}).get("encoder", {}).get("packed_pooling", "scatter")
            != "scatter"
            and not self._encoder_packed
        ):
            raise ValueError(
                "model.encoder.packed_pooling applies to the packed encoder path; set model.encoder.packed"
            )
        if self._knn_num_groups > 1:
            print(
                f"[Model] KNN acceleration enabled: num_groups={self._knn_num_groups}"
            )

        # model.kabsch_reference: the mesh the per-step Kabsch fit transforms. 'advancing' (default) = the
        # previous step's result, so every step re-transforms an already transformed mesh and rounding accumulates;
        # 'fixed' = one unchanged reference shape for the whole sequence / rollout (the first current frame), equal
        # in exact arithmetic (the fit is equivariant) without the accumulation.
        _kref = config.get("model", {}).get("kabsch_reference", "advancing")
        if _kref not in ("fixed", "advancing"):
            raise ValueError(
                f"model.kabsch_reference must be 'fixed' or 'advancing', got {_kref!r}"
            )
        self._kabsch_reference_fixed = _kref == "fixed"
        if self._kabsch_reference_fixed:
            print(
                "[Model] Kabsch reference FIXED: every step fits the same reference mesh"
            )

        # model.intersection_handling.after_kabsch_loss: which state the position-after-Kabsch loss supervises.
        # 'rigid' = the Kabsch output (the state the acceleration-after-Kabsch loss already uses), 'handled'
        # (default) = the state after the ground/object correction. The rollout state is the corrected one either way.
        _akl = (
            config.get("model", {})
            .get("intersection_handling", {})
            .get("after_kabsch_loss", "handled")
        )
        if _akl not in ("rigid", "handled"):
            raise ValueError(
                f"model.intersection_handling.after_kabsch_loss must be 'rigid' or 'handled', got {_akl!r}"
            )
        self._after_kabsch_loss_rigid = _akl == "rigid"
        if self._after_kabsch_loss_rigid:
            print(
                "[Model] position-after-Kabsch loss supervises the Kabsch output (before intersection handling)"
            )

        # Time-batched training forward (training.time_batched_forward): the network part of all teacher-forced
        # steps runs as one batch (see _forward_multi_object). Only valid when no step's network input depends on
        # an earlier step's prediction or state; the settings that would create such a dependency are refused.
        self._time_batched_forward = bool(
            config.get("training", {}).get("time_batched_forward", False)
        )
        if self._time_batched_forward:
            _enc = config.get("model", {}).get("encoder", {})
            _bad = []
            if config.get("model", {}).get(
                "if_penetration",
                config.get("model", {}).get("if_directional_distance", False),
            ):
                _bad.append(
                    "model.if_penetration/if_directional_distance (uses the previous step's decoder output)"
                )
            if not _enc.get("masked_pooling", False):
                _bad.append(
                    "model.encoder.masked_pooling=false (BatchNorm statistics would pool across steps)"
                )
            if _enc.get("use_stn", False) or _enc.get("feature_transform", False):
                _bad.append(
                    "encoder STN / feature transform (their BatchNorms would pool across steps)"
                )
            if _bad:
                raise ValueError(
                    "training.time_batched_forward is incompatible with: "
                    + "; ".join(_bad)
                )
            print(
                "[Model] time-batched training forward ACTIVE (network part of all teacher-forced steps in one "
                "batch; per-step BatchNorm statistics)",
                flush=True,
            )

        # Debug mode: from data.random_step_size.if_debug or training.loss_weights.if_debug
        self._debug_mode = config.get("data", {}).get("random_step_size", {}).get(
            "if_debug", False
        ) or config.get("training", {}).get("loss_weights", {}).get("if_debug", False)

        # Config Consistency Check: Multi-timescale training requires time_step feature
        random_step_size_config = config.get("data", {}).get("random_step_size", {})
        if random_step_size_config.get("enable", False):
            if not self._if_time_step:
                raise ValueError(
                    "Config inconsistency detected!\n"
                    "  data.random_step_size.enable = True  (multi-timescale training enabled)\n"
                    "  model.if_time_step = False  (time step feature disabled)\n"
                    "\n"
                    "Multi-timescale training requires the network to receive step_size as input.\n"
                    "Please set model.if_time_step = true in your config file."
                )
            print(
                f"[Model] Multi-timescale training enabled: step_size will be used as condition feature"
            )

        # Initialize components (multi-object only)

        # ============================================================
        # ARCHITECTURAL INVARIANTS — Translation Consistency & Permutation Equivariance
        # These properties are fundamental to the model. Enforce at init time.
        # ============================================================

        # 1. State tokens permanently removed — fixed 3D grid breaks translation consistency
        self._no_global_state = True
        self.state_repr = None
        assert (
            config["model"].get("no_global_state", True) is True
        ), "no_global_state must be True: state tokens on a fixed 3D grid break translation consistency"

        # 2. rope_set_aggregation must be True — concat mode is anchor-order sensitive
        assert config["model"]["decoder"].get("rope_set_aggregation", False) is True, (
            "rope_set_aggregation must be True: the concatenation mode is anchor-order sensitive, "
            "breaking permutation invariance of ARoPE within each object"
        )

        # 3. No sequence-index positional embeddings — would break object permutation equivariance
        assert not config["model"]["decoder"].get(
            "use_learnable_pe", False
        ), "use_learnable_pe must be False: sequence-index PE breaks object permutation equivariance"

        # ============================================================

        # Multi-scale predictor reads features from selected decoder layers (CUT3R-style)
        from embodichain_models.dynamics.rigidformer._native.models.multi_scale_predictor import (
            QueryBasedMultiScalePredictor,
        )

        # Adjust layer indices based on actual number of decoder layers (needed before decoder init)
        num_decoder_layers = config["model"]["decoder"].get("num_layers", 12)
        if num_decoder_layers >= 12:
            layer_indices = [0, 6, 9, 12]  # CUT3R's layers for 12-layer model
        elif num_decoder_layers >= 6:
            layer_indices = [0, 2, 4, num_decoder_layers]  # For 6+ layer model
        elif num_decoder_layers >= 4:
            # For 4-layer model
            layer_indices = [0, 1, 2, num_decoder_layers]  # Use all available layers
        else:
            # For 2-layer model (tiny), use all available layers
            layer_indices = list(
                range(num_decoder_layers + 1)
            )  # [0, 1, 2] for 2 layers
        # model.predictor.layer_indices overrides the rule above, e.g. [0, 1, 2, 8] for a decoder grown from 4 to 8
        # layers (scripts/grow_decoder.py) that reads the same layers as the 4-layer model
        if config["model"]["predictor"].get("layer_indices") is not None:
            layer_indices = [
                int(i) for i in config["model"]["predictor"]["layer_indices"]
            ]
            if (
                sorted(set(layer_indices)) != layer_indices
                or layer_indices[-1] != num_decoder_layers
            ):
                raise ValueError(
                    f"model.predictor.layer_indices must be increasing and end with the last decoder layer "
                    f"({num_decoder_layers}); got {layer_indices}"
                )

        # Verify that all layer indices are valid for the decoder
        max_valid_index = (
            num_decoder_layers  # 0 to num_decoder_layers inclusive (initial + N layers)
        )
        for idx in layer_indices:
            assert 0 <= idx <= max_valid_index, (
                f"Invalid layer index {idx} for decoder with {num_decoder_layers} layers. "
                f"Valid range: [0, {max_valid_index}]"
            )

        # Store layer indices for decoder to only save needed layers
        self.predictor_layer_indices = set(layer_indices)  # Use set for O(1) lookup

        # Create mapping from original layer index to position in saved list
        # For example: if layer_indices = [0, 1, 2, 4], then:
        # layer_index_mapping = {0: 0, 1: 1, 2: 2, 4: 3}
        self.layer_index_mapping = {
            idx: i for i, idx in enumerate(sorted(layer_indices))
        }

        # Gated Attention configuration (Qwen3-style)
        gated_attention_config = config["model"]["decoder"].get("gated_attention", {})
        gated_attention_enable = gated_attention_config.get("enable", False)
        gated_attention_type = gated_attention_config.get("type", "elementwise")
        gated_attention_debug = gated_attention_config.get("if_debug", False)
        # Currently only support elementwise gate
        if gated_attention_enable and gated_attention_type != "elementwise":
            raise ValueError(
                f"Unsupported gated_attention type: {gated_attention_type}. Only 'elementwise' is supported."
            )
        self._elementwise_gate = (
            gated_attention_enable and gated_attention_type == "elementwise"
        )

        # Debug print for gated attention (print to console since FileHandler not yet set)
        if gated_attention_debug:
            msg = (
                f"[Gated Attention] enable={gated_attention_enable}, type={gated_attention_type}, "
                f"elementwise_gate={self._elementwise_gate}"
            )
            print(msg)  # Console output (FileHandler not ready at model init time)
            logger.info(
                msg
            )  # Also log (will appear in file after Trainer sets FileHandler)

        # State update mode: when False, state is just learnable embeddings (no bidirectional update)
        # Use with rollout_state_reset=True to avoid train-test mismatch
        self._state_update_enabled = config["model"]["decoder"].get(
            "state_update_enabled", True
        )

        # Objects Self-Attention: Enable direct object-to-object interaction
        # Useful for physics simulation where objects need to exchange information
        self._objects_self_attention = config["model"]["decoder"].get(
            "objects_self_attention", False
        )

        # PE type must be read before decoder init (use_se3 flag needs it)
        self._pe_type = config["model"]["decoder"].get("pe_type", "arope")

        # Initialize decoder with layer_indices_to_save for memory efficiency
        # Block AttnRes config (Kimi 2026)
        _block_attn_res_cfg = config["model"]["decoder"].get("block_attn_res", {})
        _block_attn_res_enable = (
            _block_attn_res_cfg.get("enable", False)
            if isinstance(_block_attn_res_cfg, dict)
            else bool(_block_attn_res_cfg)
        )
        _block_attn_res_block_size = (
            _block_attn_res_cfg.get("block_size", 4)
            if isinstance(_block_attn_res_cfg, dict)
            else 4
        )
        _block_attn_res_final = (
            bool(_block_attn_res_cfg.get("final_aggregate", False))
            if isinstance(_block_attn_res_cfg, dict)
            else False
        )

        self.decoder = BidirectionalDecoder(
            state_dim=config["model"]["state"].get("token_dim", 768),
            num_layers=config["model"]["decoder"].get(
                "num_layers", 12
            ),  # CUT3R uses 12
            num_heads=config["model"]["decoder"].get("num_heads", 12),
            feedforward_dim=config["model"]["decoder"].get("feedforward_dim", 3072),
            dropout=config["model"]["decoder"]["dropout"],
            return_all_layers=True,  # For multi-scale fusion
            use_self_attention=config["model"]["decoder"].get(
                "use_self_attention", False
            ),  # Optional state SA
            center_rope_positions=config["model"].get("rope_center_positions", False),
            rope_set_aggregation=config["model"]["decoder"].get(
                "rope_set_aggregation", False
            ),  # Set aggregation for RoPE
            rope_dim=config["model"]["decoder"].get(
                "rope_dim", None
            ),  # Custom RoPE dimension (None = auto-compute)
            dim_per_coord_3d=config["model"]["decoder"].get(
                "dim_per_coord_3d", None
            ),  # Dim per coordinate for single-point (3D) RoPE
            dim_per_coord_multi=config["model"]["decoder"].get(
                "dim_per_coord_multi", None
            ),  # Dim per coordinate for multi-point RoPE
            layer_indices_to_save=self.predictor_layer_indices,  # Only save layers needed by predictor
            elementwise_gate=self._elementwise_gate,  # Qwen3: elementwise gated attention
            state_update_enabled=self._state_update_enabled,  # When False, state is learnable embeddings only
            objects_self_attention=self._objects_self_attention,  # Enable Objects Self-Attention for object-object interaction
            use_se3=(
                self._pe_type == "se3"
            ),  # SE(3) pairwise bias for obj self-attention
            block_attn_res=_block_attn_res_enable,  # Block Attention Residuals (Kimi 2026)
            block_attn_res_block_size=_block_attn_res_block_size,  # Layers per block
            block_attn_res_final_aggregate=_block_attn_res_final,  # final depth aggregation as the last features
            film_conditioning=(
                config["model"]["decoder"].get("film_conditioning", False)
                and self._film_conditioning
                and self._if_time_step
            ),  # Per-layer FiLM: requires decoder.film_conditioning + model.film_conditioning + model.if_time_step
        )

        # ============ Anchor-Vertex Pooling (AVP) ============
        avp_cfg = config.get("model", {}).get("avp", {})
        self._avp_enabled = avp_cfg.get("enable", False)
        self._avp_output_dim = avp_cfg.get("output_dim", 256)
        if self._avp_enabled:
            from embodichain_models.dynamics.rigidformer._native.models.anchor_vertex_pooling import (
                AnchorVertexPooling,
            )

            _avp_vertex_dim = (
                config.get("model", {})
                .get("encoder", {})
                .get(
                    "backbone_dim",
                    config.get("model", {}).get("encoder", {}).get("output_dim", 1024),
                )
            )
            self.avp = AnchorVertexPooling(
                vertex_dim=_avp_vertex_dim,
                output_dim=self._avp_output_dim,
                init_sigma=avp_cfg.get("init_sigma", 0.1),
            )
            print(
                f"[Model] AVP ENABLED: vertex_dim={_avp_vertex_dim}, output_dim={self._avp_output_dim}, "
                f"init_sigma={avp_cfg.get('init_sigma', 0.1)}"
            )
        else:
            self.avp = None

        # Dynamic predictor input_dim calculation:
        # Base: 12D (9D query: dist_vec+vel+rel_pos + 3D ref_pose)
        # + phys_params (if if_phys_param=true): +3D = 15D
        # + time_step (if if_time_step=true AND NOT film): +2D = 17D
        # + AVP (if avp.enable=true): +avp_output_dim
        predictor_input_dim = 12  # Base: 9D query + 3D ref_pose
        if self._if_phys_param:
            predictor_input_dim += 3  # Physics params
        if self._if_time_step and not self._film_conditioning:
            predictor_input_dim += (
                2  # Time step (s and s²) — skipped when FiLM handles it
            )
        if self._avp_enabled:
            predictor_input_dim += self._avp_output_dim  # AVP features per anchor
            print(
                f"[Model] Predictor input_dim: {predictor_input_dim} "
                f"(base 12 + phys {3 if self._if_phys_param else 0} + AVP {self._avp_output_dim})"
            )

        self.predictor = QueryBasedMultiScalePredictor(
            state_dim=config["model"]["state"].get("token_dim", 768),
            query_dim=predictor_input_dim,  # Dynamically computed based on flags
            hidden_dim=config["model"]["predictor"].get("hidden_dims", [512, 256])[0],
            output_dim=config["model"]["predictor"]["output_dim"],
            layer_indices=layer_indices,
            layer_index_mapping=self.layer_index_mapping,  # Pass the mapping for memory-optimized decoder
            num_heads=config["model"]["predictor"].get(
                "num_heads", config["model"]["decoder"].get("num_heads", 8)
            ),  # Use predictor's num_heads if specified
            dropout=config["model"]["predictor"]["dropout"],
            use_ref_pose=config["model"]["predictor"].get(
                "use_ref_pose", True
            ),  # Enable ref pose
            norm_type=config["model"]["predictor"].get("norm_type", "layer_norm"),
            qk_norm=config["model"]["predictor"].get("qk_norm", False),
            fusion_activation="swiglu",
            use_rope=config["model"]["predictor"].get(
                "use_rope", True
            ),  # Enable RoPE for position-aware attention
            rope_dim=config["model"]["decoder"].get(
                "rope_dim", None
            ),  # Use same RoPE dim as decoder
            dim_per_coord_3d=config["model"]["decoder"].get(
                "dim_per_coord_3d", None
            ),  # Single-Point RoPE config
            dim_per_coord_multi=config["model"]["decoder"].get(
                "dim_per_coord_multi", None
            ),  # Multi-Point RoPE config
            elementwise_gate=self._elementwise_gate,  # Pass elementwise_gate config
            cross_object_attention=config["model"]["predictor"].get(
                "cross_object_attention", False
            ),  # Cross-object anchor attention
            cross_object_kv=config["model"]["predictor"].get(
                "cross_object_kv", False
            ),  # Cross-object key-value attention
            no_global_state=config["model"].get(
                "no_global_state", False
            ),  # Removes unused state input channels
        )
        # model.predictor.rmsnorm_eps: training_dtype -> the predictor's RMSNorm modules keep the eps they have in
        # bf16-autocast training also when evaluated in fp32 (their default eps follows the input dtype; see
        # pin_rmsnorm_eps). Applied here, so every construction of the model computes the configured function.
        _rms_eps_mode = config["model"]["predictor"].get("rmsnorm_eps")
        if _rms_eps_mode == "training_dtype":
            _n_pinned = self.predictor.pin_rmsnorm_eps()
            logger.info(
                f"[Predictor] RMSNorm eps pinned to the bf16-training values on {_n_pinned} modules"
            )
        elif _rms_eps_mode not in (None, "default"):
            raise ValueError(
                "model.predictor.rmsnorm_eps must be 'training_dtype' or 'default'"
            )
        # model.predictor.fp32_output: the acceleration read-out layer in fp32 under bf16 training autocast (evaluation
        # runs it in fp32 already, so evaluation is unchanged)
        self.predictor.fp32_output = bool(
            config["model"]["predictor"].get("fp32_output", False)
        )
        if self.predictor.fp32_output:
            logger.info(
                "[Predictor] acceleration read-out (fusion_out) in fp32 during training"
            )

        # Store attention flags for reference
        self._cross_object_attention = config["model"]["predictor"].get(
            "cross_object_attention", False
        )
        self._cross_object_kv = config["model"]["predictor"].get(
            "cross_object_kv", False
        )

        # Log attention configuration summary
        logger.info("=" * 60)
        logger.info("[Attention Config Summary]")
        logger.info(
            f"  Decoder Objects Self-Attention: {'ENABLED' if self._objects_self_attention else 'DISABLED'}"
        )
        logger.info(
            f"  Predictor Cross-Object Anchor Attention: {'ENABLED' if self._cross_object_attention else 'DISABLED'}"
        )
        logger.info(
            f"  Predictor Cross-Object Key-Value: {'ENABLED' if self._cross_object_kv else 'DISABLED'}"
        )
        logger.info("=" * 60)

        # Store actual dt for video/visualization
        self.dt = config["data"].get("dt", 0.01)  # Actual timestep (0.01 = 100 FPS)
        # Verlet integration always uses dt=1; the network learns accelerations in this unit system.
        self.dt_verlet = 1.0

        # Create transform layer (configurable)
        transform_cfg = config.get("model", {}).get("transform", {})
        transform_method = transform_cfg.get("method", "kabsch")
        self.learnable_transform = transform_method == "learnable"

        # RoMa differentiable Kabsch (if enabled, no STE needed)
        self.if_roma = bool(transform_cfg.get("if_roma", False))
        if self.if_roma and not ROMA_AVAILABLE:
            raise ImportError(
                "RoMa is required when if_roma=true. Install with: pip install roma"
            )

        # Build kwargs per method to avoid passing unsupported args
        if self.learnable_transform:
            tl_kwargs = {
                "use_6d_rotation": transform_cfg.get("use_6d_rotation", True),
                "learnable_mode": transform_cfg.get("learnable_mode", "absolute"),
            }
        else:
            tl_kwargs = {
                "regularize_rotation": transform_cfg.get("regularize_rotation", True)
            }
        self.transform_layer = create_transform_layer(
            method=transform_method, if_roma=self.if_roma, **tl_kwargs
        )

        # ============== Multi-object (MOVi) specific modules ==============
        self.enable_multi_object = bool(
            self.config.get("data", {}).get("multi_object", False)
        )
        if self.enable_multi_object:
            # Cache for mesh faces (dataset_name:obj_idx -> faces array)
            self._faces_cache = {}

            # Shared PointNet encoder for arbitrary number of objects (weights shared)
            enc_cfg = config.get("model", {}).get("encoder", {})
            # Dynamic encoder input_dim calculation:
            # Base: 9D (dist_vec + velocity + rel_pos)
            # + phys_params (if if_phys_param=true): +3D
            # + time_step (if if_time_step=true and no FiLM): +2D (s and s²)
            encoder_input_dim = 9  # Base
            if self._if_phys_param:
                encoder_input_dim += 3  # Physics params
            if self._if_time_step and not self._film_conditioning:
                encoder_input_dim += (
                    2  # Time step (s and s²) — skipped when FiLM handles it
                )
            if enc_cfg.get("height_input", False):
                encoder_input_dim += 1  # Height above the ground plane (absolute z)
            self._encoder_input_dim = encoder_input_dim  # Save for logging

            self.shared_obj_encoder = RigidBodyEncoder(
                input_dim=encoder_input_dim,
                hidden_dims=enc_cfg.get("hidden_dims", [512, 1024]),
                output_dim=enc_cfg.get("backbone_dim", enc_cfg.get("output_dim", 896)),
                backbone_dim=enc_cfg.get(
                    "backbone_dim", enc_cfg.get("output_dim", 896)
                ),
                use_stn=enc_cfg.get("use_stn", False),
                use_batch_norm=enc_cfg.get("use_batch_norm", True),
                masked_pooling=enc_cfg.get("masked_pooling", False),
                freeze_bn_stats=enc_cfg.get("freeze_bn_stats", False),
                height_input=enc_cfg.get("height_input", False),
                hierarchical_subsets=enc_cfg.get("hierarchical_subsets", False),
                packed_pooling=enc_cfg.get("packed_pooling", "scatter"),
                subset_small_objects=enc_cfg.get("subset_small_objects", "all_below_8"),
                feature_transform=enc_cfg.get("feature_transform", False),
                feature_transform_dim=enc_cfg.get("feature_transform_dim", 64),
                pooling_type=enc_cfg.get(
                    "pooling_type", "max"
                ),  # New: enhanced pooling
                num_hierarchical_levels=enc_cfg.get(
                    "num_hierarchical_levels", 2
                ),  # New: hierarchical levels
            )
            # Project per-object latent to model token dim
            token_dim = config["model"]["state"].get("token_dim", 768)
            enc_out_dim = enc_cfg.get("backbone_dim", enc_cfg.get("output_dim", 896))
            # Project PointNet encoder output (per-object) to token dimension for Decoder
            self.obj_feature_proj = nn.Linear(enc_out_dim, token_dim)
            # Anchor query dimension is always 9D (dist_vec + vel + rel_pos)
            anchor_query_dim = 9
            # Projection for anchor query features (for Predictor)
            self.anchor_query_proj = nn.Linear(anchor_query_dim, token_dim)

            # FiLM conditioning: modulate encoder output with [s, s²] instead of
            # concatenating step size to the input features
            if self._film_conditioning and self._if_time_step:
                from embodichain_models.dynamics.rigidformer._native.models.encoder import (
                    StepSizeFiLM,
                )

                self.encoder_film = StepSizeFiLM(time_dim=2, feature_dim=token_dim)
                print(
                    f"[Model] FiLM conditioning ENABLED: step-size modulates encoder output (dim={token_dim})"
                )
                print(
                    f"[Model]   Encoder input: {encoder_input_dim}D (time NOT concatenated, handled by FiLM)"
                )
                print(
                    f"[Model]   Predictor query: {predictor_input_dim}D (time NOT in query)"
                )
            else:
                self.encoder_film = None
                if self._film_conditioning and not self._if_time_step:
                    print(
                        f"[Model] WARNING: film_conditioning=True but if_time_step=False; FiLM disabled"
                    )

            # ============== PE Type for ARoPE Ablation ==============
            # pe_type controls what 12D positional descriptor is fed to the decoder's RoPE path.
            # "arope" (default): original 4 anchors × 3D = 12D
            # "obb": OBB center + 3 half-axis vectors = 12D
            # "pca": PCA centroid + √λ·eigenvectors = 12D
            self._pe_type = config["model"]["decoder"].get("pe_type", "arope")
            _VALID_PE_TYPES = ("arope", "obb", "pca", "se3")
            if self._pe_type not in _VALID_PE_TYPES:
                raise ValueError(
                    f"Invalid pe_type='{self._pe_type}'. Must be one of {_VALID_PE_TYPES}."
                )
            _pe_msg = f"[PE Type] Using pe_type='{self._pe_type}' for decoder/predictor positional encoding."
            print(_pe_msg)
            logger.info(_pe_msg)

            self._geometry_encoder = None
            if self._pe_type in ("obb", "pca"):
                from embodichain_models.dynamics.rigidformer._native.models.geometry_encoders import (
                    OBBEncoder,
                    PCAFrameEncoder,
                )

                if self._pe_type == "obb":
                    self._geometry_encoder = OBBEncoder()
                    logger.info(
                        "[PE Type] Initialized OBBEncoder (center + 3 half-axis vectors -> 12D)"
                    )
                elif self._pe_type == "pca":
                    self._geometry_encoder = PCAFrameEncoder()
                    logger.info(
                        "[PE Type] Initialized PCAFrameEncoder (centroid + scaled eigenvectors -> 12D)"
                    )

            # SE(3) pairwise bias module
            self._se3_bias_module = None
            if self._pe_type == "se3":
                from embodichain_models.dynamics.rigidformer._native.models.se3_attention import (
                    SE3PairwiseBias,
                )

                _num_heads = config["model"]["decoder"].get("num_heads", 6)
                _se3_hidden = config["model"]["decoder"].get("se3_hidden_dim", 64)
                self._se3_bias_module = SE3PairwiseBias(
                    num_heads=_num_heads, hidden_dim=_se3_hidden, use_scale_ratio=True
                )
                logger.info(
                    f"[PE Type] Initialized SE3PairwiseBias (7D -> {_se3_hidden} -> {_num_heads} heads, zero-init, scale-norm)"
                )

            # ============== RoPE frequency configuration summary ==============
            _rope_dim = config["model"]["decoder"].get("rope_dim", None)
            _dim_per_coord_3d = config["model"]["decoder"].get("dim_per_coord_3d", None)
            _dim_per_coord_multi = config["model"]["decoder"].get(
                "dim_per_coord_multi", None
            )
            _hidden_dim = config["model"]["decoder"].get("hidden_dim", 768)
            _num_heads = config["model"]["decoder"].get("num_heads", 6)
            _head_dim = _hidden_dim // _num_heads if _num_heads > 0 else 0
            _anchor_k = config.get("data", {}).get("anchor_k", 4)
            _n_coords = _anchor_k * 3
            _rope_msg = (
                f"[RoPE Freq] rope_dim={_rope_dim} | "
                f"dim_per_coord_multi={_dim_per_coord_multi} (freq/coord) | "
                f"dim_per_coord_3d={_dim_per_coord_3d} | "
                f"hidden_dim={_hidden_dim} | num_heads={_num_heads} | head_dim={_head_dim} | "
                f"anchor_k={_anchor_k} | n_coords={_n_coords} | "
                f"check: {_n_coords}x{_dim_per_coord_multi}={_n_coords * _dim_per_coord_multi if _dim_per_coord_multi else '?'} "
                f"{'== ' + str(_rope_dim) if _dim_per_coord_multi and _n_coords * _dim_per_coord_multi == _rope_dim else '!= ' + str(_rope_dim) + ' (MISMATCH)'}"
            )
            print(_rope_msg)
            logger.info(_rope_msg)

    def forward(self, batch, return_all=False, teacher_forcing_ratio=1.0):
        """
        Forward pass through the model.

        Must remain consistent with rollout():
        1. Reference pose is frame 0 of the full sequence (not batch[:, 0])
        2. Input: current position only (no velocity to encoder)
        3. Output: acceleration, integrated via Verlet to get the next position
        4. Predictor maps (current_pos, ref_pose_frame0) -> acceleration
        5. Verlet: x_{t+1} = a*dt² + 2*x_t - x_{t-1}

        Args:
            batch: Dictionary with body positions and velocities
            return_all: Whether to return intermediate results
            teacher_forcing_ratio: Probability of using ground truth (1.0 = always teacher, 0.0 = always student)

        Returns:
            predictions: Dictionary with predicted positions, velocities, etc.
        """
        # Multi-object path only
        if "obj_positions" in batch and getattr(self, "enable_multi_object", False):
            return self._forward_multi_object(
                batch,
                return_all=return_all,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
        else:
            raise ValueError("Expected multi-object batch with 'obj_positions'.")

    # ================= Multi-object forward (MOVi) =================
    def _forward_multi_object(self, batch, return_all=False, teacher_forcing_ratio=1.0):
        """
        Full multi-object forward path using shared per-object PointNet encoder and
        mask-aware per-vertex queries across all objects.
        """
        time_step_id = batch["time_step_id"]  # (B,)
        # network inputs: the rigid-noise augmentation's perturbed frames when present (the losses keep comparing
        # against batch['obj_positions'])
        obj_pos = batch.get(
            "obj_positions_input", batch["obj_positions"]
        )  # (B,T,Nobj,Nv,3)
        obj_mask = batch["valid_obj_mask"]  # (B,Nobj)
        vtx_mask = batch["valid_vtx_mask"]  # (B,T,Nobj,Nv)
        ref_obj_positions = batch["ref_obj_positions"]  # (B?,Nobj,Nv,3) or (Nobj,Nv,3)

        # Read physics parameters and time step if enabled
        phys_params = None  # (B, Nobj, 3): [mass, friction, restitution]
        skip_step_size_batch = None  # (B,)
        if self._if_phys_param:
            if "phys_params" not in batch:
                raise KeyError(
                    "Batch is missing 'phys_params' required when if_phys_param=true"
                )
            phys_params = batch["phys_params"].to(obj_pos.device)  # (B, Nobj, 3)
        if self._if_time_step:
            if "skip_step_size" not in batch:
                raise KeyError(
                    "Batch is missing 'skip_step_size' required when if_time_step=true"
                )
            skip_step_size_batch = batch["skip_step_size"].to(obj_pos.device)  # (B,)

        B, T, Nobj, Nv, _ = obj_pos.shape
        device = obj_pos.device

        # Normalize reference pose to (B,Nobj,Nv,3)
        if ref_obj_positions.dim() == 3:
            ref_pose = ref_obj_positions.unsqueeze(0).expand(B, -1, -1, -1).to(device)
        else:
            ref_pose = ref_obj_positions.to(device)  # (B,Nobj,Nv,3)

        # ========== Center scene by first frame ==========
        # If enabled, subtract the global center of the first frame from all coordinates;
        # relative distances and object shapes are unaffected.
        if self.config.get("data", {}).get("center_scene_by_t0", False):
            for b in range(B):
                vmask_b0 = vtx_mask[b, 0]  # (Nobj, Nv)
                v_t0 = obj_pos[b, 0][vmask_b0]  # Valid vertices at t=0
                if v_t0.shape[0] > 0:
                    center_t0 = v_t0.mean(dim=0)  # (3,)

                    # Log centering (only once)
                    if not hasattr(self, "_center_t0_logged_train"):
                        logger.info(
                            f"[Center-T0 TRAIN] Enabled. Sample b={b}: center_t0={center_t0.cpu().numpy().tolist()}, "
                            f"obj_pos range before: [{obj_pos[b].min().item():.4f}, {obj_pos[b].max().item():.4f}]"
                        )

                    # Shift positions
                    obj_pos[b] = obj_pos[b] - center_t0
                    ref_pose[b] = ref_pose[b] - center_t0

                    if not hasattr(self, "_center_t0_logged_train"):
                        logger.info(
                            f"[Center-T0 TRAIN] After centering b={b}: "
                            f"obj_pos range: [{obj_pos[b].min().item():.4f}, {obj_pos[b].max().item():.4f}], "
                            f"ref_pose range: [{ref_pose[b].min().item():.4f}, {ref_pose[b].max().item():.4f}]"
                        )
            if not hasattr(self, "_center_t0_logged_train"):
                self._center_t0_logged_train = True
                logger.info(
                    f"[Center-T0 TRAIN] Scene centering applied to obj_pos and ref_pose for batch of {B} samples"
                )

        # Require anchors for query and supervision (no fallback)
        if "anchor_idx" not in batch:
            raise KeyError(
                "Batch is missing 'anchor_idx' required for anchor-based querying"
            )
        anchor_idx = batch["anchor_idx"].to(device).long()  # (B,Nobj,Nq)
        if (
            anchor_idx.dim() != 3
            or anchor_idx.shape[0] != B
            or anchor_idx.shape[1] != Nobj
        ):
            raise ValueError(
                f"anchor_idx must be (B,Nobj,Nq); got {tuple(anchor_idx.shape)} vs (B={B},Nobj={Nobj})"
            )
        Nq = int(anchor_idx.shape[2])

        # Initialize persistent state tokens and 3D positions for RoPE
        # When no_global_state=True, pass None to decoder (skips all state cross-attention)
        if self._no_global_state:
            state_tokens = None
            state_pos = None
        else:
            state_tokens, state_pos = self.state_repr.get_state_and_positions(B, device)

        # Storage across time steps
        all_pred_pos_for_rollout = (
            []
        )  # STE positions for autoregressive input (B, Nobj, Nv, 3)
        all_pred_pos_raw = []  # Raw positions for loss supervision (B, Nobj, Nv, 3)
        all_pred_pos_raw_norm = []  # Raw normalized positions for loss (before Kabsch)
        all_pred_pos_icp_norm = (
            []
        )  # ICP/Kabsch-transformed normalized positions for loss (after Kabsch)
        all_accel_no_icp_norm_anchor_seq = []  # (B, Nobj, Nq, 3)
        all_accel_icp_norm_seq = []
        all_pen_pred_vertices_seq = (
            []
        )  # (B, Nobj, Nv, 1) per step if enabled (ALL vertices)

        # Registration refs per object (initialized with GT at the first step)
        registration_refs = None

        # Previous step's features for penetration prediction (t>1)
        prev_obj_layers = None
        prev_state_layers = None

        # ========== ANCHOR VALIDATION (whole window, vectorized, one host sync) ==========
        # Every anchor of a valid object must sit on a valid vertex at every frame the loop reads
        # (vtx_mask[:, t] for t in 1..T-2). Checked once per batch instead of per (t, object).
        if T > 2:
            if (
                anchor_idx.dim() != 3
                or anchor_idx.shape[0] != B
                or anchor_idx.shape[2] != Nq
            ):
                raise RuntimeError(
                    f"anchor_idx shape mismatch: got {tuple(anchor_idx.shape)}, expected (B={B},Nobj,Nq={Nq})"
                )
            _vmask_win = vtx_mask[:, 1 : T - 1]  # (B,Tw,Nobj,Nv)
            _numv_win = _vmask_win.sum(dim=3)  # (B,Tw,Nobj)
            _idx_win = anchor_idx.unsqueeze(1).expand(
                -1, T - 2, -1, -1
            )  # (B,Tw,Nobj,Nq)
            _at_anchor = torch.gather(_vmask_win, 3, _idx_win)  # (B,Tw,Nobj,Nq)
            _ovalid = obj_mask.unsqueeze(1)  # (B,1,Nobj)
            _zero_bad = _ovalid & (_numv_win == 0)
            _short_bad = _ovalid & (_numv_win < Nq)
            _anch_bad = (~_at_anchor) & _ovalid.unsqueeze(-1)
            _flags = torch.stack([_zero_bad.any(), _short_bad.any(), _anch_bad.any()])
            if bool(
                _flags.any().item()
            ):  # the single sync; everything below is the error path
                zb, sb, ab = [bool(x.item()) for x in _flags]
                _where = (
                    _zero_bad if zb else (_short_bad if sb else _anch_bad.any(dim=-1))
                ).nonzero(as_tuple=False)
                _b0, _tw0, _i0 = [int(v) for v in _where[0][:3]]
                _nv0 = int(_numv_win[_b0, _tw0, _i0].item())
                ds_name = batch.get("dataset_name", None)
                ds_name_str = (
                    ds_name
                    if isinstance(ds_name, str)
                    else (ds_name[0] if hasattr(ds_name, "__getitem__") else "unknown")
                )
                logger.error(
                    f"[Anchor Debug] dataset={ds_name_str} t={_tw0 + 1} obj={_i0} sample={_b0} Nv={Nv} anchor_k={Nq} "
                    f"valid_vertices={_nv0} zero_any={zb} short_any={sb} bad_anchor_any={ab}"
                )
                if zb:
                    raise RuntimeError(
                        f"Zero valid vertices encountered at t={_tw0 + 1}, obj={_i0} for some valid samples; dataset likely corrupted."
                    )
                if sb:
                    raise RuntimeError(
                        f"Not enough valid vertices for anchors at t={_tw0 + 1}, obj={_i0}: valid={_nv0} < anchor_k={Nq}"
                    )
                raise RuntimeError(
                    f"Invalid anchors: anchor indices refer to invalid/padded vertices at t={_tw0 + 1}, obj={_i0}. "
                    f"Consider reducing anchor_k or ensuring consistent Nv per frame."
                )

        # Object slots that hold a valid object in at least one sample (fixed for the batch). The per-object
        # KNN skips the others: they are padding everywhere and their distance vectors are zeroed anyway.
        _knn_active_slots = (
            torch.nonzero(obj_mask.any(dim=0)).flatten().tolist()
            if self._knn_num_groups <= 1
            else None
        )
        # Constant scalars of the decoder attention mask and the valid-object index set used by the Kabsch
        # fit: both are fixed for the batch, so they are built once instead of at every step (the per-step
        # versions cost two host-to-device copies and one nonzero sync each step).
        _attn_zero_fwd = torch.tensor(0.0, dtype=torch.float32, device=obj_mask.device)
        _attn_ninf_fwd = torch.tensor(
            float("-inf"), dtype=torch.float32, device=obj_mask.device
        )
        _kabsch_valid_idx_fwd = obj_mask.reshape(-1).nonzero(as_tuple=True)[0]
        # Valid-vertex packs (model.knn_packed / model.encoder.packed), fixed for the batch: every frame of a
        # sequence has the same vertex mask, which is checked once here.
        _knn_pack_fwd = _enc_pack_fwd = None
        if self._knn_packed or self._encoder_packed:
            _vm1 = vtx_mask[:, 1]
            if not bool((vtx_mask[:, 1 : T - 1] == _vm1.unsqueeze(1)).all()):
                raise ValueError(
                    "valid-vertex packing needs the same vertex mask in every frame of a sequence"
                )
            if self._knn_packed:
                _knn_pack_fwd = knn_pack(_vm1, obj_mask)
            if self._encoder_packed:
                _enc_m = self.shared_obj_encoder
                _vr0 = batch.get("vtx_rank")
                _enc_pack_fwd = EncoderPack(
                    _vm1.reshape(B * Nobj, Nv),
                    _vr0.reshape(B * Nobj, Nv) if _vr0 is not None else None,
                    len(_enc_m.hierarchical_layers),
                    _enc_m.hierarchical_subsets,
                    1,
                    small=_enc_m.subset_small_objects,
                )

        def _net_step(
            t,
            curr_pos_bt,
            prev_pos_bt,
            B,
            Nobj,
            Nv,
            Nq,
            obj_mask,
            vtx_mask_t,
            anchor_idx,
            ref_pose,
            phys_params,
            skip_step_size_batch,
            vtx_rank_all,
            state_tokens,
            bn_groups,
            knn_chunks,
            knn_slots,
            enc_pack=None,
            knn_pk=None,
        ):
            """Network part of one step (distance features -> encoder -> AVP -> decoder -> predictor):
            per-anchor accelerations for the batch given current/previous positions. Called once per step,
            or once for all teacher-forced steps stacked along the batch (bn_groups = knn_chunks = steps).
            """
            # Velocities
            vel_bt = curr_pos_bt - prev_pos_bt  # (B,Nobj,Nv,3)

            # Build per-object features and per-object latents
            per_obj_latents = []  # each (B, D)
            # Build per-object normalized query features and ref poses (padded)
            per_obj_vel_norm = []
            per_obj_relpos_norm = []
            per_obj_distvec_norm = []

            # Prepare far-away filler for invalid vertices in reference lists
            FAR = 1e6

            # (Anchor validation runs once per batch before the time loop: it depends only on the
            #  ground-truth masks and anchor indices, never on predictions.)

            # ========== VECTORIZED KNN FOR ALL OBJECTS AT ONCE ==========
            # curr_pos_bt: (B, Nobj, Nv, 3), vtx_mask_t: (B, Nobj, Nv)
            vmask_t = vtx_mask_t  # (B, Nobj, Nv)

            # Compute KNN distance vectors for ALL objects simultaneously
            # Only computes for valid objects and valid vertices!
            # Returns: (B, Nobj, Nv, 3) - vector from each vertex to nearest point on OTHER objects
            if knn_pk is not None:
                # packed valid vertices; with batched time steps the step-sized chunks share one pack
                _cs = B // knn_chunks
                dist_vec_all = (
                    compute_knn_packed(curr_pos_bt, knn_pk)
                    if knn_chunks == 1
                    else torch.cat(
                        [
                            compute_knn_packed(
                                curr_pos_bt[_c * _cs : (_c + 1) * _cs], knn_pk
                            )
                            for _c in range(knn_chunks)
                        ],
                        dim=0,
                    )
                )
            elif knn_chunks == 1:
                dist_vec_all = compute_knn_all_objects_vectorized(
                    curr_pos_bt,
                    vmask_t,
                    FAR,
                    valid_obj_mask=obj_mask,
                    num_groups=self._knn_num_groups,
                    active_slots=knn_slots,
                )
            else:
                # batched time steps: one call per step-sized chunk (memory of the per-object cdist stays
                # that of a single step)
                _cs = B // knn_chunks
                dist_vec_all = torch.cat(
                    [
                        compute_knn_all_objects_vectorized(
                            curr_pos_bt[_c * _cs : (_c + 1) * _cs],
                            vmask_t[_c * _cs : (_c + 1) * _cs],
                            FAR,
                            valid_obj_mask=obj_mask[_c * _cs : (_c + 1) * _cs],
                            num_groups=self._knn_num_groups,
                            active_slots=knn_slots,
                        )
                        for _c in range(knn_chunks)
                    ],
                    dim=0,
                )

            # Apply plane constraint adjustment (vectorized)
            # Reshape for distance_2_plane: (B, Nobj*Nv, 3)
            # Use reshape() instead of view() to handle non-contiguous tensors
            curr_pos_flat = curr_pos_bt.reshape(B, Nobj * Nv, 3)
            dist_vec_flat = dist_vec_all.reshape(B, Nobj * Nv, 3)
            dist_vec2_flat = distance_2_plane(
                curr_pos_flat, [0, 0, 0], "z", dist_vec_flat
            )
            dist_vec2_stacked = dist_vec2_flat.reshape(B, Nobj, Nv, 3)

            # Relative position (vectorized): curr_pos - ref_pose
            relpos_stacked = curr_pos_bt - ref_pose  # (B, Nobj, Nv, 3)

            # Velocity (already in correct shape)
            vel_stacked = vel_bt  # (B, Nobj, Nv, 3)

            # Noise augmentation (vectorized)
            if self.training:
                aug_config = self.config["data"].get("augmentation", {})
                if aug_config.get("dist_vec_noise", 0) > 0:
                    std = aug_config["dist_vec_noise"]
                    dist_vec2_stacked = (
                        dist_vec2_stacked + torch.randn_like(dist_vec2_stacked) * std
                    )
                if aug_config.get("velocity_noise", 0) > 0:
                    std = aug_config["velocity_noise"]
                    vel_stacked = vel_stacked + torch.randn_like(vel_stacked) * std
                if aug_config.get("rel_pos_noise", 0) > 0:
                    std = aug_config["rel_pos_noise"]
                    relpos_stacked = (
                        relpos_stacked + torch.randn_like(relpos_stacked) * std
                    )

            # Apply vertex masks BEFORE normalization (vectorized across all objects)
            vmask_all = vtx_mask_t.unsqueeze(-1)  # (B,Nobj,Nv,1)
            dist_vec2_stacked = torch.where(
                vmask_all, dist_vec2_stacked, torch.zeros_like(dist_vec2_stacked)
            )
            vel_stacked = torch.where(
                vmask_all, vel_stacked, torch.zeros_like(vel_stacked)
            )
            relpos_stacked = torch.where(
                vmask_all, relpos_stacked, torch.zeros_like(relpos_stacked)
            )

            # Apply network-predicted directional sign to ALL vertices' dist_vec (before encoder)
            # Only for t>1 (may have penetration in student forcing with predicted positions)
            # For t=1, GT positions (no penetration, sign=+1 implicit)
            # Use if_penetration as master switch (fallback to if_directional_distance for backwards compatibility)
            penetration_enabled = self.config["model"].get(
                "if_penetration",
                self.config["model"].get("if_directional_distance", False),
            )
            if penetration_enabled and t > 1:
                # Predict penetration sign using PREVIOUS step's features and CURRENT positions
                # curr_pos_bt was already selected by teacher/student forcing above
                # In student forcing mode: curr_pos_bt = all_pred_pos_for_rollout[-1] = blended_next (after Kabsch transform)
                # GT supervision will use predictions['obj_positions_ste'] to match
                pen_pred_all_vertices = self.predictor.predict_penetration_vertices(
                    prev_obj_layers,  # Use previous step's decoder outputs
                    prev_state_layers,  # Use previous step's state features
                    pos_full=curr_pos_bt,  # AFTER Kabsch transform (student forcing) or GT (teacher forcing/t=1)
                )  # (B, Nobj, Nv, 1) in [-1,1]

                # Apply predicted sign to dist_vec for ALL vertices
                pen_sign_all = torch.sign(
                    pen_pred_all_vertices + 1e-6
                )  # (B, Nobj, Nv, 1) in {-1,+1}
                dist_vec2_stacked = dist_vec2_stacked * pen_sign_all  # (B, Nobj, Nv, 3)

                # Store predicted signs for loss supervision (will compute GT later in validation)
                all_pen_pred_vertices_seq.append(pen_pred_all_vertices)
            elif penetration_enabled and t == 1:
                # First step: GT positions, no penetration prediction
                # Append placeholder (all +1) to maintain time dimension alignment
                # This will NOT be used in loss (masked out), but keeps dimensions consistent
                B, Nobj, Nv = (
                    dist_vec2_stacked.shape[0],
                    dist_vec2_stacked.shape[1],
                    dist_vec2_stacked.shape[2],
                )
                placeholder = torch.ones(
                    (B, Nobj, Nv, 1), dtype=torch.float32, device=self.device
                )
                all_pen_pred_vertices_seq.append(placeholder)

            # No normalization — use raw values directly
            dist_vec2_norm_stacked = dist_vec2_stacked
            vel_norm_stacked = vel_stacked
            relpos_norm_stacked = relpos_stacked

            # Broadcast physics parameters and time_step to vertex dimension if enabled
            phys_broadcast = None  # (B, Nobj, Nv, 3)
            time_broadcast = None  # (B, Nobj, Nv, 2) - contains [s, s²]

            if self._if_phys_param:
                # Broadcast per-object physics params (B,Nobj,3) to all vertices (B,Nobj,Nv,3)
                phys_broadcast = phys_params.unsqueeze(2).expand(
                    B, Nobj, Nv, 3
                )  # (B,Nobj,Nv,3)
                # Mask with vertex mask to ensure padding vertices get zero physics params
                phys_broadcast = torch.where(
                    vmask_all, phys_broadcast, torch.zeros_like(phys_broadcast)
                )

            if self._if_time_step:
                # Broadcast both s and s² to all objects and vertices (B,Nobj,Nv,2)
                # Input both s and s² because:
                #   - s: original step_size for temporal context
                #   - s²: the scaling factor that will be applied to network output
                #   - Network can leverage both for better learning
                step_size_s = skip_step_size_batch.float()  # (B,)
                step_size_sq = (skip_step_size_batch**2).float()  # (B,)
                # Stack s and s² -> (B, 2)
                time_features = torch.stack(
                    [step_size_s, step_size_sq], dim=-1
                )  # (B, 2)
                time_broadcast = time_features.view(B, 1, 1, 2).expand(
                    B, Nobj, Nv, 2
                )  # (B,Nobj,Nv,2)
                # Time step doesn't need masking (it's a scalar property of the batch, not vertex-specific)
                # But for consistency, we can zero it for padding vertices
                # vmask_all is (B,Nobj,Nv,1), expand to match (B,Nobj,Nv,2)
                vmask_time = vmask_all.expand(-1, -1, -1, 2)  # (B,Nobj,Nv,2)
                time_broadcast = torch.where(
                    vmask_time, time_broadcast, torch.zeros_like(time_broadcast)
                )

            # PointNet Encoder: process each object's Nv vertices -> per-object global feature
            # Input dimension depends on flags:
            #   Base: 9D (dist_vec + velocity + rel_pos)
            #   +if_phys_param=true: +3D
            #   +if_time_step=true (no FiLM): +2D (s and s²)
            # Split into components and reshape to (B*Nobj, Nv, D) for encoder
            dist_reshaped = dist_vec2_norm_stacked.view(B * Nobj, Nv, 3)
            vel_reshaped = vel_norm_stacked.view(B * Nobj, Nv, 3)
            relpos_reshaped = relpos_norm_stacked.view(B * Nobj, Nv, 3)

            phys_reshaped = None
            if self._if_phys_param:
                phys_reshaped = phys_broadcast.view(B * Nobj, Nv, 3)

            time_reshaped = None
            if self._if_time_step and not self._film_conditioning:
                time_reshaped = time_broadcast.view(B * Nobj, Nv, 2)  # (s, s²)

            # Prepare vertex mask for encoder pooling
            # vmask_all is (B,Nobj,Nv,1), need (B*Nobj, Nv)
            vtx_mask_for_encoder = vmask_all.squeeze(-1).reshape(
                B * Nobj, Nv
            )  # (B*Nobj, Nv)
            height_reshaped = None
            if self.shared_obj_encoder.height_input:
                _h = torch.where(
                    vmask_all,
                    curr_pos_bt[..., 2:3],
                    torch.zeros_like(curr_pos_bt[..., 2:3]),
                )
                height_reshaped = _h.reshape(B * Nobj, Nv, 1)
            rank_reshaped = None
            if self.shared_obj_encoder.hierarchical_subsets:
                if vtx_rank_all is None:
                    raise KeyError(
                        "model.encoder.hierarchical_subsets requires data.fps_order: true (batch lacks 'vtx_rank')"
                    )
                rank_reshaped = vtx_rank_all.reshape(B * Nobj, Nv)

            # Call encoder with all features (encoder will concatenate them internally)
            # AVP: capture per-point features if AVP is enabled
            obj_features_encoded, _per_vtx_feat, _ = self.shared_obj_encoder(
                dist_reshaped,
                vel_reshaped,
                relpos_reshaped,
                phys_params=phys_reshaped,
                time_step=time_reshaped,
                valid_mask=vtx_mask_for_encoder,
                height=height_reshaped,
                vtx_rank=rank_reshaped,
                bn_groups=bn_groups,
                pack=enc_pack,
            )  # obj_features_encoded: (B*Nobj, enc_out_dim), _per_vtx_feat: (B*Nobj, backbone_dim, Nv)

            # AVP: Anchor-Vertex Pooling — enrich anchor queries with collision-aware features
            _avp_anchor_features = (
                None  # Will be (B, Nobj, Nq, avp_output_dim) if AVP enabled
            )
            if self.avp is not None and _per_vtx_feat is not None:
                # _per_vtx_feat is (B*Nobj, backbone_dim, Nv) — Conv1d channels-first format
                # Transpose to (B*Nobj, Nv, backbone_dim) for AVP
                _pvf = _per_vtx_feat.transpose(1, 2)  # (B*Nobj, Nv, backbone_dim)
                # Anchor positions in flat format
                _curr_pos_flat = curr_pos_bt.reshape(B * Nobj, Nv, 3)  # (B*Nobj, Nv, 3)
                _anchor_pos_flat = torch.gather(
                    _curr_pos_flat,
                    dim=1,
                    index=anchor_idx.reshape(B * Nobj, -1)
                    .unsqueeze(-1)
                    .expand(-1, -1, 3),
                )  # (B*Nobj, Nq, 3)
                _avp_anchor_features = self.avp(
                    _pvf, _curr_pos_flat, _anchor_pos_flat, vtx_mask_for_encoder
                )  # (B*Nobj, Nq, avp_output_dim)
                _avp_anchor_features = _avp_anchor_features.view(
                    B, Nobj, anchor_idx.shape[2], -1
                )  # (B, Nobj, Nq, avp_output_dim)
                # Zero out padding objects (proj bias makes padding non-zero; consistent with other features)
                _avp_anchor_features = (
                    _avp_anchor_features * obj_mask.unsqueeze(-1).unsqueeze(-1).float()
                )

            obj_features_encoded = obj_features_encoded.view(
                B, Nobj, -1
            )  # (B, Nobj, enc_out_dim)

            # Project to token dimension for Decoder
            obj_features = self.obj_feature_proj(obj_features_encoded)  # (B, Nobj, D)

            # FiLM: modulate encoder features with step-size [s, s²]
            if self.encoder_film is not None and skip_step_size_batch is not None:
                time_feat_film = torch.stack(
                    [skip_step_size_batch.float(), (skip_step_size_batch**2).float()],
                    dim=-1,
                )  # (B, 2)
                obj_features = self.encoder_film(
                    obj_features, time_feat_film
                )  # (B, Nobj, D)

            # Object anchor positions for RoPE (like RenderFormer's triangle approach)
            # RenderFormer: Triangle = 3 vertices (9D)
            # Our approach: Object = Nq anchors (Nq*3 D)
            # Gather anchor positions: (B, Nobj, Nq, 3)
            # Replace padding objects' positions with zeros before the gather:
            # padding objects have anchor_idx=0, which would gather arbitrary values into RoPE inputs.
            obj_mask_pos = obj_mask.unsqueeze(-1).unsqueeze(-1)  # (B, Nobj, 1, 1)
            safe_curr_pos_bt = torch.where(
                obj_mask_pos, curr_pos_bt, torch.zeros_like(curr_pos_bt)
            )

            idx_exp_3_for_decoder = anchor_idx.unsqueeze(-1).expand(
                -1, -1, -1, 3
            )  # (B, Nobj, Nq, 3)
            obj_anchor_positions = torch.gather(
                safe_curr_pos_bt, dim=2, index=idx_exp_3_for_decoder
            )  # (B, Nobj, Nq, 3)
            # Flatten to (B, Nobj, Nq*3) for multi-point RoPE
            Nq = anchor_idx.shape[2]  # Number of anchors per object

            # ============== PE Type Routing (forward) ==============
            if self._pe_type in ("arope", "se3"):
                # SE(3) uses arope anchors for cross-attention RoPE; SE(3) bias computed separately
                obj_anchor_flat = obj_anchor_positions.view(
                    B, Nobj, Nq * 3
                )  # (B, Nobj, 12)
            elif self._pe_type in ("obb", "pca"):
                obj_anchor_flat = self._geometry_encoder(
                    safe_curr_pos_bt, vtx_mask_t, obj_mask=obj_mask
                )  # (B, Nobj, 12)
            else:
                raise ValueError(f"Unknown pe_type='{self._pe_type}' in forward().")

            # ============== SE(3) Pairwise Bias Computation ==============
            _se3_bias = None
            if self._pe_type == "se3" and self._se3_bias_module is not None:
                from embodichain_models.dynamics.rigidformer._native.models.se3_attention import (
                    extract_object_pose_features,
                )

                _rotmats, _centers, _scales = extract_object_pose_features(
                    safe_curr_pos_bt, ref_pose, vtx_mask_t, obj_mask
                )
                _se3_bias = self._se3_bias_module(_rotmats, _centers, _scales, obj_mask)

            # Decoder attention mask: SDPA additive format (0=attend, -inf=mask)
            # Prevents padding objects from participating in Decoder self-attention.
            # Combined with feature zeroing to ensure clean inputs.
            obj_attn_mask = torch.where(
                obj_mask.unsqueeze(1).unsqueeze(1),  # (B, 1, 1, Nobj)
                _attn_zero_fwd,
                _attn_ninf_fwd,
            )  # (B, 1, 1, Nobj) float32

            # Zero out padding object features and anchor positions BEFORE Decoder
            # to prevent garbage from padding objects from destabilizing attention
            pad_mask_float = obj_mask.unsqueeze(-1).to(dtype=obj_features.dtype)
            obj_features = obj_features * pad_mask_float  # (B, Nobj, D)
            obj_anchor_flat = obj_anchor_flat * obj_mask.unsqueeze(-1).to(
                dtype=obj_anchor_flat.dtype
            )

            # Decoder interaction with per-object features
            # Per-layer FiLM: pass time_features for step-size conditioning in each layer
            _decoder_time_feat = (
                time_feat_film
                if (self._film_conditioning and self._if_time_step)
                else None
            )
            all_state_layers, all_obj_layers = self.decoder(
                state_tokens,
                obj_features,
                state_pos,
                obj_anchor_flat,
                mask=obj_attn_mask,  # SDPA additive mask for padding objects
                se3_bias=_se3_bias,
                time_features=_decoder_time_feat,  # Per-layer FiLM
            )

            # Update state for next step (skip when no_global_state)
            if not self._no_global_state:
                state_tokens = all_state_layers[-1]

            # Build anchor query for Predictor (sample at anchor points Nq)
            # Query dimension depends on flags:
            #   Base: 9D (dist_vec + vel + rel_pos)
            #   +if_phys_param=true: +3D = 12D
            #   +if_time_step=true (no FiLM): +2D (s and s²)
            idx_exp_3 = anchor_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B,Nobj,Nq,3)
            distvec_at_anchors = torch.gather(
                dist_vec2_norm_stacked, dim=2, index=idx_exp_3
            )  # (B,Nobj,Nq,3)
            vel_at_anchors = torch.gather(
                vel_norm_stacked, dim=2, index=idx_exp_3
            )  # (B,Nobj,Nq,3)
            relpos_at_anchors = torch.gather(
                relpos_norm_stacked, dim=2, index=idx_exp_3
            )  # (B,Nobj,Nq,3)

            # Build query features list
            query_features = [distvec_at_anchors, vel_at_anchors, relpos_at_anchors]

            # Add physics parameters at anchors if enabled
            if self._if_phys_param:
                phys_at_anchors = torch.gather(
                    phys_broadcast, dim=2, index=idx_exp_3
                )  # (B,Nobj,Nq,3)
                query_features.append(phys_at_anchors)

            # Add time step at anchors if enabled AND not using FiLM
            if self._if_time_step and not self._film_conditioning:
                idx_exp_2 = anchor_idx.unsqueeze(-1).expand(
                    -1, -1, -1, 2
                )  # (B,Nobj,Nq,2)
                time_at_anchors = torch.gather(
                    time_broadcast, dim=2, index=idx_exp_2
                )  # (B,Nobj,Nq,2)
                query_features.append(time_at_anchors)

            # AVP: Append collision-aware anchor features to query
            if _avp_anchor_features is not None:
                query_features.append(
                    _avp_anchor_features
                )  # (B, Nobj, Nq, avp_output_dim)

            # Concatenate all query features
            anchor_query = torch.cat(
                query_features, dim=-1
            )  # (B,Nobj,Nq, 9/12 + avp_dim)

            # Extract anchor positions for RoPE (current frame positions at anchors)
            anchor_pos_at_curr = torch.gather(
                curr_pos_bt, dim=2, index=idx_exp_3
            )  # (B,Nobj,Nq,3)

            # Penetration sign was already applied to all vertices' dist_vec before the encoder;
            # the anchor query therefore carries the sign information as well.

            _ref_src = (
                _object_centered(ref_pose, vmask_all.squeeze(-1).bool())
                if self._ref_pose_object_centered
                else ref_pose
            )
            if self._ref_pose_object_centered and not getattr(
                self, "_refpose_logged_fwd", False
            ):
                self._refpose_logged_fwd = True
                print(
                    f"[RefPose] object-centred (x/y only, z absolute) reference pose ACTIVE (forward): |shift| max={((ref_pose - _ref_src).abs().max()).item():.3f} m",
                    flush=True,
                )
            ref_pose_at_anchors = torch.gather(
                _ref_src, dim=2, index=idx_exp_3
            )  # (B,Nobj,Nq,3)

            # Check for NaN/Inf before predictor
            if DEBUG_NAN and (
                torch.isnan(anchor_query).any() or torch.isinf(anchor_query).any()
            ):
                logger.error(f"NaN/Inf detected in anchor_query!")
                logger.error(
                    f"  distvec_at_anchors: nan={torch.isnan(distvec_at_anchors).sum().item()}, inf={torch.isinf(distvec_at_anchors).sum().item()}"
                )
                logger.error(
                    f"  vel_at_anchors: nan={torch.isnan(vel_at_anchors).sum().item()}, inf={torch.isinf(vel_at_anchors).sum().item()}"
                )
                logger.error(
                    f"  relpos_at_anchors: nan={torch.isnan(relpos_at_anchors).sum().item()}, inf={torch.isinf(relpos_at_anchors).sum().item()}"
                )

            if DEBUG_NAN and (
                torch.isnan(ref_pose_at_anchors).any()
                or torch.isinf(ref_pose_at_anchors).any()
            ):
                logger.error(f"NaN/Inf detected in ref_pose_at_anchors!")
                logger.error(
                    f"  ref_pose_at_anchors range: [{ref_pose_at_anchors.min().item():.4f}, {ref_pose_at_anchors.max().item():.4f}]"
                )

            for layer_idx, obj_layer in enumerate(all_obj_layers if DEBUG_NAN else ()):
                if torch.isnan(obj_layer).any() or torch.isinf(obj_layer).any():
                    logger.error(f"NaN/Inf detected in all_obj_layers[{layer_idx}]!")
                    logger.error(
                        f"  Shape: {obj_layer.shape}, nan={torch.isnan(obj_layer).sum().item()}, inf={torch.isinf(obj_layer).sum().item()}"
                    )

            # Predict accelerations per anchor using grouped format (B, Nobj, Nq, *)
            accel_grouped = self.predictor(
                all_obj_layers,
                anchor_query,  # (B, Nobj, Nq, 9/12)
                ref_pose=ref_pose_at_anchors,  # (B, Nobj, Nq, 3)
                query_pos=anchor_pos_at_curr,  # (B, Nobj, Nq, 3)
                key_pos=obj_anchor_flat,  # (B, Nobj, Nq*3) - multi-point RoPE
                multi_scale_state_features=(
                    None if self._no_global_state else all_state_layers
                ),
                state_pos=None if self._no_global_state else state_pos,
                obj_mask=obj_mask,  # Object validity mask for padding
            )  # Returns (B, Nobj, Nq, 3)

            # Check for NaN/Inf after predictor
            if DEBUG_NAN and (
                torch.isnan(accel_grouped).any() or torch.isinf(accel_grouped).any()
            ):
                logger.error(f"NaN/Inf detected in predictor output!")
                logger.error(
                    f"  accel_grouped: nan={torch.isnan(accel_grouped).sum().item()}, inf={torch.isinf(accel_grouped).sum().item()}"
                )
                logger.error(
                    f"  accel_grouped range: min={accel_grouped[~torch.isnan(accel_grouped)].min().item() if (~torch.isnan(accel_grouped)).any() else 'all nan'}, max={accel_grouped[~torch.isnan(accel_grouped)].max().item() if (~torch.isnan(accel_grouped)).any() else 'all nan'}"
                )

            return (
                vmask_t,
                accel_grouped,
                all_obj_layers,
                all_state_layers,
                state_tokens,
            )

        # ========== TIME-BATCHED NETWORK PASS (training.time_batched_forward) ==========
        # With pure teacher forcing every step's network input is built from ground-truth frames, so the network
        # part of all T-2 steps runs once on the steps stacked along the batch (step-major: rows [g*B, (g+1)*B)
        # belong to step g+1). The encoder keeps per-step BatchNorm statistics (grouped masked BN: same
        # statistics, gradients and running-statistics updates as step-by-step calls), and the per-step physics
        # (Verlet, Kabsch, ground handling) below is unchanged. Equal to the step-by-step path up to
        # floating-point rounding and the dropout random stream.
        _tb_accel = None
        if self.training and self._time_batched_forward and T > 2:
            if teacher_forcing_ratio < 1.0:
                raise ValueError(
                    "training.time_batched_forward requires teacher_forcing_ratio == 1 (inputs must be ground truth)"
                )
            Tw = T - 2
            rope_memo_reset()

            def _rep(x):
                return (
                    x.unsqueeze(0)
                    .expand(Tw, *x.shape)
                    .reshape(Tw * x.shape[0], *x.shape[1:])
                )

            _ref_b = (
                _rep(ref_pose)
                if (
                    torch.is_tensor(ref_pose)
                    and ref_pose.dim() == 4
                    and ref_pose.shape[0] == B
                )
                else ref_pose
            )
            _vr = batch.get("vtx_rank")
            _, _acc_s, _, _, _ = _net_step(
                None,
                obj_pos[:, 1 : T - 1].transpose(0, 1).reshape(Tw * B, Nobj, Nv, 3),
                obj_pos[:, 0 : T - 2].transpose(0, 1).reshape(Tw * B, Nobj, Nv, 3),
                Tw * B,
                Nobj,
                Nv,
                Nq,
                _rep(obj_mask),
                vtx_mask[:, 1 : T - 1].transpose(0, 1).reshape(Tw * B, Nobj, Nv),
                _rep(anchor_idx),
                _ref_b,
                _rep(phys_params) if phys_params is not None else None,
                (
                    _rep(skip_step_size_batch)
                    if skip_step_size_batch is not None
                    else None
                ),
                _rep(_vr) if _vr is not None else None,
                None,
                Tw,
                Tw,
                _knn_active_slots,
                enc_pack=(
                    EncoderPack(
                        vtx_mask[:, 1 : T - 1]
                        .transpose(0, 1)
                        .reshape(Tw * B * Nobj, Nv),
                        (
                            _rep(_vr).reshape(Tw * B * Nobj, Nv)
                            if _vr is not None
                            else None
                        ),
                        len(self.shared_obj_encoder.hierarchical_layers),
                        self.shared_obj_encoder.hierarchical_subsets,
                        Tw,
                        small=self.shared_obj_encoder.subset_small_objects,
                    )
                    if self._encoder_packed
                    else None
                ),
                knn_pk=_knn_pack_fwd,
            )
            _tb_accel = _acc_s.view(Tw, B, *_acc_s.shape[1:])

        # Time loop predicts for steps t in [1, T-2] -> next positions x_{t+1}
        for t in range(1, T - 1):
            rope_memo_reset()  # rotary angles are memoized within one step only
            # Teacher vs student forcing (no special-case first step)
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

            if use_teacher_forcing or len(all_pred_pos_for_rollout) == 0:
                prev_pos_bt = obj_pos[:, t - 1]  # (B,Nobj,Nv,3)
                curr_pos_bt = obj_pos[:, t]
            else:
                # Use last predicted (STE blended) as current for autoregressive stability
                curr_pos_bt = all_pred_pos_for_rollout[-1]  # (B,Nobj,Nv,3)
                if len(all_pred_pos_for_rollout) >= 2:
                    prev_pos_bt = all_pred_pos_for_rollout[-2]
                else:
                    prev_pos_bt = obj_pos[:, t - 1]

            if _tb_accel is not None:
                # time-batched: this step's accelerations were computed with all teacher-forced steps
                vmask_t = vtx_mask[:, t]
                accel_grouped = _tb_accel[t - 1]
                all_obj_layers = all_state_layers = None
            else:
                (
                    vmask_t,
                    accel_grouped,
                    all_obj_layers,
                    all_state_layers,
                    state_tokens,
                ) = _net_step(
                    t,
                    curr_pos_bt,
                    prev_pos_bt,
                    B,
                    Nobj,
                    Nv,
                    Nq,
                    obj_mask,
                    vtx_mask[:, t],
                    anchor_idx,
                    ref_pose,
                    phys_params,
                    skip_step_size_batch,
                    batch.get("vtx_rank"),
                    state_tokens,
                    1,
                    1,
                    _knn_active_slots,
                    enc_pack=_enc_pack_fwd,
                    knn_pk=_knn_pack_fwd,
                )
            # Already in grouped format, no need to reshape
            accel_flat = accel_grouped  # (B, Nobj, Nq, 3)
            accel_base_norm = (
                accel_flat  # Network output: base acceleration (step_size-independent)
            )

            # Verlet integration (dt_verlet=1)
            dt_squared = self.dt_verlet * self.dt_verlet
            with torch.amp.autocast("cuda", enabled=False):
                # No normalizer — network output is raw acceleration
                accel_base_denorm = accel_base_norm

                # ============================================================
                # Explicit step_size² scaling for multi-timescale training
                # ============================================================
                # Physical reasoning:
                #   - Network outputs "base acceleration" a_base (step_size-independent)
                #   - Actual acceleration = a_base × step_size²
                #   - This explicit scaling removes the burden from network to learn s² relationship
                #   - Benefit: Better generalization to unseen step_sizes, easier optimization
                # ============================================================
                if self._if_time_step and skip_step_size_batch is not None:
                    step_size_sq = (skip_step_size_batch**2).view(
                        B, 1, 1, 1
                    )  # (B,1,1,1)
                    accel_scaled_denorm = (
                        accel_base_denorm * step_size_sq
                    )  # (B,Nobj,Nq,3)
                    # Also scale normalized version for loss computation
                    accel_no_icp_norm_anchor = accel_base_norm * step_size_sq
                else:
                    accel_scaled_denorm = accel_base_denorm
                    accel_no_icp_norm_anchor = accel_base_norm

                # Build base next positions for all vertices as zero-accel extrapolation
                # No detach: allow gradients to flow through position history for long-horizon training
                next_pos_bt = 2 * curr_pos_bt - prev_pos_bt  # (B,Nobj,Nv,3)

                # Vectorized scatter using scatter_reduce (faster than per-object loop)
                # Step 1: Gather all anchor positions at once
                anchor_idx_exp = anchor_idx.unsqueeze(-1).expand(
                    -1, -1, -1, 3
                )  # (B,Nobj,Nq,3)
                curr_anchors = torch.gather(
                    curr_pos_bt, dim=2, index=anchor_idx_exp
                )  # (B,Nobj,Nq,3)
                prev_anchors = torch.gather(
                    prev_pos_bt, dim=2, index=anchor_idx_exp
                )  # (B,Nobj,Nq,3)

                # Step 2: Compute next positions at anchors using Verlet (with scaled acceleration)
                next_anchors = (
                    accel_scaled_denorm * dt_squared + 2 * curr_anchors - prev_anchors
                )  # (B,Nobj,Nq,3)

                # Step 3: Compute updates (delta from base prediction)
                base_anchors = torch.gather(
                    next_pos_bt, dim=2, index=anchor_idx_exp
                )  # (B,Nobj,Nq,3)
                anchor_updates = next_anchors - base_anchors  # (B,Nobj,Nq,3)

                # Mask padding objects' updates to zero
                # Padding objects have anchor_idx=0 which would scatter to wrong positions
                obj_mask_scatter = (
                    obj_mask.unsqueeze(-1).unsqueeze(-1).float()
                )  # (B,Nobj,1,1)
                anchor_updates = (
                    anchor_updates * obj_mask_scatter
                )  # padding objects -> zero updates

                # Step 4: Vectorized scatter using scatter_reduce
                # Flatten batch and object dimensions for efficient scatter
                # Use reshape() to handle non-contiguous tensors
                B_Nobj_Nv = B * Nobj * Nv
                next_pos_flat = next_pos_bt.reshape(B_Nobj_Nv, 3)  # (B*Nobj*Nv, 3)

                # Create flat indices for scatter_reduce
                # anchor_idx is (B, Nobj, Nq), we need indices in flat (B*Nobj*Nv) space
                obj_offsets = (
                    torch.arange(Nobj, device=anchor_idx.device).view(1, Nobj, 1) * Nv
                )  # (1,Nobj,1)
                batch_offsets = torch.arange(B, device=anchor_idx.device).view(
                    B, 1, 1
                ) * (
                    Nobj * Nv
                )  # (B,1,1)
                flat_anchor_idx = (
                    anchor_idx + obj_offsets + batch_offsets
                )  # (B,Nobj,Nq)
                flat_anchor_idx = flat_anchor_idx.reshape(B * Nobj * Nq)  # (B*Nobj*Nq,)

                # Flatten updates
                anchor_updates_flat = anchor_updates.reshape(
                    B * Nobj * Nq, 3
                )  # (B*Nobj*Nq, 3)

                # Apply updates using scatter_reduce (more efficient than loop)
                next_pos_flat = torch.scatter_reduce(
                    next_pos_flat,
                    dim=0,
                    index=flat_anchor_idx.unsqueeze(-1).expand(-1, 3),
                    src=anchor_updates_flat,
                    reduce="sum",
                    include_self=True,
                )

                # Reshape back
                next_pos_bt = next_pos_flat.reshape(B, Nobj, Nv, 3)

                # Per-object rigid projection using anchors (Kabsch on anchors), then apply to full Nv
                if registration_refs is None:
                    registration_refs = curr_pos_bt.detach().clone()  # (B,Nobj,Nv,3)

                # Vectorized Kabsch: Process ONLY valid objects (filter padding BEFORE SVD)
                # Step 1: Gather anchor positions for all objects at once
                anchor_idx_exp = anchor_idx.unsqueeze(-1).expand(
                    -1, -1, -1, 3
                )  # (B,Nobj,Nq,3)
                last_anchor_all = torch.gather(
                    registration_refs, dim=2, index=anchor_idx_exp
                )  # (B,Nobj,Nq,3)
                curr_anchor_all = torch.gather(
                    next_pos_bt, dim=2, index=anchor_idx_exp
                )  # (B,Nobj,Nq,3)

                # Reshape to (B*Nobj, Nq, 3) for batch processing
                B_Nobj = B * Nobj
                last_anchor_flat = last_anchor_all.view(B_Nobj, Nq, 3)
                curr_anchor_flat = curr_anchor_all.view(B_Nobj, Nq, 3)

                # Filter padding objects before SVD; they would produce garbage R/t
                obj_mask_flat = obj_mask.view(B_Nobj)  # (B*Nobj,)
                valid_idx = _kabsch_valid_idx_fwd  # indices of valid objects (fixed within the forward)
                n_valid = valid_idx.shape[0]

                # Initialize R/t for ALL objects (identity R, zero t)
                R_flat = (
                    torch.eye(3, device=device, dtype=last_anchor_flat.dtype)
                    .unsqueeze(0)
                    .expand(B_Nobj, -1, -1)
                    .clone()
                )
                t_vec_flat = torch.zeros(
                    B_Nobj, 1, 3, device=device, dtype=last_anchor_flat.dtype
                )

                if n_valid > 0:
                    # Gather only valid objects' anchors
                    last_anchor_valid = last_anchor_flat[valid_idx]  # (Nvalid, Nq, 3)
                    curr_anchor_valid = curr_anchor_flat[valid_idx]  # (Nvalid, Nq, 3)

                    if self.if_roma:
                        # RoMa mode: batch process only valid objects
                        # RoMa requires FP32 for numerical stability
                        with torch.amp.autocast("cuda", enabled=False):
                            R_valid, t_vec_valid = roma.rigid_points_registration(
                                last_anchor_valid.float(), curr_anchor_valid.float()
                            )
                        R_valid = R_valid.float()  # Ensure FP32 for downstream
                        t_vec_valid = t_vec_valid.float().unsqueeze(1)  # (Nvalid, 1, 3)

                    else:
                        # Standard SVD Kabsch: batch process only valid objects
                        centroid_last = last_anchor_valid.mean(
                            dim=1, keepdim=True
                        )  # (Nvalid,1,3)
                        centroid_curr = curr_anchor_valid.mean(
                            dim=1, keepdim=True
                        )  # (Nvalid,1,3)
                        last_c = last_anchor_valid - centroid_last
                        curr_c = curr_anchor_valid - centroid_curr

                        # Batch SVD for valid objects only
                        H = torch.bmm(last_c.transpose(1, 2), curr_c)  # (Nvalid,3,3)
                        with torch.amp.autocast("cuda", enabled=False):
                            H_float = H.float()
                            U, S, Vt = torch.linalg.svd(H_float)  # Batch SVD!
                        U = U.to(H.dtype)
                        Vt = Vt.to(H.dtype)
                        R_valid = torch.bmm(
                            Vt.transpose(1, 2), U.transpose(1, 2)
                        )  # (Nvalid,3,3)

                        # Fix reflections
                        with torch.amp.autocast("cuda", enabled=False):
                            det_R = torch.det(R_valid.float()).to(R_valid.dtype)
                        need_flip = det_R < 0
                        if need_flip.any():
                            Fm = (
                                torch.eye(3, device=R_valid.device, dtype=R_valid.dtype)
                                .unsqueeze(0)
                                .expand(n_valid, -1, -1)
                                .clone()
                            )
                            Fm[need_flip, 2, 2] = -1
                            R_valid = torch.bmm(
                                torch.bmm(Vt.transpose(1, 2), Fm), U.transpose(1, 2)
                            )

                        # Translation
                        rotated_centroid = torch.bmm(
                            R_valid, centroid_last.transpose(1, 2)
                        ).transpose(1, 2)
                        t_vec_valid = centroid_curr - rotated_centroid  # (Nvalid,1,3)

                    # Scatter valid R/t back to full tensors
                    R_flat[valid_idx] = R_valid
                    t_vec_flat[valid_idx] = t_vec_valid

                # Reshape back to (B,Nobj,3,3) and (B,Nobj,1,3)
                R_all = R_flat.view(B, Nobj, 3, 3)
                t_vec_all = t_vec_flat.view(B, Nobj, 1, 3)

                # ========== VECTORIZED TRANSFORMATION APPLICATION ==========
                # Apply R, t to all objects in a single bmm call (no per-object loop)
                # registration_refs: (B, Nobj, Nv, 3)
                # R_all: (B, Nobj, 3, 3)
                # t_vec_all: (B, Nobj, 1, 3)

                # Reshape for batch bmm: (B, Nobj, Nv, 3) -> (B*Nobj, Nv, 3)
                last_flat = registration_refs.view(B * Nobj, Nv, 3)
                R_flat = R_all.view(B * Nobj, 3, 3)
                t_flat = t_vec_all.view(B * Nobj, 1, 3)

                # Single bmm for ALL objects: y = R @ x + t (RoMa convention)
                # last_flat.transpose(1, 2): (B*Nobj, 3, Nv)
                # bmm result: (B*Nobj, 3, Nv)
                # transpose back + t: (B*Nobj, Nv, 3)
                y_flat = (
                    torch.bmm(R_flat, last_flat.transpose(1, 2)).transpose(1, 2)
                    + t_flat
                )
                # Use reshape() because transpose() makes tensor non-contiguous
                yproj_bt = y_flat.reshape(
                    B, Nobj, Nv, 3
                ).float()  # FP32 for numerical stability

                # Raw position BEFORE Kabsch/STE (for position loss supervision)
                next_pos_raw_norm = next_pos_bt

                # Apply gradient strategy based on transform type
                if self.if_roma:
                    # RoMa mode: fully differentiable, gradients flow through Kabsch
                    blended_next = yproj_bt
                else:
                    # Standard mode: straight-through estimator (STE)
                    # Use Kabsch value but only pass gradient to original prediction
                    blended_next = next_pos_bt + (yproj_bt - next_pos_bt).detach()

                # Recompute accelerations from blended positions (for acceleration_after_icp loss)
                accel_icp_denorm = (
                    blended_next - 2 * curr_pos_bt + prev_pos_bt
                ) / dt_squared
                accel_icp_norm = accel_icp_denorm

            # ========== INTERSECTION HANDLING ==========
            # Resolve object-ground and object-object penetrations (STE-differentiable)
            intersection_cfg = self.config.get("model", {}).get(
                "intersection_handling", {}
            )
            rigid_next = (
                blended_next  # Kabsch output before the ground/object correction
            )
            blended_next = handle_intersections(
                blended_next, obj_mask, vmask_t, intersection_cfg
            )

            # Update registration refs to the post-correction position (consistent with rollout); with
            # model.kabsch_reference 'fixed' every step fits the same reference mesh (the first current frame)
            if not self._kabsch_reference_fixed:
                registration_refs = blended_next.detach()

            # Store positions:
            # - STE blended positions for autoregressive rollout (more stable)
            # - Raw positions for loss supervision (direct gradient)
            # - ICP-transformed positions for position_after_icp loss
            all_pred_pos_for_rollout.append(blended_next)  # For next step input
            all_pred_pos_raw.append(next_pos_bt)  # For position_before_icp loss
            all_pred_pos_raw_norm.append(
                next_pos_raw_norm
            )  # For position_before_icp loss (normalized)
            # ICP/Kabsch-transformed positions for position_after_icp loss. after_kabsch_loss 'rigid': the Kabsch
            # output itself, the state the acceleration_after_icp loss uses (the stated after-Kabsch losses);
            # 'handled': the state after the intersection correction.
            blended_next_norm = (
                rigid_next if self._after_kabsch_loss_rigid else blended_next
            )
            all_pred_pos_icp_norm.append(
                blended_next_norm
            )  # For position_after_icp loss
            all_accel_no_icp_norm_anchor_seq.append(accel_no_icp_norm_anchor)

            # Save current features for next step's penetration prediction
            prev_obj_layers = all_obj_layers
            prev_state_layers = all_state_layers
            all_accel_icp_norm_seq.append(accel_icp_norm)

        # Stack time dimension
        # Design principle:
        # - Position outputs: RAW predictions (before Kabsch) AND ICP-transformed (after Kabsch)
        # - Acceleration outputs: before and after Kabsch transform
        # - Autoregressive rollout: uses ICP-transformed positions internally for stability
        output = {
            "obj_positions_denorm": torch.stack(
                all_pred_pos_raw, dim=1
            ),  # (B,T-2,Nobj,Nv,3) - raw (before Kabsch)
            "obj_positions_norm": torch.stack(
                all_pred_pos_raw_norm, dim=1
            ),  # (B,T-2,Nobj,Nv,3) - raw normalized (before Kabsch)
            "obj_positions_icp_norm": torch.stack(
                all_pred_pos_icp_norm, dim=1
            ),  # (B,T-2,Nobj,Nv,3) - ICP-transformed normalized (after Kabsch)
            "obj_positions_ste": torch.stack(
                all_pred_pos_for_rollout, dim=1
            ),  # (B,T-2,Nobj,Nv,3) - ICP blended (matches rollout)
            "obj_accelerations_no_icp_norm_anchor": torch.stack(
                all_accel_no_icp_norm_anchor_seq, dim=1
            ),  # before Kabsch
            "obj_accelerations_norm": torch.stack(
                all_accel_icp_norm_seq, dim=1
            ),  # after Kabsch
        }

        # Include penetration predictions for ALL vertices if collected
        penetration_enabled = self.config["model"].get(
            "if_penetration", self.config["model"].get("if_directional_distance", False)
        )
        if penetration_enabled and len(all_pen_pred_vertices_seq) > 0:
            try:
                output["obj_pen_pred_vertices"] = torch.stack(
                    all_pen_pred_vertices_seq, dim=1
                )  # (B,T-2,Nobj,Nv,1)
            except Exception as _e:
                logger.warning(f"Failed to stack pen predictions: {_e}")

        if return_all:
            # Not storing states for multi-object yet
            pass

        return output

    def rollout(
        self, initial_batch, num_steps, dt=0.01, return_extras=True, warmup_frames=2
    ):
        """
        Rollout predictions from initial conditions (multi-object).

        Args:
            initial_batch: dict from dataloader containing at least
                - obj_positions (B, T0, Nobj, Nv, 3)
                - valid_obj_mask (B, Nobj)
                - valid_vtx_mask (B, T0, Nobj, Nv)
                - ref_obj_positions (B or 1, Nobj, Nv, 3) [optional]
            num_steps: total frames to output (including provided GT warmup frames)
            dt: timestep (unused; integration handled in model forward)
            warmup_frames: number of GT warmup frames (default=2 for our model, HopNet uses 4)
        Returns:
            dict with 'obj_positions' of shape (B, num_steps, Nobj, Nv, 3)
                and optionally 'obj_body_dist_vec2' and 'obj_accelerations_anchor' if return_extras=True
        """
        self.eval()
        with torch.no_grad():
            if "obj_positions" not in initial_batch:
                raise ValueError(
                    "rollout expects multi-object 'obj_positions' in initial_batch"
                )

            obj_pos_all = initial_batch["obj_positions"]  # (B,T0,Nobj,Nv,3)
            valid_obj_mask = initial_batch["valid_obj_mask"]  # (B,Nobj)
            valid_vtx_mask_all = initial_batch["valid_vtx_mask"]  # (B,T0,Nobj,Nv)

            # Read physics parameters and time step if enabled
            phys_params = None  # (B, Nobj, 3): [mass, friction, restitution]
            skip_step_size_batch = None  # (B,)
            if self._if_phys_param:
                if "phys_params" not in initial_batch:
                    raise KeyError(
                        "rollout requires 'phys_params' in initial_batch when if_phys_param=true"
                    )
                phys_params = initial_batch["phys_params"].to(
                    obj_pos_all.device
                )  # (B, Nobj, 3)
            if self._if_time_step:
                if "skip_step_size" not in initial_batch:
                    raise KeyError(
                        "rollout requires 'skip_step_size' in initial_batch when if_time_step=true"
                    )
                skip_step_size_batch = initial_batch["skip_step_size"].to(
                    obj_pos_all.device
                )  # (B,)

                # Optional debug logging of the rollout step size
                if_debug = (
                    self.config.get("data", {})
                    .get("random_step_size", {})
                    .get("if_debug", False)
                )
                if if_debug:
                    logger.info(
                        f"[DEBUG Rollout] step_size for rollout: {skip_step_size_batch.cpu().numpy().tolist()}, num_steps={num_steps}"
                    )

                # Validate step_size is within training range
                # Check if multi-timescale training is enabled and validate step_size
                random_step_config = self.config.get("data", {}).get(
                    "random_step_size", {}
                )
                if random_step_config.get("enable", False):
                    trained_step_sizes = random_step_config.get(
                        "step_sizes", [1]
                    )  # e.g., [1, 2, 5, 10]
                    for b_idx, step_size_val in enumerate(
                        skip_step_size_batch.cpu().numpy()
                    ):
                        if step_size_val not in trained_step_sizes:
                            logger.warning(
                                f"[Rollout Warning] Sample {b_idx}: step_size={step_size_val} is NOT in trained range {trained_step_sizes}!\n"
                                f"Model was only trained on step_sizes {trained_step_sizes}. Prediction may be inaccurate due to extrapolation."
                            )
                        elif if_debug:
                            logger.info(
                                f"[Rollout Validation] Sample {b_idx}: step_size={step_size_val} is within trained range {trained_step_sizes}"
                            )

            B, T0, Nobj, Nv, _ = obj_pos_all.shape
            device = obj_pos_all.device

            # Require at least warmup_frames GT frames (minimum 2 for Verlet initialization)
            min_required = max(2, warmup_frames)
            if T0 < min_required:
                raise ValueError(
                    f"rollout requires at least {min_required} GT frames (warmup_frames={warmup_frames}), got {T0}"
                )

            # Reference pose: MUST be global frame 0 (same as training)
            if "ref_obj_positions" not in initial_batch:
                raise ValueError(
                    "rollout requires 'ref_obj_positions' (global frame 0) in initial_batch"
                )
            ref_pose = initial_batch["ref_obj_positions"]  # (B?,Nobj,Nv,3)
            if ref_pose.dim() == 3:
                ref_pose = ref_pose.unsqueeze(0).expand(B, -1, -1, -1).to(device)
            else:
                ref_pose = ref_pose.to(device)

            # ========== Center scene by first frame (eval/rollout alignment) ==========
            # If enabled, subtract the global center of the first frame from all coordinates
            # so the rollout coordinate system matches the training setup.
            # center_t0 is stored per sample so visualization can apply the same shift to disk-loaded GT.
            _center_t0_per_sample = (
                []
            )  # Store per-sample center for visualization GT alignment
            if self.config.get("data", {}).get("center_scene_by_t0", False):
                for b in range(B):
                    vmask_b0 = valid_vtx_mask_all[b, 0]  # (Nobj, Nv)
                    v_t0 = obj_pos_all[b, 0][vmask_b0]  # Valid vertices at t=0
                    if v_t0.shape[0] > 0:
                        center_t0 = v_t0.mean(dim=0)  # (3,)
                        _center_t0_per_sample.append(center_t0.cpu().numpy())

                        logger.info(
                            f"[Center-T0 ROLLOUT] b={b}: center_t0={center_t0.cpu().numpy().tolist()}, "
                            f"obj_pos range before: [{obj_pos_all[b].min().item():.4f}, {obj_pos_all[b].max().item():.4f}]"
                        )

                        # Shift positions
                        obj_pos_all[b] = obj_pos_all[b] - center_t0
                        ref_pose[b] = ref_pose[b] - center_t0

                        logger.info(
                            f"[Center-T0 ROLLOUT] b={b} after: "
                            f"obj_pos range: [{obj_pos_all[b].min().item():.4f}, {obj_pos_all[b].max().item():.4f}]"
                        )
                    else:
                        _center_t0_per_sample.append(None)
                logger.info(
                    f"[Center-T0 ROLLOUT] Applied centering to {len(_center_t0_per_sample)} samples"
                )

            # Require anchors for query and supervision (no fallback)
            if "anchor_idx" not in initial_batch:
                raise KeyError(
                    "rollout requires 'anchor_idx' in initial_batch for anchor-based querying"
                )
            anchor_idx = initial_batch["anchor_idx"].to(device).long()  # (B,Nobj,Nq)
            if (
                anchor_idx.dim() != 3
                or anchor_idx.shape[0] != B
                or anchor_idx.shape[1] != Nobj
            ):
                raise ValueError(
                    f"anchor_idx must be (B,Nobj,Nq); got {tuple(anchor_idx.shape)} vs (B={B},Nobj={Nobj})"
                )
            Nq = int(anchor_idx.shape[2])

            # Initialize previous/current positions (use last 2 frames of warmup for Verlet)
            # For warmup_frames=2: use frame 0, 1
            # For warmup_frames=4: use frame 2, 3 (GT frames 0,1,2,3 then predict from frame 4)
            prev_idx = warmup_frames - 2
            curr_idx = warmup_frames - 1
            prev_pos_bt = obj_pos_all[:, prev_idx]  # (B,Nobj,Nv,3)
            curr_pos_bt = obj_pos_all[:, curr_idx]
            registration_refs = curr_pos_bt.detach().clone()

            # Initialize persistent state tokens (carry across steps like training)
            # When no_global_state=True, pass None to decoder
            if self._no_global_state:
                state_tokens = None
                state_pos = None
            else:
                state_tokens, state_pos = self.state_repr.get_state_and_positions(
                    B, device
                )

            # Collect frames: include ALL GT warmup frames first
            out_seq = [obj_pos_all[:, i] for i in range(warmup_frames)]

            # Steps to predict beyond the provided GT frames in the output horizon
            remaining = max(0, num_steps - warmup_frames)

            # Collect per-step input body_dist_vec2 for visualization (curr frame at each step)
            # Only collect if return_extras=True (video gen) to save memory during RMSE eval
            all_dist_vec2_seq = (
                [] if return_extras else None
            )  # list of (B, Nobj, Nv, 3)
            # Collect predicted anchor accelerations (denormalized) for visualization
            all_accel_anchor_denorm_seq = (
                [] if return_extras else None
            )  # list of (B, Nobj, Nq, 3)

            # Previous step's features for penetration prediction (step>0)
            prev_obj_layers_rollout = None
            prev_state_layers_rollout = None

            # Optional CUDA-graph fast path (RIGIDFORMER_CUDA_GRAPH=1): after the
            # early steps whose vertex-mask index is still advancing, every step
            # executes an identical op stream on fixed shapes, so one step is
            # captured as a CUDA graph and replayed for the remainder. Cross-step
            # tensors are kept in fixed-address buffers that the graph itself
            # rotates, which makes each replay self-contained.
            use_cuda_graph = (
                os.environ.get("RIGIDFORMER_CUDA_GRAPH", "0") == "1"
                and not return_extras
                and torch.cuda.is_available()
            )
            # Constant scalars used inside the step; created once so the step body
            # performs no host-to-device transfers (required for graph capture).
            _attn_zero = torch.tensor(0.0, dtype=torch.float32, device=device)
            _attn_ninf = torch.tensor(float("-inf"), dtype=torch.float32, device=device)
            # Kabsch padding filter: valid_obj_mask is fixed for the whole rollout,
            # so the valid-object index set is computed once (nonzero would force a
            # host sync every step and cannot run under graph capture).
            _kabsch_mask_flat = valid_obj_mask.reshape(B * Nobj)
            _kabsch_valid_idx = _kabsch_mask_flat.nonzero(as_tuple=True)[0]
            _kabsch_n_valid = int(_kabsch_valid_idx.shape[0])
            use_cuda_graph = use_cuda_graph and bool(getattr(self, "if_roma", False))
            graph_obj = None
            graph_obj2 = None
            graph_capture_step = T0 - 1  # first step index with the saturated mask
            if use_cuda_graph and remaining <= graph_capture_step + 1:
                use_cuda_graph = False

            # Progress bar for rollout (only show on rank 0)
            import torch.distributed as dist
            from tqdm import tqdm

            show_pbar = True
            if dist.is_available() and dist.is_initialized():
                show_pbar = dist.get_rank() == 0

            # Object slots holding a valid object in at least one sample; fixed for the rollout, so
            # computed once here (a host sync outside the step loop and any graph capture).
            _knn_active_slots_r = (
                torch.nonzero(valid_obj_mask.any(dim=0)).flatten().tolist()
                if self._knn_num_groups <= 1
                else None
            )
            # Valid-vertex packs, one per mask index the loop uses (min(warmup_frames - 1 + step, T0 - 1)); built here, outside
            # the step loop and any graph capture.
            _knn_packs_r, _enc_packs_r = {}, {}
            if self._knn_packed or self._encoder_packed:
                _vr_r = initial_batch.get("vtx_rank")
                for _mi in range(1, valid_vtx_mask_all.shape[1]):
                    _vm_r = valid_vtx_mask_all[:, _mi]
                    if self._knn_packed:
                        _knn_packs_r[_mi] = knn_pack(_vm_r, valid_obj_mask)
                    if self._encoder_packed:
                        _enc_packs_r[_mi] = EncoderPack(
                            _vm_r.reshape(B * Nobj, Nv),
                            _vr_r.reshape(B * Nobj, Nv) if _vr_r is not None else None,
                            len(self.shared_obj_encoder.hierarchical_layers),
                            self.shared_obj_encoder.hierarchical_subsets,
                            1,
                            small=self.shared_obj_encoder.subset_small_objects,
                        )

            # Anchor validation. Anchors and vertex masks are fixed for the whole
            # rollout, so every per-step check is equivalent to checking each mask
            # index once up front; this keeps the step loop free of host syncs.
            for _mask_idx in range(1, T0):
                for i in range(Nobj):
                    vmask_curr = valid_vtx_mask_all[:, _mask_idx, i]  # (B,Nv)
                    idx_i_chk = anchor_idx[:, i]  # (B,Nq)
                    mask_at_anchors = torch.gather(vmask_curr, dim=1, index=idx_i_chk)
                    obj_valid_b = valid_obj_mask[:, i]  # (B,)
                    num_valid = vmask_curr.sum(dim=1)  # (B,)
                    if int(((num_valid == 0) & obj_valid_b).sum().item()) > 0:
                        raise RuntimeError(
                            f"Zero valid vertices for rollout at mask_idx={_mask_idx}, obj={i} for valid samples"
                        )
                    need = (num_valid < Nq) & obj_valid_b
                    if bool(need.any()):
                        min_valid = (
                            int(num_valid[obj_valid_b].min().item())
                            if bool(obj_valid_b.any())
                            else 0
                        )
                        raise RuntimeError(
                            f"Not enough valid vertices for rollout at mask_idx={_mask_idx}, obj={i}: min_valid={min_valid} < anchor_k={Nq}"
                        )
                    bad_mask = (~mask_at_anchors) & obj_valid_b.unsqueeze(1)
                    if bool(bad_mask.any()):
                        bad = bad_mask.sum().item()
                        raise RuntimeError(
                            f"Invalid anchors for rollout: {bad} indices invalid at mask_idx={_mask_idx}, obj={i}."
                        )

            pbar = tqdm(
                range(remaining), desc="Rollout", disable=not show_pbar, leave=False
            )
            for step in pbar:
                rope_memo_reset()  # rotary angles are memoized within one step only (keeps graph capture self-contained)
                _capturing = use_cuda_graph and step == graph_capture_step
                if _capturing:
                    # Move cross-step carries into fixed-address buffers, then
                    # record this step. Kernels are only recorded during capture,
                    # so the step is materialized by an immediate replay below.
                    prev_pos_bt = prev_pos_bt.clone()
                    curr_pos_bt = curr_pos_bt.clone()
                    registration_refs = registration_refs.clone()
                    if prev_obj_layers_rollout is not None:
                        prev_obj_layers_rollout = [
                            x.clone() for x in prev_obj_layers_rollout
                        ]
                    if prev_state_layers_rollout is not None:
                        prev_state_layers_rollout = [
                            x.clone() for x in prev_state_layers_rollout
                        ]
                    torch.cuda.synchronize()
                    graph_obj = torch.cuda.CUDAGraph()
                    _graph_stream = torch.cuda.Stream()
                    _prev_stream = torch.cuda.current_stream()
                    torch.cuda.set_stream(_graph_stream)
                    # thread_local: NCCL watchdog / DataLoader threads must not
                    # poison the capture under DDP evaluation.
                    graph_obj.capture_begin(capture_error_mode="thread_local")
                # ========== STATE RESET (optional, to avoid train-test mismatch) ==========
                # When rollout_state_reset=True, reset state tokens at each step to match
                # training distribution (training only sees ~2 state updates per sequence).
                # This prevents state drift during long rollouts (e.g., 400 frames).
                if self._rollout_state_reset:
                    if self._no_global_state:
                        state_tokens = None
                        state_pos = None
                    else:
                        state_tokens, state_pos = (
                            self.state_repr.get_state_and_positions(B, device)
                        )

                # Compute velocities from last two frames
                vel_bt = curr_pos_bt - prev_pos_bt  # (B,Nobj,Nv,3)

                # Prepare far-away filler for invalid vertices in reference lists
                FAR = 1e6

                mask_idx_curr = min(
                    warmup_frames - 1 + step, T0 - 1
                )  # current frame's mask

                # ========== VECTORIZED KNN FOR ALL OBJECTS AT ONCE ==========
                # curr_pos_bt: (B, Nobj, Nv, 3)
                # valid_vtx_mask_all[:, mask_idx_curr]: (B, Nobj, Nv)
                vmask_step = valid_vtx_mask_all[:, mask_idx_curr]  # (B, Nobj, Nv)

                # Compute KNN distance vectors for ALL objects simultaneously
                # Only computes for valid objects and valid vertices!
                if self._knn_packed:
                    dist_vec_all = compute_knn_packed(
                        curr_pos_bt, _knn_packs_r[mask_idx_curr]
                    )
                else:
                    dist_vec_all = compute_knn_all_objects_vectorized(
                        curr_pos_bt,
                        vmask_step,
                        FAR,
                        valid_obj_mask=valid_obj_mask,
                        num_groups=self._knn_num_groups,
                        active_slots=_knn_active_slots_r,
                    )

                # Apply plane constraint adjustment (vectorized)
                # Use reshape() instead of view() to handle non-contiguous tensors
                curr_pos_flat = curr_pos_bt.reshape(B, Nobj * Nv, 3)
                dist_vec_flat = dist_vec_all.reshape(B, Nobj * Nv, 3)
                dist_vec2_flat = distance_2_plane(
                    curr_pos_flat, [0, 0, 0], "z", dist_vec_flat
                )
                dist_vec2_bt = dist_vec2_flat.reshape(B, Nobj, Nv, 3)

                # Store raw dist_vec2 for visualization (only if return_extras=True)
                if return_extras:
                    all_dist_vec2_seq.append(dist_vec2_bt)

                # Relative position (vectorized): curr_pos - ref_pose
                relpos_cat = curr_pos_bt - ref_pose  # (B, Nobj, Nv, 3)

                # Velocity (already in correct shape)
                vel_cat = vel_bt  # (B, Nobj, Nv, 3)

                # Dist vectors
                distvec_cat = dist_vec2_bt  # (B, Nobj, Nv, 3)

                # Get current step's vertex mask for all objects (vectorized)
                mask_idx_curr_all = mask_idx_curr
                vmask_curr_all = valid_vtx_mask_all[:, mask_idx_curr_all]  # (B,Nobj,Nv)
                vmask_curr_all_3d = vmask_curr_all.unsqueeze(-1)  # (B,Nobj,Nv,1)

                # Apply masks BEFORE normalization (vectorized across all objects)
                distvec_cat = torch.where(
                    vmask_curr_all_3d, distvec_cat, torch.zeros_like(distvec_cat)
                )
                vel_cat = torch.where(
                    vmask_curr_all_3d, vel_cat, torch.zeros_like(vel_cat)
                )
                relpos_cat = torch.where(
                    vmask_curr_all_3d, relpos_cat, torch.zeros_like(relpos_cat)
                )

                # Apply network-predicted directional sign to ALL vertices' dist_vec (before normalization)
                # Only for step>0 (student forcing steps with predicted positions, may have penetration)
                # For step=0, use GT positions from warmup (no penetration, sign=+1)
                penetration_enabled = self.config["model"].get(
                    "if_penetration",
                    self.config["model"].get("if_directional_distance", False),
                )
                if penetration_enabled and step > 0:
                    # Predict penetration sign using PREVIOUS step's features and CURRENT predicted positions
                    # curr_pos_bt = blended_next (AFTER Kabsch + intersection handling)
                    pen_pred_all_vertices = self.predictor.predict_penetration_vertices(
                        prev_obj_layers_rollout,  # Use previous step's decoder outputs
                        prev_state_layers_rollout,  # Use previous step's state features
                        pos_full=curr_pos_bt,  # AFTER Kabsch transform (matches training forward)
                    )  # (B, Nobj, Nv, 1) in [-1,1]

                    # Apply predicted sign to dist_vec for ALL vertices
                    pen_sign_all = torch.sign(
                        pen_pred_all_vertices + 1e-6
                    )  # (B, Nobj, Nv, 1) in {-1,+1}
                    distvec_cat = distvec_cat * pen_sign_all  # (B, Nobj, Nv, 3)

                # No normalization — raw values, matching the forward path
                # (features are masked once, then the penetration sign is applied)

                # Broadcast physics parameters and time_step to vertex dimension if enabled (rollout)
                phys_broadcast = None  # (B, Nobj, Nv, 3)
                time_broadcast = None  # (B, Nobj, Nv, 2)

                if self._if_phys_param:
                    # Broadcast per-object physics params (B,Nobj,3) to all vertices (B,Nobj,Nv,3)
                    phys_broadcast = phys_params.unsqueeze(2).expand(
                        B, Nobj, Nv, 3
                    )  # (B,Nobj,Nv,3)
                    # Mask with vertex mask to ensure padding vertices get zero physics params
                    phys_broadcast = torch.where(
                        vmask_curr_all_3d,
                        phys_broadcast,
                        torch.zeros_like(phys_broadcast),
                    )

                if self._if_time_step:
                    # Broadcast both s and s² to all objects and vertices (B,Nobj,Nv,2)
                    # Must match the forward path
                    step_size_s = skip_step_size_batch.float()  # (B,)
                    step_size_sq = (skip_step_size_batch**2).float()  # (B,)
                    time_features = torch.stack(
                        [step_size_s, step_size_sq], dim=-1
                    )  # (B, 2)
                    time_broadcast = time_features.view(B, 1, 1, 2).expand(
                        B, Nobj, Nv, 2
                    )  # (B,Nobj,Nv,2)
                    # Mask for consistency
                    # vmask_curr_all_3d is (B,Nobj,Nv,1), expand to match (B,Nobj,Nv,2)
                    vmask_time = vmask_curr_all_3d.expand(
                        -1, -1, -1, 2
                    )  # (B,Nobj,Nv,2)
                    time_broadcast = torch.where(
                        vmask_time, time_broadcast, torch.zeros_like(time_broadcast)
                    )

                # PointNet Encoder: process each object's Nv vertices -> per-object global feature
                # Input dimension depends on flags:
                #   Base: 9D (dist_vec + velocity + rel_pos)
                #   +if_phys_param=true: +3D
                #   +if_time_step=true (no FiLM): +2D (s and s²)
                # Split into components and reshape to (B*Nobj, Nv, D) for encoder
                dist_reshaped = distvec_cat.view(B * Nobj, Nv, 3)
                vel_reshaped = vel_cat.view(B * Nobj, Nv, 3)
                relpos_reshaped = relpos_cat.view(B * Nobj, Nv, 3)

                # Optional features

                phys_reshaped = None
                if self._if_phys_param:
                    phys_reshaped = phys_broadcast.view(B * Nobj, Nv, 3)

                time_reshaped = None
                if self._if_time_step and not self._film_conditioning:
                    time_reshaped = time_broadcast.view(B * Nobj, Nv, 2)  # (s, s²)

                # Prepare vertex mask for encoder pooling
                # vmask_curr_all_3d is (B,Nobj,Nv,1), need (B*Nobj, Nv)
                vtx_mask_for_encoder = vmask_curr_all_3d.squeeze(-1).reshape(
                    B * Nobj, Nv
                )  # (B*Nobj, Nv)
                height_reshaped = None
                if self.shared_obj_encoder.height_input:
                    _h = torch.where(
                        vmask_curr_all_3d,
                        curr_pos_bt[..., 2:3],
                        torch.zeros_like(curr_pos_bt[..., 2:3]),
                    )
                    height_reshaped = _h.reshape(B * Nobj, Nv, 1)
                rank_reshaped = None
                if self.shared_obj_encoder.hierarchical_subsets:
                    if "vtx_rank" not in initial_batch:
                        raise KeyError(
                            "model.encoder.hierarchical_subsets requires data.fps_order: true (batch lacks 'vtx_rank')"
                        )
                    rank_reshaped = initial_batch["vtx_rank"].reshape(B * Nobj, Nv)

                # Call encoder with all features (encoder will concatenate them internally)
                # AVP: capture per-point features if AVP is enabled
                obj_features_encoded, _per_vtx_feat, _ = self.shared_obj_encoder(
                    dist_reshaped,
                    vel_reshaped,
                    relpos_reshaped,
                    phys_params=phys_reshaped,
                    time_step=time_reshaped,
                    valid_mask=vtx_mask_for_encoder,
                    height=height_reshaped,
                    vtx_rank=rank_reshaped,
                    pack=(
                        _enc_packs_r.get(mask_idx_curr_all)
                        if self._encoder_packed
                        else None
                    ),
                )  # obj_features_encoded: (B*Nobj, enc_out_dim), _per_vtx_feat: (B*Nobj, backbone_dim, Nv)

                # AVP: Anchor-Vertex Pooling — enrich anchor queries with collision-aware features
                _avp_anchor_features = (
                    None  # Will be (B, Nobj, Nq, avp_output_dim) if AVP enabled
                )
                if self.avp is not None and _per_vtx_feat is not None:
                    # _per_vtx_feat is (B*Nobj, backbone_dim, Nv) — Conv1d channels-first format
                    # Transpose to (B*Nobj, Nv, backbone_dim) for AVP
                    _pvf = _per_vtx_feat.transpose(1, 2)  # (B*Nobj, Nv, backbone_dim)
                    # Anchor positions in flat format
                    _curr_pos_flat = curr_pos_bt.reshape(
                        B * Nobj, Nv, 3
                    )  # (B*Nobj, Nv, 3)
                    _anchor_pos_flat = torch.gather(
                        _curr_pos_flat,
                        dim=1,
                        index=anchor_idx.reshape(B * Nobj, -1)
                        .unsqueeze(-1)
                        .expand(-1, -1, 3),
                    )  # (B*Nobj, Nq, 3)
                    _avp_anchor_features = self.avp(
                        _pvf, _curr_pos_flat, _anchor_pos_flat, vtx_mask_for_encoder
                    )  # (B*Nobj, Nq, avp_output_dim)
                    _avp_anchor_features = _avp_anchor_features.view(
                        B, Nobj, anchor_idx.shape[2], -1
                    )  # (B, Nobj, Nq, avp_output_dim)
                    # Zero out padding objects (proj bias makes padding non-zero; consistent with other features)
                    _avp_anchor_features = (
                        _avp_anchor_features
                        * valid_obj_mask.unsqueeze(-1).unsqueeze(-1).float()
                    )

                obj_features_encoded = obj_features_encoded.view(
                    B, Nobj, -1
                )  # (B, Nobj, enc_out_dim)

                # Project to token dimension for Decoder
                obj_features = self.obj_feature_proj(
                    obj_features_encoded
                )  # (B, Nobj, D)

                # FiLM: modulate encoder features with step-size [s, s²]
                if self.encoder_film is not None and skip_step_size_batch is not None:
                    time_feat_film = torch.stack(
                        [
                            skip_step_size_batch.float(),
                            (skip_step_size_batch**2).float(),
                        ],
                        dim=-1,
                    )  # (B, 2)
                    obj_features = self.encoder_film(
                        obj_features, time_feat_film
                    )  # (B, Nobj, D)

                # Object anchor positions for RoPE
                # Same approach as forward(): gather anchors, flatten to multi-point for RoPE.
                # Replace padding objects' positions with zeros before the gather:
                # padding objects have anchor_idx=0, which would gather arbitrary values into RoPE inputs.
                obj_mask_pos = valid_obj_mask.unsqueeze(-1).unsqueeze(
                    -1
                )  # (B, Nobj, 1, 1)
                safe_curr_pos_bt = torch.where(
                    obj_mask_pos, curr_pos_bt, torch.zeros_like(curr_pos_bt)
                )

                idx_exp_3_for_decoder = anchor_idx.unsqueeze(-1).expand(
                    -1, -1, -1, 3
                )  # (B, Nobj, Nq, 3)
                obj_anchor_positions = torch.gather(
                    safe_curr_pos_bt, dim=2, index=idx_exp_3_for_decoder
                )  # (B, Nobj, Nq, 3)
                # Flatten to (B, Nobj, Nq*3) for multi-point RoPE
                Nq = anchor_idx.shape[2]  # Number of anchors per object

                # ============== PE Type Routing (rollout) ==============
                # valid_vtx_mask_t is needed for geometry encoder
                valid_vtx_mask_t = valid_vtx_mask_all[:, mask_idx_curr]  # (B, Nobj, Nv)
                if self._pe_type in ("arope", "se3"):
                    # SE(3) uses arope anchors for cross-attention RoPE; SE(3) bias computed separately
                    obj_anchor_flat = obj_anchor_positions.view(
                        B, Nobj, Nq * 3
                    )  # (B, Nobj, 12)
                elif self._pe_type in ("obb", "pca"):
                    obj_anchor_flat = self._geometry_encoder(
                        safe_curr_pos_bt, valid_vtx_mask_t, obj_mask=valid_obj_mask
                    )  # (B, Nobj, 12)
                else:
                    raise ValueError(f"Unknown pe_type='{self._pe_type}' in rollout().")

                # ============== SE(3) Pairwise Bias Computation (rollout) ==============
                _se3_bias = None
                if self._pe_type == "se3" and self._se3_bias_module is not None:
                    from embodichain_models.dynamics.rigidformer._native.models.se3_attention import (
                        extract_object_pose_features,
                    )

                    _rotmats, _centers, _scales = extract_object_pose_features(
                        safe_curr_pos_bt, ref_pose, valid_vtx_mask_t, valid_obj_mask
                    )
                    _se3_bias = self._se3_bias_module(
                        _rotmats, _centers, _scales, valid_obj_mask
                    )

                # Decoder: SDPA additive mask + feature zeroing (same as forward)
                obj_attn_mask = torch.where(
                    valid_obj_mask.unsqueeze(1).unsqueeze(1), _attn_zero, _attn_ninf
                )
                pad_mask_float = valid_obj_mask.unsqueeze(-1).to(
                    dtype=obj_features.dtype
                )
                obj_features = obj_features * pad_mask_float
                obj_anchor_flat = obj_anchor_flat * valid_obj_mask.unsqueeze(-1).to(
                    dtype=obj_anchor_flat.dtype
                )

                # Per-layer FiLM: pass time_features for step-size conditioning in each layer
                _decoder_time_feat = (
                    time_feat_film
                    if (self._film_conditioning and self._if_time_step)
                    else None
                )
                all_state_layers, all_obj_layers = self.decoder(
                    state_tokens,
                    obj_features,
                    state_pos,
                    obj_anchor_flat,
                    mask=obj_attn_mask,
                    se3_bias=_se3_bias,
                    time_features=_decoder_time_feat,  # Per-layer FiLM
                )
                # No need to remove any tokens - all_obj_layers directly contains object features

                # Update state (skip when no_global_state)
                if not self._no_global_state:
                    state_tokens = all_state_layers[-1]

                # Build anchor query for Predictor (sample at anchor points Nq)
                # Query dimension depends on flags:
                #   Base: 9D (dist_vec + vel + rel_pos)
                #   +if_phys_param=true: +3D = 12D
                #   +if_time_step=true (no FiLM): +2D (s and s²)
                idx_exp_3 = anchor_idx.unsqueeze(-1).expand(
                    -1, -1, -1, 3
                )  # (B,Nobj,Nq,3)
                distvec_at_anchors = torch.gather(distvec_cat, dim=2, index=idx_exp_3)
                vel_at_anchors = torch.gather(vel_cat, dim=2, index=idx_exp_3)
                relpos_at_anchors = torch.gather(relpos_cat, dim=2, index=idx_exp_3)

                # Build query features list
                query_features = [distvec_at_anchors, vel_at_anchors, relpos_at_anchors]

                # Add physics parameters at anchors if enabled
                if self._if_phys_param:
                    phys_at_anchors = torch.gather(
                        phys_broadcast, dim=2, index=idx_exp_3
                    )  # (B,Nobj,Nq,3)
                    query_features.append(phys_at_anchors)

                # Add time step at anchors if enabled AND not using FiLM
                if self._if_time_step and not self._film_conditioning:
                    idx_exp_2 = anchor_idx.unsqueeze(-1).expand(
                        -1, -1, -1, 2
                    )  # (B,Nobj,Nq,2)
                    time_at_anchors = torch.gather(
                        time_broadcast, dim=2, index=idx_exp_2
                    )  # (B,Nobj,Nq,2)
                    query_features.append(time_at_anchors)

                # AVP: Append collision-aware anchor features to query
                if _avp_anchor_features is not None:
                    query_features.append(
                        _avp_anchor_features
                    )  # (B, Nobj, Nq, avp_output_dim)

                # Concatenate all query features
                anchor_query = torch.cat(
                    query_features, dim=-1
                )  # (B,Nobj,Nq, 9/12 + avp_dim)

                # Extract anchor positions for RoPE (current frame positions at anchors)
                anchor_pos_at_curr = torch.gather(
                    curr_pos_bt, dim=2, index=idx_exp_3
                )  # (B,Nobj,Nq,3)

                # Penetration sign was already applied to all vertices' dist_vec;
                # the anchor query therefore carries the sign information as well.

                _ref_src = (
                    _object_centered(ref_pose, vmask_step.bool())
                    if self._ref_pose_object_centered
                    else ref_pose
                )
                if self._ref_pose_object_centered and not getattr(
                    self, "_refpose_logged_roll", False
                ):
                    self._refpose_logged_roll = True
                    print(
                        f"[RefPose] object-centred (x/y only, z absolute) reference pose ACTIVE (rollout): |shift| max={((ref_pose - _ref_src).abs().max()).item():.3f} m",
                        flush=True,
                    )
                ref_pose_at_anchors = torch.gather(
                    _ref_src, dim=2, index=idx_exp_3
                )  # (B,Nobj,Nq,3)

                # Predict accelerations at anchors using grouped format (B, Nobj, Nq, *)
                accel_norm_anchor = self.predictor(
                    all_obj_layers,
                    anchor_query,  # (B, Nobj, Nq, 9/12)
                    ref_pose=ref_pose_at_anchors,  # (B, Nobj, Nq, 3)
                    query_pos=anchor_pos_at_curr,  # (B, Nobj, Nq, 3)
                    key_pos=obj_anchor_flat,  # (B, Nobj, Nq*3) - multi-point RoPE
                    multi_scale_state_features=(
                        None if self._no_global_state else all_state_layers
                    ),
                    state_pos=None if self._no_global_state else state_pos,
                    obj_mask=valid_obj_mask,  # Object validity mask for padding
                )  # Returns (B, Nobj, Nq, 3)

                # Force FP32 for Verlet integration and position computation
                # (must match the training forward path for numerical stability)
                with torch.amp.autocast("cuda", enabled=False):
                    accel_base_denorm = accel_norm_anchor.float()

                    # Explicit step_size² scaling for multi-timescale rollout
                    # (must match the forward path)
                    if self._if_time_step and skip_step_size_batch is not None:
                        step_size_sq = (skip_step_size_batch.float() ** 2).view(
                            B, 1, 1, 1
                        )  # (B,1,1,1) FP32
                        accel_scaled_denorm = (
                            accel_base_denorm * step_size_sq
                        )  # FP32 * FP32 = FP32
                    else:
                        accel_scaled_denorm = accel_base_denorm

                    # Store scaled acceleration for visualization (matches forward behavior)
                    if return_extras:
                        all_accel_anchor_denorm_seq.append(accel_scaled_denorm)

                    # Verlet integration on anchors; initialize full next positions as zero-accel extrapolation
                    dt_squared = self.dt_verlet * self.dt_verlet
                    # Positions must be FP32 for numerical stability
                    curr_pos_bt_f = curr_pos_bt.float()
                    prev_pos_bt_f = prev_pos_bt.float()
                    next_pos_bt = (
                        2 * curr_pos_bt_f - prev_pos_bt_f
                    )  # (B,Nobj,Nv,3) - FP32

                    # Vectorized scatter using scatter_reduce (same optimization as forward)
                    # Step 1: Gather all anchor positions at once (use FP32 positions)
                    anchor_idx_exp = anchor_idx.unsqueeze(-1).expand(
                        -1, -1, -1, 3
                    )  # (B,Nobj,Nq,3)
                    curr_anchors = torch.gather(
                        curr_pos_bt_f, dim=2, index=anchor_idx_exp
                    )  # (B,Nobj,Nq,3)
                    prev_anchors = torch.gather(
                        prev_pos_bt_f, dim=2, index=anchor_idx_exp
                    )  # (B,Nobj,Nq,3)

                    # Step 2: Compute next positions at anchors using Verlet (with scaled acceleration)
                    next_anchors = (
                        accel_scaled_denorm * dt_squared
                        + 2 * curr_anchors
                        - prev_anchors
                    )  # (B,Nobj,Nq,3)

                    # Step 3: Compute updates (delta from base prediction)
                    base_anchors = torch.gather(
                        next_pos_bt, dim=2, index=anchor_idx_exp
                    )  # (B,Nobj,Nq,3)
                    anchor_updates = next_anchors - base_anchors  # (B,Nobj,Nq,3)

                    # Mask padding objects' updates to zero
                    # Padding objects have anchor_idx=0 which would scatter to wrong positions
                    obj_mask_scatter = (
                        valid_obj_mask.unsqueeze(-1).unsqueeze(-1).float()
                    )  # (B,Nobj,1,1)
                    anchor_updates = (
                        anchor_updates * obj_mask_scatter
                    )  # padding objects -> zero updates

                    # Step 4: Vectorized scatter using scatter_reduce
                    # Flatten batch and object dimensions for efficient scatter
                    # Use reshape() to handle non-contiguous tensors
                    B_Nobj_Nv = B * Nobj * Nv
                    next_pos_flat = next_pos_bt.reshape(B_Nobj_Nv, 3)  # (B*Nobj*Nv, 3)

                    # Create flat indices for scatter_reduce
                    obj_offsets = (
                        torch.arange(Nobj, device=anchor_idx.device).view(1, Nobj, 1)
                        * Nv
                    )  # (1,Nobj,1)
                    batch_offsets = torch.arange(B, device=anchor_idx.device).view(
                        B, 1, 1
                    ) * (
                        Nobj * Nv
                    )  # (B,1,1)
                    flat_anchor_idx = (
                        anchor_idx + obj_offsets + batch_offsets
                    )  # (B,Nobj,Nq)
                    flat_anchor_idx = flat_anchor_idx.reshape(
                        B * Nobj * Nq
                    )  # (B*Nobj*Nq,)

                    # Flatten updates
                    anchor_updates_flat = anchor_updates.reshape(
                        B * Nobj * Nq, 3
                    )  # (B*Nobj*Nq, 3)

                    # Apply updates using scatter_reduce (more efficient than loop)
                    next_pos_flat = torch.scatter_reduce(
                        next_pos_flat,
                        dim=0,
                        index=flat_anchor_idx.unsqueeze(-1).expand(-1, 3),
                        src=anchor_updates_flat,
                        reduce="sum",
                        include_self=True,
                    )

                    # Reshape back
                    next_pos_bt = next_pos_flat.reshape(B, Nobj, Nv, 3)

                # Per-object rigid projection using anchors; apply to full Nv
                # Vectorized Kabsch: Process ONLY valid objects (filter padding BEFORE SVD)
                # Step 1: Gather anchor positions for all objects at once
                anchor_idx_exp = anchor_idx.unsqueeze(-1).expand(
                    -1, -1, -1, 3
                )  # (B,Nobj,Nq,3)
                last_anchor_all = torch.gather(
                    registration_refs, dim=2, index=anchor_idx_exp
                )  # (B,Nobj,Nq,3)
                curr_anchor_all = torch.gather(
                    next_pos_bt, dim=2, index=anchor_idx_exp
                )  # (B,Nobj,Nq,3)

                # Reshape to (B*Nobj, Nq, 3) for batch processing
                B_Nobj = B * Nobj
                last_anchor_flat = last_anchor_all.view(B_Nobj, Nq, 3)
                curr_anchor_flat = curr_anchor_all.view(B_Nobj, Nq, 3)

                # Filter padding objects before SVD; they would produce garbage R/t
                # (indices precomputed before the loop; the mask never changes)
                valid_idx = _kabsch_valid_idx
                n_valid = _kabsch_n_valid

                # Initialize R/t for ALL objects (identity R, zero t)
                # FP32 so the RoMa FP32 output is not downcast during scatter
                R_flat = (
                    torch.eye(3, device=device, dtype=torch.float32)
                    .unsqueeze(0)
                    .expand(B_Nobj, -1, -1)
                    .clone()
                )
                t_vec_flat = torch.zeros(
                    B_Nobj, 1, 3, device=device, dtype=torch.float32
                )

                if n_valid > 0:
                    # Gather only valid objects' anchors
                    last_anchor_valid = last_anchor_flat[valid_idx]  # (Nvalid, Nq, 3)
                    curr_anchor_valid = curr_anchor_flat[valid_idx]  # (Nvalid, Nq, 3)

                    if _capturing:
                        # The SVD inside the rigid solve is not graph-capturable, so
                        # the step is recorded as two graphs around an eager solve.
                        # Close graph 1 here and execute it to materialize its outputs.
                        graph_obj.capture_end()
                        graph_obj.replay()
                        torch.cuda.current_stream().synchronize()

                    if self.if_roma:
                        # RoMa mode: batch process only valid objects
                        # RoMa requires FP32 for numerical stability
                        with torch.amp.autocast("cuda", enabled=False):
                            R_valid, t_vec_valid = roma.rigid_points_registration(
                                last_anchor_valid.float(), curr_anchor_valid.float()
                            )
                        R_valid = R_valid.float()  # Ensure FP32 for downstream
                        t_vec_valid = t_vec_valid.float().unsqueeze(1)  # (Nvalid, 1, 3)

                    else:
                        # Standard SVD Kabsch: batch process only valid objects
                        centroid_last = last_anchor_valid.mean(
                            dim=1, keepdim=True
                        )  # (Nvalid,1,3)
                        centroid_curr = curr_anchor_valid.mean(
                            dim=1, keepdim=True
                        )  # (Nvalid,1,3)
                        last_c = last_anchor_valid - centroid_last
                        curr_c = curr_anchor_valid - centroid_curr

                        # Batch SVD for valid objects only
                        H = torch.bmm(last_c.transpose(1, 2), curr_c)  # (Nvalid,3,3)
                        with torch.amp.autocast("cuda", enabled=False):
                            H_float = H.float()
                            U, S, Vt = torch.linalg.svd(H_float)  # Batch SVD!
                        U = U.to(H.dtype)
                        Vt = Vt.to(H.dtype)
                        R_valid = torch.bmm(
                            Vt.transpose(1, 2), U.transpose(1, 2)
                        )  # (Nvalid,3,3)

                        # Fix reflections
                        with torch.amp.autocast("cuda", enabled=False):
                            det_R = torch.det(R_valid.float()).to(R_valid.dtype)
                        need_flip = det_R < 0
                        if need_flip.any():
                            Fm = (
                                torch.eye(3, device=R_valid.device, dtype=R_valid.dtype)
                                .unsqueeze(0)
                                .expand(n_valid, -1, -1)
                                .clone()
                            )
                            Fm[need_flip, 2, 2] = -1
                            R_valid = torch.bmm(
                                torch.bmm(Vt.transpose(1, 2), Fm), U.transpose(1, 2)
                            )

                        # Translation
                        rotated_centroid = torch.bmm(
                            R_valid, centroid_last.transpose(1, 2)
                        ).transpose(1, 2)
                        t_vec_valid = centroid_curr - rotated_centroid  # (Nvalid,1,3)

                    if _capturing:
                        # Fixed-address buffers carry the solve result into graph 2.
                        _rt_R_buf = R_valid.clone()
                        _rt_t_buf = t_vec_valid.clone()
                        R_valid = _rt_R_buf
                        t_vec_valid = _rt_t_buf
                        graph_obj2 = torch.cuda.CUDAGraph()
                        graph_obj2.capture_begin(
                            pool=graph_obj.pool(), capture_error_mode="thread_local"
                        )
                    elif graph_obj2 is not None:
                        _rt_R_buf.copy_(R_valid)
                        _rt_t_buf.copy_(t_vec_valid)
                        R_valid = _rt_R_buf
                        t_vec_valid = _rt_t_buf

                    # Scatter valid R/t back to full tensors
                    R_flat[valid_idx] = R_valid
                    t_vec_flat[valid_idx] = t_vec_valid

                # Reshape back to (B,Nobj,3,3) and (B,Nobj,1,3)
                R_all = R_flat.view(B, Nobj, 3, 3)
                t_vec_all = t_vec_flat.view(B, Nobj, 1, 3)

                # ========== VECTORIZED TRANSFORMATION APPLICATION ==========
                # Apply R, t to all objects in a single bmm call (no per-object loop)
                # registration_refs: (B, Nobj, Nv, 3)
                # R_all: (B, Nobj, 3, 3)
                # t_vec_all: (B, Nobj, 1, 3)

                # Reshape for batch bmm: (B, Nobj, Nv, 3) -> (B*Nobj, Nv, 3)
                last_flat = registration_refs.view(B * Nobj, Nv, 3)
                R_flat = R_all.view(B * Nobj, 3, 3)
                t_flat = t_vec_all.view(B * Nobj, 1, 3)

                # Single bmm for ALL objects: y = R @ x + t (RoMa convention)
                # Convert to FP32 to prevent BF16 precision loss during autoregressive rollout
                y_flat = (
                    torch.bmm(R_flat, last_flat.transpose(1, 2)).transpose(1, 2)
                    + t_flat
                )
                # Use reshape() because transpose() makes tensor non-contiguous
                yproj_bt = y_flat.reshape(
                    B, Nobj, Nv, 3
                ).float()  # FP32 for numerical stability

                # Apply gradient strategy based on transform type
                if self.if_roma:
                    # RoMa mode: fully differentiable, gradients flow through Kabsch
                    blended_next = yproj_bt
                else:
                    # Standard mode: straight-through estimator (STE)
                    blended_next = next_pos_bt + (yproj_bt - next_pos_bt).detach()

                # ========== INTERSECTION HANDLING (EVAL) ==========
                intersection_cfg = self.config.get("model", {}).get(
                    "intersection_handling", {}
                )
                blended_next = handle_intersections(
                    blended_next, valid_obj_mask, vmask_step, intersection_cfg
                )

                if _capturing:
                    # Rotate the carry buffers in place inside the captured region
                    # so every replay advances the state by one step.
                    prev_pos_bt.copy_(curr_pos_bt)
                    curr_pos_bt.copy_(blended_next)
                    if not self._kabsch_reference_fixed:
                        registration_refs.copy_(blended_next)
                    if prev_obj_layers_rollout is not None:
                        for _buf, _new in zip(prev_obj_layers_rollout, all_obj_layers):
                            _buf.copy_(_new)
                    if prev_state_layers_rollout is not None:
                        for _buf, _new in zip(
                            prev_state_layers_rollout, all_state_layers
                        ):
                            _buf.copy_(_new)
                    graph_obj2.capture_end()
                    torch.cuda.set_stream(_prev_stream)
                    torch.cuda.synchronize()
                    graph_obj2.replay()  # capture only records; execute part 2 now
                    torch.cuda.synchronize()
                    out_seq.append(curr_pos_bt.clone())
                    pbar.close()
                    break
                # Registration refs use the post-collision position, matching the forward path ('advancing');
                # model.kabsch_reference 'fixed' keeps the last warm-up frame as the reference for every step
                if not self._kabsch_reference_fixed:
                    registration_refs = blended_next.detach()

                out_seq.append(blended_next)

                # Save current features for next step's penetration prediction
                prev_obj_layers_rollout = all_obj_layers
                prev_state_layers_rollout = all_state_layers

                prev_pos_bt = curr_pos_bt
                curr_pos_bt = blended_next

            if graph_obj is not None:
                target_len = warmup_frames + remaining
                while len(out_seq) < target_len:
                    graph_obj.replay()
                    with torch.amp.autocast("cuda", enabled=False):
                        _R_new, _t_new = roma.rigid_points_registration(
                            last_anchor_valid.float(), curr_anchor_valid.float()
                        )
                    _rt_R_buf.copy_(_R_new.float())
                    _rt_t_buf.copy_(_t_new.float().unsqueeze(1))
                    graph_obj2.replay()
                    out_seq.append(curr_pos_bt.clone())

            if graph_obj is not None:
                # Tear the graphs down and return their private pools to the
                # allocator: the evaluator calls rollout once per batch and per
                # step size, and retired pools would otherwise accumulate.
                graph_obj.reset()
                if graph_obj2 is not None:
                    graph_obj2.reset()
                graph_obj = None
                graph_obj2 = None
                if os.environ.get("RIGIDFORMER_GRAPH_KEEPCACHE", "0") != "1":
                    torch.cuda.empty_cache()

            full = torch.stack(out_seq, dim=1)  # (B, 2+remaining, Nobj, Nv, 3)

            # Single numerical-health check per rollout (one host sync total, not
            # one per step): any non-finite prediction aborts loudly here.
            if not torch.isfinite(full).all():
                raise RuntimeError(
                    "Non-finite values in rollout output; "
                    "inspect the checkpoint and input data."
                )

            # ========== Map back to original absolute space ==========
            # If center_scene_by_t0 was applied, add center_t0 back to predictions
            # so that the output is in the original world coordinate frame;
            # centering remains a purely internal operation of the model.
            if (
                self.config.get("data", {}).get("center_scene_by_t0", False)
                and len(_center_t0_per_sample) > 0
            ):
                for b in range(B):
                    if _center_t0_per_sample[b] is not None:
                        center_t0_tensor = torch.from_numpy(
                            _center_t0_per_sample[b]
                        ).to(full.device)
                        full[b] = full[b] + center_t0_tensor
                logger.info(
                    f"[Center-T0 ROLLOUT] Mapped predictions back to original absolute space"
                )

            output = {"obj_positions": full}

            # Return body_dist_vec2 and accelerations only if requested (for visualization)
            # Skip during RMSE eval to save ~4GB memory on long rollouts
            if return_extras and remaining > 0:
                output["obj_body_dist_vec2"] = torch.stack(all_dist_vec2_seq, dim=1)
                output["obj_accelerations_anchor"] = torch.stack(
                    all_accel_anchor_denorm_seq, dim=1
                )

            return output
