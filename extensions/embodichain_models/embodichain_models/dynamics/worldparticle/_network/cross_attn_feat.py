import torch
from torch import nn

from .encodings.rope import ParticleRotaryEmbedding, freqs_to_cos_sin
from .layers.attention import MultiHeadAttentionWithTOME


class CrossAttentionWeightedPoswithFeature(nn.Module):

    def __init__(
        self,
        dim: int,
        filter_extent: float,
        n_heads: int = 6,
        iters: int = 3,
        eps: float = 1e-8,
        use_rope: bool = True,  # Make RoPE optional
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.filter_extent = filter_extent
        self.n_heads = n_heads
        self.iters = iters
        self.eps = eps
        self.nh_f = n_heads
        self.use_rope = use_rope

        # RoPE (3D) for inputs & slots - optional
        if self.use_rope:
            dh_f = dim // n_heads
            assert dh_f * n_heads == dim, "dim must be divisible by n_heads"
            self.dh_f = dh_f
            self.input_rope = ParticleRotaryEmbedding(dim=dh_f)

        # ---------- content head projections (features) ----------
        self.self_attn_with_tome = nn.ModuleList()
        for _ in range(self.iters):
            self.self_attn_with_tome.append(
                MultiHeadAttentionWithTOME(
                    query_dim=dim,
                    num_heads=n_heads,
                    ffn_hidden_dim=dim,
                    kv_dim=None,  # self attention
                    bias=True,
                    qk_norm=True,
                    dropout=dropout,
                )
            )

    def forward(
        self,
        inputs_feat_batch: torch.Tensor,
        input_pos_batch: torch.Tensor,
        k_partite: int = -1,
    ):
        if inputs_feat_batch.dim() != 3 or input_pos_batch.dim() != 3:
            raise ValueError(
                "inputs_feat_batch and input_pos_batch must have shapes [B, N, C] and [B, N, 3]."
            )
        if inputs_feat_batch.shape[:2] != input_pos_batch.shape[:2]:
            raise ValueError(
                f"Feature/position batch mismatch: {inputs_feat_batch.shape[:2]} vs "
                f"{input_pos_batch.shape[:2]}"
            )

        batch_size = inputs_feat_batch.shape[0]
        if batch_size > 1:
            feature_parts = []
            position_parts = []
            for batch_idx in range(batch_size):
                features, positions = self._forward_impl(
                    inputs_feat_batch=inputs_feat_batch[batch_idx : batch_idx + 1],
                    input_pos_batch=input_pos_batch[batch_idx : batch_idx + 1],
                    k_partite=k_partite,
                )
                feature_parts.append(features)
                position_parts.append(positions)
            return torch.cat(feature_parts, dim=0), torch.cat(position_parts, dim=0)

        return self._forward_impl(
            inputs_feat_batch=inputs_feat_batch,
            input_pos_batch=input_pos_batch,
            k_partite=k_partite,
        )

    # ---------------- forward ----------------
    def _forward_impl(
        self,
        inputs_feat_batch: torch.Tensor,
        input_pos_batch: torch.Tensor,
        k_partite: int,
    ):
        """
        inputs_feat    : (B, N, D)
        input_pos : (B, N, 3)
        """

        # Initialize positions as input positions
        pos_batch = input_pos_batch.clone()  # (B, N, 3)

        feat = inputs_feat_batch
        size_token = None
        trace_source = None

        for _iter in range(self.iters):
            # Update RoPE embedding with current positions at each iteration
            if self.use_rope:
                rope_ctx = self.input_rope(pos_batch)
                rope_cos, rope_sin = freqs_to_cos_sin(rope_ctx, head_dim=self.dh_f)

            # Calculate number of tokens to remove (with safety check)
            current_num_tokens = feat.shape[1]
            target_tokens = current_num_tokens // k_partite
            num_to_remove_tokens = current_num_tokens - target_tokens

            assert (
                num_to_remove_tokens >= 0
            ), "num_to_remove_tokens should be non-negative"

            feat, size_token, trace_source = self.self_attn_with_tome[_iter](
                query=feat,
                num_to_remove_tokens=num_to_remove_tokens,
                size_token=size_token,
                src_key_padding_mask=None,
                rope_cos=rope_cos if self.use_rope else None,
                rope_sin=rope_sin if self.use_rope else None,
                is_trace_source=True,
                k_partite=k_partite,
                trace_source=trace_source,
            )  # (B, K_new, D) where K_new = current_num_tokens - num_to_remove_tokens

            # Calculate new positions using trace_source matrix
            # trace_source: (B, K_new, K_old) - binary matrix indicating which old tokens contribute to new tokens
            # Normalize trace_source to get weights (handle case where sum is 0)
            trace_weights = trace_source.float()  # (B, K_new, K_old)
            trace_sum = trace_weights.sum(dim=-1, keepdim=True).clamp(
                min=1.0
            )  # (B, K_new, 1)
            trace_weights = trace_weights / trace_sum  # (B, K_new, K_old)

            # Weighted sum of positions: (B, K_new, K_old) @ (B, K_old, 3) -> (B, K_new, 3)
            pos_batch = torch.bmm(trace_weights, input_pos_batch)  # (B, K_new, 3)

        slots_pos_batch = pos_batch  # Final positions after all iterations

        return feat, slots_pos_batch
