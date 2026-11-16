import os
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encodings.rope import (
    ParticleRotaryEmbedding,
    apply_rotary_emb_cossin,
    apply_rotary_emb_one_cossin,
    apply_rotary_emb,
)

from ..encodings.rms_norm import RMSNorm

EPS = 1e-6

ATTN = os.environ.get("ATTN_IMPL", "flash_attn")
assert ATTN in ["flash_attn", "sdpa"], "ATTN_IMPL must be either 'flash_attn' or 'sdpa'"
if ATTN == "flash_attn":
    try:
        from flash_attn import (
            flash_attn_qkvpacked_func,
            flash_attn_varlen_qkvpacked_func,
            flash_attn_varlen_kvpacked_func,
        )
        from flash_attn.bert_padding import pad_input, unpad_input
        from einops import rearrange
    except ImportError:
        raise ImportError(
            "WorldParticle requires a FlashAttention build matching PyTorch/CUDA. Install the optional WorldParticle runtime before selecting this solver."
        )


class FeedForwardSwiGLU(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Feed forward layer with SwiGLU activation.
        Args:
            dim (int): input dimension
            hidden_dim (int): feed forward hidden dim
            dropout (float): dropout rate, default 0.1
        """
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x))))


class FeedForwardGeLU(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Feed forward layer with GeLU activation.
        Args:
            dim (int): input dimension
            hidden_dim (int): feed forward hidden dim
            dropout (float): dropout rate, default 0.1
        """
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.w2(self.dropout(F.gelu(self.w1(x)))))


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        query_dim,
        num_heads,
        kv_dim=None,
        bias=True,
        qk_norm=False,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.apply_rope_cossin = apply_rotary_emb_cossin

        self.num_heads = num_heads
        self.is_self_attn = kv_dim is None
        kv_dim = query_dim if kv_dim is None else kv_dim

        if self.is_self_attn:
            self.in_proj = nn.Linear(query_dim, 3 * query_dim, bias=bias)
        else:
            self.q_proj = nn.Linear(query_dim, query_dim, bias=bias)
            self.k_proj = nn.Linear(kv_dim, query_dim, bias=bias)
            self.v_proj = nn.Linear(kv_dim, query_dim, bias=bias)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=bias)

        if qk_norm:
            if norm_type == "layer_norm":
                norm_module = nn.LayerNorm
            elif norm_type == "rms_norm":
                norm_module = RMSNorm
            else:
                raise ValueError(
                    "Unsupported normalization type. Choose from 'layer_norm' and 'rms_norm'."
                )
            self.q_norm = norm_module(query_dim, eps=EPS)
            self.k_norm = norm_module(query_dim, eps=EPS)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        q,
        k,
        v,
        src_key_padding_mask=None,
        rope_cos=None,
        rope_sin=None,
        rope_ctx_cos=None,
        rope_ctx_sin=None,
        force_sdpa=False,
    ):
        # src_key_padding_mask: (B, N), key padding mask, things you want to attend to is True
        bs, src_len = q.shape[0], q.shape[1]
        ctx_len = k.shape[1]

        if self.is_self_attn:
            q, k, v = self.in_proj(q).chunk(3, dim=-1)
        else:
            q = self.q_proj(q)
            k = self.k_proj(k)
            v = self.v_proj(v)

        # qk normalization
        q = self.q_norm(q).type(v.dtype)
        k = self.k_norm(k).type(v.dtype)

        q = q.view(bs, src_len, self.num_heads, -1).transpose(
            1, 2
        )  # (bs, num_heads, src_len, head_dim)
        k = k.view(bs, ctx_len, self.num_heads, -1).transpose(1, 2)
        v = v.view(bs, ctx_len, self.num_heads, -1).transpose(1, 2)

        # apply rope
        if rope_cos is not None:
            if rope_ctx_cos is None:
                q, k = self.apply_rope_cossin(q, k, rope_cos, rope_sin)
            else:
                q = apply_rotary_emb_one_cossin(q, rope_cos, rope_sin)
                k = apply_rotary_emb_one_cossin(k, rope_ctx_cos, rope_ctx_sin)

        if ATTN == "sdpa" or force_sdpa:
            # create attention mask
            if src_key_padding_mask is not None:
                # things you want to attend to is True
                assert src_key_padding_mask.shape == (
                    bs,
                    ctx_len,
                ), f"expecting key_padding_mask shape of {(bs, ctx_len)}, but got {src_key_padding_mask.shape}"
                # but now, things you want to attend to is True
                src_key_padding_mask = torch.zeros_like(
                    src_key_padding_mask, dtype=torch.bool
                ).masked_fill_(src_key_padding_mask, True)
                attn_mask = (
                    src_key_padding_mask.view(bs, 1, 1, ctx_len)
                    .expand(-1, self.num_heads, -1, -1)
                    .reshape(bs * self.num_heads, 1, ctx_len)
                )
                attn_mask = attn_mask.view(bs, self.num_heads, -1, ctx_len)
            else:
                attn_mask = None

            attn_output = (
                F.scaled_dot_product_attention(
                    query=q.type(v.dtype),
                    key=k.type(v.dtype),
                    value=v,
                    attn_mask=attn_mask,
                )
                .transpose(1, 2)
                .contiguous()
                .view(bs, src_len, -1)
            )
        elif ATTN == "flash_attn":
            # self-attn
            if self.is_self_attn:
                if src_key_padding_mask is not None:
                    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
                        q.transpose(1, 2), src_key_padding_mask
                    )
                    k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(
                        k.transpose(1, 2), src_key_padding_mask
                    )
                    v_unpad, indices_v, cu_seqlens_v, max_seqlen_v, _ = unpad_input(
                        v.transpose(1, 2), src_key_padding_mask
                    )
                    qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
                    out_unpad = flash_attn_varlen_qkvpacked_func(
                        qkv_unpad,
                        cu_seqlens_q,
                        max_seqlen_q,
                    )
                    attn_output = (
                        pad_input(out_unpad, indices_q, bs, src_len)
                        .contiguous()
                        .view(bs, src_len, -1)
                    )
                else:
                    qkv = torch.stack(
                        [
                            q.transpose(1, 2),
                            k.transpose(1, 2),
                            v.transpose(1, 2),
                        ],
                        dim=2,
                    )
                    attn_output = (
                        flash_attn_qkvpacked_func(qkv)
                        .contiguous()
                        .view(bs, src_len, -1)
                    )
            # cross-attn
            else:
                if src_key_padding_mask is not None:
                    # With padding mask: use variable-length attention
                    q_unpad = rearrange(q, "b h s d -> (b s) h d")
                    cu_seqlens_q = torch.arange(
                        0,
                        (bs + 1) * src_len,
                        step=src_len,
                        dtype=torch.int32,
                        device=q_unpad.device,
                    )
                    max_seqlen_q = src_len

                    k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(
                        k.transpose(1, 2), src_key_padding_mask
                    )
                    v_unpad, _, _, _, _ = unpad_input(
                        v.transpose(1, 2), src_key_padding_mask
                    )
                    kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)

                    out_unpad = flash_attn_varlen_kvpacked_func(
                        q_unpad,
                        kv_unpad,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                    )
                    attn_output = (
                        rearrange(out_unpad, "(b s) h d -> b s h d", b=bs)
                        .contiguous()
                        .view(bs, src_len, -1)
                    )
                else:
                    # Without padding mask: use regular flash attention
                    # Reshape for flash_attn_kvpacked_func
                    from flash_attn import flash_attn_kvpacked_func

                    # q: [bs, num_heads, src_len, head_dim] -> [bs, src_len, num_heads, head_dim]
                    q_flash = q.transpose(1, 2)
                    # k, v: [bs, num_heads, ctx_len, head_dim] -> [bs, ctx_len, num_heads, head_dim]
                    k_flash = k.transpose(1, 2)
                    v_flash = v.transpose(1, 2)

                    # Stack k and v for kvpacked format
                    kv = torch.stack(
                        [k_flash, v_flash], dim=2
                    )  # [bs, ctx_len, 2, num_heads, head_dim]

                    # Apply flash attention
                    attn_output = (
                        flash_attn_kvpacked_func(q_flash, kv)
                        .contiguous()
                        .view(bs, src_len, -1)
                    )
        else:
            raise ValueError(
                "Unsupported attention type. Choose from 'flash_attn' and 'sdpa'."
            )

        return self.out_proj(attn_output)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


SWIN_ATTN_MASK_CACHE = {}


def get_swin_attn_mask(H, W, window_size, shift_size, device):
    """
    Get the attention mask for Swin Transformer. (Original implementation)
    Args:
        H (int): height of image
        W (int): width of image
        window_size (int): window size
        shift_size (int): shift size
        device (torch.device): device to store the attention mask
    Returns:
        attn_mask: (num_windows, num_windows)
    """
    if (H, W, window_size, shift_size) in SWIN_ATTN_MASK_CACHE:
        return SWIN_ATTN_MASK_CACHE[(H, W, window_size, shift_size)]
    else:
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
        h_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = (attn_mask == 0).to(
            torch.bool
        )  # nW, window_size * window_size, window_size * window_size
        SWIN_ATTN_MASK_CACHE[(H, W, window_size, shift_size)] = attn_mask
        return attn_mask


class SwinSelfAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        shift_size: int = 0,
        bias=True,
        qk_norm=False,
        norm_type="layer_norm",
    ):
        """
        Args:
            dim (int): input dimension
            num_heads (int): number of attention heads
            window_size (int): window size
            shift_size (int): shift size, if None, no shift
            bias (bool): whether to use bias, default True
            qk_norm (bool): whether to normalize query and key, default False
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.in_proj = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        if qk_norm:
            if norm_type == "layer_norm":
                norm_module = nn.LayerNorm
            elif norm_type == "rms_norm":
                norm_module = RMSNorm
            else:
                raise ValueError(
                    "Unsupported normalization type. Choose from 'layer_norm' and 'rms_norm'."
                )
            self.q_norm = norm_module(dim, eps=EPS)
            self.k_norm = norm_module(dim, eps=EPS)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        nW = H * W // self.window_size // self.window_size

        # swin related operations
        if self.shift_size > 0:
            attn_mask = get_swin_attn_mask(
                H, W, self.window_size, self.shift_size, x.device
            )  # nW, window_size * window_size, window_size * window_size
            attn_mask = attn_mask.repeat(B, 1, 1)[
                :, None
            ]  # B * nW, 1, window_size * window_size, window_size * window_size
        else:
            attn_mask = None

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # B * nW, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # B * nW, window_size * window_size, C

        q, k, v = self.in_proj(x_windows).chunk(3, dim=-1)

        # qk normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # (B * nW, num_heads, window_size * window_size, head_dim)
        q = q.view(
            B * nW, self.window_size * self.window_size, self.num_heads, -1
        ).transpose(1, 2)
        k = k.view(
            B * nW, self.window_size * self.window_size, self.num_heads, -1
        ).transpose(1, 2)
        v = v.view(
            B * nW, self.window_size * self.window_size, self.num_heads, -1
        ).transpose(1, 2)

        # apply attention
        attn_output = (
            F.scaled_dot_product_attention(
                query=q.type(v.dtype),
                key=k.type(v.dtype),
                value=v,
                attn_mask=attn_mask,
            )
            .transpose(1, 2)
            .contiguous()
            .view(B * nW, self.window_size * self.window_size, -1)
        )

        attn_windows = self.out_proj(
            attn_output
        )  # B * nW, window_size * window_size, C
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # reverse cyclic shift
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H W C
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H, W, C)
        return x


class AttentionLayer(nn.Module):

    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        kv_dim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True,
        bias_kv: bool = False,
        activation: str = "swiglu",
        norm_type: Literal["layer_norm", "rms_norm"] = "layer_norm",
        disable_q_norm: bool = False,
        disable_kv_norm: bool = False,
        qk_norm: bool = False,
        add_self_attn: bool = False,
        use_swin_attn: bool = False,
        window_size: int = 8,
        shift_size: int = 0,
    ):
        """
        Attention layer with feed forward and pre-norm.
        Args:
            query_dim (int): input dimension
            kv_dim (int): key and value dimension, if None, set to query_dim (self-attention)
            num_heads (int): number of attention heads
            hidden_dim (int): feed forward hidden dim
            dropout (float): dropout, default 0.1
            bias (bool): whether to use bias, default True
            bias_kv (bool): whether to use bias for key and value, default False
            activation (str): activation function, choose from 'gelu' and 'swiglu', default 'swiglu'
            norm_type (str): normalization type, choose from 'layer_norm' and 'rms_norm', default 'layer_norm'
            disable_q_norm (bool): disable query normalization, default False
            disable_kv_norm (bool): disable key and value normalization, default False
            qk_norm (bool): whether to apply normalization to query and key, default False
            add_self_attn (bool): whether to add self-attention after cross-attention (cross-attn, self-attn, ffn), default False
            use_swin_attn (bool): whether to use swin self-attention, default False
            window_size (int): window size for swin self-attention, default 8
            shift_size (int): shift size for swin self-attention, default 0 (no shift)
        Returns:
            torch.Tensor: (B, N, query_dim)
        """
        super().__init__()
        self.multihead_attn = MultiHeadAttention(
            query_dim=query_dim,
            num_heads=num_heads,
            kv_dim=kv_dim,
            bias=bias,
            qk_norm=qk_norm,
            norm_type=norm_type,
        )
        print(f"dropout: {dropout}")
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if bias_kv:
            raise NotImplementedError("Bias for key and value is not supported for now")

        if norm_type == "layer_norm":
            norm_module = nn.LayerNorm
        elif norm_type == "rms_norm":
            norm_module = RMSNorm
        else:
            raise ValueError(
                "Unsupported normalization type. Choose from 'layer_norm' and 'rms_norm'."
            )

        self.query_norm = (
            norm_module(query_dim, eps=EPS) if not disable_q_norm else nn.Identity()
        )
        kv_dim = query_dim if kv_dim is None else kv_dim
        if not self.multihead_attn.is_self_attn:
            self.kv_norm = (
                norm_module(kv_dim, eps=EPS) if not disable_kv_norm else nn.Identity()
            )

        self.add_self_attn = add_self_attn
        self.use_swin_attn = use_swin_attn
        if add_self_attn:
            if use_swin_attn:
                self.self_attn = SwinSelfAttention(
                    dim=query_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    bias=bias,
                    qk_norm=qk_norm,
                    norm_type=norm_type,
                )
            else:
                self.self_attn = MultiHeadAttention(
                    query_dim=query_dim,
                    num_heads=num_heads,
                    kv_dim=None,
                    bias=bias,
                    qk_norm=qk_norm,
                    norm_type=norm_type,
                )
            self.self_attn_norm = (
                norm_module(query_dim, eps=EPS) if not disable_q_norm else nn.Identity()
            )

        if activation == "swiglu":
            self.ffn = FeedForwardSwiGLU(
                query_dim, hidden_dim=ffn_hidden_dim, dropout=dropout, bias=bias
            )
        elif activation == "gelu":
            self.ffn = FeedForwardGeLU(
                query_dim, hidden_dim=ffn_hidden_dim, dropout=dropout, bias=bias
            )
        else:
            raise ValueError(
                "Unsupported activation function. Choose from 'gelu' and 'swiglu'."
            )

        self.ffn_norm = norm_module(query_dim, eps=EPS)

    def forward(
        self,
        query,
        kv=None,
        src_key_padding_mask=None,
        rope_cos=None,
        rope_sin=None,
        rope_ctx_cos=None,
        rope_ctx_sin=None,
        force_sdpa=False,
        patch_h=None,
        patch_w=None,
    ):
        """
        Args:
            query (torch.Tensor): (B, N, query_dim)
            kv (torch.Tensor): (B, N, kv_dim), key and value, if None, set to query (self-attention)
            src_key_padding_mask (torch.Tensor): (B, N), key padding mask, things you want to attend to is True
            rope_cos (torch.Tensor): (B, 1, N, head_dim), cosine tensor for RoPE, None if no RoPE is applied
            rope_sin (torch.Tensor): (B, 1, N, head_dim), sine tensor for RoPE, None if no RoPE is applied
            rope_ctx_cos (torch.Tensor): (B, 1, N, head_dim), cosine tensor for RoPE, None if no RoPE is applied
            rope_ctx_sin (torch.Tensor): (B, 1, N, head_dim), sine tensor for RoPE, None if no RoPE is applied
            patch_h (int): height of the patch, used for swin self-attention
            patch_w (int): width of the patch, used for swin self-attention
        Returns:
            torch.Tensor: (B, N, query_dim)
        """

        bs = query.shape[0]

        q = self.query_norm(query)
        if self.multihead_attn.is_self_attn:
            kv = q
        else:
            kv = self.kv_norm(kv)

        # multihead attention
        attn_output = self.dropout(
            self.multihead_attn(
                q,
                kv,
                kv,
                src_key_padding_mask,
                rope_cos,
                rope_sin,
                rope_ctx_cos,
                rope_ctx_sin,
                force_sdpa=force_sdpa,
            )
        )
        query = query + attn_output

        if self.add_self_attn:
            q = self.self_attn_norm(query)
            if self.use_swin_attn:
                q = q.view(bs, patch_h, patch_w, -1)
                self_attn_output = self.self_attn(q)
                self_attn_output = self_attn_output.view(bs, patch_h * patch_w, -1)
            else:
                self_attn_output = self.self_attn(
                    q, q, q, None, rope_cos, rope_sin, force_sdpa=force_sdpa
                )
            query = query + self.dropout(self_attn_output)

        # feed forward
        query = query + self.dropout(self.ffn(self.ffn_norm(query)))
        return query


class CrossAttentionWithRoPE(nn.Module):
    """
    Cross-attention that applies RoPE to queries/keys.

    RenderFormer enhancement: QK-normalization for numerical stability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        qk_norm: bool = True,  # RenderFormer: QK-normalization
        rope_dim: int = None,  # Custom RoPE dimension (None = auto-compute)
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.use_rope = bool(use_rope)
        self.qk_norm = qk_norm

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        # QK-normalization: RMSNorm per head (RenderFormer recommendation)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # 3D Rotary embedding (rope-only); rotate first rope_dim dims
        if rope_dim is None:
            # Auto-compute: multiple of 6, capped at head_dim
            self.rope_dim = max(6, (self.head_dim // 6) * 6)
            self.rope_dim = min(self.rope_dim, self.head_dim)
        else:
            # Use custom rope_dim (must be multiple of 6 for 3D RoPE)
            assert (
                rope_dim % 6 == 0
            ), f"rope_dim must be multiple of 6 for 3D RoPE, got {rope_dim}"
            assert (
                rope_dim <= self.head_dim
            ), f"rope_dim {rope_dim} exceeds head_dim {self.head_dim}"
            self.rope_dim = rope_dim

        # ParticleRotaryEmbedding expects dim parameter but outputs dim//2 frequencies
        # Since we need rope_dim output, we pass rope_dim*2
        self.rope3d = ParticleRotaryEmbedding(
            dim=self.rope_dim * 2,
        )
        # Relative position bias per head, from 3D delta
        self.relpos_bias = nn.Linear(3, self.num_heads, bias=False)

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
        B, Nq, _ = query.shape
        _, Nk, _ = key_value.shape

        q = self._reshape_to_heads(self.q_proj(query))  # (B,H,Nq,Hd)
        k = self._reshape_to_heads(self.k_proj(key_value))  # (B,H,Nk,Hd)
        v = self._reshape_to_heads(self.v_proj(key_value))  # (B,H,Nk,Hd)

        # QK-normalization (RenderFormer): normalize Q and K before RoPE
        if self.qk_norm:
            q = self.q_norm(q)  # (B,H,Nq,Hd) - applied per head
            k = self.k_norm(k)  # (B,H,Nk,Hd) - applied per head

        # RoPE: detect single-point (3D) vs multi-point (Nq*3 D) input
        if pos_q is None or pos_k is None:
            raise ValueError(
                "CrossAttentionWithRoPE expects pos_q and pos_k (rope-only)"
            )

        # Compute RoPE embeddings based on input dimensionality
        rope_q = self.rope3d(pos_q)  # (B,Nq,rope_dim)
        rope_k = self.rope3d(pos_k)  # (B,Nk,rope_dim)

        # Expand to heads: (B,H,N,rope_dim)
        rope_q = rope_q.unsqueeze(1).expand(B, self.num_heads, Nq, self.rope_dim)
        rope_k = rope_k.unsqueeze(1).expand(B, self.num_heads, Nk, self.rope_dim)

        import pdb

        pdb.set_trace()  # 1. check rot pass dim, check self.rope3d
        # Apply rotation to q/k
        q_rot = q[..., : self.rope_dim]
        k_rot = k[..., : self.rope_dim]
        q_pass = q[..., self.rope_dim :]
        k_pass = k[..., self.rope_dim :]
        q_rot = apply_rotary_emb(rope_q, q_rot)
        k_rot = apply_rotary_emb(rope_k, k_rot)
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)

        # Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,Nq,Nk)

        pos_q_3d = pos_q  # (B, Nq, 3)
        pos_k_3d = pos_k  # (B, Nk, 3)
        delta = pos_q_3d.unsqueeze(2) - pos_k_3d.unsqueeze(1)  # (B, Nq, Nk, 3)
        bias = self.relpos_bias(delta)  # (B, Nq, Nk, H)
        bias = bias.permute(0, 3, 1, 2).contiguous()  # (B, H, Nq, Nk)
        attn_scores = attn_scores + bias
        if attn_mask is not None:
            # attn_mask expected shape broadcastable to (B, 1, Nq, Nk) or (B,H,Nq,Nk)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)  # (B,H,Nq,Hd)
        out = self._merge_heads(out)  # (B,Nq,D)
        out = self.out_proj(out)
        return out


from .merge import merge_source, merge_wavg, k_partite_soft_matching


class MultiHeadAttentionWithTOME(nn.Module):

    def __init__(
        self,
        query_dim,
        num_heads,
        ffn_hidden_dim,
        kv_dim=None,
        dropout=0.1,
        bias=True,
        bias_kv=False,
        activation: str = "swiglu",
        norm_type="layer_norm",
        disable_q_norm=False,
        disable_kv_norm=False,
        qk_norm=False,
        add_self_attn: bool = False,
    ):
        super().__init__()
        self.apply_rope_cossin = apply_rotary_emb_cossin

        self.num_heads = num_heads
        self.is_self_attn = kv_dim is None
        kv_dim = query_dim if kv_dim is None else kv_dim

        self.in_proj = nn.Linear(query_dim, 3 * query_dim, bias=bias)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=bias)

        if qk_norm:
            if norm_type == "layer_norm":
                norm_module = nn.LayerNorm
            elif norm_type == "rms_norm":
                norm_module = RMSNorm
            else:
                raise ValueError(
                    "Unsupported normalization type. Choose from 'layer_norm' and 'rms_norm'."
                )
            self.q_norm = norm_module(query_dim, eps=EPS)
            self.k_norm = norm_module(query_dim, eps=EPS)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        assert bias_kv is False, "Bias for key and value is not supported for now"
        self.query_norm = (
            norm_module(query_dim, eps=EPS) if not disable_q_norm else nn.Identity()
        )

        if activation == "swiglu":
            self.ffn = FeedForwardSwiGLU(
                query_dim, hidden_dim=ffn_hidden_dim, dropout=dropout, bias=bias
            )
        elif activation == "gelu":
            self.ffn = FeedForwardGeLU(
                query_dim, hidden_dim=ffn_hidden_dim, dropout=dropout, bias=bias
            )
        else:
            raise ValueError(
                "Unsupported activation function. Choose from 'gelu' and 'swiglu'."
            )
        self.ffn_norm = norm_module(query_dim, eps=EPS)

    def forward(
        self,
        query,
        num_to_remove_tokens,
        size_token=None,  # size of each token
        src_key_padding_mask=None,
        rope_cos=None,
        rope_sin=None,
        is_trace_source=False,
        trace_source=None,
        force_sdpa=False,
        k_partite=-1,
    ):

        # src_key_padding_mask: (B, N), key padding mask, things you want to attend to is True
        bs, src_len = query.shape[0], query.shape[1]

        q = self.query_norm(query)
        kv = q
        ctx_len = kv.shape[1]

        # multihead self attention
        q, k, v = self.in_proj(q).chunk(3, dim=-1)

        # qk normalization
        q = self.q_norm(q).type(v.dtype)
        k = self.k_norm(k).type(v.dtype)

        q = q.view(bs, src_len, self.num_heads, -1).transpose(
            1, 2
        )  # (bs, num_heads, src_len, head_dim)
        k = k.view(bs, ctx_len, self.num_heads, -1).transpose(1, 2)
        v = v.view(bs, ctx_len, self.num_heads, -1).transpose(1, 2)

        # apply rope
        if rope_cos is not None:
            q, k = self.apply_rope_cossin(q, k, rope_cos, rope_sin)

        if ATTN == "sdpa" or force_sdpa:
            # create attention mask
            if src_key_padding_mask is not None:
                # things you want to attend to is True
                assert src_key_padding_mask.shape == (
                    bs,
                    ctx_len,
                ), f"expecting key_padding_mask shape of {(bs, ctx_len)}, but got {src_key_padding_mask.shape}"
                # but now, things you want to attend to is True
                src_key_padding_mask = torch.zeros_like(
                    src_key_padding_mask, dtype=torch.bool
                ).masked_fill_(src_key_padding_mask, True)
                attn_mask = (
                    src_key_padding_mask.view(bs, 1, 1, ctx_len)
                    .expand(-1, self.num_heads, -1, -1)
                    .reshape(bs * self.num_heads, 1, ctx_len)
                )
                attn_mask = attn_mask.view(bs, self.num_heads, -1, ctx_len)
            else:
                attn_mask = None

            attn_output = (
                F.scaled_dot_product_attention(
                    query=q.type(v.dtype),
                    key=k.type(v.dtype),
                    value=v,
                    attn_mask=attn_mask,
                )
                .transpose(1, 2)
                .contiguous()
                .view(bs, src_len, -1)
            )
        elif ATTN == "flash_attn":
            # self-attn
            if src_key_padding_mask is not None:
                q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
                    q.transpose(1, 2), src_key_padding_mask
                )
                k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(
                    k.transpose(1, 2), src_key_padding_mask
                )
                v_unpad, indices_v, cu_seqlens_v, max_seqlen_v, _ = unpad_input(
                    v.transpose(1, 2), src_key_padding_mask
                )
                qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
                out_unpad = flash_attn_varlen_qkvpacked_func(
                    qkv_unpad,
                    cu_seqlens_q,
                    max_seqlen_q,
                )
                attn_output = (
                    pad_input(out_unpad, indices_q, bs, src_len)
                    .contiguous()
                    .view(bs, src_len, -1)
                )
            else:
                qkv = torch.stack(
                    [
                        q.transpose(1, 2),
                        k.transpose(1, 2),
                        v.transpose(1, 2),
                    ],
                    dim=2,
                )
                attn_output = (
                    flash_attn_qkvpacked_func(qkv).contiguous().view(bs, src_len, -1)
                )
        else:
            raise ValueError(
                "Unsupported attention type. Choose from 'flash_attn' and 'sdpa'."
            )

        x_attn = self.out_proj(attn_output)
        metric = k.mean(dim=1)  # (B, N, head_dim)

        # merge
        # TODO the self attn does not include size_token yet
        query = query + self.dropout(x_attn)
        if num_to_remove_tokens > 0:
            assert k_partite >= 2, "k_partite should be at least 2"
            merge, _ = k_partite_soft_matching(
                metric,
                k=k_partite,
                class_token=False,
                distill_token=False,
            )

            if is_trace_source:
                trace_source = merge_source(merge, query, trace_source)

            query, size_token = merge_wavg(merge, query, size_token)

        x = query + self.dropout(self.ffn(self.ffn_norm(query)))
        return x, size_token, trace_source
