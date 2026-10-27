# Derived from Zhiyang Dou's RigidFormer, revision 160fe0852c51cffae7b7440452b58d0b49db6df6.
# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Adaptations: package-relative imports; framework-owned public interfaces.
# The complete MIT license is in ../THIRD_PARTY_LICENSES.

"""
PointNet-based encoder for rigid body representation.

Contains the per-object point cloud encoder (PointNet backbone with spatial
and feature transforms), multi-strategy global pooling, and step-size FiLM
conditioning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class STN3d(nn.Module):
    """Spatial Transformer Network for 3D point alignment."""

    def __init__(self, input_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_dim, 64, 1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv3 = nn.Conv1d(128, 1024, 1, bias=False)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.fc3 = nn.Linear(256, input_dim * input_dim, bias=False)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input point cloud (B, input_dim, N)
        Returns:
            Transformation matrix (B, input_dim, input_dim)
        """
        batch_size = x.size(0)

        # Extract features (use SiLU for smoother nonlinearity)
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        x = F.silu(self.bn3(self.conv3(x)))

        # Global feature via max pooling
        x = torch.max(x, 2, keepdim=False)[0]

        # Transform to matrix
        x = F.silu(self.bn4(self.fc1(x)))
        x = F.silu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Add identity matrix for stability
        identity = torch.eye(self.input_dim, dtype=x.dtype, device=x.device)
        identity = identity.view(1, -1).repeat(batch_size, 1)
        x = x + identity
        x = x.view(batch_size, self.input_dim, self.input_dim)

        return x


def grouped_masked_batch_norm(
    bn: nn.Module, x: torch.Tensor, mask: torch.Tensor, groups: int
) -> torch.Tensor:
    """Training-mode BatchNorm over x (G*M, C, N) with separate statistics for each of G groups.

    Group g is the contiguous block of rows [g*M, (g+1)*M); its statistics use only the masked-valid
    positions (mask (G*M, N), True = valid). This equals G separate calls of the packed masked BatchNorm
    (RigidBodyEncoder._bn) on the G blocks, including the running-statistics updates (applied in the
    closed form of G sequential momentum updates), without packing: fixed shapes and no host syncs. Under
    torch.distributed the per-group sums are all-reduced with an autograd-aware collective, so both the
    statistics and their gradients are those of SyncBatchNorm applied per group. Padded positions of the
    output are zero, as in the packed path.
    Sums run in fp32 with fp64 accumulation (no fp64 copy of x); results match the packed path up to
    floating-point rounding (different reduction order)."""
    GM, C, N = x.shape
    if GM % groups != 0:
        raise ValueError(
            f"grouped_masked_batch_norm: {GM} rows are not divisible into {groups} groups"
        )
    M = GM // groups
    if mask.shape != (GM, N):
        raise ValueError(
            f"grouped_masked_batch_norm: mask must be {(GM, N)}; got {tuple(mask.shape)}"
        )
    m = mask.view(groups, M, 1, N)
    xf = x.float().view(groups, M, C, N)
    xm = torch.where(m, xf, torch.zeros((), dtype=xf.dtype, device=x.device))
    s1 = xm.sum(dim=(1, 3), dtype=torch.float64)  # (G, C)
    s2 = (xm * xm).sum(dim=(1, 3), dtype=torch.float64)  # (G, C)
    cnt = m.sum(dim=(1, 3), dtype=torch.float64)  # (G, 1)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        from torch.distributed.nn.functional import all_reduce as _ar_autograd

        packed = _ar_autograd(
            torch.cat([s1, s2, cnt], dim=1)
        )  # (G, 2C+1), gradients all-reduced too
        s1, s2, cnt = packed[:, :C], packed[:, C : 2 * C], packed[:, 2 * C :]
    cnt_c = cnt.clamp_min(1.0)
    mean = s1 / cnt_c  # (G, C)
    var = (s2 / cnt_c - mean * mean).clamp_min(0.0)  # biased (normalisation), (G, C)
    if bn.track_running_stats and bn.running_mean is not None:
        with torch.no_grad():
            mom = bn.momentum if bn.momentum is not None else 0.1
            unbiased = var * (cnt / (cnt - 1.0).clamp_min(1.0))
            # G sequential updates r <- (1-mom) r + mom * stat_g, in closed form
            w = mom * (1.0 - mom) ** torch.arange(
                groups - 1, -1, -1, dtype=torch.float64, device=x.device
            )
            decay = (1.0 - mom) ** groups
            rm = bn.running_mean.double() * decay + (
                w.view(groups, 1) * mean.detach()
            ).sum(0)
            rv = bn.running_var.double() * decay + (
                w.view(groups, 1) * unbiased.detach()
            ).sum(0)
            bn.running_mean.copy_(rm.to(bn.running_mean.dtype))
            bn.running_var.copy_(rv.to(bn.running_var.dtype))
            bn.num_batches_tracked.add_(groups)
    inv = torch.rsqrt(var.float() + bn.eps).view(groups, 1, C, 1)
    y = (xf - mean.float().view(groups, 1, C, 1)) * inv
    if bn.affine:
        y = y * bn.weight.float().view(1, 1, C, 1) + bn.bias.float().view(1, 1, C, 1)
    y = torch.where(m, y, torch.zeros((), dtype=y.dtype, device=y.device))
    return y.view(GM, C, N).to(x.dtype)


def grouped_rows_batch_norm(
    bn: nn.Module, x: torch.Tensor, group_sizes: list
) -> torch.Tensor:
    """Training-mode BatchNorm over packed rows x (P, C) with separate statistics for G contiguous row blocks of the
    given sizes: the packed counterpart of grouped_masked_batch_norm (same fp64 accumulation, autograd-aware
    all-reduce of the per-group sums under torch.distributed, closed-form running-statistics updates of G
    sequential momentum steps)."""
    G = len(group_sizes)
    P, C = x.shape
    xf = x.float()
    blocks = xf.split(group_sizes, dim=0)
    s1 = torch.stack([b.sum(dim=0, dtype=torch.float64) for b in blocks])  # (G, C)
    s2 = torch.stack(
        [(b * b).sum(dim=0, dtype=torch.float64) for b in blocks]
    )  # (G, C)
    cnt = torch.tensor(group_sizes, dtype=torch.float64, device=x.device).view(G, 1)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        from torch.distributed.nn.functional import all_reduce as _ar_autograd

        packed = _ar_autograd(torch.cat([s1, s2, cnt], dim=1))
        s1, s2, cnt = packed[:, :C], packed[:, C : 2 * C], packed[:, 2 * C :]
    cnt_c = cnt.clamp_min(1.0)
    mean = s1 / cnt_c
    var = (s2 / cnt_c - mean * mean).clamp_min(0.0)
    if bn.track_running_stats and bn.running_mean is not None:
        with torch.no_grad():
            mom = bn.momentum if bn.momentum is not None else 0.1
            unbiased = var * (cnt / (cnt - 1.0).clamp_min(1.0))
            w = mom * (1.0 - mom) ** torch.arange(
                G - 1, -1, -1, dtype=torch.float64, device=x.device
            )
            decay = (1.0 - mom) ** G
            rm = bn.running_mean.double() * decay + (w.view(G, 1) * mean.detach()).sum(
                0
            )
            rv = bn.running_var.double() * decay + (
                w.view(G, 1) * unbiased.detach()
            ).sum(0)
            bn.running_mean.copy_(rm.to(bn.running_mean.dtype))
            bn.running_var.copy_(rv.to(bn.running_var.dtype))
            bn.num_batches_tracked.add_(G)
    # Per-block broadcast of the group statistics (the same elementwise ops as indexing them per row, whose
    # backward is a heavily duplicated index_put)
    inv, mean_f = torch.rsqrt(var.float() + bn.eps), mean.float()  # (G, C)
    y = torch.cat(
        [(b - mean_f[g]) * inv[g] for g, b in enumerate(blocks)], dim=0
    )  # (P, C)
    if bn.affine:
        y = y * bn.weight.float() + bn.bias.float()
    return y.to(x.dtype)


def hierarchical_subset_size(
    n_valid: torch.Tensor, level: int, small: str
) -> torch.Tensor:
    """Vertices pooled by subsampled level `level` (0-based) of an object with n_valid valid vertices: the first
    n_valid // 2^(level+1) of its FPS ordering (50/25/12.5% for levels 0/1/2). small = 'all_below_8': a level
    with fewer than 8 vertices uses all of them; 'fraction': the stated fraction for every object, at least 1 vertex.
    """
    k = n_valid // (2 ** (level + 1))
    if small == "all_below_8":
        return torch.where(k < 8, n_valid, k)
    if small == "fraction":
        return k.clamp_min(1)
    raise ValueError(
        f"subset_small_objects must be 'all_below_8' or 'fraction', got {small!r}"
    )


class EncoderPack:
    """Packed layout of the valid vertices of an encoder call (rows = objects, N = vertex slots): flat indices of the
    valid vertices, of every hierarchical level's subset, and the per-group row counts for grouped BatchNorm.
    The masks do not change over a batch or a rollout, so the pack is built once (a few host syncs) and reused.
    """

    def __init__(
        self,
        valid_mask: torch.Tensor,
        vtx_rank: Optional[torch.Tensor],
        num_levels: int,
        subsets: bool,
        groups: int = 1,
        small: str = "all_below_8",
    ):
        R, N = valid_mask.shape
        if R % groups != 0:
            raise ValueError(
                f"EncoderPack: {R} rows are not divisible into {groups} groups"
            )
        self.R, self.N, self.groups = R, N, groups
        mask = valid_mask.bool()
        self.idx = mask.reshape(-1).nonzero(as_tuple=True)[0]  # (P,) sorted
        rows_per_group = R // groups
        g_of = torch.div(
            torch.div(self.idx, N, rounding_mode="floor"),
            rows_per_group,
            rounding_mode="floor",
        )
        self.group_sizes = (
            torch.bincount(g_of, minlength=groups).tolist()
            if groups > 1
            else [int(self.idx.numel())]
        )
        self.level_idx, self.level_rows, self.level_masks, self.level_group_sizes = (
            [],
            [],
            [],
            [],
        )
        n_valid = mask.sum(dim=1)
        # Segment layout for pooling over the packed rows (packed_pooling='segment'): row r's valid vertices are
        # the contiguous packed rows of segment r (idx is sorted), lengths = their counts, seg = segment per row.
        self.lengths, self.seg = n_valid, torch.div(self.idx, N, rounding_mode="floor")
        self.level_lengths, self.level_seg = [], []
        for level in range(num_levels):
            if subsets:
                if vtx_rank is None:
                    raise ValueError("EncoderPack: hierarchical subsets need vtx_rank")
                k = hierarchical_subset_size(n_valid, level, small)
                lmask = mask & (vtx_rank < k.unsqueeze(1))
            else:
                lmask = mask
            li = lmask.reshape(-1).nonzero(as_tuple=True)[0]  # sorted, subset of idx
            # positions of the level's vertices inside the packed rows (idx is sorted, li a subset of it)
            self.level_rows.append(torch.searchsorted(self.idx, li))
            self.level_idx.append(li)
            self.level_masks.append(lmask)
            lg = torch.div(
                torch.div(li, N, rounding_mode="floor"),
                rows_per_group,
                rounding_mode="floor",
            )
            self.level_group_sizes.append(
                torch.bincount(lg, minlength=groups).tolist()
                if groups > 1
                else [int(li.numel())]
            )
            self.level_lengths.append(lmask.sum(dim=1))
            self.level_seg.append(torch.div(li, N, rounding_mode="floor"))
        self.mask = mask


class RigidBodyEncoder(nn.Module):
    """
    PointNet encoder for rigid body representation.

    Follows the original PointNet architecture (STN, feature transform,
    shared MLPs), with two pooling variants beyond plain max pooling:
    - Enhanced pooling: max + mean + std for richer global features
    - Hierarchical pooling: multi-scale aggregation (PointNet++ style)
    """

    def __init__(
        self,
        input_dim: int = 9,  # 3 for dist_vec + 3 for velocity + 3 for relative_pos
        hidden_dims: list = None,  # Will use original PointNet dims by default
        output_dim: int = 1024,  # Final fused output dim used by downstream modules (MultiBodyEncoder maps to this)
        backbone_dim: int = None,  # Internal PointNet feature width before pooling/ fusion; defaults to output_dim
        use_stn: bool = True,  # Enable STN like original
        use_batch_norm: bool = True,
        feature_transform: bool = True,  # Enable feature transform like original
        feature_transform_dim: int = 64,  # K=64 in original paper
        pooling_type: str = "enhanced",  # 'max' | 'enhanced' (max+mean+std) | 'hierarchical' (PointNet++)
        num_hierarchical_levels: int = 2,
        masked_pooling: bool = False,  # Exact set-function pooling / BatchNorm over valid vertices only
        freeze_bn_stats: bool = False,  # Training uses the stored running statistics (fixed per-channel affine); no re-estimation
        height_input: bool = False,  # Extra input channel: vertex height above the ground plane (absolute z)
        hierarchical_subsets: bool = False,  # Levels pool the first 50/25/12.5/... percent of an order-independent FPS ordering
        packed_pooling: str = "scatter",  # Packed path only: 'scatter' (pool the padded layout) | 'segment' (pool the packed rows)
        subset_small_objects: str = "all_below_8",  # Subset levels below 8 vertices: 'all_below_8' (all vertices) | 'fraction'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.backbone_dim = backbone_dim if backbone_dim is not None else output_dim
        self.use_stn = use_stn
        self.feature_transform = feature_transform
        self.feature_transform_dim = feature_transform_dim
        self.pooling_type = pooling_type
        self.num_hierarchical_levels = num_hierarchical_levels
        # Set-function mode: pooling and BatchNorm statistics ignore padded vertices,
        # and every hierarchical level pools over the full valid set.
        self.masked_pooling = masked_pooling
        self.freeze_bn_stats = bool(freeze_bn_stats)
        self.height_input = bool(height_input)
        self.hierarchical_subsets = bool(hierarchical_subsets)
        if packed_pooling not in ("scatter", "segment"):
            raise ValueError(
                f"packed_pooling must be 'scatter' or 'segment', got {packed_pooling!r}"
            )
        self.packed_pooling = packed_pooling
        if subset_small_objects not in ("all_below_8", "fraction"):
            raise ValueError(
                f"subset_small_objects must be 'all_below_8' or 'fraction', got {subset_small_objects!r}"
            )
        self.subset_small_objects = subset_small_objects

        # Use original PointNet architecture dimensions if not specified:
        # conv(64) -> conv(64) -> [feature transform] -> conv(64) -> conv(128) -> conv(1024)
        if hidden_dims is None:
            hidden_dims = [64, 128, 1024]

        # Spatial Transformer Network (for input alignment)
        if use_stn:
            self.stn = STN3d(input_dim)

        # First set of convolutions (before feature transform)
        self.conv1 = nn.Conv1d(input_dim, 64, 1, bias=not use_batch_norm)
        self.bn1 = nn.BatchNorm1d(64) if use_batch_norm else nn.Identity()

        self.conv2 = nn.Conv1d(64, 64, 1, bias=not use_batch_norm)
        self.bn2 = nn.BatchNorm1d(64) if use_batch_norm else nn.Identity()

        # Feature transform network at 64-dimensional features
        if feature_transform:
            self.feature_stn = STN3d(feature_transform_dim)

        # Second set of convolutions (after feature transform)
        self.conv3 = nn.Conv1d(64, 64, 1, bias=not use_batch_norm)
        self.bn3 = nn.BatchNorm1d(64) if use_batch_norm else nn.Identity()

        self.conv4 = nn.Conv1d(64, 128, 1, bias=not use_batch_norm)
        self.bn4 = nn.BatchNorm1d(128) if use_batch_norm else nn.Identity()

        self.conv5 = nn.Conv1d(128, self.backbone_dim, 1, bias=not use_batch_norm)
        self.bn5 = (
            nn.BatchNorm1d(self.backbone_dim) if use_batch_norm else nn.Identity()
        )

        # Enhanced pooling: project (max+mean+std) back to backbone_dim
        if pooling_type == "enhanced":
            self.pooling_proj = nn.Sequential(
                nn.Linear(self.backbone_dim * 3, self.backbone_dim * 2),
                nn.ReLU(),
                nn.Linear(self.backbone_dim * 2, self.backbone_dim),
            )

        # Hierarchical pooling: multi-scale feature aggregation (PointNet++ inspired)
        # Designed to handle variable vertex counts robustly
        if pooling_type == "hierarchical":
            self.hierarchical_layers = nn.ModuleList()
            for i in range(num_hierarchical_levels):
                # Each level: Conv1d for local feature extraction
                self.hierarchical_layers.append(
                    nn.Sequential(
                        nn.Conv1d(self.backbone_dim, self.backbone_dim, 1),
                        (
                            nn.BatchNorm1d(self.backbone_dim)
                            if use_batch_norm
                            else nn.Identity()
                        ),
                        nn.ReLU(),
                    )
                )
            # Project concatenated multi-scale features
            # Input: global (backbone_dim*2) + level1 (backbone_dim*2) + ... + levelN (backbone_dim*2)
            # Total: backbone_dim * 2 * (1 + num_hierarchical_levels)
            proj_input_dim = self.backbone_dim * 2 * (1 + num_hierarchical_levels)
            self.hierarchical_proj = nn.Sequential(
                nn.Linear(proj_input_dim, self.backbone_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),  # Dropout for regularization
                nn.Linear(self.backbone_dim * 2, self.backbone_dim),
            )

    def _bn(
        self,
        bn: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        mask_idx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        groups: int = 1,
    ) -> torch.Tensor:
        """BatchNorm over (B, C, N) whose training statistics come from valid vertices only.

        The valid vertices are packed into one (1, C, P) tensor and normalised by the stock
        (Sync)BatchNorm module, so batch statistics, running statistics and the gradient
        through the statistics are exactly those of a BatchNorm applied to the un-padded
        set. Padded rows of the output are zero; the masked pooling ignores them.
        Evaluation uses the running statistics, a per-channel affine map that is already
        independent of padding.

        mask_idx: optional mask.nonzero(as_tuple=True), shared by calls that use the same mask. Indexing
        with it is what boolean-mask indexing does internally, so results are identical; it only avoids
        recomputing the nonzero (a host sync) for the pack and again for the scatter.

        groups > 1: the rows are G contiguous blocks (e.g. G time steps batched together) whose statistics
        must stay separate; grouped_masked_batch_norm computes them per block without packing.
        """
        is_bn = isinstance(bn, torch.nn.modules.batchnorm._BatchNorm)
        if self.freeze_bn_stats and self.training and is_bn:
            # Fine-tuning a converged model: keep the stored statistics (the downstream weights
            # were trained against them), so training and evaluation normalise identically and
            # padded rows cannot influence anything. The affine parameters remain trainable.
            if not getattr(self, "_frozen_bn_logged", False):
                self._frozen_bn_logged = True
                print(
                    f"[Encoder] BatchNorm statistics FROZEN in training (module={type(bn).__name__}; stored running stats, affine trainable)",
                    flush=True,
                )
            return torch.nn.functional.batch_norm(
                x,
                bn.running_mean,
                bn.running_var,
                bn.weight,
                bn.bias,
                False,
                0.0,
                bn.eps,
            )
        if not (self.masked_pooling and mask is not None and self.training and is_bn):
            return bn(x)
        if not getattr(self, "_packed_bn_logged", False):
            self._packed_bn_logged = True
            print(
                f"[Encoder] packed masked BatchNorm ACTIVE: module={type(bn).__name__}, valid={int(mask.sum())}/{mask.numel()} vertices",
                flush=True,
            )
        if groups > 1:
            return grouped_masked_batch_norm(bn, x, mask, groups)
        if mask_idx is None:
            mask_idx = mask.nonzero(as_tuple=True)
        xt = x.transpose(1, 2)  # (B, N, C)
        packed = xt[mask_idx]  # (P, C), same rows and order as xt[mask]
        y = bn(packed.t().unsqueeze(0)).squeeze(0).t()  # (P, C)
        out = xt.new_zeros(xt.shape)
        out[mask_idx] = y.to(out.dtype)
        return out.transpose(1, 2)

    @staticmethod
    def _masked_max(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg = torch.finfo(f.dtype).min
        out = f.masked_fill(~mask.unsqueeze(1), neg).max(dim=2)[0]
        return torch.where(mask.any(dim=1, keepdim=True), out, torch.zeros_like(out))

    @staticmethod
    def _masked_mean(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Accumulate in fp32 so the mean does not depend on the padded length under autocast.
        m = mask.unsqueeze(1).float()
        return ((f.float() * m).sum(dim=2) / m.sum(dim=2).clamp_min(1.0)).to(f.dtype)

    @staticmethod
    def _segment_max_mean(h: torch.Tensor, lengths: torch.Tensor, seg: torch.Tensor):
        """_masked_max and _masked_mean of the packed rows h (P, C) without the padded layout: segment r is the
        contiguous block of lengths[r] rows (seg = segment of every row). The max has the same value and passes its
        gradient to the first maximal row of the segment, which is the index torch.max returns on the padded layout
        (the packed order is the slot order); the mean accumulates in fp32. Empty segments give zeros.
        """
        P = h.shape[0]
        if P >= 2**24:
            raise ValueError(
                f"_segment_max_mean: {P} rows exceed the exact fp32 row positions"
            )
        with torch.no_grad():
            mx = torch.segment_reduce(
                h, "max", lengths=lengths, axis=0, unsafe=True
            )  # (R, C)
            pos = torch.arange(P, device=h.device, dtype=torch.float32).unsqueeze(1)
            first = torch.segment_reduce(
                torch.where(h == mx.index_select(0, seg), pos, float(P)),
                "min",
                lengths=lengths,
                axis=0,
                unsafe=True,
            )
            first = first.clamp_max(P - 1).long()  # (R, C)
        mx = torch.where(
            (lengths > 0).unsqueeze(1),
            h.gather(0, first),
            torch.zeros((), dtype=h.dtype, device=h.device),
        )
        mean = (
            torch.segment_reduce(h.float(), "sum", lengths=lengths, axis=0, unsafe=True)
            / lengths.clamp_min(1).unsqueeze(1).float()
        ).to(h.dtype)
        return mx, mean

    def _bn_rows(
        self, bn: nn.Module, x: torch.Tensor, group_sizes: list, groups: int
    ) -> torch.Tensor:
        """BatchNorm over packed valid rows x (P, C): the statistics of the masked path without packing/scattering."""
        if not isinstance(bn, torch.nn.modules.batchnorm._BatchNorm):
            return bn(x)
        if self.freeze_bn_stats and self.training:
            return torch.nn.functional.batch_norm(
                x,
                bn.running_mean,
                bn.running_var,
                bn.weight,
                bn.bias,
                False,
                0.0,
                bn.eps,
            )
        if self.training and groups > 1:
            return grouped_rows_batch_norm(bn, x, group_sizes)
        return bn(x)

    @staticmethod
    def _rows_linear(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, conv.weight.squeeze(-1), conv.bias)

    def _forward_packed(self, x: torch.Tensor, pack: "EncoderPack", bn_groups: int):
        """Encoder over the valid vertices only (model.encoder.packed): every pointwise layer runs on the packed rows,
        each hierarchical level only on its own vertex subset, and the features are scattered back (zeros at
        padding) for the unchanged masked pooling and for anchor-vertex pooling. Equal to the padded path at every
        valid vertex up to floating-point rounding (different GEMM shapes / reduction order).
        """
        if (
            not (self.masked_pooling and self.pooling_type == "hierarchical")
            or self.use_stn
            or self.feature_transform
        ):
            raise ValueError(
                "the packed encoder path needs masked hierarchical pooling without STN / feature transform"
            )
        R, N, Cin = x.shape
        if (R, N) != (pack.R, pack.N) or bn_groups != pack.groups:
            raise ValueError(
                f"EncoderPack built for {(pack.R, pack.N, pack.groups)}, called with {(R, N, bn_groups)}"
            )
        if not getattr(self, "_packed_path_logged", False):
            self._packed_path_logged = True
            print(
                f"[Encoder] packed vertex path ACTIVE: {int(pack.idx.numel())}/{R * N} vertex slots valid, "
                f"levels {[int(i.numel()) for i in pack.level_idx]}",
                flush=True,
            )
        gs = pack.group_sizes
        h = x.reshape(R * N, Cin).index_select(0, pack.idx)
        h = F.silu(
            self._bn_rows(self.bn1, self._rows_linear(self.conv1, h), gs, bn_groups)
        )
        h = F.silu(
            self._bn_rows(self.bn2, self._rows_linear(self.conv2, h), gs, bn_groups)
        )
        h = F.silu(
            self._bn_rows(self.bn3, self._rows_linear(self.conv3, h), gs, bn_groups)
        )
        h = F.silu(
            self._bn_rows(self.bn4, self._rows_linear(self.conv4, h), gs, bn_groups)
        )
        h = self._bn_rows(
            self.bn5, self._rows_linear(self.conv5, h), gs, bn_groups
        )  # (P, C)
        C = h.shape[1]
        # Per-vertex features in the padded layout (zeros at padding) for anchor-vertex pooling
        per_point_features = (
            h.new_zeros(R * N, C)
            .index_copy(0, pack.idx, h)
            .view(R, N, C)
            .transpose(1, 2)
        )
        segment = self.packed_pooling == "segment"
        if segment and not getattr(self, "_segment_pool_logged", False):
            self._segment_pool_logged = True
            print(
                "[Encoder] packed pooling: segment max/mean over the packed rows (no padded level tensors)",
                flush=True,
            )
        if segment:
            feats = [
                torch.cat(self._segment_max_mean(h, pack.lengths, pack.seg), dim=1)
            ]
        else:
            mask = pack.mask
            feats = [
                torch.cat(
                    [
                        self._masked_max(per_point_features, mask),
                        self._masked_mean(per_point_features, mask),
                    ],
                    dim=1,
                )
            ]
        for level, layer in enumerate(self.hierarchical_layers):
            hl = self._rows_linear(layer[0], h.index_select(0, pack.level_rows[level]))
            hl = layer[2](
                self._bn_rows(layer[1], hl, pack.level_group_sizes[level], bn_groups)
            )
            if segment:
                feats.append(
                    torch.cat(
                        self._segment_max_mean(
                            hl, pack.level_lengths[level], pack.level_seg[level]
                        ),
                        dim=1,
                    )
                )
                continue
            lf = (
                hl.new_zeros(R * N, C)
                .index_copy(0, pack.level_idx[level], hl)
                .view(R, N, C)
                .transpose(1, 2)
            )
            lmask = pack.level_masks[level]
            feats.append(
                torch.cat(
                    [self._masked_max(lf, lmask), self._masked_mean(lf, lmask)], dim=1
                )
            )
        latent = self.hierarchical_proj(torch.cat(feats, dim=1))
        return latent, per_point_features, None

    def forward(
        self,
        dist_vec_norm: torch.Tensor,
        velocity_norm: torch.Tensor,
        relative_pos: torch.Tensor,
        phys_params: Optional[torch.Tensor] = None,
        time_step: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        height: Optional[torch.Tensor] = None,
        vtx_rank: Optional[torch.Tensor] = None,
        bn_groups: int = 1,
        pack: Optional["EncoderPack"] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode rigid body point cloud into latent representation.
        Follows the original PointNet architecture.

        Args:
            dist_vec_norm: Normalized distance vectors (B, N, 3) where N is number of vertices
            velocity_norm: Normalized velocities (B, N, 3)
            relative_pos: Position relative to reference pose (B, N, 3)
            phys_params: Optional physics parameters (B, N, 3) - [mass, friction, restitution] broadcasted to all vertices
            time_step: Optional time step size (B, N, 1) - skip_step_size broadcasted to all vertices
            valid_mask: Optional vertex validity mask (B, N) - True for valid vertices, False for padding

        Returns:
            latent: Global latent representation (B, output_dim)
            per_point_features: Per-point features before pooling (B, output_dim, N)
            trans_feat: Feature transformation matrix if feature_transform=True
        """
        B, N, _ = dist_vec_norm.shape

        # Concatenate all input features: (B, N, input_dim)
        # Base: 9D (dist_vec + velocity + rel_pos)
        # +phys_params: +3D = 12D
        # +time_step: +2D (s and s^2) = 14D
        features = [dist_vec_norm, velocity_norm, relative_pos]

        if phys_params is not None:
            features.append(phys_params)

        if time_step is not None:
            features.append(time_step)
        if self.height_input:
            if height is None:
                raise ValueError(
                    "RigidBodyEncoder(height_input=True) requires the `height` tensor (B, N, 1)"
                )
            features.append(height)  # last channel: height above the ground plane
            if not getattr(self, "_height_logged", False):
                self._height_logged = True
                print(
                    f"[Encoder] height-above-ground input channel ACTIVE (input_dim={self.input_dim}, "
                    f"height range [{height.min().item():.2f}, {height.max().item():.2f}] m)",
                    flush=True,
                )
        x = torch.cat(features, dim=-1)

        if pack is not None:
            return self._forward_packed(x, pack, bn_groups)

        # Transpose for Conv1d: (B, N, C) -> (B, C, N)
        x = x.transpose(1, 2)

        # Apply input spatial transformer if enabled
        if self.use_stn:
            trans = self.stn(x)
            x = x.transpose(2, 1)  # (B, C, N) -> (B, N, C)
            x = torch.bmm(x, trans)  # Apply transformation
            x = x.transpose(2, 1)  # (B, N, C) -> (B, C, N)

        # The five backbone BatchNorms share one vertex mask: compute its index set once (training with
        # masked statistics only; evaluation and frozen statistics never pack).
        _bn_idx = (
            valid_mask.nonzero(as_tuple=True)
            if (
                self.masked_pooling
                and valid_mask is not None
                and self.training
                and not self.freeze_bn_stats
                and bn_groups == 1
            )
            else None
        )

        # First block of convolutions
        x = F.silu(self._bn(self.bn1, self.conv1(x), valid_mask, _bn_idx, bn_groups))
        x = F.silu(self._bn(self.bn2, self.conv2(x), valid_mask, _bn_idx, bn_groups))

        # Apply feature transform at 64d features
        trans_feat = None
        if self.feature_transform:
            trans_feat = self.feature_stn(x)
            x_transformed = x.transpose(2, 1)  # (B, 64, N) -> (B, N, 64)
            x_transformed = torch.bmm(x_transformed, trans_feat)  # Apply transformation
            x = x_transformed.transpose(2, 1)  # (B, N, 64) -> (B, 64, N)

        # Second block of convolutions (after feature transform)
        x = F.silu(self._bn(self.bn3, self.conv3(x), valid_mask, _bn_idx, bn_groups))
        x = F.silu(self._bn(self.bn4, self.conv4(x), valid_mask, _bn_idx, bn_groups))
        x = self._bn(
            self.bn5, self.conv5(x), valid_mask, _bn_idx, bn_groups
        )  # No ReLU on last layer before pooling

        # Store per-point features
        per_point_features = x  # (B, backbone_dim, N)

        # With zero-valued padding, standard pooling is safe: zeros rarely
        # dominate the max and only mildly dilute the mean, so valid_mask is
        # accepted for interface compatibility but not applied here.
        if self.masked_pooling and valid_mask is not None:
            _key = "_masked_pool_logged_" + ("train" if self.training else "eval")
            if not getattr(self, _key, False):
                setattr(self, _key, True)
                print(
                    f"[Encoder] masked set-function pooling ACTIVE ({'train' if self.training else 'eval'} mode): "
                    f"pooling={self.pooling_type}, levels={getattr(self, 'num_hierarchical_levels', None)}, "
                    f"valid={int(valid_mask.sum())}/{valid_mask.numel()} vertices, mask dtype={valid_mask.dtype}",
                    flush=True,
                )
            mask = valid_mask.bool()
            if self.pooling_type == "max":
                latent = self._masked_max(per_point_features, mask)
            elif self.pooling_type == "enhanced":
                mean_feat = self._masked_mean(per_point_features, mask)
                m = mask.unsqueeze(1).to(per_point_features.dtype)
                var = (((per_point_features - mean_feat.unsqueeze(-1)) ** 2) * m).sum(
                    dim=2
                ) / (m.sum(dim=2) - 1).clamp_min(1.0)
                combined = torch.cat(
                    [
                        self._masked_max(per_point_features, mask),
                        mean_feat,
                        var.clamp_min(1e-12).sqrt(),
                    ],
                    dim=1,
                )
                latent = self.pooling_proj(combined)
            elif self.pooling_type == "hierarchical":
                # Every level pools over the full valid set: a set function of the
                # vertices, independent of their order, count, and padding.
                feats = [
                    torch.cat(
                        [
                            self._masked_max(per_point_features, mask),
                            self._masked_mean(per_point_features, mask),
                        ],
                        dim=1,
                    )
                ]
                if self.hierarchical_subsets:
                    if vtx_rank is None:
                        raise ValueError(
                            "RigidBodyEncoder(hierarchical_subsets=True) requires `vtx_rank` (B, N)"
                        )
                    n_valid = mask.sum(dim=1)  # (B,)
                    if not getattr(self, "_subsets_logged", False):
                        self._subsets_logged = True
                        print(
                            f"[Encoder] hierarchical FPS subsets ACTIVE: levels={len(self.hierarchical_layers)}, "
                            f"fractions={[1.0 / 2 ** (l + 1) for l in range(len(self.hierarchical_layers))]}, "
                            f"small objects={self.subset_small_objects}",
                            flush=True,
                        )
                for level, layer in enumerate(self.hierarchical_layers):
                    if self.hierarchical_subsets:
                        # n // 2^(level+1) vertices; small objects per subset_small_objects
                        k = hierarchical_subset_size(
                            n_valid, level, self.subset_small_objects
                        )
                        level_mask = mask & (vtx_rank < k.unsqueeze(1))
                    else:
                        level_mask = mask
                    h = layer[0](per_point_features)
                    h = layer[2](self._bn(layer[1], h, level_mask, None, bn_groups))
                    feats.append(
                        torch.cat(
                            [
                                self._masked_max(h, level_mask),
                                self._masked_mean(h, level_mask),
                            ],
                            dim=1,
                        )
                    )
                latent = self.hierarchical_proj(torch.cat(feats, dim=1))
            else:
                raise ValueError(f"Unknown pooling_type: {self.pooling_type}")
            return latent, per_point_features, trans_feat

        # Unmasked pooling over all slots (PADDING_VALUE=0).
        if self.pooling_type == "max":
            latent = torch.max(per_point_features, dim=2)[0]  # (B, backbone_dim)

        elif self.pooling_type == "enhanced":
            max_feat = torch.max(per_point_features, dim=2)[0]  # (B, backbone_dim)
            mean_feat = torch.mean(per_point_features, dim=2)  # (B, backbone_dim)
            std_feat = torch.std(per_point_features, dim=2)  # (B, backbone_dim)

            combined = torch.cat(
                [max_feat, mean_feat, std_feat], dim=1
            )  # (B, backbone_dim*3)
            latent = self.pooling_proj(combined)  # (B, backbone_dim)

        elif self.pooling_type == "hierarchical":
            # Hierarchical: multi-scale feature aggregation (PointNet++ inspired)
            global_max = torch.max(per_point_features, dim=2)[0]  # (B, backbone_dim)
            global_mean = torch.mean(per_point_features, dim=2)  # (B, backbone_dim)
            global_feat = torch.cat(
                [global_max, global_mean], dim=1
            )  # (B, backbone_dim*2)

            local_features = [global_feat]

            for level, layer in enumerate(self.hierarchical_layers):
                stride = 2 ** (level + 1)
                sampled_feat = per_point_features[
                    :, :, ::stride
                ]  # (B, backbone_dim, N//stride)

                if sampled_feat.shape[2] < 8:
                    sampled_feat = per_point_features

                local_feat = layer(sampled_feat)  # (B, backbone_dim, num_sampled)
                local_max = torch.max(local_feat, dim=2)[0]  # (B, backbone_dim)
                local_mean = torch.mean(local_feat, dim=2)  # (B, backbone_dim)

                local_combined = torch.cat(
                    [local_max, local_mean], dim=1
                )  # (B, backbone_dim*2)
                local_features.append(local_combined)

            all_features = torch.cat(local_features, dim=1)
            latent = self.hierarchical_proj(all_features)  # (B, backbone_dim)

        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")

        return latent, per_point_features, trans_feat


class StepSizeFiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) conditioned on step-size.

    Generates per-feature scale (γ) and shift (β) from [s, s²]:
        output = features * (1 + γ) + β

    This gives the step size a stronger influence than concatenating
    [s, s²] as two extra dimensions of the encoder input.

    Reference: Perez et al., "FiLM: Visual Reasoning with a General
    Conditioning Layer", AAAI 2018.
    """

    def __init__(self, time_dim: int = 2, feature_dim: int = 768):
        super().__init__()
        self.film_net = nn.Sequential(
            nn.Linear(time_dim, 256),
            nn.SiLU(),
            nn.Linear(256, feature_dim * 2),  # γ and β
        )
        # Zero-init the head so γ = β = 0 and FiLM starts as the identity
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias)

    def forward(
        self, features: torch.Tensor, time_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: (B, Nobj, D) or (B*Nobj, Nq, D) — encoder/predictor features
            time_features: (B, 2) — [s, s²] per sample

        Returns:
            Modulated features, same shape as input
        """
        gamma_beta = self.film_net(time_features)  # (B, 2*D)
        D = features.shape[-1]
        gamma = gamma_beta[..., :D]  # (B, D)
        beta = gamma_beta[..., D:]  # (B, D)

        # Broadcast over spatial dimensions (Nobj or Nq)
        ndim_diff = features.dim() - gamma.dim()
        for _ in range(ndim_diff):
            gamma = gamma.unsqueeze(-2)  # (B, 1, D) or (B, 1, 1, D)
            beta = beta.unsqueeze(-2)

        return features * (1.0 + gamma) + beta
