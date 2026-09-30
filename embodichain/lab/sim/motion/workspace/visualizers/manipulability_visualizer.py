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

"""Manipulability-aware workspace visualization.

Colors reachable workspace points by their Yoshikawa manipulability ``w``
while keeping unreachable points visually distinct, and lets a small subset of
points be inspected in detail (``w``, Jacobian condition number, joint
configuration and a translational manipulability ellipsoid).

The module is split so that everything a test needs is pure numerics:

* :func:`align_manipulability_scores` maps a raw score vector onto the points
  of an analysis result, for joint-space results (scores aligned with
  ``workspace_points``) and for Cartesian/plane results (scores aligned with
  ``reachable_points``, optionally scattered back onto ``all_points``).
* :func:`normalize_manipulability` performs robust percentile-clipped, optionally
  logarithmic normalization and reports the raw score range untouched.
* :func:`map_manipulability_colors` turns normalized values into RGBA colors and
  point sizes, with a distinct color for unreachable points.
* :func:`select_inspection_indices` and :func:`inspect_points` build the
  detail records for *selected / top-N / bottom-N* points only, so no ellipsoid
  is ever computed for the full point set.
* :func:`translational_manipulability_ellipsoid` derives ellipsoid semi-axes
  from the translational Jacobian block via
  :func:`embodichain.compute.kinematics.select_jacobian_rows`.

:class:`ManipulabilityVisualizer` is the thin rendering layer on top: it draws a
Matplotlib figure with a visible color bar, or hands the computed colors to the
existing point-cloud backends (``open3d``, ``viser``, ``sim_manager``), or just
returns the mapping arrays with the ``data`` backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np
import torch

from embodichain.compute.kinematics import (
    condition_number,
    select_jacobian_rows,
)
from embodichain.lab.sim.motion.workspace.configs import (
    VisualizationType,
)
from embodichain.lab.sim.motion.workspace.visualizers.base_visualizer import (
    BaseVisualizer,
)
from embodichain.utils import configclass, logger

__all__ = [
    "ManipulabilityColorCfg",
    "ManipulabilityPointSet",
    "ManipulabilityNormalization",
    "ManipulabilityColorMapping",
    "InspectionSelection",
    "PointInspection",
    "align_manipulability_scores",
    "normalize_manipulability",
    "map_manipulability_colors",
    "select_inspection_indices",
    "inspect_points",
    "translational_manipulability_ellipsoid",
    "ellipsoid_surface",
    "ManipulabilityVisualizer",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@configclass
class ManipulabilityColorCfg:
    """Colors and normalization for manipulability visualization."""

    colormap: str = "viridis"
    """Matplotlib colormap name used for reachable points."""

    percentile_clip: tuple[float, float] = (2.0, 98.0)
    """Lower/upper percentile used as color-scale bounds.

    Robust against the heavy tails of manipulability distributions: a handful of
    near-singular (``w`` close to zero) or exceptionally well-conditioned points
    would otherwise flatten the whole color range. Use ``(0.0, 100.0)`` for a
    plain min/max scale.
    """

    log_scale: bool = False
    """Normalize ``log10(w)`` instead of ``w``.

    Manipulability spans orders of magnitude near singularities; the log scale
    keeps the interior of the workspace from saturating to a single color.
    """

    log_epsilon: float = 1e-12
    """Lower clamp applied before ``log10`` so exact zeros stay finite."""

    unreachable_color: tuple[float, float, float, float] = (0.62, 0.62, 0.62, 0.45)
    """RGBA for unreachable points; deliberately desaturated and translucent."""

    reachable_alpha: float = 0.9
    """Alpha applied to colormap output for reachable points."""

    point_size: float = 6.0
    """Marker size for reachable points."""

    unreachable_point_size: float = 1.5
    """Marker size for unreachable points; smaller so they recede visually."""


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ManipulabilityPointSet:
    """Per-point scores aligned with the points of an analysis result.

    Attributes:
        points: Point positions, shape ``(P, 3)``.
        scores: Per-point manipulability, shape ``(P,)``. ``NaN`` where the
            point is unreachable and therefore has no joint configuration.
        reachable_mask: Boolean mask, shape ``(P,)``.
        score_indices: Row of the raw score vector each point maps to, shape
            ``(P,)``; ``-1`` for unreachable points. Use it to index
            ``joint_configurations``.
        joint_configurations: Joint configurations of the scored points, shape
            ``(M, DOF)`` with ``M = reachable_mask.sum()``, or ``None``.
        mode: Analysis mode string the alignment was derived from.
    """

    points: np.ndarray
    scores: np.ndarray
    reachable_mask: np.ndarray
    score_indices: np.ndarray
    joint_configurations: np.ndarray | None = None
    mode: str = ""

    @property
    def num_reachable(self) -> int:
        """Number of reachable (scored) points."""
        return int(self.reachable_mask.sum())

    @property
    def num_unreachable(self) -> int:
        """Number of points shown but not scored."""
        return int(self.reachable_mask.size - self.reachable_mask.sum())


@dataclass
class ManipulabilityNormalization:
    """Outcome of robust manipulability normalization.

    Attributes:
        normalized: Values in ``[0, 1]``, shape ``(N,)``; ``NaN`` for entries
            that carry no score.
        raw_range: ``(min, max)`` of the *raw* scores over valid entries, before
            any clipping or log transform. ``(nan, nan)`` when nothing is valid.
        clip_range: ``(low, high)`` color-scale bounds expressed in raw score
            units, i.e. the values that map to normalized ``0`` and ``1``.
        log_scale: Whether the normalization ran on ``log10(w)``.
        num_valid: Number of entries that contributed to the statistics.
        num_clipped: Number of valid entries that fell outside ``clip_range``.
    """

    normalized: np.ndarray
    raw_range: tuple[float, float]
    clip_range: tuple[float, float]
    log_scale: bool
    num_valid: int
    num_clipped: int


@dataclass
class ManipulabilityColorMapping:
    """Colors, sizes and normalization metadata for a point set.

    Attributes:
        colors: RGBA colors, shape ``(N, 4)``.
        sizes: Marker sizes, shape ``(N,)``.
        reachable_mask: Boolean mask, shape ``(N,)``.
        normalization: The normalization that produced ``colors``.
        colormap: Colormap name used for reachable points.
    """

    colors: np.ndarray
    sizes: np.ndarray
    reachable_mask: np.ndarray
    normalization: ManipulabilityNormalization
    colormap: str

    @property
    def rgb(self) -> np.ndarray:
        """RGB view of :attr:`colors`, shape ``(N, 3)``."""
        return self.colors[:, :3]

    @property
    def raw_range(self) -> tuple[float, float]:
        """Raw score range, forwarded from :attr:`normalization`."""
        return self.normalization.raw_range

    @property
    def clip_range(self) -> tuple[float, float]:
        """Color-scale bounds in raw units, forwarded from :attr:`normalization`."""
        return self.normalization.clip_range


@dataclass
class InspectionSelection:
    """Indices chosen for detailed (ellipsoid) inspection.

    Attributes:
        indices: Point indices, shape ``(K,)``; ordered explicit selection
            first, then descending-``w`` picks, then ascending-``w`` picks.
        labels: Per-index origin, one of ``"selected"``, ``"top"``, ``"bottom"``.
    """

    indices: np.ndarray
    labels: tuple[str, ...]

    def __len__(self) -> int:
        """Number of selected points."""
        return int(self.indices.size)


@dataclass
class PointInspection:
    """Detail record for a single inspected workspace point.

    Attributes:
        index: Index into the visualized point set.
        label: Selection origin (``"selected"``, ``"top"`` or ``"bottom"``).
        position: Cartesian position, shape ``(3,)``.
        manipulability: Yoshikawa index ``w`` at this point.
        condition_number: Jacobian condition number, or ``None`` when not
            requested.
        joint_configuration: Joint configuration, shape ``(DOF,)``, or ``None``.
        ellipsoid_radii: Translational ellipsoid semi-axis lengths in descending
            order, shape ``(3,)``, or ``None``.
        ellipsoid_axes: Principal directions as columns, shape ``(3, 3)``, or
            ``None``.
    """

    index: int
    label: str
    position: np.ndarray
    manipulability: float
    condition_number: float | None = None
    joint_configuration: np.ndarray | None = None
    ellipsoid_radii: np.ndarray | None = None
    ellipsoid_axes: np.ndarray | None = None

    @property
    def anisotropy(self) -> float | None:
        """Ratio of longest to shortest ellipsoid semi-axis, or ``None``."""
        if self.ellipsoid_radii is None:
            return None
        smallest = float(self.ellipsoid_radii[-1])
        if smallest <= 0.0:
            return float("inf")
        return float(self.ellipsoid_radii[0]) / smallest

    def summary_lines(self) -> list[str]:
        """Render the inspection as human-readable lines."""
        lines = [
            f"point #{self.index} ({self.label})",
            "pos = [" + ", ".join(f"{v:.3f}" for v in self.position) + "]",
            f"w = {self.manipulability:.6g}",
        ]
        if self.condition_number is not None:
            lines.append(f"cond(J) = {self.condition_number:.4g}")
        if self.ellipsoid_radii is not None:
            radii = ", ".join(f"{v:.4g}" for v in self.ellipsoid_radii)
            lines.append(f"ellipsoid semi-axes = [{radii}]")
            anisotropy = self.anisotropy
            if anisotropy is not None:
                lines.append(f"anisotropy = {anisotropy:.3g}")
        if self.joint_configuration is not None:
            qpos = ", ".join(f"{v:+.3f}" for v in self.joint_configuration)
            lines.append(f"qpos = [{qpos}]")
        return lines


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _to_numpy(data: torch.Tensor | np.ndarray | Sequence[Any]) -> np.ndarray:
    """Convert tensors, arrays or sequences to a NumPy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _as_float_vector(data: torch.Tensor | np.ndarray | Sequence[Any]) -> np.ndarray:
    """Convert to a 1D float64 vector.

    Raises:
        ValueError: If the input is not one-dimensional.
    """
    array = np.asarray(_to_numpy(data), dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"expected a 1D score vector; got shape {array.shape}.")
    return array


def _as_points(data: torch.Tensor | np.ndarray | Sequence[Any]) -> np.ndarray:
    """Convert to an ``(N, 3)`` float64 point array.

    Raises:
        ValueError: If the input does not have shape ``(N, 3)``.
    """
    array = np.asarray(_to_numpy(data), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3); got shape {array.shape}.")
    return array


def align_manipulability_scores(
    results: Dict[str, Any],
    scores: torch.Tensor | np.ndarray | None = None,
    *,
    include_unreachable: bool = True,
    verify_positions: bool = True,
    position_atol: float = 1e-5,
) -> ManipulabilityPointSet:
    """Align a manipulability score vector with the points of an analysis result.

    ``WorkspaceAnalyzer`` computes one score per stored joint configuration, so
    the score at row ``i`` belongs to:

    * joint-space results: ``workspace_points[i]`` — every sampled point is
      reachable by construction;
    * Cartesian / plane results: ``reachable_points[i]`` — the joint
      configurations are the IK solutions collected in ascending sample order,
      so ``all_points[reachability_mask][i] == reachable_points[i]``.

    When diagnostics were retained (``all_points`` and ``reachability_mask``
    present), the scores are scattered back onto the full sample set so
    unreachable points can be drawn alongside, with ``NaN`` scores. Otherwise
    only the reachable points are returned.

    Args:
        results: Analysis dictionary returned by ``WorkspaceAnalyzer.analyze``.
        scores: Score vector to align. Defaults to
            ``results["manipulability_scores"]``.
        include_unreachable: Scatter scores onto ``all_points`` when the result
            retains them. When ``False``, only reachable points are returned.
        verify_positions: Cross-check that ``all_points[mask]`` equals
            ``reachable_points`` before scattering, so a silently reordered
            result is rejected instead of mis-colored.
        position_atol: Absolute tolerance for that cross-check.

    Returns:
        The aligned :class:`ManipulabilityPointSet`.

    Raises:
        ValueError: If no scores are available, if the result carries no points,
            or if any length / ordering invariant is violated.
    """
    raw = scores if scores is not None else results.get("manipulability_scores")
    if raw is None:
        raise ValueError(
            "no manipulability scores available; enable MetricType.MANIPULABILITY "
            "on the analyzer or pass scores explicitly."
        )
    raw = _as_float_vector(raw)
    mode = str(results.get("mode", ""))

    qpos = results.get("joint_configurations")
    qpos = None if qpos is None else np.asarray(_to_numpy(qpos), dtype=np.float64)
    if qpos is not None and len(qpos) != len(raw):
        raise ValueError(
            f"joint_configurations has {len(qpos)} rows but the score vector has "
            f"{len(raw)}; the analysis result is inconsistent."
        )

    reachable_points = results.get("reachable_points")
    if reachable_points is None:
        # Joint-space results: every stored point is scored.
        points = results.get("workspace_points")
        if points is None:
            raise ValueError(
                f"result (mode={mode!r}) contains neither 'reachable_points' nor "
                "'workspace_points'."
            )
        points = _as_points(points)
        if len(points) != len(raw):
            raise ValueError(
                f"workspace_points has {len(points)} rows but the score vector has "
                f"{len(raw)}; scores cannot be aligned for mode {mode!r}."
            )
        return ManipulabilityPointSet(
            points=points,
            scores=raw.copy(),
            reachable_mask=np.ones(len(points), dtype=bool),
            score_indices=np.arange(len(points), dtype=np.int64),
            joint_configurations=qpos,
            mode=mode,
        )

    reachable = _as_points(reachable_points)
    if len(reachable) != len(raw):
        raise ValueError(
            f"reachable_points has {len(reachable)} rows but the score vector has "
            f"{len(raw)}; scores cannot be aligned for mode {mode!r}."
        )

    all_points = results.get("all_points")
    mask = results.get("reachability_mask")
    if not include_unreachable or all_points is None or mask is None:
        return ManipulabilityPointSet(
            points=reachable,
            scores=raw.copy(),
            reachable_mask=np.ones(len(reachable), dtype=bool),
            score_indices=np.arange(len(reachable), dtype=np.int64),
            joint_configurations=qpos,
            mode=mode,
        )

    all_points = _as_points(all_points)
    mask = np.asarray(_to_numpy(mask)).astype(bool).reshape(-1)
    if len(mask) != len(all_points):
        raise ValueError(
            f"reachability_mask has {len(mask)} entries but all_points has "
            f"{len(all_points)} rows."
        )
    positions = np.flatnonzero(mask)
    if positions.size != len(reachable):
        raise ValueError(
            f"reachability_mask selects {positions.size} points but "
            f"reachable_points has {len(reachable)} rows."
        )
    if (
        verify_positions
        and positions.size
        and not np.allclose(all_points[positions], reachable, atol=position_atol)
    ):
        raise ValueError(
            "all_points[reachability_mask] does not match reachable_points; "
            "refusing to color points with scores that may belong elsewhere."
        )

    full_scores = np.full(len(all_points), np.nan, dtype=np.float64)
    full_scores[positions] = raw
    score_indices = np.full(len(all_points), -1, dtype=np.int64)
    score_indices[positions] = np.arange(len(raw), dtype=np.int64)
    return ManipulabilityPointSet(
        points=all_points,
        scores=full_scores,
        reachable_mask=mask,
        score_indices=score_indices,
        joint_configurations=qpos,
        mode=mode,
    )


def normalize_manipulability(
    scores: torch.Tensor | np.ndarray | Sequence[float],
    *,
    percentile_clip: tuple[float, float] = (2.0, 98.0),
    log_scale: bool = False,
    log_epsilon: float = 1e-12,
    valid_mask: np.ndarray | None = None,
) -> ManipulabilityNormalization:
    """Normalize manipulability scores to ``[0, 1]`` with percentile clipping.

    ``NaN`` entries (unreachable points) never contribute to the statistics and
    stay ``NaN`` in the output. The raw score range is reported untouched so a
    caller can always show the true magnitudes next to the color bar.

    Degenerate inputs are handled explicitly: an empty or fully invalid input
    yields an all-``NaN`` result with ``NaN`` ranges, and a constant input (a
    single point, or all-equal scores) maps to a flat ``0.5`` instead of
    dividing by a zero span.

    Args:
        scores: Score vector, shape ``(N,)``.
        percentile_clip: ``(low, high)`` percentiles in ``[0, 100]`` with
            ``low < high``, evaluated on the transformed values.
        log_scale: Normalize ``log10(max(w, log_epsilon))`` instead of ``w``.
        log_epsilon: Lower clamp applied before ``log10``.
        valid_mask: Optional additional mask of entries to consider valid.

    Returns:
        The :class:`ManipulabilityNormalization` record.

    Raises:
        ValueError: If ``percentile_clip`` is not an increasing pair inside
            ``[0, 100]``, or ``log_epsilon`` is not positive.
    """
    low, high = (float(percentile_clip[0]), float(percentile_clip[1]))
    if not (0.0 <= low < high <= 100.0):
        raise ValueError(
            "percentile_clip must satisfy 0 <= low < high <= 100; got "
            f"{percentile_clip!r}."
        )
    if log_epsilon <= 0.0:
        raise ValueError(f"log_epsilon must be positive; got {log_epsilon}.")

    values = _as_float_vector(scores)
    valid = np.isfinite(values)
    if valid_mask is not None:
        mask = np.asarray(_to_numpy(valid_mask)).astype(bool).reshape(-1)
        if mask.size != values.size:
            raise ValueError(
                f"valid_mask has {mask.size} entries but scores has {values.size}."
            )
        valid &= mask

    normalized = np.full(values.shape, np.nan, dtype=np.float64)
    if not valid.any():
        return ManipulabilityNormalization(
            normalized=normalized,
            raw_range=(float("nan"), float("nan")),
            clip_range=(float("nan"), float("nan")),
            log_scale=log_scale,
            num_valid=0,
            num_clipped=0,
        )

    raw_valid = values[valid]
    raw_range = (float(raw_valid.min()), float(raw_valid.max()))

    if log_scale:
        transformed = np.log10(np.maximum(raw_valid, log_epsilon))
    else:
        transformed = raw_valid

    lo = float(np.percentile(transformed, low))
    hi = float(np.percentile(transformed, high))
    span = hi - lo
    if span <= 0.0:
        # Constant (or numerically constant) data: a flat mid-scale color is
        # honest, whereas 0/0 would be NaN and min/max would be arbitrary.
        normalized[valid] = 0.5
        num_clipped = 0
    else:
        scaled = (transformed - lo) / span
        num_clipped = int(np.count_nonzero((scaled < 0.0) | (scaled > 1.0)))
        normalized[valid] = np.clip(scaled, 0.0, 1.0)

    clip_range = (10.0**lo, 10.0**hi) if log_scale else (lo, hi)
    return ManipulabilityNormalization(
        normalized=normalized,
        raw_range=raw_range,
        clip_range=(float(clip_range[0]), float(clip_range[1])),
        log_scale=log_scale,
        num_valid=int(valid.sum()),
        num_clipped=num_clipped,
    )


def map_manipulability_colors(
    scores: torch.Tensor | np.ndarray | Sequence[float],
    *,
    reachable_mask: np.ndarray | None = None,
    cfg: ManipulabilityColorCfg | None = None,
) -> ManipulabilityColorMapping:
    """Map manipulability scores to RGBA colors and marker sizes.

    Reachable points take the configured colormap over the robustly normalized
    score; unreachable points take ``cfg.unreachable_color`` and a smaller
    marker so they stay visible but visually subordinate.

    Args:
        scores: Score vector, shape ``(N,)``. ``NaN`` marks an unscored point.
        reachable_mask: Optional boolean mask, shape ``(N,)``. Defaults to the
            finite entries of ``scores``. Entries that are masked reachable but
            carry a non-finite score are treated as unreachable.
        cfg: Color configuration. Defaults to :class:`ManipulabilityColorCfg`.

    Returns:
        The :class:`ManipulabilityColorMapping`.

    Raises:
        ImportError: If Matplotlib (the colormap source) is unavailable.
        ValueError: If ``reachable_mask`` length does not match ``scores``.
    """
    cfg = cfg or ManipulabilityColorCfg()
    values = _as_float_vector(scores)
    finite = np.isfinite(values)
    if reachable_mask is None:
        mask = finite
    else:
        mask = np.asarray(_to_numpy(reachable_mask)).astype(bool).reshape(-1)
        if mask.size != values.size:
            raise ValueError(
                f"reachable_mask has {mask.size} entries but scores has "
                f"{values.size}."
            )
        mask = mask & finite

    normalization = normalize_manipulability(
        values,
        percentile_clip=tuple(cfg.percentile_clip),
        log_scale=cfg.log_scale,
        log_epsilon=cfg.log_epsilon,
        valid_mask=mask,
    )

    colormap = _get_colormap(cfg.colormap)
    colors = np.tile(
        np.asarray(cfg.unreachable_color, dtype=np.float64), (values.size, 1)
    )
    sizes = np.full(values.size, float(cfg.unreachable_point_size), dtype=np.float64)
    if mask.any():
        mapped = np.asarray(colormap(normalization.normalized[mask]), dtype=np.float64)
        mapped[:, 3] = float(cfg.reachable_alpha)
        colors[mask] = mapped
        sizes[mask] = float(cfg.point_size)

    return ManipulabilityColorMapping(
        colors=colors,
        sizes=sizes,
        reachable_mask=mask,
        normalization=normalization,
        colormap=cfg.colormap,
    )


def _get_colormap(name: str):
    """Look up a Matplotlib colormap by name.

    Raises:
        ImportError: If Matplotlib is not installed.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Matplotlib is required for manipulability color mapping. "
            "Install it with: pip install matplotlib"
        ) from exc
    return matplotlib.colormaps[name]


def select_inspection_indices(
    scores: torch.Tensor | np.ndarray | Sequence[float],
    *,
    selected: Sequence[int] = (),
    top_k: int = 1,
    bottom_k: int = 1,
    valid_mask: np.ndarray | None = None,
) -> InspectionSelection:
    """Choose the few points that deserve a detailed inspection.

    Ellipsoids are expensive and unreadable in bulk, so only an explicit
    selection plus the best and worst conditioned points are returned. Ties are
    broken by ascending index so the selection is deterministic.

    Args:
        scores: Per-point scores, shape ``(N,)``; ``NaN`` marks unscored points.
        selected: Explicit point indices to always include, highest priority.
        top_k: Number of highest-``w`` points to add.
        bottom_k: Number of lowest-``w`` points to add.
        valid_mask: Optional additional validity mask, shape ``(N,)``.

    Returns:
        The deduplicated :class:`InspectionSelection`.

    Raises:
        ValueError: If ``top_k`` or ``bottom_k`` is negative, or an explicitly
            selected index is out of range or carries no score.
    """
    if top_k < 0 or bottom_k < 0:
        raise ValueError(
            f"top_k and bottom_k must be non-negative; got {top_k}, {bottom_k}."
        )
    values = _as_float_vector(scores)
    valid = np.isfinite(values)
    if valid_mask is not None:
        mask = np.asarray(_to_numpy(valid_mask)).astype(bool).reshape(-1)
        if mask.size != values.size:
            raise ValueError(
                f"valid_mask has {mask.size} entries but scores has {values.size}."
            )
        valid &= mask

    indices: list[int] = []
    labels: list[str] = []
    seen: set[int] = set()

    def _add(index: int, label: str) -> None:
        if index in seen:
            return
        seen.add(index)
        indices.append(index)
        labels.append(label)

    for raw_index in selected:
        index = int(raw_index)
        if index < 0 or index >= values.size:
            raise ValueError(
                f"selected index {index} is out of range for {values.size} points."
            )
        if not valid[index]:
            raise ValueError(
                f"selected index {index} has no manipulability score "
                "(unreachable point)."
            )
        _add(index, "selected")

    valid_indices = np.flatnonzero(valid)
    if valid_indices.size:
        # Stable order: primary key score, secondary key index.
        order = np.lexsort((valid_indices, values[valid_indices]))
        ascending = valid_indices[order]
        for index in ascending[::-1][:top_k]:
            _add(int(index), "top")
        for index in ascending[:bottom_k]:
            _add(int(index), "bottom")

    return InspectionSelection(
        indices=np.asarray(indices, dtype=np.int64),
        labels=tuple(labels),
    )


def translational_manipulability_ellipsoid(
    jacobian: torch.Tensor | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive translational manipulability ellipsoids from spatial Jacobians.

    The ellipsoid ``{J_t q : ||q|| <= 1}`` has semi-axes ``sqrt(eig(J_t J_t^T))``
    along the corresponding eigenvectors, where ``J_t`` is the translational
    row block selected with
    :func:`embodichain.compute.kinematics.select_jacobian_rows`.

    Args:
        jacobian: Batched spatial Jacobian, shape ``(N, 6, DOF)``.

    Returns:
        Tuple of semi-axis lengths ``(N, 3)`` in descending order and the
        matching principal directions ``(N, 3, 3)`` stored as columns.

    Raises:
        ValueError: If ``jacobian`` is not batched 3D or lacks translational
            rows.
    """
    tensor = torch.as_tensor(_to_numpy(jacobian), dtype=torch.float64)
    translational = select_jacobian_rows(tensor, "translational")
    gram = translational @ translational.transpose(-2, -1)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    # ``eigh`` returns ascending eigenvalues; flip to descending semi-axes and
    # flip the matching eigenvector columns with them.
    radii = torch.sqrt(torch.clamp(eigenvalues, min=0.0)).flip(-1)
    axes = eigenvectors.flip(-1)
    return radii.cpu().numpy(), axes.cpu().numpy()


def inspect_points(
    selection: InspectionSelection | Sequence[int],
    *,
    points: torch.Tensor | np.ndarray,
    scores: torch.Tensor | np.ndarray,
    jacobian_fn: Callable[[np.ndarray], Any] | None = None,
    joint_configurations: torch.Tensor | np.ndarray | None = None,
    score_indices: np.ndarray | None = None,
    with_ellipsoid: bool = True,
    with_condition_number: bool = True,
) -> list[PointInspection]:
    """Build detail records for the selected points only.

    ``jacobian_fn`` is invoked exactly once with the joint configurations of the
    selected points, so the number of Jacobians (and therefore ellipsoids)
    computed is ``len(selection)`` regardless of how many points are drawn.

    Args:
        selection: Selection returned by :func:`select_inspection_indices`, or a
            plain sequence of point indices.
        points: Point positions, shape ``(P, 3)``.
        scores: Per-point scores, shape ``(P,)``.
        jacobian_fn: Callable mapping joint configurations ``(K, DOF)`` to
            spatial Jacobians ``(K, 6, DOF)``. Required for ellipsoids and
            condition numbers.
        joint_configurations: Joint configurations of the *scored* points,
            shape ``(M, DOF)``.
        score_indices: Mapping from point index to score/joint row, shape
            ``(P,)``, as produced by :func:`align_manipulability_scores`.
            Defaults to identity.
        with_ellipsoid: Compute translational ellipsoids.
        with_condition_number: Compute Jacobian condition numbers.

    Returns:
        One :class:`PointInspection` per selected index, in selection order.

    Raises:
        ValueError: If an index has no joint configuration while Jacobians were
            requested, or ``jacobian_fn`` returns the wrong number of Jacobians.
    """
    if isinstance(selection, InspectionSelection):
        indices = np.asarray(selection.indices, dtype=np.int64)
        labels = list(selection.labels)
    else:
        indices = np.asarray(list(selection), dtype=np.int64)
        labels = ["selected"] * indices.size

    point_array = _as_points(points)
    score_array = _as_float_vector(scores)
    if indices.size and (indices.min() < 0 or indices.max() >= len(point_array)):
        raise ValueError(
            f"inspection indices out of range for {len(point_array)} points."
        )

    qpos = (
        None
        if joint_configurations is None
        else np.asarray(_to_numpy(joint_configurations), dtype=np.float64)
    )
    if score_indices is None:
        rows = indices
    else:
        row_map = np.asarray(_to_numpy(score_indices), dtype=np.int64).reshape(-1)
        rows = row_map[indices] if indices.size else indices

    want_jacobians = (
        bool(indices.size)
        and jacobian_fn is not None
        and (with_ellipsoid or with_condition_number)
    )
    radii = axes = conditions = None
    selected_qpos = None
    if want_jacobians:
        if qpos is None:
            raise ValueError(
                "joint_configurations are required to compute Jacobians for "
                "inspected points."
            )
        if rows.size and (rows.min() < 0 or rows.max() >= len(qpos)):
            raise ValueError(
                "an inspected point has no joint configuration; select only "
                "reachable points."
            )
        selected_qpos = qpos[rows]
        jacobians = torch.as_tensor(
            _to_numpy(jacobian_fn(selected_qpos)), dtype=torch.float64
        )
        if jacobians.ndim != 3 or len(jacobians) != len(selected_qpos):
            raise ValueError(
                f"jacobian_fn must return {len(selected_qpos)} Jacobians of shape "
                f"(6, DOF); got shape {tuple(jacobians.shape)}."
            )
        if with_condition_number:
            conditions = condition_number(jacobians).cpu().numpy()
        if with_ellipsoid:
            radii, axes = translational_manipulability_ellipsoid(jacobians)
    elif qpos is not None and indices.size:
        valid_rows = (rows >= 0) & (rows < len(qpos))
        selected_qpos = np.full((indices.size, qpos.shape[1]), np.nan)
        selected_qpos[valid_rows] = qpos[rows[valid_rows]]

    inspections: list[PointInspection] = []
    for position, index in enumerate(indices):
        inspections.append(
            PointInspection(
                index=int(index),
                label=labels[position],
                position=point_array[index],
                manipulability=float(score_array[index]),
                condition_number=(
                    None if conditions is None else float(conditions[position])
                ),
                joint_configuration=(
                    None if selected_qpos is None else selected_qpos[position]
                ),
                ellipsoid_radii=None if radii is None else radii[position],
                ellipsoid_axes=None if axes is None else axes[position],
            )
        )
    return inspections


def ellipsoid_surface(
    radii: np.ndarray,
    axes: np.ndarray,
    center: np.ndarray | None = None,
    resolution: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tessellate one ellipsoid for surface plotting.

    Args:
        radii: Semi-axis lengths, shape ``(3,)``.
        axes: Principal directions as columns, shape ``(3, 3)``.
        center: Optional ellipsoid center, shape ``(3,)``. Defaults to origin.
        resolution: Number of samples per angular direction.

    Returns:
        ``(x, y, z)`` grids of shape ``(resolution, resolution)``.
    """
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    unit = np.stack(
        (
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
        )
    )
    scaled = np.asarray(radii, dtype=np.float64).reshape(3, 1, 1) * unit
    rotated = np.einsum("ij,jkl->ikl", np.asarray(axes, dtype=np.float64), scaled)
    if center is not None:
        rotated = rotated + np.asarray(center, dtype=np.float64).reshape(3, 1, 1)
    return rotated[0], rotated[1], rotated[2]


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------


class ManipulabilityVisualizer(BaseVisualizer):
    """Render workspace points colored by manipulability.

    The visualizer owns no numerics of its own: it calls
    :func:`map_manipulability_colors` and then either draws a Matplotlib figure
    with a visible color bar, forwards the colors to a point-cloud backend, or
    returns the mapping arrays.

    Attributes:
        color_cfg: Color and normalization configuration.
        sim_manager: Simulation manager used by the forwarding backends.
        control_part_name: Control part name used to name forwarded point clouds.
        scores: Default score vector used when ``visualize`` is called without
            one, so generic callers such as ``WorkspaceAnalyzer.visualize`` can
            supply the scores at construction time.
        reachable_mask: Default reachability mask paired with :attr:`scores`.
    """

    def __init__(
        self,
        backend: str = "matplotlib",
        color_cfg: ManipulabilityColorCfg | None = None,
        config: Dict[str, Any] | None = None,
        sim_manager: Any | None = None,
        control_part_name: str | None = None,
        scores: torch.Tensor | np.ndarray | None = None,
        reachable_mask: torch.Tensor | np.ndarray | None = None,
    ):
        """Initialize the manipulability visualizer.

        Args:
            backend: One of ``'matplotlib'``, ``'data'``, ``'open3d'``,
                ``'viser'`` or ``'sim_manager'``. Defaults to ``'matplotlib'``
                because the color bar is the point of this visualization.
            color_cfg: Color and normalization configuration.
            config: Optional generic visualizer configuration dictionary.
            sim_manager: SimulationManager for the forwarding backends.
            control_part_name: Control part name used when forwarding.
            scores: Default per-point manipulability scores.
            reachable_mask: Default reachability mask for ``scores``.
        """
        super().__init__(backend, config)
        self.color_cfg = color_cfg or ManipulabilityColorCfg()
        self.sim_manager = sim_manager
        self.control_part_name = control_part_name
        self.scores = scores
        self.reachable_mask = reachable_mask

    def get_type_name(self) -> str:
        """Return the registered visualization type name."""
        return VisualizationType.MANIPULABILITY.value

    def map_colors(
        self,
        scores: torch.Tensor | np.ndarray,
        reachable_mask: np.ndarray | None = None,
    ) -> ManipulabilityColorMapping:
        """Map scores to colors using this visualizer's configuration.

        Args:
            scores: Per-point scores, shape ``(N,)``.
            reachable_mask: Optional reachability mask, shape ``(N,)``.

        Returns:
            The :class:`ManipulabilityColorMapping`.
        """
        return map_manipulability_colors(
            scores, reachable_mask=reachable_mask, cfg=self.color_cfg
        )

    @staticmethod
    def _resolve_scores(
        num_points: int,
        scores: torch.Tensor | np.ndarray,
        reachable_mask: torch.Tensor | np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Line a score vector up with the drawn points.

        Accepts either one score per drawn point, or the analyzer's compact
        ``(M,)`` reachable-only score vector together with a ``(N,)``
        reachability mask selecting exactly those ``M`` points, which is then
        scattered in order (same contract as
        :func:`align_manipulability_scores`).

        Args:
            num_points: Number of points being drawn.
            scores: Score vector, shape ``(N,)`` or ``(M,)``.
            reachable_mask: Optional boolean mask, shape ``(N,)`` or ``(M,)``.

        Returns:
            Tuple of the per-point scores and the matching mask (or ``None``).

        Raises:
            ValueError: If neither layout applies.
        """
        values = _as_float_vector(scores)
        mask = (
            None
            if reachable_mask is None
            else np.asarray(_to_numpy(reachable_mask)).astype(bool).reshape(-1)
        )
        if values.size == num_points:
            return values, (
                mask if mask is not None and mask.size == num_points else None
            )
        if (
            mask is not None
            and mask.size == num_points
            and int(mask.sum()) == values.size
        ):
            full = np.full(num_points, np.nan, dtype=np.float64)
            full[np.flatnonzero(mask)] = values
            return full, mask
        raise ValueError(
            f"{values.size} scores cannot be aligned with {num_points} points; "
            "pass one score per point, or a reachability mask selecting exactly "
            "the scored points."
        )

    def visualize(
        self,
        points: torch.Tensor | np.ndarray,
        colors: torch.Tensor | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any:
        """Visualize points colored by manipulability.

        Scores win over ``colors``: this visualizer exists to color by
        manipulability, so a generic caller that also passes reachability colors
        (as ``WorkspaceAnalyzer.visualize`` does) still gets the manipulability
        mapping. ``colors`` is used only when no scores are available.

        Args:
            points: Point positions, shape ``(N, 3)``.
            colors: Precomputed colors, used only as a fallback when no scores
                are available.
            **kwargs: Additional parameters:

                - ``scores``: per-point manipulability, shape ``(N,)`` or
                  ``(M,)`` together with a ``(N,)`` ``reachable_mask`` selecting
                  ``M`` points. Defaults to :attr:`scores`.
                - ``reachable_mask``: boolean mask, shape ``(N,)``.
                - ``point_set``: a :class:`ManipulabilityPointSet` providing
                  ``points``/``scores``/``reachable_mask`` in one object.
                - ``title``: figure title (Matplotlib backend).
                - ``highlight``: indices to mark with a cross (Matplotlib).
                - ``elev`` / ``azim``: 3D view angles (Matplotlib).

        Returns:
            A Matplotlib figure, a backend-specific handle, or the mapping data
            dictionary for the ``data`` backend.

        Raises:
            ValueError: If no scores are supplied and no colors are given, or
                the backend is unsupported.
        """
        point_set: ManipulabilityPointSet | None = kwargs.pop("point_set", None)
        scores = kwargs.pop("scores", None)
        reachable_mask = kwargs.pop("reachable_mask", None)
        kwargs.pop("sizes", None)  # generic callers pass reachability sizes
        if point_set is not None:
            points = point_set.points
            scores = point_set.scores if scores is None else scores
            reachable_mask = (
                point_set.reachable_mask if reachable_mask is None else reachable_mask
            )
        if scores is None:
            scores = self.scores
            if reachable_mask is None:
                reachable_mask = self.reachable_mask

        points = self._to_numpy(points)
        self._validate_points(points)

        mapping: ManipulabilityColorMapping | None = None
        if scores is not None:
            scores, reachable_mask = self._resolve_scores(
                len(points), scores, reachable_mask
            )
            mapping = self.map_colors(scores, reachable_mask)
            colors = mapping.colors
        elif colors is not None:
            colors = self._validate_colors(colors, len(points))
        else:
            raise ValueError(
                "scores (or a point_set, or colors) are required to visualize."
            )

        if self.backend == "data":
            data = {
                "points": points,
                "colors": colors,
                "type": self.get_type_name(),
            }
            if mapping is not None:
                data.update(
                    {
                        "sizes": mapping.sizes,
                        "normalized": mapping.normalization.normalized,
                        "reachable_mask": mapping.reachable_mask,
                        "raw_range": mapping.raw_range,
                        "clip_range": mapping.clip_range,
                        "log_scale": mapping.normalization.log_scale,
                        "colormap": mapping.colormap,
                    }
                )
            self._last_visualization = {"data": data, "mapping": mapping}
            return data

        if self.backend == "matplotlib":
            if mapping is None:
                raise ValueError(
                    "the matplotlib backend needs scores to build the color bar."
                )
            figure = self._create_matplotlib_figure(points, mapping, **kwargs)
            self._last_visualization = {"figure": figure, "mapping": mapping}
            return figure

        if self.backend in ("open3d", "viser", "sim_manager"):
            handle = self._forward_to_point_cloud(points, np.asarray(colors)[:, :3])
            self._last_visualization = {"handle": handle, "mapping": mapping}
            return handle

        raise ValueError(f"Unsupported backend: {self.backend}")

    def visualize_inspection(
        self,
        inspections: Sequence[PointInspection],
        **kwargs: Any,
    ) -> Any:
        """Draw the detail view for inspected points.

        Each inspected point gets a translational manipulability ellipsoid plus
        its ``w``, condition number and joint configuration.

        Args:
            inspections: Records from :func:`inspect_points`.
            **kwargs: Additional parameters:

                - ``title``: figure title.
                - ``log_scale``: annotate ``w`` on a log scale (cosmetic only).
                - ``elev`` / ``azim``: 3D view angles.

        Returns:
            The Matplotlib figure.

        Raises:
            ValueError: If ``inspections`` is empty, or the backend cannot draw.
        """
        if not inspections:
            raise ValueError("no inspections to visualize.")
        if self.backend not in ("matplotlib", "data"):
            raise ValueError(
                "point inspection rendering requires the 'matplotlib' backend; "
                f"got {self.backend!r}."
            )
        figure = self._create_inspection_figure(inspections, **kwargs)
        self._last_visualization = {"figure": figure, "inspections": list(inspections)}
        return figure

    # -- rendering helpers -------------------------------------------------

    def _create_matplotlib_figure(
        self,
        points: np.ndarray,
        mapping: ManipulabilityColorMapping,
        title: str = "Workspace manipulability",
        highlight: Sequence[int] | None = None,
        elev: float = 22.0,
        azim: float = -58.0,
        figsize: tuple[float, float] = (9.5, 7.5),
    ):
        """Build the 3D scatter figure with a color bar and range annotation."""
        import matplotlib

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, Normalize

        normalization = mapping.normalization
        reachable = mapping.reachable_mask
        unreachable = ~reachable

        figure = plt.figure(figsize=figsize)
        axes = figure.add_subplot(111, projection="3d")

        if unreachable.any():
            axes.scatter(
                points[unreachable, 0],
                points[unreachable, 1],
                points[unreachable, 2],
                c=mapping.colors[unreachable],
                s=mapping.sizes[unreachable],
                marker="x",
                linewidths=0.7,
                depthshade=False,
                label=f"unreachable ({int(unreachable.sum())})",
            )
        if reachable.any():
            axes.scatter(
                points[reachable, 0],
                points[reachable, 1],
                points[reachable, 2],
                c=mapping.colors[reachable],
                s=mapping.sizes[reachable],
                depthshade=False,
                label=f"reachable ({int(reachable.sum())})",
            )
        if highlight is not None and len(highlight):
            index = np.asarray(list(highlight), dtype=np.int64)
            axes.scatter(
                points[index, 0],
                points[index, 1],
                points[index, 2],
                facecolors="none",
                edgecolors="crimson",
                s=140.0,
                linewidths=1.8,
                depthshade=False,
                label="inspected",
            )

        axes.set_xlabel("X [m]")
        axes.set_ylabel("Y [m]")
        axes.set_zlabel("Z [m]")
        axes.set_title(title)
        axes.view_init(elev=elev, azim=azim)
        axes.legend(loc="upper left", fontsize=8, framealpha=0.85)

        low, high = normalization.clip_range
        if np.isfinite(low) and np.isfinite(high) and high > low:
            norm = (
                LogNorm(vmin=max(low, np.finfo(float).tiny), vmax=high)
                if normalization.log_scale
                else Normalize(vmin=low, vmax=high)
            )
        else:
            norm = Normalize(vmin=0.0, vmax=1.0)
        mappable = matplotlib.cm.ScalarMappable(
            norm=norm, cmap=_get_colormap(mapping.colormap)
        )
        mappable.set_array(np.asarray([]))
        color_bar = figure.colorbar(mappable, ax=axes, shrink=0.72, pad=0.10)
        scale = "log10 scale" if normalization.log_scale else "linear scale"
        color_bar.set_label(f"Yoshikawa manipulability w  ({scale})")

        figure.text(
            0.01,
            0.02,
            self._range_annotation(normalization),
            fontsize=8,
            family="monospace",
            va="bottom",
        )
        figure.tight_layout()
        return figure

    @staticmethod
    def _range_annotation(normalization: ManipulabilityNormalization) -> str:
        """Compose the raw/clipped score range annotation text."""
        raw_low, raw_high = normalization.raw_range
        clip_low, clip_high = normalization.clip_range
        return (
            f"raw w range   : [{raw_low:.4g}, {raw_high:.4g}]   "
            f"(scored points: {normalization.num_valid})\n"
            f"color bar span: [{clip_low:.4g}, {clip_high:.4g}]   "
            f"(percentile-clipped, {normalization.num_clipped} points saturated)"
        )

    def _create_inspection_figure(
        self,
        inspections: Sequence[PointInspection],
        title: str = "Selected-point manipulability inspection",
        elev: float = 20.0,
        azim: float = -60.0,
        figsize: tuple[float, float] | None = None,
    ):
        """Build one ellipsoid subplot per inspected point, with detail text."""
        import matplotlib.pyplot as plt

        count = len(inspections)
        figsize = figsize or (4.6 * count, 5.6)
        figure = plt.figure(figsize=figsize)
        figure.suptitle(title)

        for position, inspection in enumerate(inspections):
            axes = figure.add_subplot(1, count, position + 1, projection="3d")
            radii = inspection.ellipsoid_radii
            if radii is not None and inspection.ellipsoid_axes is not None:
                x, y, z = ellipsoid_surface(radii, inspection.ellipsoid_axes)
                axes.plot_surface(
                    x,
                    y,
                    z,
                    color="tab:blue",
                    alpha=0.35,
                    linewidth=0.2,
                    edgecolor="tab:blue",
                    rstride=2,
                    cstride=2,
                )
                extent = float(np.max(radii))
                extent = extent if extent > 0.0 else 1.0
                axes.set_xlim(-extent, extent)
                axes.set_ylim(-extent, extent)
                axes.set_zlim(-extent, extent)
            else:
                axes.text2D(
                    0.5,
                    0.5,
                    "no ellipsoid",
                    ha="center",
                    transform=axes.transAxes,
                )
            axes.set_title(
                f"{inspection.label}: w = {inspection.manipulability:.4g}",
                fontsize=10,
            )
            axes.set_xlabel("vx [m/s]", fontsize=8)
            axes.set_ylabel("vy [m/s]", fontsize=8)
            axes.set_zlabel("vz [m/s]", fontsize=8)
            axes.tick_params(labelsize=6)
            axes.view_init(elev=elev, azim=azim)
            axes.text2D(
                0.0,
                -0.16,
                "\n".join(inspection.summary_lines()),
                transform=axes.transAxes,
                fontsize=7,
                family="monospace",
                va="top",
            )

        figure.subplots_adjust(bottom=0.30, top=0.90)
        return figure

    def _forward_to_point_cloud(self, points: np.ndarray, rgb: np.ndarray) -> Any:
        """Hand the computed colors to the shared point-cloud visualizer."""
        from embodichain.lab.sim.motion.workspace.visualizers.point_cloud_visualizer import (
            PointCloudVisualizer,
        )

        delegate = PointCloudVisualizer(
            backend=self.backend,
            point_size=float(self.color_cfg.point_size),
            config=self.config,
            sim_manager=self.sim_manager,
            control_part_name=self.control_part_name,
        )
        handle = delegate.visualize(points, rgb)
        self._delegate = delegate
        return handle

    def _save_impl(self, filepath: Path, **kwargs: Any) -> None:
        """Persist the last visualization.

        Args:
            filepath: Destination path.
            **kwargs: Forwarded to ``Figure.savefig`` for Matplotlib output.
        """
        if "figure" in self._last_visualization:
            kwargs.setdefault("dpi", 150)
            kwargs.setdefault("bbox_inches", "tight")
            self._last_visualization["figure"].savefig(filepath, **kwargs)
            return
        if "data" in self._last_visualization:
            data = {
                key: value
                for key, value in self._last_visualization["data"].items()
                if isinstance(value, (np.ndarray, list, tuple))
            }
            np.savez(filepath, **data)
            return
        delegate = getattr(self, "_delegate", None)
        if delegate is not None:
            delegate.save(filepath, **kwargs)
            return
        raise RuntimeError("nothing to save for the current backend.")

    def _show_impl(self, **kwargs: Any) -> None:
        """Display the last visualization."""
        if "figure" in self._last_visualization:
            import matplotlib.pyplot as plt

            plt.show()
            return
        delegate = getattr(self, "_delegate", None)
        if delegate is not None:
            delegate.show(**kwargs)
            return
        logger.log_warning(
            f"Backend {self.backend!r} has nothing to display interactively."
        )
