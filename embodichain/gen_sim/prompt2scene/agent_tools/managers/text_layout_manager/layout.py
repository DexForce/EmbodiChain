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


from __future__ import annotations

from typing import Any

import numpy as np

from embodichain.gen_sim.prompt2scene.agent_tools.managers.optimization_manager import (
    _center_xy_aabb_layout,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.text_layout_manager.optimization import (
    _optimize_text_layout_slp,
)
__all__ = [
    "_layout_text_objects_grid",
]

def _transitive_closure(
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Floyd–Warshall transitive closure over a small set of nodes."""
    if not nodes or not edges:
        return list(edges)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    adj = [[False] * n for _ in range(n)]
    for src, dst in edges:
        if src in idx and dst in idx:
            adj[idx[src]][idx[dst]] = True
    for k in range(n):
        for i in range(n):
            if adj[i][k]:
                row_k = adj[k]
                row_i = adj[i]
                for j in range(n):
                    if row_k[j]:
                        row_i[j] = True
    closed: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                closed.append((nodes[i], nodes[j]))
    return closed



def _longest_path_ranks(
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> dict[str, int]:
    """Assign integer ranks satisfying ``(A,B)`` → rank[A] < rank[B].

    Uses topological sort + longest-path DP.  Returns a rank dict for every
    node in *nodes* (default 0 for isolated nodes).
    """
    ranks: dict[str, int] = {n: 0 for n in nodes}
    if not edges:
        return ranks
    # Build adjacency and in-degree
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    in_deg: dict[str, int] = {n: 0 for n in nodes}
    present = set(nodes)
    for src, dst in edges:
        if src not in present or dst not in present:
            continue
        adj[src].append(dst)
        in_deg[dst] += 1
    # Kahn topological sort
    queue = [n for n in nodes if in_deg[n] == 0]
    order: list[str] = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in adj[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    # Longest path
    for u in order:
        for v in adj[u]:
            if ranks[v] < ranks[u] + 1:
                ranks[v] = ranks[u] + 1
    # Remaining nodes (cycles / isolated) keep rank 0
    return ranks



def _layout_text_objects_grid(
    *,
    object_ids: list[str],
    xy_sizes: dict[str, np.ndarray],
    spatial_relations: list[dict[str, Any]],
    table_constraints: list[dict[str, Any]] | None = None,
    grid_spacing: float = 0.02,
    padding_ratio: float = 0.08,
) -> dict[str, Any]:
    """Lay out text-scene objects — transitive closure + longest-path ranks.

    1. Transitive closure of left_of / front_of.
    2. Pick centre: explicit 9‑grid ʻcenterʼ, else highest-degree node.
    3. Longest-path rank assignment (left_of→X, front_of→Y).
    4. Shift 9‑grid anchors to their grid positions.
    5. Free objects auto‑wrap below.
    6. Convert ranks→XY using per‑column/row max sizes + gaps.
    7. SA point optimisation + mesh AABB collision cleanup.
    """
    if not object_ids:
        return {
            "centers": {},
            "initial_centers": {},
            "metadata": {
                "method": "transitive_closure_longest_path_with_9grid",
                "iterations": 0,
            },
        }

    # Parse spatial relations.
    left_of_edges: list[tuple[str, str]] = []
    front_of_edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for rel in spatial_relations:
        subject = str(rel.get("subject") or "")
        obj = str(rel.get("object") or "")
        relation = str(rel.get("relation") or "")
        if not subject or not obj or subject == obj:
            continue
        key = (subject, relation, obj)
        if key in seen:
            continue
        seen.add(key)
        if relation == "left_of":
            left_of_edges.append((subject, obj))
        elif relation == "front_of":
            front_of_edges.append((subject, obj))

    # Compute transitive closures.
    left_of_closed = _transitive_closure(object_ids, left_of_edges)
    front_of_closed = _transitive_closure(object_ids, front_of_edges)

    # Parse nine-grid constraints.
    # −Y = front, so front row = 0, back row = 2
    _GRID_TO_RC: dict[str, tuple[int, int]] = {
        "left_front": (0, 0), "center_front": (1, 0), "right_front": (2, 0),
        "left_center": (0, 1), "center": (1, 1), "right_center": (2, 1),
        "left_back": (0, 2), "center_back": (1, 2), "right_back": (2, 2),
        "front": (1, 0), "back": (1, 2),
        "left": (0, 1), "right": (2, 1),
    }
    grid_targets: dict[str, tuple[int, int]] = {}
    for tc in (table_constraints or []):
        asset = str(tc.get("asset") or "")
        grid_name = str(tc.get("grid") or "").strip()
        if asset in object_ids and grid_name in _GRID_TO_RC:
            grid_targets[asset] = _GRID_TO_RC[grid_name]

    # Select a center object when none is explicit.
    auto_center_oid: str | None = None
    has_explicit_center = any(
        tc.get("grid") == "center" for tc in (table_constraints or [])
    )
    if not has_explicit_center:
        # Degree = appearances in left_of + front_of (subject or object)
        degree: dict[str, int] = {oid: 0 for oid in object_ids}
        for src, dst in left_of_closed + front_of_closed:
            if src in degree:
                degree[src] += 1
            if dst in degree:
                degree[dst] += 1
        max_deg = max(degree.values()) if degree else 0
        if max_deg > 0:
            candidates = [oid for oid, d in degree.items() if d == max_deg]
            # Tie-breaker: largest AABB area
            centre_oid = max(
                candidates,
                key=lambda oid: float(xy_sizes[oid][0]) * float(xy_sizes[oid][1]),
            )
            grid_targets[centre_oid] = (1, 1)  # 9‑grid centre
            auto_center_oid = centre_oid

    # Derive ranks from the transitive closures.
    x_rank = _longest_path_ranks(object_ids, left_of_closed)
    # −Y = front:  A front_of B  →  A.y < B.y  →  row[A] < row[B].
    # _longest_path_ranks gives rank[src] < rank[dst]; edges are
    # already (A,B) for "A front_of B", so NO reversal needed.
    y_rank = _longest_path_ranks(object_ids, front_of_closed)

    # Apply nine-grid shifts.
    # Pin 9‑grid objects to their target ranks; shift all connected
    # objects (both upstream and downstream) to preserve topology.
    if grid_targets:
        # Build undirected connected-components via relation edges
        all_edges = left_of_closed + front_of_closed
        neighbours: dict[str, set[str]] = {oid: set() for oid in object_ids}
        for src, dst in all_edges:
            if src in neighbours and dst in neighbours:
                neighbours[src].add(dst)
                neighbours[dst].add(src)
        for oid in grid_targets:
            neighbours.setdefault(oid, set())

        # For each 9‑grid object, BFS the component and shift uniformly
        shifted: set[str] = set()
        for oid, (target_col, target_row) in grid_targets.items():
            if oid in shifted:
                continue
            dx = target_col - x_rank.get(oid, 0)
            dy = target_row - y_rank.get(oid, 0)

            # BFS to collect the full connected component
            component: set[str] = {oid}
            queue = [oid]
            while queue:
                u = queue.pop(0)
                for v in neighbours.get(u, set()):
                    if v not in component:
                        component.add(v)
                        queue.append(v)

            for oid2 in component:
                if oid2 not in grid_targets:  # only shift non‑anchored objects
                    x_rank[oid2] = x_rank.get(oid2, 0) + dx
                    y_rank[oid2] = y_rank.get(oid2, 0) + dy
            shifted.update(component)

    # Propagate row and column alignment.
    # left_of A B  →  same row  (y_rank[A] = y_rank[B])
    # front_of A B →  same col  (x_rank[A] = x_rank[B])
    # Priority (higher wins): 9‑grid > higher degree > larger area.
    _prio = {
        oid: (
            oid in grid_targets,
            sum(1 for e in left_of_closed + front_of_closed if oid in e),
            float(xy_sizes[oid][0]) * float(xy_sizes[oid][1]),
        )
        for oid in object_ids
    }
    for src, dst in left_of_closed:
        if _prio[src] >= _prio[dst]:
            y_rank[dst] = y_rank.get(src, 0)
        else:
            y_rank[src] = y_rank.get(dst, 0)
    for src, dst in front_of_closed:
        if _prio[src] >= _prio[dst]:
            x_rank[dst] = x_rank.get(src, 0)
        else:
            x_rank[src] = x_rank.get(dst, 0)

    # Normalise to >= 0
    min_x = min(x_rank.values()) if x_rank else 0
    min_y = min(y_rank.values()) if y_rank else 0
    for oid in object_ids:
        x_rank[oid] = x_rank.get(oid, 0) - min_x
        y_rank[oid] = y_rank.get(oid, 0) - min_y

    # Resolve cell collisions: spread objects sharing the same (col, row)
    cell_occupants: dict[tuple[int, int], list[str]] = {}
    for oid in object_ids:
        cell = (x_rank[oid], y_rank[oid])
        cell_occupants.setdefault(cell, []).append(oid)
    for (col, row), occupants in cell_occupants.items():
        if len(occupants) > 1:
            for offset, oid in enumerate(occupants[1:], start=1):
                x_rank[oid] = col + offset

    # Place unconstrained objects in wrapped rows.
    constrained = set()
    for src, dst in left_of_closed + front_of_closed:
        constrained.update([src, dst])
    constrained.update(grid_targets)
    free_objects = [oid for oid in object_ids if oid not in constrained]

    if free_objects:
        free_row = max(y_rank.values()) + 1 if y_rank else 0
        # Max row width ≈ existing union width × 1.5 (at least 3 cols)
        col_keys = list(x_rank.values())
        existing_cols = max(col_keys) - min(col_keys) + 1 if col_keys else 1
        max_cols_per_row = max(existing_cols, 3)
        free_sorted = sorted(
            free_objects,
            key=lambda oid: float(xy_sizes[oid][0]),
            reverse=True,
        )
        col = 0
        row_offset = 0
        for oid in free_sorted:
            x_rank[oid] = col
            y_rank[oid] = free_row + row_offset
            col += 1
            if col >= max_cols_per_row:
                col = 0
                row_offset += 1

    # Convert ranks to XY positions.
    col_widths: dict[int, float] = {}
    row_heights: dict[int, float] = {}
    for oid in object_ids:
        c = x_rank[oid]
        r = y_rank[oid]
        col_widths[c] = max(col_widths.get(c, 0.0), float(xy_sizes[oid][0]))
        row_heights[r] = max(row_heights.get(r, 0.0), float(xy_sizes[oid][1]))

    x_cumsum: dict[int, float] = {}
    cumulative = 0.0
    for c in sorted(col_widths):
        x_cumsum[c] = cumulative
        cumulative += col_widths[c] + grid_spacing

    y_cumsum: dict[int, float] = {}
    cumulative = 0.0
    for r in sorted(row_heights):
        y_cumsum[r] = cumulative
        cumulative += row_heights[r] + grid_spacing

    centers: dict[str, np.ndarray] = {}
    for oid in object_ids:
        c = x_rank[oid]
        r = y_rank[oid]
        cx = x_cumsum[c] + 0.5 * float(xy_sizes[oid][0])
        cy = y_cumsum[r] + 0.5 * float(xy_sizes[oid][1])
        centers[oid] = np.array([cx, cy], dtype=np.float64)

    centers = _center_xy_aabb_layout(centers=centers, xy_sizes=xy_sizes)

    initial_centers = {oid: c.copy() for oid, c in centers.items()}

    # Snap initial grid positions as 9‑grid spring targets
    grid_spring_targets: dict[str, np.ndarray] = {
        oid: initial_centers[oid].copy()
        for oid in grid_targets
        if oid in initial_centers
    }

    # Optimize positions and remove mesh AABB collisions.
    optimized = _optimize_text_layout_slp(
        object_ids=object_ids,
        xy_sizes=xy_sizes,
        initial_centers=initial_centers,
        left_of_edges=left_of_closed,
        front_of_edges=front_of_closed,
        grid_spring_targets=grid_spring_targets,
        padding_ratio=padding_ratio,
    )
    centers = optimized["centers"]
    optimization_metadata = optimized["metadata"]

    # Collect layout metadata.
    metadata = {
        "method": "transitive_closure_longest_path_with_9grid_and_sa",
        "grid_spacing": grid_spacing,
        "auto_center_oid": auto_center_oid,
        "has_explicit_center": has_explicit_center,
        "table_constraint_count": len(grid_targets),
        "left_of_count": len(left_of_edges),
        "left_of_closed_count": len(left_of_closed),
        "front_of_count": len(front_of_edges),
        "front_of_closed_count": len(front_of_closed),
        "free_object_count": len(free_objects),
        "x_ranks": {oid: x_rank.get(oid, 0) for oid in object_ids},
        "y_ranks": {oid: y_rank.get(oid, 0) for oid in object_ids},
        "optimization": optimization_metadata,
    }
    return {
        "centers": centers,
        "initial_centers": initial_centers,
        "metadata": metadata,
    }
