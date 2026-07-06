from embodichain.gen_sim.prompt2scene.agent_tools.managers.layout_manager.sa_state import (
    SceneState,
    Tempo_SceneState,
)
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import trimesh
from scipy.optimize import minimize


def _parse_coordinate_range(
    coordinate_range: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Optional[List[float]]]]:
    """
    Convert a Node 2 ``coordinate_range`` (expressed in cm) into the optimization
    frame (expressed in meters).

    Node 2 output shape (cm)::

        {"x": [x_min, x_max], "y": [y_min, y_max]}

    where the whole field may be null, or ``x`` / ``y`` may individually be null.

    Returns (m)::

        {"x": [x_min, x_max] | None, "y": [y_min, y_max] | None}

    or ``None`` when no usable range is present.
    """
    if not isinstance(coordinate_range, dict):
        return None

    def _axis(vals) -> Optional[List[float]]:
        if not isinstance(vals, (list, tuple)) or len(vals) < 2:
            return None
        try:
            lo = float(vals[0]) * 0.01
            hi = float(vals[1]) * 0.01
        except (TypeError, ValueError):
            return None
        if lo > hi:
            lo, hi = hi, lo
        return [lo, hi]

    x_axis = _axis(coordinate_range.get("x"))
    y_axis = _axis(coordinate_range.get("y"))

    if x_axis is None and y_axis is None:
        return None
    return {"x": x_axis, "y": y_axis}


def _intersect_axis_range(
    a: Optional[List[float]], b: Optional[List[float]]
) -> Optional[List[float]]:
    """
    Intersect two 1D intervals ``[lo, hi]``.

    ``None`` means "no constraint on this axis". When the intersection would be
    empty (conflicting user input), fall back to the most recently provided
    interval ``b`` so the constraint stays well-formed instead of infeasible.
    """
    if a is None:
        return b
    if b is None:
        return a
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if lo > hi:
        return b
    return [lo, hi]


def _effective_axis_bounds(
    group: Dict[str, Any], W: float, H: float
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Compute the effective per-axis bounds for a group center, in meters.

    Priority per axis::

        explicit coordinate_range  >  coarse region_box

    The explicit ``coordinate_range`` extracted by Node 2 overrides the coarse
    ``region_box`` on a per-axis basis so that conflicting coarse regions never
    invalidate an exact user constraint. The result is finally clamped to the
    physical table ``[0, W] x [0, H]``.

    Returns ``(xbounds, ybounds)`` where each is ``[lo, hi]`` or ``None``.
    """
    box = group.get("region_box")
    xb = [float(box[0]), float(box[1])] if box is not None else None
    yb = [float(box[2]), float(box[3])] if box is not None else None

    cr = group.get("coordinate_range")
    if isinstance(cr, dict):
        if cr.get("x") is not None:
            xb = [float(cr["x"][0]), float(cr["x"][1])]
        if cr.get("y") is not None:
            yb = [float(cr["y"][0]), float(cr["y"][1])]

    if xb is not None:
        xb = [max(0.0, xb[0]), min(float(W), xb[1])]
        if xb[0] > xb[1]:
            xb = None
    if yb is not None:
        yb = [max(0.0, yb[0]), min(float(H), yb[1])]
        if yb[0] > yb[1]:
            yb = None
    return xb, yb


def _is_coordinate_point(value: Any) -> bool:
    """
    A relation target may now be either an object-id string (e.g. ``"laptop_0"``)
    or an absolute coordinate point ``[x, y]`` in cm (e.g. ``[10.0, 20.0]``).

    Return True only for the coordinate-point case.
    """
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and not isinstance(value[0], bool)
        and isinstance(value[1], (int, float))
        and not isinstance(value[1], bool)
    )


def _region_box(region: str, W: float, H: float):
    if region == "left_area":
        return [0.0, W * 0.33, 0.0, H]
    elif region == "center_area":
        return [W * 0.33, W * 0.66, 0.0, H]
    elif region == "right_area":
        return [W * 0.66, W, 0.0, H]
    elif region == "front_area":
        return [0.0, W, 0.0, H * 0.5]
    elif region == "back_area":
        return [0.0, W, H * 0.5, H]
    elif region == "front_left_area":
        return [0.0, W * 0.33, 0.0, H * 0.5]
    elif region == "front_right_area":
        return [W * 0.66, W, 0.0, H * 0.5]
    elif region == "back_left_area":
        return [0.0, W * 0.33, H * 0.5, H]
    elif region == "back_right_area":
        return [W * 0.66, W, H * 0.5, H]
    return None


def _region_seed(region: str, W: float, H: float) -> Tuple[float, float]:
    if region == "left_area":
        return (W * 0.15, H * 0.5)
    elif region == "center_area":
        return (W * 0.5, H * 0.5)
    elif region == "right_area":
        return (W * 0.85, H * 0.5)
    elif region == "front_area":
        return (W * 0.5, H * 0.2)
    elif region == "back_area":
        return (W * 0.5, H * 0.8)
    elif region == "front_left_area":
        return (W * 0.15, H * 0.2)
    elif region == "front_right_area":
        return (W * 0.85, H * 0.2)
    elif region == "back_left_area":
        return (W * 0.15, H * 0.8)
    elif region == "back_right_area":
        return (W * 0.85, H * 0.8)
    return (W * 0.5, H * 0.5)


def _build_stack_groups_center(
    raw: Dict[str, Dict],
    init_layout: Dict[str, Dict],
    table_size: Tuple[float, float],
):
    """
    Stack_on / Inside_of are treated as binding relations.
    One optimization variable per group root.
    Only x/y are optimized.

    init_layout[obj_id]["init_coordinate"] is expected to be:
        [x, y, rotation]
    Here we only use x/y for optimization seed.
    """
    W, H = [v * 0.01 for v in table_size]

    parent = {}
    for obj_id, obj in raw.items():
        ct = obj.get("contact", {})
        if isinstance(ct, dict) and ct.get("type") in ("Stack_on", "Inside_of"):
            target = ct.get("target")
            if target and target in raw:
                parent[obj_id] = target

    def find_root(node_id: str) -> str:
        seen = set()
        cur = node_id
        while cur in parent:
            if cur in seen:
                break
            seen.add(cur)
            cur = parent[cur]
        return cur

    def depth_of(node_id: str) -> int:
        depth = 0
        seen = {node_id}
        cur = node_id
        while cur in parent:
            nxt = parent[cur]
            if nxt in seen:
                break
            seen.add(nxt)
            cur = nxt
            depth += 1
        return depth

    root_to_members = defaultdict(list)
    object_to_root = {}

    for obj_id in raw.keys():
        root = find_root(obj_id)
        root_to_members[root].append(obj_id)
        object_to_root[obj_id] = root

    groups = {}
    for root_id, members in root_to_members.items():
        members_sorted = sorted(members, key=lambda oid: depth_of(oid))

        root_obj = raw.get(root_id)
        region = (
            root_obj.get("region", "unspecified")
            if root_obj is not None
            else "unspecified"
        )

        fixed_xy = None
        fixed_source = None

        for oid in members_sorted:
            coord = raw[oid].get("coordinate", None)
            if coord is not None:
                fixed_xy = (float(coord[0] * 0.01), float(coord[1] * 0.01))
                fixed_source = oid
                break

        coordinate_range = None
        for oid in members_sorted:
            cr = _parse_coordinate_range(raw[oid].get("coordinate_range", None))
            if cr is None:
                continue
            if coordinate_range is None:
                coordinate_range = {"x": cr["x"], "y": cr["y"]}
            else:
                coordinate_range["x"] = _intersect_axis_range(
                    coordinate_range["x"], cr["x"]
                )
                coordinate_range["y"] = _intersect_axis_range(
                    coordinate_range["y"], cr["y"]
                )

        init_xy = None
        if root_id in init_layout and "init_coordinate" in init_layout[root_id]:
            coord = init_layout[root_id]["init_coordinate"]
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                init_xy = (float(coord[0] * 0.01), float(coord[1] * 0.01))
        else:
            for oid in members_sorted:
                if oid in init_layout and "init_coordinate" in init_layout[oid]:
                    coord = init_layout[oid]["init_coordinate"]
                    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                        init_xy = (float(coord[0]), float(coord[1]))
                        break

        if init_xy is None:
            init_xy = _region_seed(region, W, H)

        region_box = _region_box(region, W, H)

        if fixed_xy is not None:
            init_xy = fixed_xy
        else:
            xb, yb = _effective_axis_bounds(
                {"region_box": region_box, "coordinate_range": coordinate_range},
                W,
                H,
            )
            sx, sy = float(init_xy[0]), float(init_xy[1])
            if xb is not None:
                sx = float(np.clip(sx, xb[0], xb[1]))
            if yb is not None:
                sy = float(np.clip(sy, yb[0], yb[1]))
            init_xy = (sx, sy)

        groups[root_id] = {
            "root_id": root_id,
            "members": members_sorted,
            "depth_order": {oid: depth_of(oid) for oid in members_sorted},
            "region": region,
            "fixed_xy": fixed_xy,
            "fixed_source": fixed_source,
            "coordinate_range": coordinate_range,
            "init_xy": [float(init_xy[0]), float(init_xy[1])],
            "region_box": region_box,
        }

    return groups, object_to_root


def _load_objects_cfg_by_uid(ec_root: str | Path) -> Dict[str, Dict[str, Any]]:
    ec_root = Path(ec_root)
    cfg_path = ec_root / "objects_config_scaled.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"objects_config_scaled.json not found: {cfg_path}")

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    rigid = data.get("rigid_object", [])
    out = {}
    for obj in rigid:
        uid = obj.get("uid")
        if uid:
            out[uid] = obj
    return out


def rotate_mesh_to_optimization_frame(mesh):
    # Match gym_export._sim_world_xy_aabb: sim/layout XY is GLB X and -GLB Z.
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)

    mesh.vertices = mesh.vertices @ R.T
    return mesh


def _load_mesh_simple(mesh_path: str | Path):
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        print(f"[WARN] Mesh file not found: {mesh_path}")
        return None

    try:
        mesh = trimesh.load(mesh_path, force="mesh")
    except Exception as e:
        print(f"[WARN] Failed to load mesh {mesh_path}: {e}")
        return None

    if isinstance(mesh, trimesh.Scene):
        geoms = []
        for geom in mesh.geometry.values():
            if geom is not None and geom.faces is not None and len(geom.faces) > 0:
                geoms.append(geom.copy())
        if not geoms:
            print(f"[WARN] Scene has no geometry: {mesh_path}")
            return None
        mesh = trimesh.util.concatenate(geoms)

    return mesh


def _apply_body_scale(mesh: trimesh.Trimesh, body_scale):
    if body_scale is None:
        return mesh
    try:
        scale = np.asarray(body_scale, dtype=np.float64).reshape(-1)
        if scale.size == 3 and not np.allclose(scale, [1.0, 1.0, 1.0]):
            mesh = mesh.copy()
            # body_scale is stored in sim axes [X, Y, Z].  Trimesh vertices are
            # still in GLB axes here, so convert back to [GLB_X, GLB_Y, GLB_Z].
            mesh.apply_scale([scale[0], scale[2], scale[1]])
    except Exception:
        pass
    return mesh


def _get_cfg_init_rot(cfg: Dict[str, Any]) -> np.ndarray:
    value = cfg.get("init_rot")
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return np.zeros(3, dtype=np.float64)
    try:
        rot = np.asarray(value[:3], dtype=np.float64)
    except (TypeError, ValueError):
        return np.zeros(3, dtype=np.float64)
    if rot.shape != (3,) or not np.all(np.isfinite(rot)):
        return np.zeros(3, dtype=np.float64)
    return rot


def _apply_cfg_rotation(mesh: trimesh.Trimesh, cfg: Dict[str, Any]) -> trimesh.Trimesh:
    init_rot = _get_cfg_init_rot(cfg)
    if np.all(np.abs(init_rot) <= 1.0e-8):
        return mesh
    import trimesh.transformations as tt

    rotated = mesh.copy()
    transform = tt.euler_matrix(
        float(np.deg2rad(init_rot[0])),
        float(np.deg2rad(init_rot[1])),
        float(np.deg2rad(init_rot[2])),
        axes="sxyz",
    )
    rotated.apply_transform(transform)
    return rotated


def _prepare_collision_mesh(
    mesh: trimesh.Trimesh,
    cfg: Dict[str, Any],
) -> trimesh.Trimesh:
    mesh = _apply_body_scale(mesh, cfg.get("body_scale", [1.0, 1.0, 1.0]))
    mesh = rotate_mesh_to_optimization_frame(mesh)
    return _apply_cfg_rotation(mesh, cfg)


def _get_init_rot_deg(state, uid: str) -> float:
    init_layout = getattr(state, "init_layout", {})
    if isinstance(init_layout, dict) and uid in init_layout:
        coord = init_layout[uid].get("init_coordinate", None)
        if isinstance(coord, (list, tuple)) and len(coord) >= 3:
            try:
                return float(coord[2])
            except Exception:
                pass
    return 0.0


def _pose_from_center_xy(mesh: trimesh.Trimesh, x: float, y: float, z: float = 0.0):
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center_xy = 0.5 * (bounds[0, :2] + bounds[1, :2])
    min_z = float(bounds[0, 2])
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [
        float(x) - float(center_xy[0]),
        float(y) - float(center_xy[1]),
        float(z) - min_z,
    ]
    return pose


def _world_bounds(mesh: trimesh.Trimesh, pose: np.ndarray):
    m = mesh.copy()
    m.apply_transform(pose)
    bmin, bmax = m.bounds
    bmin = np.asarray(bmin, dtype=float).ravel()
    bmax = np.asarray(bmax, dtype=float).ravel()
    return bmin, bmax


def _render_collision_mesh_topdown(
    *,
    output_path: Path,
    mesh_dict: Dict[str, trimesh.Trimesh],
    pose_dict: Dict[str, np.ndarray],
    table_size: Tuple[float, float],
    title: str,
) -> None:
    if not mesh_dict:
        return
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
    except Exception as exc:
        print(f"[WARN] Failed to import matplotlib for collision render: {exc}")
        return

    W, H = [float(v) * 0.01 for v in table_size]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(mesh_dict), 1)))

    all_xy: list[np.ndarray] = []
    for index, uid in enumerate(sorted(mesh_dict)):
        mesh = mesh_dict[uid].copy()
        pose = pose_dict.get(uid)
        if pose is not None:
            mesh.apply_transform(pose)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            continue

        xy = vertices[:, :2]
        all_xy.append(xy)
        polygons = xy[faces]
        color = colors[index % len(colors)]
        collection = PolyCollection(
            polygons,
            facecolors=[color],
            edgecolors=[(0.0, 0.0, 0.0, 0.18)],
            linewidths=0.15,
            alpha=0.34,
        )
        ax.add_collection(collection)

        bounds = mesh.bounds
        center_xy = 0.5 * (bounds[0, :2] + bounds[1, :2])
        ax.text(
            float(center_xy[0]),
            float(center_xy[1]),
            uid,
            fontsize=6,
            ha="center",
            va="center",
            color="black",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.0},
        )

    ax.add_patch(
        plt.Rectangle(
            (0.0, 0.0),
            W,
            H,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
            linestyle="--",
        )
    )
    ax.set_title(title)
    ax.set_xlabel("SA/layout X (m)")
    ax.set_ylabel("SA/layout Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    if all_xy:
        stacked = np.vstack(all_xy)
        min_xy = np.minimum(stacked.min(axis=0), np.array([0.0, 0.0]))
        max_xy = np.maximum(stacked.max(axis=0), np.array([W, H]))
    else:
        min_xy = np.array([0.0, 0.0])
        max_xy = np.array([W, H])
    span = np.maximum(max_xy - min_xy, 1.0e-3)
    pad = np.maximum(span * 0.08, 0.02)
    ax.set_xlim(float(min_xy[0] - pad[0]), float(max_xy[0] + pad[0]))
    ax.set_ylim(float(min_xy[1] - pad[1]), float(max_xy[1] + pad[1]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def _relation_direction_vector(relation_name: str) -> Optional[np.ndarray]:
    if relation_name == "left_of":
        return np.array([-1.0, 0.0], dtype=float)
    if relation_name == "right_of":
        return np.array([1.0, 0.0], dtype=float)
    if relation_name == "front_of":
        return np.array([0.0, -1.0], dtype=float)
    if relation_name in {"back_of", "behind"}:
        return np.array([0.0, 1.0], dtype=float)
    return None


def _build_relation_direction_map(
    object_items: Dict[str, Dict[str, Any]],
    object_to_group: Dict[str, str],
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    relation_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for src_id, obj in object_items.items():
        src_gid = object_to_group.get(src_id, src_id)
        rel = obj.get("relation", {}) or {}
        if not isinstance(rel, dict):
            continue
        for relation_name in ("left_of", "right_of", "front_of", "back_of"):
            direction = _relation_direction_vector(relation_name)
            if direction is None:
                continue
            for target in rel.get(relation_name, []) or []:
                if _is_coordinate_point(target):
                    continue
                target_id = str(target)
                target_gid = object_to_group.get(target_id, target_id)
                if not target_gid or src_gid == target_gid:
                    continue
                relation_map[(src_gid, target_gid)].append(
                    {
                        "source": src_id,
                        "target": target_id,
                        "relation": relation_name,
                        "direction": direction.tolist(),
                    }
                )
    return relation_map


def _relation_candidates_for_pair(
    group_a: str,
    group_b: str,
    relation_direction_map: Optional[Dict[Tuple[str, str], List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    if not relation_direction_map:
        return []
    candidates: List[Dict[str, Any]] = []
    for item in relation_direction_map.get((group_a, group_b), []) or []:
        candidates.append(
            {**item, "direction": np.asarray(item["direction"], dtype=float).tolist()}
        )
    for item in relation_direction_map.get((group_b, group_a), []) or []:
        candidates.append(
            {
                **item,
                "source": item.get("target", ""),
                "target": item.get("source", ""),
                "direction": (-np.asarray(item["direction"], dtype=float)).tolist(),
                "relation": f"inverse_{item.get('relation', '')}",
            }
        )
    return candidates


def _pair_separation_from_bounds(
    bmin_a,
    bmax_a,
    bmin_b,
    bmax_b,
    *,
    direction_2d: Optional[np.ndarray] = None,
    margin: float = 0.02,
):
    bmin_a = np.asarray(bmin_a, dtype=float)
    bmax_a = np.asarray(bmax_a, dtype=float)
    bmin_b = np.asarray(bmin_b, dtype=float)
    bmax_b = np.asarray(bmax_b, dtype=float)

    overlap_x = min(bmax_a[0], bmax_b[0]) - max(bmin_a[0], bmin_b[0])
    overlap_y = min(bmax_a[1], bmax_b[1]) - max(bmin_a[1], bmin_b[1])
    overlap_z = min(bmax_a[2], bmax_b[2]) - max(bmin_a[2], bmin_b[2])

    if overlap_x <= 0 or overlap_y <= 0 or overlap_z <= 0:
        return None, None, None, None

    center_a = 0.5 * (bmin_a + bmax_a)
    center_b = 0.5 * (bmin_b + bmax_b)

    if direction_2d is not None:
        direction = np.asarray(direction_2d, dtype=float).reshape(2)
        if np.linalg.norm(direction) < 1e-8:
            return None, None, None, None
        if abs(float(direction[0])) >= abs(float(direction[1])):
            sign = 1.0 if float(direction[0]) >= 0.0 else -1.0
            axis = 0
            direction_2d = np.array([sign, 0.0], dtype=float)
            overlap_axis = overlap_x
        else:
            sign = 1.0 if float(direction[1]) >= 0.0 else -1.0
            axis = 1
            direction_2d = np.array([0.0, sign], dtype=float)
            overlap_axis = overlap_y
        half_a = 0.5 * float(bmax_a[axis] - bmin_a[axis])
        half_b = 0.5 * float(bmax_b[axis] - bmin_b[axis])
        required_gap = half_a + half_b + float(margin)
        current_gap = sign * float(center_a[axis] - center_b[axis])
        push_distance = max(0.0, required_gap - current_gap)
        severity = float(overlap_axis)
        return direction_2d, required_gap, push_distance, severity

    if overlap_x <= overlap_y:
        sign = 1.0 if center_a[0] >= center_b[0] else -1.0
        direction_2d = np.array([sign, 0.0], dtype=float)
        half_a = 0.5 * float(bmax_a[0] - bmin_a[0])
        half_b = 0.5 * float(bmax_b[0] - bmin_b[0])
        required_gap = half_a + half_b + float(margin)
        current_gap = sign * float(center_a[0] - center_b[0])
        push_distance = max(0.0, required_gap - current_gap)
        severity = float(overlap_x)
    else:
        sign = 1.0 if center_a[1] >= center_b[1] else -1.0
        direction_2d = np.array([0.0, sign], dtype=float)
        half_a = 0.5 * float(bmax_a[1] - bmin_a[1])
        half_b = 0.5 * float(bmax_b[1] - bmin_b[1])
        required_gap = half_a + half_b + float(margin)
        current_gap = sign * float(center_a[1] - center_b[1])
        push_distance = max(0.0, required_gap - current_gap)
        severity = float(overlap_y)

    return direction_2d, required_gap, push_distance, severity


def _clamp_group_center(
    gid: str, xy: np.ndarray, groups: Dict[str, Dict], table_size: Tuple[float, float]
) -> np.ndarray:
    info = groups[gid]
    if info.get("fixed_xy") is not None:
        return np.asarray(info["fixed_xy"], dtype=float)

    W, H = [v * 0.01 for v in table_size]
    x, y = float(xy[0]), float(xy[1])

    x = float(np.clip(x, 0.0, W))
    y = float(np.clip(y, 0.0, H))

    xb, yb = _effective_axis_bounds(info, W, H)
    if xb is not None:
        x = float(np.clip(x, xb[0], xb[1]))
    if yb is not None:
        y = float(np.clip(y, yb[0], yb[1]))

    return np.array([x, y], dtype=float)


def _print_collision_item(item: Dict[str, Any]):
    print(
        f"[COLLISION] {item['a']}({item['group_a']}) <-> {item['b']}({item['group_b']}) | "
        f"dir={item['direction_2d'].tolist()} | required_sep={item['required_sep']:.4f} | "
        f"severity={item['severity']:.4f} | "
        f"overlaps: x={item['overlap_x']:.4f}, y={item['overlap_y']:.4f}, z={item['overlap_z']:.4f}"
    )


def _greedy_push_apart(
    current_centers: Dict[str, np.ndarray],
    collisions: List[Dict[str, Any]],
    groups: Dict[str, Dict],
    table_size: Tuple[float, float],
    push_scale: float = 0.05,
):
    new_centers = {
        gid: np.asarray(xy, dtype=float).reshape(2).copy()
        for gid, xy in current_centers.items()
    }

    for item in collisions:
        ga = item["group_a"]
        gb = item["group_b"]

        dir2d = np.asarray(item["direction_2d"], dtype=float).reshape(2)
        norm = float(np.linalg.norm(dir2d))
        if norm < 1e-8:
            continue
        dir2d = dir2d / norm

        step = max(0.0, float(item.get("push_distance", item["required_sep"])))
        step *= float(push_scale)

        a_fixed = groups[ga].get("fixed_xy") is not None
        b_fixed = groups[gb].get("fixed_xy") is not None

        if a_fixed and b_fixed:
            continue
        elif a_fixed and not b_fixed:
            new_centers[gb] -= dir2d * step
        elif b_fixed and not a_fixed:
            new_centers[ga] += dir2d * step
        else:
            new_centers[ga] += dir2d * (step * 0.5)
            new_centers[gb] -= dir2d * (step * 0.5)

    for gid in list(new_centers.keys()):
        new_centers[gid] = _clamp_group_center(
            gid, new_centers[gid], groups, table_size
        )

    return new_centers


def _local_z_interval(mesh: trimesh.Trimesh, rot_deg: float):
    m = mesh.copy()
    center = np.asarray(m.bounds.mean(axis=0), dtype=np.float64)

    theta = np.deg2rad(float(rot_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))

    t1 = np.eye(4, dtype=np.float64)
    t1[:3, 3] = -center

    rz = np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    m.apply_transform(rz @ t1)
    bmin, bmax = m.bounds
    return float(bmin[2]), float(bmax[2])


def _refine_stack_group_z(
    state: SceneState,
    ec_root: str | Path,
    optimized_layout: Dict[str, Dict],
    groups: Dict[str, Dict],
    z_gap: float = 0.01,
):
    cfg_by_uid = _load_objects_cfg_by_uid(ec_root)

    final_layout = copy.deepcopy(optimized_layout)

    mesh_cache = {}
    for uid in final_layout.keys():
        cfg = cfg_by_uid.get(uid)
        if cfg is None:
            continue

        shape = cfg.get("shape", {})
        fpath = shape.get("fpath")
        if not fpath:
            continue

        mesh_path = Path(ec_root) / fpath
        mesh = _load_mesh_simple(mesh_path)
        if mesh is None:
            continue

        mesh = _apply_body_scale(mesh, cfg.get("body_scale", [1.0, 1.0, 1.0]))
        mesh = rotate_mesh_to_optimization_frame(mesh)
        mesh_cache[uid] = mesh

    for gid, group in groups.items():
        members = group["members"]
        if not members:
            continue

        root_id = members[0]
        root_z = 0.0

        if root_id in final_layout and "z" in final_layout[root_id]:
            try:
                root_z = float(final_layout[root_id]["z"])
            except Exception:
                pass

        prev_top = None
        for idx, uid in enumerate(members):
            if uid not in mesh_cache:
                continue

            rot_deg = _get_init_rot_deg(state, uid)
            local_min_z, local_max_z = _local_z_interval(
                mesh_cache[uid], rot_deg=rot_deg
            )

            if idx == 0:
                z_center = float(root_z)
            else:
                z_center = float(prev_top - local_min_z + z_gap)

            prev_top = z_center + local_max_z

            if uid not in final_layout:
                final_layout[uid] = {}

            final_layout[uid]["z"] = float(z_center)
            if "center_2d" in final_layout[uid]:
                final_layout[uid]["center_3d"] = [
                    float(final_layout[uid]["center_2d"][0]),
                    float(final_layout[uid]["center_2d"][1]),
                    float(z_center),
                ]

    return final_layout


def _solve_group_model(
    model: Dict[str, Any], seed_centers: Optional[Dict[str, np.ndarray]] = None
):
    group_ids = model["group_ids"]
    group_index = model["group_index"]
    groups = model["groups"]

    if seed_centers is None:
        seed_centers = {
            gid: np.asarray(groups[gid]["init_xy"], dtype=float) for gid in group_ids
        }

    x0 = []
    for gid in group_ids:
        xy = np.asarray(
            seed_centers.get(gid, groups[gid]["init_xy"]), dtype=float
        ).reshape(2)
        x0.extend([float(xy[0]), float(xy[1])])
    x0 = np.asarray(x0, dtype=float)

    def unpack(xvec):
        out = {}
        for gid, idx in group_index.items():
            out[gid] = np.array([xvec[2 * idx], xvec[2 * idx + 1]], dtype=float)
        return out

    def objective(xvec):
        coords = unpack(xvec)
        loss = 0.0

        for gid in group_ids:
            init_xy = np.asarray(
                seed_centers.get(gid, groups[gid]["init_xy"]), dtype=float
            )
            loss += 5 * float(np.sum((coords[gid] - init_xy) ** 2))

        min_dist = 0.01
        for i in range(len(group_ids)):
            for j in range(i + 1, len(group_ids)):
                a = group_ids[i]
                b = group_ids[j]
                d = float(np.linalg.norm(coords[a] - coords[b]))
                overlap = max(0.0, min_dist - d)
                loss += 0.05 * (overlap**2)

        return loss

    constraints = []

    for row, b in zip(model["A_ub"], model["b_ub"]):
        row = np.asarray(row, dtype=float)
        b = float(b)
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda xvec, row=row, b=b: b - float(np.dot(row, xvec)),
            }
        )

    for row, b in zip(model["A_eq"], model["b_eq"]):
        row = np.asarray(row, dtype=float)
        b = float(b)
        constraints.append(
            {
                "type": "eq",
                "fun": lambda xvec, row=row, b=b: float(np.dot(row, xvec)) - b,
            }
        )

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-6, "disp": False},
    )

    solved = unpack(result.x)
    return result, solved


def _build_collision_scene(
    state: SceneState | Tempo_SceneState,
    ec_root: str | Path,
    group_centers: Dict[str, np.ndarray],
    object_to_group: Dict[str, str],
):
    cfg_by_uid = _load_objects_cfg_by_uid(ec_root)

    try:
        cm = trimesh.collision.CollisionManager()
    except Exception:
        cm = None

    mesh_dict = {}
    pose_dict = {}

    for uid, cfg in cfg_by_uid.items():
        gid = object_to_group.get(uid)
        if gid not in group_centers:
            continue

        shape = cfg.get("shape", {})
        fpath = shape.get("fpath")
        if not fpath:
            continue

        mesh_path = Path(ec_root) / fpath
        mesh = _load_mesh_simple(mesh_path)
        if mesh is None:
            continue

        mesh = _prepare_collision_mesh(mesh, cfg)

        center_xy = np.asarray(group_centers[gid], dtype=float).reshape(2)
        pose = _pose_from_center_xy(mesh, float(center_xy[0]), float(center_xy[1]))

        mesh_dict[uid] = mesh
        pose_dict[uid] = pose

        if cm is not None:
            try:
                cm.add_object(uid, mesh, transform=pose)
            except Exception:
                pass

    return cm, mesh_dict, pose_dict


def _detect_collision_pairs(
    cm,
    mesh_dict: Dict[str, trimesh.Trimesh],
    pose_dict: Dict[str, np.ndarray],
    object_to_group: Dict[str, str],
    relation_direction_map: Optional[Dict[Tuple[str, str], List[Dict[str, Any]]]] = None,
    separation_margin: float = 0.02,
):
    results = []
    ids = list(mesh_dict.keys())

    seen = set()
    if cm is not None:
        for uid in ids:
            try:
                names = cm.in_collision_other(
                    mesh_dict[uid], transform=pose_dict[uid], return_names=True
                )
            except Exception:
                names = []
            for other in names or []:
                if other == uid:
                    continue
                key = tuple(sorted((uid, other)))
                if key in seen:
                    continue
                seen.add(key)

                bmin_a, bmax_a = _world_bounds(mesh_dict[uid], pose_dict[uid])
                bmin_b, bmax_b = _world_bounds(mesh_dict[other], pose_dict[other])
                group_a = object_to_group.get(uid, uid)
                group_b = object_to_group.get(other, other)

                candidates = _relation_candidates_for_pair(
                    group_a, group_b, relation_direction_map
                )
                relation_choice = None
                best_sep = None
                for candidate in candidates:
                    sep = _pair_separation_from_bounds(
                        bmin_a,
                        bmax_a,
                        bmin_b,
                        bmax_b,
                        direction_2d=np.asarray(candidate["direction"], dtype=float),
                        margin=separation_margin,
                    )
                    if sep[0] is None:
                        continue
                    if best_sep is None or float(sep[2]) < float(best_sep[2]):
                        best_sep = sep
                        relation_choice = candidate
                if best_sep is None:
                    best_sep = _pair_separation_from_bounds(
                        bmin_a,
                        bmax_a,
                        bmin_b,
                        bmax_b,
                        margin=separation_margin,
                    )
                dir2d, required_sep, push_distance, severity = best_sep
                if dir2d is None:
                    continue

                overlap_x = min(bmax_a[0], bmax_b[0]) - max(bmin_a[0], bmin_b[0])
                overlap_y = min(bmax_a[1], bmax_b[1]) - max(bmin_a[1], bmin_b[1])
                overlap_z = min(bmax_a[2], bmax_b[2]) - max(bmin_a[2], bmin_b[2])

                results.append(
                    {
                        "a": uid,
                        "b": other,
                        "group_a": group_a,
                        "group_b": group_b,
                        "direction_2d": dir2d,
                        "required_sep": required_sep,
                        "push_distance": push_distance,
                        "severity": severity,
                        "relation_driven": relation_choice is not None,
                        "relation_choice": relation_choice,
                        "overlap_x": float(overlap_x),
                        "overlap_y": float(overlap_y),
                        "overlap_z": float(overlap_z),
                    }
                )
    else:
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = ids[i]
                b = ids[j]
                bmin_a, bmax_a = _world_bounds(mesh_dict[a], pose_dict[a])
                bmin_b, bmax_b = _world_bounds(mesh_dict[b], pose_dict[b])
                group_a = object_to_group.get(a, a)
                group_b = object_to_group.get(b, b)
                candidates = _relation_candidates_for_pair(
                    group_a, group_b, relation_direction_map
                )
                relation_choice = None
                best_sep = None
                for candidate in candidates:
                    sep = _pair_separation_from_bounds(
                        bmin_a,
                        bmax_a,
                        bmin_b,
                        bmax_b,
                        direction_2d=np.asarray(candidate["direction"], dtype=float),
                        margin=separation_margin,
                    )
                    if sep[0] is None:
                        continue
                    if best_sep is None or float(sep[2]) < float(best_sep[2]):
                        best_sep = sep
                        relation_choice = candidate
                if best_sep is None:
                    best_sep = _pair_separation_from_bounds(
                        bmin_a,
                        bmax_a,
                        bmin_b,
                        bmax_b,
                        margin=separation_margin,
                    )
                dir2d, required_sep, push_distance, severity = best_sep
                if dir2d is None:
                    continue
                overlap_x = min(bmax_a[0], bmax_b[0]) - max(bmin_a[0], bmin_b[0])
                overlap_y = min(bmax_a[1], bmax_b[1]) - max(bmin_a[1], bmin_b[1])
                overlap_z = min(bmax_a[2], bmax_b[2]) - max(bmin_a[2], bmin_b[2])
                results.append(
                    {
                        "a": a,
                        "b": b,
                        "group_a": group_a,
                        "group_b": group_b,
                        "direction_2d": dir2d,
                        "required_sep": required_sep,
                        "push_distance": push_distance,
                        "severity": severity,
                        "relation_driven": relation_choice is not None,
                        "relation_choice": relation_choice,
                        "overlap_x": float(overlap_x),
                        "overlap_y": float(overlap_y),
                        "overlap_z": float(overlap_z),
                    }
                )

    results.sort(key=lambda item: float(item["severity"]), reverse=True)
    return results


def _add_pair_separation_constraint(
    model: Dict[str, Any],
    ga: str,
    gb: str,
    direction_2d: np.ndarray,
    required_sep: float,
):
    group_index = model["group_index"]
    if ga not in group_index or gb not in group_index:
        return

    ia = group_index[ga]
    ib = group_index[gb]

    dx, dy = float(direction_2d[0]), float(direction_2d[1])
    row = np.zeros(2 * len(model["group_ids"]), dtype=float)
    row[2 * ia] = -dx
    row[2 * ia + 1] = -dy
    row[2 * ib] = dx
    row[2 * ib + 1] = dy

    model["A_ub"].append(row.tolist())
    model["b_ub"].append(float(-required_sep))


def _build_work_model(base_model: Dict[str, Any], collision_terms: List[Dict[str, Any]]):
    model = copy.deepcopy(base_model)
    for term in collision_terms:
        _add_pair_separation_constraint(
            model,
            term["ga"],
            term["gb"],
            np.asarray(term["direction_2d"], dtype=float),
            float(term["margin"]),
        )
    return model


def _group_half_extents_xy(
    *,
    ec_root: str | Path,
    groups: Dict[str, Dict],
    object_to_group: Dict[str, str],
) -> Dict[str, np.ndarray]:
    cfg_by_uid = _load_objects_cfg_by_uid(ec_root)
    extents: Dict[str, np.ndarray] = {
        gid: np.zeros(2, dtype=np.float64) for gid in groups
    }
    for uid, cfg in cfg_by_uid.items():
        gid = object_to_group.get(uid)
        if gid not in extents:
            continue
        shape = cfg.get("shape", {})
        fpath = shape.get("fpath")
        if not fpath:
            continue
        mesh_path = Path(ec_root) / fpath
        mesh = _load_mesh_simple(mesh_path)
        if mesh is None:
            continue
        mesh = _prepare_collision_mesh(mesh, cfg)
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        if bounds.shape != (2, 3):
            continue
        half_xy = 0.5 * np.maximum(bounds[1, :2] - bounds[0, :2], 0.0)
        extents[gid] = np.maximum(extents[gid], half_xy)
    return extents


def run_node_3_5(state: Tempo_SceneState, ec_root: str | Path) -> Tempo_SceneState:
    print(">>> Node 3.5: compiling center-point model and optimizing...")

    raw = state.raw_object_dict
    if not raw or state.table_size is None or not state.init_layout:
        state.messages.append("Node 3.5 skipped: missing inputs")
        return state

    object_items = {
        obj_id: obj for obj_id, obj in raw.items() if not obj_id.startswith("table_")
    }

    W, H = [v * 0.01 for v in state.table_size]

    groups, object_to_group = _build_stack_groups_center(
        object_items, state.init_layout, state.table_size
    )
    state.stack_groups = groups
    group_half_extents = _group_half_extents_xy(
        ec_root=ec_root,
        groups=groups,
        object_to_group=object_to_group,
    )
    relation_direction_map = _build_relation_direction_map(
        object_items,
        object_to_group,
    )

    group_ids = list(groups.keys())
    group_index = {gid: i for i, gid in enumerate(group_ids)}

    def gid_of(obj_id: str) -> str:
        return object_to_group.get(obj_id, obj_id)

    A_ub, b_ub = [], []
    A_eq, b_eq = [], []
    relation_terms = []
    clearance = 0.03

    def add_ub(coeffs: Dict[int, float], rhs: float):
        row = np.zeros(2 * len(group_ids), dtype=float)
        for idx, val in coeffs.items():
            row[idx] = float(val)
        A_ub.append(row.tolist())
        b_ub.append(float(rhs))

    def add_eq(coeffs: Dict[int, float], rhs: float):
        row = np.zeros(2 * len(group_ids), dtype=float)
        for idx, val in coeffs.items():
            row[idx] = float(val)
        A_eq.append(row.tolist())
        b_eq.append(float(rhs))

    for gid in group_ids:
        i = group_index[gid]

        add_ub({2 * i: -1.0}, 0.0)
        add_ub({2 * i: 1.0}, W)
        add_ub({2 * i + 1: -1.0}, 0.0)
        add_ub({2 * i + 1: 1.0}, H)

        if groups[gid]["fixed_xy"] is not None:
            fx, fy = groups[gid]["fixed_xy"]
            add_eq({2 * i: 1.0}, fx)
            add_eq({2 * i + 1: 1.0}, fy)
            continue

        xb, yb = _effective_axis_bounds(groups[gid], W, H)
        if xb is not None:
            add_ub({2 * i: -1.0}, -xb[0])
            add_ub({2 * i: 1.0}, xb[1])
        if yb is not None:
            add_ub({2 * i + 1: -1.0}, -yb[0])
            add_ub({2 * i + 1: 1.0}, yb[1])

    def add_relation_constraints(
        src_i: int,
        src_gid: str,
        src_id: str,
        targets,
        rel_type: str,
    ):
        for tgt in targets or []:
            if _is_coordinate_point(tgt):
                px = float(tgt[0]) * 0.01
                py = float(tgt[1]) * 0.01
                if rel_type == "left_of":
                    add_ub({2 * src_i: 1.0}, px - clearance)
                elif rel_type == "right_of":
                    add_ub({2 * src_i: -1.0}, -(px + clearance))
                elif rel_type == "front_of":
                    add_ub({2 * src_i + 1: 1.0}, py - clearance)
                elif rel_type == "back_of":
                    add_ub({2 * src_i + 1: -1.0}, -(py + clearance))
                relation_terms.append(
                    {
                        "source": src_id,
                        "target_point": [px, py],
                        "type": rel_type,
                        "source_group": src_gid,
                        "target_group": None,
                        "gap": clearance,
                    }
                )
                continue

            tgt_id = tgt
            tgt_gid = gid_of(tgt_id)
            if tgt_gid == src_gid or tgt_gid not in group_index:
                continue
            tgt_j = group_index[tgt_gid]
            src_half = group_half_extents.get(src_gid, np.zeros(2, dtype=np.float64))
            tgt_half = group_half_extents.get(tgt_gid, np.zeros(2, dtype=np.float64))
            if rel_type == "left_of":
                gap = float(src_half[0] + tgt_half[0])
                add_ub({2 * src_i: 1.0, 2 * tgt_j: -1.0}, -gap)
            elif rel_type == "right_of":
                gap = float(src_half[0] + tgt_half[0])
                add_ub({2 * tgt_j: 1.0, 2 * src_i: -1.0}, -gap)
            elif rel_type == "front_of":
                gap = float(src_half[1] + tgt_half[1])
                add_ub({2 * src_i + 1: 1.0, 2 * tgt_j + 1: -1.0}, -gap)
            elif rel_type == "back_of":
                gap = float(src_half[1] + tgt_half[1])
                add_ub({2 * tgt_j + 1: 1.0, 2 * src_i + 1: -1.0}, -gap)
            else:
                gap = clearance
            relation_terms.append(
                {
                    "source": src_id,
                    "target": tgt_id,
                    "type": rel_type,
                    "source_group": src_gid,
                    "target_group": tgt_gid,
                    "gap": gap,
                }
            )

    for src_id, obj in object_items.items():
        src_gid = gid_of(src_id)
        if src_gid not in group_index:
            continue
        src_i = group_index[src_gid]

        rel = obj.get("relation", {}) or {}
        if not isinstance(rel, dict):
            rel = {}

        add_relation_constraints(
            src_i, src_gid, src_id, rel.get("left_of", []), "left_of"
        )
        add_relation_constraints(
            src_i, src_gid, src_id, rel.get("right_of", []), "right_of"
        )
        add_relation_constraints(
            src_i, src_gid, src_id, rel.get("front_of", []), "front_of"
        )
        add_relation_constraints(
            src_i, src_gid, src_id, rel.get("back_of", []), "back_of"
        )

        if rel.get("towards_to"):
            relation_terms.append(
                {
                    "source": src_id,
                    "type": "towards_to",
                    "targets": rel.get("towards_to", []),
                    "ignored_in_stage": True,
                }
            )

    base_model = {
        "variable_type": "group_root_center_2d",
        "group_ids": group_ids,
        "variable_order": group_ids,
        "group_index": group_index,
        "object_to_group": object_to_group,
        "groups": groups,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "A_eq": A_eq,
        "b_eq": b_eq,
        "relation_terms": relation_terms,
        "table_bounds": {"x_range": [0.0, W], "y_range": [0.0, H]},
        "collision_terms": [],
        "notes": [
            "Center-point optimization only.",
            "Stack_on and Inside_of are bound by group roots.",
            "rotation is not part of optimization variables.",
            "rotation is only used when building collision geometry.",
            "collision constraints are added only to temporary working models.",
            "coordinate_range (cm->m) is added as per-axis hard bounds and "
            "overrides the coarse region box.",
            "relation targets may be object ids or absolute points [x, y] (cm); "
            "points become single-variable bounds.",
        ],
    }

    state.optimization_model = base_model

    result, solved = _solve_group_model(base_model, seed_centers=None)
    if not result.success:
        state.messages.append(
            f"Node 3.5 warning: optimizer did not fully converge: {result.message}"
        )
        print(f"[WARN] optimizer did not fully converge: {result.message}")

    current_centers = {
        gid: np.asarray(solved[gid], dtype=float).reshape(2) for gid in group_ids
    }

    max_rounds = 8
    max_added_pairs = 64
    collision_margin = 0.02

    seen_pair_keys = set()
    collision_terms_history: List[Dict[str, Any]] = []
    debug_renders: Dict[str, str] = {}

    cm_initial, mesh_dict_initial, pose_dict_initial = _build_collision_scene(
        state,
        ec_root,
        current_centers,
        base_model["object_to_group"],
    )
    before_render_path = Path(ec_root) / "collision_mesh_before.png"
    _render_collision_mesh_topdown(
        output_path=before_render_path,
        mesh_dict=mesh_dict_initial,
        pose_dict=pose_dict_initial,
        table_size=state.table_size,
        title="SA collision mesh before collision refinement",
    )
    if before_render_path.is_file():
        debug_renders["collision_mesh_before"] = str(before_render_path)

    for round_idx in range(max_rounds):
        print(f"\n>>> Collision refinement round {round_idx + 1}")

        cm, mesh_dict, pose_dict = _build_collision_scene(
            state,
            ec_root,
            current_centers,
            base_model["object_to_group"],
        )
        collisions = _detect_collision_pairs(
            cm,
            mesh_dict,
            pose_dict,
            base_model["object_to_group"],
            relation_direction_map=relation_direction_map,
            separation_margin=collision_margin,
        )

        if not collisions:
            state.messages.append(
                f"Node 3.5 collision refinement finished at round {round_idx}: no collisions"
            )
            break

        added_this_round = 0

        for item in collisions[:max_added_pairs]:
            ga = item["group_a"]
            gb = item["group_b"]
            if ga == gb:
                continue

            pair_key = tuple(sorted((ga, gb)))
            if pair_key in seen_pair_keys:
                continue

            dir2d = np.asarray(item["direction_2d"], dtype=float).reshape(2)
            if np.linalg.norm(dir2d) < 1e-8:
                continue

            required_gap = float(item["required_sep"])

            collision_terms_history.append(
                {
                    "ga": ga,
                    "gb": gb,
                    "direction_2d": dir2d.tolist(),
                    "margin": required_gap,
                    "push_distance": float(item.get("push_distance", 0.0)),
                    "relation_driven": bool(item.get("relation_driven", False)),
                    "relation_choice": item.get("relation_choice"),
                }
            )
            seen_pair_keys.add(pair_key)
            added_this_round += 1

        if added_this_round == 0:
            print(
                "[COLLISION] no new constraints from current round, using greedy fallback"
            )
            current_centers = _greedy_push_apart(
                current_centers,
                collisions,
                groups,
                state.table_size,
                push_scale=1.0,
            )
            continue

        work_model = _build_work_model(base_model, collision_terms_history)

        result, solved = _solve_group_model(work_model, seed_centers=current_centers)
        if not result.success:
            msg = f"Node 3.5 collision refinement warning round {round_idx}: optimizer failed: {result.message}"
            state.messages.append(msg)
            print(f"[WARN] {msg}")

            current_centers = _greedy_push_apart(
                current_centers,
                collisions,
                groups,
                state.table_size,
                push_scale=1.0,
            )
            continue

        new_centers = {
            gid: np.asarray(solved[gid], dtype=float).reshape(2)
            for gid in base_model["group_ids"]
        }

        max_move = 0.0
        for gid in base_model["group_ids"]:
            delta = np.linalg.norm(new_centers[gid] - current_centers[gid])
            max_move = max(max_move, float(delta))

        current_centers = new_centers
        state.messages.append(
            f"Node 3.5 collision refinement round {round_idx + 1}: "
            f"constraints={len(collision_terms_history)}, max_move={max_move:.4f}"
        )
        print(
            f"[SOLVE] done | constraints={len(collision_terms_history)} | max_move={max_move:.4f}"
        )

        cm2, mesh_dict2, pose_dict2 = _build_collision_scene(
            state,
            ec_root,
            current_centers,
            base_model["object_to_group"],
        )
        post_collisions = _detect_collision_pairs(
            cm2,
            mesh_dict2,
            pose_dict2,
            base_model["object_to_group"],
            relation_direction_map=relation_direction_map,
            separation_margin=collision_margin,
        )

        if not post_collisions:
            state.messages.append(
                f"Node 3.5 collision refinement converged at round {round_idx + 1}"
            )
            print("[COLLISION] fully resolved")
            break
        else:
            print(
                f"[COLLISION] still unresolved after solve: {len(post_collisions)} pairs"
            )
            for item in post_collisions:
                _print_collision_item(item)

        if max_move < 1e-4:
            break

    for safety_round in range(5):
        cm_s, mesh_dict_s, pose_dict_s = _build_collision_scene(
            state,
            ec_root,
            current_centers,
            base_model["object_to_group"],
        )
        remaining = _detect_collision_pairs(
            cm_s,
            mesh_dict_s,
            pose_dict_s,
            base_model["object_to_group"],
            relation_direction_map=relation_direction_map,
            separation_margin=collision_margin,
        )
        if not remaining:
            break
        print(
            f"[COLLISION] safety push round {safety_round + 1}, remaining={len(remaining)}"
        )
        current_centers = _greedy_push_apart(
            current_centers,
            remaining,
            groups,
            state.table_size,
            push_scale=1.0,
        )

    optimized_layout = {}
    for gid in group_ids:
        root_xy = current_centers[gid]
        members = groups[gid]["members"]

        for stack_level, obj_id in enumerate(members):
            optimized_layout[obj_id] = {
                "group_root": gid,
                "stack_level": stack_level,
                "is_fixed": groups[gid]["fixed_xy"] is not None,
                "center_2d": [float(root_xy[0]), float(root_xy[1])],
                "region": groups[gid]["region"],
                "rotation_deg": float(_get_init_rot_deg(state, obj_id)),
                "contact": object_items[obj_id].get("contact", {}),
                "relation": object_items[obj_id].get("relation", {}),
            }

    optimized_layout = _refine_stack_group_z(
        state=state,
        ec_root=ec_root,
        optimized_layout=optimized_layout,
        groups=groups,
        z_gap=0.01,
    )

    state.optimization_model = base_model
    state.optimization_model["collision_terms"] = collision_terms_history
    state.optimization_model["debug_renders"] = debug_renders
    state.optimized_layout = optimized_layout
    state.optimized_group_centers = {
        gid: [float(v[0]), float(v[1])] for gid, v in current_centers.items()
    }

    cm_f, mesh_dict_f, pose_dict_f = _build_collision_scene(
        state,
        ec_root,
        current_centers,
        base_model["object_to_group"],
    )
    after_render_path = Path(ec_root) / "collision_mesh_after.png"
    _render_collision_mesh_topdown(
        output_path=after_render_path,
        mesh_dict=mesh_dict_f,
        pose_dict=pose_dict_f,
        table_size=state.table_size,
        title="SA collision mesh after collision refinement",
    )
    if after_render_path.is_file():
        debug_renders["collision_mesh_after"] = str(after_render_path)
        state.optimization_model["debug_renders"] = debug_renders

    final_collisions = _detect_collision_pairs(
        cm_f,
        mesh_dict_f,
        pose_dict_f,
        base_model["object_to_group"],
        relation_direction_map=relation_direction_map,
        separation_margin=collision_margin,
    )
    if final_collisions:
        state.messages.append(
            f"Node 3.5 finished with remaining collisions: {len(final_collisions)}"
        )
        print(f"[COLLISION] final remaining collisions: {len(final_collisions)}")
        for item in final_collisions:
            _print_collision_item(item)
    else:
        state.messages.append("Node 3.5 completed with no collisions")
        print("[COLLISION] final state clean")

    state.messages.append(
        f"Node 3.5 completed: optimized {len(optimized_layout)} objects"
    )
    return state
