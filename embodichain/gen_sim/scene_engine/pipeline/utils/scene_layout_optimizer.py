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

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from embodichain.gen_sim.scene_engine.core.scene_graph import SceneGraphRelation
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_generation_utils import (
    layout_object_to_transform_matrix,
    load_glb_mesh,
    transform_matrix_to_layout_object,
)


@dataclass(frozen=True)
class SceneLayoutOptimizerConfig:
    """Numerical controls shared by each graph-layout solve."""

    relation_clearance_m: float = 0.03
    collision_margin_m: float = 0.02
    max_slsqp_iterations: int = 500
    slsqp_ftol: float = 1e-6
    max_collision_rounds: int = 8
    max_added_collision_pairs: int = 64
    imported_seed_weight: float = 5.0
    min_center_distance_m: float = 0.01
    min_center_distance_weight: float = 0.05

    def __post_init__(self) -> None:
        """Reject invalid numerical controls before assembling a layout problem."""
        if self.relation_clearance_m < 0.0:
            raise ValueError("relation_clearance_m must be non-negative.")
        if self.collision_margin_m < 0.0:
            raise ValueError("collision_margin_m must be non-negative.")
        if self.max_slsqp_iterations <= 0:
            raise ValueError("max_slsqp_iterations must be positive.")
        if self.slsqp_ftol <= 0.0:
            raise ValueError("slsqp_ftol must be positive.")
        if self.max_collision_rounds <= 0:
            raise ValueError("max_collision_rounds must be positive.")
        if self.max_added_collision_pairs <= 0:
            raise ValueError("max_added_collision_pairs must be positive.")


class SceneLayoutOptimizer:
    """Solve graph-constrained XY layouts and apply resulting poses."""

    def __init__(self, *, config: SceneLayoutOptimizerConfig | None = None) -> None:
        self.config = config if config is not None else SceneLayoutOptimizerConfig()

    def optimize_table_root_xy(
        self,
        *,
        assets_by_id: dict[str, SceneObject],
        root_ids: list[str],
        root_seed_xy_by_id: dict[str, list[float]],
        imported_root_ids: set[str],
        fixed_root_xy_by_id: dict[str, list[float] | None],
        root_table_regions_by_id: dict[str, str | None],
        table_optimization_rect_xy: list[list[float]],
        root_relations: list[SceneGraphRelation],
    ) -> dict[str, list[float]]:
        """Solve direct table-child centers with graph and AABB constraints."""
        return _optimize_table_root_xy(
            assets_by_id=assets_by_id,
            root_ids=root_ids,
            root_seed_xy_by_id=root_seed_xy_by_id,
            imported_root_ids=imported_root_ids,
            fixed_root_xy_by_id=fixed_root_xy_by_id,
            root_table_regions_by_id=root_table_regions_by_id,
            table_optimization_rect_xy=table_optimization_rect_xy,
            root_relations=root_relations,
            config=self.config,
        )

    def optimize_parent_child_xy(
        self,
        *,
        assets_by_id: dict[str, SceneObject],
        child_ids: list[str],
        child_seed_xy_by_id: dict[str, list[float]],
        imported_child_ids: set[str],
        fixed_child_xy_by_id: dict[str, list[float] | None],
        parent_aabb_xy: list[list[float]],
    ) -> dict[str, list[float]]:
        """Solve direct on-children inside one parent's current XY AABB."""
        child_half_extents_xy = _asset_half_extents_xy(
            assets_by_id=assets_by_id,
            object_ids=child_ids,
        )
        inequality_constraints: list[tuple[np.ndarray, float]] = []
        equality_constraints: list[tuple[np.ndarray, float]] = []
        child_index = {child_id: index for index, child_id in enumerate(child_ids)}
        parent_bounds = _bounds_from_points(parent_aabb_xy)
        for child_id in child_ids:
            _append_aabb_center_bounds(
                constraints=inequality_constraints,
                root_index=child_index,
                root_id=child_id,
                bounds=parent_bounds,
                half_extents_xy=child_half_extents_xy[child_id],
            )
            fixed_xy = fixed_child_xy_by_id[child_id]
            if fixed_xy is not None:
                _append_fixed_root_constraints(
                    constraints=equality_constraints,
                    root_index=child_index,
                    root_id=child_id,
                    fixed_xy=fixed_xy,
                )

        solved_child_xy_by_id = _solve_root_xy(
            root_ids=child_ids,
            root_seed_xy_by_id=child_seed_xy_by_id,
            imported_root_ids=imported_child_ids,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            config=self.config,
        )
        return _refine_root_collisions(
            root_ids=child_ids,
            root_seed_xy_by_id=child_seed_xy_by_id,
            imported_root_ids=imported_child_ids,
            root_half_extents_xy=child_half_extents_xy,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            fixed_root_xy_by_id=fixed_child_xy_by_id,
            solved_root_xy_by_id=solved_child_xy_by_id,
            config=self.config,
        )

    @staticmethod
    def scene_object_z_up_world_aabb(
        *,
        scene_object: SceneObject,
    ) -> list[list[float]]:
        """Return one object's current z-up world AABB as [min, max]."""
        return _scene_object_z_up_world_aabb(scene_object=scene_object)

    @staticmethod
    def update_scene_object_y_up_pose_from_z_up_support(
        *,
        scene_object: SceneObject,
        support_region_z: float,
        center_xy: list[float],
        clearance_m: float = 0.02,
    ) -> None:
        """Place one SimReady asset on a horizontal z-up support region."""
        _update_scene_object_y_up_pose_from_z_up_support(
            scene_object=scene_object,
            support_region_z=support_region_z,
            center_xy=center_xy,
            clearance_m=clearance_m,
        )

    @staticmethod
    def translate_scene_object_y_up_by_z_up_delta(
        *,
        scene_object: SceneObject,
        delta_xy: list[float],
    ) -> None:
        """Translate one existing y-up pose by a solved z-up XY delta."""
        _translate_scene_object_y_up_by_z_up_delta(
            scene_object=scene_object,
            delta_xy=delta_xy,
        )


def _optimize_table_root_xy(
    *,
    assets_by_id: dict[str, SceneObject],
    root_ids: list[str],
    root_seed_xy_by_id: dict[str, list[float]],
    imported_root_ids: set[str],
    fixed_root_xy_by_id: dict[str, list[float] | None],
    root_table_regions_by_id: dict[str, str | None],
    table_optimization_rect_xy: list[list[float]],
    root_relations: list[SceneGraphRelation],
    config: SceneLayoutOptimizerConfig,
) -> dict[str, list[float]]:
    """Solve direct table-child centers with graph and AABB constraints."""
    root_half_extents_xy = _asset_half_extents_xy(
        assets_by_id=assets_by_id,
        object_ids=root_ids,
    )
    inequality_constraints, equality_constraints = _build_table_root_constraints(
        root_ids=root_ids,
        root_half_extents_xy=root_half_extents_xy,
        root_relations=root_relations,
        root_table_regions_by_id=root_table_regions_by_id,
        table_optimization_rect_xy=table_optimization_rect_xy,
        fixed_root_xy_by_id=fixed_root_xy_by_id,
        config=config,
    )
    solved_root_xy_by_id = _solve_root_xy(
        root_ids=root_ids,
        root_seed_xy_by_id=root_seed_xy_by_id,
        imported_root_ids=imported_root_ids,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        config=config,
    )
    return _refine_root_collisions(
        root_ids=root_ids,
        root_seed_xy_by_id=root_seed_xy_by_id,
        imported_root_ids=imported_root_ids,
        root_half_extents_xy=root_half_extents_xy,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        fixed_root_xy_by_id=fixed_root_xy_by_id,
        solved_root_xy_by_id=solved_root_xy_by_id,
        config=config,
    )


def _update_scene_object_y_up_pose_from_z_up_support(
    *,
    scene_object: SceneObject,
    support_region_z: float,
    center_xy: list[float],
    clearance_m: float = 0.02,
) -> None:
    """Place one SimReady asset on a horizontal z-up support region.

    ``SceneObject`` stores poses in y-up before export. The target center and
    support height are z-up values because layout optimization uses that frame.
    """
    if not np.isfinite(support_region_z):
        raise ValueError("support_region_z must be finite.")
    if clearance_m < 0.0 or not np.isfinite(clearance_m):
        raise ValueError("clearance_m must be finite and non-negative.")
    target_xy = _two_floats(center_xy, field_name="center_xy")
    rotation_y_up = _three_floats_or_default(
        scene_object.rot,
        field_name="rot",
        default=[0.0, 0.0, 0.0],
    )
    mesh = _asset_z_up_mesh_at_zero_translation(
        scene_object=scene_object,
        rotation_y_up=rotation_y_up,
    )
    target_position_z_up = np.array(
        [
            target_xy[0] - float(mesh.bounds[:, 0].mean()),
            target_xy[1] - float(mesh.bounds[:, 1].mean()),
            float(support_region_z) + clearance_m - float(mesh.bounds[0, 2]),
        ]
    )
    z_up_to_y_up = np.linalg.inv(_y_up_to_z_up_matrix())
    # Persist the y-up pose that SceneExporter later converts back to z-up.
    scene_object.pos = (z_up_to_y_up[:3, :3] @ target_position_z_up).tolist()
    scene_object.rot = rotation_y_up
    scene_object.center_xy = target_xy


def _translate_scene_object_y_up_by_z_up_delta(
    *,
    scene_object: SceneObject,
    delta_xy: list[float],
) -> None:
    """Translate one existing y-up pose by a solved z-up XY delta."""
    dx, dy = _two_floats(delta_xy, field_name="delta_xy")
    current_pos = _three_floats_or_default(
        scene_object.pos,
        field_name="pos",
        default=None,
    )
    # z-up x maps to y-up x, while z-up y maps to negative y-up z.
    scene_object.pos = [
        current_pos[0] + dx,
        current_pos[1],
        current_pos[2] - dy,
    ]
    if scene_object.center_xy is not None:
        scene_object.center_xy = [
            scene_object.center_xy[0] + dx,
            scene_object.center_xy[1] + dy,
        ]


def _scene_object_z_up_world_aabb(
    *,
    scene_object: SceneObject,
) -> list[list[float]]:
    """Measure one current SceneObject pose in z-up world coordinates."""
    position_y_up = _three_floats_or_default(
        scene_object.pos,
        field_name="pos",
        default=None,
    )
    mesh = _asset_z_up_mesh_at_zero_translation(scene_object=scene_object)
    position_z_up = _y_up_to_z_up_matrix()[:3, :3] @ np.asarray(
        position_y_up,
        dtype=float,
    )
    mesh.apply_translation(position_z_up)
    return mesh.bounds.tolist()


def _build_table_root_constraints(
    *,
    root_ids: list[str],
    root_half_extents_xy: dict[str, np.ndarray],
    root_relations: list[SceneGraphRelation],
    root_table_regions_by_id: dict[str, str | None],
    table_optimization_rect_xy: list[list[float]],
    fixed_root_xy_by_id: dict[str, list[float] | None],
    config: SceneLayoutOptimizerConfig,
) -> tuple[list[tuple[np.ndarray, float]], list[tuple[np.ndarray, float]]]:
    """Build hard table, region, planar, and fixed-root constraints."""
    root_index = {root_id: index for index, root_id in enumerate(root_ids)}
    table_bounds = _bounds_from_points(table_optimization_rect_xy)
    inequality_constraints: list[tuple[np.ndarray, float]] = []
    equality_constraints: list[tuple[np.ndarray, float]] = []

    for root_id in root_ids:
        region_bounds = _table_region_bounds(
            table_bounds=table_bounds,
            table_region=root_table_regions_by_id[root_id],
        )
        _append_aabb_center_bounds(
            constraints=inequality_constraints,
            root_index=root_index,
            root_id=root_id,
            bounds=region_bounds,
            half_extents_xy=root_half_extents_xy[root_id],
        )
        fixed_xy = fixed_root_xy_by_id[root_id]
        if fixed_xy is not None:
            _append_fixed_root_constraints(
                constraints=equality_constraints,
                root_index=root_index,
                root_id=root_id,
                fixed_xy=fixed_xy,
            )

    for relation in root_relations:
        _append_planar_relation_constraint(
            constraints=inequality_constraints,
            root_index=root_index,
            source_id=relation.source_id,
            relation=relation.relation,
            target_id=relation.target_id,
            source_half_extents_xy=root_half_extents_xy[relation.source_id],
            target_half_extents_xy=root_half_extents_xy[relation.target_id],
            relation_clearance_m=config.relation_clearance_m,
        )
    return inequality_constraints, equality_constraints


def _solve_root_xy(
    *,
    root_ids: list[str],
    root_seed_xy_by_id: dict[str, list[float]],
    imported_root_ids: set[str],
    inequality_constraints: list[tuple[np.ndarray, float]],
    equality_constraints: list[tuple[np.ndarray, float]],
    config: SceneLayoutOptimizerConfig,
) -> dict[str, list[float]]:
    """Solve one root-group center model with the legacy SLSQP settings."""
    root_index = {root_id: index for index, root_id in enumerate(root_ids)}
    initial_xy = np.asarray(
        [root_seed_xy_by_id[root_id] for root_id in root_ids], dtype=float
    )
    x0 = initial_xy.reshape(-1)

    def unpack(values: np.ndarray) -> dict[str, list[float]]:
        return {
            root_id: [float(values[2 * index]), float(values[2 * index + 1])]
            for root_id, index in root_index.items()
        }

    def objective(values: np.ndarray) -> float:
        coordinates = values.reshape(-1, 2)
        loss = 0.0
        for root_id, index in root_index.items():
            if root_id in imported_root_ids:
                delta = coordinates[index] - initial_xy[index]
                loss += config.imported_seed_weight * float(delta @ delta)
        for first_index in range(len(root_ids)):
            for second_index in range(first_index + 1, len(root_ids)):
                distance = float(
                    np.linalg.norm(coordinates[first_index] - coordinates[second_index])
                )
                shortfall = max(0.0, config.min_center_distance_m - distance)
                loss += config.min_center_distance_weight * shortfall**2
        return loss

    constraints: list[dict[str, object]] = []
    for row, bound in inequality_constraints:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda values, row=row, bound=bound: bound - float(row @ values),
            }
        )
    for row, bound in equality_constraints:
        constraints.append(
            {
                "type": "eq",
                "fun": lambda values, row=row, bound=bound: float(row @ values) - bound,
            }
        )

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        constraints=constraints,
        options={
            "maxiter": config.max_slsqp_iterations,
            "ftol": config.slsqp_ftol,
            "disp": False,
        },
    )
    if not result.success:
        raise ValueError(f"Table layout optimization failed: {result.message}")
    return unpack(np.asarray(result.x, dtype=float))


def _refine_root_collisions(
    *,
    root_ids: list[str],
    root_seed_xy_by_id: dict[str, list[float]],
    imported_root_ids: set[str],
    root_half_extents_xy: dict[str, np.ndarray],
    inequality_constraints: list[tuple[np.ndarray, float]],
    equality_constraints: list[tuple[np.ndarray, float]],
    fixed_root_xy_by_id: dict[str, list[float] | None],
    solved_root_xy_by_id: dict[str, list[float]],
    config: SceneLayoutOptimizerConfig,
) -> dict[str, list[float]]:
    """Separate only sibling pairs that include a layout-variable object."""
    seen_pairs: set[tuple[str, str]] = set()
    current_xy_by_id = solved_root_xy_by_id
    for _ in range(config.max_collision_rounds):
        all_overlaps = _root_aabb_overlaps(
            root_ids=root_ids,
            root_half_extents_xy=root_half_extents_xy,
            xy_by_id=current_xy_by_id,
        )
        # Fixed/fixed siblings are outside this edit and therefore cannot be solved here.
        overlaps = [
            overlap
            for overlap in all_overlaps
            if fixed_root_xy_by_id[overlap[1]] is None
            or fixed_root_xy_by_id[overlap[2]] is None
        ]
        if not overlaps:
            return current_xy_by_id
        added_constraint_count = 0
        for _, first_id, second_id in overlaps[: config.max_added_collision_pairs]:
            pair_key = tuple(sorted((first_id, second_id)))
            if pair_key in seen_pairs:
                continue
            inequality_constraints.append(
                _aabb_separation_constraint(
                    root_ids=root_ids,
                    first_id=first_id,
                    second_id=second_id,
                    first_half_extents_xy=root_half_extents_xy[first_id],
                    second_half_extents_xy=root_half_extents_xy[second_id],
                    first_xy=current_xy_by_id[first_id],
                    second_xy=current_xy_by_id[second_id],
                    collision_margin_m=config.collision_margin_m,
                )
            )
            seen_pairs.add(pair_key)
            added_constraint_count += 1
        if added_constraint_count == 0:
            break
        current_xy_by_id = _solve_root_xy(
            root_ids=root_ids,
            root_seed_xy_by_id=current_xy_by_id,
            imported_root_ids=imported_root_ids,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            config=config,
        )

    remaining_pairs = [
        f"{first_id}/{second_id}"
        for _, first_id, second_id in _root_aabb_overlaps(
            root_ids=root_ids,
            root_half_extents_xy=root_half_extents_xy,
            xy_by_id=current_xy_by_id,
        )
        if fixed_root_xy_by_id[first_id] is None
        or fixed_root_xy_by_id[second_id] is None
    ]
    raise ValueError(
        "Table-root AABB collisions remain after layout refinement: "
        f"{remaining_pairs}."
    )


def _asset_half_extents_xy(
    *,
    assets_by_id: dict[str, SceneObject],
    object_ids: list[str],
) -> dict[str, np.ndarray]:
    """Measure each asset's oriented z-up footprint around its XY center."""
    half_extents_xy: dict[str, np.ndarray] = {}
    for object_id in object_ids:
        asset = assets_by_id.get(object_id)
        if asset is None:
            raise ValueError(f"Table root {object_id!r} is not an asset.")
        half_extents_xy[object_id] = _asset_half_extent_xy(asset)
    return half_extents_xy


def _asset_half_extent_xy(asset: SceneObject) -> np.ndarray:
    """Measure one SimReady GLB with its current orientation and scale."""
    mesh = _asset_z_up_mesh_at_zero_translation(scene_object=asset)
    return (mesh.bounds[1, :2] - mesh.bounds[0, :2]) / 2.0


def _asset_z_up_mesh_at_zero_translation(
    *,
    scene_object: SceneObject,
    rotation_y_up: list[float] | None = None,
):
    """Load one SimReady GLB in z-up with orientation and scale but no position."""
    asset = scene_object
    if asset.simready_glb_path is None:
        raise ValueError(f"Asset {asset.id!r} has no SimReady GLB path.")
    y_up_layout = {
        "id": asset.id,
        "rot": (
            rotation_y_up
            if rotation_y_up is not None
            else _three_floats_or_default(
                asset.rot,
                field_name="rot",
                default=[0.0, 0.0, 0.0],
            )
        ),
        "pos": [0.0, 0.0, 0.0],
        "scale": _three_floats_or_default(
            asset.scale,
            field_name="scale",
            default=[1.0, 1.0, 1.0],
        ),
    }
    y_up_to_z_up = _y_up_to_z_up_matrix()
    z_up_layout = transform_matrix_to_layout_object(
        asset.id,
        y_up_to_z_up
        @ layout_object_to_transform_matrix(y_up_layout)
        @ np.linalg.inv(y_up_to_z_up),
    )
    mesh = load_glb_mesh(asset.simready_glb_path)
    mesh.apply_transform(y_up_to_z_up)
    mesh.apply_transform(layout_object_to_transform_matrix(z_up_layout))
    return mesh


def _y_up_to_z_up_matrix() -> np.ndarray:
    """Return the coordinate transform used by SceneExporter and layout stages."""
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    return matrix


def _two_floats(value: object, *, field_name: str) -> list[float]:
    """Validate one finite two-value vector."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must contain two values.")
    vector = [float(component) for component in value]
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{field_name} must contain finite values.")
    return vector


def _three_floats_or_default(
    value: object,
    *,
    field_name: str,
    default: list[float] | None,
) -> list[float]:
    """Return a finite three-value vector or the canonical SimReady default."""
    if value is None:
        if default is None:
            raise ValueError(f"{field_name} must contain three values.")
        return list(default)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must contain three values.")
    vector = [float(component) for component in value]
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{field_name} must contain finite values.")
    return vector


def _bounds_from_points(points: list[list[float]]) -> np.ndarray:
    """Return [[min_x, min_y], [max_x, max_y]] from finite XY points."""
    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
        raise ValueError("XY bounds must contain at least two points.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("XY bounds must contain finite values.")
    return np.stack([coordinates.min(axis=0), coordinates.max(axis=0)])


def _table_region_bounds(
    *,
    table_bounds: np.ndarray,
    table_region: str | None,
) -> np.ndarray:
    """Return the requested 3x3 table region, with y increasing toward front."""
    if table_region is None:
        return table_bounds.copy()
    column_by_region = {
        "left_back": 0,
        "left_center": 0,
        "left_front": 0,
        "back_center": 1,
        "center": 1,
        "front_center": 1,
        "right_back": 2,
        "right_center": 2,
        "right_front": 2,
    }
    row_by_region = {
        "left_back": 0,
        "back_center": 0,
        "right_back": 0,
        "left_center": 1,
        "center": 1,
        "right_center": 1,
        "left_front": 2,
        "front_center": 2,
        "right_front": 2,
    }
    if table_region not in column_by_region:
        raise ValueError(f"Unsupported table region {table_region!r}.")
    minimum, maximum = table_bounds
    cell_size = (maximum - minimum) / 3.0
    region_minimum = minimum + cell_size * np.array(
        [column_by_region[table_region], row_by_region[table_region]]
    )
    return np.stack([region_minimum, region_minimum + cell_size])


def _append_aabb_center_bounds(
    *,
    constraints: list[tuple[np.ndarray, float]],
    root_index: dict[str, int],
    root_id: str,
    bounds: np.ndarray,
    half_extents_xy: np.ndarray,
) -> None:
    """Keep one root's complete AABB inside the given rectangular bounds."""
    minimum = bounds[0] + half_extents_xy
    maximum = bounds[1] - half_extents_xy
    if np.any(minimum > maximum):
        raise ValueError(
            f"Asset {root_id!r} cannot fit inside its assigned table region."
        )
    variable_count = 2 * len(root_index)
    root_offset = 2 * root_index[root_id]
    for axis in range(2):
        upper_row = np.zeros(variable_count)
        upper_row[root_offset + axis] = 1.0
        constraints.append((upper_row, float(maximum[axis])))
        lower_row = np.zeros(variable_count)
        lower_row[root_offset + axis] = -1.0
        constraints.append((lower_row, -float(minimum[axis])))


def _append_fixed_root_constraints(
    *,
    constraints: list[tuple[np.ndarray, float]],
    root_index: dict[str, int],
    root_id: str,
    fixed_xy: list[float],
) -> None:
    """Use equality constraints so unchanged formal objects remain fixed."""
    variable_count = 2 * len(root_index)
    root_offset = 2 * root_index[root_id]
    for axis, coordinate in enumerate(fixed_xy):
        row = np.zeros(variable_count)
        row[root_offset + axis] = 1.0
        constraints.append((row, float(coordinate)))


def _append_planar_relation_constraint(
    *,
    constraints: list[tuple[np.ndarray, float]],
    root_index: dict[str, int],
    source_id: str,
    relation: str,
    target_id: str,
    source_half_extents_xy: np.ndarray,
    target_half_extents_xy: np.ndarray,
    relation_clearance_m: float,
) -> None:
    """Require directional relations to clear both sibling AABB footprints."""
    if source_id not in root_index or target_id not in root_index:
        raise ValueError("Table-root planar relations must reference table roots.")
    axis, source_sign = {
        "left_of": (0, 1.0),
        "right_of": (0, -1.0),
        "behind": (1, 1.0),
        "in_front_of": (1, -1.0),
    }.get(relation, (None, None))
    if axis is None or source_sign is None:
        raise ValueError(f"Unsupported planar relation {relation!r}.")
    row = np.zeros(2 * len(root_index))
    row[2 * root_index[source_id] + axis] = source_sign
    row[2 * root_index[target_id] + axis] = -source_sign
    required_distance = (
        source_half_extents_xy[axis]
        + target_half_extents_xy[axis]
        + relation_clearance_m
    )
    constraints.append((row, -float(required_distance)))


def _root_aabb_overlaps(
    *,
    root_ids: list[str],
    root_half_extents_xy: dict[str, np.ndarray],
    xy_by_id: dict[str, list[float]],
) -> list[tuple[float, str, str]]:
    """Return root pairs whose current XY AABBs overlap without a margin."""
    overlaps: list[tuple[float, str, str]] = []
    for first_index, first_id in enumerate(root_ids):
        first_xy = np.asarray(xy_by_id[first_id], dtype=float)
        first_half_extents = root_half_extents_xy[first_id]
        for second_id in root_ids[first_index + 1 :]:
            second_xy = np.asarray(xy_by_id[second_id], dtype=float)
            second_half_extents = root_half_extents_xy[second_id]
            overlap_xy = np.minimum(
                first_xy + first_half_extents,
                second_xy + second_half_extents,
            ) - np.maximum(
                first_xy - first_half_extents,
                second_xy - second_half_extents,
            )
            if np.all(overlap_xy > 1e-9):
                overlaps.append((float(np.min(overlap_xy)), first_id, second_id))
    return sorted(overlaps, reverse=True)


def _aabb_separation_constraint(
    *,
    root_ids: list[str],
    first_id: str,
    second_id: str,
    first_half_extents_xy: np.ndarray,
    second_half_extents_xy: np.ndarray,
    first_xy: list[float],
    second_xy: list[float],
    collision_margin_m: float,
) -> tuple[np.ndarray, float]:
    """Separate one overlapping pair along its shallowest penetration axis."""
    root_index = {root_id: index for index, root_id in enumerate(root_ids)}
    first_xy_array = np.asarray(first_xy, dtype=float)
    second_xy_array = np.asarray(second_xy, dtype=float)
    overlap_xy = np.minimum(
        first_xy_array + first_half_extents_xy,
        second_xy_array + second_half_extents_xy,
    ) - np.maximum(
        first_xy_array - first_half_extents_xy,
        second_xy_array - second_half_extents_xy,
    )
    axis = int(np.argmin(overlap_xy))
    first_is_lower = first_xy_array[axis] < second_xy_array[axis] or (
        first_xy_array[axis] == second_xy_array[axis] and first_id < second_id
    )
    row = np.zeros(2 * len(root_ids))
    first_coefficient = 1.0 if first_is_lower else -1.0
    row[2 * root_index[first_id] + axis] = first_coefficient
    row[2 * root_index[second_id] + axis] = -first_coefficient
    required_distance = (
        first_half_extents_xy[axis] + second_half_extents_xy[axis] + collision_margin_m
    )
    return row, -float(required_distance)
