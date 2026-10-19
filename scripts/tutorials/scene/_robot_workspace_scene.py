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

"""Shared physical inputs and measured checks for the workspace demonstration.

This module deliberately has no Gym or Task Program dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from embodichain.robot_registry import resolve_robot_cfg
from embodichain.utils.config_paths import resolve_config_path
from embodichain.lab.sim.cfg import RigidObjectCfg, RobotCfg
from embodichain.lab.sim.motion.workspace.analyzer import (
    AnalysisMode,
    WorkspaceAnalyzer,
    WorkspaceAnalyzerConfig,
)
from embodichain.lab.sim.motion.workspace.cfg import RobotWorkspaceCfg
from embodichain.lab.sim.motion.workspace.configs.cache_config import CacheConfig
from embodichain.lab.sim.motion.workspace.configs.metric_config import MetricConfig
from embodichain.lab.sim.motion.workspace.configs.sampling_config import SamplingConfig
from embodichain.lab.sim.motion.workspace.configs.visualization_config import (
    VisualizationConfig,
)
from embodichain.lab.sim.motion.workspace.caches.results_cache import ResultsCache
from embodichain.lab.sim.scene import (
    SceneFragment,
    SceneCheck,
    SceneCheckReport,
    SceneView,
    SettlingCfg,
    RobotSceneWorkspace,
    capture_state,
    compare_states,
    restore_state,
    settle_scene,
)
from embodichain.lab.sim.scene.expansion import ScenePoseChange, SceneVariant
from embodichain.lab.sim.scene.operations import SceneChangeSet, apply_changes

__all__: list[str] = []

TABLE_Z = 0.75
CUBE_SIZE = 0.05
CUBE_Z = TABLE_Z + CUBE_SIZE / 2
INITIAL = (-0.42, -0.08, CUBE_Z)
GOAL = (-0.40, 0.48, CUBE_Z)
NEGATIVE = (-0.98, 0.12, CUBE_Z)
TABLE_CENTER = (0.0, 0.0)
TABLE_RADIUS = 1.2
TABLE_SECTIONS = 96
TABLE_LOWER = tuple(value - TABLE_RADIUS for value in TABLE_CENTER)
TABLE_UPPER = tuple(value + TABLE_RADIUS for value in TABLE_CENTER)
CANDIDATE_XY_BOUNDS = ((-0.55, -0.18), (-0.30, 0.12))
CONTROL_DT = 0.04
SCENE_ID = "ur5-workspace-round-table/v1"


@dataclass(frozen=True)
class _Candidate:
    variant: SceneVariant
    joint_seed: tuple[float, ...]
    cache_index: int
    original_pose: tuple[tuple[float, ...], ...]
    checks: SceneCheckReport


def _embodiment() -> dict[str, Any]:
    # Importing embodichain_tasks auto-registers Gym environments. Resolve the
    # physical resource through the core path helper instead, including wheels.
    path = resolve_config_path(
        "extensions/embodichain_tasks/configs/components/embodiments/ur5_dh_pgi_140_80.yaml",
        base_dir=Path(__file__).parent,
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _robot_cfg() -> RobotCfg:
    physical = _embodiment()["simulation"]
    physical.update(
        init_pos=[0.0, 0.0, TABLE_Z],
        init_qpos=[0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0, 0.0],
    )
    return resolve_robot_cfg("ur5", overrides=physical)


def _fragment() -> SceneFragment:
    import trimesh

    top = trimesh.creation.cylinder(
        radius=TABLE_RADIUS, height=0.04, sections=TABLE_SECTIONS
    )
    pedestal = trimesh.creation.cylinder(
        radius=0.18, height=TABLE_Z - 0.04, sections=48
    )
    pedestal.apply_translation((0.0, 0.0, (TABLE_Z - 0.04) / 2 - (TABLE_Z - 0.02)))
    foot = trimesh.creation.cylinder(radius=0.42, height=0.04, sections=64)
    foot.apply_translation((0.0, 0.0, 0.02 - (TABLE_Z - 0.02)))
    mesh = trimesh.util.concatenate((top, pedestal, foot))
    table = RigidObjectCfg.from_dict(
        {
            "uid": "support_surface",
            "body_type": "static",
            "shape": {
                "shape_type": "Mesh",
                "vertices": mesh.vertices.tolist(),
                "triangles": mesh.faces.tolist(),
                "collision": {"approximation": "triangle_mesh"},
            },
            "init_pos": [*TABLE_CENTER, TABLE_Z - 0.02],
            "attrs": {
                "material_props": {"static_friction": 0.99, "dynamic_friction": 0.97}
            },
        }
    )
    cube = RigidObjectCfg.from_dict(
        {
            "uid": "cube",
            "body_type": "dynamic",
            "shape": {"shape_type": "Cube", "size": [CUBE_SIZE] * 3},
            "init_pos": list(INITIAL),
            "attrs": {
                "mass_props": {"mass": 0.05},
                "rigid_props": {"linear_damping": 0.2, "angular_damping": 0.2},
                "material_props": {"static_friction": 0.99, "dynamic_friction": 0.97},
            },
        }
    )
    from embodichain.lab.sim.material import VisualMaterialCfg

    cube.shape.visual_material = VisualMaterialCfg(
        uid="workspace_cube_blue",
        base_color=[0.08, 0.25, 0.9, 1.0],
        roughness=0.6,
    )
    return SceneFragment([table, cube])


def _check_dict(report: SceneCheckReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "checks": [
            {
                "name": c.name,
                "status": c.status,
                "message": c.message,
                "metrics": dict(c.metrics),
            }
            for c in report.checks
        ],
    }


def _make_cache(robot: Any, output: Path, seed: int) -> Path:
    analyzer = WorkspaceAnalyzer(
        robot,
        config=WorkspaceAnalyzerConfig(
            mode=AnalysisMode.JOINT_SPACE,
            sampling=SamplingConfig(num_samples=8192, seed=seed, batch_size=2048),
            cache=CacheConfig(enabled=False),
            metric=MetricConfig(enabled_metrics=[]),
            visualization=VisualizationConfig(enabled=False),
            control_part_name="arm",
        ),
    )
    result = analyzer.analyze(visualize=False)
    joints = result["joint_configurations"]
    # Store real current-arena FK positions aligned with every sampled qpos row.
    poses = robot.compute_batch_fk(
        qpos=joints.unsqueeze(0), name="arm", env_ids=[0], to_matrix=True
    )
    result["workspace_points"] = poses[0, :, :3, 3]
    path = ResultsCache(output / "workspace_cache").save(
        "ur5-arm",
        result,
        metadata={
            "robot": "ur5",
            "control_part": "arm",
            "seed": seed,
            "base_position": [0.0, 0.0, TABLE_Z],
            "sample_count": 8192,
        },
    )
    cfg = RobotWorkspaceCfg(cache_path=str(path))
    cfg.validate()
    robot.cfg.workspace_cfg = {"arm": cfg}
    return path / "results.npz"


def _choose_grasp(
    robot: Any, workspace: RobotSceneWorkspace, seed: int
) -> torch.Tensor:
    import trimesh
    from embodichain.toolkits.graspkit.pose_generator import (
        get_parallel_jaw_gripper_model,
    )
    from embodichain.toolkits.graspkit.pg_grasp.pose_generator import (
        AntipodalGraspPoseGenerator,
        AntipodalGraspPoseGeneratorCfg,
        ParallelJawGraspCollisionCfg,
    )

    mesh = trimesh.creation.box(extents=[CUBE_SIZE] * 3)
    generator = AntipodalGraspPoseGenerator(
        get_parallel_jaw_gripper_model("dh_pgi_140_80"),
        algorithm_cfg=AntipodalGraspPoseGeneratorCfg(
            sample_count=1000,
            max_candidates=32,
            approach_deviation_angle=0.0,
            approach_direction_samples=1,
        ),
        collision_cfg=ParallelJawGraspCollisionCfg(filter_ground_collision=False),
    )
    obj = torch.eye(4, device=robot.device)
    obj[:3, 3] = obj.new_tensor(INITIAL)
    numpy_state = np.random.get_state()
    try:
        np.random.seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            rows = generator.get_valid_grasp_poses(
                mesh_vertices=torch.as_tensor(
                    mesh.vertices.copy(), dtype=torch.float32, device=robot.device
                ),
                mesh_triangles=torch.as_tensor(
                    mesh.faces.copy(), dtype=torch.long, device=robot.device
                ),
                obj_poses=obj.unsqueeze(0),
                approach_direction=obj.new_tensor([0.0, 0.0, -1.0]),
            )
    finally:
        np.random.set_state(numpy_state)
    for world_pose in rows[0][0]:
        local = torch.linalg.inv(obj) @ world_pose
        if workspace.check_object_pose(
            ScenePoseChange("cube", obj.tolist()), local
        ).passed:
            return local.detach().clone()
    raise RuntimeError(
        "No generated downward grasp passes full-pose IK/FK at the baseline"
    )


def _prepare_candidates(
    robot: Any,
    output: Path,
    seed: int,
    count: int,
) -> tuple[RobotSceneWorkspace, torch.Tensor, tuple[_Candidate, ...], dict[str, Any]]:
    cache = _make_cache(robot, output, seed)
    workspace = RobotSceneWorkspace(robot, control_part="arm")
    grasp = _choose_grasp(robot, workspace, seed)
    samples = workspace.sample_object_poses(
        "cube",
        grasp,
        num_samples=128,
        seed=seed,
        max_attempts=1024,
        position_bounds=(
            (*CANDIDATE_XY_BOUNDS[0], -1.0),
            (*CANDIDATE_XY_BOUNDS[1], 2.0),
        ),
    )
    candidates = []
    diagnostics = []
    for sample in samples:
        adjusted = torch.eye(4, device=robot.device)
        adjusted[:2, 3] = adjusted.new_tensor(sample.pose_change.arena_pose)[:2, 3]
        adjusted[2, 3] = CUBE_Z
        change = ScenePoseChange("cube", adjusted.tolist())
        check = workspace.check_object_pose(change, grasp, joint_seed=sample.joint_seed)
        diagnostics.append(
            {"cache_index": sample.cache_index, "checks": _check_dict(check)}
        )
        if not check.passed or any(
            np.linalg.norm(
                np.asarray(c.variant.pose_changes[0].arena_pose)[:2, 3]
                - adjusted[:2, 3].cpu().numpy()
            )
            < 0.05
            for c in candidates
        ):
            continue
        variant = SceneVariant(
            SCENE_ID,
            seed,
            len(candidates),
            (change,),
            (
                ("generator", "robot_workspace_demo/v1"),
                ("cache_index", str(sample.cache_index)),
            ),
        )
        candidates.append(
            _Candidate(
                variant,
                sample.joint_seed,
                sample.cache_index,
                sample.pose_change.arena_pose,
                check,
            )
        )
        if len(candidates) == count:
            break
    info = {
        "cache": str(cache),
        "requested_cache_samples": 8192,
        "cache_samples": len(robot.get_workspace(name="arm").qpos),
        "sampling_budget": 1024,
        "returned_samples": len(samples),
        "screened_samples": diagnostics,
        "object_grasp_pose": grasp.cpu().tolist(),
    }
    if len(candidates) != count:
        raise RuntimeError(
            f"Candidate budget exhausted: {len(candidates)}/{count} distinct valid poses"
        )
    info["reachability"] = _make_reachability_map(
        workspace, grasp, tuple(candidates), robot.get_qpos(name="arm")[0], output
    )
    return workspace, grasp, tuple(candidates), info


def _make_reachability_map(
    workspace: RobotSceneWorkspace,
    grasp: torch.Tensor,
    candidates: tuple[_Candidate, ...],
    joint_seed: torch.Tensor,
    output: Path,
) -> dict[str, Any]:
    """Check tabletop object origins with one fixed grasp, not FK occupancy."""
    margin = CUBE_SIZE / 2
    axes = [
        np.arange(lower + margin, upper - margin + 1e-8, 0.06)
        for lower, upper in zip(TABLE_LOWER, TABLE_UPPER, strict=True)
    ]
    corners = np.asarray(list(product((-margin, margin), repeat=2)))
    points = [
        (float(x), float(y), CUBE_Z)
        for x, y in product(*axes)
        if _round_table_contains(corners + (x, y))
    ]
    candidate_indices = list(range(len(points), len(points) + len(candidates)))
    points.extend(
        tuple(np.asarray(candidate.variant.pose_changes[0].arena_pose)[:3, 3])
        for candidate in candidates
    )
    negative_index = len(points)
    points.append(NEGATIVE)
    seeds = [tuple(joint_seed.detach().cpu().tolist())] * len(points)
    for index, candidate in zip(candidate_indices, candidates, strict=True):
        seeds[index] = candidate.joint_seed
    passed, position_errors, orientation_errors, details = [], [], [], []
    for point, seed in zip(points, seeds, strict=True):
        pose = np.eye(4)
        pose[:3, 3] = point
        check = workspace.check_object_pose(
            ScenePoseChange("cube", pose.tolist()), grasp, joint_seed=seed
        )
        passed.append(check.passed)
        metrics = check.checks[0].metrics
        position_errors.append(metrics.get("position_error_m", np.nan))
        orientation_errors.append(metrics.get("orientation_error_rad", np.nan))
        details.append(check.checks[0].message)
    valid = np.asarray(passed, dtype=bool)
    if not valid[candidate_indices].all() or valid[negative_index]:
        raise RuntimeError(
            "Reachability colors disagree with the selected red/green examples"
        )
    colors = np.where(valid[:, None], (40, 220, 100), (245, 65, 65)).astype(np.uint8)
    path = output / "workspace_reachability.npz"
    np.savez_compressed(
        path,
        points=np.asarray(points, dtype=np.float32),
        reachable=valid,
        colors=colors,
        joint_seeds=np.asarray(seeds, dtype=np.float32),
        object_grasp_pose=grasp.detach().cpu().numpy(),
        position_error_m=position_errors,
        orientation_error_rad=orientation_errors,
        details=np.asarray(details),
    )
    return {
        "cache": str(path),
        "definition": "upright_object_fixed_grasp_ik_joint_limits_full_fk",
        "object_center_height_m": CUBE_Z,
        "grid_spacing_m": 0.06,
        "point_count": len(points),
        "reachable_count": int(valid.sum()),
        "rejected_count": int((~valid).sum()),
        "candidate_indices": candidate_indices,
        "negative_index": negative_index,
    }


def _round_table_contains(corners: np.ndarray) -> np.bool_:
    """Conservatively contain every footprint corner in the faceted round top."""
    # The mesh is an inscribed regular polygon. Its inradius also keeps corner
    # checks inside the actual collision surface between adjacent rim vertices.
    support_radius = TABLE_RADIUS * np.cos(np.pi / TABLE_SECTIONS)
    return np.all(np.linalg.norm(corners - TABLE_CENTER, axis=-1) <= support_radius)


def _initial_checks(
    sim: Any,
    workspace: RobotSceneWorkspace,
    grasp: torch.Tensor,
    joint_seed: tuple[float, ...] | None = None,
) -> SceneCheckReport:
    cube = sim.get_rigid_object("cube")
    pose = cube.get_local_pose(to_matrix=True)[0]
    vertices = pose.new_tensor(list(product((-CUBE_SIZE / 2, CUBE_SIZE / 2), repeat=3)))
    vertices = vertices @ pose[:3, :3].T + pose[:3, 3]
    lower = vertices.min(dim=0).values
    supported = (
        bool(_round_table_contains(vertices[:, :2].detach().cpu().numpy()))
        and abs(float(lower[2]) - TABLE_Z) <= 0.005
    )
    linear = float(torch.linalg.vector_norm(cube.body_state[0, 7:10]))
    angular = float(torch.linalg.vector_norm(cube.body_state[0, 10:13]))
    reachable = workspace.check_object_pose(
        ScenePoseChange("cube", pose.tolist()), grasp, joint_seed=joint_seed
    )
    return SceneCheckReport(
        (
            SceneCheck(
                "scene.support",
                "passed" if supported else "failed",
                "Measured whole-cube footprint against the circular work surface",
                {"bottom_z": float(lower[2])},
            ),
            SceneCheck(
                "demo.initial_velocities",
                "passed" if linear <= 0.03 and angular <= 0.2 else "failed",
                "Measured initial velocities",
                {"linear_speed": linear, "angular_speed": angular},
            ),
            *reachable.checks,
        )
    )


def _negative(seed: int, ordinal: int) -> SceneVariant:
    pose = np.eye(4)
    pose[:3, 3] = NEGATIVE
    return SceneVariant(
        SCENE_ID,
        seed,
        ordinal,
        (ScenePoseChange("cube", pose.tolist()),),
        (("case", "supported_but_unreachable"),),
    )


def _settle(view: SceneView, step: Any) -> SceneCheckReport:
    return settle_scene(view, step, _settling_cfg())


def _settling_cfg() -> SettlingCfg:
    # Object thresholds match the demo's acceptance contract. The held robot
    # uses a separate measured joint-velocity threshold (rad/s).
    return SettlingCfg(
        mode="convergence",
        max_steps=600,
        consecutive_steps=10,
        linear_threshold=0.03,
        angular_threshold=0.2,
        joint_threshold=0.05,
    )


def _restore(view: SceneView, snapshot: Any) -> None:
    restore_state(view, snapshot, qpos_tolerance=1e-4)
    if not compare_states(snapshot, capture_state(view)).passed:
        raise RuntimeError("Measured scene restore differs from its snapshot")


def _run_scene(
    sim: Any, robot: Any, output: Path, seed: int, count: int, display: Any
) -> dict[str, Any]:
    display.stage = "candidate_generation"
    workspace, grasp, candidates, info = _prepare_candidates(robot, output, seed, count)
    view = SceneView(sim)
    baseline = capture_state(view)
    rows, snapshots = [], []
    display.candidates(candidates, grasp, info, sim=sim)
    negative = _negative(seed, count)
    display.stage = "negative_preparation"
    apply_changes(
        view, SceneChangeSet({"cube": negative.pose_changes[0].arena_pose}, ("cube",))
    )
    _settle(view, lambda n: display.advance(sim, n))
    negative_checks = _initial_checks(sim, workspace, grasp)
    if (
        negative_checks.passed
        or negative_checks.checks[0].status != "passed"
        or negative_checks.checks[-1].status != "failed"
    ):
        raise RuntimeError("Red start must be supported but fail workspace IK")
    display.frame(sim, "negative_rejected", force=True)
    initialization_order = [count]
    for candidate in candidates:
        display.stage = f"candidate_{candidate.variant.ordinal}_preparation"
        _restore(view, baseline)
        apply_changes(
            view,
            SceneChangeSet(
                {c.entity_uid: c.arena_pose for c in candidate.variant.pose_changes},
                ("cube",),
            ),
        )
        stable = _settle(view, lambda n: display.advance(sim, n))
        actual = capture_state(view)
        checks = _initial_checks(sim, workspace, grasp, candidate.joint_seed)
        snapshots.append(actual)
        row = {
            "variant_id": candidate.variant.variant_id,
            "cache_index": candidate.cache_index,
            "joint_seed": candidate.joint_seed,
            "original_pose": candidate.original_pose,
            "adjusted_pose": candidate.variant.pose_changes[0].arena_pose,
            "measured_pose": sim.get_rigid_object("cube")
            .get_local_pose(to_matrix=True)[0]
            .tolist(),
            "settling": _check_dict(stable),
            "initial_validation": _check_dict(checks),
        }
        rows.append(row)
        initialization_order.append(candidate.variant.ordinal)
        suffix = "settled" if stable.passed and checks.passed else "rejected"
        display.frame(
            sim, f"candidate_{candidate.variant.ordinal}_{suffix}", force=True
        )
        if not stable.passed or not checks.passed:
            raise RuntimeError(f"Initial candidate checks failed: {row}")
    restore_reports = []
    for index in (0, 1 if count > 1 else 0, 0):
        display.stage = f"candidate_{index}_restoration"
        _restore(view, snapshots[index])
        comparison = compare_states(snapshots[index], capture_state(view))
        restore_reports.append(_check_dict(comparison))
        display.frame(sim, f"restored_{index}", force=True)
        if not comparison.passed:
            raise RuntimeError("A/B/A physical state restoration failed")
    return {
        "workspace": info,
        "candidates": rows,
        "restoration": restore_reports,
        "negative": _check_dict(negative_checks),
        "initialization_order": initialization_order,
        "accepted": True,
    }
