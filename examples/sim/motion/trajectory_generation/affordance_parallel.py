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

"""Generate masked trajectory batches from real antipodal cube grasps.

Run in the ``embodichain2`` environment from the repository root::

    conda activate embodichain2
    python examples/sim/motion/trajectory_generation/affordance_parallel.py \
        --output /tmp/affordance-parallel --trajectories 8 --seed 13

The existing cube scene supplies four real UR5/gripper instances. One actual
AntipodalGraspPoseGenerator samples the cube triangle mesh; no hand-authored yaw
poses are used as candidate input. Each candidate PickUp includes transit,
pre-grasp approach, closing, settling and lift. Candidate jobs run in batches
of up to four physical rows and are filtered independently.

``trajectories.npz`` contains ``qpos[B,N,D]``, arrival intervals, valid lengths,
padding masks and candidate/grasp identities. ``report.json`` records the
actual output count and rejection reasons. The requested count is an upper
bound: empty or partial output is a normal bounded-generation result.

This is PLANNING ONLY. The sampler checks target-local gripper geometry with
the tutorial gripper model; support-plane and full-world collision checks are
not enabled. No generated trajectory is executed, no physical grasp success
is asserted, and no expert episode is committed. Scene preparation uses CPU
physics, but the shared headless scene still requires its rendering GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback
from typing import TYPE_CHECKING

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if TYPE_CHECKING:
    from embodichain.lab.trajectory_generation.integrations.atomic_candidates import (
        AtomicGenerationResult,
    )
    from embodichain.toolkits.graspkit import GraspCandidateBatch

__all__ = ["run_affordance_parallel", "main"]

_PHYSICAL_ROWS = 4
_CONTROL_DT = 0.05
_PHYSICS_DT = 0.005
_MAX_GRASPS = 32
_MAX_PROPOSALS = 64


def _save_result(
    output_dir: Path,
    result: AtomicGenerationResult,
    grasps: GraspCandidateBatch,
    *,
    seed: int,
    requested: int,
    gripper_model_id: str,
) -> dict[str, object]:
    """Persist compact planning arrays and an explicit non-expert audit."""
    batch = result.trajectories
    np.savez_compressed(
        output_dir / "trajectories.npz",
        qpos=batch.positions.detach().cpu().numpy(),
        dt=batch.dt.detach().cpu().numpy(),
        valid_length=batch.valid_length.detach().cpu().numpy(),
        valid_mask=batch.valid_mask.detach().cpu().numpy(),
        joint_names=np.asarray(batch.joint_names, dtype=np.str_),
        candidate_ids=np.asarray(
            [identity.candidate_id for identity in batch.identities], dtype=np.str_
        ),
        source_row_indices=batch.source_row_indices.detach().cpu().numpy(),
        input_grasp_poses=grasps.poses.detach().cpu().numpy(),
        input_grasp_costs=grasps.costs.detach().cpu().numpy(),
        input_grasp_valid_mask=grasps.valid_mask.detach().cpu().numpy(),
        input_grasp_ids=np.asarray(grasps.grasp_ids, dtype=np.str_),
        input_grasp_opening_widths=(
            np.empty(0, dtype=np.float32)
            if grasps.opening_widths is None
            else grasps.opening_widths.detach().cpu().numpy()
        ),
    )
    report: dict[str, object] = {
        "source": "AntipodalGraspPoseGenerator.get_grasp_candidates",
        "planning_only": True,
        "physical_validation": False,
        "world_collision_validation": False,
        "expert_episodes_committed": 0,
        "physical_envs": _PHYSICAL_ROWS,
        "canonical_sources": 1,
        "seed": seed,
        "requested_trajectories": requested,
        "output_shape": list(batch.positions.shape),
        "valid_input_grasps": int(grasps.valid_mask.sum()),
        "gripper_model_id": gripper_model_id,
        "control_dt": _CONTROL_DT,
        "planning_strategy": "ik_interp",
        "grasp_frame_to_eef": "identity; tutorial UR5/parallel-jaw TCP convention",
        "filter_ground_collision": False,
        "summary": dict(result.summary),
        "branches": [dict(item) for item in result.branch_metadata],
        "rejections": [asdict(item) for item in result.rejections],
        "planning_checks": [
            [
                {
                    "check_id": check.check_id,
                    "status": check.status,
                    "detail": check.detail,
                    "metrics": dict(check.metrics),
                }
                for check in checks.checks
            ]
            for checks in result.planning_checks
        ],
        "phases": [[asdict(phase) for phase in phases] for phases in batch.phases],
        "limitations": [
            "No rollout, contact validation, tracking verification or dataset write.",
            "Sampler collision checks cover the target-local tutorial gripper only.",
            "Opening widths are metadata; the configured hand command is unchanged.",
            "Four real physical rows; independent single-instance candidate batching is unsupported.",
        ],
    }
    encoded = json.dumps(report, indent=2, allow_nan=False)
    (output_dir / "report.json").write_text(encoded, encoding="utf-8")
    return json.loads(encoded)


def run_affordance_parallel(
    output_dir: Path,
    *,
    trajectories: int = 8,
    seed: int = 13,
    sample_count: int = 10000,
    cuda_device: int = 0,
) -> dict[str, object]:
    """Sample real antipodal grasps and generate bounded parallel PickUp plans.

    Args:
        output_dir: New or empty directory for the NPZ batch and JSON audit.
        trajectories: Maximum output count, between one and 64. Failures can
            yield fewer trajectories, including zero.
        seed: Local sampler and source seed; global Torch RNG is not reset.
        sample_count: Positive number of antipodal surface-ray samples.
        cuda_device: GPU used by the shared scene's headless renderer.

    Returns:
        Persisted planning report, including compact output shape and failures.

    Raises:
        ValueError: For invalid arguments or an occupied output directory.
        RuntimeError: For initialization, backend or stale-scene errors. These
            are not converted into ordinary failed-grasp records.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("output_dir must be new or empty")
    if type(trajectories) is not int or not 1 <= trajectories <= _MAX_PROPOSALS:
        raise ValueError("trajectories must be an integer in [1, 64]")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if type(sample_count) is not int or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    if type(cuda_device) is not int or cuda_device < 0:
        raise ValueError("cuda_device must be a nonnegative integer")

    from embodichain.lab.sim import SimulationManager

    original_error: BaseException | None = None
    try:
        return _run_affordance_parallel_scene(
            output_dir,
            trajectories=trajectories,
            seed=seed,
            sample_count=sample_count,
            cuda_device=cuda_device,
        )
    except BaseException as error:
        original_error = error
        # An exception traceback retains the inner frame's engine, camera,
        # host and native geometry. Clear that completed frame before native
        # teardown; clearing the currently executing outer frame is skipped.
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        try:
            # destroy() deliberately defers teardown until deep scene-owning
            # locals have gone out of scope. This outer function owns none.
            # The queue already performs its required garbage collection.
            SimulationManager.flush_cleanup_queue()
        except Exception as cleanup_error:
            if original_error is None:
                raise
            print(
                f"Additional native cleanup error: {cleanup_error!r}",
                file=sys.stderr,
            )


def _run_affordance_parallel_scene(
    output_dir: Path,
    *,
    trajectories: int,
    seed: int,
    sample_count: int,
    cuda_device: int,
) -> dict[str, object]:
    """Own all live scene references; queue destruction before this scope ends."""
    # Lazy imports keep CLI help and output-format tests free of scene creation.
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.atomic_actions import (
        ControlPartCommandProfile,
        GraspGoal,
        MotionPolicy,
        PickUpOptions,
        create_simulation_atomic_action_engine,
    )
    from embodichain.lab.sim.motion.expansion import (
        SceneCase,
        ValidationCheck,
        ValidationResult,
    )
    from embodichain.lab.trajectory_generation.initial_state import (
        FixedSceneHost,
        InitialStateProfile,
    )
    from embodichain.lab.trajectory_generation.integrations.atomic_candidates import (
        AtomicCandidateGenerationCfg,
        AtomicTrajectoryGenerator,
    )
    from embodichain.lab.trajectory_generation.integrations.sim import (
        SimInitialStateAdapter,
    )
    from embodichain.lab.trajectory_generation.replicas import SceneReplicaPool
    from embodichain.lab.visualization import VisualizationCfg
    from embodichain.toolkits.graspkit.pg_grasp import (
        AntipodalGraspPoseGenerator,
        AntipodalGraspPoseGeneratorCfg,
        GraspAnnotationCfg,
        ParallelJawGraspCollisionCfg,
    )
    from examples.sim.motion.trajectory_generation.cube_grasp_parallel import (
        _prepare_cube_scene,
    )
    from scripts.tutorials.atomic_action.tutorial_utils import (
        TUTORIAL_PARALLEL_JAW_MODEL,
        create_antipodal_semantics,
        create_toppra_motion_generator,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device="cpu",
            num_envs=_PHYSICAL_ROWS,
            physics_dt=_PHYSICS_DT,
            arena_space=4.0,
            gpu_id=cuda_device,
            visualization=VisualizationCfg(),
        )
    )
    motion_generator = host = None
    try:
        robot, cube, _camera, initial, _unused_yaw_poses, hand_open, hand_close = (
            _prepare_cube_scene(sim)
        )
        # Only the shared physical scene/initial state is reused. Its manually
        # authored yaw poses are deliberately NOT the candidate source.
        objects = {
            uid: sim.get_rigid_object(uid) for uid in sim.get_rigid_object_uid_list()
        }
        # The preview helper solves/settles each row separately and allows
        # small row-dependent numerical differences. A replica source instead
        # prepares the first row's measured, limit-resolved state in every real
        # environment, before capture. Do not increase equality tolerances.
        measured_initial = robot.get_qpos().clone()
        setup_metrics = {
            "helper_requested_qpos_error": float(
                (measured_initial - initial).abs().max()
            ),
            "helper_qpos_replica_error": float(
                (measured_initial - measured_initial[:1]).abs().max()
            ),
        }
        initial = measured_initial[:1].expand_as(measured_initial).clone()
        measured_objects = {
            uid: obj.get_local_pose(to_matrix=True).clone()
            for uid, obj in objects.items()
        }
        initial_objects = {
            uid: poses[:1].expand_as(poses).clone()
            for uid, poses in measured_objects.items()
        }
        for uid, poses in measured_objects.items():
            setup_metrics[f"helper_{uid}_replica_error"] = float(
                (poses - poses[:1]).abs().max()
            )
        zero = torch.zeros_like(initial)

        def prepare() -> None:
            robot.set_qpos(initial, target=False)
            robot.set_qvel(zero, target=False)
            robot.set_qpos(initial, target=True)
            robot.set_qvel(zero, target=True)
            robot.set_qf(zero)
            for uid, obj in objects.items():
                obj.set_local_pose(initial_objects[uid])
                obj.clear_dynamics()

        def signature() -> str:
            # This example constructs one common robot/object configuration
            # across all rows. Include fixed control/geometry/physics settings
            # so changing them invalidates a captured host binding.
            return json.dumps(
                {
                    "robot": robot.cfg.to_dict(),
                    "scene": {uid: obj.cfg.to_dict() for uid, obj in objects.items()},
                    "physics": sim.sim_config.physics_config.to_dict(),
                    "control_dt": _CONTROL_DT,
                    "gripper_model": TUTORIAL_PARALLEL_JAW_MODEL.to_dict(),
                },
                sort_keys=True,
                default=str,
            )

        case = SceneCase(
            "cube_antipodal",
            "fixed_initial",
            "four_replicated_5cm_cubes_on_bench_v1",
            "cube_pickup",
            "ur5_dh_pgi",
        )

        def verify(cases: tuple[SceneCase, ...]) -> ValidationResult:
            metrics = {
                **setup_metrics,
                "case_identity_matches": int(cases == (case,) * _PHYSICAL_ROWS),
                "qpos_initial_error": float((robot.get_qpos() - initial).abs().max()),
                "qvel_initial_error": float(robot.get_qvel().abs().max()),
                "qpos_replica_error": float(
                    (robot.get_qpos() - robot.get_qpos()[:1]).abs().max()
                ),
            }
            for uid, obj in objects.items():
                observed = obj.get_local_pose(to_matrix=True)
                metrics[f"{uid}_initial_error"] = float(
                    (observed - initial_objects[uid]).abs().max()
                )
                metrics[f"{uid}_replica_error"] = float(
                    (observed - observed[:1]).abs().max()
                )
            passed = (
                cases == (case,) * _PHYSICAL_ROWS
                and torch.allclose(robot.get_qpos(), initial, atol=1e-6, rtol=0)
                and torch.allclose(robot.get_qvel(), zero, atol=1e-6, rtol=0)
                and torch.allclose(
                    initial, initial[:1].expand_as(initial), atol=1e-6, rtol=0
                )
                and all(
                    torch.allclose(
                        obj.get_local_pose(to_matrix=True),
                        initial_objects[uid],
                        atol=1e-6,
                        rtol=0,
                    )
                    and torch.allclose(
                        initial_objects[uid],
                        initial_objects[uid][:1].expand_as(initial_objects[uid]),
                        atol=1e-6,
                        rtol=0,
                    )
                    for uid, obj in objects.items()
                )
            )
            return ValidationResult(
                (
                    ValidationCheck(
                        "replicated_case_initial_state",
                        "passed" if passed else "failed",
                        "Common scene configuration and identical local robot/object initial states",
                        metrics,
                    ),
                )
            )

        host = FixedSceneHost(
            SimInitialStateAdapter(sim, robot),
            InitialStateProfile(
                profile_id="affordance_parallel_initial",
                prepare=prepare,
                signature=signature,
                verify=verify,
                physics_dt=_PHYSICS_DT,
            ),
        )
        prepared = host.acquire_case((case,) * _PHYSICAL_ROWS)
        pool = SceneReplicaPool.from_host(host, prepared)
        semantics = create_antipodal_semantics(cube, label="cube")
        grasp_generator = AntipodalGraspPoseGenerator(
            TUTORIAL_PARALLEL_JAW_MODEL,
            algorithm_cfg=AntipodalGraspPoseGeneratorCfg(
                sample_count=sample_count, max_candidates=_MAX_GRASPS
            ),
            collision_cfg=ParallelJawGraspCollisionCfg(
                opening_margin=0.03,
                point_sample_density=0.012,
                # Same target-only collision policy as the atomic tutorials;
                # no table or full-world collision claim is made in this demo.
                filter_ground_collision=False,
            ),
            annotation_cfg=GraspAnnotationCfg(selection_mode="whole_mesh"),
        )
        affordance = semantics.affordance
        grasps = grasp_generator.get_grasp_candidates(
            mesh_vertices=affordance.mesh_vertices,
            mesh_triangles=affordance.mesh_triangles,
            obj_poses=pool.source_snapshot(0).entity_poses[cube.uid].unsqueeze(0),
            approach_direction=torch.tensor([0.0, 0.0, -1.0], device=robot.device),
            generator=torch.Generator(device=robot.device).manual_seed(seed),
            frame="local_arena",
        )
        motion_generator = create_toppra_motion_generator(robot)
        engine = create_simulation_atomic_action_engine(
            motion_generator,
            scene_entities=tuple(objects.values()),
            control_profiles={
                "hand": ControlPartCommandProfile.joint_positions(
                    open=hand_open, grasp=hand_close
                )
            },
            grasp_pose_generators={"hand": grasp_generator},
        )
        pickup = engine.make_invocation(
            "pick_up",
            GraspGoal(semantics),
            invocation_id="candidate_pickup",
            control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
            motion_policy=MotionPolicy(strategy="ik_interp", sample_count=240),
            skill_options=PickUpOptions(
                pre_grasp_distance=0.16,
                lift_height=0.18,
                hand_interp_steps=20,
                grasp_settle_steps=20,
                rotate_upright=None,
            ),
        )
        result = AtomicTrajectoryGenerator(engine, replica_pool=pool).generate(
            invocations=(pickup,),
            context=engine.initial_context(control_dt=_CONTROL_DT),
            candidate_inputs={(pickup.invocation_id, "primary"): grasps},
            cfg=AtomicCandidateGenerationCfg(
                max_grasps_per_case=_MAX_GRASPS,
                max_output_trajectories=trajectories,
                max_proposals=_MAX_PROPOSALS,
                max_wall_time_s=300.0,
                variants="feasible_rolls",
                seed=seed,
            ),
        )
        host.assert_current(prepared)
        return _save_result(
            output_dir,
            result,
            grasps,
            seed=seed,
            requested=trajectories,
            gripper_model_id=TUTORIAL_PARALLEL_JAW_MODEL.model_id,
        )
    finally:
        original_error = sys.exc_info()[1]
        first_cleanup_error: Exception | None = None
        cleanups = []
        if host is not None:
            cleanups.append(host.close)
        if motion_generator is not None:
            cleanups.append(motion_generator.planner.close)
        cleanups.append(lambda: sim.destroy(exit_process=False))
        for cleanup in cleanups:
            try:
                cleanup()
            except Exception as cleanup_error:
                if original_error is None and first_cleanup_error is None:
                    first_cleanup_error = cleanup_error
                else:
                    print(
                        f"Additional scene resource cleanup error: {cleanup_error!r}",
                        file=sys.stderr,
                    )
        if first_cleanup_error is not None:
            raise first_cleanup_error


def main() -> None:
    """Run the CLI; candidate exhaustion is reported separately from backend errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sample-count", type=int, default=10000)
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()
    report = None
    try:
        report = run_affordance_parallel(
            args.output,
            trajectories=args.trajectories,
            seed=args.seed,
            sample_count=args.sample_count,
            cuda_device=args.cuda_device,
        )
    except Exception as error:
        diagnostic = traceback.TracebackException.from_exception(error)
        traceback.clear_frames(error.__traceback__)
        print("".join(diagnostic.format()), file=sys.stderr, end="")
    if report is None:
        raise SystemExit(1)
    print(json.dumps({"planning_only": True, **report["summary"]}))


if __name__ == "__main__":
    main()
