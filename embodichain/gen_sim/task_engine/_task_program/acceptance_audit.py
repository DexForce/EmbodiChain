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

"""Compare versioned observers on one unchanged pre-reset native episode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from embodichain.gen_sim.task_spec import TaskSpec
from embodichain.utils.utility import load_config
from ..contracts import canonical_hash
from .control_evidence import capture_control_operations
from .coordinated_evidence import capture_coordinated_holds
from .coordinated_placement_evidence import capture_transport_holds
from .handover_evidence import capture_handover_operations
from .placement_evidence import _native_mesh
from .pour_evidence import capture_pour_motions
from .task_spec_runtime import (
    capture_final_goals,
    capture_placement_milestones,
    evaluate_runtime_evidence,
)

__all__: list[str] = []


def _state(env: Any, deployment: Any) -> dict:
    target = getattr(env, "unwrapped", env)
    bridge = getattr(target, "_active_task_program_bridge", None)
    return {
        "step_counts": target._traj_steps.tolist(),
        "robot_qpos": target.robot.get_qpos().detach().cpu().tolist(),
        "clock": (
            None
            if bridge is None
            else {
                "step": bridge._clock.step_index,
                "time": bridge._clock.now(),
                "dt": bridge._clock.step_dt,
            }
        ),
        "articulations": {
            value.entity_id: {
                "pose": target.sim.get_articulation(value.simulation_uid)
                .get_local_pose(to_matrix=True)
                .detach()
                .cpu()
                .tolist(),
                "qpos": target.sim.get_articulation(value.simulation_uid)
                .get_qpos()
                .detach()
                .cpu()
                .tolist(),
            }
            for value in deployment.integration.registration.scene_binding.articulations
        },
        "rigid_poses": {
            value.entity_id: target.sim.get_rigid_object(value.simulation_uid)
            .get_local_pose(to_matrix=True)
            .detach()
            .cpu()
            .tolist()
            for value in deployment.integration.registration.scene_binding.rigid_objects
        },
    }


def evaluate_episode_observers(
    env: Any,
    deployment: Any,
    bundle: Path,
    graph: dict,
    runtime: dict,
    contract: dict,
    count: int,
    output: Path,
) -> dict:
    """Keep legacy audit separate from the current selected verdict and execution."""
    target = getattr(env, "unwrapped", env)
    spec = TaskSpec.from_dict(load_config(bundle / "task_spec.json"))
    program = load_config(bundle / "task_program/program.yaml")
    constraints = load_config(bundle / "task_program/constraints.json")
    before = _state(env, deployment)
    runtime_digest = canonical_hash(runtime)
    reader = getattr(target, "_gen_sim_acceptance_contacts", None)
    timeline = None if reader is None else reader.snapshot()
    controls = (
        capture_control_operations(deployment, graph, runtime, program, count) or None
    )
    handovers = (
        capture_handover_operations(
            env, deployment, graph, runtime, program, constraints, count
        )
        or None
    )
    holds = (
        capture_coordinated_holds(
            env, deployment, graph, runtime, program, constraints, count
        )
        or None
    )
    pours = capture_pour_motions(env, deployment, graph, runtime, program, count)
    evaluations = {}
    for legacy in (True, False):
        final = capture_final_goals(
            env,
            deployment,
            bundle,
            graph,
            contract,
            count,
            runtime=runtime,
            pour_motions=None if legacy else pours,
            legacy_observer=legacy,
        )
        placements = capture_placement_milestones(
            env,
            deployment,
            graph,
            contract,
            runtime,
            count,
            spec=spec,
            legacy_observer=legacy,
            program=program,
        )
        coordinated = holds
        if not legacy and timeline is not None:
            coordinated = {
                **(holds or {}),
                **capture_transport_holds(graph, runtime, program, timeline),
            }
        evaluations["legacy" if legacy else "current"] = evaluate_runtime_evidence(
            spec,
            contract,
            graph,
            runtime,
            final,
            num_envs=count,
            run_id=output.as_posix(),
            program=program,
            constraints=constraints,
            placement_milestones=placements,
            control_milestones=controls,
            handover_milestones=handovers,
            coordinated_milestones=coordinated,
            pour_milestones=None if legacy else pours,
            contact_timeline=None if legacy else timeline,
            legacy_observer=legacy,
        )
    after = _state(env, deployment)
    if before != after or canonical_hash(runtime) != runtime_digest:
        raise ValueError(
            "TaskSpec observer comparison modified native state or runtime evidence."
        )
    arrays, mapping = {"terminal_steps": np.asarray(before["step_counts"])}, {}
    for index, binding in enumerate(
        deployment.integration.registration.scene_binding.rigid_objects
    ):
        vertices, faces = _native_mesh(target.sim, binding, count)
        prefix = f"rigid_{index}"
        arrays[f"{prefix}_vertices"] = vertices.numpy()
        arrays[f"{prefix}_faces"] = faces.numpy()
        arrays[f"{prefix}_terminal_pose"] = np.asarray(
            before["rigid_poses"][binding.entity_id]
        )
        mapping[binding.entity_id] = {
            "prefix": prefix,
            "simulation_uid": binding.simulation_uid,
        }
    np.savez_compressed(output / "task_spec_native_geometry.npz", **arrays)
    (output / "task_spec_terminal_snapshot.json").write_text(
        json.dumps(
            {
                "schema_version": "gen_sim.task-spec-terminal/v1",
                **before,
                "geometry": mapping,
                "geometry_file": "task_spec_native_geometry.npz",
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    if timeline is not None:
        (output / "task_spec_contact_observations.json").write_text(
            json.dumps(timeline, ensure_ascii=False, allow_nan=False) + "\n"
        )
    (output / "task_spec_legacy_evaluation.json").write_text(
        json.dumps(evaluations["legacy"], ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )
    comparison = {
        "schema_version": "gen_sim.task-spec-observer-comparison/v1",
        "same_native_episode": True,
        "native_state_unchanged": True,
        "state_digest": canonical_hash(before),
        "runtime_digest": runtime_digest,
        "task_spec_digest": canonical_hash(spec.to_dict()),
        "program_digest": canonical_hash(program),
        "constraints_digest": canonical_hash(constraints),
        "legacy_affects_selected_verdict": False,
        "legacy": [
            {key: row[key] for key in ("env_id", "status", "certificate_ready")}
            for row in evaluations["legacy"]["environments"]
        ],
        "current": [
            {key: row[key] for key in ("env_id", "status", "certificate_ready")}
            for row in evaluations["current"]["environments"]
        ],
    }
    (output / "task_spec_observer_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return evaluations["current"]
