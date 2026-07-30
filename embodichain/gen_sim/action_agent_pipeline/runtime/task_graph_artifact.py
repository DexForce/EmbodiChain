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

"""Persist one grounded runtime Task graph per environment and episode."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import torch

from embodichain.gen_sim.action_agent_pipeline.domain.seed_task_graph import (
    seed_task_graph_hash,
)
from embodichain.utils.logger import log_warning

__all__ = ["RuntimeTaskGraphRecorder"]

_SAFE_COMPONENT_RE = re.compile(r"[^0-9A-Za-z._-]+")


class RuntimeTaskGraphRecorder:
    """Maintain isolated JSON execution records for a vectorized environment."""

    def __init__(
        self,
        seed_graph: Mapping[str, Any],
        *,
        env: Any,
        run_id: str | None,
        episode_index: int,
        graph_renderer: Callable[[Mapping[str, Any]], bytes] | None = None,
    ) -> None:
        self.seed_graph = deepcopy(dict(seed_graph))
        self.num_envs = int(env.num_envs)
        self.run_id = _safe_component(run_id or _new_run_id())
        self.episode_index = int(episode_index)
        self.graph_renderer = graph_renderer
        self.robot_profile = str(getattr(env, "agent_robot_profile", "unknown"))
        self.output_dir = (
            _outputs_root()
            / _safe_component(str(seed_graph["task"]))
            / "runs"
            / self.run_id
            / f"episode_{self.episode_index:04d}"
        )
        self.documents = [
            self._initial_document(env_id) for env_id in range(self.num_envs)
        ]
        self._edge_by_id = [
            {str(edge["id"]): edge for edge in document["edges"]}
            for document in self.documents
        ]
        self._step_by_id = [
            {str(step["id"]): step for step in document["semantic_steps"]}
            for document in self.documents
        ]
        self._step_id_by_edge = {
            str(edge_id): str(step["id"])
            for step in self.seed_graph["semantic_steps"]
            for edge_id in step["edge_ids"]
        }
        self._incoming_by_node: dict[str, list[str]] = {}
        for edge in self.seed_graph["edges"]:
            self._incoming_by_node.setdefault(str(edge["target"]), []).append(
                str(edge["id"])
            )

    def begin_step(
        self,
        step: Any,
        *,
        assignments: Sequence[str | None],
        object_pose: torch.Tensor | None,
        reference_pose: torch.Tensor | None,
        active_mask: torch.Tensor,
        selection_failed_mask: torch.Tensor,
        physical_control_parts: Sequence[str | None] | None = None,
        arrangement_metadata: Sequence[Mapping[str, Any]] | None = None,
        candidate_failures: Sequence[Mapping[str, str | None]] | None = None,
    ) -> None:
        for env_id, document in enumerate(self.documents):
            record = self._step_by_id[env_id][step.id]["runtime"]
            selection_failed = bool(selection_failed_mask[env_id].item())
            if selection_failed:
                record["status"] = "failed"
                failures = (
                    dict(candidate_failures[env_id])
                    if candidate_failures is not None
                    else {}
                )
                failed_phases = ", ".join(
                    f"{arm}={phase}"
                    for arm, phase in failures.items()
                    if phase is not None
                )
                record["failure_reason"] = "no feasible arm candidate" + (
                    f" ({failed_phases})" if failed_phases else ""
                )
                record["candidate_failures"] = failures
            else:
                record["status"] = (
                    "grounding" if bool(active_mask[env_id].item()) else "skipped"
                )
            record["assigned_arm"] = assignments[env_id]
            record["physical_control_part"] = (
                physical_control_parts[env_id]
                if physical_control_parts is not None
                else assignments[env_id]
            )
            record["observed_object_pose"] = _pose_at(object_pose, env_id)
            record["observed_reference_pose"] = _pose_at(reference_pose, env_id)
            if arrangement_metadata is not None:
                record["arrangement"] = deepcopy(dict(arrangement_metadata[env_id]))
            document["status"] = "running"

    def record_edge(
        self,
        edge_id: str,
        *,
        assignments: Sequence[str | None],
        grounded_actions: Sequence[Any],
        failed_before: torch.Tensor,
        failed_after: torch.Tensor,
        grounding_failed: torch.Tensor,
        action_steps: int,
        arm_actions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        failure_class: str = "fatal",
    ) -> None:
        if failure_class not in {"cleanup", "fatal"}:
            raise ValueError(f"Unsupported runtime failure class: {failure_class!r}.")
        for env_id in range(self.num_envs):
            edge = self._edge_by_id[env_id][edge_id]
            was_active = not bool(failed_before[env_id].item())
            failed_during_grounding = bool(grounding_failed[env_id].item())
            did_fail = failed_during_grounding or (
                bool(failed_after[env_id].item()) and was_active
            )
            if len(edge["actions"]) == 1 and grounded_actions:
                assigned_arm = assignments[env_id]
                grounded_for_env = next(
                    (
                        grounded
                        for grounded in grounded_actions
                        if grounded.action_spec.get("robot_name") == assigned_arm
                    ),
                    grounded_actions[0],
                )
                grounded_pairs = ((edge["actions"][0], grounded_for_env),)
            elif len(edge["actions"]) == len(grounded_actions):
                grounded_pairs = tuple(zip(edge["actions"], grounded_actions))
            else:
                raise ValueError(
                    f"Runtime edge {edge_id!r} action count changed during grounding."
                )
            env_arm_actions = (
                arm_actions[env_id]
                if not isinstance(arm_actions, Mapping)
                else arm_actions
            )
            for action_record, grounded in grounded_pairs:
                assigned_arm = (
                    assignments[env_id]
                    if len(edge["actions"]) == 1
                    else str(grounded.action_spec["robot_name"])
                )
                runtime = action_record.setdefault("runtime", {})
                selected_action = _selected_arm_action(
                    assigned_arm,
                    env_arm_actions,
                )
                resolved_object_pose = getattr(
                    selected_action,
                    "resolved_object_target_pose",
                    None,
                )
                resolved_eef_pose = getattr(
                    selected_action,
                    "resolved_eef_target_pose",
                    None,
                )
                resolved_left_eef_pose = getattr(
                    selected_action,
                    "resolved_left_eef_target_pose",
                    None,
                )
                resolved_right_eef_pose = getattr(
                    selected_action,
                    "resolved_right_eef_target_pose",
                    None,
                )
                resolved_target_positions = _pose_positions(resolved_object_pose)
                if resolved_target_positions is None:
                    resolved_target_positions = grounded.target_object_pose
                runtime.update(
                    {
                        "assigned_arm": assigned_arm,
                        "semantic_arm": assigned_arm,
                        "physical_control_part": self._step_by_id[env_id][
                            self._step_id_by_edge[edge_id]
                        ]["runtime"].get("physical_control_part"),
                        "phase": (
                            "pickup"
                            if action_record.get("atomic_action_class") == "PickUp"
                            else action_record.get("target_binding", {}).get("phase")
                        ),
                        "status": (
                            "failed"
                            if did_fail
                            else ("executed" if was_active else "skipped")
                        ),
                        "observed_object_pose": _pose_at(grounded.object_pose, env_id),
                        "observed_reference_pose": _pose_at(
                            grounded.reference_pose, env_id
                        ),
                        "resolved_target_object_pose": _pose_at(
                            resolved_object_pose, env_id
                        )
                        or _target_pose_at(
                            grounded.target_object_pose,
                            grounded.object_pose,
                            env_id,
                        ),
                        "resolved_target_position": _position_at(
                            resolved_target_positions,
                            env_id,
                        ),
                        "resolved_eef_pose": (
                            {
                                "left_arm": _pose_at(resolved_left_eef_pose, env_id),
                                "right_arm": _pose_at(resolved_right_eef_pose, env_id),
                            }
                            if resolved_left_eef_pose is not None
                            or resolved_right_eef_pose is not None
                            else _pose_at(resolved_eef_pose, env_id)
                        ),
                        "resolved_motion_policy": _motion_policy_at(
                            grounded.motion_policy,
                            env_id,
                            self.num_envs,
                        ),
                        "planning": _planning_record(
                            assigned_arm,
                            env_id,
                            env_arm_actions,
                            did_fail=did_fail,
                            was_active=was_active or failed_during_grounding,
                        ),
                        "execution": {
                            "status": (
                                "skipped"
                                if not was_active
                                else ("failed" if did_fail else "executed")
                            ),
                            "action_step_count": (
                                int(action_steps) if was_active else 0
                            ),
                        },
                        "failure_reason": (
                            (
                                self._step_by_id[env_id][
                                    self._step_id_by_edge[edge_id]
                                ]["runtime"].get(
                                    "failure_reason",
                                    "no feasible arm candidate",
                                )
                                if failed_during_grounding
                                else "IK or motion planning failed"
                            )
                            if did_fail
                            else None
                        ),
                        "failure_class": failure_class if did_fail else None,
                    }
                )
            if did_fail and failure_class == "cleanup":
                self.documents[env_id]["cleanup_failures"].append(edge_id)
            target_node = str(edge["target"])
            for node in self.documents[env_id]["nodes"]:
                if node["id"] == target_node:
                    node["runtime_status"] = self._joined_node_status(
                        env_id,
                        target_node,
                    )
                    break

    def complete_step(
        self,
        step_id: str,
        *,
        success: torch.Tensor,
        failed_mask: torch.Tensor,
        observed_positions: torch.Tensor,
        target_positions: torch.Tensor | None,
        position_error: torch.Tensor | None,
        tolerance: float | None,
        cleanup_failed_mask: torch.Tensor | None = None,
    ) -> None:
        for env_id, document in enumerate(self.documents):
            record = self._step_by_id[env_id][step_id]["runtime"]
            succeeded = bool(success[env_id].item())
            prior_status = record["status"]
            if prior_status != "skipped":
                record["status"] = "success" if succeeded else "failed"
            record["postcondition"] = {
                "evaluated": prior_status not in {"failed", "skipped"},
                "success": succeeded,
                "observed_object_position": _position_at(observed_positions, env_id),
                "target_position": _position_at(target_positions, env_id),
                "position_error": (
                    float(position_error[env_id].item())
                    if position_error is not None
                    else None
                ),
                "tolerance": tolerance,
            }
            record["cleanup_status"] = (
                "degraded"
                if cleanup_failed_mask is not None
                and bool(cleanup_failed_mask[env_id].item())
                else "complete"
            )
            if bool(failed_mask[env_id].item()) and not succeeded:
                record.setdefault("failure_reason", "postcondition not satisfied")
            if prior_status != "skipped":
                document["last_completed_semantic_step"] = step_id
        self.checkpoint()

    def checkpoint(self) -> None:
        """Atomically replace every environment JSON after one semantic step."""
        files = []
        for env_id, document in enumerate(self.documents):
            directory = self._env_dir(env_id)
            directory.mkdir(parents=True, exist_ok=True)
            files.append(
                (
                    directory / "task_graph.json",
                    json.dumps(document, ensure_ascii=False, indent=4) + "\n",
                )
            )
        _write_file_transaction(files)

    def finalize(
        self,
        failed_mask: torch.Tensor | None,
        *,
        aborted_reason: str | None = None,
        relation_success: torch.Tensor | None = None,
        cleanup_failed_mask: torch.Tensor | None = None,
    ) -> None:
        """Publish final JSON and PNG even when execution aborted."""
        files: list[tuple[Path, str | bytes]] = []
        for env_id, document in enumerate(self.documents):
            if aborted_reason is not None:
                document["status"] = "aborted"
                document["failure_reason"] = aborted_reason
            elif failed_mask is not None and bool(failed_mask[env_id].item()):
                document["status"] = "failed"
            else:
                document["status"] = "success"
            document["relation_success"] = (
                bool(relation_success[env_id].item())
                if relation_success is not None
                else None
            )
            document["cleanup_degraded"] = (
                bool(cleanup_failed_mask[env_id].item())
                if cleanup_failed_mask is not None
                else False
            )
            document["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            directory = self._env_dir(env_id)
            directory.mkdir(parents=True, exist_ok=True)
            files.append(
                (
                    directory / "task_graph.json",
                    json.dumps(document, ensure_ascii=False, indent=4) + "\n",
                )
            )
            if self.graph_renderer is not None:
                try:
                    rendered_graph = self.graph_renderer(document)
                except Exception as exc:
                    log_warning(
                        "Runtime Task graph visualization failed for "
                        f"env {env_id}: {exc}"
                    )
                else:
                    files.append(
                        (
                            directory / "task_graph.png",
                            rendered_graph,
                        )
                    )
        _write_file_transaction(files)

    def _initial_document(self, env_id: int) -> dict[str, Any]:
        nodes = deepcopy(self.seed_graph["nodes"])
        for node in nodes:
            node["runtime_status"] = (
                "reached" if node["id"] == self.seed_graph["start"] else "pending"
            )
        steps = deepcopy(self.seed_graph["semantic_steps"])
        for step in steps:
            step["runtime"] = {
                "status": "pending",
                "assigned_arm": None,
                "postcondition": {"evaluated": False, "success": None},
            }
        edges = deepcopy(self.seed_graph["edges"])
        for edge in edges:
            for action in edge["actions"]:
                action["runtime"] = {
                    "status": "pending",
                    "assigned_arm": None,
                }
        return {
            "schema_version": "runtime_task_graph_v3",
            "task": self.seed_graph["task"],
            "run_id": self.run_id,
            "episode_index": self.episode_index,
            "env_id": env_id,
            "robot_profile": self.robot_profile,
            "seed_graph_schema_version": self.seed_graph["schema_version"],
            "seed_graph_hash": seed_task_graph_hash(self.seed_graph),
            "motion_policy_version": self.seed_graph["motion_policy_version"],
            "start": self.seed_graph["start"],
            "goal": self.seed_graph["goal"],
            "nodes": nodes,
            "edges": edges,
            "semantic_step_schema_version": self.seed_graph[
                "semantic_step_schema_version"
            ],
            "semantic_steps": steps,
            "allocation_groups": deepcopy(self.seed_graph.get("allocation_groups", [])),
            "status": "pending",
            "cleanup_failures": [],
            "cleanup_degraded": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def _env_dir(self, env_id: int) -> Path:
        return self.output_dir / f"env_{env_id:04d}"

    def _joined_node_status(self, env_id: int, node_id: str) -> str:
        """Reach a fork/join node only after all of its incoming actions finish."""
        incoming = self._incoming_by_node.get(node_id, [])
        statuses = []
        for edge_id in incoming:
            edge = self._edge_by_id[env_id][edge_id]
            statuses.extend(
                action.get("runtime", {}).get("status", "pending")
                for action in edge["actions"]
            )
        if any(status == "pending" for status in statuses):
            return "pending"
        if any(status == "failed" for status in statuses):
            return "failed"
        if statuses and all(status == "skipped" for status in statuses):
            return "skipped"
        return "reached"


def _planning_record(
    assignment: str | None,
    env_id: int,
    arm_actions: Mapping[str, Any],
    *,
    did_fail: bool,
    was_active: bool,
) -> dict[str, Any]:
    side = (
        "left"
        if assignment == "left_arm"
        else (
            "right"
            if assignment == "right_arm"
            else ("left" if assignment == "coordinated" else None)
        )
    )
    executed = arm_actions.get(side) if side is not None else None
    action = getattr(executed, "action", None)
    trajectory_steps = (
        int(action.shape[-2]) if action is not None and action.ndim >= 2 else 0
    )
    return {
        "status": (
            "skipped" if not was_active else ("failed" if did_fail else "planned")
        ),
        "ik_success": None if not was_active else not did_fail,
        "trajectory_step_count": trajectory_steps if was_active else 0,
    }


def _motion_policy_at(
    policy: Mapping[str, Any],
    env_id: int,
    num_envs: int,
) -> dict[str, Any]:
    """Project per-environment resolved policy arrays into one Task graph."""
    result = deepcopy(dict(policy))
    heights = result.pop("resolved_retreat_height_by_env", None)
    if heights is not None:
        if not isinstance(heights, Sequence) or len(heights) != num_envs:
            raise ValueError(
                "resolved_retreat_height_by_env must match the environment batch."
            )
        result["resolved_retreat_height"] = float(heights[env_id])
    return result


def _selected_arm_action(
    assignment: str | None,
    arm_actions: Mapping[str, Any],
) -> Any:
    side = (
        "left"
        if assignment == "left_arm"
        else (
            "right"
            if assignment == "right_arm"
            else ("left" if assignment == "coordinated" else None)
        )
    )
    if side is None:
        return None
    return arm_actions.get(side)


def _pose_at(pose: torch.Tensor | None, env_id: int) -> dict[str, Any] | None:
    if pose is None:
        return None
    if pose.ndim not in {2, 3} or tuple(pose.shape[-2:]) != (4, 4):
        raise ValueError(
            f"Runtime pose must have shape (4, 4) or (N, 4, 4), got "
            f"{tuple(pose.shape)}."
        )
    value = pose[env_id] if pose.ndim == 3 else pose
    return {
        "position": [float(item) for item in value[:3, 3].detach().cpu().tolist()],
        "rotation_matrix": [
            [float(item) for item in row]
            for row in value[:3, :3].detach().cpu().tolist()
        ],
    }


def _pose_positions(pose: torch.Tensor | None) -> torch.Tensor | None:
    if pose is None:
        return None
    if pose.ndim == 2 and tuple(pose.shape) == (4, 4):
        return pose[:3, 3].unsqueeze(0)
    if pose.ndim == 3 and tuple(pose.shape[-2:]) == (4, 4):
        return pose[:, :3, 3]
    raise ValueError(
        f"Runtime pose must have shape (4, 4) or (N, 4, 4), got "
        f"{tuple(pose.shape)}."
    )


def _target_pose_at(
    positions: torch.Tensor | None,
    orientation_source: torch.Tensor | None,
    env_id: int,
) -> dict[str, Any] | None:
    if positions is None:
        return None
    rotation = (
        orientation_source[env_id, :3, :3].detach().cpu().tolist()
        if isinstance(orientation_source, torch.Tensor)
        else torch.eye(3).tolist()
    )
    return {
        "position": [float(item) for item in positions[env_id].detach().cpu().tolist()],
        "rotation_matrix": [[float(item) for item in row] for row in rotation],
    }


def _position_at(
    positions: torch.Tensor | None,
    env_id: int,
) -> list[float] | None:
    if positions is None:
        return None
    value = positions if positions.ndim == 1 else positions[env_id]
    return [float(item) for item in value.detach().cpu().tolist()]


def _safe_component(value: str) -> str:
    result = _SAFE_COMPONENT_RE.sub("_", str(value)).strip("._")
    if not result:
        raise ValueError("Runtime graph path component must not be empty.")
    return result


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _outputs_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "setup.py").is_file() and (parent / "embodichain").is_dir():
            return parent / "outputs" / "graph"
    raise RuntimeError("Unable to resolve outputs/graph for runtime Task graphs.")


def _write_file_transaction(files: list[tuple[Path, str | bytes]]) -> None:
    """Atomically publish one runtime checkpoint without cross-layer imports."""
    staged: list[tuple[Path, Path]] = []
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for destination, content in files:
            is_binary = isinstance(content, bytes)
            descriptor, temp_name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                text=not is_binary,
            )
            temp_path = Path(temp_name)
            mode = "wb" if is_binary else "w"
            with os.fdopen(
                descriptor,
                mode,
                encoding=None if is_binary else "utf-8",
            ) as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            temp_path.chmod(0o644)
            staged.append((destination, temp_path))
        for destination, temp_path in staged:
            if destination.exists():
                descriptor, backup_name = tempfile.mkstemp(
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".bak",
                )
                os.close(descriptor)
                backup_path = Path(backup_name)
                backup_path.unlink()
                os.replace(destination, backup_path)
                backups[destination] = backup_path
            os.replace(temp_path, destination)
            installed.append(destination)
    except BaseException:
        for destination in reversed(installed):
            destination.unlink(missing_ok=True)
        for destination, backup_path in reversed(list(backups.items())):
            if backup_path.exists():
                os.replace(backup_path, destination)
        raise
    else:
        for backup_path in backups.values():
            backup_path.unlink(missing_ok=True)
    finally:
        for _, temp_path in staged:
            temp_path.unlink(missing_ok=True)
