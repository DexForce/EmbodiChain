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

"""Strict optional scene sampling and pose coverage declarations."""

from __future__ import annotations
import math
import numpy as np

__all__: list[str] = []

_ROBOT_LAYOUT_STRATEGIES = ("robot_first", "bidirectional")
_LAYOUT_STRATEGIES = ("scene_first", *_ROBOT_LAYOUT_STRATEGIES)


def validate_sampling(value: dict) -> None:
    """Normalize optional settings without changing legacy configurations."""
    from ._pose_cells import _grid

    allowed = {
        "strategy",
        "layout",
        "task_mode",
        "screening",
        "robot_anchors",
        "workspace_source",
        "control_part",
        "sample_count",
        "uniform_fraction",
        "targets",
        "max_rollouts",
        "max_candidates",
        "max_ik_calls",
        "max_planning_calls",
    }
    if not any(k in value for k in ("scene_sampling", "scheduling", "pose_coverage")):
        return
    sampling = value.setdefault("scene_sampling", {})
    schedule = value.setdefault("scheduling", {})
    if not isinstance(sampling, dict) or set(sampling) - allowed:
        raise ValueError("Unknown scene_sampling settings")
    if not isinstance(schedule, dict) or set(schedule) - {
        "policy",
        "pool_size",
        "exploration_fraction",
    }:
        raise ValueError("Unknown pose scheduling settings")
    for k, v in {
        "strategy": "bounded_pose",
        "workspace_source": "joint_uniform_fk",
        "control_part": "arm",
        "sample_count": 4096,
        "uniform_fraction": 0.25,
        "targets": {},
        "max_candidates": 4096,
        "max_rollouts": value["engine"]["max_attempts"],
        "max_ik_calls": 32768,
        "max_planning_calls": 4096,
    }.items():
        sampling.setdefault(k, v)
    for k, v in {
        "policy": "fifo",
        "pool_size": 64,
        "exploration_fraction": 0.25,
    }.items():
        schedule.setdefault(k, v)
    if sampling["strategy"] not in (
        "bounded_pose",
        "workspace_joint",
        *_LAYOUT_STRATEGIES,
    ):
        raise ValueError(
            "scene_sampling.strategy must be bounded_pose, workspace_joint, "
            "scene_first, robot_first or bidirectional"
        )
    if sampling["workspace_source"] not in ("cache", "joint_uniform_fk"):
        raise ValueError(
            "workspace_source must explicitly select cache or joint_uniform_fk"
        )
    if schedule["policy"] not in ("fifo", "task_pose_coverage"):
        raise ValueError("Pose scheduling supports fifo or task_pose_coverage")
    for k in (
        "sample_count",
        "max_candidates",
        "max_rollouts",
        "max_ik_calls",
        "max_planning_calls",
    ):
        if type(sampling[k]) is not int or not 0 < sampling[k] <= 1000000:
            raise ValueError(f"{k} must be a bounded positive integer")
    if sampling["sample_count"] > 65536:
        raise ValueError("Workspace sample_count must be <= 65536")
    if sampling["max_rollouts"] < value["engine"].get("episodes_per_operator", 1):
        raise ValueError("max_rollouts must cover the requested episode quota")
    if sampling["max_candidates"] < value["engine"]["max_attempts"]:
        raise ValueError("max_candidates must cover at least one proposal per attempt")
    if type(schedule["pool_size"]) is not int or not 1 <= schedule["pool_size"] <= 64:
        raise ValueError("pool_size must be between 1 and 64")
    for obj, key in (
        (sampling, "uniform_fraction"),
        (schedule, "exploration_fraction"),
    ):
        if (
            type(obj[key]) not in (int, float)
            or not math.isfinite(obj[key])
            or not 0 <= obj[key] <= 1
        ):
            raise ValueError(f"{key} must be a probability")
    if (
        type(sampling["control_part"]) is not str
        or not sampling["control_part"].strip()
    ):
        raise ValueError("control_part must be explicit")
    if not isinstance(sampling["targets"], dict):
        raise ValueError(
            "workspace targets must map scene entities to object-from-TCP matrices"
        )
    if sampling["strategy"] == "workspace_joint" and not sampling["targets"]:
        raise ValueError(
            "workspace_joint requires explicit object-local target transforms"
        )
    for uid, pose in sampling["targets"].items():
        m = np.asarray(pose, dtype=float)
        if (
            uid not in value["scene"]
            or "follow" in value["scene"][uid]
            or m.shape != (4, 4)
            or not np.isfinite(m).all()
            or not np.allclose(m[3], [0, 0, 0, 1])
            or not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(m[:3, :3]), 1)
        ):
            raise ValueError(
                "Workspace targets require a declared independent scene entity and rigid transform"
            )
    if sampling["strategy"] == "workspace_joint" and "layout" in sampling:
        raise ValueError(
            "workspace_joint does not consume layout conditions; choose scene_first or a screened bounded_pose baseline"
        )
    if sampling["strategy"] in _LAYOUT_STRATEGIES or "layout" in sampling:
        if (
            value["engine"].get("batch_size", 1)
            * value["engine"].get("workload_count", 1)
            != 1
        ):
            raise ValueError("Layout task-path screening currently requires B=1")
        layout = sampling.get("layout")
        if (
            not isinstance(layout, dict)
            or set(layout) - {"support_uid", "parent_by_uid", "policy", "conditions"}
            or not {"support_uid", "parent_by_uid", "policy"} <= layout.keys()
        ):
            raise ValueError(
                "Layout sampling requires support_uid, parent_by_uid and policy"
            )
        if (
            not isinstance(layout["support_uid"], str)
            or not layout["support_uid"].strip()
            or not isinstance(layout["parent_by_uid"], dict)
            or not layout["parent_by_uid"]
            or any(
                type(k) is not str or type(v) is not str
                for k, v in layout["parent_by_uid"].items()
            )
        ):
            raise ValueError(
                "layout requires an explicit support UID and parent mapping"
            )
        if not isinstance(layout["policy"], dict):
            raise ValueError("layout.policy must be a mapping")
        mode = sampling.get("task_mode")
        if (
            not isinstance(mode, dict)
            or set(mode) - {"mode_id", "phases"}
            or not {"mode_id", "phases"} <= mode.keys()
            or not isinstance(mode["phases"], list)
            or not mode["phases"]
        ):
            raise ValueError(
                "Layout sampling requires a declared task_mode phase graph"
            )
        screening = sampling.setdefault("screening", {})
        if not isinstance(screening, dict) or set(screening) - {
            "beam_width",
            "ik_seeds",
            "max_rrt_nodes",
            "planning_timeout_s",
        }:
            raise ValueError("Unknown scene-first screening budgets")
        for key, default in {
            "beam_width": 2,
            "ik_seeds": 2,
            "max_rrt_nodes": 64,
            "planning_timeout_s": 0.5,
        }.items():
            screening.setdefault(key, default)
        for key in ("beam_width", "ik_seeds", "max_rrt_nodes"):
            if type(screening[key]) is not int or not 1 <= screening[key] <= 4096:
                raise ValueError("Screening counts must be positive and bounded")
        if (
            type(screening["planning_timeout_s"]) not in (int, float)
            or not math.isfinite(screening["planning_timeout_s"])
            or not 0 < screening["planning_timeout_s"] <= 60
        ):
            raise ValueError("Screening timeout must be positive and bounded")
        from embodichain.lab.sim.scene.expansion.layout import _policy_data
        from embodichain.lab.sim.scene.feasibility import request_from_mapping

        normalized_policy = _policy_data(layout["policy"])
        if any(
            "follow" in value["scene"].get(uid, {})
            for uid in normalized_policy["intra_group"]["targets"]
        ):
            raise ValueError(
                "Rigid scene followers cannot also be independent intra-group targets"
            )
        parents = layout["parent_by_uid"]
        support = layout["support_uid"]
        for uid in parents:
            seen = set()
            current = uid
            while current != support:
                if current in seen or current not in parents:
                    raise ValueError("Layout parents must form one rooted support tree")
                seen.add(current)
                current = parents[current]
        identities = {support, *parents, *value["scene"]}
        request = request_from_mapping(
            dict(
                task_id="preflight",
                robot_profile_id="preflight",
                policy_id="preflight",
                validator_id="preflight",
                parent_scene_id="preflight",
                entity_poses={uid: np.eye(4).tolist() for uid in identities},
                variables=[],
                modes=[mode],
            )
        )
        for uid, parent in parents.items():
            if (
                parent == support
                and uid in value["scene"]
                and "follow" not in value["scene"][uid]
            ):
                bounds = value["scene"][uid]
                if bounds["yaw_rad"] != [0, 0] or bounds["translation_m"][2] != [0, 0]:
                    raise ValueError("Root layout domains currently support XY only")
        conditions = layout.get("conditions", {})
        if not isinstance(conditions, dict):
            raise ValueError("layout.conditions must map UIDs to arena poses")
        for uid, m in conditions.items():
            m = np.asarray(m, dtype=float)
            if (
                m.shape != (4, 4)
                or not np.isfinite(m).all()
                or not np.allclose(m[3], [0, 0, 0, 1])
                or not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(m[:3, :3]), 1)
            ):
                raise ValueError("layout conditions require proper rigid matrices")
        if (
            "robot_anchors" in sampling
            or sampling["strategy"] in _ROBOT_LAYOUT_STRATEGIES
        ):
            from ._robot_first_sampling import _anchor_targets

            anchors = _anchor_targets(
                request.modes[0],
                layout,
                normalized_policy,
                sampling.get("robot_anchors"),
            )
            if any(
                a.root_uid not in value["scene"]
                or "follow" in value["scene"][a.root_uid]
                for a in anchors
            ):
                raise ValueError(
                    "A robot anchor requires an independently declared scene root"
                )
    elif "task_mode" in sampling:
        raise ValueError("task_mode screening requires an explicit layout")
    elif "robot_anchors" in sampling:
        raise ValueError("robot_anchors requires an explicit layout and task_mode")
    coverage = value.get("pose_coverage")
    if coverage is None:
        if schedule["policy"] == "task_pose_coverage":
            raise ValueError(
                "task_pose_coverage scheduling requires pose_coverage grids"
            )
        return
    if not isinstance(coverage, dict) or set(coverage) != {
        "entity",
        "place_reference",
        "grids",
    }:
        raise ValueError("pose_coverage requires entity, place_reference and grids")
    if (
        any(coverage[k] not in value["scene"] for k in ("entity", "place_reference"))
        or coverage["entity"] == coverage["place_reference"]
    ):
        raise ValueError(
            "Coverage entity and place_reference must select distinct scene entities"
        )
    if not isinstance(coverage["grids"], dict) or set(coverage["grids"]) != {
        "pick",
        "place",
    }:
        raise ValueError("Coverage requires pick and place grids")
    coverage["grids"] = {k: _grid(v) for k, v in coverage["grids"].items()}
