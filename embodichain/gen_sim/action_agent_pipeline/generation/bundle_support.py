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

"""Provide robot-aware factories and metadata shared by bundle builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.action_agent_templates import (
    make_sensor_config as _make_sensor_config,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_router import (
    TaskRouteSpec,
)

__all__ = [
    "_make_sensor_config_factory_for_robot",
    "_make_sensor_config_for_robot",
    "_runtime_object_registry",
    "_with_task_route_summary",
]


def _make_sensor_config_for_robot(
    robot_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sensors = _make_sensor_config()
    wrist_parent_by_uid = {
        "cam_wrist_left": _robot_solver_end_link(robot_config, "left_arm"),
        "cam_wrist_right": _robot_solver_end_link(robot_config, "right_arm"),
    }
    for sensor in sensors:
        parent = wrist_parent_by_uid.get(str(sensor.get("uid", "")))
        if not parent:
            continue
        extrinsics = sensor.get("extrinsics")
        if isinstance(extrinsics, dict):
            extrinsics["parent"] = parent
    return sensors


def _make_sensor_config_factory_for_robot(
    robot_config: Mapping[str, Any],
) -> Callable[[], list[dict[str, Any]]]:
    def sensor_config_factory() -> list[dict[str, Any]]:
        return _make_sensor_config_for_robot(robot_config)

    return sensor_config_factory


def _robot_solver_end_link(
    robot_config: Mapping[str, Any], arm_name: str
) -> str | None:
    solver_cfg = robot_config.get("solver_cfg", {})
    if not isinstance(solver_cfg, Mapping):
        return None
    arm_solver_cfg = solver_cfg.get(arm_name, {})
    if not isinstance(arm_solver_cfg, Mapping):
        return None
    end_link_name = arm_solver_cfg.get("end_link_name")
    if end_link_name is None:
        return None
    return str(end_link_name)


def _with_task_route_summary(
    bundle: Mapping[str, Any],
    route: TaskRouteSpec,
) -> dict[str, Any]:
    routed_bundle = dict(bundle)
    routed_summary = dict(routed_bundle.get("summary", {}))
    routed_summary["task_route"] = route.to_summary()
    routed_bundle["summary"] = routed_summary
    return routed_bundle


def _runtime_object_registry(
    runtime_uids_by_source_uid: Mapping[str, str],
    *,
    by_uid: Mapping[str, SceneObject],
) -> list[dict[str, str]]:
    entries = []
    for source_uid, runtime_uid in sorted(
        runtime_uids_by_source_uid.items(),
        key=lambda item: item[1],
    ):
        obj = by_uid.get(source_uid)
        if obj is None:
            continue
        entries.append(
            {
                "runtime_uid": str(runtime_uid),
                "source_uid": str(source_uid),
                "source_role": obj.source_role,
                "description": str(obj.config.get("description", "")).strip(),
            }
        )
    return entries
