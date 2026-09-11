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

"""Task Engine-owned E1-E5 service decoding and deployment composition."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from embodichain.lab.task_program.integrations._configured_composition import (
    _compose_integration_payload,
    _resolve_task_program_components,
)
from embodichain.lab.task_program.integrations.configured import (
    _mapping,
    _identifier,
    _identifier_tuple,
    _sequence,
    _finite_tuple,
    _real,
    _decode_scene,
    _decode_robot_profile,
    _decode_runtime_services,
    _decode_registered_lowerer as _decode_shared_lowerer,
)
from embodichain.lab.task_program.integrations.catalog import (
    SimulationTaskProgramRegistration,
)
from embodichain.lab.task_program.language import TaskProgramIntegrationCfg
from embodichain.lab.task_program.semantics import (
    SemanticCallDescriptor,
    RegisteredSemanticCall,
    builtin_semantic_call_catalog,
)
from ..contracts import canonical_hash
from .services import (
    _AbsolutePoseTarget,
    _SceneEntityTarget,
    _PickRoute,
    make_pick_factory,
    _MoveHeldObjectRoute,
    _MoveHeldObjectLowererFactory,
    _CoordinatedTransportRoute,
    _CoordinatedTransportLowererFactory,
    _CoordinatedHoldLowererFactory,
    _RelativePlaceRoute,
    _RelativePlaceLowererFactory,
)

__all__: list[str] = []


def _decode_goal_pose(
    value: object, *, path: str
) -> _AbsolutePoseTarget | _SceneEntityTarget:
    """Decode one closed absolute or scene-relative target declaration."""
    common = _mapping(
        value,
        path=path,
        required=frozenset({"kind"}),
        optional=frozenset(
            {"position", "quaternion_wxyz", "entity_id", "relative_pose"}
        ),
    )
    kind = _identifier(common["kind"], path=f"{path}.kind")
    if kind == "pose":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "position", "quaternion_wxyz"}),
        )
        return _AbsolutePoseTarget(
            _finite_tuple(
                config["position"], path=f"{path}.position", expected_length=3
            ),
            _finite_tuple(
                config["quaternion_wxyz"],
                path=f"{path}.quaternion_wxyz",
                expected_length=4,
            ),
        )
    if kind == "scene_entity":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "entity_id"}),
            optional=frozenset({"relative_pose"}),
        )
        return _SceneEntityTarget(
            _identifier(config["entity_id"], path=f"{path}.entity_id"),
            relative_pose=(
                None
                if "relative_pose" not in config
                else _finite_tuple(
                    config["relative_pose"],
                    path=f"{path}.relative_pose",
                    expected_length=16,
                )
            ),
        )
    raise ValueError(
        f"Unsupported {path}.kind {kind!r}; expected pose or scene_entity."
    )


def decode_task_lowerer(value: object, *, path: str) -> Any:
    """Decode only the E1-E5 task-owned routes or established shared services."""
    if type(value) is not dict:
        raise TypeError(f"{path} must be a mapping.")
    kind = _identifier(value.get("kind"), path=f"{path}.kind")
    if kind == "pick":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "routes"}),
            optional=frozenset({"call_id"}),
        )
        call_id = _identifier(
            config.get("call_id", "simulation.pick"), path=f"{path}.call_id"
        )
        if call_id != "simulation.pick" and not call_id.startswith("gen_sim.pick."):
            raise ValueError("Task Pick aliases must use the gen_sim.pick namespace.")
        routes: list[_PickRoute] = []
        for index, raw in enumerate(_sequence(config["routes"], path=f"{path}.routes")):
            route_path = f"{path}.routes[{index}]"
            route = _mapping(
                raw,
                path=route_path,
                required=frozenset({"object_id", "target_id"}),
                optional=frozenset(
                    {
                        "release_clearance_object_pose",
                        "release_clearance_plane_z",
                        "release_clearance_safety_margin",
                        "grasp_region",
                    }
                ),
            )
            routes.append(
                _PickRoute(
                    object_id=_identifier(
                        route["object_id"], path=f"{route_path}.object_id"
                    ),
                    target_id=_identifier(
                        route["target_id"], path=f"{route_path}.target_id"
                    ),
                    release_clearance_object_pose=(
                        None
                        if "release_clearance_object_pose" not in route
                        else _decode_goal_pose(
                            route["release_clearance_object_pose"],
                            path=f"{route_path}.release_clearance_object_pose",
                        )
                    ),
                    release_clearance_plane_z=(
                        None
                        if "release_clearance_plane_z" not in route
                        else _real(
                            route["release_clearance_plane_z"],
                            path=f"{route_path}.release_clearance_plane_z",
                        )
                    ),
                    release_clearance_safety_margin=_real(
                        route.get("release_clearance_safety_margin", 0.0),
                        path=f"{route_path}.release_clearance_safety_margin",
                        minimum=0.0,
                    ),
                    grasp_region=(
                        None
                        if "grasp_region" not in route
                        else _identifier(
                            route["grasp_region"], path=f"{route_path}.grasp_region"
                        )
                    ),
                )
            )
        return make_pick_factory(tuple(routes), call_id)
    if kind == "move_held_object":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "routes"}),
            optional=frozenset({"phase_protection"}),
        )
        routes: list[_MoveHeldObjectRoute] = []
        for index, raw in enumerate(_sequence(config["routes"], path=f"{path}.routes")):
            route_path = f"{path}.routes[{index}]"
            route = _mapping(
                raw,
                path=route_path,
                required=frozenset({"object_id", "target_id", "pose"}),
            )
            routes.append(
                _MoveHeldObjectRoute(
                    object_id=_identifier(
                        route["object_id"], path=f"{route_path}.object_id"
                    ),
                    target_id=_identifier(
                        route["target_id"], path=f"{route_path}.target_id"
                    ),
                    pose=_decode_goal_pose(route["pose"], path=f"{route_path}.pose"),
                )
            )
        return _MoveHeldObjectLowererFactory(
            tuple(routes),
            phase_protection=(
                _identifier(config["phase_protection"], path=f"{path}.phase_protection")
                if "phase_protection" in config
                else None
            ),
        )
    if kind in {"coordinated_transport", "coordinated_hold"}:
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "routes"}),
        )
        routes: list[_CoordinatedTransportRoute] = []
        for index, route_value in enumerate(
            _sequence(config["routes"], path=f"{path}.routes")
        ):
            route_path = f"{path}.routes[{index}]"
            route = _mapping(
                route_value,
                path=route_path,
                required=frozenset({"object_id", "target_id"}),
                optional=frozenset(
                    {
                        "reference_entity_id",
                        "relative_pose",
                        "world_displacement",
                    }
                ),
            )
            has_reference = "reference_entity_id" in route
            has_pose = "relative_pose" in route
            has_displacement = "world_displacement" in route
            if has_reference != has_pose or has_reference == has_displacement:
                raise ValueError(
                    f"{route_path} must declare exactly one of "
                    "reference_entity_id with relative_pose or "
                    "world_displacement."
                )
            routes.append(
                _CoordinatedTransportRoute(
                    object_id=_identifier(
                        route["object_id"], path=f"{route_path}.object_id"
                    ),
                    target_id=_identifier(
                        route["target_id"], path=f"{route_path}.target_id"
                    ),
                    reference_entity_id=(
                        _identifier(
                            route["reference_entity_id"],
                            path=f"{route_path}.reference_entity_id",
                        )
                        if has_reference
                        else None
                    ),
                    relative_pose=(
                        _finite_tuple(
                            route["relative_pose"],
                            path=f"{route_path}.relative_pose",
                            expected_length=16,
                        )
                        if has_pose
                        else None
                    ),
                    world_displacement=(
                        _finite_tuple(
                            route["world_displacement"],
                            path=f"{route_path}.world_displacement",
                            expected_length=3,
                        )
                        if has_displacement
                        else None
                    ),
                )
            )
        factory_type = (
            _CoordinatedTransportLowererFactory
            if kind == "coordinated_transport"
            else _CoordinatedHoldLowererFactory
        )
        return factory_type(routes=tuple(routes))
    if kind == "place_relative":
        config = _mapping(
            value,
            path=path,
            required=frozenset({"kind", "routes"}),
        )
        routes: list[_RelativePlaceRoute] = []
        for index, route_value in enumerate(
            _sequence(config["routes"], path=f"{path}.routes")
        ):
            route_path = f"{path}.routes[{index}]"
            route = _mapping(
                route_value,
                path=route_path,
                required=frozenset(
                    {
                        "object_id",
                        "reference_entity_id",
                        "relation",
                        "world_displacement",
                    }
                ),
            )
            routes.append(
                _RelativePlaceRoute(
                    object_id=_identifier(
                        route["object_id"],
                        path=f"{route_path}.object_id",
                    ),
                    reference_entity_id=_identifier(
                        route["reference_entity_id"],
                        path=f"{route_path}.reference_entity_id",
                    ),
                    relation=_identifier(
                        route["relation"],
                        path=f"{route_path}.relation",
                    ),
                    world_displacement=_finite_tuple(
                        route["world_displacement"],
                        path=f"{route_path}.world_displacement",
                        expected_length=3,
                    ),
                )
            )
        return _RelativePlaceLowererFactory(routes=tuple(routes))
    if kind in {"park", "pour", "axis_align"}:
        return _decode_shared_lowerer(value, path=path)
    raise ValueError(
        f"{path}: unsupported Task Engine service {kind!r}; expected E1-E5 services."
    )


@dataclass(frozen=True, slots=True)
class TaskIntegration:
    registration: SimulationTaskProgramRegistration
    integration_fingerprint: str
    grasp_factories: tuple[tuple[str, Any], ...]
    adapter_factory: Any = None


@dataclass(frozen=True, slots=True)
class TaskDeployment:
    integration_id: str
    program_id: str
    program_path: Path
    selection: TaskProgramIntegrationCfg
    integration: TaskIntegration
    scene_binding: dict[str, Any]


def compose_deployment(
    *, task_program: object, skill_profile: object, base_dir: str | Path
) -> TaskDeployment:
    """Reuse component contracts while owning E1-E5 service decoding locally."""
    program_path, task, policy = _resolve_task_program_components(
        task_program,
        base_dir=Path(base_dir).expanduser(),
    )
    skill_profile = _mapping(
        skill_profile,
        path="embodiment component.skill_profile",
        required=frozenset(
            {"contract_id", "profile_id", "resources", "command_presets"}
        ),
        optional=frozenset({"runtime_services"}),
    )
    scene = _mapping(
        task["scene_binding"],
        path="task integration.scene_binding",
        required=frozenset({"contract_id", "registry_id"}),
        optional=frozenset(
            {"rigid_objects", "articulations", "links", "collision_world_mode"}
        ),
    )
    payload = _compose_integration_payload(
        task=task, policy=policy, skill_profile=skill_profile, scene=scene
    )
    raw_services = deepcopy(payload.get("runtime_services", {}))
    raw_lowerers = raw_services.pop("registered_semantic_lowerers", ())
    lowerers = tuple(
        decode_task_lowerer(
            value, path=f"runtime_services.registered_semantic_lowerers[{i}]"
        )
        for i, value in enumerate(
            _sequence(raw_lowerers, path="registered_semantic_lowerers")
        )
    )
    services = _decode_runtime_services(raw_services)
    call_catalog = builtin_semantic_call_catalog()
    for factory in lowerers:
        call_catalog = call_catalog.with_descriptor(
            SemanticCallDescriptor(
                call_id=factory.call_id,
                spec_type=RegisteredSemanticCall,
                target_descriptor=factory.target_descriptor,
            )
        )
    registration = SimulationTaskProgramRegistration(
        scene_binding=_decode_scene(payload["scene"]),
        robot_profile_binding=_decode_robot_profile(payload["robot_profile"]),
        call_catalog=call_catalog,
        handover_pose_providers=services.handover_pose_providers,
        control_part_evidence_factory=services.control_part_evidence,
        registered_semantic_lowerer_factories=lowerers,
    )
    fingerprint = canonical_hash(
        {
            "local_decoder_revision": 1,
            "registration": registration.fingerprint,
            "grasp_pose_generators": {
                key: asdict(value) for key, value in services.grasp_pose_generators
            },
        }
    )
    return TaskDeployment(
        integration_id=_identifier(task["integration_id"], path="integration_id"),
        program_id=_identifier(task["program_id"], path="program_id"),
        program_path=program_path,
        selection=TaskProgramIntegrationCfg(
            robot_profile=registration.robot_profile_binding.profile_id,
            scene_registry=registration.scene_binding.registry_id,
            runtime_preset=_identifier(
                policy["preset_id"], path="execution_policy.preset_id"
            ),
        ),
        integration=TaskIntegration(
            registration, fingerprint, services.grasp_pose_generators
        ),
        scene_binding=deepcopy(dict(scene)),
    )
