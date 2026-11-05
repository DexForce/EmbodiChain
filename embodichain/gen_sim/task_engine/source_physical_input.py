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

"""Compose prepared source scenes through independent physical declaration owners."""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Any
from embodichain.data.asset_dependencies import collect_declared_asset_files
from embodichain.utils.config_paths import resolve_config_path
from embodichain.utils.utility import load_config
from embodichain.lab.sim.scene.component_declarations import (
    source_scene_component_declaration,
)
from embodichain.lab.sim.scene.source_contact import source_grasp_contact_clearances
from embodichain.lab.sim.scene.source_envelopes import source_obstacle_bounds
from embodichain.lab.sim.scene.prismatic_declarations import (
    source_prismatic_physics_declarations,
)
from embodichain.lab.task_program.integrations.simulation.source_deployment import (
    source_task_program_environment_declaration,
)
from embodichain_robots.robotiq.deployment import (
    source_robotiq_opening_declaration,
)
from embodichain_robots.task_program import (
    TaskGripEffortCandidate,
    declare_task_source_execution_calibration,
    declare_source_task_embodiment,
    resolve_task_hand_calibrations,
    declare_task_robot_mount,
    declare_task_contact_pads,
)
from embodichain.toolkits.graspkit import resolve_parallel_jaw_gripper_model
from .asset_preparation import (
    fit_prepared_scene_assets,
    normalize_prepared_scene_assets,
)
from .orchestration.source_scene import PreparedScene
from .physical_input import FrozenPhysicalInput
from .contracts import canonical_hash
from .robot_profiles import canonical_robot_profile

__all__ = ["SourcePhysicalPreparation", "prepare_source_physical_input"]


@dataclass(frozen=True, slots=True)
class SourcePhysicalPreparation:
    """Fresh immutable physical input plus source adaptation receipts."""

    physical: FrozenPhysicalInput
    artifacts: Mapping[str, Any]


def _resource_effort_candidates(
    graph: Mapping[str, Any], scene: PreparedScene
) -> tuple[TaskGripEffortCandidate, ...]:
    # Reduce semantic relationships only; model calibration stays in robots.
    picks = []
    blocked = set()
    pour = set()
    press = set()
    carriers = {}
    for node in graph["nodes"]:
        call = node["call"]
        resources = set(call.get("resources", {}).values())
        if node["task_type"] == "E9":
            press.update(resources)
            continue
        if node["task_type"] not in {"E1", "E2", "E3", "E5"}:
            blocked.update(resources)
            continue
        if node["task_type"] == "E5":
            if call.get("call_id") in {
                "simulation.coordinated_hold",
                "simulation.coordinated_transport",
            }:
                for resource in resources:
                    carriers.setdefault(resource, set()).add(
                        call["arguments"]["object"]
                    )
            continue
        if call["kind"] == "pick":
            uid = call["object"]
        elif call["kind"] == "registered" and (
            call["call_id"] == "simulation.pick"
            or call["call_id"].startswith("gen_sim.pick.")
        ):
            uid = call["arguments"]["object"]
        else:
            continue
        resource = call.get("resources", {}).get("primary", "left")
        picks.append((uid, resource))
        if node["task_type"] == "E3":
            pour.add(resource)
    if not picks:
        return ()
    blocked.update(press - pour)
    objects = {value["uid"]: value for value in scene.rigid_objects}
    result = []
    for resource in {value for _, value in picks}:
        payloads = {uid for uid, owner in picks if owner == resource} | carriers.get(
            resource, set()
        )
        masses = [
            objects.get(uid, {}).get("attrs", {}).get("mass_props", {}).get("mass")
            for uid in payloads
        ]
        result.append(
            TaskGripEffortCandidate(
                resource_id=resource,
                eligible=resource not in blocked,
                pouring=resource in pour,
                payload_masses=tuple(masses),
            )
        )
    return tuple(result)


def prepare_source_physical_input(
    scene: PreparedScene,
    graph: Mapping[str, Any],
    *,
    robot_profile: str,
    cache_root: str | Path,
    runtime_profile: str = "default",
    fit_grasp_assets: bool = False,
    max_episodes: int | None = None,
    max_episode_steps: int | None = None,
    prismatic_uids: tuple[str, ...] = (),
    source_mass_owned_uids: tuple[str, ...] = (),
    twist_support_lift_m: float = 0.001,
    twist_grip_depth_m: float | None = None,
    twist_mass_source: bool = False,
    press_extra_distance: float | None = None,
) -> SourcePhysicalPreparation:
    """Prepare a normal source scene before pure bundle declaration/publication.

    Args:
        scene: Source PreparedScene, never a previously frozen snapshot.
        graph: Validated semantic declarations selecting acquisition/resource eligibility.
        robot_profile: Existing qualified source robot selector.
        cache_root: Separate derived asset/model stage, outside final bundle output.
        runtime_profile: Explicit resolved default/contact/composition policy family.
        fit_grasp_assets: Original optional uniform grasp-fit preparation.
        max_episodes: Requested positive episode limit or source default one.
        max_episode_steps: Requested control-step limit or source default 10000.
        prismatic_uids: Explicit native articulation owners requiring source calibration.
        source_mass_owned_uids: Owners retaining independent mass source calibration.
        twist_support_lift_m: Explicit original source-control support lift.
        twist_grip_depth_m: Optional measured source-control depth candidate.
        twist_mass_source: Explicit derived mass-source preparation selection.
        press_extra_distance: Original route overtravel provenance.

    Returns:
        Fresh physical snapshot and source receipts. Frozen input callers must
        bypass this function entirely. No Robot, simulator, IK, skill, model
        inference or bundle is constructed; derived assets are prepared separately.
        Control-specific geometric scale/mass preparation remains its own owner.
    """
    if type(scene) is not PreparedScene:
        raise TypeError("Source physical preparation requires a PreparedScene.")
    scene.assert_source_assets()
    if type(fit_grasp_assets) is not bool:
        raise TypeError("fit_grasp_assets must be a boolean.")
    if type(twist_mass_source) is not bool:
        raise ValueError("E8 deployment mass source must be an explicit boolean.")
    has_twist = any(value["task_type"] == "E8" for value in graph["nodes"])
    if twist_mass_source and not has_twist:
        raise ValueError("Deployment mass source requires a standalone E8 deployment.")
    if twist_grip_depth_m is not None and not has_twist:
        raise ValueError("Explicit knob depth candidates require an E8 deployment.")
    if press_extra_distance is not None and (
        type(press_extra_distance) not in (int, float)
        or not math.isfinite(press_extra_distance)
        or not 0.0 <= press_extra_distance <= 0.010
    ):
        raise ValueError("press_extra_distance must be between 0 and 0.010 m or None.")
    profiles = {
        "dual_franka": "dual_franka_robotiq_arg2f_140",
        "dual_franka_robotiq_arg2f_140": "dual_franka_robotiq_arg2f_140",
        "dual_ur10": "dual_ur10_robotiq_arg2f_140",
        "tianji_marvin": "tianji_marvin",
    }
    normalized = canonical_robot_profile(str(robot_profile).strip())
    if normalized not in profiles:
        raise ValueError("Source physical input requires a qualified robot profile.")
    if runtime_profile not in {
        "default",
        "contact_experiment",
        "composition_experiment",
    }:
        raise ValueError(
            "Source physical preparation requires a resolved runtime profile."
        )
    cache = Path(cache_root).expanduser().resolve()
    component = resolve_config_path(
        f"extensions/embodichain_tasks/configs/components/embodiments/{profiles[normalized]}.yaml"
    )
    policy_path = resolve_config_path(
        "extensions/embodichain_tasks/configs/components/execution_policies/dual_arm_trajectory_verified.yaml"
    )
    native_marvin = normalized == "tianji_marvin"
    selected_embodiment = load_config(component)
    embodiment = (
        declare_source_task_embodiment(selected_embodiment)
        if native_marvin
        else source_robotiq_opening_declaration(selected_embodiment)
    )
    if runtime_profile == "contact_experiment":
        embodiment = declare_task_contact_pads(embodiment)
    initial_files = {component.resolve(), policy_path.resolve()}
    if scene.source_config_path.is_file():
        initial_files.add(scene.source_config_path.resolve())
    for value in (
        dict(
            background=scene.background,
            rigid_object=scene.rigid_objects,
            articulation=scene.articulations,
        ),
        embodiment,
    ):
        initial_files.update(
            collect_declared_asset_files(value, base_dir=scene.scene_dir)
        )
    initial_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in initial_files
    }
    artifacts = {}
    if fit_grasp_assets:
        hands = resolve_task_hand_calibrations(embodiment)
        generators = embodiment["skill_profile"]["runtime_services"][
            "grasp_pose_generators"
        ]
        scene, adaptation = fit_prepared_scene_assets(
            scene,
            graph,
            openings={
                hand.resource_id: resolve_parallel_jaw_gripper_model(
                    generators[hand.grasp_target_id]["model"]
                ).max_opening_width
                for hand in hands
            },
            contact_clearances=source_grasp_contact_clearances(
                scene.rigid_objects, embodiment["simulation"]
            ),
        )
        artifacts["asset_adaptation.json"] = adaptation
    scene = normalize_prepared_scene_assets(scene, str(cache))
    if "asset_adaptation.json" in artifacts:
        runtime = {value["uid"]: value for value in scene.asset_provenance}
        for record in artifacts["asset_adaptation.json"]["records"]:
            derived = runtime.get(record["object_id"])
            if derived is not None and derived["status"] == "preserved_invalid_source":
                raise ValueError(
                    f"Adapted grasp asset {record['object_id']!r} could not be baked."
                )
            if derived is not None:
                record.update(
                    runtime_sha256=derived["runtime_sha256"],
                    runtime_path=derived["runtime_path"],
                )
    control_input_files = ()
    has_controls = any(value["task_type"] in {"E8", "E9"} for value in graph["nodes"])
    if has_controls:
        from .control_physical_preparation import prepare_source_control_physics

        controls = prepare_source_control_physics(
            scene,
            graph,
            embodiment,
            cache_root=cache,
            twist_support_lift_m=twist_support_lift_m,
            twist_grip_depth_m=twist_grip_depth_m,
            twist_mass_source=twist_mass_source,
            press_extra_distance=press_extra_distance,
        )
        scene = controls.scene
        artifacts.update(controls.artifacts)
        control_input_files = controls.input_files
        source_mass_owned_uids = tuple(
            dict.fromkeys((*source_mass_owned_uids, *controls.mass_owned_uids))
        )
    table_bounds = None
    if native_marvin:
        table = next(
            (value for value in scene.background if value["uid"] == "table"), None
        )
        if table is None:
            raise ValueError("Marvin floor deployment requires a measured scene table.")
        table_bounds = source_obstacle_bounds(dict(table))
    embodiment = declare_task_robot_mount(
        embodiment, table_top_z=scene.table_top_z, table_bounds=table_bounds
    )
    first = graph["nodes"][0]["call"]
    embodiment, posture = declare_task_source_execution_calibration(
        embodiment,
        _resource_effort_candidates(graph, scene),
        runtime_profile=runtime_profile,
        requires_prismatic_stiffness=bool(prismatic_uids),
        initial_prismatic_clearance=(
            first.get("call_id") == "gen_sim.articulation_slide"
        ),
        active_resource_id=first.get("resources", {}).get("primary"),
    )
    if posture is not None:
        artifacts["ur10_initial_posture_selection.json"] = posture
    component_scene = source_scene_component_declaration(
        "source_prepared_scene",
        background=scene.background,
        rigid_objects=scene.rigid_objects,
        articulations=scene.articulations,
    )
    if prismatic_uids:
        component_scene["simulation"]["articulation"] = list(
            source_prismatic_physics_declarations(
                component_scene["simulation"]["articulation"],
                bound_uids=prismatic_uids,
                source_mass_owned_uids=source_mass_owned_uids,
            )
        )
    policy = load_config(policy_path)
    policy["tracking"].update(consecutive_acceptances=5, terminal_settle_timeout=3.0)
    episode_steps = int(max_episode_steps or 10000)
    if (
        len(graph.get("task_groups", ())) == 1
        and graph["task_groups"][0]["task_type"] == "E8"
    ):
        episode_steps = min(episode_steps, 15000)
    deployment = source_task_program_environment_declaration(
        tuple(str(value["uid"]) for value in scene.rigid_objects),
        max_episodes=int(max_episodes or 1),
        max_episode_steps=episode_steps,
    )
    files = {component.resolve(), policy_path.resolve(), *control_input_files}
    if scene.source_config_path.is_file():
        files.add(scene.source_config_path.resolve())
    for source in (component_scene, embodiment):
        files.update(collect_declared_asset_files(source, base_dir=cache))
    for record in scene.asset_provenance:
        for key in ("source_path", "runtime_path"):
            if record.get(key):
                files.add(Path(record[key]).resolve())
    for name, expected in initial_hashes.items():
        if (
            not Path(name).is_file()
            or hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected
        ):
            raise ValueError(
                f"Source physical input changed during preparation: {name}"
            )
    files.update(Path(name) for name in initial_hashes)
    input_files = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }
    simulation = component_scene["simulation"]
    physical = FrozenPhysicalInput(
        scene, simulation, embodiment, policy, deployment, normalized, input_files, ""
    )
    plain = asdict(scene)
    plain["source_config_path"] = str(scene.source_config_path)
    plain["scene_dir"] = str(scene.scene_dir)
    physical = replace(
        physical,
        physical_digest=canonical_hash(
            dict(
                scene=plain,
                scene_simulation=simulation,
                embodiment=embodiment,
                execution_policy=policy,
                deployment=deployment,
                robot_profile=normalized,
                input_files=input_files,
            )
        ),
    )
    physical.assert_unchanged()
    return SourcePhysicalPreparation(physical, deepcopy(artifacts))
