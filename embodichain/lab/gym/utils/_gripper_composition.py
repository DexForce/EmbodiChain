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

"""Execution-free physical hand binding shared by embodiment composition."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

from embodichain.robot_registry import resolve_gripper_defaults
from embodichain.robot_registry import _validate_gripper_transform
from embodichain.toolkits.urdf_assembly.name_normalizer import NameNormalizer

__all__: list[str] = []


def _expand_grippers(
    simulation: dict, profile: dict | None, declarations: object
) -> None:
    """Expand explicitly selected hands without assets, Robot classes or backends."""
    if not isinstance(declarations, Mapping):
        raise TypeError("embodiment.grippers must be a mapping.")
    bindings = {}
    urdf = simulation.get("urdf_cfg", {})
    prefixes = dict(
        urdf.get(
            "component_prefix",
            (
                ("left_arm", "left_"),
                ("right_arm", "right_"),
                ("left_hand", "left_"),
                ("right_hand", "right_"),
            ),
        )
    )
    normalizer = NameNormalizer(
        urdf.get("name_case", {"joint": "original", "link": "original"})
    )
    controls = simulation.setdefault("control_parts", {})
    components = urdf.get("components", [])
    if isinstance(components, dict):
        components = [{"component_type": k, **v} for k, v in components.items()]
    else:
        components = deepcopy(components)
    for instance, raw in declarations.items():
        if type(instance) is not str or not instance or not isinstance(raw, Mapping):
            raise ValueError("Grippers need explicit instance IDs and mapping values.")
        allowed = {
            "class_type",
            "variant",
            "mode",
            "component_type",
            "motion_part",
            "control_part",
            "transform",
            "root_from_tcp",
            "root_from_contact_frame",
            "commands",
            "control_joint_names",
        }
        unknown = raw.keys() - allowed
        if (
            unknown
            or not {"class_type", "mode", "motion_part", "control_part"} <= raw.keys()
        ):
            raise ValueError(
                f"Invalid gripper binding {instance!r}; unsupported fields: {sorted(unknown)}"
            )
        defaults = resolve_gripper_defaults(
            raw["class_type"], variant=raw.get("variant")
        )
        mode = raw["mode"]
        if mode not in ("mounted", "existing"):
            raise ValueError("Gripper mode must be mounted or existing.")
        if mode == "existing" and "transform" in raw:
            raise ValueError(
                "Existing grippers cannot declare another mount transform."
            )
        component = raw.get("component_type")
        if mode == "mounted":
            if not component:
                raise ValueError("Mounted grippers require component_type.")
            if any(x["component_type"] == component for x in components):
                raise ValueError(
                    f"Gripper {instance!r} duplicates component {component!r}."
                )
            transform = _validate_gripper_transform(
                raw.get(
                    "transform",
                    ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
                )
            )
            components.append(
                {
                    "component_type": component,
                    "urdf_path": defaults["fpath"],
                    "transform": transform,
                    **(
                        {
                            "source_root_link": defaults["root_link"],
                            "mesh_paths": defaults.get("mesh_paths", {}),
                        }
                        if defaults.get("extract_subtree")
                        else {}
                    ),
                }
            )
        names = defaults["control_parts"]["eef"]
        selected = list(raw.get("control_joint_names", names))
        if (
            not selected
            or len(set(selected)) != len(selected)
            or not set(selected) <= set(names)
        ):
            raise ValueError(
                "control_joint_names must select distinct native hand joints."
            )
        prefix = prefixes.get(component)
        mapped = {n: normalizer.component_name("joint", n, prefix) for n in names}
        part = raw["control_part"]
        if part in controls:
            raise ValueError(
                f"Gripper {instance!r} owns control_parts.{part}; remove its duplicate."
            )
        controls[part] = [mapped[n] for n in selected]
        if mode == "mounted":
            import re

            drives = simulation.setdefault("joint_drive_props", {})
            for field, default in defaults["joint_drive_props"].items():
                if field == "drive_type":
                    drives.setdefault(field, default)
                    continue
                explicit = drives.get(field)
                if explicit is not None and not isinstance(explicit, Mapping):
                    continue
                values = drives.setdefault(field, {})
                for native, actual in mapped.items():
                    if not any(
                        pattern == part or re.fullmatch(pattern, actual)
                        for pattern in values
                    ):
                        if isinstance(default, Mapping):
                            matches = [
                                value
                                for pattern, value in default.items()
                                if pattern == "eef" or re.fullmatch(pattern, native)
                            ]
                            if len(matches) > 1:
                                raise ValueError(
                                    f"Ambiguous native gripper drive patterns for {native!r}."
                                )
                            if not matches:
                                continue
                            value = matches[0]
                        else:
                            value = default
                        values[actual] = deepcopy(value)
        if defaults.get("limit_overrides"):
            limits = simulation.setdefault("qpos_limits", {})
            if isinstance(limits, dict):
                for native, value in defaults["limit_overrides"].items():
                    limits.setdefault(mapped[native], deepcopy(value))
        for command, values in raw.get("commands", {}).items():
            if (
                not isinstance(values, Mapping)
                or set(values) != set(names)
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    for v in values.values()
                )
            ):
                raise ValueError(
                    "Each explicit command must cover all native hand joints with finite positions."
                )
            defaults["joint_commands"][command] = dict(values)
        for values in defaults["joint_commands"].values():
            for joint, value in values.items():
                low, high = defaults["qpos_limits"][joint]
                if not low <= value <= high:
                    raise ValueError(
                        f"Gripper command exceeds {joint!r} source limits."
                    )
        if "root_from_tcp" in raw:
            defaults["default_root_from_tcp"] = _validate_gripper_transform(
                raw["root_from_tcp"]
            )
        if "root_from_contact_frame" in raw:
            defaults["root_from_contact_frame"] = _validate_gripper_transform(
                raw["root_from_contact_frame"]
            )
        defaults["binding"] = {"instance": instance, **deepcopy(dict(raw))}
        bindings[instance] = {
            "declaration": deepcopy(dict(raw)),
            "defaults": defaults,
            "joint_names": mapped,
            "link_names": {
                n: normalizer.component_name("link", n, prefix)
                for n in defaults["collision_links"]
            },
            "control_joint_names": selected,
            "reference_link": (
                normalizer.component_name("link", defaults["reference_link"], prefix)
                if defaults.get("reference_link")
                else None
            ),
            "root_link": normalizer.component_name(
                "link", defaults["root_link"], prefix
            ),
        }
        explicit_tcp = (
            simulation.get("solver_cfg", {}).get(raw["motion_part"], {}).get("tcp")
        )
        if explicit_tcp is not None:
            bindings[instance]["explicit_solver_tcp"] = _validate_gripper_transform(
                explicit_tcp
            )
    if any(x["declaration"]["mode"] == "mounted" for x in bindings.values()):
        simulation.setdefault("urdf_cfg", {})["components"] = components
    simulation["gripper_bindings"] = bindings
    if profile is None:
        return
    presets = profile.setdefault("command_presets", [])
    services = profile.setdefault("runtime_services", {})
    generators = services.setdefault("grasp_pose_generators", {})
    for resource in profile.get("resources", []):
        for endpoint in resource.get("endpoints", []):
            instance = endpoint.pop("gripper", None)
            if instance is None:
                continue
            if instance not in bindings:
                raise ValueError(f"Unknown endpoint gripper {instance!r}.")
            binding = bindings[instance]
            part = binding["declaration"]["control_part"]
            if endpoint[
                "control_part"
            ] != part or "interaction.grasp" not in endpoint.get("capabilities", []):
                raise ValueError(
                    "Gripper references require an explicit matching grasp endpoint."
                )
            preset_id = endpoint.setdefault("command_preset", f"gripper.{instance}")
            if any(x["preset_id"] == preset_id for x in presets):
                raise ValueError(
                    f"Gripper {instance!r} duplicates explicit command preset {preset_id!r}."
                )
            defaults = binding["defaults"]
            presets.append(
                {
                    "preset_id": preset_id,
                    "control_part": part,
                    "commands": {
                        key: [value[n] for n in binding["control_joint_names"]]
                        for key, value in defaults["joint_commands"].items()
                    },
                }
            )
            policy = generators.pop(part, {})
            if set(policy) - {
                "kind",
                "sample_count",
                "approach_direction_samples",
                "approach_deviation_angle",
                "max_candidates",
                "center_mode",
                "point_spacing",
                "min_opening_width",
                "max_opening_width",
                "filter_ground_collision",
                "force_refresh",
            }:
                raise ValueError(
                    "A bound gripper owns generator physical data; only policy fields may be repeated."
                )
            supported = defaults["grasp_generators"]
            selected_kind = policy.get("kind")
            if selected_kind is None:
                if len(supported) != 1:
                    raise ValueError(
                        "A grasp endpoint must explicitly select one declared generator kind."
                    )
                selected_kind = supported[0]
            if selected_kind not in supported:
                raise ValueError(
                    f"Gripper {instance!r} does not declare generator {selected_kind!r}."
                )
            generators[part] = {
                "kind": selected_kind,
                **policy,
                "gripper": deepcopy(defaults),
            }
