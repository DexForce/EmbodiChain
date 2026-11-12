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

"""Prepare a native Pick → typed transport → measured hold deployment."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

from embodichain.utils.config_paths import resolve_config_path

__all__: list[str] = []


def prepare(output: Path, *, duration_s: float = 2.0) -> Path:
    """Write and statically validate an independently owned present example."""
    from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
    from embodichain.lab.task_program.integrations.composition import (
        load_configured_task_program_deployment,
    )

    source = resolve_config_path(
        "embodichain_tasks/configs/tasks/manipulation/carton_stocking/task.panda.yaml"
    ).resolve()
    base = source.parent
    task = yaml.safe_load(source.read_text())
    integration = yaml.safe_load(
        (base / task["task_program"]["integration"]).read_text()
    )
    policy = yaml.safe_load(
        (base / task["task_program"]["execution_policy"]).read_text()
    )
    template = yaml.safe_load((base / task["task_program"]["program"]).read_text())
    integration["program_id"] = "carton_present"
    integration["integration_id"] = "carton_present"
    services = integration["runtime_services"]
    services.pop("nominal_motion_sources", None)
    services.pop("registered_semantic_lowerers", None)
    profile = integration["profile"]
    profile["skill_options"].pop("simulation.move_held_object", None)
    profile["skill_options"]["move_held_object"] = {"kind": "move_held_object"}
    profile["effect_monitors"]["move_held_object"] = {
        "monitor_id": "builtin.composite_effect",
        "revision": "1",
        "params": {},
    }
    destination = deepcopy(template["targets"]["above_worktop"]["values"][0])
    program = {
        "program_id": "carton_present",
        "instruction": "Pick the carton, move it above the worktop and hold it.",
        "targets": {"presentation": {"kind": "pose", "frame": "world", **destination}},
        "program": [
            {"segment": "pick_carton", "steps": [{"pick": {"object": "carton"}}]},
            {
                "segment": "present_carton",
                "steps": [
                    {"move_held_object": {"object": "carton", "at": "presentation"}}
                ],
                "post": [
                    {
                        "hold": {
                            "object": "carton",
                            "at": "presentation",
                            "duration_s": duration_s,
                            "position_tolerance": 0.03,
                            "orientation_tolerance": 0.15,
                        }
                    }
                ],
            },
        ],
    }
    task.pop("real2sim", None)
    task.pop("collection", None)
    task["id"] = "CartonPresentPanda-v1"
    for name in ("environment", "embodiment"):
        task[name]["component"] = str((base / task[name]["component"]).resolve())
    task["task_program"] = {
        "program": "program.yaml",
        "integration": "integration.yaml",
        "execution_policy": "policy.yaml",
    }
    physical = _resolve_gym_components(task, base_dir=base)
    # Validate before creating any output, using the normal configured composition.
    deployment = load_configured_task_program_deployment(
        task_program={
            "program": {"inline": program},
            "integration": {"inline": integration},
            "execution_policy": {"inline": policy},
        },
        skill_profile=physical.embodiment_skill_profile,
        base_dir=base,
    )
    deployment.load_program()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    for filename, value in (
        ("task.yaml", task),
        ("program.yaml", program),
        ("integration.yaml", integration),
        ("policy.yaml", policy),
    ):
        (output / filename).write_text(yaml.safe_dump(value, sort_keys=False))
    return output / "task.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=2.0)
    args = parser.parse_args()
    print(prepare(args.output, duration_s=args.duration))


if __name__ == "__main__":
    main()
