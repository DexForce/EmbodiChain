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

"""Object-bound native Atomic Skill declarations and raw demonstration gaps."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from embodichain.data_pipeline.skill_fitting import file_digest

__all__: list[str] = []


def write_native_files(
    directory: Path,
    arrays: dict[str, np.ndarray],
    report: dict[str, Any],
    config: dict[str, Any],
    timeline: list[dict[str, Any]],
    collision_assets: dict[str, Any],
    step_dt: float,
) -> None:
    """Retain only required scene/model assets and demonstration tensors."""
    for span in timeline:
        if span["kind"] == "demo_skill":
            continue
        fit = report["fits"][span["fit_index"]]
        skill = span["skill_id"]
        goal = deepcopy(fit["native_goal"])
        if fit.get("scene") and skill == "pick_up":
            # Store a relative calibration, so a moved/settled object is grounded
            # from its current pose when this invocation actually starts.
            relative = np.linalg.inv(np.asarray(goal["object_pose"])) @ np.asarray(
                goal["grasp_xpos"]
            )
            goal = dict(
                kind="object_grasp",
                object_id=goal["object_id"],
                object_to_tcp=relative.tolist(),
            )
        else:
            goal = dict(
                kind={
                    "pick_up": "pickup_motion",
                    "slide": "slide_motion",
                    "twist": "twist_motion",
                    "hand_over": "handover_motion",
                }[skill],
                **goal,
            )
        factor = fit.get("replay_interpolation_factor", 1)
        span.update(
            execution="native_atomic_skill",
            goal=goal,
            control_parts=fit["selected_participants"],
            options=deepcopy(fit["native_options"]),
            motion_policy=deepcopy(fit["native_motion_policy"]),
            duration_scale=factor,
        )
    config["robot"]["solver_cfg"] = deepcopy(report["robot_profile"]["solvers"])
    for part, solver in config["robot"]["solver_cfg"].items():
        solver.pop("urdf_path", None)  # RobotCfg uses its packaged model path.
        solver["joint_names"] = report["robot_profile"]["control_parts"][part]
    config["max_episode_steps"] = 200_000
    config["env"][
        "task_instruction"
    ] = "Execute native atomic skills and recorded demonstration transitions."
    (directory / "task.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = dict(
        layout="DOF x n_frame",
        joint_names=report["joint_names"],
        step_dt=step_dt,
        timeline=timeline,
        scene_collision=collision_assets,
        schema="native-skill-sequence/v1",
        execution="native-skills-and-recorded-gaps",
        task_config="task.yaml",
        robot_profile="robot_profile.json",
        initial_qpos=arrays["original_qpos"][0].tolist(),
        connection=dict(kind="quintic_joint", minimum_duration_s=0.25),
        physical_success="unknown",
    )
    manifest["files"] = {
        p.relative_to(directory).as_posix(): file_digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name not in {"skills.yaml", "README.md"}
    }
    (directory / "skills.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
    )
    from ._sequence_expansion import write_sequence_expansion_profile

    # Expansion settings are editable and intentionally outside the sealed
    # physical-input inventory. A changed profile does not change the recording.
    write_sequence_expansion_profile(directory, timeline)
    (directory / "README.md").write_text(
        "# Native atomic skills + recorded transitions\n\n"
        "```bash\nconda activate py312_bk\n"
        "embodichain run-task --gym_config task.yaml --skill-sequence skills.yaml --headless --viser\n```\n\n"
        "Open the printed Viser URL. Commands execute automatically with physics enabled. "
        "For the native desktop window, omit `--viser` and use `--no-headless`.\n\n"
        "```bash\nembodichain run-task --gym_config task.yaml --skill-sequence skills.yaml --no-headless\n```\n\n"
        "For fixed-scene spatial variants of native atomic spans, run:\n\n"
        "```bash\nembodichain run-task --gym_config task.yaml --skill-sequence skills.yaml "
        "--expansion-profile expansion.yaml --expansion-output-dir ./expanded --headless --device cpu\n```\n\n"
        "The supplied profile runs nominal plus two variants when native spans exist; Demo-only sequences "
        "run nominal. DemoSkill source and command paths stay "
        "unchanged; connectors may depend on the preceding native posture. The expansion host uses one "
        "environment, records actual commands and T+1 measured states to chunked HDF5, and writes "
        "expansion_manifest.json with generation and execution results. Choose a new output directory "
        "for each run. Scene/affordance/timing/visual expansion and LeRobot export are not enabled here.\n\n"
        "Atomic spans contain only typed goals, object-relative grasp calibration, selected control parts, "
        "native options and motion policy. AtomicSkillEngine replans them from the live robot/object state; "
        "there is no atomic-skill trajectory PT or full-episode replay PT. "
        "Only `skills/*execute_demo_skill.pt` retain original full-DOF qpos matrices (DOF x n_frame) "
        "and source timestamps. These gaps are plain joint demonstrations, not fabricated monitored "
        "ExecuteDemoSkill contracts. Quintic joint connections enforce continuous command boundaries "
        "and configured velocity/acceleration limits. Runtime-generated motion is held to the control clock; "
        "duration_scale preserves the pickup slow-down setting.\n\n"
        "Gripper open/full-close targets are `robot_profile.json` fields "
        "`hand_commands.<grasp_part>.open/grasp`, in that part's declared joint order and native units. "
        "Native spans update only their explicitly bound parts. Other joints retain their previous "
        "command targets, so a loaded gripper keeps its closing preload while the opposite arm acts. "
        "Demo spans still execute their original commands, including authored release events. "
        "TCP calibration is `solvers.<motion_part>.tcp` (end-link-from-TCP, metres); export copies it "
        "to `task.yaml` under `robot.solver_cfg.<motion_part>.tcp`. Refit/export after changing the "
        "source profile so calibration and sealed resource hashes remain consistent.\n\n"
        "Meshes, colliders, robot model and configuration are local. Convex parts are loaded directly. "
        "Runtime writes `execution.json` with command completion and observed object heights; "
        "command completion alone does not certify physical task success. "
        "Use the separate fitting-input bundle to refit or change annotations.\n"
    )
