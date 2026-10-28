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

"""Replay saved synthetic prototypes through the verified Task Program.

The explicit diagnostic option permits unverified collision checks and never
labels such a run as dataset-qualified. This script does not commit a dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium
import numpy as np

from embodichain.data_pipeline.real2sim import load_prototype
from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
from embodichain.lab.scripts.run_task import (
    _create_parser,
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import PrototypeMotionProvider
from embodichain.lab.task_program.integrations.nominal import (
    BoundTaskProgramNominalSources,
)
from embodichain.lab.task_program.integrations.expansion import (
    TaskProgramMappedRetimeFactory,
)

__all__: list[str] = []

_TASK = Path(__file__).with_name("task.ur5.yaml")


class _ReplayFactory:
    """Select saved source units by compiled call occurrence."""

    def __init__(
        self, program, prototypes, *, diagnostic: bool, retarget_pick: bool = False
    ) -> None:
        self._bound = BoundTaskProgramNominalSources(program, {})
        self._prototypes = prototypes
        self._diagnostic = diagnostic
        self._retarget_pick = retarget_pick

    @property
    def program_fingerprint(self) -> str:
        return self._bound.program_fingerprint

    def validate_program(self, program) -> None:
        self._bound.validate_program(program)

    def create_nominal_source(self, request, *, engine):
        index = request.analysis_call_index
        if index not in self._prototypes:
            raise ValueError(f"synthetic reference has no call {index}")
        authority = "qpos" if index == 0 and not self._retarget_pick else "eef_pose"
        return PrototypeMotionProvider(
            self._prototypes[index],
            object_entity_id="cube",
            target_entity_id="platform" if index else None,
            authority=authority,
            gripper_mode="semantic",
            motion_generator=(
                engine.motion_generator
                if authority != "qpos" or not self._diagnostic
                else None
            ),
            require_collision_validation=not self._diagnostic,
            pose_tolerance=0.002,
        )


def main() -> None:
    """Run one saved-source episode and export its actual state and commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diagnostic-no-collision-check", action="store_true")
    parser.add_argument("--diagnostic-duration-scale", type=float)
    options = parser.parse_args()
    if (
        options.diagnostic_duration_scale is not None
        and not options.diagnostic_no_collision_check
    ):
        raise ValueError(
            "diagnostic retiming requires the explicit collision diagnostic flag"
        )
    reference = options.reference_dir.resolve()
    source_report = json.loads((reference / "report.json").read_text(encoding="utf-8"))
    if (
        source_report.get("origin") != "synthetic_reference"
        or not source_report.get("baseline_verified")
        or not source_report.get("source_artifacts_complete")
    ):
        raise ValueError(
            "reference must come from a completed verified synthetic capture"
        )
    prototypes = {
        index: load_prototype(reference / "prototypes" / str(index) / "prototype.json")
        for index in range(3)
    }
    output = options.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    args = _create_parser().parse_args(
        [
            "--gym-config",
            str(_TASK),
            "--device",
            options.device,
            "--headless",
            "--renderer",
            "no-render",
            "--disable_sensor",
            "--filter_dataset_saving",
            "--seed",
            str(options.seed),
        ]
    )
    discover_task_packages()
    execute_init_hooks()
    cfg, gym_config, action_config = build_env_cfg_from_args(args)
    cfg.record_trajectory = True
    cfg.trajectory_auto_save = False
    env = gymnasium.make(id=gym_config["id"], cfg=cfg, **action_config)
    try:
        env.reset(seed=options.seed, options={"save_data": False})
        target = env.unwrapped
        program = target.compile_task_program(target.cfg.task_program)
        factory = _ReplayFactory(
            program, prototypes, diagnostic=options.diagnostic_no_collision_check
        )
        if factory.program_fingerprint != source_report["program_fingerprint"]:
            raise ValueError("saved references belong to a different compiled program")
        retimer = None
        if options.diagnostic_duration_scale is not None:
            velocity = target.robot.get_qvel_limits()[0].detach().cpu().tolist()
            # The arm is additionally capped by its motion policy. This
            # permissive gripper assumption is diagnostic, not qualification.
            retimer = TaskProgramMappedRetimeFactory(
                duration_scale=options.diagnostic_duration_scale,
                velocity_limits=velocity,
                acceleration_limits=[50.0] * len(velocity),
                call_indices=(0, 1, 2),
                diagnostic_unchecked=True,
            )
        result = execute_demo_episode(
            env,
            task_program=program,
            nominal_source_factory=factory,
            plan_transform_factory=retimer,
        )
        report = {
            "origin": "synthetic_reference",
            "task_verified": bool(result.completed and result.all_success),
            "collision_check": (
                "unverified" if options.diagnostic_no_collision_check else "sampled"
            ),
            "dataset_committed": False,
            "terminal_reason": result.terminal_reason,
            "length": result.length,
            "program_fingerprint": factory.program_fingerprint,
            "segments": result.to_metadata()["segments"],
            "diagnostic_duration_scale": options.diagnostic_duration_scale,
            "retimed_call_indices": (
                []
                if retimer is None
                else [record.call_index for record in retimer.records]
            ),
        }
        (output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        if retimer is not None:
            arrays = {}
            for record in retimer.records:
                prefix = f"call_{record.call_index}"
                mapped = record.result.time_map
                normalized = prototypes[record.call_index].unit.report.time_map
                arrays[f"{prefix}_source_sample_offset"] = np.array(
                    record.source_sample_offset, dtype=np.int64
                )
                for name in (
                    "left",
                    "right",
                    "weight",
                    "source_event_indices",
                    "execution_event_indices",
                ):
                    arrays[f"{prefix}_retime_{name}"] = getattr(mapped, name)
                    arrays[f"{prefix}_normalization_{name}"] = getattr(normalized, name)
            np.savez_compressed(output / "plan_time_maps.npz", **arrays)
        count = int(target._traj_steps[0].item())
        buffer = target._traj_buffer
        if buffer is not None and 0 < count < buffer.shape[1]:
            states = buffer["states"]
            np.savez_compressed(
                output / "actual_rollout.npz",
                qpos=states["robot", "qpos"][0, : count + 1].cpu().numpy(),
                qvel=states["robot", "qvel"][0, : count + 1].cpu().numpy(),
                object_pose=states["rigid_objects", "cube", "pose"][0, : count + 1]
                .cpu()
                .numpy(),
                actions=buffer["actions"][0, :count].cpu().numpy(),
                timestamps=np.arange(count + 1) * target.step_dt,
            )
        if not report["task_verified"]:
            raise RuntimeError("saved-source replay failed a verified task check")
        env.reset(options={"save_data": False})
    finally:
        env.unwrapped.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
