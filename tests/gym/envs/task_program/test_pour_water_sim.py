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

"""Measured regression coverage for bottle retention through PourWater."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import numpy as np
import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_DIRECTORY = (
    _REPOSITORY_ROOT
    / "embodichain/extensions/tasks/configs/tasks/manipulation/tableware/pour_water"
)
# Keep the bottle near its authored pour center while rotating by 60 degrees.
_MAX_POUR_POSITION_ERROR_M = 0.02
_MIN_POUR_TILT_DEGREES = 45.0


def _run_episode(variant: str, seed: int) -> dict[str, Any]:
    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import discover_task_packages, make
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.task_program.language.decoder import _decode_target
    from embodichain.lab.task_program.integrations.configured import _pose_matrix
    from embodichain.utils.utility import load_config

    discover_task_packages()
    source = _CONFIG_DIRECTORY / f"task.{variant}.yaml"
    payload = load_config(source)
    payload.update(
        num_envs=1,
        headless=True,
        renderer="no-render" if variant == "cobotmagic" else "hybrid",
        device="cuda:0" if variant == "cobotmagic" else "cpu",
        enable_sensor=False,
        seed=seed,
    )
    cfg = config_to_cfg(payload, source_path=source)
    cfg.dataset = None
    cfg.filter_dataset_saving = True
    cfg.filter_visual_rand = True
    cfg.init_rollout_buffer = False
    cfg.record_trajectory = False
    program = load_config(source.parent / payload["task_program"]["program"])
    target = None
    if variant != "cobotmagic":
        target = np.array(
            _decode_target(
                program["targets"]["bottle_pour_pose"],
                path=("targets", "bottle_pour_pose"),
            ).position
        )
    env = None
    try:
        env = make(payload["id"], cfg=cfg)
        env.reset(seed=seed, options={"save_data": False})
        if target is None:
            integration = load_config(
                source.parent / payload["task_program"]["integration"]
            )
            transport = next(
                lowerer
                for lowerer in integration["runtime_services"][
                    "registered_semantic_lowerers"
                ]
                if lowerer.get("target_id") == "cup_pour_pose"
            )
            relative = np.array(
                _pose_matrix(transport["relative_pose"], path="transport.relative_pose")
            ).reshape(4, 4)
            cup = (
                env.sim.get_rigid_object("cup")
                .get_local_pose(to_matrix=True)[0]
                .cpu()
                .numpy()
            )
            target = (cup @ relative)[:3, 3]
        poses = []
        original_step = env.step

        def tracked_step(*args: Any, **kwargs: Any) -> Any:
            result = original_step(*args, **kwargs)
            poses.append(
                env.sim.get_rigid_object("bottle")
                .get_local_pose(to_matrix=True)[0]
                .cpu()
                .numpy()
                .copy()
            )
            if variant == "cobotmagic" and len(poses) == round(1.0 / env.step_dt):
                # The seed-0 bottle is still upright and stationary at one second.
                # A particle outside its bounding cylinder has escaped before pouring.
                water = env.sim.get_fluid_object("water")
                points = (
                    (water.data.particle_pos_w[0] - env.sim.arena_offsets[0])
                    .cpu()
                    .numpy()
                )
                pose = poses[-1]
                local = (points - pose[:3, 3]) @ pose[:3, :3]
                inside = (
                    (np.linalg.norm(local[:, :2], axis=1) <= 0.0295)
                    & (local[:, 2] >= -0.0865)
                    & (local[:, 2] <= 0.0705)
                )
                assert (
                    inside.all()
                ), f"{np.count_nonzero(~inside)} particles outside upright bottle"
            return result

        env.step = tracked_step
        result = execute_demo_episode(env)
        episode = result.to_metadata()
        assert result.all_success, episode["terminal_reason"]
        segment = next(
            item for item in episode["segments"] if item["name"] == "pour_water"
        )
        pouring = np.stack(poses[segment["start_step"] : segment["end_step"]])
        if variant == "cobotmagic":
            volume = cfg.observations.received_volume
            received_m3 = volume.func(env, {}, **volume.params)
            assert received_m3[0, 0].item() >= 0.00005
            # Counterfactual terminal state: preserve robot, object poses and
            # completed program, relocate only the water outside the receiver.
            water = env.sim.get_fluid_object("water")
            spill = water.data.particle_state_w.clone()
            spill[..., 0] = env.sim.arena_offsets[:, None, 0] + 2.0
            water.write_particle_state_to_sim(spill)
            assert volume.func(env, {}, **volume.params)[0, 0].item() == 0.0
            assert not env.is_task_success()[0].item()

        errors = np.linalg.norm(pouring[:, :3, 3] - target, axis=-1)
        tilts = np.degrees(np.arccos(np.clip(pouring[:, 2, 2], -1.0, 1.0)))
        return {
            "completed": bool(result.completed),
            "success": bool(result.all_success),
            "steps": len(poses),
            "simulation_seconds": len(poses) * env.step_dt,
            "max_pour_position_error_m": float(errors.max()),
            "max_pour_tilt_degrees": float(tilts.max()),
            "terminal_reason": episode["terminal_reason"],
            "segments": [
                {
                    "name": item["name"],
                    "success": item["success"],
                    "failure_reason": item["failure_reason"],
                }
                for item in episode["segments"]
            ],
        }
    finally:
        if env is not None:
            env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def _run_child(variant: str, seed: int, destination: str) -> None:
    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    code = 1
    try:
        report = _run_episode(variant, seed)
        Path(destination).write_text(json.dumps(report), encoding="utf-8")
        code = 0
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    ("variant", "seed"),
    [
        ("cobotmagic", 0),
        ("dual_ur5_dh_pgi_140_80", 0),
        ("tianji_marvin", 0),
        ("dual_ur5_dh_pgi_140_80", 1),
        ("dual_ur5_dh_pgi_140_80", 2),
        ("tianji_marvin", 1),
    ],
)
def test_pour_keeps_bottle_grasped_tilts_and_returns(
    variant: str, seed: int, tmp_path: Path
) -> None:
    """Actual bottle motion, retention and all measured segment gates must pass."""
    output = tmp_path / "outcome.json"
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; runpy.run_path(sys.argv[1])['_run_child'](sys.argv[2],int(sys.argv[3]),sys.argv[4])",
            str(Path(__file__).resolve()),
            variant,
            str(seed),
            str(output),
        ],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=1800 if variant == "cobotmagic" else 180,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(output.read_text())
    assert report["completed"] and report["success"], report
    assert report["max_pour_position_error_m"] < _MAX_POUR_POSITION_ERROR_M, report
    assert report["max_pour_tilt_degrees"] > _MIN_POUR_TILT_DEGREES, report
