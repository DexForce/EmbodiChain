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

"""Execute a preflighted deployment through the ordinary Gym demo owner."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__: list[str] = []


def _execute_deployment(
    path: Path,
    *,
    device: str | None,
    headless: bool,
    reset_observer: Callable[[Any], None] | None = None,
    step_observer: Callable[[Any, Any, Any], None] | None = None,
) -> dict[str, Any]:
    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import (
        discover_task_packages,
        execute_init_hooks,
        get_env_spec,
    )
    from embodichain.lab.sim import SimulationManager
    from embodichain.utils.utility import load_config

    discover_task_packages()
    execute_init_hooks()
    data = load_config(path)
    cfg = config_to_cfg(data, source_path=path)
    if device is not None:
        cfg.sim_cfg.device = device
    cfg.sim_cfg.headless = headless
    env = None
    try:
        env = get_env_spec(data["id"]).make(cfg=cfg)
        reset_result = env.reset(options={"save_data": False})
        if reset_observer is not None:
            reset_observer(reset_result[0])
        kwargs = {"episode_index": 0}
        if step_observer is not None:
            kwargs["step_observer"] = step_observer
        result = execute_demo_episode(env, **kwargs)
        report = result.to_metadata()
        report.update(
            status="succeeded" if result.completed and result.all_success else "failed",
            all_success=result.all_success,
            effect_assurance="verified",
        )
        env.reset(options={"save_data": result.completed and result.all_success})
        return report
    finally:
        try:
            if env is not None:
                env.close(exit_process=False)
        finally:
            SimulationManager.flush_cleanup_queue()
