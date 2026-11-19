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

"""Measured drawer travel of the open drawer task."""

from __future__ import annotations

import pytest


def _run_ur5_open_drawer_episode() -> None:
    """Open the Default drawer once and measure how far its rail moved."""
    from pathlib import Path

    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import make
    from embodichain.utils.utility import load_config

    task = (
        Path(__file__).resolve().parents[4]
        / "extensions/embodichain_tasks/configs/tasks/manipulation/open_drawer"
    )
    source = task / "task.ur5.yaml"
    payload = load_config(source)
    payload.update(
        headless=True, device="cpu", renderer="no-render", enable_sensor=False, seed=0
    )
    cfg = config_to_cfg(payload, source_path=source)
    cfg.dataset = None
    cfg.filter_dataset_saving = True
    integration = load_config(task / "task_program/integration.yaml")
    slide = integration["profile"]["skill_options"][
        "simulation.articulation_link_slide"
    ]

    env = make(payload["id"], cfg=cfg)
    try:
        env.reset(seed=0, options={"save_data": False})
        result = execute_demo_episode(env)
        assert result.completed and result.all_success, result.to_metadata()
        # The program has no opening validator, so a stalled rail still
        # completes the skill. Require the configured pull within the skill's
        # 20 mm goal tolerance.
        opened = float(env.sim.get_articulation("drawer").get_qpos()[0, 0])
        assert opened >= slide["translation_distance"] - 0.02, opened
    finally:
        env.close(exit_process=False)


@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
@pytest.mark.slow
def test_ur5_open_drawer_pulls_the_rail_open() -> None:
    """Reject a completed episode whose drawer did not travel the pull."""
    from pathlib import Path
    import subprocess
    import sys

    path = Path(__file__).resolve()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])['_run_ur5_open_drawer_episode']()",
            str(path),
        ],
        cwd=path.parents[4],
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
