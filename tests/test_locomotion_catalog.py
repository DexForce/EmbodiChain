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

"""Locomotion catalog deployment and PPO associations."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from embodichain.cli import _task_catalog as catalog
from embodichain.cli.show_task import main as show_task

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "embodichain_tasks/configs/tasks"
_TASK_PATHS = (
    "locomotion/velocity/g1_flat",
    "locomotion/velocity/h1_2_flat",
    "locomotion/velocity/go1_flat",
    "locomotion/velocity/go2_flat",
    "locomotion/velocity/anymal_c_flat",
    "locomotion/velocity/microduck_flat",
    "classic_control/humanoid",
)


@pytest.fixture(scope="module")
def locomotion_catalog() -> list:
    return catalog._load_catalog({"embodichain_tasks": _CONFIG_ROOT})


@pytest.mark.parametrize("task_path", _TASK_PATHS)
def test_locomotion_deployments_resolve_matching_ppo(
    locomotion_catalog: list, task_path: str
) -> None:
    task = catalog._select_task(locomotion_catalog, Path(task_path).name)
    assert task.default_deployment == "default"
    assert task.readme == _CONFIG_ROOT / task_path / "README.md"
    assert task.readme.is_file()
    assert {deployment.name for deployment in task.deployments} == {"default", "newton"}
    for deployment in task.deployments:
        suffix = ".newton" if deployment.name == "newton" else ""
        prefix = f"embodichain_tasks/configs/tasks/{task_path}"
        env_path = _CONFIG_ROOT / task_path / f"env{suffix}.yaml"
        env = yaml.safe_load(env_path.read_text())
        agent = yaml.safe_load(
            (_CONFIG_ROOT / task_path / f"agents/ppo{suffix}.yaml").read_text()
        )
        assert deployment.resource == env_path
        assert deployment.config_ref == agent["trainer"]["gym_config"]
        assert deployment.agent_refs == (f"{prefix}/agents/ppo{suffix}.yaml",)
        assert deployment.env_id == env["id"]
        assert deployment.physics == deployment.name == env["physics"]
        assert deployment.capabilities == {catalog._RL}
        assert deployment.validation is None


def test_locomotion_details_and_gallery(
    locomotion_catalog: list, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = [
        catalog._select_task(locomotion_catalog, Path(path).name)
        for path in _TASK_PATHS
    ]
    gallery = catalog._render_html(tasks, static_only=True)
    for task in tasks:
        show_task(
            [task.qualified_key, "--config-root", f"embodichain_tasks={_CONFIG_ROOT}"]
        )
        detail = capsys.readouterr().out
        assert task.title in detail and task.title in gallery
        assert detail.count("Supported uses: RL") == 2
        assert detail.count("Qualification: unavailable") == 2
        assert task.readme.as_uri() in gallery
        for deployment in task.deployments:
            assert deployment.agent_refs[0] in detail
            assert deployment.agent_refs[0] in gallery
            assert deployment.resource.as_uri() in gallery
