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

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from embodichain.toolkits.scaffold.generator import generate_task
from embodichain.toolkits.scaffold.naming import default_env_class, default_gym_id
from embodichain.toolkits.scaffold.post_process import patch_tasks_init
from embodichain.toolkits.scaffold.render import render_task_py
from embodichain.toolkits.scaffold.spec import TaskSpec


def test_naming_defaults():
    assert default_env_class("pick_place") == "PickPlaceEnv"
    assert default_gym_id("pick_place") == "PickPlace-v1"


def test_invalid_snake_raises():
    with pytest.raises(ValueError, match="snake_case"):
        TaskSpec(task_snake="PickPlace", workflow="demo")


def test_render_demo_contains_register():
    spec = TaskSpec(
        task_snake="scaffold_test_demo",
        workflow="demo",
        target="inrepo",
        category="special",
        gym_id="ScaffoldTestDemo-v0",
        task_class="ScaffoldTestDemoEnv",
        include_test=False,
    )
    text = render_task_py(spec)
    assert '@register_env("ScaffoldTestDemo-v0"' in text
    assert "create_demo_action_list" in text


def test_dry_run_inrepo_paths(tmp_path: Path):
    spec = TaskSpec(
        task_snake="dry_run_task",
        workflow="config-only",
        target="inrepo",
        category="special",
        gym_id="DryRunTask-v0",
        dry_run=True,
        include_test=True,
        repo_root=tmp_path,
    )
    paths = generate_task(spec)
    assert not spec.task_py_path().exists()
    assert len(paths) >= 2


def test_generate_extension_project(tmp_path: Path):
    out = tmp_path / "my_ext"
    spec = TaskSpec(
        task_snake="ext_task",
        workflow="demo",
        target="extension",
        gym_id="ExtTask-v1",
        package_name="my_ext_pkg",
        project_name="my-ext",
        output_dir=out,
        include_test=True,
        run_black=False,
        repo_root=tmp_path,
    )
    written = generate_task(spec)
    assert (out / "pyproject.toml").is_file()
    assert (out / "my_ext_pkg" / "tasks" / "ext_task.py").is_file()
    assert (out / "configs" / "ext_task" / "gym.json").is_file()
    gym = json.loads((out / "configs" / "ext_task" / "gym.json").read_text())
    assert gym["id"] == "ExtTask-v1"
    init = (out / "my_ext_pkg" / "tasks" / "__init__.py").read_text()
    assert "from .ext_task import ExtTaskEnv" in init
    assert len(written) >= 5


def test_patch_tasks_init_idempotent(tmp_path: Path):
    spec = TaskSpec(
        task_snake="foo",
        workflow="demo",
        target="inrepo",
        category="special",
        gym_id="Foo-v1",
        task_class="FooEnv",
        repo_root=tmp_path,
    )
    init = spec.tasks_init_path()
    init.parent.mkdir(parents=True, exist_ok=True)
    init.write_text("__all__ = []\n", encoding="utf-8")
    patch_tasks_init(spec)
    patch_tasks_init(spec)
    content = init.read_text()
    assert "FooEnv" in content
    assert "from embodichain.lab.gym.envs.tasks.special.foo import FooEnv" in content
