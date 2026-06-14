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

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from embodichain.toolkits.scaffold.presets import gym_config_to_json
from embodichain.toolkits.scaffold.spec import TaskSpec

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=select_autoescape(enabled_extensions=()),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)


def _ctx(spec: TaskSpec) -> dict:
    return {
        "spec": spec,
        "task_snake": spec.task_snake,
        "task_class": spec.task_class,
        "gym_id": spec.gym_id,
        "description": spec.description,
        "max_episode_steps": spec.max_episode_steps,
        "package_name": spec.package_name or "",
        "project_name": spec.project_name or "",
        "gym_json": gym_config_to_json(spec),
    }


def render_task_py(spec: TaskSpec) -> str:
    if spec.workflow == "rl":
        template = "inrepo/task_rl.py.j2"
        if spec.target == "extension":
            template = "extension/task_rl.py.j2"
    elif spec.workflow == "config-only":
        template = "inrepo/task_config_only.py.j2"
        if spec.target == "extension":
            template = "extension/task_config_only.py.j2"
    else:
        template = "inrepo/task_demo.py.j2"
        if spec.target == "extension":
            template = "extension/task_demo.py.j2"
    return _env.get_template(template).render(**_ctx(spec))


def render_test_py(spec: TaskSpec) -> str:
    template = (
        "extension/test_task.py.j2"
        if spec.target == "extension"
        else "inrepo/test_task.py.j2"
    )
    return _env.get_template(template).render(**_ctx(spec))


def render_extension_file(name: str, spec: TaskSpec) -> str:
    return _env.get_template(f"extension/{name}.j2").render(**_ctx(spec))
