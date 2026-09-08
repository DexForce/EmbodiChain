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

"""Focused setuptools coverage for packaged Task Program resources."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import NamedTuple

import pytest
from setuptools import Distribution
from setuptools.command.build_py import build_py

from setup import get_package_dir

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SETUP_PATH = _REPOSITORY_ROOT / "setup.py"
_CONFIG_PACKAGE = "embodichain_tasks.configs"
_CONFIG_SOURCE = _REPOSITORY_ROOT / "embodichain_tasks" / "configs"
_PROGRAMS = {
    Path("tasks/manipulation/repeated_pick_place/task_program/program.yaml"): (
        "repeated_cube_pick_place"
    ),
    Path(
        "tasks/manipulation/open_drawer/task_program/program.yaml"
    ): "slide_open_drawer",
    Path(
        "tasks/manipulation/hand_over/task_program/program.yaml"
    ): "dual_ur5_hand_over",
    Path(
        "tasks/manipulation/tableware/pour_water/task_program/program.yaml"
    ): "pour_water_with_right_arm",
}
_DEPLOYMENTS = {
    Path("tasks/manipulation/repeated_pick_place/task.ur5.yaml"): (
        "repeated_cube_pick_place",
        "task_program_repeated_pick_place",
        "ur5_dh_pgi_140_80",
    ),
    Path("tasks/manipulation/repeated_pick_place/task.franka.yaml"): (
        "repeated_cube_pick_place",
        "task_program_repeated_pick_place",
        "franka_panda",
    ),
    Path("tasks/manipulation/open_drawer/task.ur5.yaml"): (
        "slide_open_drawer",
        "task_program_open_drawer",
        "ur5_dh_pgi_140_80",
    ),
    Path("tasks/manipulation/open_drawer/task.franka.yaml"): (
        "slide_open_drawer",
        "task_program_open_drawer",
        "franka_panda",
    ),
    Path("tasks/manipulation/hand_over/task.dual_ur5_dh_pgi_140_80.yaml"): (
        "dual_ur5_hand_over",
        "dual_ur5_handover_v1",
        "dual_ur5_dh_pgi_140_80",
    ),
    Path("tasks/manipulation/tableware/pour_water/task.cobotmagic.yaml"): (
        "pour_water_with_right_arm",
        "task_program_pour_water",
        "cobotmagic",
    ),
}
_RESOURCE_PATHS = frozenset(
    {
        *_PROGRAMS,
        *_DEPLOYMENTS,
        Path("components/execution_policies/motion_gen_verified.yaml"),
        Path("components/execution_policies/trajectory_open_loop.yaml"),
        Path("components/execution_policies/trajectory_open_loop_dense.yaml"),
        Path("components/embodiments/cobotmagic.yaml"),
        Path("components/embodiments/dual_ur5_dh_pgi_140_80.yaml"),
        Path("components/embodiments/franka_panda.yaml"),
        Path("components/embodiments/ur5_dh_pgi_140_80.yaml"),
        Path("tasks/manipulation/hand_over/env.yaml"),
        Path("tasks/manipulation/hand_over/task_program/integration.yaml"),
        Path("tasks/manipulation/open_drawer/env.yaml"),
        Path("tasks/manipulation/open_drawer/task_program/integration.yaml"),
        Path("tasks/manipulation/repeated_pick_place/env.yaml"),
        Path("tasks/manipulation/repeated_pick_place/task_program/integration.yaml"),
        Path("tasks/manipulation/tableware/pour_water/env.yaml"),
        Path("tasks/manipulation/tableware/pour_water/task_program/integration.yaml"),
    }
)


class _StagedConfigPackage(NamedTuple):
    """Isolated setuptools output and the setup options that produced it."""

    build_lib: Path
    relative_outputs: frozenset[Path]
    package_data: dict[str, list[str]]
    include_package_data: bool


def _literal_setup_keyword(keyword_name: str) -> object:
    """Read one literal keyword from the repository's setup() call."""
    tree = ast.parse(_SETUP_PATH.read_text(encoding="utf-8"), filename=str(_SETUP_PATH))
    setup_calls = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    if len(setup_calls) != 1:
        raise AssertionError("setup.py must contain exactly one setup() call.")
    keywords = {
        keyword.arg: keyword.value
        for keyword in setup_calls[0].keywords
        if keyword.arg is not None
    }
    if keyword_name not in keywords:
        raise AssertionError(f"setup.py does not declare {keyword_name!r}.")
    return ast.literal_eval(keywords[keyword_name])


@pytest.fixture
def staged_config_package(tmp_path: Path) -> _StagedConfigPackage:
    """Stage official Task Program resources through the real build command."""
    package_data = _literal_setup_keyword("package_data")
    include_package_data = _literal_setup_keyword("include_package_data")
    assert type(package_data) is dict
    assert type(include_package_data) is bool

    isolated_source = tmp_path / "source" / "embodichain_tasks" / "configs"
    isolated_source.mkdir(parents=True)
    shutil.copyfile(_CONFIG_SOURCE / "__init__.py", isolated_source / "__init__.py")
    for relative_path in _RESOURCE_PATHS:
        destination = isolated_source / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_CONFIG_SOURCE / relative_path, destination)

    build_lib = tmp_path / "build_lib"
    distribution = Distribution(
        {
            "packages": [_CONFIG_PACKAGE],
            "package_dir": {_CONFIG_PACKAGE: str(isolated_source)},
            "package_data": package_data,
            "include_package_data": include_package_data,
        }
    )
    distribution.script_name = str(_SETUP_PATH)
    command = build_py(distribution)
    command.build_lib = str(build_lib)
    command.ensure_finalized()

    def reject_manifest_command(command_name: str) -> None:
        raise AssertionError(
            f"Focused package-data staging must not run {command_name!r}."
        )

    command.run_command = reject_manifest_command
    relative_outputs = frozenset(
        Path(output).resolve().relative_to(build_lib.resolve())
        for output in command.get_outputs(include_bytecode=False)
    )
    command.run()
    return _StagedConfigPackage(
        build_lib=build_lib,
        relative_outputs=relative_outputs,
        package_data=package_data,
        include_package_data=include_package_data,
    )


def test_setup_stages_all_official_task_programs(
    staged_config_package: _StagedConfigPackage,
) -> None:
    """The actual setup patterns put nested Task Programs in wheel staging."""
    assert staged_config_package.include_package_data is False
    assert get_package_dir()[_CONFIG_PACKAGE] == "embodichain_tasks/configs"
    assert staged_config_package.package_data[_CONFIG_PACKAGE] == [
        "**/*.json",
        "**/*.yaml",
        "**/*.yml",
    ]
    expected_outputs = {
        Path("embodichain_tasks") / "configs" / relative_path
        for relative_path in _RESOURCE_PATHS
    }
    assert expected_outputs <= staged_config_package.relative_outputs


def test_staged_task_program_resources_decode_through_installed_paths(
    staged_config_package: _StagedConfigPackage,
    tmp_path: Path,
) -> None:
    """A clean process resolves and decodes all files from wheel staging."""
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    expected_deployments = {
        relative_path.as_posix(): identifiers
        for relative_path, identifiers in _DEPLOYMENTS.items()
    }
    script = """
import json
from pathlib import Path
import sys

import embodichain_tasks.configs as config_package
from embodichain.lab.task_program import load_task_program
from embodichain.lab.task_program.integrations._configured_composition import (
    _load_configured_task_program_deployment,
)
from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
from embodichain.utils.utility import load_config
from embodichain_tasks.configs import get_config_path

build_lib = Path(sys.argv[1]).resolve()
expected_deployments = json.loads(sys.argv[2])
module_path = Path(config_package.__file__).resolve()
assert module_path.is_relative_to(build_lib), (module_path, build_lib)
decoded_deployments = {}
for relative_path, expected_ids in expected_deployments.items():
    env_path = get_config_path(relative_path).resolve()
    assert env_path.is_relative_to(build_lib), (env_path, build_lib)
    config = load_config(env_path)
    physical = _resolve_gym_components(config, base_dir=env_path.parent)
    assert physical.embodiment_skill_profile is not None
    deployment = _load_configured_task_program_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=env_path.parent,
    )
    program = load_task_program(
        deployment.program_path,
        integration=deployment.selection,
        validation_context=deployment.integration.registration.catalog,
    )
    deployment.integration.registration.catalog.preflight(program)
    actual_ids = [
        program.program_id,
        deployment.integration.registration.scene_binding.registry_id,
        deployment.integration.registration.robot_profile_binding.profile_id,
    ]
    assert actual_ids == expected_ids
    decoded_deployments[relative_path] = actual_ids
print(json.dumps(decoded_deployments, sort_keys=True))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(staged_config_package.build_lib), str(_REPOSITORY_ROOT))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(staged_config_package.build_lib),
            json.dumps(expected_deployments, sort_keys=True),
        ],
        cwd=runtime_dir,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout.splitlines()[-1]) == {
        path: list(identifiers) for path, identifiers in expected_deployments.items()
    }
