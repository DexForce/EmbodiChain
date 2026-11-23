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
import re
import shutil
import subprocess
import sys
from typing import NamedTuple

import pytest
from setuptools import Distribution
from setuptools.command.build_py import build_py

from setup import get_package_data, get_packages

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SETUP_PATH = _REPOSITORY_ROOT / "setup.py"
_PROGRAMS = (
    Path("tasks/manipulation/drawer_storage/task_program/program.long_horizon.yaml"),
    *(
        Path(
            f"tasks/manipulation/gaussian_pick_place/task_program/program.{scene}.yaml"
        )
        for scene in ("office", "discoverse", "robosimgs")
    ),
    Path("tasks/manipulation/carton_stocking/task_program/program.yaml"),
    Path("tasks/manipulation/hand_over/task_program/program.bar.yaml"),
    Path("tasks/manipulation/repeated_pick_place/task_program/program.tabletop.yaml"),
    Path("tasks/manipulation/repeated_pick_place/task_program/program.workbench.yaml"),
    Path("tasks/manipulation/hand_over/task_program/program.tabletop.yaml"),
    Path("tasks/manipulation/tableware/stack_blocks_two/task_program/program.yaml"),
    Path("tasks/manipulation/tableware/blocks_ranking_rgb/task_program/program.yaml"),
    Path("tasks/manipulation/tableware/pour_water/task_program/program.workbench.yaml"),
    Path("tasks/manipulation/rubiks_cube_pick_place/task_program/program.yaml"),
    Path("tasks/manipulation/repeated_pick_place/task_program/program.yaml"),
    Path("tasks/manipulation/open_drawer/task_program/program.yaml"),
    Path("tasks/manipulation/hand_over/task_program/program.yaml"),
    Path("tasks/manipulation/tableware/pour_water/task_program/program.yaml"),
    Path(
        "tasks/manipulation/tableware/pour_water/task_program/program.tianji_marvin.yaml"
    ),
)
_DEPLOYMENTS = {
    Path("tasks/manipulation/cable/franka_rj45_cycle/task.franka.yaml"): (
        "franka_rj45_cycle",
        "rj45_cycle",
        "franka_rj45",
    ),
    Path("tasks/manipulation/drawer_storage/task.ur5.long_horizon.yaml"): (
        "drawer_storage_long_horizon",
        "task_program_drawer_storage_long_horizon",
        "ur5_dh_pgi_140_80",
    ),
    **{
        Path(f"tasks/manipulation/gaussian_pick_place/task.{scene}.yaml"): (
            "franka_gs_pick_place",
            "franka_gs_tabletop",
            "franka_panda",
        )
        for scene in ("office", "discoverse", "robosimgs")
    },
    Path("tasks/manipulation/repeated_pick_place/task.tianji_marvin.yaml"): (
        "repeated_tabletop_cube_pick_place",
        "repeated_pick_place_tabletop",
        "tianji_marvin",
    ),
    Path("tasks/manipulation/hand_over/task.tianji_marvin.yaml"): (
        "tabletop_hand_over",
        "tabletop_handover",
        "tianji_marvin",
    ),
    Path("tasks/manipulation/tableware/pour_water/task.tianji_marvin.yaml"): (
        "pour_water_with_right_arm",
        "pour_water_tianji_marvin",
        "tianji_marvin",
    ),
    Path("tasks/manipulation/tableware/stack_blocks_two/task.cobotmagic.yaml"): (
        "stack_two_blocks",
        "task_program_stack_blocks_two",
        "cobotmagic",
    ),
    Path("tasks/manipulation/tableware/blocks_ranking_rgb/task.cobotmagic.yaml"): (
        "rank_blocks_rgb",
        "task_program_blocks_ranking_rgb",
        "cobotmagic",
    ),
    Path("tasks/manipulation/rubiks_cube_pick_place/task.ur5.yaml"): (
        "rubiks_cube_pick_place",
        "task_program_rubiks_cube_pick_place",
        "ur5_dh_pgi_140_80",
    ),
    Path("tasks/manipulation/repeated_pick_place/task.ur5.yaml"): (
        "repeated_cube_pick_place",
        "task_program_repeated_pick_place",
        "ur5_dh_pgi_140_80",
    ),
    Path("tasks/manipulation/repeated_pick_place/task.ur5.objective.yaml"): (
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
    Path("tasks/manipulation/tableware/pour_cup/task.franka.fluid.yaml"): (
        "pick_aim_pour_return_cup",
        "task_program_pour_cup_fluid",
        "franka_panda_fast_fluid",
    ),
}
_DEPLOYMENTS[
    Path("tasks/manipulation/hand_over/task.dual_franka_robotiq_arg2f_140.yaml")
] = (
    "upright_can_hand_over",
    "dual_franka_robotiq140_handover",
    "dual_franka_robotiq_arg2f_140_handover",
)
for _variant in (
    "dual_franka_robotiq_arg2f_140",
    "dual_ur10_robotiq_arg2f_140",
    "dual_franka_robotiq_arg2f_140_gensim",
    "dual_ur10_robotiq_arg2f_140_gensim",
):
    _DEPLOYMENTS[
        Path(f"tasks/manipulation/repeated_pick_place/task.{_variant}.yaml")
    ] = ("repeated_cube_pick_place_workbench", "robotiq140_tabletop", _variant)
_DEPLOYMENTS[Path("tasks/manipulation/carton_stocking/task.panda.rgbd.yaml")] = (
    "real2sim_carton_stocking",
    "carton_worktop",
    "franka_panda_robotiq85",
)
_DUAL_ARM_VARIANTS = (
    "dual_ur5_dh_pgi_140_80",
    "tianji_marvin",
    "dexforce_w1_030_pika",
)
_DUAL_ARM_PROGRAMS = {
    "hand_over": ("tabletop_hand_over", "tabletop_handover"),
    "repeated_pick_place": (
        "repeated_tabletop_cube_pick_place",
        "repeated_pick_place_tabletop",
    ),
    "tableware/pour_water": ("pour_water_with_right_arm", "pour_water_tianji_marvin"),
    "tableware/stack_blocks_two": ("stack_two_blocks", "task_program_stack_blocks_two"),
    "tableware/blocks_ranking_rgb": (
        "rank_blocks_rgb",
        "task_program_blocks_ranking_rgb",
    ),
}
for _task, (_program, _scene) in _DUAL_ARM_PROGRAMS.items():
    for _variant in _DUAL_ARM_VARIANTS:
        _DEPLOYMENTS.setdefault(
            Path(f"tasks/manipulation/{_task}/task.{_variant}.yaml"),
            (_program, _scene, _variant),
        )
_DEPLOYMENTS[Path("tasks/manipulation/hand_over/task.dual_ur5.yaml")] = (
    "tabletop_hand_over",
    "tabletop_handover",
    "dual_ur5_dh_pgi_140_80",
)
_DEPLOYMENTS[Path("tasks/manipulation/hand_over/task.dexforce_w1_030_pika.yaml")] = (
    "tabletop_hand_over",
    "tabletop_bar_handover",
    "dexforce_w1_030_pika",
)
_DUAL_ARM_RESOURCES = {
    Path("components/embodiments/dexforce_w1_030_pika.yaml"),
    *(
        Path(f"tasks/manipulation/{task}/env.dexforce_w1_030_pika.yaml")
        for task in (
            *_DUAL_ARM_PROGRAMS,
            "tableware/blocks_ranking_size",
            "tableware/match_object_container",
            "tableware/place_object_drawer",
            "tableware/stack_cups",
        )
    ),
    *(
        Path(
            f"tasks/manipulation/{task}/task_program/integration.dexforce_w1_030_pika.yaml"
        )
        for task in _DUAL_ARM_PROGRAMS
        if task
        not in {
            "tableware/blocks_ranking_rgb",
            "tableware/stack_blocks_two",
            "repeated_pick_place",
        }
    ),
    Path("components/execution_policies/motion_gen_projected_manipulation.yaml"),
    Path(
        "components/execution_policies/trajectory_open_loop_dual_arm_manipulation.yaml"
    ),
    *(
        Path(f"tasks/manipulation/tableware/{task}/{filename}")
        for task in (
            "blocks_ranking_size",
            "match_object_container",
            "place_object_drawer",
            "stack_cups",
        )
        for filename in (
            "env.json",
            "env.yaml",
            *(f"task.{variant}.yaml" for variant in _DUAL_ARM_VARIANTS),
        )
    ),
}

_RESOURCE_PATHS = frozenset(
    {
        *(
            Path(f"tasks/manipulation/gaussian_pick_place/env.{scene}.yaml")
            for scene in ("office", "discoverse", "robosimgs")
        ),
        Path("components/embodiments/franka_gs_dual_camera.yaml"),
        Path("components/execution_policies/franka_gs_verified.yaml"),
        Path("tasks/manipulation/gaussian_pick_place/task_program/integration.yaml"),
        Path("tasks/manipulation/gaussian_pick_place/catalog.yaml"),
        Path("tasks/manipulation/gaussian_pick_place/README.md"),
        Path("tasks/manipulation/cable/route_the_cable/cable.usda"),
        Path("components/embodiments/franka_flat_pads.yaml"),
        *(
            Path("tasks/manipulation/cable/franka_rope_wrap") / filename
            for filename in (
                "README.md",
                "cable.usda",
                "catalog.yaml",
                "config.json",
                "env.yaml",
                "task.franka.yaml",
                "task_program/evaluation.yaml",
                "task_program/integration.yaml",
                "task_program/program.yaml",
            )
        ),
        Path("components/embodiments/franka_rj45.yaml"),
        *(
            Path("tasks/manipulation/cable/franka_rj45_cycle") / filename
            for filename in (
                "catalog.yaml",
                "config.json",
                "env.yaml",
                "task.franka.yaml",
                "task_program/evaluation.yaml",
                "task_program/execution_policy.yaml",
                "task_program/integration.yaml",
                "task_program/program.yaml",
            )
        ),
        Path("tasks/manipulation/tableware/pour_cup/task.franka.fluid.yaml"),
        Path("tasks/manipulation/tableware/pour_cup/env.fluid.yaml"),
        Path("tasks/manipulation/tableware/pour_cup/task_program/program.fluid.yaml"),
        Path(
            "tasks/manipulation/tableware/pour_cup/task_program/integration.fluid.yaml"
        ),
        Path("components/embodiments/franka_panda_fast_fluid.yaml"),
        Path("components/execution_policies/trajectory_open_loop_fluid_pour.yaml"),
        Path("components/embodiments/franka_panda_robotiq85_rgbd.yaml"),
        Path(
            "tasks/manipulation/carton_stocking/task_program/integration.ordinary.yaml"
        ),
        Path("tasks/manipulation/repeated_pick_place/README.md"),
        Path("tasks/manipulation/hand_over/README.md"),
        Path("components/embodiments/dual_franka_robotiq_arg2f_140_handover.yaml"),
        Path("components/embodiments/dual_franka_robotiq_arg2f_140.yaml"),
        Path("components/embodiments/dual_ur10_robotiq_arg2f_140.yaml"),
        Path("components/embodiments/dual_franka_robotiq_arg2f_140_gensim.yaml"),
        Path("components/embodiments/dual_ur10_robotiq_arg2f_140_gensim.yaml"),
        Path(
            "tasks/manipulation/repeated_pick_place/envs/dual_franka_robotiq140.default.yaml"
        ),
        Path(
            "tasks/manipulation/repeated_pick_place/envs/dual_ur10_robotiq140.default.yaml"
        ),
        Path(
            "tasks/manipulation/repeated_pick_place/task_program/integration.robotiq140.yaml"
        ),
        Path(
            "tasks/manipulation/repeated_pick_place/task_program/program.robotiq_workbench.yaml"
        ),
        Path("components/execution_policies/sequential_ik_verified.yaml"),
        Path("components/execution_policies/sequential_ik_projected_manipulation.yaml"),
        Path("components/embodiments/franka_panda_cartesian.yaml"),
        Path("tasks/manipulation/hand_over/envs/dual_franka_robotiq140.default.yaml"),
        Path("tasks/manipulation/hand_over/task_program/integration.robotiq140.yaml"),
        Path("tasks/manipulation/hand_over/task_program/program.upright_can.yaml"),
        *(
            Path("tasks/manipulation/carton_stocking") / name
            for name in (
                "README.md",
                "catalog.yaml",
                "env.yaml",
                "task.panda.yaml",
                "task_program/integration.yaml",
                "real2sim/collection.yaml",
            )
        ),
        Path("components/execution_policies/real2sim_verified.yaml"),
        *_DUAL_ARM_RESOURCES,
        *(
            Path("tasks/manipulation/drawer_storage") / name
            for name in (
                "env.long_horizon.yaml",
                "catalog.yaml",
                "README.md",
                "task_program/evaluation.long_horizon.yaml",
                "task_program/integration.long_horizon.yaml",
                "task_program/execution_policy.long_horizon.yaml",
            )
        ),
        *_PROGRAMS,
        *_DEPLOYMENTS,
        *(
            Path("tasks/manipulation/tableware") / task_name / relative_path
            for task_name in ("stack_blocks_two", "blocks_ranking_rgb")
            for relative_path in (
                "env.yaml",
                "catalog.yaml",
                "README.md",
                "task_program/integration.yaml",
            )
        ),
        Path("tasks/manipulation/repeated_pick_place/catalog.yaml"),
        Path("tasks/manipulation/repeated_pick_place/README.md"),
        Path("tasks/manipulation/repeated_pick_place/objective.yaml"),
        Path("tasks/manipulation/push_cube/catalog.yaml"),
        Path("tasks/manipulation/push_cube/README.md"),
        Path(
            "tasks/manipulation/tableware/stack_blocks_two/task_program/evaluation.yaml"
        ),
        Path("tasks/manipulation/tableware/stack_cups/catalog.yaml"),
        Path("tasks/manipulation/tableware/stack_cups/README.md"),
        Path("components/execution_policies/motion_gen_verified.yaml"),
        Path("components/execution_policies/trajectory_open_loop.yaml"),
        Path("components/expansion_policies/task_program_episode.yaml"),
        Path("components/randomization_profiles/cube_initial_pose.yaml"),
        Path("components/randomization_profiles/rgb_visual.yaml"),
        Path("components/embodiments/cobotmagic.yaml"),
        Path("components/embodiments/dual_ur5_dh_pgi_140_80.yaml"),
        Path("components/embodiments/franka_panda.yaml"),
        Path("components/embodiments/tianji_marvin.yaml"),
        Path("components/embodiments/ur5_dh_pgi_140_80.yaml"),
        Path("tasks/manipulation/hand_over/env.yaml"),
        Path("tasks/manipulation/hand_over/README.md"),
        Path("tasks/manipulation/hand_over/envs/tabletop.default.yaml"),
        Path(
            "tasks/manipulation/hand_over/task_program/integration.tianji_marvin.yaml"
        ),
        Path("tasks/manipulation/hand_over/task_program/integration.yaml"),
        Path("tasks/manipulation/open_drawer/envs/default.yaml"),
        Path("tasks/manipulation/open_drawer/envs/newton.yaml"),
        Path("tasks/manipulation/open_drawer/expansion/open_drawer.yaml"),
        Path("tasks/manipulation/open_drawer/README.md"),
        Path("tasks/manipulation/open_drawer/task_program/integration.yaml"),
        Path("tasks/manipulation/repeated_pick_place/envs/default.yaml"),
        Path("tasks/manipulation/repeated_pick_place/envs/tabletop.default.yaml"),
        Path(
            "tasks/manipulation/repeated_pick_place/task_program/integration.tabletop.yaml"
        ),
        Path("components/embodiments/dexforce_w1_030_cloth_fold.yaml"),
        Path("tasks/manipulation/cloth_fold/env.yaml"),
        Path("tasks/manipulation/cloth_fold/task.w1.yaml"),
        Path("tasks/manipulation/cloth_fold/catalog.yaml"),
        Path("tasks/manipulation/cloth_fold/README.md"),
        Path("tasks/manipulation/cloth_fold/task_program/program.yaml"),
        Path("tasks/manipulation/cloth_fold/task_program/integration.yaml"),
        Path("tasks/manipulation/cloth_fold/task_program/evaluation.yaml"),
        Path("tasks/manipulation/cloth_fold/task_program/trajectory.npz"),
        Path("tasks/manipulation/repeated_pick_place/envs/newton.yaml"),
        Path(
            "tasks/manipulation/repeated_pick_place/expansion/"
            "repeated_pick_place.yaml"
        ),
        Path("tasks/manipulation/repeated_pick_place/task_program/integration.yaml"),
        Path("tasks/manipulation/rubiks_cube_pick_place/env.yaml"),
        Path("tasks/manipulation/rubiks_cube_pick_place/task_program/integration.yaml"),
        Path("tasks/manipulation/tableware/pour_water/env.yaml"),
        Path("tasks/manipulation/tableware/pour_water/env.tianji_marvin.yaml"),
        Path("tasks/manipulation/tableware/pour_water/README.md"),
        Path("tasks/manipulation/tableware/pour_water/task_program/integration.yaml"),
        Path("tasks/manipulation/tableware/pour_water/task_program/evaluation.yaml"),
        Path(
            "tasks/manipulation/tableware/pour_water/task_program/integration.dual_arm.yaml"
        ),
    }
)


class _StagedConfigPackage(NamedTuple):
    """Isolated setuptools output and the setup options that produced it."""

    build_lib: Path
    relative_outputs: frozenset[Path]
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


@pytest.fixture(scope="module")
def staged_config_package(
    tmp_path_factory: pytest.TempPathFactory,
) -> _StagedConfigPackage:
    """Stage official Task Program resources through the real build command."""
    package_data = get_package_data()
    include_package_data = _literal_setup_keyword("include_package_data")
    assert type(package_data) is dict
    assert type(include_package_data) is bool

    build_lib = tmp_path_factory.mktemp("task-config-build")
    packages = get_packages()
    assert "embodichain.extensions.tasks.manipulation.cable" in packages
    distribution = Distribution(
        {
            "packages": packages,
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
        include_package_data=include_package_data,
    )


def test_setup_stages_all_official_task_programs(
    staged_config_package: _StagedConfigPackage,
) -> None:
    """The actual setup patterns put nested Task Programs in wheel staging."""
    assert staged_config_package.include_package_data is False
    expected_outputs = {
        Path("embodichain/extensions/tasks/configs") / relative_path
        for relative_path in _RESOURCE_PATHS
    }
    assert expected_outputs <= staged_config_package.relative_outputs
    assert not (
        staged_config_package.build_lib
        / "embodichain/extensions/tasks/configs/tasks/manipulation/tableware/pour_cup/assets"
    ).exists()


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

build_lib = Path(sys.argv[1]).resolve()
expected_deployments = json.loads(sys.argv[2])
import embodichain.extensions.tasks.configs as config_package
from embodichain.lab.task_program import load_task_program
from embodichain.lab.task_program.integrations._configured_composition import (
    _load_configured_task_program_deployment,
)
from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
from embodichain.utils.utility import load_config
from embodichain.extensions.tasks.configs import get_config_path

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
        evaluation=physical.evaluation,
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
        (
            str(staged_config_package.build_lib),
            str(_REPOSITORY_ROOT),
            environment.get("PYTHONPATH", ""),
        )
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
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1]) == {
        path: list(identifiers) for path, identifiers in expected_deployments.items()
    }


def test_drawer_storage_readme_links_survive_wheel_installation(
    staged_config_package: _StagedConfigPackage,
) -> None:
    """Relative README links target files shipped in the installed package."""
    readme = (
        staged_config_package.build_lib
        / "embodichain/extensions/tasks/configs/tasks/manipulation/drawer_storage/README.md"
    )
    targets = re.findall(r"\]\(([^)\s#]+)", readme.read_text(encoding="utf-8"))
    relative = [
        readme.parent / target
        for target in targets
        if re.match(r"[a-z][a-z0-9+.-]*:", target) is None
    ]

    assert relative
    assert all(path.is_file() for path in relative)


def test_installed_wheel_registers_configured_tasks(tmp_path: Path) -> None:
    """Register drawer/GS deployments from installed YAML and discover their IDs."""
    source = tmp_path / "source"
    source.mkdir()
    for name in ("setup.py", "pyproject.toml", "README.md", "VERSION", "MANIFEST.in"):
        shutil.copyfile(_REPOSITORY_ROOT / name, source / name)
    for name in ("embodichain", "examples"):
        shutil.copytree(
            _REPOSITORY_ROOT / name,
            source / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )

    wheel_dir = tmp_path / "wheels"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--no-index",
            "--wheel-dir",
            str(wheel_dir),
            ".",
        ],
        cwd=source,
        check=True,
    )
    (wheel,) = wheel_dir.glob("embodichain-*.whl")
    installed = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(installed),
            str(wheel),
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys

installed = Path(sys.argv[1])
sys.path.insert(0, str(installed))
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import get_env_spec
from embodichain.utils.utility import load_config
from gymnasium import registry
from embodichain.lab.sim.cfg import RobotPresetCfg
from embodichain.utils.config_paths import resolve_config_path

import embodichain
assert Path(embodichain.__file__).is_relative_to(installed)

configs = installed / "embodichain/extensions/tasks/configs/tasks/manipulation/drawer_storage"
assert not (installed / "embodichain/extensions/tasks/manipulation/drawer_storage.py").exists()
assert (configs / "task_program/evaluation.yaml").is_file()
assert "embodichain.extensions.tasks" not in sys.modules

for name, open_positions, contained, clearance_fields, finger_stiffness in (
    ("ur5", (0.0,), "cube_contained", 3, 1000.0),
    ("franka", (0.04, 0.04), "cube_contained", 3, None),
    ("ur5.long_horizon", (0.0,), "paper_cup_contained", 6, 3000.0),
):
    path = configs / f"task.{name}.yaml"
    raw = load_config(path)
    assert raw["id"] not in registry
    cfg = config_to_cfg(raw, source_path=path)
    if finger_stiffness is not None:
        stiffness = cfg.robot.joint_drive_props.stiffness
        assert stiffness["gripper_finger1_joint_1"] == finger_stiffness
    spec = get_env_spec(raw["id"])
    assert spec.cls is EmbodiedEnv
    registration = spec.task_program_registration
    rules = registration.evaluation_conditions
    assert rules["hand_open"].parameters["hand_open"]["positions"] == open_positions
    assert len(rules["fingers_clear"].required_fields) == clearance_fields
    assert contained in registration.evaluation_final_refs

gaussian_ids = {
    "office": "FrankaGSPickPlace-Office-v1",
    "discoverse": "FrankaGSPickPlace-Discoverse-v1",
    "robosimgs": "FrankaGSPickPlace-Robosimgs-v1",
}
gaussian_calibration = {
    "office": ([710.4, 710.4, 480, 270], [1.5, -2, 1.65], [0.25, 0, 1]),
    "discoverse": ([480, 480, 480, 270], [-0.55, 0.1, 1.5], [-0.02, 0.99, 1.03]),
    "robosimgs": ([480, 480, 480, 270], [0.85, -0.65, 0.85], [0.3, 0, 0.38]),
}
shared_embodiments = set()
for scene, expected_id in gaussian_ids.items():
    path = resolve_config_path(
        "embodichain/extensions/tasks/configs/tasks/manipulation/"
        f"gaussian_pick_place/task.{scene}.yaml",
        base_dir=Path.cwd(),
    )
    assert path.is_relative_to(installed)
    raw = load_config(path)
    assert raw["id"] == expected_id
    component = resolve_config_path(raw["embodiment"]["component"], base_dir=path.parent)
    assert component.is_relative_to(installed)
    shared_embodiments.add(component)
    cfg = config_to_cfg(raw, source_path=path)
    assert get_env_spec(expected_id).cls is EmbodiedEnv
    robot = (
        cfg.robot.resolve(cfg.sim_cfg.physics_cfg)
        if isinstance(cfg.robot, RobotPresetCfg) else cfg.robot
    )
    cameras = {camera.uid: camera for camera in [*cfg.sensor, *robot.sensors]}
    assert set(cameras) == {"cam_third", "cam_wrist"}
    assert cameras["cam_wrist"].extrinsics.parent == "fr3_hand"
    assert list(cameras["cam_wrist"].intrinsics) == [430, 430, 480, 270]
    assert list(cameras["cam_wrist"].extrinsics.pos) == [0.1, 0, 0.065]
    assert list(cameras["cam_wrist"].extrinsics.quat) == [
        0.576704088924317, 0.576704088924317, 0.4091605966096607, 0.4091605966096607
    ]
    intrinsics, eye, target = gaussian_calibration[scene]
    third = cameras["cam_third"]
    assert list(third.intrinsics) == intrinsics
    assert list(third.extrinsics.eye) == eye
    assert list(third.extrinsics.target) == target
    assert list(third.extrinsics.up) == [0, 0, 1]
    assert all((camera.width, camera.height) == (960, 540) for camera in cameras.values())
    assert all(camera.enable_color and not camera.enable_depth for camera in cameras.values())
assert len(shared_embodiments) == 1
assert not (installed / "embodichain/extensions/tasks/manipulation/gaussian_pick_place.py").exists()
assert not (installed / "share/embodichain/examples/gym/gaussian_pick_place/configs").exists()

import embodichain.extensions.tasks
from embodichain.cli.list_task import _collect_environment_entries
from embodichain.cli.main import main

assert Path(embodichain.extensions.tasks.__file__).is_relative_to(installed)
assert "embodichain.extensions.tasks.manipulation.drawer_storage" not in sys.modules
assert {entry.env_id for entry in _collect_environment_entries()} >= {
    "TaskProgramDrawerStorage-UR5-v1",
    "TaskProgramDrawerStorage-Franka-v1",
    "TaskProgramDrawerStorage-UR5-LongHorizon-v1",
    *gaussian_ids.values(),
}
with redirect_stdout(StringIO()) as details:
    main(["show-task", "embodichain.extensions.tasks:pour_water"])
assert "pour_water" in details.getvalue()
assert "embodichain/extensions/tasks/configs/tasks/manipulation/tableware/pour_water/" in details.getvalue()
""",
            str(installed),
        ],
        cwd=tmp_path,
        check=True,
    )
