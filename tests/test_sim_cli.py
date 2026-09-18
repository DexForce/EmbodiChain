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

import argparse

import pytest

from embodichain.cli.sim import (
    add_sim_args_to_parser,
    add_seed_arg_to_parser,
    resolve_seed,
)

pytestmark = pytest.mark.no_sim


def test_sim_parser_only_exposes_simulation_options() -> None:
    parser = argparse.ArgumentParser()
    add_sim_args_to_parser(parser)
    args = parser.parse_args([])
    assert (args.num_envs, args.device, args.physics) == (1, None, "default")
    assert not hasattr(args, "seed")
    assert not hasattr(args, "gym_config")
    assert not hasattr(args, "action_config")
    assert not hasattr(args, "record_trajectory")
    args = parser.parse_args(["--physics", "newton", "--viser", "--num_envs", "4"])
    assert args.physics == "newton" and args.viser and args.num_envs == 4


def test_seed_is_opt_in_and_explicit_values_override_default() -> None:
    parser = argparse.ArgumentParser()
    add_sim_args_to_parser(parser)
    add_seed_arg_to_parser(parser, default=0, scope="scene perturbations")
    assert parser.parse_args([]).seed == 0
    assert parser.parse_args(["--seed", "42"]).seed == 42
    assert parser.parse_args(["--seed", "-1"]).seed == -1
    assert "scene perturbations" in parser.format_help()


@pytest.mark.parametrize("value", ["-2", "4294967296", "abc"])
def test_seed_rejects_values_outside_portable_range(value: str) -> None:
    parser = argparse.ArgumentParser()
    add_seed_arg_to_parser(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--seed", value])


def test_resolve_seed_does_not_seed_global_generators(monkeypatch) -> None:
    monkeypatch.setattr("secrets.randbits", lambda bits: 123)
    assert resolve_seed(-1) == 123
    assert resolve_seed(42) == 42
    with pytest.raises(ValueError):
        resolve_seed(-2)


# Keep this inventory broad: new standalone entry points inherit the same check.
def _example_scripts():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return [
        p
        for directory in (root / "examples", root / "scripts/tutorials")
        for p in sorted(directory.rglob("*.py"))
        if "def build_parser()" in p.read_text()
        and "add_sim_args_to_parser" in p.read_text()
    ]


@pytest.mark.parametrize("script", _example_scripts(), ids=lambda p: p.name)
def test_example_help_without_site_packages(script) -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--physics" in result.stdout
    assert "--gym_config" not in result.stdout


def test_common_parser_does_not_import_runtime_dependencies() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import argparse, sys; "
            "from embodichain.cli.sim import add_sim_args_to_parser; "
            "add_sim_args_to_parser(argparse.ArgumentParser()); "
            "assert not {'torch','dexsim','gymnasium','curobo'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def _load_cli_namespace(script):
    import ast

    source = script.read_text()
    tree = ast.parse(source)
    # Entry points intentionally define their parser before the first main guard.
    boundary = next(
        node.lineno - 1
        for node in tree.body
        if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
    )
    namespace = {"__name__": "parser_contract", "__file__": str(script)}
    exec(
        compile("\n".join(source.splitlines()[:boundary]), str(script), "exec"),
        namespace,
    )
    return namespace


def _load_parser_only(script):
    return _load_cli_namespace(script)["build_parser"]()


@pytest.mark.parametrize("script", _example_scripts(), ids=lambda p: p.name)
def test_example_argument_destinations_have_single_owners(script) -> None:
    parser = _load_parser_only(script)
    destinations = [action.dest for action in parser._actions]
    assert len(destinations) == len(set(destinations))
    assert parser.parse_args(["--arena-space", "3"]).arena_space == 3
    assert parser.parse_args(["--arena_space", "4"]).arena_space == 4


@pytest.mark.parametrize(
    "script",
    [
        p
        for p in _example_scripts()
        if p.name
        in {
            "curobo_planner.py",
            "open_drawer.py",
            "grasp_cup_to_caffe.py",
            "modular_env.py",
            "random_reach.py",
        }
    ],
    ids=lambda p: p.name,
)
def test_seeded_examples_have_concrete_defaults_and_overrides(script) -> None:
    parser = _load_parser_only(script)
    assert parser.parse_args([]).seed == 0
    assert parser.parse_args(["--seed", "17"]).seed == 17
    assert parser.parse_args(["--seed", "-1"]).seed == -1


def test_standalone_consumers_do_not_reintroduce_gym_launcher() -> None:
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for directory in (root / "examples", root / "scripts/tutorials"):
        for script in directory.rglob("*.py"):
            tree = ast.parse(script.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert all(
                        name.name != "add_env_launcher_args_to_parser"
                        for name in node.names
                    ), str(script)


@pytest.mark.parametrize("script", _example_scripts(), ids=lambda p: p.name)
def test_example_defaults_pass_pre_runtime_validation(script) -> None:
    namespace = _load_cli_namespace(script)
    parse = namespace.get("parse_args", namespace.get("parse_arguments"))
    args = (
        parse([]) if parse is not None else namespace["build_parser"]().parse_args([])
    )
    assert args.num_envs >= 1
