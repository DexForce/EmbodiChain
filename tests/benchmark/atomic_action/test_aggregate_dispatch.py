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

"""Tests for the aggregate dispatcher's in-process path.

``--in_process`` runs a benchmark inside the aggregate process, so the child
namespace has to carry the arguments that benchmark reads. Building it from a
hand-kept list left every newly registered action without its own
case-selection arguments, and each one raised ``AttributeError`` on its first
line of work.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
import types
from pathlib import Path

import pytest

from scripts.benchmark.atomic_action import run_benchmark
from scripts.benchmark.atomic_action.run_benchmark import (
    ACTION_MODULES,
    _make_child_args,
    _run_in_process_benchmarks,
    add_benchmark_args,
)

ARGS_ATTRIBUTE = re.compile(r"\bargs\.([a-z_]+)")


def _aggregate_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return aggregate benchmark arguments as the CLI would parse them."""
    parser = argparse.ArgumentParser(add_help=False)
    add_benchmark_args(parser)
    return parser.parse_args(argv or [])


@pytest.mark.parametrize("action_name", sorted(ACTION_MODULES))
def test_every_action_receives_the_arguments_its_benchmark_reads(
    action_name: str,
) -> None:
    """Each dispatched benchmark finds every ``args`` attribute it reads."""
    module = importlib.import_module(ACTION_MODULES[action_name])
    expected = set(ARGS_ATTRIBUTE.findall(inspect.getsource(module)))

    child_args = _make_child_args(_aggregate_args(), action_name)

    missing = sorted(name for name in expected if not hasattr(child_args, name))
    assert not missing, f"{action_name} would raise AttributeError for {missing}"


@pytest.mark.parametrize("action_name", sorted(ACTION_MODULES))
def test_each_action_keeps_its_own_case_selection_defaults(action_name: str) -> None:
    """A dispatched benchmark runs the cases its own parser would select."""
    module = importlib.import_module(ACTION_MODULES[action_name])
    parser = argparse.ArgumentParser(add_help=False)
    module.add_benchmark_args(parser)
    standalone = parser.parse_args([])

    child_args = _make_child_args(_aggregate_args(), action_name)

    case_arguments = [name for name in vars(standalone) if name.endswith("_cases")]
    assert case_arguments, f"{action_name} declares no case selection."
    for name in case_arguments:
        assert getattr(child_args, name) == getattr(standalone, name)


def test_shared_settings_override_the_module_defaults() -> None:
    """Device, profile, repeat and video settings come from the aggregate run."""
    args = _aggregate_args(["--profile", "smoke", "--device", "cpu"])

    child_args = _make_child_args(args, "open_door")

    assert child_args.device == "cpu"
    assert child_args.profile == "smoke"
    assert child_args.smoke is True
    assert child_args.repeat == 1
    assert child_args.video_dir == args.video_dir
    assert child_args.n_sample == 1000


def test_object_selection_is_forwarded_only_when_the_run_gives_one() -> None:
    """An unset object selection leaves the benchmark's own default in place."""
    module = importlib.import_module(ACTION_MODULES["pick_up"])
    parser = argparse.ArgumentParser(add_help=False)
    module.add_benchmark_args(parser)
    standalone = parser.parse_args([])

    default_args = _make_child_args(_aggregate_args(), "pick_up")
    selected_args = _make_child_args(
        _aggregate_args(["--object_types", "cube"]), "pick_up"
    )

    assert default_args.object_types == standalone.object_types
    assert selected_args.object_types == ["cube"]


def test_a_skill_without_grasp_sampling_is_not_given_grasp_arguments() -> None:
    """Shared settings are applied only where the benchmark declares them."""
    args = _aggregate_args()
    child_args = _make_child_args(args, "move_joints")

    assert not hasattr(child_args, "n_sample")
    assert not hasattr(child_args, "force_reannotate")
    assert child_args.device == args.device


def test_each_in_process_benchmark_starts_from_a_released_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The simulator singleton is torn down between benchmarks.

    Leaving the previous scene standing makes the next benchmark a second
    simulator instance, and its planner then resolves against the first
    benchmark's robots instead of its own.
    """
    calls: list[str] = []

    def fake_import(module_name: str) -> types.SimpleNamespace:
        def run_all_benchmarks(args: argparse.Namespace) -> Path:
            del args
            calls.append(f"run:{module_name}")
            return Path(f"{module_name}.md")

        return types.SimpleNamespace(
            add_benchmark_args=lambda parser: None,
            run_all_benchmarks=run_all_benchmarks,
        )

    monkeypatch.setattr(run_benchmark.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        run_benchmark, "release_simulation", lambda: calls.append("release")
    )

    reports = _run_in_process_benchmarks(_aggregate_args(), ["press", "hand_over"])

    assert calls == [
        "run:" + ACTION_MODULES["press"],
        "release",
        "run:" + ACTION_MODULES["hand_over"],
        "release",
    ]
    assert len(reports) == 2
