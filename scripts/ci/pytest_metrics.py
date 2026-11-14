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

"""Optional pytest phase and collection metrics, isolated per worker process."""

from __future__ import annotations

from collections.abc import Generator
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import pytest

__all__ = [
    "pytest_addoption",
    "pytest_configure",
    "pytest_collection_finish",
    "pytest_collection_modifyitems",
    "pytest_runtest_makereport",
    "pytest_make_collect_report",
    "pytest_sessionfinish",
]

_STATE = pytest.StashKey[dict[str, Any]]()
_HISTORY = pytest.StashKey[dict[str, float]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register an opt-in output directory without inheritable environment flags."""
    parser.addoption(
        "--ci-metrics-dir", type=Path, help="Write per-process test metrics"
    )

    parser.addoption(
        "--ci-duration-history",
        type=Path,
        help="CPU metrics directory for group ordering",
    )


def _group(item: pytest.Item) -> str:
    names = sorted(
        {
            str(mark.kwargs.get("name", mark.args[0] if mark.args else ""))
            for mark in item.iter_markers("xdist_group")
        }
    )
    return "_".join(names) if names else item.nodeid.split("::", 1)[0]


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Prioritize expensive CPU groups while preserving their internal order."""
    history = config.stash.get(_HISTORY, None)
    if history:
        items.sort(key=lambda item: -history.get(_group(item), 0.0))


def pytest_configure(config: pytest.Config) -> None:
    """Initialize only the explicitly requested process-local metrics recorder."""
    history_path = config.getoption("--ci-duration-history")
    if history_path is not None:
        if not history_path.is_dir():
            raise pytest.UsageError(
                f"Missing duration history directory: {history_path}"
            )
        weights: dict[str, float] = {}
        for path in sorted(history_path.glob("*.json")):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                for phase in report["phases"]:
                    group = phase["group"]
                    weights[group] = weights.get(group, 0.0) + float(phase["seconds"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise pytest.UsageError(
                    f"Invalid duration history {path}: {error}"
                ) from error
        config.stash[_HISTORY] = weights
    directory = config.getoption("--ci-metrics-dir")
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    worker = getattr(config, "workerinput", {}).get("workerid", "main")
    stream = (directory / f"phases-{worker}-{os.getpid()}.jsonl").open(
        "w", encoding="utf-8", buffering=1
    )
    config.stash[_STATE] = {
        "version": 1,
        "pid": os.getpid(),
        "worker": getattr(config, "workerinput", {}).get("workerid", "main"),
        "started": time.monotonic(),
        "directory": directory,
        "stream": stream,
        "arguments": [str(arg) for arg in config.invocation_params.args],
        "collection": [],
        "collection_reports": [],
        "phases": [],
    }


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(
    collector: pytest.Collector,
) -> Generator[None, Any, None]:
    """Keep module-level skips and import errors distinct from runnable cases."""
    outcome = yield
    state = collector.config.stash.get(_STATE, None)
    if state is not None:
        report = outcome.get_result()
        if report.failed or report.skipped:
            state["collection_reports"].append(
                {
                    "nodeid": report.nodeid,
                    "outcome": report.outcome,
                    "reason": str(report.longrepr),
                }
            )


def pytest_collection_finish(session: pytest.Session) -> None:
    """Record the final ordered selection including resource and skip markers."""
    state = session.config.stash.get(_STATE, None)
    if state is None:
        return
    state["collection"] = [
        {
            "nodeid": item.nodeid,
            "markers": sorted({m.name for m in item.iter_markers()}),
        }
        for item in session.items
    ]
    state["collection_seconds"] = time.monotonic() - state["started"]
    directory = state["directory"] / "collection"
    directory.mkdir(exist_ok=True)
    path = directory / f"{state['worker']}-{state['pid']}.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "collection": state["collection"],
                "collection_reports": state["collection_reports"],
            }
        ),
        encoding="utf-8",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, Any, None]:
    """Capture setup failures, skips and teardown failures as well as test calls."""
    outcome = yield
    state = item.config.stash.get(_STATE, None)
    if state is None:
        return
    report = outcome.get_result()
    phase = {
        "nodeid": report.nodeid,
        "group": _group(item),
        "phase": report.when,
        "outcome": report.outcome,
        "seconds": report.duration,
        "wasxfail": getattr(report, "wasxfail", None),
        "reason": str(report.longrepr) if report.failed or report.skipped else None,
    }
    if (
        report.when == "teardown"
        and item.get_closest_marker("requires_sim") is not None
        and item.get_closest_marker("subprocess_sim") is None
    ):
        state["native_observed"] = True
        phase["native_resources"] = _native_resources()
    state["phases"].append(phase)
    state["stream"].write(json.dumps(phase) + "\n")


def _native_resources() -> dict[str, Any] | None:
    """Observe already-loaded native state without creating a context or scene."""
    native = sys.modules.get("dexsim")
    manager_module = sys.modules.get("embodichain.lab.sim.sim_manager")
    if native is None or manager_module is None:
        return None
    try:
        manager = manager_module.SimulationManager
        return {
            "worlds": int(native.get_world_num()),
            "managers": len(manager._instances),
            "cleanup_queue": manager._cleanup_queue.qsize(),
        }
    except Exception as error:
        return {"observation_error": repr(error)}


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write one independent file per PID; no child process can overwrite a peer."""
    state = session.config.stash.get(_STATE, None)
    if state is None:
        return
    state["stream"].close()
    if state.get("native_observed", False):
        state["native_final_resources"] = _native_resources()
    result = {
        key: value
        for key, value in state.items()
        if key not in {"started", "directory", "stream"}
    }
    result.update(
        elapsed_seconds=time.monotonic() - state["started"], exitstatus=int(exitstatus)
    )
    try:
        import resource

        result["max_rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["max_rss_unit"] = "bytes" if sys.platform == "darwin" else "KiB"
    except ImportError:
        result["max_rss"] = None
        result["max_rss_unit"] = None
    path = state["directory"] / f"{state['worker']}-{state['pid']}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
