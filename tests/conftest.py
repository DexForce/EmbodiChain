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

from functools import lru_cache

import hashlib
import inspect
import json
import os
import re
from pathlib import Path
import pytest

os.environ.setdefault("EMBODICHAIN_SIM_EXIT_PROCESS", "0")


@pytest.fixture(scope="session")
def _discover_task_packages():
    """Discover all installed task packages once per test session.

    Official task environments now ship as the ``embodichain_tasks`` import
    package inside the main EmbodiChain distribution. Third-party packages can
    also declare an ``embodichain.tasks`` entry point. Tasks are no longer
    auto-registered by importing
    ``embodichain.lab.gym.envs``, so tests that build task envs via
    ``make(env_id)`` / ``build_env`` / ``train_from_config`` need discovery to
    run first. This mirrors what the ``run_task`` and ``train-rl`` CLIs do at
    startup.
    """
    from embodichain.lab.gym.utils.registration import (
        discover_task_packages,
        execute_init_hooks,
    )

    discover_task_packages()
    execute_init_hooks()


@pytest.fixture(autouse=True)
def _discover_task_packages_for_marked_tests(request):
    """Discover task packages only for tests that consume registered tasks."""
    if request.node.get_closest_marker("requires_tasks") is not None:
        request.getfixturevalue("_discover_task_packages")
    yield


def pytest_addoption(parser):
    parser.addoption(
        "--renderer",
        action="store",
        default="hybrid",
        help="Specify the renderer backend: hybrid, or fast-rt",
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run tests marked gpu; these are skipped by default to limit VRAM use.",
    )
    parser.addoption(
        "--ci-collect-manifest",
        type=Path,
        help="Write the ordered resource-lane selection during collect-only.",
    )
    parser.addoption(
        "--ci-expected-collection",
        type=Path,
        help="Require the same ordered selection before executing a resource lane.",
    )


def pytest_configure(config):
    if config.getoption("--ci-collect-manifest") is not None and (
        not config.option.collectonly
        or config.getoption("--ci-expected-collection") is not None
    ):
        raise pytest.UsageError("--ci-collect-manifest requires collect-only discovery")
    renderer = config.getoption("--renderer")
    if renderer:
        if renderer not in ["hybrid", "fast-rt"]:
            pytest.exit(
                f"Invalid renderer: {renderer}. Must be one of 'hybrid', 'fast-rt'"
            )

        # DexSim initialization is intentionally deferred to the first real-simulation
        # test.  Most of the suite consists of pure-Python tests and should not acquire
        # a CUDA/Vulkan context merely because pytest has started.


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session: pytest.Session) -> None:
    """Record discovery or reject a changed selection before any test setup."""
    config = session.config
    output = config.getoption("--ci-collect-manifest")
    expected = config.getoption("--ci-expected-collection")
    if output is None and expected is None:
        return
    if session.testsfailed:
        if expected is not None:
            raise pytest.UsageError(
                "Resource-lane collection failed after discovery; refusing to execute."
            )
        return
    nodes = [item.nodeid for item in session.items]
    manifest = {
        "version": 1,
        "files": list(dict.fromkeys(node.split("::", 1)[0] for node in nodes)),
        "nodeids": nodes,
        "markers": {
            item.nodeid: sorted({marker.name for marker in item.iter_markers()})
            for item in session.items
        },
    }
    if output is not None:
        output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if expected is not None:
        try:
            recorded = json.loads(expected.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise pytest.UsageError(
                f"Cannot read resource-lane selection: {error}"
            ) from error
        if recorded != manifest:
            raise pytest.UsageError(
                "Resource-lane collection changed after discovery; "
                "refusing to execute a different ordered test selection."
            )


_SIMULATION_MANAGER_CONSTRUCTOR = re.compile(r"\bSimulationManager\s*\(")


@lru_cache(maxsize=None)
def _source_constructs_real_sim(obj):
    """Return whether an object's source directly constructs a simulation manager."""
    try:
        return (
            _SIMULATION_MANAGER_CONSTRUCTOR.search(inspect.getsource(obj)) is not None
        )
    except (OSError, TypeError):
        return False


def _callable_constructs_real_sim(obj):
    """Check a callable and the module-level helpers it directly references."""
    if _source_constructs_real_sim(obj):
        return True

    code = getattr(obj, "__code__", None)
    module = inspect.getmodule(obj)
    if code is None or module is None:
        return False

    for name in code.co_names:
        helper = vars(module).get(name)
        if inspect.isfunction(helper) and _source_constructs_real_sim(helper):
            return True
    return False


def _requires_real_sim(item):
    """Return whether a test item creates a real SimulationManager."""
    if item.get_closest_marker("subprocess_sim") is not None:
        return False
    if item.get_closest_marker("requires_sim") is not None:
        return True
    if item.get_closest_marker("no_sim") is not None:
        return False

    module = getattr(item, "module", None)
    if module is not None and "SimulationManager" in vars(module):
        return True

    if _callable_constructs_real_sim(item.obj):
        return True

    test_class = getattr(item, "cls", None)
    if test_class is not None:
        for cls in test_class.__mro__:
            if _source_constructs_real_sim(cls):
                return True

    fixture_info = getattr(item, "_fixtureinfo", None)
    fixture_defs_by_name = getattr(fixture_info, "name2fixturedefs", {})
    for fixture_defs in fixture_defs_by_name.values():
        for fixture_def in fixture_defs:
            if _callable_constructs_real_sim(fixture_def.func):
                return True

    return False


def _initialize_sim_engine(renderer):
    """Initialize DexSim once, immediately before the first real-sim test."""
    from embodichain.lab.sim import cfg

    cfg.DEFAULT_RENDERER = renderer

    import dexsim
    import dexsim.types

    if dexsim.get_world_num() != 0:
        return

    renderer_map = {
        "hybrid": dexsim.types.Renderer.HYBRID,
        "fast-rt": dexsim.types.Renderer.FASTRT,
    }
    sim_config = dexsim.WorldConfig()
    # Keep the process-wide engine initialization aligned with the manager's
    # default. DexSim's native auto worker setting changes contact ordering
    # and adds substantial scheduling overhead to these small test scenes.
    sim_config.cpu_num = 1
    sim_config.renderer = renderer_map[renderer]
    sim_config.backend = dexsim.types.Backend.VULKAN
    sim_config.open_windows = False
    dexsim.init_sim_engine(sim_config)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Classify real-simulation tests for fast and resource-aware test selection."""
    for item in items:
        nodeid = item.nodeid.lower()
        if "cuda" in nodeid or "gpu" in nodeid:
            item.add_marker(pytest.mark.gpu)
        requires_sim = _requires_real_sim(item)
        if requires_sim:
            item.add_marker(pytest.mark.requires_sim)
            item.add_marker(pytest.mark.xdist_group(name="simulation"))
        else:
            module_name = getattr(item.module, "__name__", "unknown")
            group = f"module-{module_name.replace('.', '-')}"
            if (
                item.get_closest_marker("slow") is not None
                and item.get_closest_marker("requires_sim") is None
                and item.get_closest_marker("subprocess_sim") is None
                and item.get_closest_marker("gpu") is None
                and item.get_closest_marker("xdist_group") is None
            ):
                # Slow CPU cases are independent; distribute large reference
                # matrices while preserving explicit resource-sharing groups.
                group = f"slow-cpu-{hashlib.sha256(item.nodeid.encode()).hexdigest()}"
            item.add_marker(pytest.mark.xdist_group(name=group))
        if requires_sim and (
            "/sensors/" in nodeid or "hybrid" in nodeid or "fastrt" in nodeid
        ):
            item.add_marker(pytest.mark.renderer)

        if item.get_closest_marker("gpu") is not None and not config.getoption(
            "--run-gpu"
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason="GPU tests require --run-gpu and should run in a serial job."
                )
            )


def pytest_runtest_setup(item):
    """Select the renderer after skip checks, before class-scoped scenes.

    Conftest setup hooks run before the runner's fixture setup. Keep default
    priority so pytest's tryfirst skip checks can avoid native initialization.
    """
    if _requires_real_sim(item):
        _initialize_sim_engine(item.config.getoption("--renderer"))


@pytest.fixture(autouse=True, scope="function")
def wait_scene_destruction_after_test(request):
    """Ensure C++ engine scenes are fully destructed globally after each test exits."""
    if not _requires_real_sim(request.node):
        yield
        return

    yield

    # [Improvement - delayed destruction]: top-level dequeue and traceback cleanup.
    # Pytest retains Tracebacks on failure; breaking the exception stack ensures
    # that local variables of temporary objects on the stack can be garbage collected.
    import sys
    import gc

    sys.last_traceback = None
    sys.last_value = None
    sys.last_type = None

    # [Core fix]: drain the cleanup queue to consume SimManager and related objects
    from embodichain.lab.sim.sim_manager import SimulationManager

    SimulationManager.flush_cleanup_queue()
