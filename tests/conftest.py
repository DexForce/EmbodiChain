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

import inspect
import os
import pytest

os.environ.setdefault("EMBODICHAIN_SIM_EXIT_PROCESS", "0")


def pytest_addoption(parser):
    parser.addoption(
        "--renderer",
        action="store",
        default="hybrid",
        help="Specify the renderer backend: hybrid, or fast-rt",
    )


def pytest_configure(config):
    renderer = config.getoption("--renderer")
    if renderer:
        if renderer not in ["hybrid", "fast-rt"]:
            pytest.exit(
                f"Invalid renderer: {renderer}. Must be one of 'hybrid', 'fast-rt'"
            )

        # Override the global default renderer in the simulation config
        from embodichain.lab.sim import cfg

        cfg.DEFAULT_RENDERER = renderer

        # DexSim initialization is intentionally deferred to the first real-simulation
        # test.  Most of the suite consists of pure-Python tests and should not acquire
        # a CUDA/Vulkan context merely because pytest has started.


def _requires_real_sim(item):
    """Return whether a test module creates a real SimulationManager."""
    if item.get_closest_marker("requires_sim") is not None:
        return True
    module = getattr(item, "module", None)
    if module is not None and "SimulationManager" in vars(module):
        return True

    # Some planner regression tests intentionally import the simulation manager
    # inside the test body to keep their module import lightweight.
    try:
        return "SimulationManager" in inspect.getsource(item.obj)
    except (OSError, TypeError):
        return False


def _initialize_sim_engine(renderer):
    """Initialize DexSim once, immediately before the first real-sim test."""
    import dexsim
    import dexsim.types

    if dexsim.get_world_num() != 0:
        return

    renderer_map = {
        "hybrid": dexsim.types.Renderer.HYBRID,
        "fast-rt": dexsim.types.Renderer.FASTRT,
    }
    sim_config = dexsim.WorldConfig()
    sim_config.renderer = renderer_map[renderer]
    sim_config.backend = dexsim.types.Backend.VULKAN
    sim_config.open_windows = False
    dexsim.init_sim_engine(sim_config)


def pytest_collection_modifyitems(config, items):
    """Classify real-simulation tests for fast and resource-aware test selection."""
    for item in items:
        nodeid = item.nodeid.lower()
        requires_sim = _requires_real_sim(item)
        if requires_sim:
            item.add_marker(pytest.mark.requires_sim)
        if "cuda" in nodeid or "gpu" in nodeid:
            item.add_marker(pytest.mark.gpu)
        if requires_sim and (
            "/sensors/" in nodeid or "hybrid" in nodeid or "fastrt" in nodeid
        ):
            item.add_marker(pytest.mark.renderer)


@pytest.fixture(autouse=True, scope="function")
def wait_scene_destruction_after_test(request):
    """Ensure C++ engine scenes are fully destructed globally after each test exits."""
    if not _requires_real_sim(request.node):
        yield
        return

    _initialize_sim_engine(request.config.getoption("--renderer"))
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
