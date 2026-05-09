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

import os
import pytest


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

        # PREVENT IMPLICIT INITIALIZATION BY EXPLICITLY INITIALIZING DEXSIM HERE
        import dexsim
        import dexsim.types

        # Map string to dexsim configuration types
        renderer_map = {
            "hybrid": dexsim.types.Renderer.HYBRID,
            "fast-rt": dexsim.types.Renderer.FASTRT,
        }
        backend_map = {
            "hybrid": dexsim.types.Backend.VULKAN,
            "fast-rt": dexsim.types.Backend.VULKAN,
        }

        if dexsim.get_world_num() == 0:
            sim_config = dexsim.WorldConfig()
            sim_config.renderer = renderer_map.get(
                renderer, dexsim.types.Renderer.HYBRID
            )
            sim_config.backend = backend_map.get(renderer, dexsim.types.Backend.VULKAN)
            sim_config.open_windows = False
            # This triggers initialization with the correct properties immediately.
            dexsim.init_sim_engine(sim_config)


@pytest.fixture(autouse=True, scope="function")
def wait_scene_destruction_after_test():
    """Ensure C++ engine scenes are fully destructed globally after each test exits."""
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
