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
        help="Specify the renderer backend: legacy, hybrid, or fast-rt",
    )


def pytest_configure(config):
    renderer = config.getoption("--renderer")
    if renderer:
        if renderer not in ["legacy", "hybrid", "fast-rt"]:
            pytest.exit(
                f"Invalid renderer: {renderer}. Must be one of 'legacy', 'hybrid', 'fast-rt'"
            )

        # Override the global default renderer in the simulation config
        from embodichain.lab.sim import cfg

        cfg.DEFAULT_RENDERER = renderer

@pytest.fixture(autouse=True, scope="function")
def wait_scene_destruction_after_test():
    """Ensure C++ engine scenes are fully destructed globally after each test exits."""
    yield
    
    # [改良方案 - 延迟销毁]: 顶层出队与报错清理。
    # Pytest 会在失败时保留 Traceback，打断异常栈可以确保栈上的临时对象的局部变量能被垃圾回收。
    import sys
    import gc
    sys.last_traceback = None
    sys.last_value = None
    sys.last_type = None
    
    # [核心修补]: 统一消费清理队列内的 SimManager 和相关对象
    from embodichain.lab.sim.sim_manager import SimulationManager
    SimulationManager.flush_cleanup_queue()
