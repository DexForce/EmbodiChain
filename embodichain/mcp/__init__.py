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

"""Cross-domain MCP access layer for EmbodiChain capabilities."""

from __future__ import annotations

from .adapters import MCPAdapter, MCPAdapterRegistry
from .backend import (
    InMemorySimulationBackend,
    SimulationBackend,
    SimulationManagerBackend,
)
from .server import cli, create_server, serve
from .service import EmbodiChainMCPService
from .simulation import SimulationMCPAdapter
from .urdf import URDFAssemblyAdapter

__all__ = [
    "MCPAdapter",
    "MCPAdapterRegistry",
    "EmbodiChainMCPService",
    "InMemorySimulationBackend",
    "SimulationBackend",
    "SimulationManagerBackend",
    "SimulationMCPAdapter",
    "URDFAssemblyAdapter",
    "cli",
    "create_server",
    "serve",
]
