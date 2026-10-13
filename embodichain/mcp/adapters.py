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

"""Provider registration primitives for the EmbodiChain MCP server."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

__all__ = ["MCPAdapter", "MCPAdapterRegistry"]


ToolRegistrar = Callable[..., None]


class MCPAdapter(Protocol):
    """Protocol implemented by one project-domain MCP provider.

    An adapter owns the protocol registrations for one domain.  It can expose
    tools, resources, and prompts without requiring the central server to know
    the domain's implementation details.
    """

    name: str

    def register(self, server: Any, *, register_tool: ToolRegistrar) -> None:
        """Register the adapter's MCP tools, resources, and prompts."""

    def capabilities(self) -> Sequence[str]:
        """Return the operation names contributed by this adapter."""

    def close(self) -> None:
        """Release adapter-owned handles and executors."""


class MCPAdapterRegistry:
    """Compose independently owned MCP domain adapters.

    Args:
        adapters: Initial adapters to register.  Adapter names must be unique.

    The registry deliberately does not impose a backend type.  A pure asset
    adapter can therefore be registered beside a simulator adapter without
    importing or starting the simulator for asset-only requests.
    """

    def __init__(self, adapters: Iterable[MCPAdapter] = ()) -> None:
        self._adapters: dict[str, MCPAdapter] = {}
        for adapter in adapters:
            self.add(adapter)

    @property
    def adapters(self) -> tuple[MCPAdapter, ...]:
        """Return adapters in their registration order."""
        return tuple(self._adapters.values())

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered adapter names in their registration order."""
        return tuple(self._adapters)

    def add(self, adapter: MCPAdapter) -> None:
        """Add one adapter, rejecting missing or duplicate names.

        Args:
            adapter: Domain provider implementing :class:`MCPAdapter`.

        Raises:
            ValueError: If the adapter name is empty or already registered.
        """
        name = getattr(adapter, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("MCP adapters must expose a non-empty string name")
        if name in self._adapters:
            raise ValueError(f"MCP adapter {name!r} is already registered")
        capabilities = tuple(adapter.capabilities())
        if any(
            not isinstance(capability, str) or not capability
            for capability in capabilities
        ):
            raise ValueError(f"MCP adapter {name!r} exposes an invalid capability name")
        occupied = {
            capability
            for existing in self._adapters.values()
            for capability in existing.capabilities()
        }
        collisions = sorted(set(capabilities) & occupied)
        if collisions:
            raise ValueError(
                f"MCP adapter {name!r} collides with registered capabilities: "
                f"{', '.join(collisions)}"
            )
        self._adapters[name] = adapter

    def register(self, server: Any, *, register_tool: ToolRegistrar) -> None:
        """Register every adapter with one MCP server instance.

        Args:
            server: MCP server receiving registrations.
            register_tool: Central tool wrapper used for annotations and errors.
        """
        for adapter in self._adapters.values():
            adapter.register(server, register_tool=register_tool)

    def capabilities(self) -> list[str]:
        """Return the union of adapter operation names without duplicates."""
        result: list[str] = []
        seen: set[str] = set()
        for adapter in self._adapters.values():
            for capability in adapter.capabilities():
                if capability not in seen:
                    result.append(capability)
                    seen.add(capability)
        return result

    def close(self) -> None:
        """Close adapters in reverse registration order."""
        errors: list[Exception] = []
        for adapter in reversed(self.adapters):
            close = getattr(adapter, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as error:  # noqa: BLE001 - close every provider.
                    errors.append(error)
        if errors:
            raise RuntimeError(
                "One or more MCP adapters failed to close: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            ) from errors[0]
