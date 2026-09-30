# MCP integration

The MCP package exposes coarse-grained EmbodiChain operations to MCP Hosts. It
is a protocol boundary over existing domain APIs, not a second implementation
of physics, IK, planning, learning, or data processing. Simulation is the first
adapter; future adapters remain domain-owned.

## Entry points

| Responsibility | Source |
|---|---|
| MCP server and transport | `embodichain/mcp/server.py` |
| Tool/resource/prompt facade | `embodichain/mcp/service.py` |
| Backend contract and adapters | `embodichain/mcp/backend.py` |
| CLI command | `embodichain/cli/main.py` (`embodichain mcp`) |

## Phase-one boundary

The simulation adapter uses explicit `world_id`, `scene_revision`, `run_id`, and
`snapshot_id` handles. Results carry backend, frame, units, seed, diagnostics,
and artifact metadata. High-frequency physics and controller loops remain
inside the backend; MCP calls operate on worlds, trajectories, and rollout
jobs. The default server uses `SimulationManagerBackend`; the deterministic
`InMemorySimulationBackend` is intended for protocol tests and examples.

Phase one does not expose real-device control, arbitrary code execution, or
per-physics-step Agent calls. Tool annotations describe risk, but authorization
and safety enforcement remain execution-layer responsibilities. The package
must remain importable without loading simulation libraries or the optional MCP
SDK. Domain runtime imports belong inside the adapter operations that need them.

## Validation

Focused protocol coverage is in `tests/lab/mcp/test_service.py`. Run it with
the optional `mcp` dependency installed; tests that exercise the protocol SDK
skip when that optional dependency is unavailable. Run the public API checker
after changing the MCP package exports.
