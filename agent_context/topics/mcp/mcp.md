# MCP integration

The MCP adapter exposes coarse-grained, simulation-only EmbodiChain operations
to MCP Hosts. It is a protocol boundary over existing simulation and motion
APIs, not a second implementation of physics, IK, or planning.

## Entry points

| Responsibility | Source |
|---|---|
| MCP server and transport | `embodichain/lab/mcp/server.py` |
| Tool/resource/prompt facade | `embodichain/lab/mcp/service.py` |
| Backend contract and adapters | `embodichain/lab/mcp/backend.py` |
| CLI command | `embodichain/cli/main.py` (`embodichain mcp`) |

## Phase-one boundary

The service uses explicit `world_id`, `scene_revision`, `run_id`, and
`snapshot_id` handles. Results carry backend, frame, units, seed, diagnostics,
and artifact metadata. High-frequency physics and controller loops remain
inside the backend; MCP calls operate on worlds, trajectories, and rollout
jobs. The default server uses `SimulationManagerBackend`; the deterministic
`InMemorySimulationBackend` is intended for protocol tests and examples.

Phase one does not expose real-device control, arbitrary code execution, or
per-physics-step Agent calls. Tool annotations describe risk, but authorization
and safety enforcement remain execution-layer responsibilities.

## Validation

Focused protocol coverage is in `tests/lab/mcp/test_service.py`. Run it with
the optional `mcp` dependency installed; tests that exercise the protocol SDK
skip when that optional dependency is unavailable. Run the public API checker
after changing the MCP package exports.
