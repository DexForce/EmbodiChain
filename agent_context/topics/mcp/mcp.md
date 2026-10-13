# MCP integration

The MCP package exposes coarse-grained EmbodiChain operations to MCP Hosts. It
is a protocol boundary over existing domain APIs, not a second implementation
of physics, IK, planning, learning, data processing, or asset assembly. Domain
adapters remain independently owned and are composed by the MCP registry.

## Entry points

| Responsibility | Source |
|---|---|
| MCP server and transport | `embodichain/mcp/server.py` |
| Adapter composition | `embodichain/mcp/adapters.py` |
| Tool/resource/prompt facade | `embodichain/mcp/service.py` |
| Backend contract and adapters | `embodichain/mcp/backend.py` |
| Simulation MCP adapter | `embodichain/mcp/simulation.py` |
| URDF asset adapter | `embodichain/mcp/urdf.py` |
| Optional URDF simulation verifier | `embodichain/mcp/urdf_simulation.py` |
| CLI command | `embodichain/cli/main.py` (`embodichain mcp`) |
| Codex trajectory demo | `examples/mcp/codex_trajectory_demo/` |
| URDF assembly demo | `examples/mcp/urdf_assembly_demo/` |

## Phase-one boundary

The phase-one command serves stdio only. Streamable HTTP is deferred until an
authenticated gateway owns authorization, approval, and resource quotas.

The simulation adapter uses explicit `world_id`, `scene_revision`, `run_id`,
and `snapshot_id`, and `trajectory_id` handles. Results carry backend, frame,
units, seed, diagnostics, and artifact metadata. High-frequency physics and
controller loops remain inside the backend; MCP calls operate on worlds,
trajectories, and rollout jobs, and offscreen camera recordings. The default
server registers the simulation facade and the pure `URDFAssemblyAdapter`; the
deterministic `InMemorySimulationBackend` is intended for protocol tests and
examples.

The URDF adapter accepts registered relative asset identifiers, composes models
through `embodichain.toolkits.urdf_assembly`, validates XML, topology, and mesh
references without starting a simulator, and exposes an explicit temporary
simulation verification operation. It must not accept arbitrary filesystem
paths from an MCP Host.

`health` reports the service and backend scope without starting a simulation.
World operations accept optional `expected_scene_revision` values and reject
stale callers before touching simulator state. Trajectories retain the scene
revision observed during generation and are rejected at execution time when
the world has changed.

Phase one does not expose real-device control, arbitrary code execution, or
per-physics-step Agent calls. Tool annotations describe risk, but authorization
and safety enforcement remain execution-layer responsibilities. The package
must remain importable without loading simulation libraries or the optional MCP
SDK. Domain runtime imports belong inside the adapter operations that need them.

## Validation

Focused protocol coverage is in `tests/lab/mcp/test_service.py`. Run it with
the optional `mcp` dependency installed; camera recording additionally uses
the `mcp-demo` extra for ImageIO/FFmpeg. Tests that exercise the protocol SDK
skip when that optional dependency is unavailable. Run the public API checker
after changing the MCP package exports.
