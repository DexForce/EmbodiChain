MCP integration
===============

EmbodiChain's MCP integration exposes coarse-grained simulation and motion
capabilities to an MCP Host. The integration is a protocol boundary over
existing EmbodiChain APIs; physics, rendering, kinematics, planning, and
resource cleanup remain owned by the simulation backend.

Phase-one scope
---------------

The phase-one server uses the stdio transport and starts with
``python -m embodichain mcp``. It provides:

* discovery tools for server health, robots, tasks, physics backends, and
  capabilities;
* world lifecycle tools for creation, scene loading, stepping, reset,
  snapshots, restore, and destruction;
* FK, IK, reachability, joint-limit validation, motion planning, and bounded
  trajectory generation;
* bounded rollout jobs with status, metrics, cancellation, and comparison;
* JSON resources for catalogs, world manifests, trajectories, and rollout
  status;
* prompts for robot inspection, motion validation, and trajectory generation.

The server is simulation-only. It does not provide real-device control,
arbitrary Python execution, or an agent call for every physics step. Streamable
HTTP belongs behind an authenticated gateway that owns authorization, approval,
and resource quotas.

State and safety model
----------------------

World, snapshot, trajectory, and rollout operations use explicit handles.
World mutations carry an optimistic ``scene_revision`` token, and generated
trajectories retain the revision observed during planning. Results include the
backend, frame, units, seed, diagnostics, and artifact metadata needed by a
Host to decide whether a result is usable.

A collision response reports the scope that was actually checked. Joint-limit
validation and geometric reachability do not establish collision-free motion.
Hosts should validate a trajectory immediately before execution, request user
intent for state-changing operations, and destroy worlds after a workflow
finishes.

Architecture
------------

.. mermaid::

   flowchart LR
      Host[MCP Host] -->|stdio JSON-RPC| Server[embodichain.mcp.server]
      Server --> Service[EmbodiChainMCPService]
      Service --> Backend[SimulationBackend]
      Backend --> Sim[SimulationManager or InMemory backend]
      Service --> Resources[Resources and handles]

The public Python entry point is ``embodichain.mcp``. Inject
``InMemorySimulationBackend`` for protocol tests; the default server creates a
headless ``SimulationManagerBackend`` lazily.

Continue with
-------------

* :doc:`/guides/mcp` — install the extra, register Codex, run the Franka demo,
  and troubleshoot common calls.
* :doc:`/api_reference/embodichain/embodichain.mcp` — public Python API.
* ``examples/mcp/codex_trajectory_demo/`` — complete stdio client and editable
  scene/trajectory inputs.
