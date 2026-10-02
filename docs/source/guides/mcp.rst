.. _tutorial_mcp:

Use EmbodiChain through MCP
===========================

This guide connects an MCP Host to EmbodiChain's local, simulation-only
stdio server. You will inspect the server, create a simulation world, load a
scene, generate and validate a joint trajectory, and optionally record the
trajectory with an offscreen camera.

The protocol boundary lives in the top-level ``embodichain.mcp`` package. The
simulation backend owns physics, rendering, kinematics, and high-frequency
loops; the MCP client sends task-level tool calls and receives structured
results.

Prerequisites
-------------

Install the protocol extra from the repository root:

.. code-block:: console

   $ python -m pip install -e '.[mcp]'

Install ``mcp-demo`` when you want to run the camera recording portion. It adds
ImageIO and FFmpeg support:

.. code-block:: console

   $ python -m pip install -e '.[mcp-demo]'

The recording workflow also needs the simulator dependencies and the cached
Franka asset used by the demo. The protocol tests use the deterministic
``InMemorySimulationBackend`` and do not start a simulator.

Start the stdio server
----------------------

The unified CLI starts the server with the stdio transport:

.. code-block:: console

   $ python -m embodichain mcp

An MCP Host normally starts this command as a child process. For Codex, register
the command once:

.. code-block:: console

   $ codex mcp add embodichain -- python -m embodichain mcp
   $ codex mcp list

The phase-one server intentionally serves stdio only. A remote HTTP gateway is
outside this tutorial because authorization, approval, and resource quotas must
be owned by that gateway.

The first tool calls
--------------------

Ask the Host to call ``server_info``, ``health``, and ``list_capabilities``. A
healthy response identifies the protocol phase and backend scope without
starting a simulation. The normal world workflow is:

.. code-block:: text

   create_world(backend="default", seed=7)
     -> world_id, scene_revision=0, seed
   load_task_or_scene(world_id, scene=...)
     -> scene_revision=1
   get_world_state(world_id)
   solve_ik / forward_kinematics / plan_motion
   validate_trajectory(world_id, robot_id, positions)
   execute_trajectory(trajectory_id)
   destroy_world(world_id)

Every operation after ``create_world`` carries an explicit ``world_id``. State
changing calls can include ``expected_scene_revision``. The server rejects a
stale revision before touching the world, so a Host should read the latest
world state after a revision conflict and retry with the new value.

Run the Franka example
----------------------

The repository example contains a complete stdio client and editable JSON
inputs in ``examples/mcp/codex_trajectory_demo/``. Run the protocol-only part
with:

.. code-block:: console

   $ python examples/mcp/codex_trajectory_demo/run_demo.py

Add ``--record`` to execute the trajectory through the offscreen camera and
write RGB and depth artifacts:

.. code-block:: console

   $ python examples/mcp/codex_trajectory_demo/run_demo.py --record

The client first reads the built-in
``embodichain://demos/codex-trajectory/scene`` resource and checks it against
``scene.json``. It then creates a seeded world, reads the current robot qpos,
generates a trajectory from ``trajectory.json``, and reports the trajectory
resource URI. With ``--record``, the server applies the trajectory and writes:

.. code-block:: text

   outputs/mcp/codex_franka.mp4
   outputs/mcp/codex_franka.depth.npz

The demo keeps the physics loop inside the backend. The Host chooses waypoints
and makes one task-level recording call instead of issuing a tool call for each
physics step.

Resources and handles
---------------------

The Host can read these resources after the corresponding handle exists:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Resource
     - Purpose
   * - ``embodichain://tasks/catalog``
     - Discover installed task metadata and deployments.
   * - ``embodichain://robots/{robot_id}/config``
     - Inspect a robot record and its capabilities.
   * - ``embodichain://worlds/{world_id}/manifest``
     - Read the current world state and scene revision.
   * - ``embodichain://trajectories/{trajectory_id}``
     - Inspect generated waypoints, samples, and validation.
   * - ``embodichain://runs/{run_id}/status``
     - Follow a bounded asynchronous rollout.
   * - ``embodichain://runs/{run_id}/metrics``
     - Read terminal or partial rollout metrics.

World, snapshot, trajectory, and rollout handles are scoped to the service
instance. A Host should save the handle together with the reported
``scene_revision`` and backend metadata, and destroy worlds after a workflow
finishes.

Safety and scope
----------------

Phase one exposes simulation and read-only analysis. It does not provide
real-device control, arbitrary Python execution, or an interactive per-step
agent control loop. Collision results state their exact scope; a joint-limit
check or geometric reachability result does not certify collision-free motion.

Treat ``execute_trajectory`` and ``record_trajectory`` as state-changing
operations. Validate the trajectory immediately before execution and ask for
explicit user intent before executing a motion when the user only requested a
plan or validation.

Troubleshooting
---------------

``MCP support requires the optional dependency``
   Install ``.[mcp]`` in the same Python environment used by the Host.

``Unknown robot_id``
   Read the world state after ``load_task_or_scene`` and use the robot ID that
   the backend registered. A successful world creation does not load a scene.

Camera recording fails to start
   Install ``.[mcp-demo]``, verify that the scene contains an enabled camera,
   and check that the Franka asset and simulator runtime are available.

``scene_revision mismatch``
   Refresh ``embodichain://worlds/{world_id}/manifest`` and retry the operation
   with the returned revision. Do not reuse a trajectory generated for an older
   scene.

Further reference
-----------------

* :doc:`/api_reference/embodichain/embodichain.mcp` — public Python API.
* ``examples/mcp/codex_trajectory_demo/README.md`` — complete Host prompt and
  local demo notes.
* :doc:`/tutorial/simulation/index` — simulation setup and scene fundamentals.
