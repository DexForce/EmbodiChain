.. _guide_mcp:

Use EmbodiChain through MCP
===========================

This guide connects an MCP Host to EmbodiChain's local, project-level stdio
server. You will compose a modular URDF without starting a simulator, validate
the generated asset, optionally verify it in a temporary simulation world, and
then inspect the existing simulation and motion tools.

The protocol boundary lives in the top-level ``embodichain.mcp`` package.
Domain adapters own their existing implementations: the URDF adapter calls
the standalone assembly toolkit, while the simulation adapter owns physics,
rendering, kinematics, and high-frequency loops. The MCP client sends
coarse-grained tool calls and receives structured results.

Prerequisites
-------------

Install the protocol extra from the repository root:

.. code-block:: console

   $ python -m pip install -e '.[mcp]'

Install ``mcp-demo`` when you want to run the camera recording portion. It adds
ImageIO and FFmpeg support:

.. code-block:: console

   $ python -m pip install -e '.[mcp-demo]'

The URDF assembly example can run without creating a simulator world. The
optional simulation verification and Franka recording workflows use the
headless simulator. The ``mcp-demo`` extra adds the ImageIO and FFmpeg
packages required by recording. The protocol tests use the deterministic
``InMemorySimulationBackend`` and the URDF tests use local XML fixtures.

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
outside this guide because authorization, approval, and resource quotas must
be owned by that gateway.

The first tool calls
--------------------

Ask the Host to call ``server_info``, ``health``, and ``list_capabilities``. A
healthy response identifies the installed domain adapters without starting a
simulation. The normal URDF workflow is:

.. code-block:: text

   urdf_list_assets()
     -> registered asset identifiers
   urdf_compose(components=[arm, hand])
     -> assembly_id, model resource, manifest resource
   urdf_validate(assembly_id)
   urdf_verify_in_simulation(assembly_id)  # optional

The simulation workflow remains available when a world is needed:

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

Run the URDF assembly showcase
------------------------------

The repository example composes a UR5 arm and a DH PGC gripper through the
pure URDF adapter. It validates XML topology and copied mesh references before
optionally loading the generated model in a temporary headless world:

.. code-block:: console

   $ python examples/mcp/urdf_assembly_demo/run_demo.py

Use ``--no-verify`` to stop after the simulator-independent checks:

.. code-block:: console

   $ python examples/mcp/urdf_assembly_demo/run_demo.py --no-verify

Generated files are written under ``outputs/mcp/urdf/``. The MCP input uses
registered asset identifiers rather than arbitrary filesystem paths. The
generated model and manifest are available as:

.. code-block:: text

   embodichain://urdf/assemblies/{assembly_id}/model
   embodichain://urdf/assemblies/{assembly_id}/manifest
   embodichain://urdf/assemblies/{assembly_id}/validation

Run the Franka trajectory example
---------------------------------

The repository example contains a complete stdio client and editable JSON
inputs in ``examples/mcp/codex_trajectory_demo/``. Both commands below need
the simulator dependencies and cached Franka asset; ``--record`` additionally
needs the ``mcp-demo`` media dependencies. Run the trajectory workflow without
recording with:

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

For a simulator-free protocol check, run the deterministic contract tests:

.. code-block:: console

   $ pytest -q -c /dev/null tests/lab/mcp/test_service.py tests/lab/mcp/test_backend_contracts.py

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
   * - ``embodichain://urdf/assemblies/{assembly_id}/model``
     - Read a generated modular URDF model.
   * - ``embodichain://urdf/assemblies/{assembly_id}/manifest``
     - Read component references, artifact digest, and validation metadata.
   * - ``embodichain://urdf/assemblies/{assembly_id}/validation``
     - Read pure XML, topology, mimic, and mesh-reference checks.
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

Phase one exposes project adapters for asset composition, simulation, and
read-only analysis. It does not provide real-device control, arbitrary Python
execution, or an interactive per-step agent control loop. URDF composition
writes generated files under the adapter output root; simulation verification
uses a temporary world and removes it before returning. Collision results state
their exact scope; a joint-limit check or geometric reachability result does
not certify collision-free motion.

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
* ``examples/mcp/urdf_assembly_demo/README.md`` — modular arm and gripper
  composition workflow.
* :doc:`/tutorial/simulation/index` — simulation setup and scene fundamentals.
