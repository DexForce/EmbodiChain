Simulation Framework
====================

.. currentmodule:: embodichain.lab.sim

The ``embodichain.lab.sim`` package is EmbodiChain's backend-neutral runtime
for building and running embodied scenes. It brings together scene assets,
physics, rendering, sensors, robot kinematics, motion generation, and reusable
manipulation actions behind one batched API. A
:class:`SimulationManager` owns the simulation world, the scene declaration and
registries, the selected physics backend, and the explicit update/reset
lifecycle.

Scene composition is configuration-driven: asset and sensor configs are
declared first, translated through Spawn, and exposed as stable EmbodiChain
facades after preparation.

The simulation layer is deliberately separate from task orchestration. It owns
physical state, simulation time, render-state publication, and sensor readback;
Gym environments own the ``reset``/``step`` contract and may own episode
recording. Task Program owns semantic sequencing and verification contracts,
while application code owns learning and other orchestration. The same scene
can therefore be used by a standalone script, a vectorized learning
environment, a data-generation pipeline, or a declarative Task Program
integration.

.. important::

   Declare the complete initial scene before calling
   :meth:`SimulationManager.prepare`. Read state and metadata only after that
   boundary, and advance time explicitly with :meth:`SimulationManager.update`.
   The manager does not advance physics in the background.

Architecture
------------

Read the runtime from configuration and scene assembly down to control and
observation:

.. code-block:: text

    SimulationManagerCfg + asset/sensor configs
                         |
                         v
    SimulationManager
    |-- SpawnScene -> finalized Scene (topology and native handles)
    |-- PhysicsBackend -> Default or Newton
    |-- registries -> assets, sensors, materials, gizmos, and markers
    |-- renderer -> native DexSim window or optional Viser browser runtime
    `-- explicit prepare / update / reset / destroy lifecycle
                         |
                         v
    batched scene state and sensor observations
                         |
        +----------------+----------------+
        v                                 v
    motion capabilities              Gym / Task Program
    (solvers, planners,              (actions, rewards, semantic
     workspace, execution)            calls, recording, evaluation)
        |                                 |
        +-------------> controllers <-----+

``SpawnScene`` and ``PhysicsBackend`` are implementation boundaries owned by
the manager; applications should use the manager and the public asset/sensor
APIs rather than reaching into native scene builders or backend handles.

Core components
---------------

.. list-table::
   :header-rows: 1
   :widths: 23 34 43

   * - Component
     - Responsibility
     - Public boundary
   * - ``SimulationManager``
     - Creates the world, lays out ``num_envs`` replicated arenas, coordinates scene
       preparation, imports/exports USD, advances physics, optionally profiles
       updates, publishes render state, and tears down native resources.
     - Use ``add_*``/``get_*`` methods and :meth:`SimulationManager.prepare`,
       :meth:`SimulationManager.update`, and ``reset``/``destroy``; task logic
       does not belong here.
   * - Scene assets
     - Rigid objects and groups, articulations, robots, volume/surface
       deformables, lights, materials, rigid constraints, gizmos, and markers.
     - Configure with the types in ``embodichain.lab.sim.cfg``. Dynamic state
       and control views are batched by arena and become metadata-complete after
       preparation.
   * - Sensors
     - Camera, stereo-camera, and contact observations, including image
       modalities, disparity, segmentation, and contact query data.
     - Attach sensors through the manager; consume their batched tensors and
       capability metadata instead of backend-native buffers.
   * - Motion
     - Kinematic solving, trajectory planning, workspace analysis, timed
       playback, and bounded trajectory augmentation.
     - Import from ``embodichain.lab.sim.motion`` and its owning subpackages;
       the parent package loads them lazily and does not flatten their APIs.
   * - Atomic Actions
     - Typed manipulation plans and runtime commands such as move, pick, place,
       and coordinated actions.
     - Actions compose robot and motion resources but return plans/commands;
       the caller or an environment performs stepping and endpoint I/O.
   * - Visualization
     - Native DexSim rendering, camera capture, point-cloud/debug overlays, and
       optional browser-based Viser scene inspection and controls.
     - Select rendering with ``render_cfg`` and opt into Viser with
       ``visualization.backend="viser"``. Viser forces headless operation and
       is mutually exclusive with the native window; visualization does not
       own physics time.
   * - Gym and Task Program integrations
     - Convert the simulation runtime into vectorized episodes, expert
       trajectory workflows, or provider-independent semantic programs.
     - ``BaseEnv``/``EmbodiedEnv`` own episode timing and manager functors;
       Task Program compiles Semantic Calls and lowers them to Atomic Skills.

Asset declarations may be procedural or source-backed (USD/URDF). Use
``asset_physics_mode="preserve"`` to retain source physics or
``asset_physics_mode="overlay"`` to apply the typed EmbodiChain properties;
the object pages document the backend-specific fields and precedence.

Physics backends
----------------

The concrete type of ``SimulationManagerCfg.physics_cfg`` selects the physics
backend. Rendering is configured independently by ``render_cfg``; choosing
Newton does not select a different camera renderer. ``default`` and ``newton``
are the only public backend identifiers. ``headless=True`` disables the native
window while still permitting configured offscreen camera rendering.

The selected physics config owns its default compute device (``cpu`` for
Default and ``cuda:0`` for Newton). An explicit ``SimulationManagerCfg.device``
or legacy ``sim_device`` value overrides that choice; ``gpu_id`` primarily
selects the rendering GPU and supplies the index for an unqualified CUDA
device. Environment tensors derive from ``sim.device``; do not introduce a
second tensor-device selector in task configuration.

.. list-table::
   :header-rows: 1
   :widths: 17 25 29 29

   * - Backend
     - Configuration
     - Good fit
     - Current boundary
   * - Default
     - ``DefaultPhysicsCfg``
     - CPU or GPU rigid-body and articulation scenes, including manager-owned
       rigid constraints and the established Constraint Dynamics path.
     - Volume and surface deformables are not implemented by this backend.
   * - Newton
     - ``NewtonPhysicsCfg``
     - GPU rigid, particle, and deformable scenes, solver-specific experiments,
       and articulated/deformable coupling through DexUni.
     - Support depends on the resolved solver, device, finalized scene, and
       requested operation. Manager-owned rigid constraints are unsupported by
       Newton; the differentiable route is the documented ``semi_implicit``
       kinematics bridge.

``physics: default`` and ``physics: newton`` in a Gym configuration identify
the integration, not a blanket capability guarantee. Newton's automatic solver
is resolved only after the complete scene is finalized, so qualify a runtime
after :meth:`SimulationManager.prepare` (for example, by inspecting
``sim.physics.solver_type`` and the sensor capabilities). See
:doc:`sim_manager/physics/index` for the capability matrix,
:doc:`sim_manager/physics/default` for the Default path, and
:doc:`sim_manager/physics/newton` for Newton, DexUni, and runtime-qualification
details.

For a runnable Gym deployment, the environment file or selected component owns
the ``physics`` value and its matching ``physics_config``; launcher
``--physics`` can confirm that choice but cannot switch it. Use separate
configuration files for backend-specific deployments.

Lifecycle and time model
------------------------

All backends share one declaration-to-step contract:

.. code-block:: text

    construct SimulationManager
        -> declare initial assets and sensors
        -> register pre-prepare Newton controls (when needed)
        -> sim.prepare()
        -> apply targets / read state
        -> sim.update(...)
        -> reset selected arenas or destroy the manager

Call :meth:`SimulationManager.prepare` after the initial ``add_*`` calls and
before reading joint/link metadata, native handles, or state that depends on a
finalized topology. Preparation is idempotent for an unchanged scene. It
commits the Spawn topology, initializes backend runtime buffers, binds the
declared EmbodiChain facades in place, restores camera attachments, and
publishes the initial render state. Newton defers physical model construction
until this boundary; Default may materialize some native entities earlier but
still prepares its GPU/runtime buffers here. Preparation does not advance
simulation time. If a supported topology change is made later, call
``prepare()`` again and reacquire any backend views. The older
``init_gpu_physics()`` and ``finalize_newton_physics()`` names remain
compatibility aliases; new code should use ``prepare()``.

Physics advances only through :meth:`SimulationManager.update`; there is no
background physics polling path. ``BaseEnv`` inserts the same readiness and
update calls into ``env.reset``/``env.step``. A standalone caller can register
Newton kinematic joint or nodal trajectories with
``register_kinematic_joint_trajectory()`` or
``register_kinematic_nodal_trajectory()``, and contact-material schedules with
``register_contact_material_schedule()`` or
``register_particle_contact_material_schedule()`` before preparation; these
controls are expanded per arena by the manager and do not require access to the
private Spawn builder. The former ``set_manual_update()`` switch is no longer
public; explicit updates are the manager's only stepping mode.

Reset is arena-scoped: selected environment rows are restored from their
captured physical defaults before the environment's reset events and task
initialization run. Keep episode bookkeeping in ``BaseEnv`` or the Task Program
integration; the manager only restores simulation state and resources.

The three useful time scales are:

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Interval
     - Definition
     - Meaning
   * - Physics step
     - ``physics_dt``
     - One physics step; ``update(step=N)`` advances ``N`` such intervals.
   * - Environment control step
     - ``physics_dt * sim_steps_per_control``
     - One action/observation interval in a Gym environment.
   * - Newton solver substep
     - ``physics_dt / num_substeps``
     - Integration refinement inside one physics step; it does not change the
       environment control period.

Data and coordinate conventions
-------------------------------

Simulation entities and sensor outputs use a leading arena dimension. With
``num_envs=B``, common layouts are ``[B, ...]`` for rigid state,
``[B, links, ...]`` for articulations, and ``[B, nodes, ...]`` for deformable
state. Spawn replication isolates each arena's per-environment rigid
collisions, including dynamic, kinematic, static, and articulation-link
shapes; global physics resources explicitly marked non-per-environment remain
shared. Use the public UID and name queries to select rows, joints, links, and
sensors.

All EmbodiChain-owned public poses and quaternions use ``xyzw`` ordering. A
7-dimensional pose is ``(px, py, pz, qx, qy, qz, qw)`` and the identity is
``(0, 0, 0, 1)``. Backend adapters perform any native-order conversion at the
boundary. This convention applies to object and link state, robot FK/IK,
sensor offsets, semantic poses, and task configuration.

Volume and surface deformables share the ``DeformableObject`` state contract
(``nodal_pos_w``, ``nodal_vel_w``, and ``nodal_state_w``), while their mesh
topologies and configuration remain distinct. In the current integration they
are Newton particle sets and require a supported CUDA solver; the Default
backend intentionally rejects unsupported deformable declarations.

Motion and task integration
---------------------------

The motion namespace is organized by capability rather than by one monolithic
planner:

.. list-table::
   :header-rows: 1
   :widths: 29 71

   * - Package
     - Role
   * - ``motion.solvers``
     - Forward, inverse, and differential kinematics for robot and articulation
       chains.
   * - ``motion.planners``
     - Joint-space and Cartesian paths, collision-aware planning, interpolation,
       and time parameterization (including TOPPRA, trapezoidal/Double-S,
       Neural, and optional cuRobo backends).
   * - ``motion.motion_generator``
     - Stateful facade that resolves the configured strategy into validated,
       timed trajectories for direct callers and Atomic Skills.
   * - ``motion.workspace``
     - Reachability analysis, workspace sampling, visualization, and caches.
   * - ``motion.execution``
     - Fixed-cadence playback that retimes planner output to an integer number
       of unchanged physics steps.
   * - ``motion.expansion``
     - Host-independent trajectory candidates, allowed operators, coverage, and
       bounded generation bookkeeping.

Use ``embodichain.lab.sim.motion.*`` for these capabilities. The former
``embodichain.lab.sim.solvers``, ``embodichain.lab.sim.planners``, and
``embodichain.lab.sim.workspace`` paths are not compatibility packages.

Planner results that contain positions carry per-waypoint arrival intervals.
:func:`~embodichain.lab.sim.motion.execution.play_joint_trajectory` can retime
those results to a fixed command period without changing ``physics_dt``; the
executed period must be an integer multiple of the physics period. The
shared timing utilities live under ``embodichain.compute.trajectory``, and the
expansion package deliberately does not step Gym or persist datasets.
Physical initial-state restoration, rollout execution, task validation, and
durable storage belong to an explicit host integration.

For higher-level manipulation, Atomic Actions provide a typed boundary:

.. code-block:: text

    grounded goal / Semantic Call
        -> AtomicActionEngine
        -> MotionGenerator and planner/solver resources
        -> ActionPlan or RuntimeCommandFrame
        -> application endpoint / simulation controller
        -> sim.update() and measured state

``plan`` and ``compile`` are planning-only entry points; ``start``/``tick``
supports observed execution and bounded recovery. None of these APIs advances
the simulator by itself. Task Program sits one level above this boundary:

.. code-block:: text

    JSON/YAML/Python Task Program
        -> decode / validate / compile
        -> Semantic Calls
        -> Atomic Skills
        -> Gym or simulation integration

Task Program owns semantic identity, effects, evidence, sequencing, settling,
and barriers; the simulation runtime remains responsible for physical state and
time. Semantic Calls are provider-independent intent and become executable only
through an integration that invokes Atomic Skills.

Typical workflow
----------------

1. Choose a ``DefaultPhysicsCfg`` or ``NewtonPhysicsCfg`` and configure the
   renderer/device independently.
2. Construct a :class:`SimulationManager` from
   :class:`SimulationManagerCfg` and declare robots, articulations, objects,
   deformables, lights, and sensors.
3. Register any Newton runtime controls that must be part of the initial scene.
4. Call :meth:`SimulationManager.prepare` once the initial topology is
   complete.
5. Read batched state, run FK/IK or
   :class:`~embodichain.lab.sim.motion.motion_generator.MotionGenerator`, and
   apply action or controller targets.
6. Advance explicitly with :meth:`SimulationManager.update`, then collect
   state, sensor tensors, render frames, or trajectory evidence.
7. For learning or declarative tasks, place the same manager logic under
   ``EmbodiedEnv`` or the Task Program integration and let that layer own
   episode/segment bookkeeping.
8. Call ``destroy(exit_process=False)`` for embedded/test lifecycles, then call
   ``SimulationManager.flush_cleanup_queue()`` after local scene references are
   released.

Choosing where to start
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Goal
     - Start here
   * - Create a scene, choose a backend, or understand explicit stepping
     - :doc:`sim_manager` and :doc:`sim_manager/physics/index`
   * - Configure rigid objects, articulations, robots, deformables, or USD
     - :doc:`sim_assets` and the object pages underneath it
   * - Add RGB, depth, stereo, segmentation, or contact observations
     - :doc:`sensors/index`
   * - Inspect a headless or remote scene in a browser
     - :doc:`viser_visualization`
   * - Tune native rendering, DLSS, or frame timing
     - :doc:`sim_manager/rendering/index`
   * - Solve kinematics or generate a timed robot trajectory
     - :doc:`motion/index`, :doc:`motion/solvers/index`, and
       :doc:`motion/planners/index`
   * - Analyze reachability or sample a robot workspace
     - :doc:`/features/workspace_analyzer/index`
   * - Play, augment, or account for expert trajectories
     - :doc:`motion/index` and :doc:`/tutorial/trajectory_planning`
   * - Record/replay expert trajectories or evaluate a policy
     - :doc:`/overview/gym/index` and :doc:`/guides/policy_evaluation`
   * - Build reusable scripted manipulation
     - :doc:`atomic actions <atomic_actions/index>`
   * - Share canonical scene identity with semantic calls and planner obstacles
     - :doc:`/overview/task_program/scene_registry`
   * - Bind an embodiment, motion resources, and execution policies
     - :doc:`/overview/task_program/robot_profiles`
   * - Author semantic calls, effects, settling, or parallel barriers
     - :doc:`/overview/task_program/index`

Documentation quality notes
---------------------------

Pages in this section should follow the same declaration, preparation,
control, stepping, and observation flow. State tensor shapes and coordinate
frames explicitly, identify backend-specific capability boundaries, and link
to the owning asset, sensor, solver, or planner page instead of duplicating API
tables.

See Also
--------

.. toctree::
   :maxdepth: 1

   sim_manager.md
   sim_assets.md
   sensors/index
   viser_visualization.md
   motion/index
   atomic_actions/index
