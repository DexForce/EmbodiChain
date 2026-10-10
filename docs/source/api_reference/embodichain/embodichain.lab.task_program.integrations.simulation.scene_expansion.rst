embodichain.lab.task_program.integrations.simulation.scene_expansion
====================================================================

``SimulationSceneExpansionHost`` provides physical preparation and initial-state
restoration for the single-row Task Program scene expansion API. It supports
pose changes to existing rigid objects while capturing the supported rigid,
robot, and articulation state of a fixed scene.

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.scene_expansion

.. autosummary::
   :nosignatures:

   SimulationSceneExpansionHost
   SimulationSceneInitialState

Reset ordering and recording
----------------------------

The callable host passes a trusted zero-argument ``scene_expansion_prepare``
callback to ``BaseEnv.reset``. The callback runs after normal reset and reset
events, before physical-objective initialization and the final observation.
``EmbodiedEnv.reset`` then clears the old bridge and seeds recording from that
observation. The hook is removed from forwarded options, so it does not become
an observation or event-functor argument. A preparation hook requires a complete
B=1 reset.

The host preflights entity identity and state capture, applies arena-local rigid
poses, clears moved-body dynamics, and advances the configured number of physics
substeps for settling. Settling is preparation rather than a recorded expert
action. Sensors and observation history are reset again after state changes;
the initial validator then checks the actual settled scene. Subsequent Task
Program bridge creation constructs fresh providers and a fresh motion generator
with current collision bindings, including when the new episode starts at the
same clock value as the previous one.

Owned initial states
---------------------

``capture_initial_state`` and the host preparation callback produce immutable
numeric copies of supported entity state. The snapshot contains rigid poses and
velocities, articulation/robot roots, full joint positions and velocities,
controller position/velocity targets, joint forces, backend, control period,
and caller-supplied parent scene provenance. Obtain snapshots from the host;
their ownership token is part of restoration validation.

The content-derived ``initial_state_id`` excludes temporary physical slots.
``to_metadata`` exports the state into JSON-compatible episode provenance.
Asset contents and physical parameters must remain fixed for the host lifetime.
Entity replacement, changed joint names/body types, incompatible backend/control
period, and foreign-host snapshots are rejected before restoration writes.
Rigid groups, deformables, and explicit rigid constraints are unsupported.

``restore_initial_state`` uses the same reset hook, restores saved physical and
controller values. Measured joint positions use absolute tolerance ``1e-4``
because the public state setter clamps small native joint-limit excursions;
poses, velocities, controller targets, and efforts retain tolerance ``1e-5``.
Quaternion signs are treated as equivalent. It resets sensor/observation
history and runs the initial validator again. Restoration does not settle a
second time. The mandatory ``scene.restore_state`` check records round-trip
errors for measured joint positions and other fields, together with the source
initial-state ID. Numerical backend round trips can change the actual state's
content hash while passing these tolerances. These checks do not relax task
acceptance criteria or the robot's joint limits.

This is an episode initial-state snapshot. Clearing solver history and external
forces does not reproduce a backend's complete mid-trajectory checkpoint.

Using the concrete host
-----------------------

The application supplies task-specific initial and measured checks:

.. code-block:: python

   from collections.abc import Callable

   from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
   from embodichain.lab.sim.motion.expansion import ValidationResult
   from embodichain.lab.sim.scene_expansion import SceneVariant
   from embodichain.lab.task_program.integrations.scene_expansion import (
       SceneVariantEpisodeResult,
       execute_scene_variant,
   )
   from embodichain.lab.task_program.integrations.simulation.scene_expansion import (
       SimulationSceneExpansionHost,
       SimulationSceneInitialState,
   )

   def evaluate_and_capture(
       env: EmbodiedEnv,
       variant: SceneVariant,
       *,
       initial_validator: Callable[[], ValidationResult],
       measured_validator: Callable[[], ValidationResult],
   ) -> tuple[
       SimulationSceneExpansionHost,
       SimulationSceneInitialState,
       SceneVariantEpisodeResult,
   ]:
       host = SimulationSceneExpansionHost(
           env, initial_validator=initial_validator, settle_steps=8
       )
       result = execute_scene_variant(
           env,
           variant,
           prepare_scene=host,
           measured_validator=measured_validator,
       )
       snapshot = host.initial_state
       assert snapshot is not None
       env.reset(options={"save_data": result.accepted})
       return host, snapshot, result

For another execution from that exact saved initial state, reuse the owning host
and pass ``prepare_scene=lambda _: host.restore_initial_state(snapshot)`` to
``execute_scene_variant``. Evaluate the new execution independently. Workspace
checks may be included in ``initial_validator`` using
:mod:`embodichain.lab.task_program.integrations.simulation.workspace`; they
remain separate from collision planning and measured task acceptance.

Public interfaces
-----------------

.. autoclass:: SimulationSceneExpansionHost
   :members:

.. autoclass:: SimulationSceneInitialState
   :members:
