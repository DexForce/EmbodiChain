embodichain.lab.task_program.integrations.simulation.scene_expansion
====================================================================

``SimulationSceneExpansionHost`` provides physical preparation and initial-state
restoration for explicitly selected Task Program environment rows. It supports
pose changes to existing rigid objects while capturing the supported rigid,
robot, and articulation state of a fixed scene.

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.scene_expansion

.. autosummary::
   :nosignatures:

   SimulationSceneExpansionHost
   SimulationSceneInitialState

Reset ordering and recording
----------------------------

The callable host passes a trusted zero-argument ``episode_prepare``
callback to ``BaseEnv.reset``. The callback runs after normal reset and reset
events, before physical-objective initialization and the final observation.
``EmbodiedEnv.reset`` then clears the old bridge and seeds recording from that
observation. The hook is removed from forwarded options, so it does not become
an observation or event-functor argument. The host passes its ordered selection
as ``reset_ids``. Callback signatures and all requested rows are checked before
seeding or resetting state. ``initial_validator`` receives a physical row ID;
preparation and restoration return a mapping of independent row results.

The host preflights entity identity and state capture, applies arena-local rigid
poses, clears moved-body dynamics, and advances the configured number of physics
substeps for settling. Any settling requires the complete World selection;
partial-row hosts reject a settling policy before mutation. The host advances
physics once for the batch, never once per row. Settling is preparation rather
than a recorded expert action. Sensors and observation history are reset again after state changes;
current contact samples are refreshed without advancing physics or contact
history. The initial validator then checks the actual settled scene. Empty
initial checks remain unavailable even when restoration or convergence checks
pass. Subsequent Task
Program bridge creation constructs fresh providers and a fresh motion generator
with current collision bindings, including when the new episode starts at the
same clock value as the previous one.

Owned initial states
---------------------

``capture_initial_states`` and the host preparation callback produce immutable
numeric copies of supported entity state for every selected row. Supply
``parent_scene_ids`` keyed by exactly those rows. ``initial_states`` exposes the
last snapshots as an immutable row mapping. Each snapshot contains rigid poses and
velocities, articulation/robot roots, full joint positions and velocities,
controller position/velocity targets, joint forces, backend, control period,
and caller-supplied parent scene provenance. Obtain snapshots from the host;
their ownership token and original row assignment are part of restoration
validation. Snapshots cannot be swapped between rows even when their content
identities match.

The content-derived ``initial_state_id`` excludes temporary physical slots.
``to_metadata`` exports the state into JSON-compatible episode provenance.
Asset contents and physical parameters must remain fixed for the host lifetime.
The host measures mass, inertia, local COM, rigid friction/damping and
articulation drive/limit fields at construction. It checks them before capture
and restoration, and after reset events. Drift beyond absolute/relative
``1e-6`` is rejected. Reset-time parameter randomization requires independent
initial-state preparation and a new host with those fixed parameters.
Entity replacement, changed joint names/body types, incompatible backend/control
period, and foreign-host snapshots are rejected before restoration writes.
Rigid groups, deformables, and explicit rigid constraints are unsupported.

``restore_initial_states`` accepts exactly the selected row snapshots. It
preflights every row before any reset write, then uses the same reset hook to
restore physical and controller values. Unselected rows are preserved. Measured joint positions use absolute tolerance ``1e-4``
because the public state setter clamps small native joint-limit excursions.
The host explicitly passes this allowance to the shared restoration service.
The shared service's default remains strict, controller position targets remain
strict, and larger excursions fail before reset writes.
Poses, velocities, controller targets, and efforts retain tolerance ``1e-5``.
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

   from collections.abc import Callable, Mapping

   from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
   from embodichain.lab.sim.motion.expansion import ValidationResult
   from embodichain.lab.sim.scene.expansion import SceneVariant
   from embodichain.lab.task_program.integrations.scene_expansion import execute_scene_variants
   from embodichain.lab.task_program.integrations.simulation.scene_expansion import (
       SimulationSceneExpansionHost,
   )

   def evaluate_and_capture(
       env: EmbodiedEnv,
       variants: Mapping[int, SceneVariant],
       *,
       initial_validator: Callable[[int], ValidationResult],
       measured_validator: Callable[[int], ValidationResult],
   ):
       # With settling enabled, variants must cover the complete World.
       host = SimulationSceneExpansionHost(
           env, env_ids=tuple(variants),
           initial_validator=initial_validator, settle_steps=8,
       )
       results = execute_scene_variants(
           env, variants, prepare_scene=host, measured_validator=measured_validator,
       )
       snapshots = host.initial_states
       assert snapshots is not None
       env.reset(options={
           "reset_ids": list(variants),
           "save_data": True,
           "commit_env_ids": [row for row, result in results.items() if result.accepted],
       })
       return host, snapshots, results

For another execution from those saved initial states, reuse the owning host and
pass ``prepare_scene=lambda rows: host.restore_initial_states(snapshots)`` to
``execute_scene_variants``. Evaluate the new execution independently. Workspace
geometry is provided by :mod:`embodichain.lab.sim.scene.workspace`; its
``SceneCheckReport`` is separate from execution ``ValidationResult`` evidence.
The caller owns the explicit conversion and task-specific acceptance rules.

Public interfaces
-----------------

.. autoclass:: SimulationSceneExpansionHost
   :members:

.. autoclass:: SimulationSceneInitialState
   :members:
