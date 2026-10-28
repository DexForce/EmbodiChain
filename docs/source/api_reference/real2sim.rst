Real-to-simulation motion prototypes
====================================

These interfaces normalize explicitly annotated joint and TCP data into a
source-owned motion unit. Prepared bundles preserve source arrays, frame and
time decisions and a content digest. A prototype is a motion reference; the
host remains responsible for initial-state preparation, collision and physical
effect validation, execution and recording.

The source accepts qpos-only references for fixed-scene replay. Cartesian
transfer needs explicit TCP poses and supports tabletop XY/yaw changes. A
prototype generated from a synthetic reference remains marked synthetic.

.. automodule:: embodichain.data_pipeline.real2sim
   :members:
   :imported-members:
   :no-index:

.. automodule:: embodichain.data_pipeline.real2sim.contracts
   :members:
   :no-index:

.. automodule:: embodichain.data_pipeline.real2sim.normalize
   :members:
   :no-index:

.. automodule:: embodichain.data_pipeline.real2sim.prototype
   :members:
   :no-index:

.. automodule:: embodichain.data_pipeline.real2sim.io
   :members:
   :no-index:

Nominal motion is selected before native planning. The corresponding atomic
skill owns effect and command construction. Sampled collision evidence is
separate from a physical outcome; an unverified collision check cannot certify
the resulting rollout.

.. automodule:: embodichain.lab.sim.atomic_actions.nominal
   :members:
   :no-index:

.. automodule:: embodichain.lab.sim.atomic_actions.prototype_source
   :members:
   :no-index:

.. automodule:: embodichain.lab.sim.motion.expansion.real2sim
   :members:
   :no-index:

.. automodule:: embodichain.lab.sim.motion.expansion.mapped_timing
   :members:
   :no-index:

Mapped retiming scales each authorized free phase's total elapsed time to the
next control interval while leaving protected event neighborhoods and
contact-phase durations fixed. If those constraints cannot fit the requested
duration, it rejects the proposal instead of changing the physics timestep.

The synchronous B=1 sink accepts an explicitly validated rollout with its
terminal observation, source artifacts and action-to-plan trace. It closes
buffered LeRobot writers and checks the existing journal and artifact digests
before confirming a receipt. Callers must provide the task-specific acceptance
checks, collision and asset-compatibility evidence, and coordinate Gym reset.
The Gym submission helper can apply a receipt to an existing expansion session;
its coverage advances only after confirmation. A failed write retains its
payload for the existing idempotent retry and repair boundary.

``real2sim_lineage`` in episode metadata declares a scene revision, compiled
program SHA-256, source bundles and a digest-checked NPZ trace. The trace has
one ``step_call_index`` and ``step_plan_index`` for each submitted action;
``-1`` marks a runtime hold without a plan sample. For each ``call_<index>``,
``left``, ``right`` and ``weight`` map expanded-plan samples to the original
plan, while ``source_events`` and ``plan_events`` map exact phase boundaries.
The call's ``source_sample_offset`` separates added connection samples from
the normalized source, whose own source map stays in the archived prototype.
The sink rejects missing plan samples, unexpressible hard events and mismatched
artifact hashes. The B=1 scene integration can opt into action tracing through
the Task Program bridge. The caller must assemble the trace artifact, establish
that it describes submitted commands and supply independent physical evidence.

.. automodule:: embodichain.lab.gym.envs.managers.real2sim_sink
   :members:
   :no-index:

For fixed tabletop geometry, the sampled collision checker loads the complete
robot URDF collision geometry and requires an explicit scene-object pose at
every sample. Contact exemptions and maximum penetration are declared per
phase; adjacent robot links are excluded using URDF joint topology. The host
checks geometry fingerprints and projects commanded gripper closure to a
sampled first contact. Execution must still be checked against measured states;
these interfaces do not establish continuous-time or hardware feasibility.

.. automodule:: embodichain.lab.sim.motion.expansion.real2sim_collision
   :members:
   :no-index:

.. automodule:: embodichain.lab.task_program.integrations.simulation.real2sim_collision
   :members:
   :no-index:

The run-task owner captures actual observations and actions before submitting
them to the synchronous sink. An unconfirmed write retains a no-pickle staged
episode, so a retry uses the same payload instead of executing again.
When observed collision validation is required, the sink also verifies and
archives the measured T+1 object poses, fixed support pose and timestamps.

.. automodule:: embodichain.lab.gym.envs.managers.real2sim_collection
   :members:
   :no-index:

Configured tabletop collection supplies explicit layouts, recipes, collision
policy and per-joint motion limits. Its runtime owner coordinates scene setup,
source-bound planning, measured verification and one durable episode receipt.
Sequential Pick, MoveHeldObject and Place occurrences, including bounded
repeats, each require a pinned source. The host separately checks each Pick,
held interval and Place. A Pick from an observed support permits a change in
height and at most two degrees of tilt only after checking the support
footprint; lift remains gravity-aligned. Cartesian Place aligns its release
anchor to the semantic target and reports the applied correction.

.. automodule:: embodichain.lab.task_program.integrations.simulation.real2sim_collection
   :members:
   :no-index:

.. automodule:: embodichain.lab.sim.motion.expansion.operators
   :members: validate_motion_samples
   :no-index:
