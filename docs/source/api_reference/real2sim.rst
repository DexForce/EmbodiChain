Real-to-simulation
==================

Core motion prototypes
----------------------

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

Original DROID recordings retain measured joint states, optional measured
velocities, submitted commands and both acquisition clocks as separate streams.
The adapter requires explicit manual phases and base/flange/TCP calibration.
Its reported Cartesian poses are FK-derived, not independent TCP measurements.
Typed evidence archives the original streams alongside the self-contained HDF5;
missing velocities are not reconstructed as observations.

.. automodule:: embodichain.data_pipeline.real2sim.droid
   :members:

Read-only replay export verifies each episode's matching complete recorder
journal, receipt identities, source bundles and attachment digests. It retains
failed attempts and creates no new collection or submission identities.

.. automodule:: embodichain.data_pipeline.real2sim.replay
   :members:

Nominal motion is selected before native planning. The corresponding atomic
skill owns effect and command construction. Sampled collision evidence is
separate from a physical outcome; an unverified collision check cannot certify
the resulting rollout.

.. automodule:: embodichain.lab.sim.atomic_skills.nominal
   :members:
   :no-index:

.. automodule:: embodichain.lab.sim.atomic_skills.prototype_source
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

Gripper contact projection is explicitly selected by the host. Independent
prismatic fingers retain their existing projection. Coupled mimic grippers
advance one driver and all followers together until the first contact of any
declared finger link. Measured coupling tolerances never relax command coupling.
Internal mechanism collision pairs and their depth bounds must be named.

.. automodule:: embodichain.lab.sim.motion.expansion.gripper_contact
   :members:

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
An optional directed yaw tolerance checks observed object +X against the
requested object pose, including solver error. The default remains disabled.
``retime_phase_allowlist`` can narrow source and skill time permissions; a short
protected free phase can retain its original grid while other phases stretch.

.. automodule:: embodichain.lab.task_program.integrations.simulation.real2sim_collection
   :members:
   :no-index:

.. automodule:: embodichain.lab.sim.motion.expansion.operators
   :members: validate_motion_samples
   :no-index:

.. _real2sim-video-adaptation:

Video reconstruction and trajectory adaptation
----------------------------------------------

The following interfaces add image preparation, Scene Engine registration, and
recorded-trajectory adaptation. They keep the original robot and use a separate
registered-scene contract. They do not replace the normalized motion prototype
or Atomic Skill source-binding interfaces above. See the
:doc:`workflow guide </features/generative_sim/real2sim>` and the
`example's How to run <https://github.com/DexForce/EmbodiChain/blob/main/examples/real2sim/README.md#how-to-run>`_
for the staged command-line workflow.

Recording and scene inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~

``ReconstructionInput`` combines measured joints, an unchanged robot model and
explicit camera observations. Camera transforms are optical-to-robot-base;
depth is in metres on the original first-image grid. ``RegisteredScene`` holds
generated geometry and sampled vertices in that frame. The prepared loader
reads an existing recording directory; it is not a raw-dataset downloader.

.. autosummary::

   embodichain.data_pipeline.real2sim.adaptation.inputs.CameraObservation
   embodichain.data_pipeline.real2sim.adaptation.inputs.RobotModel
   embodichain.data_pipeline.real2sim.adaptation.inputs.ReconstructionInput
   embodichain.data_pipeline.real2sim.adaptation.inputs.RegisteredScene
   embodichain.data_pipeline.real2sim.adaptation.inputs.RefinedTrajectory
   embodichain.data_pipeline.real2sim.adaptation.prepared.load_prepared_source
   embodichain.data_pipeline.real2sim.adaptation.prepared.load_registered_scene

.. automodule:: embodichain.data_pipeline.real2sim.adaptation.inputs
   :members:

.. automodule:: embodichain.data_pipeline.real2sim.adaptation.prepared
   :members:

Image preparation and scene generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reference frames are selected by the caller from the same fixed camera.
Observed pixel recovery records donor frames and unresolved regions. Image
editing then removes robot/background/lighting clutter at the provider's image
resolution. Its generated hidden surfaces are not measured evidence. The scene
wrapper passes that image to the complete native Scene Engine.

.. autosummary::

   embodichain.data_pipeline.real2sim.adaptation.video_fusion.RecoveryCfg
   embodichain.data_pipeline.real2sim.adaptation.video_fusion.RecoveredSceneImage
   embodichain.data_pipeline.real2sim.adaptation.video_fusion.decode_video_frames
   embodichain.data_pipeline.real2sim.adaptation.video_fusion.recover_scene_image
   embodichain.gen_sim.real2sim.image_edit.ImageEditor
   embodichain.gen_sim.real2sim.image_edit.HttpImageEditor
   embodichain.gen_sim.real2sim.image_edit.edit_scene_image
   embodichain.gen_sim.real2sim.scene.prepare_scene_input
   embodichain.gen_sim.real2sim.scene.reconstruct_scene
   embodichain.gen_sim.real2sim.scene.reconstruct_video
   embodichain.gen_sim.real2sim.scene.register_scene_export

.. automodule:: embodichain.data_pipeline.real2sim.adaptation.video_fusion
   :members:

.. automodule:: embodichain.gen_sim.real2sim.image_edit
   :members:

.. automodule:: embodichain.gen_sim.real2sim.scene
   :members:

Registration and explicit robot placement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Registration fits one shared similarity transform using original RGB-D evidence;
it does not fit individual object dimensions or regenerate meshes. Robot base
placement is an explicit pre-loop operation and leaves scene objects fixed.
Its conservative clearance search is not a task-success test.

.. autosummary::

   embodichain.data_pipeline.real2sim.adaptation.registration.register_scene
   embodichain.data_pipeline.real2sim.adaptation.placement.place_robot_base

.. automodule:: embodichain.data_pipeline.real2sim.adaptation.registration
   :members:

.. automodule:: embodichain.data_pipeline.real2sim.adaptation.placement
   :members:

Semantic annotation and bounded replay loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``run_prepared_real2sim`` owns the prepared-recording command workflow;
``run_agent_refinement`` accepts explicit inference and replay callbacks for
other hosts. The agent supplies canonical Task Program annotation, failure
locations, and bounded correction parameters. Separate half-open source
bindings are projected to inclusive numerical intervals only at the numerical
boundary. Annotation does not establish executable runtime bindings.

.. autosummary::

   embodichain.gen_sim.real2sim.execution.run_prepared_real2sim
   embodichain.gen_sim.real2sim.execution.refine_prepared_recording
   embodichain.gen_sim.real2sim.agent_refinement.run_agent_refinement
   embodichain.gen_sim.real2sim.trajectory_agent.CodexTrajectoryAgent
   embodichain.gen_sim.real2sim.annotation.ground_gripper_events
   embodichain.gen_sim.real2sim.annotation.annotate_trajectory
   embodichain.gen_sim.real2sim.annotation.bind_graph_intervals
   embodichain.gen_sim.real2sim.annotation.generate_annotated_graph
   embodichain.gen_sim.real2sim.annotation.validate_program_annotation
   embodichain.gen_sim.real2sim.annotation.bound_trajectory_phases

.. automodule:: embodichain.gen_sim.real2sim.execution
   :members:

.. automodule:: embodichain.gen_sim.real2sim.agent_refinement
   :members:

.. automodule:: embodichain.gen_sim.real2sim.trajectory_agent
   :members:

.. automodule:: embodichain.gen_sim.real2sim.annotation
   :members:

Numerical correction and replay evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Structured corrections specify local TCP translation/rotation and gripper
aperture/timing residuals. Numerical code produces commands on the source grid
using the feature-specific approximate IK. Residual and URDF-limit diagnostics
do not certify a feasible grasp. Replay readers expose measured joints, object
poses, contacts and timing without deciding task success.

.. autosummary::

   embodichain.data_pipeline.real2sim.adaptation.trajectory_corrections.CORRECTION_LIMITS
   embodichain.data_pipeline.real2sim.adaptation.trajectory_corrections.apply_trajectory_corrections
   embodichain.data_pipeline.real2sim.adaptation.trajectory_tools.TrajectoryTools
   embodichain.data_pipeline.real2sim.adaptation.trajectory_tools.interpolate_joint_knots
   embodichain.data_pipeline.real2sim.adaptation.trajectory_tools.validate_segment_revision
   embodichain.data_pipeline.real2sim.adaptation.trajectory_tools.read_replay_segment
   embodichain.data_pipeline.real2sim.adaptation.evidence.prepare_trajectory_evidence

.. automodule:: embodichain.data_pipeline.real2sim.adaptation.trajectory_corrections
   :members:

.. automodule:: embodichain.data_pipeline.real2sim.adaptation.trajectory_tools
   :members:

.. automodule:: embodichain.data_pipeline.real2sim.adaptation.evidence
   :members:

Native replay and media
~~~~~~~~~~~~~~~~~~~~~~~

Replay consumes a packaged registered scene and commanded joints. Assets and
source videos may be referenced by symlink. Runtime assets, diagnostics and
videos have separate output subdirectories. The comparison exporter uses the
recorded source-frame mapping and actual render times; rendering completion
does not establish task completion.

.. autosummary::

   embodichain.gen_sim.real2sim.replay.replay_recording
   embodichain.gen_sim.real2sim.replay.export_native_inputs
   embodichain.data_pipeline.real2sim.adaptation.media.export_comparison
   embodichain.data_pipeline.real2sim.adaptation.media.create_video_stream
   embodichain.data_pipeline.real2sim.adaptation.media.encode_video_frame

.. automodule:: embodichain.gen_sim.real2sim.replay
   :members:

.. automodule:: embodichain.data_pipeline.real2sim.adaptation.media
   :members:

Earlier contact-fit composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pipeline.run_real2sim`` composes source-event grounding and geometric fitting,
with optional source graph annotation. It is distinct from the bounded
``execution.run_prepared_real2sim`` loop above. The earlier path produces a
nominal motion reference and does not execute or certify it. Its prepared CLI
wrapper requires existing source grounding artifacts.

.. autosummary::

   embodichain.data_pipeline.real2sim.adaptation.refinement.refine_trajectory
   embodichain.data_pipeline.real2sim.adaptation.refinement.detect_gripper_events
   embodichain.gen_sim.real2sim.pipeline.Real2SimResult
   embodichain.gen_sim.real2sim.pipeline.refine_recording
   embodichain.gen_sim.real2sim.pipeline.run_real2sim

.. automodule:: embodichain.data_pipeline.real2sim.adaptation.refinement
   :members:

.. automodule:: embodichain.gen_sim.real2sim.pipeline
   :members:
