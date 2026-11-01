Scene and robot installation feasibility
========================================

The joint search keeps task semantics, robot model/tool, joint limits and physics
fixed while varying declared scene groups and installation XY/yaw. It returns
conditional proposal regions and exact execution evidence. A workspace point,
an IK solution and a complete physically successful task are different results.

Install the optional physical collision dependencies and run the authored UR5
example from the repository root:

.. code-block:: bash

   pip install -e '.[feasibility]'
   embodichain search-scene-feasibility \
     --config examples/expansion/scene_robot_feasibility/search.ur5.yaml \
     --output outputs/scene_feasibility

The example uses the existing UR5 Task Program integration and an explicitly
versioned one-cycle program. Its declared mounting/structural contact exemptions
must be reviewed when adapting it to another robot. Generated files include
``atlas.json``, ``session.json``, ``report.html`` and accepted ``episodes/*.npz``.
The NPZ sink confirms atomic writes through the existing ``ExpansionSession``;
its causal records contain measured joint/object states and submitted qpos
controller targets. It does not implicitly write a LeRobot dataset.

Input and evidence contracts
----------------------------

``PoseSearchVariable`` expresses one permitted group displacement. All members
share that transform: composed dual arms keep their relative mounting, and
support groups retain their internal geometry. Coordinates are arena-local
Z-up metres and yaw radians. Bounds parameterize XY/yaw offsets about the
reference group origin, not independent member poses.

``TaskFeasibilityRequest`` fixes semantic, robot, policy, validator and parent
scene identities, authored poses, variables and explicit operation modes.
``TaskPhase`` targets are simultaneous TCP requirements; dependencies must be
in topological order. Incoming held-object transforms and phase-local contact
permissions are explicit. Post-phase object poses are planning projections,
never measured evidence. Multiple legal modes require a Python adapter that
binds the selected mode to the canonical runtime. The closed YAML adapter
supports one configured runtime-policy mode.

``RobotFeasibilityBackend`` keeps complete joint states, projects mimic joints,
uses hypothetical full-chain roots, retains bounded IK branches and checks
full-joint connections. A bounded RRT fallback can explore detours. A failure
means no solution found in that declared budget, not mathematical infeasibility.
Unknown geometry, coupled solvers outside the supported contract, or missing
adaptation are reported explicitly.

The default physical adapter reads materialized native robot-link and rigid-body
collision descriptors, including the engine's cooked convex geometry, rather
than render meshes or uncooked URDF surfaces. It requires unit robot asset scale;
other scales use a trusted explicit geometry adapter. It includes the global ground as a halfspace, checked
analytically to avoid unsupported native FCL halfspace distance combinations.
Robot/environment, self, inter-arm and held-object queries use stage-local
permissions. Closed convex meshes use solid FCL geometry; closed concave meshes also receive
bidirectional connected-component containment checks using tiled solid angles.
Open or inconsistently oriented meshes return unsupported when containment is
required. Path checks are discrete and runtime checks occur at observed control boundaries;
these results do not certify continuous collision freedom.

Search, sampling and independent validation
-------------------------------------------

Search preserves global exploration for every operation mode and locally refines
promising joint cells. The archive retains quality tradeoffs and region selection
keeps separated modes/layouts. Regions are unions of conditional joint cells;
their coordinate projections must not be independently recombined.

After search the region distribution is frozen. Held-out trials use independent
IID streams, with fixed planned counts. Reported Clopper-Pearson intervals are
marginal 95% intervals for the raw joint-region proposal distribution, including
semantic/geometric rejections. Search samples do not enter those counts.
Incomplete validation has no confidence interval. A region is not a guarantee
that every point succeeds, and no simultaneous or global optimality claim is made.

``sample_feasibility`` draws complete correlated candidates, then requires fresh
semantic and geometric prechecks. A generated candidate must still pass full
physical task acceptance before becoming successful data. Filtering changes the
sampling distribution; raw-region statistics should not be presented as a new
filtered-distribution confidence statement.

The default budget is 600 seconds with 60% reserved for search and 40% for
validation. Nested runtime IK/planning and Atomic recovery calls share the host's
context-local quota and deadline. Native calls are checked before/after return;
cooperative cancellation cannot preempt an uninterruptible native call. Numerical
CPU/CUDA batches do not change the B=1 physical execution boundary.

Host integration
----------------

``TaskProgramFeasibilityHost`` rebuilds a copied robot deployment, prepares the
scene through ``SimulationSceneExpansionHost``, screens the phase graph, restores
the actual initial state, and invokes ``execute_scene_variant``. The canonical
Task Program remains the action producer and Gym owns every physics step.
Initial validation requires both semantics and geometry. Actual controller
provenance selects runtime contact permissions. Missing mappings fail explicitly.

``MeasuredTaskMonitor`` supports explicit pose, lift, joint-range, velocity,
arena-axis relation and physical-support-bounds predicates. Ordered events retain
actual step receipts and consecutive-sample requirements. A final goal that is
already satisfied at initialization is rejected by the configured adapter.
TaskSpec/GenSim adapters can supply the same fixed request and trusted host ports
without importing their model clients into the scene core.

Public interface
----------------

.. automodule:: embodichain.lab.sim.scene.feasibility
   :members:
   :imported-members:

Implementation modules
----------------------

.. currentmodule:: embodichain.lab.sim.scene.feasibility.cfg

.. autosummary::

   SceneRobotFeasibilityCfg

.. currentmodule:: embodichain.lab.sim.scene.feasibility.contracts

.. autosummary::

   PoseSearchVariable
   PhaseTarget
   HeldObject
   TaskPhase
   TaskOperationMode
   TaskFeasibilityRequest
   RobotDeploymentCandidate
   JointSceneCandidate
   FeasibilityCheck
   CandidateEvaluation

.. currentmodule:: embodichain.lab.sim.scene.feasibility.budget

.. autosummary::

   FeasibilityBudget
   FeasibilityBudgetExhausted

.. currentmodule:: embodichain.lab.sim.scene.feasibility.collision

.. autosummary::

   CollisionBody
   FCLCollisionChecker
   UnsupportedFeasibilityError

.. currentmodule:: embodichain.lab.sim.scene.feasibility.planning

.. autosummary::

   TaskPlanningBackend
   TaskPlan
   plan_task_mode
   RobotFeasibilityBackend

.. currentmodule:: embodichain.lab.sim.scene.feasibility.search

.. autosummary::

   FeasibilityEvaluator
   PreparedFeasibilitySearch
   FeasibilityObservation
   FeasibleRegion
   FeasibilityAtlas
   prepare_feasibility
   search_feasibility
   sample_feasibility
   validate_feasibility
   binomial_interval

.. currentmodule:: embodichain.lab.sim.scene.feasibility.persistence

.. autosummary::

   request_from_mapping
   save_feasibility_atlas
   load_feasibility_atlas

.. currentmodule:: embodichain.lab.sim.scene.feasibility.visualization

.. autosummary::

   write_feasibility_report

.. automodule:: embodichain.lab.task_program.integrations.feasibility
   :members:

.. automodule:: embodichain.lab.task_program.integrations.simulation.feasibility
   :members:

.. automodule:: embodichain.lab.task_program.integrations.simulation.feasibility_geometry
   :members:

.. automodule:: embodichain.compute.geometry.planar
   :members:

.. automodule:: embodichain.compute.geometry.solid
   :members:

.. currentmodule:: embodichain.lab.sim.objects.backends

.. autosummary::

   collision_shapes_from_articulation_link

.. currentmodule:: embodichain.lab.sim.objects.backends.collision

.. autosummary::

   collision_shapes_from_articulation_link

Workspace proposal prior
------------------------

``RobotWorkspaceProposalSource`` copies valid cached FK samples, or prepares a
bounded joint-uniform FK population when no cache is configured. The configured
``workspace_proposal`` selects an explicit phase, control part and scene group.
Group positions are proposed using object-local TCP geometry at the candidate
robot installation. Height/tilt projection is followed by actual IK; a prior
sample never certifies the projected target. A retained independent uniform
component explores poses absent from sparse caches. Malformed configured caches
fail rather than becoming ordinary scene rejections.

``search_feasibility(validate_regions=False)`` returns frozen regions for a
later standalone ``validate_feasibility`` call. Adaptive proposal replay requires
the same feedback stream; seeds do not guarantee identical physics/timeouts.
Numeric batching preserves proposal prefixes for fixed feedback. Fixed search
coordinates have canonical normalized value zero. Atlas loading validates
canonical CPU geometry within 1e-12 while retaining original saved poses and
identities, and does not require the originating CUDA device. New sampling uses
CPU by default.

``observation_keys`` selects required flattened reset/step channels in the
configured adapter; records use ``env/<key>``. The host captures the actual reset
return value rather than recomputing observations/history. Missing channels
cannot accept data. ``recording_max_bytes`` bounds owned raw recording data;
shared-session reservations account for stacking/copying, and overflow interrupts
the attempt without publishing partial successful data. Diagnostic observations
are appended to ``diagnostics/attempts.jsonl`` and session accounting is retained
on errors. Execution cost for ranking is simulated controller duration; wall
latency is reported separately.

.. currentmodule:: embodichain.lab.sim.scene.feasibility.proposals

.. autosummary::

   RobotWorkspaceProposalSource

.. automethod:: embodichain.lab.sim.objects.Articulation.get_link_collision_shapes

The initial physical collision gate includes static/static layout overlap;
intended support/mount contacts must be declared explicitly. Runtime checks cover
all dynamic scene bodies as well as robot and held-object geometry. Preparation
and settling update calls share the deadline; cleanup restores the original ports
and drains resources after interruption. The current episode-session writer uses
position-mode qpos labels; position-velocity expert deployments are explicitly
unsupported by this writer rather than silently dropping required velocity labels.
