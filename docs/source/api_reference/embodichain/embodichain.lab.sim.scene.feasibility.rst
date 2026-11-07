Scene and robot installation feasibility
========================================

The joint search keeps task semantics, robot model/tool, joint limits and physics
fixed while varying declared scene groups and installation XY/yaw. It returns
conditional proposal regions and exact execution evidence. A workspace point,
an IK solution and a complete physically successful task are different results.

Install EmbodiChain and run the authored UR5 example from the repository root.
The collision query reuses the Newton/Warp packages provided by DexSim:

.. code-block:: bash

   pip install -e .
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
other scales use a trusted explicit geometry adapter. Native ``CONVEX`` shapes
remain explicitly convex; they are not inferred from render surfaces or
triangle winding. The global ground is a halfspace checked with analytic
physical-shape support. Triangle mesh and SDF inputs return unsupported rather
than being silently convexified.

``NewtonCollisionChecker`` owns an independent query model and scratch state.
It does not step physics or mutate the host's positions, filters or clock.
Robot/environment, self, inter-arm and held-object queries use stage-local
permissions. Shape/pair/contact budgets are bounded, and overflow or nonfinite
results cannot accept a candidate. Query arrays and graphs are released by the
host after each attempt.

``contact_query`` separates the native signed-surface predicate from the
conservative distance interval. Exactly represented touching/penetrating native
surfaces can be classified at the original zero rest offset; conversion
uncertainty spanning a threshold remains unavailable. ``contact_pairs`` returns
all active touching/penetrating physical-ID pairs from one pipeline call, or
raises if any pair is undecidable. Returned facts are detached from subsequent
query updates and provide finite/null metadata.

The checker retains triangle-mesh rejection by default. Explicit
``allow_triangle_meshes=True`` uses Newton's actual mesh/mesh and mesh/convex
surface route with the supplied faces, without hull conversion or a solid
containment certificate. Native midphase and contact buffers are bounded and
checked for overflow.

The shared mesh-group adapter can cull a far pair using a frozen AABB separation
lower bound after the same geometry/pose conversion allowance. It allocates no
native model and does not collide. Only a lower bound strictly above the original
threshold may skip the native predicate; a zero or unavailable bound cannot
decide contact. Unsafe coordinates and unsupported shape/precision budgets do
not become clear broad-phase results.

CPU is the default query device. ``collision_device`` selects CPU/CUDA,
independently of the environment-owned physical backend and proposal device;
``collision_use_cuda_graph`` controls CUDA replay. Reported clearance is a
nonnegative estimate capped by the declared contact detection band, never an
infinite or exact global minimum distance when no contact is generated. Backend
version, device, tolerances, detection band, budgets and verification identity
are retained in execution evidence. Path checks are discrete and runtime checks
occur at observed control boundaries; they do not certify continuous collision
freedom.

Transforms are translated to a query-local origin before applying shape-local
offsets. Native queries additionally use an owned scale-aware tolerance value,
updated before CUDA graph replay. Unrepresentable geometry, nonfinite composed
poses/distances, or precision uncertainty beyond the contact-band budget return
unsupported. The largest applied tolerance and coordinate policy are recorded
in metadata; one checker instance expects serialized calls.

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
   NativeContactResult
   NewtonCollisionChecker
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
