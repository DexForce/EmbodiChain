# Rolling workload execution

The public owner is `embodichain.lab.expansion.workloads`. This optional port is
independent of task names and downstream operator configuration. Existing
`ExpansionEngine.from_job` and cohort callers retain their interfaces and defaults.

## Capacity and ownership

`WorkloadExecutionCfg` declares W active groups and B physical candidate rows
per group; the supplied environment must contain exactly W times B rows. The
executor creates no simulation World, process or planning thread. It runs once
on the thread that owns its World. `world_info` records actual process, World,
backend, state and solver devices; selecting a visible CUDA device or renderer
does not select the physics device.

`ExpansionWorkloadCfg` declares a unique group, qualified target, attempt cap and
admission wall-time budget. Candidate ordinals and seeds belong to that group,
not its lane or physical row. Admission includes in-flight candidates in the
remaining quota, avoiding extra speculative trajectories credited beyond target.
Another queued group takes the lane when the previous group has drained. A
difficult group does not silently borrow another group's B-row allocation.

## Source ports and native accounting

`run_task_program_expansion` binds a deployment and a closed scene/motion
configuration to `run_gym_expansion_workloads`. It owns bounded pose/follower
proposals, grasp-policy snapshots, original-grid plan transforms, common measured
acceptance and aggregate results. Configuration/viewer loaders do not import the
simulator. Integrators retain task values and provenance inputs rather than
copying these binding components. `derive_grasp_expansion_deployment` exposes the
same immutable-source preparation for configuration preflight.

`launch_expansion_preview`, `preview_expansion_results` and
`stop_expansion_preview` manage a bounded Linux native service with recorded PID
ownership. The preview reader accepts native aggregate results and derives path
endpoints from recorded motion; downstream workflow-root unwrapping stays with
the integrator. These APIs never recollect or advance physical time.

`run_gym_expansion_workloads` is the public native composition for Task Program
scene/plan workloads. `GymExpansionWorkload` supplies a pinned deployment,
independent collection budget, artifacts and row callbacks. The facade opens one
World, captures the original seeded nominal scene in row zero, and composes the
existing session, frame collector, episode assembler and confirmed artifact sink.
Different integration/grasp policies may share that World only when their
physical configuration, program and cadence agree. Physical/visual/prototype
operators remain on the cohort API; unsupported payloads fail explicitly.
Explicit contact sampling conventions may differ per integration through
`SimulationTaskProgramAdapterFactory.create_adapter(grasp_contact_frames=...)`;
the declared frame must match its grasp service, while TCP, hardware geometry
and drive/command bindings remain strict and the live robot is unchanged.
Only the initial nominal reset may run scene-settling events. The facade then
freezes reset events; rolling preparation is checked against the physical step
counter so another row cannot advance outside the World-owner loop.
Frozen groups export through one bounded writer thread without World access.
Fresh output is required; this facade does not resume interrupted collection.

`WorkloadSource.prepare` runs inside the selected-row `episode_prepare` reset
callback. It must not step physics, alter an active peer or mutate World-wide
appearance/resources. Incompatible topology or shared resources require a
different deployment, rather than pretending they are independent rows.

After at least the declared physical settling substeps, `activate` captures the
initial measured boundary, performs the source's required preparation checks,
registers a native candidate and uses `ExpansionEngine.admit_row`. It returns a
fresh bridge bound exactly to that row and the environment's control cadence.
Rejected preparation returns None and never earns quota.

`started` calls `mark_row_started` after the first actual command. `observe`
captures every active T+1 boundary and performs the original physical checks;
it returns only active rows needing native safe rejection. The executor retains
the bridge acknowledgement/hold handshake. Preparing and completed peers cannot
fabricate actions or observations for another trajectory.

`complete` receives the canonical bridge result and exact first/last World
control indices. It assembles the original source evidence and calls
`ExpansionEngine.complete_row`. Only that confirmed receipt earns a qualified
count. None is a rejection, never a placeholder success. Unavailable or ambiguous
writes retain the ordinary native stage/reconciliation contract and stop the
rolling owner from issuing unsupported quota credit.

Frozen groups can call `export_expansion_episodes` from bounded writer threads.
The callback must not retain mutable live state or operate on the World.
Partial quotas remain explicit failures even when some episodes were exported.

## Planning and validation

The optional cooperative planner merges compatible fixed-grid motion requests
and pure FK/IK/Jacobian calls. Budget-bearing whole-motion requests keep their
original execution so charge semantics are preserved. Context variables and
candidate random streams remain row-local; results are independently owned.
Default CPU command gathering shares native numeric reads through a bounded
World-owner scope in `sim/objects/backends/_numeric_observations.py`. It ends
before completion callbacks and physical stepping, and starts after reset and
activation. Checked native writes, physics steps and binding topology changes
invalidate samples. Derived dynamic rigid poses share the same boundary and invalidation, with
separate CPU autocast representations. Consumers receive independent output
buffers; semantic
state, episode clocks and validation remain row-owned. CUDA and other backends
retain native reads. Raw SDK mutation or concurrent World writes inside the
read boundary are outside this internal side-effect-free planning contract.
Semantic programs and some trajectory processing still run per row. Per-control
planning readiness is a barrier; this API does not claim complete vectorization
or universal throughput improvement.

`NewtonCollisionChecker.check_batch` uses the same physical geometry, pair
exemptions, halfspace supports and precision/saturation rules as scalar checks.
It returns discrete row reports, without claiming continuous clearance. Its
query model is never stepped and is not another simulation World.
`MeasuredTaskMonitor.batch_scope` shares measured numeric inputs while preserving
each native event state machine and receipts. `ContactPermissionChecker` checks
all valid native contact slots with the original applied-impulse and loss gates;
hosts still own sampling at every required physics substep.

Default GPU dynamics use Direct GPU reporting, whose static contact identity
limitations remain explicit admission constraints. Physics devices and task
qualification must be reported separately. Existing CPU deployments do not
switch devices.

## Focused validation

`tests/lab/expansion/test_workloads_sim.py` runs a real four-row task through
native session admission and artifact receipts. It covers rejected-row safe
holds, refill, queued groups, no missing control boundaries, scalar/batched task
predicates and compatible whole-motion result parity. Geometry and contact
coverage lives in the existing native feasibility/contact suites. Per-episode
export uses the real LeRobot SDK and checks portable T+1 evidence.
