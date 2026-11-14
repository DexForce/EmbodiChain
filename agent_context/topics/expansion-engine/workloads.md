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

Default GPU dynamics may use ordinary reporting with
`DefaultPhysicsCfg(direct_gpu_api=False)` when complete static contact identities
are needed. It retains GPU dynamics; the Direct GPU buffer initialization flag
alone is not a physics-device measurement. Physics devices and task qualification
must be reported separately. Existing CPU deployments do not switch devices.

## Focused validation

`tests/lab/expansion/test_workloads_sim.py` runs a real four-row task through
native session admission and artifact receipts. It covers rejected-row safe
holds, refill, queued groups, no missing control boundaries, scalar/batched task
predicates and compatible whole-motion result parity. Geometry and contact
coverage lives in the existing native feasibility/contact suites. Per-episode
export uses the real LeRobot SDK and checks portable T+1 evidence.
