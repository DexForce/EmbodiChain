# Official Cube migration and alignment

Work is paused at the user's request. The [paused checkpoint](validation/paused-2026-10-31/README.md)
preserves reports, diagnostics and remaining work. The latest optional soft-mimic
velocity implementation is WIP: builds completed, post-build physical validation
is pending. Earlier passing counts do not qualify that latest implementation.

The requested destination is EmbodiChain/DexSim. MuJoCo supplies the reference
task, assets, controls and measured trajectories. Completion requires the physical
task and the official evaluation protocol; robot loading, numerical model agreement
and historical UR10 scores are separate intermediate checks.

## Pinned sources

| Resource | Revision / identity |
| --- | --- |
| [LeWM implementation](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac) | `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` |
| [stable-worldmodel Cube wrapper](https://github.com/galilai-group/stable-worldmodel/blob/abdced49809d5eae38e24b27dc7b635c502c4812/stable_worldmodel/envs/ogbench/cube_env.py) | `abdced49809d5eae38e24b27dc7b635c502c4812` |
| [OGBench robot and control](https://github.com/seohongpark/ogbench/blob/6eda27402c69681cfe584cc9f3a3fe2a24ec9163/ogbench/manipspace/envs/manipspace_env.py) | `6eda27402c69681cfe584cc9f3a3fe2a24ec9163` |
| [Official Cube checkpoint](https://huggingface.co/quentinll/lewm-cube/tree/b0747c5002e86d2ce8f3cd8178004b97524c587d) | `b0747c5002e86d2ce8f3cd8178004b97524c587d` |
| [Official Cube dataset](https://huggingface.co/datasets/quentinll/lewm-cube/tree/02a19a67a0dc8c9d6215f89c19e0a597691e152a) | `02a19a67a0dc8c9d6215f89c19e0a597691e152a` |
| [Paper](https://arxiv.org/abs/2603.19312) | Architecture/objective reference; the pinned runnable Cube example owns reproduction settings. |

Checkpoint SHA256:
`2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89`.
Dataset archive SHA256:
`3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600`
(46,184,624,478 bytes). Keep upstream code, assets and data outside the wheel.
Generated robot assets retain the OGBench, UR5e and Robotiq licenses.

## Parameter contract

`configs/ogbench_cube.json` records the planning/evaluation recipe.
`official-reference` records the actual compiled parameters in
`physical-contract.json`, rather than inferring them from comments or defaults.

| Boundary | Official setting |
| --- | --- |
| Robot | UR5e with the original Robotiq 2F85 mesh, inertia and closed-loop linkage; six arm and eight gripper joints |
| Policy action | Five relative commands: XYZ, yaw, normalized gripper opening |
| Action scales | `[0.05, 0.05, 0.05, 0.3, 1.0]`; gripper target derives from measured right-driver position divided by `0.8` |
| Cartesian workspace | Lower `[0.25, -0.35, 0.02]`, upper `[0.6, 0.35, 0.35]`; target orientation points down |
| Timing | Physics `0.002 s`, 25 substeps, raw action `0.05 s` |
| Reference physics | Implicit-fast integration, elliptic cone, Newton solver, `impratio=10`; compiled options are retained in the sidecar |
| Arm proportional gains | `4500, 4500, 4500, 2000, 2000, 500` |
| Effective velocity gains | `400, 400, 400, 100, 100, 100`; see the actuator caveat below |
| Arm torque limits | First three `±150`, last three `±28` N·m; joint armature `0.1` |
| Gripper | Two connect equalities, one joint equality, split tendon with coefficients `0.5/0.5`, actuator force `±5`; passive spring/damping/armature remain distinct |
| Cube | Side `0.04 m`, density `1240 kg/m³`, mass `0.07936 kg`; original collision, friction and soft-contact parameters |
| Observations | `ob_type=states` plus an opaque-arm RGB rendering from the wrapper's `front_pixels` camera, 224×224, no multiview |
| Camera | Position `[1.053, -0.014, 0.639]`, vertical FOV 45°; compiled orientation and clip planes in the sidecar |
| Model | Maximum trained history 3; evaluation observes **1 frame**. Raw action width 5, block 5, concatenated width 25; model period `0.25 s` |
| CEM | 300 samples, 30 iterations, top 30, horizon 5 blocks; solver batch size 1, shared seed-42 RNG, elite sample deviation (`correction=1`), standardized unit Gaussian, mean candidate inserted, no clipping or minimum deviation, final elite mean |
| Evaluation | Seed 42, 50 dataset queries, goal 25 raw steps ahead, raw budget 50, episode limit 100, receding horizon 5 model blocks |
| Success | Post-step 3D cube-position distance `<= 0.04 m`; orientation is not part of success |

The upstream arm code writes a second set of intended damping values into
`actuator_gainprm[:, 2]`. Fixed-gain actuators use `gainprm[0]`; their velocity
bias comes from `biasprm[2]`, which that code leaves at `-400/-100`.
Migration follows the compiled actuator law, including force clipping.

Paper and runnable-example settings are distinct: paper v1 Appendix D gives
10 CEM iterations outside PushT, while Appendix B gives 30 generally; the pinned
Cube example uses 30. Appendix E describes 10 training epochs, whereas the
pinned training configuration defaults to 100. The paper's main-method SIGReg
weight is 0.1; the example recipe uses 0.09. This migration follows the selected
runnable example and records these differences instead of calling its local
score a paper reproduction. See the [original paper](https://arxiv.org/html/2603.19312v1#A4).

The base OGBench renderer selects a different camera for state observations;
the stable-worldmodel wrapper overrides that default. The wrapper also changes
gripper colors. Port the wrapper actually used by LeWM.

## Verified components and evaluation checkpoint

Cube uses the May 20, 2026 stable-worldmodel revision, contemporaneous with the
LeWM example. The newer revision retained for historical PushT checks changes
the solver interface and elite deviation. The OGBench manipspace subtree is
unchanged between its last pre-example commit and the pinned OGBench revision.
Five receding model blocks execute **25 raw actions** before replanning. The
50-step evaluation has planning barriers at steps 0 and 25.

`official-reference --compare-cem` checks clean revisions, compiles the task,
exports robot/physical/render contracts, and records a 100-step source replay.
The retained-body candidate preserves all 21 positive-mass source bodies and
folds only massless frames. Across 100 configurations it preserves generalized
inertia, gravity and all 25 original frames. The earlier 15-body results below
used compound followers and do not validate the new candidate. Native kinematic
and mass/inertia checks advance no physics. The analytic CEM comparison checks all
candidate populations with the original solver's batch order and sample deviation.

`reference --cube-weights ... --cube-swm-root ...` verifies the checkpoint and
compares the original single-frame model path with the migrated network.
Fixed-input numerical tolerances pass; this is not a task-success measurement.

The native task implements reference IK, force-limited implicit arm drives,
passive/split-tendon forces, source self-collision exclusions, both point
connections and driver mimic. CPU and Direct-GPU tests verify finite D6 spring
compliance. Equality gains remain an approximation. With named pad materials
and material caching, both controlled source-expert replicas first reach success
at step 41 versus source step 39. Maximum cube height remains 0.232/0.231 m
versus 0.192 m in MuJoCo. With the source-based mimic mapping, final goal errors
are 4.3/8.2 mm and the largest replica separation is 9.0 mm. A cached
uniform-friction-1 diagnostic happens to
match source lift height, but has a 50.5 mm replica separation and one final
goal failure. Retain source pad values rather than selecting the uniform
counterfactual for its height score. This 160-step diagnostic is not MPC
evaluation and does not establish loaded-contact parity.

The old mimic mapping used `frequency=1/timeconst` and damping ratio 1. This
omitted MuJoCo's impedance/regularizer factors. An independent two-hinge test
measures 86.40 microradian steady error versus MuJoCo's 2.715. Mapping the base
KBIP/R force response gives native frequency 858.289 and ratio 4.47214; both
CPU and GPU measure 2.655 microradian (2.2% error). The task now uses that
mapping. It assumes matching actual/diagonal inverse inertia, verified to
about 3e-5 relative at four driver-pair states; nonlinear `solimp` and the point
constraints still need separate validation. This is a derived base-response
mapping, not a fit to the task success count.

An actual GPU audit covers generalized mass matrices and gravity at all 50
query states. Rigid-body mass matrices match to 3.43e-6 and gravity to 2.16e-5.
PhysX's matrix query omits joint armature (also reported in
[upstream issue 575](https://github.com/NVIDIA-Omniverse/PhysX/issues/575)); readback
armature matches source, and ten independent gravity/pulse cases verify its
simulated effect to a maximum velocity error of 8.22e-8. These checks did not
justify changing the inertias or arm drive gains.

The complete official archive and HDF5 are verified. HDF5 SHA256:
`0664d507c4ff12009010644c9ae950836f954e700c172ccf22e7423af1a55625`.
The literal 50-query selection retains 16 starts already inside the goal
threshold. Action moments use all 2,000,000 non-NaN action rows. Native state
restoration preserves small compliant-limit violations through explicit
`Articulation.set_state(clamp_qpos=False)`; target limits still apply.

All following runs use the same queries and official checkpoint, without
retraining. These local measurements do not claim to reproduce the paper's score.

| Run | Collision-mask fix | Named pads + cache | Current mimic mapping |
| --- | --- | --- | --- |
| Original MuJoCo World/policy/model/CEM | 34/50 | 34/50 | 34/50 baseline |
| DexSim with original preprocessing/model/CEM | 34/50 | 33/50 | **34/50 (68%)** |
| DexSim replaying reference raw actions (diagnostic) | 33/50 | 32/50 | **34/50** |
| DexSim with migrated model/planner | 33/50 | 31/50 | **32/50 (64%)** |

The main regression was in SDK arena replication: replacing all four filter
words erased the walls' authored category/exclusion masks. Robots then collided
with walls that only constrain cubes in the source task. The SDK fix replaces
arena/isolation words 0/1 while preserving body/link layer words 2/3 through
template build, native clone requests and retained descriptors. Native readback
confirms the fix in both arenas. A physical regression fails before the fix
(masked payloads incorrectly rest on a platform) and passes afterward. All 24
actor/link/filter cases pass across CPU/GPU, clone modes and thread modes.
Eight failed-state probes reduce worst first-period arm error from about 1.1 rad
to 0.0021 rad, without changing gains, masses, action scales or timestep.

The original-planner adapter reproduces all 50 reference first raw plans bit
for bit after inverse StandardScaler conversion. Source plan files store
standardized actions; native plan files store raw actions, so direct file
comparison is invalid. Current original-planner outcomes agree on 48/50:
source-only success [32], native-only [43]. The migrated model also agrees
on 48/50, with source-only successes [32,44]. All 16 initially-near goals
still succeed. Repeating the migrated run on the same worker GPUs as the
original-planner run reproduces all initial plans and outcomes. Small neural
arithmetic differences can alter CEM elites over
30 iterations: pointwise network tolerances do not imply identical plans.

Source-action replay preserves raw commands and NaN masks exactly and agrees
on 46/50: source-only successes [32,44], native-only [41,43]. Its count improves
from 32 to 34. First-period median/max cube errors are 0.318/11.67 mm.
Whole-replay median/p95/max are 3.49/54.50/268.87 mm, versus 4.83/69.36/262.90
mm at the named-pad checkpoint. Median/tail errors improve while worst error
does not. Query 32 still loses its cube during lift. Offline source-geometry
evaluation identifies unilateral pad contact before loss and larger closed-loop
residuals; those offline force values are source-law evaluations, not native
measurements. Controlled point-force variants have mixed outcomes and remain
diagnostics. Preserve batch sizes/layouts when comparing these stiff contacts.

`official-eval-reference` runs the original evaluator.
`official-eval-dexsim` keeps two 25-environment native processes alive across
both planning barriers: the Vulkan multiview path accepts at most 32 views per
process. One planner visits active global query indices in order using one RNG.
Use `--reference-planner-root` for the controlled original-planner comparison,
or `--replay` for a reference-action diagnostic. Both evaluators use dataset
pixels initially and engine pixels at step 25. Completed native rows' recorded
results freeze at success while their isolated physical arenas may keep stepping.
The native workers require the local SDK build with D6 body endpoints and
compliant-drive and per-shape contact APIs; the installed SDK is unchanged.
The task explicitly selects `DefaultPhysicsCfg(cache_material=True)`. A physical
two-platform test exposed incorrect Direct GPU friction without this cache in
the pinned PhysX 5.6 build, despite correct CPU-side material readback. The SDK
rejects uncached GPU shape overrides. This engine allocation setting changes
no source friction, timestep, gain, mass or geometry. Fixing the uncached
dependency path remains open.

Independent named pad materials now survive loading, source-property replay,
Arena cloning and reset. Source pads use friction 0.7/0.6, and cube/pad pairs use
the native minimum combine rule. The same rule does not express MuJoCo's full
priority policy for unlike pad/pad contacts; do not claim complete contact
equivalence. The supported CPU/GPU material matrix, clone/source replay,
filters and constraints pass 79 SDK tests. Project checks pass 490 tests plus
42 context tests (three optional tests skipped), black and all 3285 API exports.
The execution ledger retains failing uncached evidence and all old/new runs.

## Retained bodies and measured state

The exporter keeps fixed bodies with positive mass, including the 3.5 g pads
and their 1.3028 g silicone children, while preserving massless frames as
transforms. The robot still weighs 22.04750833884274 kg. Independent source/URDF
checks over 100 configurations give maximum generalized-inertia error 5.83e-12
and gravity error 1.21e-13; native mass/FK checks also pass. Source weld-group
and enabled parent exclusions are exported separately from the six explicit
exclusions. Native mandatory parent filtering still needs its remaining
eligibility review. This candidate requires a newly exported reference directory;
earlier 15-body assets cannot provide separate pad measurements.

`enable_source_state` adds the original 28-D state and the world external pad
wrench in source torque/force order, with torque about the robot subtree COM.
The reconstruction accounts for incoming/child reactions, inertia and gravity.
Independent MuJoCo fixtures cover a rotated, moving pad with external force on
both it and its child. On the literal 50 initial states, including 26 negative-W
quaternions, single-world encoding has maximum error 8.33e-6 and raw quaternion
error 2.88e-8. A 50-world batch records up to 3.79 micrometres of position roundoff
from world/arena float32 conversion. At reset, force is explicitly invalid
(`state_wrench_valid=false`); it becomes measured after a physics step. Reset
encoding checks supply zero wrench and do not prove source reset-force parity.

This requires the paired SDK batch accessors **and a corrected PhysX 5.6 GPU
library**. The accessors alone are insufficient: the vendor velocity-output
flush omitted acceleration-delta accounting, and partial impulse propagation
omitted force accounting before clearing scratch values. A matching CHECKED
source build reproduces both defects. The isolated three-line patch reuses
existing accumulators and is retained under the SDK's
`3rdparty/physx/patches/5.6-gpu-link-measurements.patch`, with build instructions.
Shared installed dependencies are unchanged. Re-staging the SDK can restore its
prebuilt dependency, so verify the loaded artifact before force validation.

Six known-load cases pass, including two dynamic supports checked against
independent contact impulses/moments and finite velocity differences. In a
45-control expert replay, 24 loaded normal-force comparisons have maximum
relative error 6.65e-7; 21 previously missed contacts recover. Joint positions,
velocities and cube poses remain bit-identical after the partial-propagation
repair. This verifies native measurement accounting, not MuJoCo contact laws.

The earlier TGS candidate's 50-query evaluations give 34/50 with the original planner,
32/50 with the migrated planner, and 33/50 with exact source-action replay,
versus MuJoCo 34/50. Their outcome agreements are 46/50, 48/50 and 45/50.
Replay cube error is median 3.56 mm, p95 81.11 mm, maximum 248.60 mm, despite exact
actions. Original-planner initial raw plans are bit-identical to the source
after the actual sklearn inverse transform; migrated initial plans still differ
by up to 5.379 raw units. Retained topology and the private GPU build both change
between this checkpoint and the historical 15-body results.

Under 45 identical expert commands, source/native peak pad forces are
49.52/114.44 N and maximum cube heights are 0.19168/0.23174 m. The trajectories
evolve independently: these results locate further diagnosis in loaded grasp
behavior without isolating which contact/equality parameter causes the gap.
Full physics alignment remains unverified.

## PGS recipe and current accepted validation

The task now uses PGS and force-unit point gains computed from the compiled
source `body_invweight0` and `KBIP/R`. The ordinary Default backend keeps TGS
as its default; the task selects `enable_tgs=False` explicitly. Source sleep is
disabled and gravity must match the reference. Re-export the reference to obtain
the new metadata; earlier directories remain frozen historical evidence.

Default native creation settings are submitted before World construction so
solver selection reaches the actual scene. The saved creation descriptor is
not every live mutable value: free-fall tests show gravity was already updated
by the old implementation, so no gravity-physics improvement is claimed.
Unsupported CCD plus Direct GPU raises before allocation. The SDK's existing
PGS friction-per-iteration flag is now consumed at scene creation.

The paired GPU library also repairs PGS mimic compliance. Repeated solves
previously omitted the accumulated CFM term and made soft coupling effectively
hard. The repair keeps one scalar impulse per mimic in existing GPU storage.
At backend-default iterations, independent torque tests reproduce the spring
law. A separate native pre-initialization probe configured 100 iterations.
The source-mapped equilibrium is 2.71495969 microradians, versus MuJoCo
2.71492317. All four TGS control trajectories remain bit-identical. Native
velocity-iteration policy is retained; maximum scalar transient position error
is 0.2584 microradians, so equilibrium agreement is not full dynamic parity.

This recipe requires both SDK patches under `3rdparty/physx/patches/`:
`5.6-gpu-link-measurements.patch` and `5.6-gpu-pgs-mimic-compliance.patch`.
The validated artifact is a matching CHECKED PhysX 5.6 GPU build. The prebuilt
CPU mimic solver is unchanged and CPU PGS compliance is not certified.

The restored native-drive recipe was measured on the same 50 official queries,
seed 42 and planner settings. The rejected source-servo campaign below is
retained separately; its results are not substituted for the restored runtime:

| Restored native-drive recipe | Success | Outcome agreement with MuJoCo |
| --- | --- | --- |
| MuJoCo reference | 34/50 | reference |
| DexSim + original planner | 33/50 | 49/50; source-only query 7 |
| DexSim + migrated planner | 35/50 | 49/50; native-only query 12 |
| DexSim + exact source actions | 32/50 | 48/50; source-only queries 5,7 |

All 16 initially-near queries succeed. Replay raw actions and NaN masks match
exactly. Cube trajectory error is median 3.05 mm, p95 41.46 mm and maximum
78.55 mm, versus 3.56/81.11/248.60 mm at the previous TGS checkpoint. The original
planner's initial raw plans and first 25 common finite controls match the source
bitwise after the actual sklearn inverse transform. Migrated initial plans
still differ by up to 5.379 raw units. Counts do not establish physical parity.

Identical 45-command expert replay gives:

| Physical trajectory | Maximum height | Maximum cube-position error | Mean pad force, steps 20–35 | Peak pad force |
| --- | --- | --- | --- | --- |
| MuJoCo | 0.19168 m | reference | 47.54 N | 49.52 N |
| Prior TGS | 0.23174 m | 41.05 mm | 56.87 N | 114.44 N |
| Current PGS | 0.18915 m | 5.76 mm | 47.83 N | 130.86 N |

The loaded geometry and mean force improve substantially; normal-contact
impacts remain too sharp. Source contact softness, nonlinear equality response,
soft joint limits, geometry cooking, rendering and exact planner arithmetic
remain open. In particular, prior unconstrained gravity/pulse checks isolate
armature; they do not establish complete stiff-PD integration equivalence.

Validation includes 560 Default-scope project tests, the separate 42-case
context gate (overlapping coverage), 93 SDK regressions, 100-configuration
source/native model checks, Black 26.10.1 and all 3285 API exports. The paired
SDK lacks `AutoSolverCfg`: 18 Newton configuration tests reproduce the same
failure before/after the Default fix; they are not waived or presented as green.
The registered collisionless-chain benchmark now measures 25/256 chains at
backend-default iterations. Previous four-row reports labeled 4/100 are retained
as historical artifacts, but their iteration-sweep claim is withdrawn: the public
solver/sleep methods used there were unsupported warning-only calls. The duplicate
regression and parameter labels have been removed. The task's configured root
properties use a separate native pre-GPU path. Do not infer an iteration sweep
from a request argument or test label.

## Contact and servo characterization

The native material schema now exposes constant compliant-contact stiffness
and damping through `DefaultRigidBodyMaterialCfg`, including named URDF shapes,
source overlay replay and sparse merging. It reuses the SDK's existing atomic
shape-material operation; CPU/GPU spring equilibrium and backward-Euler impact
checks pass. Constant contact springs remain a separately tested capability;
they are not substituted for the official nonlinear contact law in the task.

Independent normal fixtures use the official cube mass/size, timestep,
friction, contact dimension and mixed source materials. The compiler selects
floor/cube `solref=(0.012,1)` at equal priority and pad/cube `(0.004,1)` at pad
priority. The pad is flattened and static in this fixture; it does not reproduce
the articulated gripper's inertia. Contact forces are independently checked
against momentum balance.

| Normal fixture | MuJoCo | Native hard | Native constant source-base spring |
| --- | ---: | ---: | ---: |
| Box/floor, self-weight compression |39.275 µm|0|39.337 µm|
| Box/pad, self-weight compression |2.131 µm|0|2.129 µm|
| Box/floor, extra 20 N compression |623.425 µm|0|1050.204 µm|
| Box/pad, 0.5 m/s impact peak |26.744 N|20.618 N|20.405 N|

Matching the low-load compression does not match nonlinear impedance or the
contact transient. The constant spring mapping is therefore not adopted as a
complete replacement for source contacts. Both high-load penetration and the
per-step force curve remain acceptance gates.

## Rejected source actuator and integration study

This candidate was rolled back after its full paired evaluation regressed.
The current task again uses the preceding accepted native-drive recipe.
Candidate code, tests, numeric benchmark and source goldens remain in the
external `alignment/rejected-source-servo-production` archive.

The candidate evaluated the compiled old-state actuator law, including force
clipping, passive springs/damping and split-tendon contributions. Its native
arm position/velocity drives were disabled. The native solver resolves explicit
forces and constraints first; the task then applies the source implicitfast
velocity update:

`v_new = v_old + solve(M + dt * D, M @ (v_physics - v_old))`

`q_new = q_old + dt * v_new`

Here `D` is the negative smooth velocity derivative. Saturated actuators omit
their velocity derivative. The source model stores mass-tree sparsity, so
sibling tendon cross-terms are omitted from the effective derivative. Using
an unprojected physical tendon derivative would change the source integrator.
Its private numeric operations lived in `embodichain.compute.dynamics`; the
candidate task owned topology, source parameters and state publication. Those
candidate runtime files are archived rather than left active.
Mass uses native public link poses/joint descriptors and declared 21-body
inertias, with all 14 armature entries retained. No MuJoCo environment or
upstream runtime executes in the native task.

Across 50 frozen query states and independently sampled controls, mass agrees
with MuJoCo within `1.94e-6` absolute, generalized force within `8.96e-13`, and the
effective velocity derivative exactly. Independent coupled-hinge tests include
force saturation and contact. A gain-only `Kd-Kp*dt` adjustment works only
without saturation/contact coupling and is not used. The diagnostic constant
contact has an exact discrete coefficient map only under its stated `dt*B=1`
condition; this does not establish the official nonlinear-contact mapping.

The source controller and state observation read pinch FK cached before the
last 2 ms physics step. The candidate retained that cache and invalidated reset rows
selectively. Joint state, cube state and success use current generalized state.
Native incoming-wrench measurements are recovered before the velocity/state
rewrite, then their moment is shifted to the source subtree COM. Source reset
force semantics and complete rendering-phase agreement remain separate gates.

A production replay of the same 45 fixed expert commands gives:

| Quantity | MuJoCo | Previous PGS drives | Source-servo PGS |
| --- | ---: | ---: | ---: |
| Maximum cube height |0.191678 m|0.189152 m|0.192330 m|
| Maximum cube path error |reference|5.764 mm|3.003 mm|
| Maximum arm position error |reference|about 8.37 mrad|4.290 mrad|
| Pad mean force, controls 20–35 |47.535 N|47.829 N|48.229 N|
| Pad peak force |49.523 N|130.857 N|102.627 N|

These improvements do not complete physical parity. Contact impact softness,
point/mimic transients and soft limits remain unmatched. The measured normal
projection agrees with reported native impulses to at most `1.40e-5` relative
in this trajectory; tangential contact-row impulses remain unavailable.
The source replay is independently reverified bit-exact in MuJoCo.

Validation includes 32 focused source-task tests and 529 related Default-scope
regressions (2 optional skips, 75 unrelated Newton deselections). The numeric
benchmark uses 21 links, 14 DOFs, float64, 10 warmups and 100 repeats; completed
medians are approximately 1.05/1.06/1.09 ms for batches 1/25/50 on the selected
Blackwell GPU. This excludes native stepping, state I/O, IK and rendering and
is not a solver speedup claim. The first complete frozen source-servo campaign failed the broader alignment
check: exact-source-action replay is 32/50 with 46/50 outcome agreement; original
and migrated planners are each 33/50 with 47/50 agreement. Raw replay path error
has median/p95/max 3.273/47.501/235.731 mm, worse than the preceding PGS campaign.
The single controlled trajectory therefore does not qualify this runtime as
physically aligned, and the initial source-servo version remains unaccepted.

The regression audit found that PGS final velocities omit stabilization retained
in native joint-position integration; their difference reaches 0.556 rad/s.
Directly rebuilding positions loses it. Two private adapters restore driver
precision but fail to recover the lost queries in a controlled six-query layout,
so they are not adopted. A task-local zero-velocity-iteration configuration was also tested because
native mimic velocity iterations use zero ERP/CFM, a stage absent from the source
solver. It reduced motion/final-velocity discrepancy to 5.96e-5 rad/s, but the
same six-query outcome agreement worsened from 3/6 to 2/6. It was not retained.
The current task/test/evaluator were restored from the pre-study snapshot;
restored task tests pass and all rejected evidence remains available.

## Point impulse characterization

The SDK now exposes D6 linear velocity targets through the existing constraint
and Arena owner. `RigidConstraintCfg.linear_drive_velocity` specifies endpoint B
relative to A in A's joint axes, in m/s. The manager's
`update_rigid_constraint_linear_drive()` changes gains and targets between
explicit steps, with selected-arena input/capability preflight. Creation without
a target keeps the previous behavior. CPU/GPU rotated-frame motion and lifecycle
tests pass. An identical 45-command task replay preserves every recorded joint
state, cube pose, wrench and contact metric exactly when this capability is unused.
A fresh 50-query, 25+25 replay also preserves all state, action, NaN-mask and goal
arrays exactly. The task still uses its accepted native-drive recipe. These
capabilities use the isolated SDK build; the shared installation is unchanged.

For a source point row with consistent signs,
`R*lambda + J*v_new = (1-h*B)*J*v_old - h*K*d*r - h*Jdot*v`.
An implicit spring with the **same Jacobian** needs
`Kn=K*d/R`, `Dn=(1/h-h*K*d)/R`, and
`vtarget=((1-h*B)*J*v_old-h*Jdot*v)/(1-h*h*K*d)`.
The earlier static gain mapping checked equilibrium; it did not establish this
time-discrete response. The moving point's acceleration-bias term is essential.

Independent source/native slider tests cover 18 position, velocity and opposing
load combinations. The full target mapping gives maximum velocity error
9.94e-9 m/s and impulse error 7.45e-9 N·s. Four 500-step trajectories update
nonlinear impedance from live native states, with no MuJoCo runtime in the
native loop; maximum position/velocity errors are 1.22e-7 m and 6.42e-8 m/s.
These are isolated scalar constraints, rather than loaded-gripper parity.

Eight free-body cases add body rotation and separated anchors. Including the
acceleration bias reduces the identity-frame, coincident-anchor moving case to
1.15e-8 maximum velocity error. With separated anchors, D6 applies actor A's
force at anchor B; MuJoCo connect uses anchor A. The largest remaining component
error is 1.44e-4 in linear/angular velocity. An independent 12-dimensional
inverse-mass impulse calculation predicts the native results within 8.15e-7 and
reproduces the source results within 8e-16. A velocity target alone cannot remove
this generalized-force difference. Preserve the native D6 contract while
investigating a separate source-compatible point row.

A pure URDF recursion also matches position, velocity and acceleration bias for
both real gripper connections at all 50 official query states, with maximum
errors 2.23e-16 m, 1.95e-16 m/s and 8.54e-16 m/s². A private task adapter uses
these live-state quantities and nonlinear impedance without changing actuation
or whole-task integration. On the 45-command replay its maximum cube error is
5.729 mm versus 5.764 mm for the accepted recipe, and peak pad-force norm falls
from 130.86 to 115.41 N, still above the source's 49.52 N.

The private adapter **fails the full replay alignment gate**. On the same 50
queries and exact source actions, successes increase from 32 to 34, but outcome
agreement stays 48/50: source-only query 7 and native-only query 12. Frozen
post-step cube errors worsen from median/p95/max 3.051/41.456/78.555 mm to
3.647/62.717/120.578 mm. A supplementary comparison uses the same 973 live states
through the earliest success across the source and both native runs. Its
median/p95/max also worsen, from 4.165/39.269/65.841 mm to
7.162/63.087/102.441 mm. Thus the regression is not explained solely by traces
freezing at different success times. The adapter stays private and is rejected
as a replacement for the accepted task. Scalar impulse agreement is insufficient
for the coupled gripper, native actuation and contacts.

The point-drive benchmark covers one and 25 arenas on an RTX PRO 5000 72GB
Blackwell, float32, PGS, five blocks of 100 completed steps after 500 warmup
steps. Host drive-update means are 0.021/0.183 ms; completed-step medians are
2.70/2.82 ms, including updates and native stepping, excluding setup, rendering
and IK. Equilibrium errors are below 1e-4 m. A pre-existing GPU broad-phase
failure for entirely disabled collision shapes is independently reproduced
without any constraint on the prior engine and retained in the validation
ledger. These measurements use enabled, spatially separated collision shapes.

## Coupled constraint and integration validation

The following coupled-stage work qualifies another component capability. The
accepted task recipe and its planner measurements above remain unchanged.

`RigidConstraintCfg.linear_drive_point_anchors=True` now opts into GPU linear
spring rows with each endpoint's authored force moment arm. It uses the existing
D6 descriptor, constraint/Arena owner and Scene thread; it changes no vtable or
descriptor layout. The ordinary D6 default remains intact. A query of the
already-loaded GPU library guards the feature; CPU and older GPU providers reject
enabling it. Gain updates retain the choice. It requires the additional vendor
patch `5.6-gpu-d6-point-anchors.patch`. Independent 12D spring-impulse tests pass
for default and selected geometry, and the CPU rejection test passes. A fresh
50-query replay with the feature unused reproduces every previous qpos, qvel,
action, NaN mask and goal value exactly.

A separate full replay enables only this anchor selection, retaining the accepted
controller, gains, mimic mapping and iteration policy. It gives 35 successes but
only 47/50 outcome agreement, with source-only query 7 and native-only queries
12,41. Median/p95/max trajectory error is 3.400/53.826/129.417 mm, worse than the
accepted 3.051/41.456/78.555 mm. Thus the geometry feature also remains optional;
its isolated physical correctness does not justify a task-default change.

The real-robot oracle isolates none/point/mimic/both constraints across six
states and undamped/passive/controlled force tiers. It retains the 21 bodies,
inertias, armature and equality parameters, while removing contact, gravity and
limits only in this diagnostic. An independent dense solve matches MuJoCo's
constraint acceleration within 6.29e-11; the separate implicit-fast mass update
matches its velocity within 3.04e-12. Native free steps match the pre-integration
velocity within 5.60e-7. The source's pre/post integration difference reaches
0.302 rad/s under the compiled controller, so the two stages cannot be conflated.

PhysX advances articulated COM/angular velocities with bias acceleration before
external constraint preparation. The earlier free-body target mapping omitted
that native stage. An independent recursion matches all 1050 COM states/biases
from the 50 official queries within 1.73e-15. Including this term explains native
responses to about 1.3e-4 rad/s, versus about 0.02 without it. With corrected
targets, the point-only error against the source falls from about 0.021 to 0.0028;
selecting authored anchors reduces it further to 0.000122 rad/s.

The documented PGS mimic velocity pass separately projects the row to a hard
zero-velocity relation. A soft position-row model followed by that projection
predicts the native mimic response within 3.55e-7. In the isolated source-stage
adapter, zero velocity passes and time-discrete gains retain finite compliance.
The missing mimic target can be supplied through existing generalized-force I/O:
`sigma=(1-h*B)*(J*v_old)/(h*R)`, with an added `J.T*sigma`. The total row impulse
is the native impulse plus `h*sigma`; this constraint contribution stays outside
the smooth-force derivative used by implicit-fast integration. Mimic-only velocity
error then falls to 5.60e-7 rad/s. Both point and mimic rows, followed by the source
mass update, match within 0.000121 rad/s in this oracle.

These results do not validate loaded contacts, soft limits or whole-task controls.
A private integrated adapter combines the qualified stages. Its 45-command
expert replay reduces mean cube error from 2.528 to 1.269 mm; maximum error is
5.667 mm and peak pad-force norm is 120.90 N, still above the source's 49.52 N.
The full 50-query source-action replay rejects this adapter: success count stays
32, outcome agreement falls from 48 to 46, with source-only queries 7,32,34 and
native-only query 41. Frozen median/p95/max error worsens from
3.051/41.456/78.555 mm to 4.903/83.695/253.809 mm. The same 977 common live states
also worsen, from 4.165/55.351/78.555 mm to 7.510/87.367/253.809 mm. It remains
private; the production task has not changed.

A further 45-command control keeps the robot's requested velocity passes at zero
but restores one pass for the cube. Mean/maximum cube error improves to
1.195/3.148 mm and peak pad force to 108.73 N. However the robot's motion/final
velocity difference returns to 0.264 rad/s, versus the roughly 6e-5 float32
position-difference bound with both at zero. A separate real14 control adds a
distant, noncontacting dynamic cube requesting one velocity pass. The robot's
motion/final difference reaches 0.03855 rad/s despite its own zero-pass request.
The GPU mimic kernel runs the scene's velocity phase without a per-articulation
iteration guard; contact with that cube is not required. These controls fail the
component gate and are not promoted. Qualify an explicit per-articulation soft
mimic velocity policy before using mixed contact iteration counts.

Fresh reference exports now also retain joint limits' range, activation margin,
solver reference and impedance, DOF inverse weights/friction-loss solver
parameters, and geom mixing/margin/gap. Existing exported fields are unchanged
in an independent pinned-source comparison. Older reference directories remain
frozen. Exporting these missing parameters does not implement their native
physics; soft limits and contact response are the next isolated gates.

## TODO and improvement order

1. **P0: quantify remaining physical differences under identical commands.**
   Replication/filter loss is fixed and validated. Inspect post-fix source-action
   replay failures [5,7] and the worst remaining trace in query 12. Compare
   loaded gripper closure, soft joint limits and coupled constraint transients.
   Source clipping/sparse integration and control-FK timing were independently
   characterized above, but their combined native candidate regressed in full
   replay and was rolled back. Point/mimic coupling and integration now pass in
   the isolated real14 oracle, while the integrated adapter still regresses.
   Isolate source soft limits and contact stages before another task change.
   Preserve the rejected adapters and the same failed-query/full-replay gates.
   Source-mapped PGS repairs the prior gripper-shape failure; contact impact
   peaks and remaining trajectory error still require controlled tests. Keep
   source topology, its six exclusions and cube-wall contact. Earlier contact,
   loop and solver counterfactuals were confounded by the lost wall masks; do not
   promote their settings or treat them as clean post-fix parameter ablations.
2. **P0: complete per-geom contacts.** Independent pad friction 0.7/0.6 is wired
   through SDK-owned interfaces. Finish priority for unlike pad pairs, contact
   softness, equality impedance and the uncached GPU material limitation.
   Normal press/impact characterization above is complete; nonlinear impedance,
   friction regularization and discrete integration are still open. The source
   uses `condim=3`, so its listed torsional/rolling coefficients are inactive.
   Compare actual manifold friction and patch torque before choosing a mapping.
   Source base equality force gains are now derived and tested; their
   position-dependent impedance remains a gap. Convex support
   differs by about 0.6 mm on gripper meshes and up to 2.34 mm on the wrist
   composite. Measure cooking options without changing source geometry. Check
   air/loaded closure, slide/drop, and grasp/lift/release before another 50-query run.
3. **P1: finish state-force and rendering comparisons.** The 28-component
   encoding, retained positive-mass fixed links and external-wrench derivation
   are implemented and independently checked above. Compare measured loaded
   wrenches across engines under identical commands, resolve source reset-force
   semantics, and retain the validity mask until a completed native step.
   The per-contact API still lacks valid tangential impulses; its normal-only
   data validates projected force but cannot replace the six-component scalar.
   Camera/checker alignment, URDF colors and texture bytes are retained.
   Measure projection, lighting and latent-image differences across the 50
   states; matching scene colors does not establish pixel agreement.
4. **P1: isolate planner arithmetic and establish controls.** Match image
   preprocessing, action-encoding batch shape and candidate evaluation order.
   Check uint8 scale multiplication versus division, full-sequence versus
   per-window action encoding, and extra raw/standardized round trips during
   normalized CEM sampling. Compare the first changed elite on identical
   populations rather than relying only on pointwise network tolerances.
   Preserve raw-action and gradient contracts. Retain the original planner as
   a controlled benchmark; add paired random/zero controls on these same queries
   and inspect query-level outcomes.
5. **P2: reproduce training after physical parity.** Match `data=ogb`, seed 3072,
   split 0.9, batch 128, AdamW `5e-5/1e-3`, BF16, gradient clip 1, SIGReg
   `0.09/17/1024`, and epoch-based LinearWarmupCosineAnnealingLR. The pinned
   runnable configuration uses 100 epochs. Resolve split/scheduler semantics
   from the pinned dependencies. The historical 10,000-update UR run does not
   reproduce this recipe. Adapt rendering only after separating visual and
   physical error with the established reference baseline.

Completion requires the parameter manifest, exact asset/runtime identities,
action/IK/force checks, physical replay and rendering measurements, complete
observations and 50 paired evaluations with traceable success decisions.
Keep `dexsim_physics_alignment_verified=false` until these gates pass.
