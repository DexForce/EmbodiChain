# Official Cube migration execution ledger

Goal: reimplement the official MuJoCo LeWM single-Cube example in
EmbodiChain/DexSim with its parameters and validate the migrated physical task.
The user explicitly selected DexSim as destination and MuJoCo as reference.
The full goal remains active; model/kinematic checks alone are insufficient.

## Current checkpoint — 2026-10-31: separate-anchor API and real14 stage oracle

Goal ACTIVE, full alignment FALSE. Previous turn PROGRESS; this turn PROGRESS:
real14 phase-resolved physics evidence, implemented/verified optional GPU point
anchor geometry, metadata completion, full default neutrality and one rejected
integrated task candidate. No task/default promotion, commit/push/PR/publication.
Production task byte-identical to before-source-split-production. Installed and
shared SDK untouched. SDK context read-only; Root simulation context revised,
other routed guides still sufficient. No active validation process at final check.

CURRENT PROVIDER: isolated engine57dcde3308c4c49d9aaccf93d0d2ee04dbb42b9c88b3019cf87a7cf545230f22,
GPUd499a135a2e37a328f349ae39df9bf1c5d4d932c987862759e6c11d84c8b36db.
Actual worker /proc/maps and hashes in alignment/point-anchor-live-provider.json.
Cpp/CUDA builds completed before staging; mismatched oldGPU25d44 rejects enable
and still steps ordinary drive. Fresh113-source/runtime identity manifest
alignment/native-point-anchor-final-runtime.json,108-source archive runtime-sources-
point-anchor-final (preceding111/106snapshot preserved). Prior22/25 binaries and files in reference/before-point-anchor-
extension; prior artifacts immutable. Roota5ea7ee0/SDKf0e752c0 heads unchanged.

Retained API: supports_d6_linear_drive_point_anchors(),
set_d6_linear_drive_point_anchors(name,bool) in existing Scene-thread Constraint/
Arena owner, no new registry/thread/vtable/layout. GPU D6 descriptor496bytes
unchanged; optional private driveflagbit2 changes actor0 angularJacobian toown
anchorA. Flags0acceleration/1outputforce untouched. DefaultD6 still usesanchorB
forboth. Existinggainupdate preservesflag. Scalar exported GPU capability marker
queried from ALREADYLOADED module (no secondGPUload); CPU/olderGPU cannotenable.
RootCfg.linear_drive_point_anchors boolFalse default; positive compliant gains
required, capability preflight beforeallocation, nativefailure rollback. Public
endpoint API retained. Vendor patch 5.6-gpu-d6-point-anchors.patch plus README
under isolated SDK patches; matching CHECKEDsm120 artifact staged. Oldglobal
D6 behavior unchanged: NEW full50 default replay q/qv/actions/NaNs/goals exactly
match previous accepted arrays. It gives32/agreement48/source-only[5,7].
Original/migrated planner33/35 campaigns remain frozen withengine80 identity.

Qualified evidence (alignment/real14-coupled-impulse):
-Source72 cases:4constraintsets none/point/mimic/both x6states x3force tiers
 undamped/passive/controlled. Canonical21 bodies/M/armature/equalities, dt.002;
 contact/limits/gravity/frictionloss disabled ONLYin this isolated test. Shared
 q/v quantized tofloat32. SourceNewton100/tol1e-8 unchanged. Independentdense
 acceleration maxerror6.287e-11, postimplicitvelocity3.037e-12. Sourceimplicitfast
 readsconstraints before M-dt*qDeriv solve; controlledpre/postdelta reaches.3022rad/s.
-Nativefree no-constraint pre-implicit maxerror5.58e-7; offlineSourceD corrects
 postvelocity within5.28e-7. NativePGSmimic hard velocityprojection is independently
 predictedwithin3.55e-7, so0requestedvelocitypasses only works inallzero-pass scenes.
-PhysX articCOM/angular velocities already include h*bias acceleration before
 externaljointprep. Originalscalar/free-body mapping omittedthis. PureURDF COM
 recursion checked1050body/states: pos6.11e-16,vel1.73e-15,bias9.44e-16. Independent
 Native14D rowmodel error .0199->.000126 withbias term. Correctingtarget drops
 Nativepoint/source error .02114->.002842; newownanchorflag drops itto.00012145.
-Mimic time-discretegainmapping plus existing generalized-force velocityterm
 sigma=(1-hB)*Jvold/(hR), tau=J^T*sigma gives totalimpulse=lambdaNative+h*sigma.
 This isconstraintforce and staysOUT of smoothqDeriv. Mimic-onlysourcepreerror
5.60e-7,post5.27e-7. Coupled point+mimic+Sourcepostmass: maxpre.00012093,
 maxpost.00012095 (float32geometry/positiondiff floor). Allthrough publicowners,
 no MuJoCo inNative process. Microphysicspasses do NOT establish loaded task parity.

PRIVATE integrated trial: reference/cube_coupled_source_trial.py loads archived
force/integration code ONLYinside diagnosticprocess (production registry/files
unchanged), enablesownanchors, correctsbias, dynamicalimpedance/timegains/mimic
velocityforce, allvelocitypasses0, sourcecontrolFK timing andpostM-dtD update.
45expertmeanerror2.528->1.269mm, max5.667mm, height.192915/source.191678m;
meanpadforce49.908/source47.535, peak120.899/source49.523. Goodmean butfull50FAILS:
success32/agreement46/source-only[7,32,34],native-only[41]. Frozenpoststep cube
median/p95/max4.903/83.695/253.809mm vsaccepted3.051/41.456/78.555. Same977live
samples worsen7.510/87.367/253.809 vs4.165/55.351/78.555mm. Exactsourceactions/NaNs,
samequeries/seed/25+25/workers6/5. Reports evaluation/dexsim-coupled-source-stage-
source-replay; comparison alignment/coupled-source-full50-comparison.json. No
controller/task recipepromoted. Privateprototype full50elapsed269.35s vsdefault93.49s.


A finalminimal full50 control enablesONLYownanchors ontheacceptedtask, noother
controller/gain/mimic/iterationchanges. Success35 butagreement47, source-only[7],
native-only[12,41], poststep median/p95/max3.400/53.826/129.417mm; exactsource
actions. Thus genericgeometry capability remainsoptional andisnotpromoted alone
either. Report evaluation/dexsim-point-anchor-only-point-source-replay; summary
alignment/point-anchor-only-full50-comparison.json.

NEW CAUSAL NEXTGATE: robot0/cube1 requestedvelocitypass expertcontrol improves
mean/maxerror1.195/3.148mm andpeak108.726N, BUT robotmotion/final difference
returns to.26408rad/s. Distant NONCONTACTING dynamiccube1 real14 control also
causes.03855rad/s difference withrobot0. Thus dependence isSCENEvelocityphase,
not contact itself; GPU artiSolveInternalTendonAndMimicJointConstraints1T loops
allarticulations without per-articulationiterationguard. No mixediteration
candidateadmitted or full50rerun forit. Evidence farbody-velocity-policy.json,
coupled-cube-velocity-grasp-comparison.json and privatefarbody/source scripts.

NEXT EXACT action: qualify explicit opt-in per-articulation PGS soft mimic
velocity policy, preserving finiteERP/CFM/accumulatedimpulse during velocity
iterations instead ofglobal disablingcontactvelocityphases. Keepdefaultlegacy
behavior/TGS/CPU explicit. Reuse existingowner and loadedGPUcapability query,
no newvtable/layout/queue. Public PxArticulationFlags isPxU8 andonlybits0..2used
in5.6; privatebit3 isavailable inexistingGPU mFlags. Newmarker capabilitymaskvalue2 (bit1)
canrejectoldGPU. This is a DESIGNLEAD, NOT implemented. Firstreproduce exactsame
farbody/real14 controls onbaseline, compare3approaches, add focusedcoverage,
thenbuildmatchingprivateartifacts andcheckdefaultneutrality. Do NOT substitute
zero robotiteration count as a compliantpolicy. After this gate, isolate active
source joint softlimits and contactresponse before anotherwhole-task trial.

Parameterexport nowalsoincludes12joint/DOFfields: range/type/limited/margin/
qposadr/dofadr/solref/solimp plusdofinvweight0/frictionloss/solref/solimp, andgeom
solmix/margin/gap. Fresh pinned-source export verifies ALLpreviousfields and64geom
entries unchanged. Newartifact alignment/physical-contract-with-limits.json;
accepted reference directory unchanged. Sourcearm limits solref.02/1,
solimp.9/.95; selectedgripperlimits .005/1,.95/.99. Mimiccompiled inverseweight
397.6325562367748, usefulinstead ofinferringfromprecomputedstates. Export only,
notnativephysicalimplementation; no newexample-target tests. Actualcompiled
source execution plusexistingstatic checks coverthis change. Initial stdin
exportfailedcwd namespace shadowing; correctedRootworktree exportpassed.

Validation: missingAPIbaseline2fail/1defaultpass (v3; earlierfixtureversions had
unapplied legacyPhysicalAttrinertia andGPUbootstrap displacement, fixedwith
explicitinertias/readback andrestoredGPUposes, no toleranceweakening). SDKfocused
3pass, affected102pass/572s. Rootbaseline6newcases fail; 2puremock testsinitially
misclassifiedbynameGPU, renamed (notwaived). Root48constraints/299related pass,
1skip/52Newtondeselected. Existing18AutoSolverCfg failures remainquarantined.
Context42/map/routing pass; Black26.10.1/1734/API3285, C++clang14/PythonYapf and
bothGitwhitespace pass. Newpatch reverseapplycheckpasses. NativeGPUbench1/25
arenas default/ownanchor: hostupdate.01769/.02117ms(B1),.18376/.17610ms(B25);
completedstepmedians1.9427/1.9632ms,2.1471/2.1379ms.5x100after500warmup,
RTXPRO5000Blackwell/cuda5,float32, setup/render/IKexcluded; equilibrium<1e-4m.
No speedupclaim fromnoise or comparison toolderbusyGPU3. Current machine state
VALIDATION.json remainsfalsefor goalcomplete/physicalalignment.

## Previous checkpoint — 2026-10-31: point impulse study, task adapter rejected

Goal ACTIVE, full physical alignment FALSE. This goal turn PROGRESS through an
additive velocity-target capability, independent scalar/free-body/real-robot
oracles, and paired full50 replay. The dynamic point adapter is REJECTED as a
default: its success count improves while physical agreement regresses. Do not
present that count or single-trajectory improvement as source parity.

Retained production changes: native D6 linear velocity setter in existing
Constraint/Arena/pybind owners, Root optional RigidConstraintCfg target and
selected update_rigid_constraint_linear_drive(), focused tests and benchmark.
No vtable/layout/resource owner/thread/queue/default changed. Creation None
makes no new setter call. Runtime updates preflight inputs/capabilities, operate
between explicit steps, and preserve creation cfg; native failures may partially
update rows as documented. SDK context was read-only; Root simulation context
was revised, other routed guidance remains sufficient. No commit/push/PR update.

Current staged engine SHA022c4489d832b5602a0b345d7bc878c22ce2714fad47cafac875de2d19bd6daf,
GPU25d44eabc37e1fe32b8992d8e4cf84d30c7c0d7a3e5fac395c206dd4b6c90e39.
Private matching Release/cp311 build completed and formatted/rebuilt before
staging. Shared/installed SDK untouched. Full78-file identity manifest
alignment/native-d6-velocity-target-runtime.json and73-source frozen archive
runtime-sources-d6-velocity-target include the private trial, not just base task.
Root task byte-for-byte matches before-source-split-production. New unused API
preserves all45 recorded expert states/wrenches/contact metrics exactly. A fresh
50-query25+25 baseline also preserves ALL qpos/qvel/actions/NaN/goals arrays
exactly versus the accepted replay. New baseline32/agreement48/source-only[5,7].
Original/migrated planner33/35 reports remain frozen with their old engine80c085fb
identity; they were not rerun for the unused capability. Physics recipe unchanged.

Independent evidence:
-18 scalar source cases: residual0/10/300um, velocity0/+-.1m/s, load0/1N,
 masses1/2kg andarmature.005, dt.002. Time gains alone remove zero-v error;
 full velocity target gives maxv9.94e-9m/s, impulse7.45e-9Ns. Requested0/1native
 velocity iterations agree only for this fixture. Four live-state nonlinear
500-step native trajectories (no MuJoCo runtime in native loop) give maxq
1.211e-7m/maxv6.419e-8m/s. Reports alignment/point-step-response.
-Moving connect includes source Jdot*v.8free-body cases plus independent12D
 impulse solve distinguish it from target-frame/force-Jacobian effects. Identity
 coincident moving case1 maxv1.148e-8 afteranalytic Jdot. At nonzero separation,
 D6 appliesactorA force atanchorB; source usesanchorA. Max remaining velocity
 component error1.442e-4. Native12D model predicts within8.143e-7, source within
8e-16. Do NOT silently change global D6 semantics or assume scalar proof generalizes.
-Pure URDF kinematic recursion matches both real point rows at all50official
 states: maxposition2.22e-16m, velocity1.94e-16m/s, Jdot8.53e-16m/s2. Private
 cube_point_kinematics.py and robot-kinematics-source.json preserve exact evidence.

Private point trial uses nonlinear live d/R, time-discrete gains, Jdot and the
D6 velocity-Jacobian correction; it cannot remove D6's different moment arm.
It leaves accepted actuation/integration unchanged and uses no MuJoCo in native
loop.45-command cube maxerror5.764->5.729mm, height.189152->.189230m;
pad force norm peak130.857->115.406N (source49.523), mean47.829->47.209N
(source47.535). Maxloop.431mm, Jdot2.138m/s2, target.01372m/s. Six-query private
layout has agreement4/6 for both baseline/trial but different false positives;
these layouts are NOT substitutes for formal25+25.

Full50 exact-source-action replay (samequeries/seed/sourceactions includingNaNs,
workers6/5): baseline32, trial34, bothagreement48. Trialsource-only[7],
native-only[12], so one recovered success is offset by a new falsepositive.
Frozen post-step cube median/p95/max3.051/41.456/78.555mm ->
3.647/62.717/120.578mm. Same973live-state mask through earliest success across
all3runs ALSO worsens4.165/39.269/65.841 ->7.162/63.087/102.441mm. The regression
is not solely freezing different success times. Fullreports
 evaluation/dexsim-point-impulse-{baseline,point}-source-replay; comparison
alignment/point-impulse-full50-comparison.json. Private trial remains onlyunder
reference/ and frozen archive; NO production task/helper/compute added for it.

Validation: native missing-API baseline2fail -> CPU/GPU2pass; affected SDK99pass.
Root missing-field baseline5fail -> focused42pass; related293pass/1skip/
52Newton-deselected. Context42pass/mapcheck/routing Chinese+symbol pass;
Black26.10.1/1734files, API3285exports, C++clang-format14/PythonYapf and both
Git whitespace gates pass. Existing18AutoSolverCfg failures remain quarantined.
Point-drive benchmark1/25 arenas: hostupdate means.02075/.18253ms, completed
step medians2.6951/2.8163ms, creation.5628/1.3467ms;5x100steps after500warmup,
float32 PGS, RTXPRO5000Blackwell. Rendering/IK/setup excluded, equilibrium<1e-4m.
Private trial full50 elapsed145.63s versusbaseline93.00s is not a speedup claim.

Pre-existing GPU failure separately quarantined: all shapes collision-disabled,
GPU standalone-body pose writes afterinitialization, thenstep causes
markUpdatedPairsLaunch broadphase.cu440 write toNULL. Reproduced on prior
engine80c085fb with NO constraints and oldpybind. Baseline stage and logs
reference/d6-baseline-runtime, alignment/point-rotating-before-api-no-constraint-
sanitizer.log; currentno-target sanitizeralso fails. No production fix guessed.
Corrected oracle/benchmark use collision-enabled, spatially separated shapes
with no contact. Do not discard failure or misattribute it to new drive setter.

Next EXACT action: establish a real14 coupled point/mimic impulse oracle with
identical q/v and independent source pre-implicit constraint acceleration,
post-implicit velocity, and native PGS motion/final velocity. Isolate point-only,
mimic-only and combined rows before another integrator/task change. Then assess
a source-compatible compliant point row through existing native owner; preserve
D6 behavior and public ABI. The scalar target is valid only for matching row
Jacobians and stage boundaries; actual actuation/contact coupling is unresolved.
Do not optimize/promote the rejected NumPy adapter or fit gains to success count.
Remaining goal also includes nonlinear contact priority/friction/limits, reset
force validity, rendering phase/pixels, CEM5.379 arithmetic, pairedzero/random
controls and originaltraining. Canonical machine state VALIDATION.json remains
false for fullalignment. No live validation processes remain after final checks.

## Previous checkpoint — 2026-10-31: rejected source-servo phase rolled back

Goal ACTIVE, full physics alignment FALSE. Previous goal turn PROGRESS; this
turn PROGRESS through independent coupled/real14 physics evidence and completed
candidate implementation/evaluation, followed by verified rejection and rollback.
Do NOT present the candidate as a passing task. No commit/push/PR/publication.

Active Root task/test/evaluator files are restored EXACTLY from
reference/before-source-split-production. New candidate compute/dynamics files,
benchmark and dynamics testfixture were backed up then removed from active repo.
Rejected code/fulltrackedpatch/rollbackJSON: alignment/rejected-source-servo-
production. Prior user/unrelated edits preserved. Current51 source/runtime
hashes match lastaccepted native-compliant-contact-runtime except the updated
OFFICIAL_CUBE documentation. Fresh manifest/archive: native-restored-pgs-runtime
and runtime-sources-restored-pgs. Task focused tests24pass afterrestore; broader
restore check521pass (2skip/75deselected). Black1733/API3285/context map pass. Provider unchanged engine80c085fb /
GPU25d44, installed/shared SDK untouched. No live taskprocesses after probes.

Lastaccepted PG measurements remain source34, original33/agreement49, migrated35/
agreement49, source-action replay32/agreement48. Rejected initial source-servo
campaign: replay32/agreement46 (source-only[5,32,34],native-only[41]); original33/
agreement47 (source-only[32,34],native-only[41]); migrated33/agreement47
(source-only[32,34],native-only[12]). Those3frozenreports are preserved, not
reused as current accepted results. Allfullalignment flags remainfalse.

Valid study results: real14 mass error1.94e-6; generalized force<9e-13; source
mass-tree sparse velocity derivative exact. Source ignores sibling tendon
cross-terms2.5 in this pattern. Coupled free clip/integration matches~2e-8rad.
Controlled45 candidate improved cube5.764→3.003mm, arm~8.37→4.290mrad, padpeak
130.857→102.627N, but full replay worsened: median/p95/max3.273/47.501/235.731mm.
Hence single-path improvement was insufficient and candidate was removed.
Rejected candidate tests32+529related (2skip/75deselected), context42/API3285/
Black1736 and numericB1/25/50 benchmark passed at their frozen source revision;
these are candidate evidence, not current-tree counts after rollback.

Causal source/native constraint mismatch studied: PGS motion/final velocities
differup to.556rad/s due hard zero-ERP/CFM velocity passes. A motion-input andB
preserve-position adapters restoredriver precision but both failprivate6query
outcomes [T,T,F,F,F,T],agreement3/6. Taskroot+Cube velocityiterations0 removes
motion/final discrepancy to5.96e-5 but worsens same6outcomes to[T,F,F,F,F,T],
agreement2/6. None are retained. Baseline/private layouts differfromformal25+25;
do not substitute their counts. Source constraint regularization/impedance and
Native update law must be matched together; integrator compensation alone fails.

Next EXACT action: characterize one-step source point-equality impulses versus
native D6 for independent position/velocity/load inputs. Source row is
R*lambda+J*vnew=(1-hB)*J*vold-h*K*d*r. Matching an implicit spring requires
Kn=K*d/R, Dn=(1/h-h*K*d)/R and velocity target
vtarget=(1-hB)/(1-h^2*K*d)*J*vold (when positivity conditionshold). This is a
hypothesis derived from scalar row equations, not a verified native mapping.
Existing D6 creation sets drivevelocity0; SDK exposes velocity setter onlyfor
revolute constraint, so read existing constraint owner/design before extending
D6 API. Reuse public Spawn/constraint owner and one-step oracle, no raw mutation
or solver fork by guess. Verify actual velocity-pass policy/CFM at this gate.
Do not reintroduce whole-task sourceintegration until constraint impulse match
passes. Scope stays ALLofficial task parameters/native task/evaluation.

Archived candidate diagnostics remain runnable via
reference/load_source_servo_archive.py (import smoke passed). It adds frozen
compute path and installs rejected task module only inside diagnostic process,
without replacing production registry/files. CandidateA/B/zero-pass scripts
bootstrap this loader. Independent prototype cube_source_split.py/source export
and allv1-v4coupled logs remain under reference/alignment. No processes live.
Remaining fullgoal: nonlinear contacts/point+mimic transient/softlimits, reset
force validity, rendering phase/pixels, CEM5.379arithmetic, pairedzero/random
controls, originaltraining. Old4/100SDK mimic sweepinvalid/withdrawn; current
benchmark uses default iterations.18NewtonAutoSolverCfg failures quarantined.
Canonical machine status VALIDATION.json; do not markgoalcomplete.

## Point impulse / velocity-target phase — active

Previous goal turn PROGRESS. Current cold read verifies restored Root/SDK heads,
dirty scope and provider engine80c085fb/GPU25d44. No live preceding processes.
New18-row independent source oracle matches scalar impulse equation to1e-12.
Source parameters solref.005/1,solimp.95/.99/.001/.5/2,dt.002,1/2kg sliders,
armature.005. Grid residual0/10/300um, velocity0/+-.1m/s, opposingload0/1N.
Native baseline has zero drivevelocity. Base source-force gains zero-velocity
maxerror5.279e-4m/s; time-discrete gains reduce that to1.315e-9m/s. Nonzero-velocity
maxerror1.223e-2m/s remains. PGS requested0/1velocitypass gives sameD6spring
result (not assumed equivalent for otherconstraint types). Reports under
alignment/point-step-response, scripts point_step_response.py.

Requirement: expose existing native D6 linear velocity target in actor0 joint
axes through current owning-thread constraint/Arena and Root creation config.
Unitsm/s; positive target is actor1 relativeactor0; preserve angular target,
positiontarget,gains,motion,lifetime,existingdefaults/vtables. Shape3 finite;
released/nonD6false,unknownnameIndexError. Root optionalNone does not callnewAPI.
This is additive/local, not critical lifecycle/ABI direction. Three approaches:
A keepzero-target/nativegainonly (verifiedinsufficient for movinginput);
B expose native target through existingowner (recommended, no newsolver/state);
C customvendorrow (unnecessaryifB passes exactone-step oracle). ChooseB.
Reusing existingConfigureD6LinearDrive/SceneThread::Sync/Arena registry and Root
constrainttransaction; no thread/queue/native resource owner added. Snapshot
reference/before-d6-velocity-target includes currentfilesandruntime libs.

Acceptance: missingcapability baseline tests; nativeCPU/GPU physical spring
response withnonzero target, finiteinput/lifecycle checks; Rootcreation/selected
arenas/transactionrollback;18source row state/force impulses includingtimegains
andderivedtarget. Onlyafteroraclepasses testlongertrajectory/actualgrasp; no
currenttaskrecipechange or scoreclaim from APIalone. Benchmark creation/step
cost without timing claims beyondmeasuredscope. Nativebuildmustfinishbefore
staging; restoreGPU25d44 ifCMake stagesprebuilt. Installed/sharedSDKuntouched.
## Rejected source-servo execution record

Goal ACTIVE, full physical alignment FALSE. This goal turn PROGRESS: derived and
validated coupled source integration, implemented14-DOF clipped forces/sparse
velocity derivative/mass update/control-FK timing in the task, added independent
source goldens and benchmark, completed32focused+529related tests and3formalruns.
BUT the new source-servo runtime has not passed the full physical gate.
No commit/push/PR update/publication; installed/shared SDK remains untouched.

Current Root a5ea7ee0 worktree contains uncommitted source-servo implementation
in taskogbench_cube.py, private compute/dynamics/_inertia.py +__init__, new
benchmark_ogbench_dynamics.py and dynamics.json golden. SDK is unchanged from
paired engine80c085fb/GPU25d44. Source runtime archive/55 hashes:
alignment/native-source-servo-runtime.json, runtime-sources-source-servo.
Old accepted PG recipe/source snapshots remain under native-point-pgs-runtime,
native-compliant-contact-runtime and reference/before-source-split-production.
All3new formalruns are terminal. Do not mutate or overwrite their reports.

Frozen new runs on same50/seed/protocol (source34): exact-source-action32,
agreement46, source-only[5,32,34], native-only[41]; original planner33, agreement47,
source-only[32,34], native-only[41]; migrated33, agreement47, source-only[32,34],
native-only[12]. Raw replay median/p95/max3.273/47.501/235.731mm worsens preceding
PGS3.051/41.456/78.555mm. Thus do NOT present single-trajectory gain as fullpass.
Formal reports under evaluation/*source-servo; summary source-servo-frozen-
evaluation.json. Previous migrated35/50 is historical, not currentresults.

Verified component evidence:50query mass maxabs1.94e-6, generalized force<9e-13,
source sparse velocity derivative exact (full sibling tendon cross-terms differ
2.5 and must be projected away). Coupled free trajectories with source force
clipping+postintegration match~2e-8rad. Diagnostic constant contacts are NOT
source recipe; contact activation envelope sensitivity quantified separately.
Production45 expert improves cube5.764→3.003mm, arm~8.37→4.290mrad, padpeak
130.857→102.627N (source49.523), mean48.229/source47.535N. Physics remainsfalse.
32task tests +529Root Default related tests pass (2skip/75Newton deselected),
42context tests, API3285 and Black26.10.1/1736files pass. New numeric benchmark
B1/25/50 medians1.05/1.06/1.09ms, excludes native step/stateIO/IK/rendering.

Regression cause under test: PhysX PGS final velocities omit stabilization bias
retained in position integration. New source post-step overwrites q with qold+
h*vnew and discards it. Measured motion versus final joint velocity differs
up to.556rad/s; driver equality median worsens1.55e-6→1.21e-4rad. Native mimic
code explicitly dropsERP/CFM in velocity iterations (source inspected).

Private candidateA uses (q_physics-qold)/h as integration velocity; candidateB
keeps final velocity and preserves q_physics via h*(vnew-vfinal). B is identity
for zero derivative on both position/velocity; A changes native velocity there.
Neither candidate is retained. Controlled45 A/B cubeerror5.026/5.503mm, arm
3.567/3.539mrad, driver max.150/.155mrad. Private same-layout6-query baseline
andB both outcomes[T,T,F,F,F,T] at indices[5,7,12,32,34,41], agreement3/6, soB
restoresdriver median(.618mrad→.00268mrad) but DOES NOT fix full outcome regression.
Do not claim it resolves the failures. Private6 differs from formal25+25 layout.

Next exact action: wait for current no-velocity-pass private6 query replay,
session95589 is the completed candidateA run; the NEW live handle is recorded
in tool output, with log alignment/servo-position-subset-no-velocity-pass.log
and device3. CandidateA6 is terminal and also gives[T,T,F,F,F,T], agreement3/6;
neither A nor B is retained as a full fix. Current minimal candidate configures
robot/Cube velocity iterations0 (100position unchanged), matching absence of
an extra hard velocity pass in the source. Its45 replay observes motion/final
velocity difference5.96e-5rad/s instead of.556, confirming the causal stage is
removed. Cubeerror5.354mm, arm3.998mrad, driver max.163mrad,padpeak120.876N.
Private6 must decide whether this restores lost queries before any production
config change. No production/library mutation has occurred after initialfrozen
campaign. Preserve zero-derivative/native projection invariance in any adapter;
if no candidate fixes the full gate, back up and rollback only this phase or
return to source constraint semantics rather than stacking guesses.

Other open requirements: nonlinear contacts/point+mimic fulltransients/softlimits,
source reset-force semantics, rendering sampling phase, neural initialCEM5.379,
zero/random controls, originaltraining. Source cached pinch FK is pre-final2ms;
source cube observation and success use current jointqpos, DO NOT delay them.
Old4/100 SDK mimic benchmark sweep was invalid; only current default-iteration
benchmark is valid.18Newton AutoSolverCfg failures remain quarantined/reproduced.

## Previous contact / servo execution records

Goal ACTIVE, full physical alignment false. Previous PGS turn PROGRESS; current
turn PROGRESS. Destination remains EmbodiChain/DexSim, official MuJoCo reference.
No commit, push, PR update, publication or installed/shared SDK replacement.
Root a5ea7ee0 / SDK f0e752c0 worktrees remain dirty with prior work preserved.

Latest formal50-query results are frozen prior-PGS measurements: source34,
original planner33 (agreement49), migrated35 (agreement49), exact source actions32
(agreement48). This continuation does not claim a new50-query evaluation.
The task's contact and servo recipe is unchanged. A45-command replay after the
additive capability has bit-identical qpos, qvel, cube pose and external wrench
versus the frozen GPU25d44 result (all max differences0).

Implemented: DefaultRigidBodyMaterialCfg and SDK DexsimCollisionDesc expose
keyword-only native constant contact stiffness/damping, preserving None/sparse
intent. Reuses atomic per-shape materials, named-URDF matcher, source overlay
replay and current cache/thread owners. No C++/solver/library change. Binary
identities remain engine80c085fb / GPU25d44eab. Source/runtime text archive and51
hashes: alignment/native-compliant-contact-runtime.json and
alignment/runtime-sources-compliant-contact. Historical formal78 checks remain
in native-point-pgs-runtime.json with its own source archive, not currenthashes.

Independent12 normal fixtures: source mixed floor/cube tau.012, pad/cube tau.004,
actual solimp/priority/friction/condim, cube mass.07936kg, dt.002. Static flattened
pad is explicitly separate from articulated pad inertia. Both source/native
normal forces satisfy momentum balance. Source/base-spring low-load floor box
compression39.275/39.337um and pad2.131/2.129um agree closely. But floor extra20N
compression623.425/1050.204um and pad-box impact peaks26.744/20.405N do not.
The constant source-base spring mapping is NOT adopted as full source contacts.
Report/plots: alignment/normal-contact-source-native-comparison.{json,png,svg}.

Servo oracle:12 source cases x2native mappings, official3 gain sets/force caps,
armature.1, dt.002 on isolated inertia.1 hinges. Explicit force flag and momentum
validate native caps. URDF import already sets this flag for the task; no global
force-limit change is warranted. Kd_native=Kd-Kp*dt matches all unsaturated cases
to max3.054e-10rad, but saturation still differs up to7.0005mrad. Source clamps
old-state force and skips its velocity derivative when clamped; native implicit
drive saturation differs. Gain-only correction is NOT adopted. Report/plots:
alignment/implicit-pd-comparison.json and implicit-pd-source-native-comparison.

Checks GREEN:195 Root material/schema/source tests; separate120 task/model/state/
configuration tests (1optional GPU skip,39Newton deselections);42context tests;
Black26.10.1/1733 files; API3285/3285; context route/map and whitespace. SDK broad
110pass, then final30articulation cases pass after removing duplicate fake-sweep
coverage (overlap, not140). AdditionalRoot command initially aborted during
renderer engine initialization because SDL_VIDEODRIVER=dummy was omitted; the
identical selection passes after restoring that environment, with bothlogs saved.
The195 material cases and110 provider cases were already green. No code fix or
check waiver was used for the startup error.

IMPORTANT VALIDATION CORRECTION: old SDK mimic test/benchmark called public
solver-iteration/sleep methods that only warn and ignore input. The old4/100
sweep labels and claims are withdrawn. Removed those calls and the duplicate
regression/parameter; no new public lifecycle semantics are introduced. Rerun
benchmark25/256 chains at backend-default iterations:2/2 pass,1.443/1.559ms,
maxspringrelative2.91e-7. New compliant-contact/query benchmark25/256 independent
bodies in one Arena:2/2pass,50 measuredsteps each, no droppedrows, stablecounts.
No speedup claim. Task root100 uses its existing pre-GPU native adapter; private
native100 probe is distinct evidence, not a justification for mislabeledrows.

Classified fixture errors are retained/excluded: Builder cube size must be tuple;
GPU initialization advances2x.01s and requires resetting initial state; selecting
one dynamic body against static support requires contact match='any'; initial
PD inline fixture lacked force-units flag and requested invalid ±10rad hinge
limits. Corrected PD uses force flag, ±3rad nonbinding limits, momentum cap checks.
No production task change was based on invalid fixture results. Historical18
Newton AutoSolverCfg failures remain independently reproduced; no unrelatedrepair.

Next exact action: extend source/native servo oracle to coupled hinges plus a
support/contact case, record old-state force saturation and qDeriv, then compare
source-compatible control alternatives. Single-hinge gain agreement cannot prove
coupled contact/control parity. Keep controller/contacts unchanged until a full
physical oracle passes. Nonlinear normal impedance, friction regularization,
soft joint limits, mimic/point complete transients, reset-force semantics, initial
CEM difference5.379, rendering/latent alignment, zero/random controls and original
training remain open. Canonical machine status is VALIDATION.json; goalcomplete
and full-physics flags remain false.

## Coupled servo phase — active

Previous goal turn classified PROGRESS. Current cold read verifies Root/SDK HEAD,
dirty scope, canonical ledger and51 runtime/source hashes; engine/GPU unchanged.
No outstanding live processes. Read official mj_implicitSkip/mjd_actuator_vel:
constraint force is computed with M, then implicitfast solves
(M-h*qDeriv)*a = qfrc_smooth+qfrc_constraint and advances v/q. Saturated actuator
velocity derivatives are omitted. Native drives instead participate in the
constraint solve with their implicit response; a constant damping adjustment
cannot prove coupled/contact agreement.

Scope now: private oracle, no production change. Compare original native gains,
Kd-Kp*h, explicit clipped PD, and explicit clipped PD followed by the source
velocity integration. For fixed M the latter is vnew=vold+(M+hD)^-1 M(vphys-vold),
qnew=qold+h*vnew. Source and native model two coaxial, inertially coupled hinges
with analytically known M; independent impulse checks validate native M/armature.
Add a frictionless normal stop to isolate contact/control ordering, with all
non-official diagnostic contact choices explicitly labeled. Record source M,
qDeriv, clipped force, constraints and integrated acceleration. Reset after native
warmup and preserve supported force-unit flag. No unsupported root setters.
Acceptance is numerical state/force comparison by scenario, not a task-score fit.
Green -> qualify model/state integration plus needed public capabilities before
adoption; red -> preserve/classify the failed oracle, leave production unchanged.

## Real14 servo model — active

Coupled oracle results preserved under pd-coupled-stage-oracle-v1/v2/v3/v4.
Native impulse M agrees with analytic M (max1.5e-7 in v1). Source-order explicit
clipping plus implicit velocity solve gives free coupled-motion max2.04e-8rad,
including saturation; original drives differ~7.5mrad. A diagnostic constant
contact with h*B=1 admits a discrete PhysX coefficient map: K_native=Kd/R,
D_native=(1/h-h*Kd)/R. This is not the official nonlinear contact recipe.
With native1um offsets, positive-gap anticipatory impulses remain and produce
contact error~0.283mrad. Native0.1nm loses the initial nanometre contact due
finite-precision detection. Native10nm envelope retains initial contact and
reduces all4 stop cases to max8.331e-6rad (velocity max.00355rad/s remains at
activation), while near-limit/saturated stop cases are~1.62e-7rad. No production
contact-offset/servo change is accepted from this diagnostic alone.
Root poses are explicitly checked; cells translate along spin axis to avoid
world-X/Y lever-arm cancellation. Source/reference contact settings and failed
alternatives remain saved. Public joint/root writes are used via Root facade;
no unsupported root setters or native mutations were introduced.

Now validate actual14-DOF mass and smooth-velocity derivative before any task
adoption. Prototype uses public native link poses/parent-joint topology and
already-declared21-body inertias. Cache masks/topology; form batched Jacobians
and J^T I J; retain float64 and armature. WorkloadB=1/25, L=21,D=14, every2ms;
O(B L D^2), roughly0.6MB temporary Jacobians atB25; dispatch/transfers and small
linear solves likely dominate. Apply implement-performance before retention.
Source implicitfast projects qDeriv into mass sparsity, so explicitly compare
full and projected derivatives against MuJoCo on50 frozen query states with
independent random controls. Staged runtime51 hashes remain unchanged.
Next exact action: finish source derivative/mass export, then native model audit;
red -> classify/correct prototype, never change task based on mismatchedmatrix.
## Real14 matrix/derivative audit — verified

Private SourceDynamics prototype (reference/cube_source_split.py) validated on
50 exact official query states and independent seeded controls. Native public
link poses/topology plus declared21-body inertias reproduce source generalized
mass to1.940e-6 absolute. Analytic generalized passive+actuator forces agree
8.953e-13. Projected velocity derivative agrees exactly. Unprojected sibling
tendon cross-terms differ2.5: MuJoCo stores only mass-tree sparsity in this
model, so the mathematically full tendon derivative must NOT be used as the
source implicitfast update. This is now an explicit model option/oracle.
Reports: alignment/source-split-model/{source.npz,native.npz,native-report.json}.
Next: private real-task controlled45 replay with original drives disabled,
old-state source clipped force and sparse derivative, followed by the verified
mass-weighted velocity update. Keep contact recipe1um/PGS/source point gains
unchanged to isolate this factor; record raw-stage contact wrenches separately
from post-state observations. No production code mutation or formal score yet.

## Source servo production phase — authorized and active

Private real14 controlled45 replay confirms source-order integration plus cached
pre-last-substep control FK improves cube-path max5.764→3.003mm, arm max~8.37→4.290mrad,
height .192330m versus source .191678m, pad peak130.857→102.627N (source49.523),
mean steps20–35 48.229N (source47.535). Integration without control-phase alignment
has arm12.912mrad and is not adopted. Current gains/point/mimic/contact inputs
stay unchanged except native arm drives disabled in favor of explicit source
actuation and source implicit velocity update. This is no new50-query result.
Full physical parity still false; source nonlinear contacts/limits remain open.
Raw-stage wrench normal projections are much more accurate than reconstructing
with pre-step pose, confirming native measurement frame must be respected before
shifting moment reference to the source subtree COM. MuJoCo set_control reads
cached site FK from before the last2ms step; observation pinch FK likewise cached.
Cube observation and success read current free-joint qpos, so DO NOT delay cube
state or success. Rendering phase remains a separately unverified requirement.

Implementation choice: retain existing native PG solver/ownership and public
state APIs, adopt verified source clipped-force/sparse derivative + mass-weighted
velocity integration and control FK cache. Rejected constant-gain workaround
fails saturation/contact coupling; a vendor integrator rewrite is unnecessary
for this proven stage and remains out of scope. Shared numeric mass/integration
lives privately under compute/dynamics (no lab imports); task owns topology and
source-actuator conversion. Defaults of other tasks/backends remain unchanged.
Production rollback snapshot: reference/before-source-split-production.

Focused coverage: retain six original controller force scenarios with independent
new MuJoCo full-force/derivative goldens, add3actual14-mass states and3coupled
integration source goldens, source-control FK phase/selective-reset coverage.
No tutorial tests. Representative B1/25/50 numeric/control measurements and real
controlled45 replay precede new paired50 validation. Preserve old score/runtime
archives. Green -> paired source-action/original/migrated50 runs on one frozen
runtime; red -> minimal causal fix or revert only this phase after backup.
## Source servo frozen validation — active

Production control/integration and cached source control/state pinch FK are
implemented. Native built-in arm drives disabled; gains/force caps applied by
source actuator law. Public state APIs publish the verified velocity step.
Raw-stage native RNE recovery is cached before rewriting state; torque reference
shifts to source pre-step subtree COM, preserving measured normal projection.
Cube state and success remain current, matching source code. New private numeric
operations under compute/dynamics have no simulator imports/public exports.
Six old passive-only force tests are replaced by the same6 source scenarios
checking full force and sparse derivative;3 source mass cases,3 coupled-step
goldens and reset/frame/moment coverage give32focused passes.529related Root
Default-scope regressions pass (2skip/75Newton deselected). API3285/3285 and
Black26.10.1/1736 files pass; context check/gate pending final read.

Production45 replay reproduces private source-control-phase prototype exactly:
cube maxerror3.003mm, maxheight.192330m, arm maxerror4.290mrad, padpeak102.627N,
mean20–35 48.229N, normal-projection maxrelative1.40e-5. Source peak49.523N;
full physics still false. Pure numeric GPU benchmarkB1/25/50 has medians
1.05/1.06/1.09ms,100repeats, float64, quality residual<4e-16; excludes native
stepping/stateI/O/IK/rendering. No speedup claim or full-task latency claim.

Frozen runtime/archive: alignment/native-source-servo-runtime.json and
alignment/runtime-sources-source-servo. Staged provider remains engine80c085fb /
GPU25d44; current Root task and new numeric module are included in evaluator
hashes. New exact source-action50 replay is live under
reference/source-servo-source-action-replay.log, workers6/5, planner7. No runtime
code/library mutation until all3formalruns terminal. Green replay -> original
planner then migrated planner on same frozen code; red -> classify before any
code change. Scores still unmeasured for new recipe; never reuse prior35asnew.
## Formal source-servo result — regression under diagnosis

Exact-source-action50 run terminal:32/50, agreement46/50, source-only[5,32,34],
native-only[41]. Median/p95/max cube trace error3.273/47.501/235.731mm versus
priorPGS3.051/41.456/78.555mm. Single45trajectory improvement does NOT establish
full validation. Original planner run terminal33/50, agreement47, source-only
[32,34], native-only[41]. Migrated planner live under source-servo-migrated-
planner.log (same frozen files); finish the frozen characterization before edits.
Runtime files remain frozen, no libraries changed. Current source-servo version
is not accepted as final physical parity.

Driver equality residual median rises from prior1.55e-6 to1.21e-4rad (source
1.26e-5). Hypothesis: using final PGS velocities loses biased motion retained in
joint-position integration (mimic/contact/limit velocity iterations can remove
position bias). The source post-step then overwrites the native position
stabilization. Private MotionCube on device3 tests motion velocity=(q_physics-
q_before)/dt as integrator input and records its discrepancy from final velocity.
This is a falsifiable diagnostic, not yet a retained fix. Script
reference/probe_source_motion_velocity.py; log alignment/source-motion-velocity-
probe.log. If confirmed, validate coupling/momentum/force semantics and paired
replay before accepting a runtime edit; otherwise revert only this hypothesis.
## Initial source-servo frozen campaign — terminal, not accepted

All3 frozen runs terminal. Raw-source replay32/50, agreement46, source-only
[5,32,34], native-only[41]; original planner33/50, agreement47, source-only[32,34],
native-only[41]; migrated33/50, agreement47, source-only[32,34], native-only[12].
All16 initially-near queries succeed (verify reports in final campaign audit).
No parameter/protocol edits occurred between these runs. Frozen reports/runtime
are preserved as a rejected full-physics campaign, not erased or called parity.

Native mimic code explicitly drops ERP/CFM in velocity iterations. Independent
45-step diagnostic observes joint motion velocity versus final velocity differs
up to.556rad/s, confirming position-stabilization loss is material. CandidateA
uses position-derived motion velocity; candidateB preserves native position
projection and only adds h*(v_new-v_final). B is identity for zero smooth
velocity derivative on both native position and velocity; A changes velocity
in that case. Neither candidate is retained yet. Controlled45 A/B cube maxerror
5.026/5.503mm, arm3.567/3.539mrad, driver error.150/.155mrad; both improve driver
stabilization but require broader replay qualification.

Private6-query same-layout baseline (device3, indices5/7/12/32/34/41) gives
[T,T,F,F,F,T], agreement3/6, driver median.618mrad; different batch/layout from
formal25+25, so do not substitute its scores for formal50. CandidateB preserve-
position private replay is live under servo-position-subset-preserve.log.
No production runtime mutation until this controlled causal comparison closes.
## Velocity-pass cause — next controlled test

CandidatesA/B both private6-query outcomes equal baseline[T,T,F,F,F,T],agreement3/6;
although driver median improves to4.28/2.68e-6rad from6.18e-4, neither fixes lost
queries32/34. They are not retained as full fixes. Next minimal source-semantic
candidate: configure native velocity iterations0 for robot and dynamic Cube,
keeping100 position iterations and source controller unchanged. Source MuJoCo
has no extra hard zero-bias velocity pass; inspected native mimic code uses
ERP=CFM=0 whenever isVelocityIteration. Removing that task-local stage may make
native motion/final velocity consistent and retain source spring impulses.
This uses existing Root creation config; no SDK/PhysX edits or new owner.
Private45 script probe_source_no_velocity_pass.py runs oncuda3; it records actual
motion/final velocity discrepancy to falsify the hypothesis. Then compare the
same6-query layout before any production config change. No gain/contact tuning.
## Prior PGS checkpoint — 2026-10-31: PGS / source force units / GPU compliance

Historical checkpoint. Its 4/100 benchmark/test sweep claims are superseded
by the current correction below and must not be reused. Goal ACTIVE, full physics alignment false. Previous turn PROGRESS; this turn
PROGRESS: repaired actual scene-creation configuration timing, exposed existing
PGS selection with defaultTGS preserved, derived force-unit point gains from
source inverse weights, restored PGS friction-iteration flag and GPU mimic
compliance, and completed controlled/benchmark/paired evaluations. No commit,
push, PR update, dependency publication or installed/shared SDK replacement.

| Frozen new recipe | Success | Paired outcome agreement |
| --- | --- | --- |
| MuJoCo reference |34/50|reference|
| Native + original planner |33/50|49/50: source-only[7], native-only[]|
| Native + migrated planner |35/50|49/50: source-only[], native-only[12]|
| Native + exact source actions |32/50|48/50: source-only[5,7], native-only[]|

Same50 queries/checksum/seed42/H1/block5/CEM300x30, persistent25+25 workers6/5,
planner7. All16 initially-near queries succeed. Replay actions/NaN masks exact;
cube errors median3.051mm/p95 41.456mm/max78.555mm (priorTGS3.56/81.11/248.60mm).
Original first raw plans/first25 common finite controls bit-identical to source.
Migrated initial plan still differs5.379. Higher count is not complete parity.

Controlled45 commands: source/priorTGS/newPGS maxheight.191678/.231742/.189152m;
maximum cube path error falls41.05→5.76mm. Mean right-pad force steps20–35 is
47.535/56.865/47.829N, but peaks49.523/114.435/130.857N: impact softness is still
wrong. See expert-pgs-source-comparison.{png,svg,json}. Known point loads match
source base force units under actualPGS; mimic torque equilibrium2.71495969e-6rad
versus source2.71492317e-6,4/100 iteration independent; all4TGS control trajectories
bit-identical. Max mimic transient position discrepancy2.584e-7rad remains.

Runtime frozen at engine80c085fb / GPU25d44eab (PhysX5.6 CHECKED,Clang14,CUDA12.8,
sm120). GPU combines earlier measurement accounting plus PGS accumulated-CFM
repair; CPU PGS mimic unchanged/unverified. Source export and native audit pass
100 configurations,21 masses/inertias,25 frames,total22.04750833884274kg. New
reference alignment-point-response-pgs includes body_invweight0 and source sleep
intent; old reference directories remain historical. Actual report descriptors
confirmPGS/friction-every-iteration/gravity. Task sleeps disabled; reference
metadata/gravity/solver guards fail early. CCD+DirectGPU is explicitly unsupported.

Checks:560Default-scope Root tests pass,2optional skip/75deselected; separate42
context gate passes (overlaps selection, not an extra total).93SDK regressions
pass,3unselected unrelated shape cases. Black26.10.1/1733files,API3285/3285,
context map/routes/whitespace pass. Benchmark4/4 (25/256 chains ×4/100 position
iterations)1.40–1.60ms completed step, spring maxrelative2.91e-7; no speedup claim.
Runtime/source archive and78 hash checks: alignment/native-point-pgs-runtime.json
and alignment/runtime-sources-point-pgs-compliant. Canonical status VALIDATION.json.

Classified limits: pairedSDK lacks AutoSolverCfg;18Root Newton config tests fail
identically with the archived old backend. No waiver and no unrelated SDK API
repair. Saved creation descriptor is not every live value: measured gravity
was already applied by old code; both versions yieldvz-.300 after.1s atg=-3.
The timing fix is necessary for solver/creation fields, not a claimed gravity fix.

Next exact action: source-derived normal-contact press/drop fixture using actual
pad/cube and cube/floor mixed solref/solimp, with per-contact force/penetration
and total wrench. Inspect replay queries5/7/12 before changing contact parameters.
Also audit force-limited implicit PD dynamics separately: prior gravity/pulse
proof isolated armature, not complete stiff-servo integration. Nonlinear equality
and soft-limit gaps, reset-force semantics, neural CEM arithmetic, rendering,
zero/random controls and original training remain open. Never mark fullalignment
true from aggregate35/50 or a steady scalar check.

## Contact / PD investigation record — 2026-10-31

VALIDATION CORRECTION: public SpawnedArticulation.set_solver_iteration_counts
and set_sleep_threshold are warning-only unsupported methods. Earlier SDK
mimic test and benchmark called these without checking their result, so the
four labeled4/100 benchmark rows do NOT prove a real iteration sweep. Those
labels/claims are withdrawn pending correction. Independent private vendor
probe used the owning native setter100 and Root task applies configured100
through pre-GPU native adaptation; these are distinct evidence, not justification
for a mislabeled benchmark. Keep old rows/logs frozen as invalid sweep evidence.
New PD fixture also initially omitted DRIVE_LIMITS_ARE_FORCES; its saturation
numbers are excluded (saved implicit-pd-native-invalid-force-limits.json).
Corrected fixture uses the supported flag setter and checks torque by momentum.
No production task/controller change has been made from this provisionalprobe.


Characterization GREEN (normal-contact-{source,native}-baseline.json): source
floor/box equilibrium39.27484um, native0; pad/box2.13068um, native0. Both sides
satisfy momentum force balance (native max0.000408N on20.78N loads). Source
pair compiler confirms floor tau.012/d0.9/dmax.95 and pad tau.004/d0.95/dmax.99.
Rest/press/impact peaks differ; static-pad impact source26.744N vs hard20.618N,
so reducing every peak is not the oracle. Native fixture errors were classified:
Builder size requires tuple; InitGPUPhysics performs two.01s warmup steps, so
reset after init; query match='any' is required for a dynamic/static pair.
Invalid logs/output retained separately and excluded from physical claims.

Next capability implementation: approach2, additive keyword-only DexsimCollisionDesc
compliant_contact_stiffness/damping, defaultNone. Existing design: Spawn/Scene
ownership topic§1 and DFRigidBody::SetShapeContactAttributes material owner/cache.
Reuse that atomic per-shape call and existing named-URDF matcher, no C++ ABI,
thread/lifetime/solver or default change. Plain stiffness/damping is explicitly
an implicit PhysX model, not full source nonlinear regularization. Prototype
code snapshots are under reference/before-compliant-contact-descriptor.
Acceptance: existing named/replicated/reapplied material case plus a known-load
sphere equilibrium and independent backward-Euler impact, invalid ranges and
sparse merge coverage. Baseline constructor failure retained before addingfields.
Green -> repeat12 contact cases with source base coefficients, quantify errors;
red -> causal correction or retain diagnostic evidence and remove unproveddiff.


Recovered baseline: Root a5ea7ee0 / SDK f0e752c0 with retained dirty work;
engine80c085fb/GPU25d44eab, all three formal50 runs frozen. Additional6 PGS
known-load link cases pass; Black26.10.1/1733 files and Root whitespace rechecked.
The selected destination remains native EmbodiChain/DexSim, MuJoCo reference.

Requirement: preserve official contact parameters and quantify force/penetration,
not optimize the50-query count. First characterization has no production edits.
Compare three approaches: (1) keep hard contacts (current baseline; cannot encode
source softness), (2) expose existing PhysX per-shape implicit compliant contact
through Spawn descriptors (reuse current material owner/cache/thread, minimal
API addition but constant spring cannot reproduce nonlinear solimp), (3) change
vendor contact solver to source nonlinear regularization (large solver scope;
reserved until independent evidence rules out existing primitive). Start with
independent sphere/box normal press and impact using actual floor/cube and
pad/cube solref/solimp/priority; include reported force and momentum balance.
Follow source row equations before any mapping. Rollback unit for a later
capability change is its descriptor/adapter diff, with private file snapshots.
Protected: no new lifecycle/thread/owner, no shared installed SDK replacement,
no context writes in SDK, no source score or action-protocol changes.
Green characterization -> derive/test existing compliant primitive; red fixture
-> correct fixture and rerun, preserving failed logs. Query5/7/12 source replay
diverges before release/landing (source controls and freeze remain exact); max
cube discrepancies36.40/65.84/78.55mm do not by themselves identify contact law.

## Prior investigation sequence — 2026-10-31

### Initial phase — 2026-10-31: loaded point equality

Previous goal turn classified PROGRESS and all prior processes are terminal.
Reopened current worktrees, ledger, task force/constraint code, pinned MuJoCo
control/collision source and matching GPU718cde0a. No runtime/library mutations.
Scope now: independently test point-equality force units before changing gains.

C1 claim correction from a stronger runtime oracle: dump_scene_config prefers
the saved creation descriptor, not every live mutable value. Under requested
gravity-3, both archived-old and current Default backends produce velocity
-.299999982 after.1s. Therefore old gravity was later applied and is NOT a
measured physics defect. Earlier broad statements that all three settings were
ignored are superseded. Solver selection is independently proven ineffective
when late (TGS scalar response versus realPGS response after early submission);
creation-descriptor CCD/GPU-capacity differences and the unsupported CCD guard
remain verified. Preserve default-gravity-motion-old/fixed.json as evidence.
The preconstruction fix aligns creation settings and actual solver selection;
no gravity improvement is claimed.

Gate C1 verification: CPU requested gravity-3/bounce.42/CCDtrue is now captured
as-3/.419999987/true. Valid DirectGPU gravity-3/bounce.42/CCDfalse likewise matches,
including requested GPU capacities. A CCDtrue DirectGPU probe exposed an omitted
capability constraint and initially crashed: PhysX5.6 PxSceneDesc::isValid
explicitly rejects CCD+DirectGPU. Added pre-World ValueError and CPU-only CCD
documentation; validGPU probe passes, unsupported combination is covered.
No silent ignore or test waiver. The broader config selection has18 failures
because this paired SDK lacks AutoSolverCfg. Re-executing the archived old
Default backend reproduces the same18 plus the new timing regression (19fails),
so this is an existing Newton capability mismatch, not a C1 regression. Default
scope is202passed/1skip, with41 unrelated Newton cases deselected; the original
full result222passed/18failed/3skipped is retained. No SDK AutoSolver API change.

C2 VERIFIED: DefaultPhysicsCfg now exposes existing native enable_tgs with
True default preserved. Both True/False forwarding tests pass;203 Default
manager/config tests pass (1optional skip,41Newton deselections). Real DirectGPU
PGS selection works through this public config, without early SDK mutation;
known1N source-force residual ratio is1.00000693 (small nonlinear solimp delta).
C3 source recipe is implemented provisionally: force-unit point gains from
exported body_invweight0, PGS, source sleep disabled, gravity/source metadata
guards.31 focused production/reference tests pass; fresh reference export under
alignment-point-response-pgs passes100-state kinematic/mass/gravity checks.
No new formal50-query score is claimed yet.

C4 VERIFIED: existing PGS friction iteration flag is now consumed on its
creation path; TGS retains its prior behavior. Baseline false/true tests gave
1pass/1fail; repaired2pass. SDK library80c085fb + pairedGPU718cde0a passes91
regressions and separately6 PGS known-load cases. Current production recipe
expert45: maxheight.189335m, max cube trajectory error5.7305mm, peak pad122.769N,
mean steps20–35 47.285N. Sourceheight.191678m/peak49.523N. New formal50-query
runs are postponed because the next fidelity audit found a real open mismatch.

Formal new-recipe ORIGINAL planner COMPLETE:33/50, outcome agreement49/50,
source-only[7], no native-only successes. Initial raw plan arrays bit-identical
after actual sklearn inverse transform; first25 common finite controls exact.
Elapsed94.95s includes end-to-end evaluator scope, not solver-only timing.
Migrated planner is now live in evaluation/dexsim-prototype-point-pgs-compliant,
log point-pgs-migrated-planner.log, same frozenRoot/runtime/reference/devices.
No further production code/library edits until that process is terminal.

Formal new-recipe source-action replay COMPLETE:32/50, agreement48/50,
source-only[5,7], no native-only successes. Raw actions and NaN masks exact.
Cube median3.051mm/p95 41.456mm/max78.555mm; firstperiod median.516/max9.969mm.
The earlier TGS case was33/50 but agreement45, p95 81.110/max248.601mm; count
changes include removed false positives and restored source32/44. Actual both
workers reportPGS, friction-every-iterationTrue, gravity-9.81000042. Source
original-planner run is live under point-pgs-reference-planner.log.

C5 scope limit quantified: Source-versus-corrected-PGS two-hinge transient
position error max2.584e-7rad, while equilibrium matches1.35e-5relative. Native
velocity-iteration policy remains unchanged. Do not interpret scalar steady
agreement as proof of complete source transient dynamics. JSON comparison saved.
Benchmark4/4 cases pass (25/256 independent chains,4/100 iterations), completed
step times1.40–1.60ms and spring equilibrium maxrelative2.91e-7. No before/after
speedup claim; setup modes are not mixed. New scenario is registered as
pgs_mimic_compliance and contains no downloaded robot dependency.

C5 full provider regression GREEN: stagedGPU25d44eab + engine80c085fb pass93
SDK cases (3unselected unrelated shape cases). New reference native kinematics /
inertia audit passes21 bodies/25 frames/100 configurations. No library staging
or production-code mutation is allowed while the following formal runs use it.
Root code Black26.10.1 is frozen; Root560Default-scope tests pass (2skip,
75Newton/context-route-inapplicable deselections);18Newton AutoSolverCfg failures
remain separately reproduced on old backend and are not waived.
Formal source-action replay started: output evaluation/dexsim-source-action-
replay-point-pgs-compliant; same50query checksum, workers6/5, planner7. Actual
native scene descriptors and GPU/default-backend source hashes now included by
the evaluator. Next on successful completion: source original planner then
migrated planner on this same frozen runtime; compute paired outcomes/trajectory
errors rather than interpreting aggregate success equality as parity.

Benchmark note: first multi-Arena matrix was intentionally terminated after the
25-chain rows, before a256-Arena case incurred2s-per-arena pose-read waits.
This was a setup-workload correction, not an expired-observation restart. The
registered scenario now measures independent collisionless chains in one arena,
so it isolates mimic stepping without unrelated arena service timeouts. The
new4-case matrix (25/256 chains ×4/100 position iterations) is running under a
separate output directory, with the same spring-law validation. Original raw
logs/results are retained, no timings are mixed across setup modes.

C5 experimental result: GPU candidate25d44eab restores the source-mapped
PGS steady error2.71495969e-6rad versus MuJoCo2.71492317e-6 (1.35e-5relative).
Legacy200 response becomes50e-6rad as expected for its own spring stiffness;
post-init setter agrees. It was tested via private LD_LIBRARY_PATH, not staged.
The existing TGS250-step trajectories for all4 compliance modes are bit-identical
to baseline. Current Root Default-scope suite560passed/2skip/75deselected;
API3285/3285/context map pass. A dedicated GPU PGS spring-equilibrium regression
at4/100 position iterations fails twice on unmodified stagedGPU718cde0a and is
now being tested against candidate. Source/CPU prebuilt behavior is unchanged;
no CPU PGS compliance claim. Full vendor fix/runtime staging still pending.

C5 GPU provider VERIFIED in focused oracles: added per-step accumulated CFM
residual only for PGS GPU mimic solves in the existing storage/helper. Both
4/100-iteration tests fail onGPU718cde0a and pass on candidate25d44eab. Mapped
source2.71495969e-6rad matches MuJoCo2.71492317e-6; TGS all4 trajectories remain
bit-identical. Velocity-iteration policy and CPU prebuilt behavior unchanged.
Full source patch retained as SDK3rdparty/physx/patches/5.6-gpu-pgs-mimic-
compliance.patch, alongside measurement patch. SDK README documents GPU-only
scope and source/build requirements. CandidateGPU25d44eab staged only after
all previous processes were confirmed terminal. Engine80c085fb unchanged.
Full93-case regression is now running; do not stage/build over it.

Next action on green: freeze runtime, run the full new recipe through identical
source-action replay and both planners on the same50 queries, actual scene
configuration and complete new runtime hashes saved by evaluator. Diagnose any
new failure before further code changes. Prior34/32/33 counts are historical
21-body TGS evidence, not scores for the current PGS/source-force recipe.

C5 OPEN: independent no-contact two-hinge mimic probe under actualPGS returns
zero relative error for frequency200 and source858.2887 alike (source physical
expectation2.7149e-6rad under opposing.1Nm). The same source-static TGS probe
previously gave2.655e-6. Do NOT claim source mimic compliance verified underPGS.
Source shared computeMimicJointImpulses recomputes soft impulses without a
per-step accumulated-CFM term, and its velocity iteration becomes hard. With
PGS fixed positions this can converge to an effectively hard equality after
many iterations. Hypothesis, not a retained fix. Protect the existing TGS path.

C5 private experiment A started (not retained in SDK): confirmed GPU mimic
block types are not referenced by CPU physx/simulationcontroller/lowlevel files.
Added a per-step scalar impulse accumulator in the private GPU SOA block and
CFM*accumulatedImpulse residual in the existing shared helper, enabled only for
PGS callers. Existing velocity-iteration hard behavior and TGS path are retained
for this first hypothesis. CPU prebuilt behavior is unchanged; no CPU parity
claim. Source snapshots and full pre-experiment diff saved under reference/
physx-before-pgs-mimic. Private GPU build is running in physx-wrench-build; the
staged productionGPU718cde0a remains unchanged. Test via isolated LD_LIBRARY_PATH
before any staging or publishing. Success must be measured with the same
no-contact opposing-torque oracle, not inferred from better grasp scores.

Next exact action: inspect/reproduce PGS mimic compliance in the private pinned
PhysX5.6 GPU source. Compare preserving hardPGS (violates requested softness),
repairing accumulated compliant impulse in the existing PGS solver (recommended
experiment), or task-side custom torque/switching back to unstable TGS (not a
fidelity substitute). Use the existing solver storage/lifetime, no task-side
pose/velocity copy. One scalar accumulator per mimic/warp is private GPU data;
confirm opaque boundaries before any source edit. First test accumulated CFM
only in PGS position iterations while preserving existing velocity behavior;
then diagnose the exact same independent oracle. No formaltask score/new goal
completion claim until all requested compliance responses are verified.

C4 prerequisite: PGS newly makes the existing SDK friction-every-iteration
setting relevant. Current consumer code is commented out; config requestsTrue
but actual saved PGS scene flag isFalse. A new real scene descriptor case tests
True/False separately. Restore this existing flag in PhysicSys::CreateScene for
PGS only; TGS already performs friction each iteration and remains unchanged.
No interface/ABI/thread or dependency change. Source/task gain edits are frozen
while this provider gate runs. Preserve SDK source/binary baseline before build;
restore paired GPU718cde0a after staging (staging otherwise copies prebuilt).
Then run focused flag tests, PGS known-load measurements, affected constraints /
clone/filter/material regressions before any formal task evaluation.

Next C2 design (after C1): expose the existing native enable_tgs boolean in its
own DefaultPhysicsCfg (defaultTrue preserved). Task-local subclass override
would hide scene semantics; direct SDK mutation before World is diagnostic only.
The native config owner field is chosen; no new dependency, manager, hook or
ABI. Source point force gains use compiled inverse weights, not success fitting.
ActualPGS controls restore the standard gripper shape with both point mappings;
source force gains also match the independent scalar equilibrium and reduce
expert peak force from144.97N(PGS legacy point) to74.33N(PGS source force), while
source is49.52N. Task defaults will be changed only after the config forwarding
gate, then source metadata/helper/goldens and controlled/50-query verification.
Full contact/soft-limit/equality-impedance parity remains unproved.

Active prerequisite gate C1 — native scene construction settings:
DefaultPhysicsBackend.activate currently submits native settings AFTER World
construction. Reproduced with public gravity-3/bounce.42/CCDtrue: actual saved
PxSceneDesc remains-9.81/2.0/false. A requested PGS control was likewise still
TGS; setting the same public SDK parameter before World gives actualPGS and
source-force scalar equilibrium matches MuJoCo within1e-7 relative. Under
actualPGS plus source force gains, expert maxheight is.192164m versus source
.191678m, max cube trajectory error3.894mm versus baseline41.05mm. These are
controlled diagnostic profiles, not a promoted task setting or complete parity.

Contract: configured native scene fields must be present when World constructs
PxSceneDesc. Owner DefaultPhysicsBackend; reuse existing configure_world and
native configuration APIs. Class Complex but no new owner/hook/interface/ABI;
Default-specific submission timing is a local contract-restoring fix. A: leave
post-World submission (observably wrong). B: submit Default settings in existing
configure_world, retain post-World hook as no-op (chosen). C: globally reorder
activation across backends (unnecessary Newton lifecycle impact). No critical
crossroads or new permission needed. Before editing, archive default/base/config
sources under reference/default-scene-before. One focused mock captures settings
at native-world creation and must fail baseline; realCPU/GPU readback must pass
then affected manager/config tests. On green resume valid solver/point response
comparison; on red diagnose this same config gate. No point/mass/material/model
or task gains are changed in this gate.


Preserve masses, source filters, geometry, arm controller, mimic mapping,
rendering, model/planner and the previous frozen evaluation artifacts.

New diagnosis (not a retained fix): native expert arm errors stay below.00382rad,
but gripper geometry diverges during closure. In the source FK at the actually
recorded joint states, max connect residual is.275mm/source versus1.203mm/native.
Pinch/cube vertical offset changes from about+5mm to-35mm. Actual point inverse
inertia / source approximate inverse weight ranges1e-5..2.7 by axis; source
approximate weights are constant2.12962693/2.19659071 across45 states. The prior
point-force query experiment is known and was mixed, so it is not proof to
promote a setting. Audit files: grasp-public-joint-order.json and
 grasp-geometry-law-audit.json.

Point-load experiment checkpoint: first private URDF probe was rejected after
mass readback showed1e-4kg instead of authored1/2kg on collisionless links. This
is a fixture input failure, not evidence about point dynamics; no product code
was changed. Repeated with explicit prebuild mass/inertia overlays and checked
readback[1,1,2] before applying forces. Under0.1/1/10N, current acceleration
mapping gives34.5% excess residual; removing its extra d0 gives27.2%; source
force-unit mapping gives-3.4% versus independent MuJoCo. Disabling articulation
sleep leaves these results identical. Logs/JSON point-load-native-explicit-
masses and point-load-native-awake retain all cases. The large incorrect-mass
results remain classified invalid. Source force mapping is not yet promoted:
its residual timestep/solver difference needs characterization. A private PGS
control and a45-control real-gripper force-mapping replay are running via
existing public configuration/constraint creation, with no shared library or
production task mutation. No success-count fitting.

Contract: reproduce the pinned source point-equality response to known loads,
with official solref/solimp/dt. Class Complex, reversible task-level parameter
conversion; existing Scene/constraint creation APIs reused, no owner/thread or
ABI change. A: retain acceleration drive (normalizes with native effective
mass; suspected mismatch). B: convert source regularization to force gains
using compiled source approximate inverse weight (recommended, proof pending).
C: implement full position-dependent impedance with owner-level batched updates
(pending, larger scope). No critical architecture choice is needed for the
independent experiment. First oracle is a no-contact two-slider point equality
under opposing known forces, with all physical parameters recorded; then the
same construction checks the native force and acceleration modes. On success,
extend source export/helper plus focused behavioral goldens, rerun affected
checks and controlled grasp replay. On failure, preserve evidence and diagnose
this same oracle; no task-success-based parameter fitting.

## Current checkpoint — 2026-10-31: retained bodies / measured force

Goal remains ACTIVE. Full physical alignment is false. This continuation is
PROGRESS:21 positive-mass bodies and28-D state encoding are implemented, native
force/acceleration query defects are reproduced and repaired, controlled physics
and formal paired evaluations are complete. No commit/push/PR update/publication.

| Current21-body protocol | Successes | Outcome agreement with MuJoCo |
| --- | --- | --- |
| MuJoCo reference |34/50|reference|
| DexSim + original planner |34/50|46/50: source-only[7,32], native-only[12,43]|
| DexSim + migrated planner |32/50|48/50: source-only[32,44], native-only[]|
| DexSim + exact source actions |33/50|45/50: source-only[7,32,44], native-only[12,43]|

Raw source replay actions/NaN masks are exact. Cube error median3.56mm,
p95 81.11mm, max248.60mm. Original-planner first raw plans and first25 common
finite controls match the source exactly using the actual sklearn inverse
transform. Migrated initial raw plans still differ by5.379. All16 initially-near
queries succeed. Equal aggregate counts do not establish paired physical parity.

Verified physical/model boundaries:
-21 positive-mass bodies,25 source frames, total22.04750833884274kg;100-state
 source/URDF FK/generalized mass/gravity checks and native mass/FK checks pass.
-28-D encoding on50 literal single-world starts max8.326e-6, raw quaternions
 max2.875e-8 including26 negative-W cases. Force at reset is explicitly invalid.
 A50-world batch also quantifies3.791micrometre position roundoff; its tighter
 raw-state threshold failure is retained rather than reported as a clean pass.
-NativeGPU measurements require patched PhysX5.6 CHECKED library718cde0a. Two
 existing accounting paths omitted velocity deltas/partial impulses;6 physical
 load cases now pass.24 expert contact-normal projections maxrelative6.649e-7;
21 missing-contact steps recover, with45-control qpos/qvel/cube poses unchanged
 bitwise by the partial-impulse repair. Vendor patch/build instructions are in
 SDK3rdparty/physx/patches; C++/Python stay on the existing Scene batch boundary.
-Same45 expert commands still yield source/native peak pad forces49.52/114.44N
 and heights.19168/.23174m. Separate trajectories do not isolate one cause.
 Native measurement correctness is not MuJoCo contact-law equivalence.

Validation:591 Root tests passed,3 optional GPU tests skipped;89 SDK tests
(79 affected regression +6 physical measurement +4 material backend matrix).
Black26.10.1 passes1733 files, API3285/3285, changed-line clang-format14, context
map/tests and whitespace pass. The existing25-UR5 benchmark measured2.665ms
baseline/2.775ms patched (one sample, no performance improvement claim; masked
NVML memory values are invalid). Sources/runtime identities verified57 hashes
across3 native runs and archived in alignment/runtime-sources-fixed-bodies.

Authoritative machine-readable status: VALIDATION.json; runtime/source/build
identity: alignment/native-fixed-body-runtime.json; task TODO:
/root/sources/EmbodiChain-pr895-lewm/examples/learning/lewm/OFFICIAL_CUBE.md.
Old alignment assets and15-body reports remain separate historical evidence.
Both retained topology and private GPU build changed between checkpoints.

Next exact action: isolate loaded gripper/contact/equality response using known
loads and the now-verified body wrench; compare both pads, loop residuals and
penetration at matched configurations before another parameter change. Then
check the first changed CEM elite/preprocessing arithmetic and pixel/latent
rendering differences. Preserve the original recipe and full physical gate.

## Retained-body investigation evidence (historical sequence)

Goal ACTIVE; the 21-body candidate and source-state observation are experimental.
No claim of contact/state parity or new task success rate. Canonical assets in
alignment and previous 15-body evaluations remain frozen. Worktree remains dirty;
no commit, push, issue, dependency publication or shared-install replacement.

Verified candidate evidence: alignment-fixed-bodies retains all 21 positive-mass
source bodies and all 25 source frames, total robot mass22.04750833884274kg.
100-state independent source/URDF FK and generalized mass/gravity checks pass;
native 21-body kinematics checks pass. Logs fixed-bodies-reference-export.log
and fixed-bodies-native-kinematics.log. State construction/source RNE fixtures
pass19 task tests; generic link dynamics/frame adapter tests pass53. SDK added
Scene batch acceleration/incoming-wrench reads; noncontact physical cases pass3.

Baseline failure (resolved by gates A/B below): SDK supported-pad regression fails: at rest velocity about2.1e-6m/s,
link acceleration reports -9.810002m/s^2 and incoming pad force -9.844334N
rather than 0m/s^2 and -9.81N. Independent contact impulse gives9.844335N.
Right-pad expert probe also intermittently reconstructs near-zero external
force while independent contact normals report70–100N. Silicone-child RNE
remains zero, excluding missing attribution to the child. Preserve all failures:
reference/link-wrench-contact-awake-test.log, supported-pad-dynamics.log;
alignment/pad-contact-wrench-children-probe.json. New state_wrench_valid must not
be treated as verified until this is resolved.

Current source-backed hypothesis: PhysX5.6 has missing measurement accumulators
in GPU partial impulse propagation (articulationImpulseResponse.cuh
averageLinkImpulsesAndPropagate2) and the velocity-output flush overload
(forwardDynamic2.cu PxcFsFlushVelocity with outVelocity). The full paths record
mSolverSpatialImpulse and mSolverSpatialDeltaVel; these partial/overload paths
update actual dynamics but omit those query accumulators. This is a hypothesis
until an isolated build passes the unchanged physical assertions and replay.
The separate contact-last PR550 is not assumed applicable (SDK does not set
that flag). No task gain, mass, or contact law will be fitted to this symptom.

Contract/decision: physical readings after a completed GPU step must obey
Newton-Euler balance and the existing child-joint/world frames. Class Complex,
contained and reversible vendor maintenance; existing PhysX accumulator and
DexSim Scene batch mechanisms reused, no public ABI/queue/thread change.
A: expose raw readings and leave force invalid (safe containment, incomplete).
B: independently sum shape contacts/friction (requires new aggregation and may
omit noncontact external force). C: repair missing vendor measurement accounting
in an isolated PhysX build (chosen, source-backed, existing contract retained).
No shared dependency switch/publication is authorized or planned. Rollback is
remove the isolated patch/library and retain the captured baseline. Tests:
held/freefall/torque/supported known loads unchanged; then multi-link contact
probe. On green run all affected SDK regressions and 21-body source action
replay; on red preserve artifact and diagnose the same physical oracle first.

Build checkpoint: release PhysXGPU built successfully (145 targets), but its
World startup crashed with the existing SDK. Classified before any physics edit:
SDK adaptation links CHECKED PhysX static libraries and GPU, while the first
private build used RELEASE, which changes internal PVD/checking structures.
Rejected that binary arm, restored staged prebuilt978a4fdb, preserved both startup
logs; rebuilding matching CHECKED GPU with Clang14/CUDA12.8/sm120. Shared installs
remain untouched. The supported-contact torque oracle now uses independent
contact moments (a manifold need not share loads symmetrically); force and
acceleration expectations are unchanged. Added a sibling-pad contact case to
exercise partial branch propagation; baseline runs use separate processes so a
failed pytest traceback cannot retain a World into the next case.

Measurement-accounting gate A: matching CHECKED source baseline042eb2a3
reproduces the prebuilt supported-pad failure exactly (0.03433339N). Added only
mSolverSpatialDeltaVel accumulation to both root/link branches of the existing
velocity-output flush. Built/staged a2b7ebaf. Held/freefall/torque pass3; one-pad
and sibling-pad static support pass2 with unchanged force/acceleration oracles.
Root focused suites pass81. A45-control expert replay still reports near-zero
pad external wrench against54–55N independent normal force (steps29/30), as
predicted for the separate partial impulse accumulator omission. This is
predeclared amber, not a waived pass. Logs link-wrench-flush-accounting-*-test.log
and pad-contact-wrench-flush-accounting.log. Next gate B adds the existing
mSolverSpatialImpulse accumulator at partial propagation before clearing Z;
unchanged physical supported cases plus expert contact-normal projections must
pass. Dynamics buffers must not be edited as part of this measurement fix.

Gate B VERIFIED: the isolated partial-impulse accumulator restores the dropped
contacts. A two-dynamic-support fixture independently reproduces a5.877N error
on flush-only library a2b7ebaf (verified loaded path using LD_LIBRARY_PATH), and
passes with full718cde0a. All6 physical cases pass. Dynamic support has a small
residual acceleration; its oracle now derives inertia from finite velocity
differences rather than assuming exact equilibrium (same5e-6 force tolerance).
Both the negative baseline and positive arm use that identical corrected oracle.
The first static-only test's zero contact-torque assumption was also corrected
using independently measured contact moments, not a relaxed threshold.

45-control replay: all24 loaded normal projections now match, maximum relative
error6.649e-7, median1.072e-7; all21 missing steps recover. Joint positions,
velocities and cube poses are bit-identical before/after gate B. Comparison:
alignment/pad-contact-wrench-accounting-comparison.json. Source replay of the
same45 expert actions matches the saved MuJoCo qpos exactly. Actual force/height
still differ: source peak pad force49.523N versus native114.435N; source maxheight
.191678m versus native.231742m. These independently evolved trajectories do not
isolate contact softness alone. Plot and metrics: expert-wrench-cross-engine-
comparison.{png,svg,json}. Preserve this as physical mismatch evidence.

State encoding: literal50 single-world resets pass2e-5 tolerance, max8.326e-6,
including26 negative-W quaternions (quat max2.875e-8). At50 replicated worlds the
same tight state threshold fails7 rows due3.791micrometre world/arena position
roundoff; quaternion signs still match. Preserve both reports, do not relabel
batch error as force parity. Force is explicitly zero/invalid at reset; source
reset-force equivalence is still open.

Checks:591 Root tests passed,3 optional GPU tests skipped;79 affected SDK
regressions passed, plus6 physical measurements. Black26.10.1 check1733 files,
API3285/3285, context map, whitespace and changed-line clang-format14 pass.
Shell initially selected Black24.3; corrected with explicit py31126.10.1 and
restored two initially-clean unrelated files. Final26.10.1 check passes.
The SDK existing25-UR5 clone benchmark measured2.665ms baseline/2.775ms patched
per step (20warmup+100measured, single sample); no speedup claim. NVML memory
counters are invalid under masked ordinal mapping and are not reported as use.
Patch/build instructions retained under SDK3rdparty/physx/patches; no SDK context
writes or dependency publication. New C++ public dispatch has no style delta.

21-body source-action replay complete:33/50, paired agreement45/50, source-only
[7,32,44], native-only[12,43]. Actions and NaN masks are exact. Cube errors
median3.560mm/p95 81.110mm/max248.601mm. Firstperiod median.290/max11.457mm.
Native original-planner run complete:34/50, paired agreement46/50, source-only
[7,32], native-only[12,43]. Migrated-planner run in progress, same workers6/5,
planner7. These new runs change retained body topology AND the private GPU build
relative to old15-body runs; do not attribute count changes to one isolated cause.
Overall physical alignment remains false. Canonical old assets remain untouched.

Next exact action: finish the retained-body migrated-planner evaluation, verify
paired raw plans/actions/runtime identities, then update VALIDATION.json and the
routed TODO. Subsequent physical work starts with measured closed-loop/contact
response under the same commands; do not retune success counts.

Historical next action (superseded): build PhysX5.6 GPU in a private output tree with the missing
measurement accumulators only; compare the supported-pad test and expert probe
against baseline. Do not advance to full evaluations on unclassified failure.

## Previous checkpoint — base mimic mapping, 15-body model


The full goal remains active. Physical/contact/state-observation parity and
training reproduction are incomplete. Do not mark the goal complete.

**LATEST CHECKPOINT: base mimic mapping corrected; 50-query runs complete.**

Previous goal turn classified PROGRESS. This continuation inspected current
files/processes, diagnosed the loaded grasp, changed the production task and
completed controlled tests/evaluation. Goal remains ACTIVE, not complete.

| Run | Named pads/cache | Source-based mimic mapping |
| --- | --- | --- |
| MuJoCo reference | 34/50 | 34/50 |
| DexSim + original planner | 33/50 | **34/50 (68%)** |
| DexSim + migrated model/planner | 31/50 | **32/50 (64%)** |
| DexSim + exact source actions (diagnostic) | 32/50 | **34/50** |

- New directories end in -mimic-mapped; canonical migrated run ends in
  -mimic-mapped-matched-devices ([cuda:6,cuda:5], planner7). A separate [1,5]
  repeat has identical initial plans and outcomes. Original-planner first raw
  plans remain bit-identical to reference; first25 executed controls match on
  common finite entries. Both MPC implementations agree on48/50 outcomes:
  original source-only[32], native-only[43]; migrated source-only[32,44], no
  native-only successes. All16 initially-near goals still succeed.
- Source-action replay agrees on46/50: source-only[32,44], native-only[41,43].
  Actions/NaN masks are exact. Cube median/p95/max errors now3.49/54.50/268.87mm
  versus4.83/69.36/262.90mm. First-control median/max .318/11.67mm. Worst
  query32 still loses the cube; aggregate equality is not physical parity.
- Expert160: firstsuccess41/41 versus source39, maxheights .23183/.23179m
  versus source .19168m. Final errors .00427/.00821m; replica max .00899m.
  Height mismatch remains and no new contact/point-force setting is promoted.

**Diagnosis and implemented correction**

- recorded-grasp-source-audit.json/log evaluates SOURCE geometry/law on saved
  states (its forces are not native measurements). Before query32 loses the
  cube aroundcontrol17, source remains bilateral while native states show
  unilateral pad contact. Atcontrol10, driver equality residual .01145rad
  versus source .000516; after loss, driver errors reach .34rad while arm
  error stays below .00528rad. Queries0/46/5/44 are also audited.
- Independent no-contact two-hinge test: legacy frequency200/ratio1 gives
  8.6396e-5rad steady error; MuJoCo3.11 gives2.7149e-6. Source KBIP/R mapping
  yields858.2887/4.47214; CPU/GPU give2.6552e-6 (2.2% discretization difference).
  Before/afterGPU-init setters agree, so that setter is not ignored. PhysX
  uses frequency directly, without2*pi. This is a real parameter-mapping gap.
- Task now calls _base_mimic_compliance(solref,solimp): frequency
  d0/(dmax*timeconst*ratio*sqrt(1-d0)), damping ratio ratio/sqrt(1-d0).
  The official driver actual/approximate inverse-inertia ratio is
  1.0000065–1.0000296 at four query32 source states. Base response is verified;
  position-dependent solimp remains incomplete. Compare three approaches:
  retain1/timeconst (wrong response); base-force-law mapping (chosen);
  full dynamic impedance/mass response (remaining, needs owner-level batching).
- Point-force versus acceleration mappings have mixed task outcomes in
  grasp-equality-mapping-probe.*. Keep those diagnostic; do not fit gains to
  success counts. Batch-layout sensitivity in stiff contacts is recorded.
- One focused test/fixture added in the existing production task suite. The
  expected force stiffness/damping come from independently compiled MuJoCo
  KBIP/R, not a duplicate implementation. Baseline physical mismatch is in
  mimic-two-joint-*.json; test baseline fails before the helper exists.
  Existing IK and passive-force coverage is reused; no tutorial tests added.
- **532 tests passed,3 GPU opt-in tests skipped** (490 project+42 context).
  Task-only selection15passed. black1733 files, API3285/3285, context map and
  whitespace pass. SDK code is unchanged this continuation; previous79 SDK
  regressions remain applicable, plus independent CPU/GPU compliance probes.
  No context edit is needed for this private task mapping: existing overviews
  route to OFFICIAL_CUBE.md, now revised. No commit/push/publication.

**Next required work — concrete source-body finding**

Do NOT assume source pads are massless. Reinspection of the actual source
shows right/left pads mass .0035kg, silicone children .00130281183948kg;
only visual silicone geometry has contype/conaffinity0. Both are positive
fixed bodies currently collapsed into follower compounds. Source pad weld
owner is follower; root is ur5e/ with subtree mass22.04750833884274. cfrc_ext
uses the robot subtree COM, not the pad center. Recorded in
alignment/source-pad-wrench-frames.json. Point connect endpoints are follower
and coupler, not pad, so pad external wrench excludes those endpoint forces.

The next promising implementation is to retain original positive-mass fixed
pad/silicone bodies (no invented epsilon masses), preserve zero-mass virtual
frames, source welded-group self-collision filtering, aggregate inertia and
25 source-frame FK. Native GPU API already exposes LINK_INCOMING_JOINT_FORCE
in child-joint coordinates, plus COM acceleration/velocity. Validate force/
torque frame, inertial/gravity subtraction and child-force balance before
building the literal28-D observation and clipped six-component wrench norm.
Do not substitute normal-only contact impulses: existing ContactQuery marks
per-contact friction unavailable. Per-shape IDs are not necessarily the only
route once original pad bodies are retained. Full wrench/28-D implementation
and nonlinear contact/equality/render/training gates remain unverified.

All jobs for this checkpoint completed. Task source is archived under
alignment/runtime-sources-mimic-mapped/. The installed SDK and original
Dexverse tree remain unchanged. Full physics alignment flag stays FALSE.

**PREVIOUS CHECKPOINT: named pad materials implemented and all 50-query runs complete.**

The user-selected destination is native EmbodiChain/DexSim, with official MuJoCo
as reference. No retraining or query/checkpoint changes. Preserve the earlier
collision-filter checkpoint as historical evidence; aggregate performance has
NOT improved after restoring source pad parameters.

| Run | Collision-filter checkpoint | Current pads + cache |
| --- | --- | --- |
| MuJoCo reference | 34/50 | 34/50 (68%) |
| DexSim + original preprocessing/model/CEM | 34/50 | **33/50 (66%)** |
| DexSim + migrated model/planner | 33/50 | **31/50 (62%)** |
| DexSim + exact source raw actions (diagnostic) | 33/50 | **32/50** |

Current output directories end in `-pad-materials` under evaluation/, with the
canonical migrated run in `dexsim-prototype-pad-materials-matched-devices`. Each
contains report.json, trace.npz and paired-outcomes.json. Original-planner raw
initial plans remain bit-identical after inverse StandardScaler conversion;
source plan files are standardized while native plan files are raw. Initial 25
executed controls match on all common valid entries. Both MPC implementations
agree with source on 47/50 outcomes: original source-only [32,44], native-only
[43]; migrated source-only [7,32,44], native-only []. All 16 initially-near goals
still succeed. Reports record native material caching and 19 code/binary/asset
SHA256 values. The separate native-contact-materials.json records the actually
loaded PhysX GPU library SHA256 and real post-reset material fields.

Source-action replay retains raw commands/NaN masks exactly and agrees on
46/50 outcomes (previously 43/50): source-only [5,32,44], native-only [43]. Its
first-control cube median/max errors are .282/11.70 mm. Whole-replay
median/p95/max are 4.83/69.36/262.90 mm, versus 4.46/42.58/261.57 mm before.
This is not overall trajectory improvement. Paired-trajectory-errors.json
identifies query32 (262.9 mm cube error at step25, max arm error .00528 rad),
then queries0/46 as the largest path divergences. Investigate loaded contacts
and linkage response rather than changing verified inertias/PD gains.

**Implemented and verified:**

- SDK additive nonvirtual shape contact getters/setters, native material COW,
  stable shape indices, clone retention and per-property body setter semantics.
  Invalid-index exceptions translate after Scene work returns; the first
  ThreadX exception implementation aborted and its failing log is preserved.
- Named URDF box/sphere overrides match dimensions/local frames, since cooked
  mesh order differs from XML. Public apply_dexsim_contact_properties() restores
  materials after Core's sparse source overlay without changing body inertia.
  A new DefaultRigidBodyMaterialCfg lowers native combine modes; the task pads
  now keep 0.7/0.6 and use MIN for cube/pad contact. Newton rejects named overrides.
- Two-arena readback retains [.6,.7,1] after load, reset and 160 controls.
  Cached expert probe: first success41/41 (source39); heights .23183/.23097 m
  (source .19168); final errors .00287/.00776 m; replica max difference .00560 m.
  The cached uniform-friction1 counterfactual has heights .19104/.19091 m but
  replica difference .05047 m and one final goal failure. Do not promote it for
  its lift-height score. Figure: alignment/pad-material-expert-comparison.png/svg.
- Direct GPU two-platform friction fails with cache_material=False despite
  correct CPU readback; CPU passes. Caching makes both physical paths pass.
  Task configuration now explicitly selects DefaultPhysicsCfg(cache_material=True)
  and new GPU shape overrides reject the unsupported uncached combination.
  This selects an existing allocation/cache mechanism and does not change the
  scientific task parameters. The uncached PhysX dependency issue remains open.
- Three approaches considered: patch/rebuild the pinned PhysX (needs separate
  native validation); use existing material caching (chosen supported path);
  retain retired materials in a new queue (extra owner/lifetime growth rejected).
  Official PhysX5.6 source is pinned at 2264315594478a9aa0bda3464761a666fa107d76
  in reference/physx-5.6-reference. Buffered NpScene events and GPU material-map
  re-registration make handle reuse a plausible cause, not a completed upstream fix.
- Tests: **489 project tests passed, 3 GPU opt-in tests skipped**, plus 42 context
  tests. SDK: **79 passed** (32 clone/articulation, 4 material backend-contract
  matrix cases, 24 filters, 19 constraints). Coverage closes native material
  behavior, clone isolation, source replay and unsupported backend gaps; no
  tutorial tests added. API coverage3285/3285, black1733 files, yapf and both Git
  whitespace checks pass. Context map passes; simulation adapter guidance was
  revised. Other routed overviews still route these contracts correctly and
  require no catalog/topology updates. SDK agent context was not edited.

**Remaining gates / next work:**

1. P0 contact/equality response at queries32,0,46 and source-only replay5/44.
   Actual shapes retain torsional patch radius .1 m, zero compliant damping,
   improved patch friction and tiny contact envelope. These are not equivalents
   of MuJoCo nonlinear normal impedance, torsional/rolling coefficients or
   material priority. MIN covers cube/pad pairs but not unlike pad/pad pairs.
   Measure per-pad normal/tangential wrench, penetration and loaded closure;
   preserve geometry/topology instead of fitting task score with uniform friction.
2. P1 exact28-D observation. Native currently exposes robot qpos/qvel/qf and
   cube pose, not the source vector. Source contact scalar is literally
   clip(norm(right_pad.cfrc_ext) / 50,0,1), including all six wrench components.
   SDK contact shape identity is needed because source pads are welded into a
   native follower with other collision geometry. No privileged state feeds MPC.
3. P1 neural arithmetic/render measurements and paired zero/random controls;
   own CEM initial plans still differ despite pointwise tolerances.
4. P2 official training recipe after these physical/observation gates.

The migrated run was repeated on the same worker devices [cuda:6,cuda:5] as
the original-planner comparison. Its success count remains31/50. Initial plans
and outcomes match the first [cuda:1,cuda:5] run exactly; keep both reports.
All jobs from this checkpoint have completed; no commit, push or publication.
The installed SDK and original Dexverse tree remain unchanged. Goal ACTIVE;
full physics/contact/observation/training reproduction flags remain FALSE.

**PREVIOUS CHECKPOINT: collision-mask bug fixed and 50-query runs completed.**

SDK replication discarded authored category/exclusion words2/3, replacing walls'
[i,1,2,1] by [i,1,0,0]. Robots consequently collided with cube-only walls. The
SDK adapter now merges arena words0/1 with each body/link's authored words2/3
through template loading, native clone requests, clone link overrides and runtime
descriptor copies. No task-side wall removal, changed gains or altered timestep.
Design selected owner-level preservation over an extra policy flag or a task-side
native workaround: isolated arena semantics remain unchanged, existing authored
object filters are now honored, and unset/zero layer defaults retain old behavior.

- Source: `python/dexsim/spawn/adapters/dexsim_adapter.py` in the isolated SDK
  worktree. Existing SDK source and test files formatted with yapf 0.43.
- Baseline physical regression FAILS: masked and control payloads all incorrectly
  rest at z=1.102 m (`reference/replicate-layers-baseline-test.log`).
- Corrected physical regression + all other filter cases PASS: **24 tests**,
  CPU/GPU, clone/non-clone, both thread modes; see
  `reference/replicate-layers-full-regression.log`. Earlier selected 9 passed.
- Native PxShape readback confirms [i,1,2,1] for all walls in both arenas,
  `alignment/native-filter-readback-fixed.log`. Previous zeroed readback kept.
- Eight failed-state first-period probes now have max arm error **0.002107 rad**
  and cube error **0.002566 m**, versus about 1.1 rad / 0.20 m before the fix.
  `alignment/solver-tgs-filter-fixed-probe.json`.
- Staged through `dexsim-build-newton-jit-package`. Installed SDK is unchanged.
  All current runs use build_lewm/lib/python_package. Do not overwrite binaries
  or staged modules while jobs use them.

**Post-fix completed evaluations** (same exact 50 queries/checkpoint):

| Run | Result | Directory under evaluation/ |
| --- | --- | --- |
| Original MuJoCo | 34/50 = 68% | mujoco |
| DexSim + original preprocessing/model/CEM | **34/50 = 68%** | dexsim-reference-planner-filter-fixed |
| DexSim + migrated model/planner | **33/50 = 66%** | dexsim-prototype-filter-fixed |
| DexSim + exact source raw actions | **33/50**, diagnostic | dexsim-source-action-replay-filter-fixed |

- Original-model native first plans remain bit-identical to source; raw commands
  before termination match exactly. Source-action replay preserves the original
  actions including NaN masks exactly. Native MPC and no-image replay can still
  differ under contact; do not infer full dynamics equality from aggregate score.
- Native original-planner outcomes agree on 44/50 (source-only successes [7,32,44],
  native-only [12,41,43]); migrated outcomes agree on 47/50 (source-only [5,32],
  native-only [12]). All 16 initially-near goals succeed in both post-fix MPC runs.
- Source-action replay agrees on 43/50 (source-only [7,19,32,44], native-only
  [12,41,43]). First-period cube median/max error .0002965/.0126939 m; entire
  replay median/p95/max .0044565/.0425773/.2615696 m. Some late physical divergence
  remains. Each directory has report.json, trace.npz, and paired-outcomes.json.
- Native reports bind 13 runtime/code/asset SHA256 hashes, including SDK adapter,
  native libraries, task/model/planner, configuration and compiled contracts.
- Corrected 160-step expert diagnostic still has contact/closure differences:
  both replicas first succeed at zero-based step41; heights .2229/.2318 m vs
  source .1917 m. Final errors .0106/.0495 m; replica max difference .0394 m.
  `alignment/expert-probe-dexsim-filter-fixed.*`. One later leaves the goal;
  this raw diagnostic ignores termination and is not official MPC evaluation.

**Additional checks and current work state**:

- Actual GPU body mass matrices at 50 poses match MuJoCo without armature to
  3.43e-6 and gravity to 2.16e-5. Armature readback matches 1.5e-9; the actual
  DRIVE_LIMITS_ARE_FORCES flag is true. PhysX queries omit armature (upstream
  NVIDIA-Omniverse/PhysX#575); ten unconstrained gravity/pulse cases prove that
  simulation does include it, with max velocity error 8.22e-8. Do not change mass
  or armature based on that known query omission. Evidence: generalized-mass-*.npz
  and .json, pure-inertia-probe.json.
- SDK-only read-only diagnostic extension in `.agent_runs/cube_dynamics_probe`
  uses cached Clang14/py311 headers and Scene-thread access. Not a production
  dependency. Real joint-limit readback also matches source float32 limits.
- All older contact, loop, mimic, solver and clamp counterfactuals were BEFORE
  the mask fix and retained unintended wall contact. They do not disprove those
  API setters or isolate remaining contact-parameter effects. Do not repeat that
  search or promote disabling walls/self-contact/constraints.
- Latest broad Python suite: **261 passed, 1 deselected** against staged SDK;
  `alignment/focused-python-tests-final.log`. SDK D6 suite previously 19 passed;
  native C++ is unchanged by the Python layer fix. Final API gate 3283/3283,
  black --check across 1733 files, 42 context tests, context map and diff whitespace
  checks all PASS. Summary: VALIDATION.json. No commit, push or publication.
- Updated repo plan: examples/learning/lewm/OFFICIAL_CUBE.md. Updated scientific
  figure: alignment/validation-overview.png. Historical pre-fix evidence below
  and in original evaluation directories remains for comparison.
- No long-running evaluation should remain; verify process/log state before
  any new launch or stage mutation. Goal remains ACTIVE and full physical flags
  remain FALSE. Next: per-pad material/priority/softness and loaded linkage parity;
  28-D observation/right-pad wrench; neural CEM arithmetic/render measurements;
  same-query zero/random controls and official training recipe reproduction.

### Runtime and exact sources

- Always use `/root/miniconda3/envs/py311/bin/python`; default shell Python is
  py310 and cannot load cp311 SDK binaries. `black==26.10.1` is in py311.
- Native SDK is the clean-build runtime under
  `/root/sources/dexverse-lewm-constraints/modules/dexsim/build_lewm/lib/python_package`.
  Installed SDK and original `/root/sources/dexverse` remain unchanged. The
  `dexsim-build-newton-jit-package` target stages all matching binaries/JIT code.
  Native tests require `SDL_VIDEODRIVER=dummy`. Never replace binaries while a
  process uses this build.
- Cube's compatible stable-worldmodel pin is now
  `abdced49809d5eae38e24b27dc7b635c502c4812` in
  `reference/stable-worldmodel-20260520`. The modern pin `21446...` remains only
  for the historical PushT comparison. Its solver interface and population
  deviation do not reproduce the original Cube example.
- Original Cube evaluation observes **one frame**, although training context is
  at most three. CEM uses sample deviation (`correction=1`), batch size one and
  one RNG visiting environments sequentially. Five receding blocks execute 25
  raw actions; only two plans occur in the 50-step evaluation.
- OGBench pin is `6eda27402c69681cfe584cc9f3a3fe2a24ec9163`; manipspace is
  unchanged from last pre-example commit `1d4140997f60c52c6fb0702ec100dc988b18c548`.
- Legacy package eager imports need isolated `reference/io-deps`: hdf5plugin
  6.0.0, lancedb 0.30.0, pylance 4.0.0, overrides 7.7.0, deprecation 2.1.0,
  lance-namespace and lance-namespace-urllib3-client 0.5.2. No global overrides.

### Completed data and physical/model checks

- Both archive and extracted HDF5 are complete and fully SHA256-verified; see
  `data/archive-verified.json`, `data/hdf5-verified.json`. HDF5 size is
  101,942,558,720 bytes, SHA `0664d507c4ff12009010644c9ae950836f954e700c172ccf22e7423af1a55625`.
  Keep the existing archive/ranges; do not restart old Xet PID 2397077.
- Official data has 10,000 episodes x 201 frames, 2,000,000 finite outgoing
  actions and NaN terminal actions. Query helper `_official_data.py` prepares
  literal seed-42 selection, including upstream's final-valid-row exclusion.
  `alignment/official-queries.h5` SHA is
  `7d2d66317ab10644173308fb9a3a8eaf9a9edc32889047c89d22e54a1f9f8b8f`.
  All 50 queries retain 16 already-near goals and use the full official scaler.
- Export uses 15 physical bodies and 25 source-frame bindings. Total mass is
  22.04750833884274 kg. Across 100 configurations, generalized inertia max error
  is 4.13e-13 and gravity error 1.21e-13 N*m. Native full-tensor overlays preserve
  small inertias and collisionless mass. FK/geometry loading does not prove dynamics.
- Original single-frame model comparison passes in
  `alignment/model-reference-legacy-history1.json`: rollout max error 3.34e-6,
  goal-cost max error 3.81e-5 within recorded absolute/relative tolerances.
- Compatible CEM analytic comparison passes, all 60 populations across two
  sequential environments; max population error 2.38e-7, final actions identical.
- SDK compliant-D6 addition is built. Full CPU/Direct-GPU body-constraint suite
  passes 19 tests (`reference/sdk-constraint-compliance-regression.log`), including
  finite K=100/B=20 acceleration-spring gravity sag. Source equality mapping is
  still approximate. Prior shape-getter pose and adapter-endpoint checks pass.
- Registered `OGBenchCube-v0` has reference IK, implicit arm drives, passive and
  split tendon forces, mimic and both point links. Core supports named link
  endpoints, rollback/removal ordering, optional compliant D6 drives, explicit
  static masks, optional manager default plane, and literal physical-state restore
  via `set_state(clamp_qpos=False)` (targets still clamp).
- Native 160-step source-expert replay now succeeds in both replicas at step 42
  (zero-based). Final errors 2.84/2.53 cm, replica max difference 6.1 mm. Height
  still 0.232 m vs source 0.192 m. Preserve `hard-point-expert-probe-dexsim.*` and
  other failed baselines. This is not official dataset evaluation.

### Actual 50-query evaluations (all completed)

- `evaluation/mujoco/report.json`: original World/WorldModelPolicy/CEM + model,
  34/50 = 68%. All 16 initially-near succeed; 18 of 34 initially-far succeed.
- `evaluation/dexsim-reference-planner/report.json`: native physics/render with
  the exact original preprocessing/model/CEM, 30/50 = 60% (15 near + 15 far).
- `evaluation/dexsim-prototype/report.json`: native physics/render with migrated
  model/planner, 29/50 = 58% (15 near + 14 far).
- `evaluation/dexsim-source-action-replay/report.json`: literal source raw-action
  replay, 30/50; a physics diagnostic, not closed-loop MPC.
- Original-planner adapter reproduces all 50 first plans bit-for-bit, verified
  by `alignment/planner-first-plan-check.json`. Migrated neural FP32 tolerances
  do not imply equal CEM plans: raw first-plan max difference 5.38 and mean .069
  after 30 elite-selection iterations (some commands saturate). Do not attribute
  that entire change to physics. Potential P1 causes: preprocessing division vs
  torchvision multiplication, encoder batch 50 vs original batch 1, repeated
  action-encoder windows vs original one-time full-horizon encoding, raw-normalized
  candidate roundtrip. None has yet been retained as a production fix.
- New `_official_eval.py` and CLI `official-eval-reference` / `official-eval-dexsim`
  implement these runs. Native mode uses two persistent 25-env processes, with
  one ordered planner and RNG across both planning barriers. Vulkan accepts at
  most 32 views; initial direct 50-camera attempt failed in
  `reference/official-eval-dexsim.log`. Task now rejects >32 image envs before
  creating a World. Completed native rows' recorded results freeze at success;
  their isolated physical arenas can keep stepping. First images are dataset
  pixels in both engines, step-25 images are native/source engine pixels.
- Initial source state roundtrip max position error 9.59e-7, velocity 6.89e-8.
  Paired replay outcomes agree 40/50. Source-success/native-fail queries are
  [7,19,23,26,32,33,44]; native-success/source-fail [12,41,43]. First-step max cube
  error .204 m (query 33), median .00031 m. Entire replay p95 error .204 m.
- Extra 25-distinct-state FK/controller probe rules out initial controller/frame
  mismatch: max TCP 2.32e-6 m and arm target 6.55e-6 rad. Evidence:
  `alignment/initial-batch-dexsim.npz`, `initial-controls-reference.npy` and logs.

### Rendering, verification and active diagnostic

- First rendering was gray because imported colorless OBJ meshes took precedence
  over URDF color. Added `ArticulationCfg.urdf_override_material`; task enables it.
  Export uses MuJoCo's effective scene rgba (rather than always overriding geom
  colors with mat_rgba) and retains compiled checker texture bytes. Source and
  native cube/gripper/texture boundaries align visually; lighting still differs.
  `alignment/query-0-source.png`, `query-0-native.png`, `render-contract.json`.
- SDK geometry getter now retains actual shape local poses. Support-direction
  comparison finds gripper cooked hull differences ~0.6 mm, wrist max 2.34 mm.
  Shape order changes, so match by geometry instead of XML index.
- Current broad Python checks: 260 passed, 1 deselected before the newest camera
  capacity test. API 3283/3283; context map check passes. Re-run changed focused
  cases/checks after current edits. No commit or publication.
- Context revised in simulation-system overview, articulation-adapters and
  visual-world-model. Robot/env/visualization owners remain accurate and need no
  duplicated field inventories. OFFICIAL_CUBE.md now records actual measurements
  and priority-ordered remaining gates.
- Contact-parameter probe reads actual MuJoCo `efc_KBIP/R` for failed states:
  floor force stiffness ~4956 N/m per contact and damping125.5 Ns/m; pad stiffness
  ~77-86 kN/m, damping645-714 Ns/m. See `alignment/contact-reference-parameters.json`.
- Contact-softness diagnostic completed: `alignment/contact-softness-probe.json`.
  Eight queries x four modes. Softening compounds using source-derived force
  stiffness/damping did not fix the worst first-step errors: hard max .219 m,
  pad-soft max .368 m, all-soft max .289 m. Friction-only max .141 m but worsens
  median. These approximations are falsified as sufficient fixes and were NOT
  retained in production. Failed setter-order run is preserved separately;
  existing SDK requires enabling compliance (e.g. set_damping) before writing
  negative restitution. Native exclusions have different semantics, below.
- CURRENT diagnostic: `reference/probe_point_drives.py`, tool session to inspect
  in process/log. Compares canonical acceleration D6, zero point drives, locked
  points, and source-derived force-mode D6 across eight queries for one action.
  Source equality coefficients from `alignment/equality-force-reference.json`:
  stiffness roughly 335-464 kN/m, damping 3495-4766 Ns/m. The first run failed
  because Arena.get_constraint returns the base wrapper; use the typed D6 handle
  retained by Core's `get_rigid_constraint(...).constraint_handles[i]`. No SDK
  change was needed. Preserve `point-drive-probe-base-handle-failed.log`.
- Paper v1 was re-read directly at https://arxiv.org/html/2603.19312v1.
  Appendix D gives ten Cube CEM steps (Appendix B gives 30 generally), Appendix E
  ten training epochs; the pinned runnable configuration uses 30 steps / 100
  epochs. Main-method SIGReg weight .1 vs example .09. Plan now distinguishes
  these without silently mixing paper/config reproduction claims.
- Latest post-capacity test run: 61 passed. Broad run remains 260 passed before
  that added case; black --check . passes all 1733 Python files; compileall,
  diff whitespace, API coverage and context map checks pass. Context tests were
  launched; inspect completion before quoting their count.
- IMPORTANT: native collision mask words 2/3 are EXCLUSION masks: an intersection
  kills a pair, opposite to MuJoCo contype/conaffinity. Existing robot [type1,mask0]
  and wall [type2,mask1] correctly exclude robot-wall pairs. Do not naively replace
  them with MuJoCo contype/affinity values; that hypothesis was disproved by the
  actual filter shader (`DFPhysic.cpp:584+`).

Immediate priorities: finish contact diagnostic; implement only verified per-pad
contact/softness mappings at SDK/typed Core ownership; validate loaded mechanism
and failed first-step states; complete 28-D state/right-pad wrench; quantify visual
and CEM arithmetic differences; add paired random/zero controls and original
training recipe. Full physical-alignment flags remain false.

## Initial ownership and baseline record (historical)

- EmbodiChain worktree: `/root/sources/EmbodiChain-pr895-lewm`, baseline
  `a5ea7ee0acd5eefc0bfabbd6fb0b77ee988ee65c`, branch
  `feat/lewm-visual-world-model`. Preserve previous uncommitted UR experiments.
- SDK dependency worktree: `/root/sources/dexverse-lewm-constraints`, detached
  baseline `f0e752c00937b3afccfb24c993a6ad36023c5dcf`.
  Original `/root/sources/dexverse` is untouched. No SDK publication or merge.
- Parameter contract and ordered TODO:
  `/root/sources/EmbodiChain-pr895-lewm/examples/learning/lewm/OFFICIAL_CUBE.md`.
- Official references and hashes: `alignment/reference-report.json` and
  `alignment/physical-contract.json`. Source audit requires clean pinned checkouts.
- MuJoCo reference venv: `reference/venv` (3.11.0, dm_control 1.0.41).
  Existing global MuJoCo was restored to 3.12.0 after dependency resolution.

## Verified evidence

- Official Cube checkpoint SHA256 verified; numerical comparison with upstream:
  `alignment/model-reference.json` (includes relative/absolute tolerances).
- Official 300-sample/30-iteration/25-input CEM: all candidate populations compared
  with upstream on an analytic objective; final actions match exactly:
  `alignment/cem-reference.json`. Dataset moments are not yet used in this check.
- Native DexSim robot kinematics and positive-mass source inertias passed using
  explicit pre-build link overlays: `alignment/dexsim-kinematics-report.json`.
  This steps no physics and does not certify zero-mass frames.
- Initial direct URDF import changed small inertias/collisionless masses:
  `alignment-unmerged/dexsim-native-import-report.json` retains failed evidence.
- Focused model suite: 29 passed. Context map check passed; API docs 3281/3281.
- SDK installed baseline lacks `Arena.create_d6_constraint_bodies`.
  The new endpoint test fails with that exact AttributeError:
  `reference/sdk-baseline-d6.log`.

## Current SDK extension

Requirement: attach two Robotiq articulation links at a point, preserving all
rotational freedom. Direct GPU uses D6's supported GPU constraint preparation.

Options considered:

1. Extend the existing Arena rigid-body endpoint family with D6: reuses the
   fixed body's endpoint/ownership validation, owning Scene thread and cleanup.
   No virtual ABI/layout, public defaults or ownership semantics change.
2. Add a generic typed Scene constraint descriptor for every joint type: useful
   longer term but broadens the scene schema and backend lowering beyond this gap.
3. Add an entire MJCF source/import path: preserves more source metadata but
   requires a new asset/import/backend integration and does not solve Default
   contact differences by itself.

Selected 1, the compatible local extension. D6 linear axes stay locked; all three
angular axes are freed for the source's connect equalities. This expresses the
point topology; compliance and measured trajectories still require validation.

Changed SDK files: `Arena.h`, `Arena.cpp`, `environment.cpp` pybind, and existing
`python/test/engine/test_constraint_bodies.py`. Endpoint and owner cases are
parameterized for fixed/D6; a CPU/Direct-GPU point connection case checks motion
and automatic cleanup when an articulation is removed. Tests are not yet verified
against the new binary. Keep old failures; do not claim compilation as acceptance.

Fresh module build: `modules/dexsim/build_lewm`, owned by py311, with dependencies
in `/root/sources/dexverse/3rdparty_downloads`. Never transplant a CMake cache.
Configure log `reference/sdk-configure.log`; build log `reference/sdk-build.log`.
Build command: `cmake --build build_lewm --target DexSim pybind -j12`.
Current tool session: 22649; revalidate it before deciding to resume/restart.
Use an isolated staged SDK package for runtime tests. Do not overwrite the
installed SDK used by other work.

## Data transfer

Pinned archive revision `02a19a67a0dc8c9d6215f89c19e0a597691e152a`, expected
46,184,624,478 bytes and SHA256 in the alignment plan. First HTTP transfer
terminated with proxy/read timeouts. Current Xet/resume download PID 2397077,
tool session 44451, log `reference/dataset-download.log`; it was verified live
with about 455 MiB staged. Revalidate PID/session and file sizes; a timeout or
lock file alone is not proof of termination. Direct external HF access fails,
proxied access works intermittently. Other asset/control work can continue.

## Next work

1. Finish and test SDK D6 body API, including both CPU and Direct GPU, then stage
   its built artifacts and record exact paths/hashes. Fix only verified failures.
2. Collapse fixed source bodies with exact composite spatial inertias, preserving
   virtual source/site frame bindings. This removes native epsilon mass from
   source massless fixed frames; do not hide that gap by excluding them forever.
   The previous unmerged export/evidence is preserved in `alignment-unmerged`.
3. Port five-dimensional target/IK semantics, tendon and passive forces, contact
   material mapping and exact 25-substep timing into the registered task.
4. Validate open-loop physical trajectories, camera/rendering, then official
   50-query MuJoCo/DexSim comparisons and the requested training recipe.

Completion requires all gates in OFFICIAL_CUBE.md. Keep
`dexsim_physics_alignment_verified=false` until the full task and evaluation pass.

## Real14 coupled impulse phase — active

Previous goal turn PROGRESS; fresh78-source/runtime hashes match and no preceding
validationprocess lives. Requirement: isolate real14 source point/mimic response
at identical quantized q/v and declared mass/armature, dt.002. Four constraint
sets none/point/mimic/both and force tiers undamped/passive/controlled; contacts,
limits, frictionloss andgravity off only in the isolated diagnostic, not the task.
Preserve explicit source pre-implicit qacc/force, post-implicit q/v and native
PGS motion/final velocity. Native importer uses21positive bodies, rootfixed,
selfcollisionoff with enabled shapes (no contacts/no all-disabled broadphase
trigger), zero native drives and exact precomputed smooth generalized force.
Source derivative is exported independently; no MuJoCo in native process.
Mimic presence is selected by private copied URDF, not live native mutation.
Point gains compare accepted static mapping and verified discrete/target mapping.
Source files and production task remain unchanged during this characterization.

Options: A tune gains against full50 (already rejected, hides phase mismatch);
B isolate real coupling and integration stages through existing public owners
(recommended); C implement a new native point row immediately (not yet justified
by actual14DOF evidence). ChooseB. Acceptance: independent dense source row solve,
none-constraint free-step reference, actual native PGS creation descriptor and
quantization bounds, exact input/force/mass metadata; retain all failures. Only
expand production changes after this gate. Pinned3.11 engine_forward.c downloaded
SHAf44d0de9a2f81faf58812c70b97bf0e64070561134e4c7f6a46e3a176e8fd862 confirms
constraint qacc precedes implicitfast M-dt*qDeriv update. Native provider remains
engine022c4489/GPU25d44, installed/shared SDK untouched.

## Separate-anchor GPU D6 drive extension — specification

Real14 oracles confirm native articulated COM/angular bias velocity. Independent
model error .0199 -> .000126rad/s with that term; source1050COM rows matchbias
9.44e-16. Corrected target reduces pointsource pre-implicit maxerror .02114 ->
.002785; remaining error agrees with native anchorB-versus-source anchorA force
Jacobian. Mimic alone is predicted within3.55e-7 using soft position rows andone
hard velocity projection. Time-discrete mimic gains at0velocityiterations reduce
its maxerror .0385 -> .000412, without claiming complete moving-row parity.

Requirement: opt-in GPU D6 linear spring applies actor0 moment at authoredanchorA
and actor1 atanchorB. Linear force axes/velocity target stayactor0 jointaxes;
rotationfree, standardD6 behavior unchanged whenoff. Data/vtables/layout/lifetime/
registry/thread/batching unchanged. Source actor acceleration-bias correction is
external and separately verified. CPU or mismatchedGPU library rejects enabling.

Threeoptions: A existinggain/targetonly (proved cannotcorrectforceJacobian);
B newcustompoint PxJoint (requiresnewjoint/lifecycleadapter andGPUpreparation);
C opt-in private driveflag inexisting496-byte D6GPU descriptor/prep andexisting
Constraint/Arena owner (recommended; no new owner/solver/descriptorfields). ChooseC.
Use bit2 (bit0acceleration andbit1outputforce alreadyowned). Query exported scalar
capability marker from ALREADYLOADED GPUmodule; do notloadanotherGPUlibrary or
assume enginebuild impliesGPUsupport. Preserve opt-in acrossgainupdates. Public
SDK free/nonvirtual functions only; no newnative/thirdpartytype crossesABI.

Acceptance: missingcapability baseline; independent12D finite-point CPU-negative
andGPUdefault/opt-in spring response; toggling/lifecycle, oldGPUreject; fresh
matching stagedbuild andGPUfeaturemarker; rotating/free-body andreal14 point/
mimicoracles; default45/full50 exacttrace neutrality; affected regressions and
1/25arena benchmark. No production task promotion until fullphysical gate.
Rollback snapshots include priorengine022c4489/GPU25d44 inreference/before-point-
anchor-extension. Shared/installedSDK andSDKcontext remainuntouched.

## Soft mimic velocity policy — specification and active work

Previous goal turn PROGRESS. Fresh113-runtime/source checks match, no prior job
lives. Requirement: explicit opt-in per-articulation PGS GPU soft mimic velocity
response preserves finiteERP/CFM/accumulated impulse during scene velocity passes.
Default legacy hard velocity projection unchanged; TGS/CPU/olderGPU reject enable.
Hard rows (nonpositive frequency/damping) keep their existing projection even
when opt-in is selected. Fixedbase/selfcollision flags, ownership, threads,
vtables, descriptor layouts and solver iteration counts unchanged.

Options A globalzero velocity iterations (verified unrelated body can override,
changes contact response); B optional existingarticulation flag andcurrentmimic
helper path (recommended); C replacement joint/solver (newowner/prep/lifecycle,
unnecessary). ChooseB. Private flagbit3 in existingPxU8articulationflags; GPU
capabilitymask2 from alreadyloaded provider. Reuse existingScene/Node worker
Sync and SpawnedArticulation facade. Small shared private capability helper
factors existingD6 lookup, no newregistry/thread/cache/publicnative type. New
free exported functions and bindings, no addedvirtual member. Root runtime
method checks all selectedcapabilities beforewrites. No task default promotion.

Acceptance: missingAPI baseline, independent2hinge backwardEuler step with
distantbody forcing velocity iterations, mixedopt-in handles, reset/toggle,
zero-damping hardrow unchanged, CPU/TGS/oldGPU rejection, freshsame real14
farbody/coupled oracles, defaultTGS/PGS neutrality, fullaffectedregressions and
25/256chain completed-step benchmark. Thenexpert/full50 checks under samequery
protocol beforeany taskrecipechange. Backups before-soft-mimic-velocity include
engine57dcde33/GPUd499a135. Installed/sharedSDK andSDKcontext remainuntouched.

## User pause and PR895 publication checkpoint

Goal status PAUSED by explicit user request. Do not resume technical experiments
until requested. User clarified remote target is existing draft EmbodiChainPR895,
not the SDK internal remote. CurrentRoot changes, public-source PhysXpatches,
compact reports and project-owned diagnostics are being saved to that PR. Private
SDK implementation stays local. Soft-mimic-velocity C++/CUDA builds completed,
but post-build physical, benchmark and full-task checks are NOT qualified. Source
changes are WIP and defaulttask remainslastaccepted. MissingAPI baselines4SDK/
3Root failures predate implementation. Pausepublish checks129Rootfocuspass,
Black26.10.1/1775files,API3285/map pass. Historical102SDK/299Root counts qualify
priorpoint-anchorphase only. Stagedpackage has beenchangedbythebuild; inspect
paused STATUS.json instead ofassuming lastvalidated57dc/d499 runtime.

186GiB resources, checkpoints, thirdpartycheckouts, environments and binaries
remainlocal. Portable PR snapshot examples/learning/lewm/validation/paused-
2026-10-31 recordswhatissaved andwhatremainsunfinished. SDKHTTP/SSH authentication
failed, but no SDKremote push ispartofuser-confirmed publication. Local tracked
patch anduntrackedarchive saved under reference/paused-sdk-* for recovery.
