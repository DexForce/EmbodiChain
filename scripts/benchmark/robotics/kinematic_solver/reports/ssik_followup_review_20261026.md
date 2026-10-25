# SSIK follow-up implementation review

No actionable code findings remain in the reviewed scope. Aloha Mini and both Marvin variants passed the mandatory local gates and now use SSIK Warp arm defaults. ARX5 remains an explicit numerical hybrid; Rizon4S remains an explicit CUDA algebraic option. Hosted CI is recorded separately and is not certified by this review.

| Priority | Location | Defect | Impact | Recommended action |
| --- | --- | --- | --- | --- |
| N/A | N/A | No outstanding actionable findings | N/A | Keep the documented model and backend boundaries |

| Review item | Result |
| --- | --- |
| Target | Upstream `c5b1fdfc28ea47376933d95207b7894b7e6d54f1` through implementation `70c91d9c215f8defe8629021d1e456d16d8a8f38`, PR #873; final evidence reports reviewed separately |
| Scope | 79 tracked delta files plus three final reports; kinematics compute, solver/Robot adaptation, presets, CLI, packaging, context, documentation and validation |
| Execution | Multi-agent; independent delegated HPC/API perspectives share one reviewer context; the lead owns integration review; a separate delegated reviewer covers P5 and CI fixture verification |
| Findings | P0: 0; P1: 0; P2: 0; P3: 0 outstanding |
| Assessment | No code blocker identified for the delivered modes or Aloha/Marvin default switch; hosted CI remains pending |
| Validation | 102 strict-geometry CPU/CUDA checks; 21 RR/QZ/CUDA checks; 49 P5 config/Task Program/native checks; pinned reference and wheel checks; full CPU fast lane on P5 commit: 9,956 passed, 31 skipped; 121 post-main integration checks; post-main full CPU fast: 9,981 passed, 31 skipped, zero failures; final hermetic wheel PASS; old-head hosted CI: 10,814 passed, 35 skipped |
| Residual limits | Finite redundancy sampling; ARX5 near-singular branch incompleteness and bounded-path failures; Rizon CUDA-only runtime and one classified wide-grid miss; historical unmonitored ARX bulk denominator; hosted CI, concurrent rendering and physical-device timing |

| Role | Selection reason / scope | Executor / mode | Status / evidence | Candidate disposition |
| --- | --- | --- | --- | --- |
| architecture / robotics | Compute layering; root/TCP/joint contracts; original-model acceptance; optional offline artifacts; actual Robot boundaries and baseline identity | Lead `/root`; implementation plus separate integration review, not an independent author review | Full integration pass; source/model/input fingerprints, no-render Robot quality and timing, packaging and context | `robotics-1` false-PASS gate resolved; canonical preset relocation and mainline conflicts reconciled |
| hpc | Cooperative kernels, complete candidate pools, bounded scratch, stream lifetimes and synchronization; verification lens | `/root/arx5_cuda`; delegated pass excludes its own near-parallel mathematics | Static assigned-scope pass and inspected closure; hardware owner evidence is separately identified | `hpc-1` uninitialized-anchor defect resolved |
| api | Public config/CLI imports, resource packaging, replacement/restore semantics, Robot measurement contract | `/root/arx5_cuda`; separate perspective in the same delegated context | External-cwd wheel checks and static API pass; fresh post-main package probe is separately bound | No additional findings; missing-coverage gate closure confirmed |
| verification / P5 | Meaningful default/override tests, physical fingerprint preservation, final data qualification, CI wheel-guard fixtures | `/root/ssik_7r`; independent P5 delta and fixture review, prior mathematical/report authorship disclosed | Eight P5 files and four contract neighbors inspected; final P1/ARX/RR records reconciled; numerical sources unchanged through main merge | No actionable P5 finding; own prior mathematics is not counted as an independent mathematical audit |

Other specialist roles were not separately dispatched. The change does not add Task Program language, learning, messaging or credential behavior. The lead checked native frame/lifecycle boundaries; no native SDK modification is proposed. Numerical results belong to implementation owners and independent original-PK, SciPy/QZ or pinned-reference oracles. The review does not invent a second independent hardware run.

## Closed defects and integration changes

`hpc-1` exposed uninitialized anchor state when the first SRS elbow had no shoulder decomposition. The entire anchor block now stays inside the warp-uniform validity guard. All 32 lanes participate or skip together; candidate count, eight anchors and forty search iterations remain unchanged. The constructed five-degree nonorthogonal shoulder regression includes a distinct seed, complete masks, original-PK FK, limits, nearest/all agreement and CPU/CUDA periodic equivalence. The lead and delegated reviewer inspected the correction; 102 focused cases and all six-model CPU/CUDA quality cohorts passed afterward.

`robotics-1` exposed a benchmark false PASS when fast failure produced zero valid Robot solutions. Single and paired gates now require matching standalone validated coverage and zero FK, step and failure-hold violations. Missing comparisons remain UNVALIDATED. Behavioral cases cover missing data, fast failure and paired limits.

Vendor/model relocation invalidated hard-coded preset implementation paths. Benchmark provenance now resolves each public configuration class's actual module. The existing upstream wheel-fixture repair was reused, then superseded by the exact current mainline fixture when merging the NeRD package addition. No production guard was weakened. CLI checks reconstruct wrapped control-column text at 80/120 columns; the UR component test patches its canonical vendor module. The final merge changes no SSIK mathematical source, solver/Robot runtime, benchmark or four robot presets.

P5 switches only Aloha arm groups and Marvin bare/gripper arm groups. Torso, roots, endpoints, joint order and TCP are preserved. Complete typed and dictionary Pytorch replacements retain the original numerical settings and frame data. The P5 receipt is an immutable pre-commit snapshot; publication commit `4748f3de969b940e875bfcd6a609b128d463fd39` records the applied defaults. The independent P5 review is in the acceptance JSON.

## Validation scope and remaining limits

All accepted joint outputs are checked as returned float32 against the original URDF/PK chain and actual TCP at 1e-5 m and 1e-5 rad. Limit and enabled 0.05-rad step constraints are hard filters; failed slots retain finite seeds or previous postures. No runtime solver fallback is enabled.

The final six-model P1 CPU/CUDA cohorts pass regular, boundary and singular coverage and reject every constructed unreachable target. Strict-model control, paired control, all 18 actual Robot groups, 90 CUDA matrix rows and driver-inclusive scratch probes pass their mandatory gates. Twelve fresh original-default Pytorch bulk comparisons bind identical models, full cohorts and timed float32 target/seed bytes. The minimum median completed-call ratio is 33.237×. Original Pytorch retains its 5e-4 stopping policy and both outputs are independently judged at 1e-5; quality-matched P1 bulk timing is separately UNMEASURED.

Conditioned Rizon CUDA achieves 99.99% regular coverage, 100% boundary/singular coverage, no unreachable false positives and all 20,000 bounded trajectory frames. Current quiet-window control p99 is 3.159 ms, actual Robot p99 is 3.532 ms, scratch is 73.72 MiB and matching original/quality-matched bulk baselines exceed 2×. Its remaining regular sample is classified as a finite lock-grid miss. CUDA-only execution retains an explicit opt-in and the Pytorch default.

ARX5's six final quiet stages pass the control, Robot, matrix and memory checks. CUDA control p99 is 2.544 ms, actual Robot p99 is 2.875 ms and conservative driver growth is 34 MiB. The complete quality set has 95.225% bounded trajectory completion; the separately timed first ten trajectories have 9,045/10,000 successes, with zero false successes, step or failure-hold violations. Known extra near-singular algebraic branches are not enumerated by its nine-slot correction pool. Its old Pytorch bulk denominator is historical and unmonitored, so it is not relabeled as a fresh quiet baseline.

Final timing is accepted only for matching source/cohort and qualified measurement scope. Overlapping or unmonitored earlier windows remain historical. The root coordinated CPU quiet and recorded runtime thread/affinity settings; the one-second continuous monitor observes GPU/device activity, not CPU profiles. No unrelated training/evaluation process was paused. Actual Robot timing uses real native state retrieval with no rendering or physics stepping inside the call; it does not certify rendering load, physical transport or hard real time.

Hermetic wheel probes run outside the repository with editable import hooks removed, CUDA hidden and optional SSIK imports blocked. They verify prepared bundles, manifests, licenses, model fingerprints, defaults, complete Pytorch restoration and original-PK CPU FK. The final CPU fast run activates actual current setup.py source mappings in its process only. Earlier stale-cache and invalid-bootstrap attempts remain explicit environmental history; neither changes tests nor counts as a passing run. Hosted full CI passed on earlier commit `5acde0417` (10,814 passed, 35 skipped), including 467 GPU cases, 395 simulation cases and the fresh-process CUDA call. The distributed lane selected no tests (one deselected); it is not presented as distributed runtime coverage. Current-head hosted CI remains separate.

The [acceptance JSON](ssik_followup_acceptance_20261026.json) binds expert reports, full validation receipts, raw-data hashes and the immutable local archive index. No remote review comments or approval event were posted.
