# Native qualification: standardized 3DGS backgrounds

Measured on 2026-11-16 with Python 3.12.16, DexSim 0.6.0rc2, an NVIDIA RTX
PRO 5000 72 GB Blackwell and driver 580.126.20. Default CPU physics, Hybrid
rendering, 8 spp, OptiX offscreen denoising and `apply_display_srgb=False` were
used. Both RGB sensors output 960x540 at the 25 Hz simulation cadence.

## Published assets and evidence

- [Standard USD/AssetRevision packages][assets], immutable asset snapshot
  `3ec5b1fb7d320599c54f07bc20d4c49d570a2e2b`.
- [Complete three-task sensor video][video], [simultaneous comparison][comparison]
  and [third-person/wrist preview][preview]. Each scene also has separate
  full-resolution camera videos in the [evidence folder][evidence].
- [Machine-readable qualification][report] records asset/revision identities,
  configuration/source hashes, every preview hash, physical metrics and timing
  scope. Validation snapshot: `c819d09cd3aaf7c9fc25068288cd4d2e36fbb6ab`.

The original upstream PLY bytes were retained and hashed. Lab3's compressed
source also has a separately decoded render PLY. All packages declare metres,
kilograms and Z-up, preserve their original metric origins, and contain zero
colliders/rigid bodies. Office's source-axis conversion is authored once in USD.
AssetRevision 2.0-draft package admission passed for all three; quality remains
incomplete and no industrial certification is granted.

All three registered datasets were downloaded through the shared downloader
into an initially empty cache from the immutable Hub commit, extracted, and
checked against their published AssetRevision identities and hashes. The native
task runs below used those downloaded packages through `VisualObjectCfg` USD
references, rather than the original local PLYs.

## Live Task Program outcomes

One seed-42 attempt was made per scene; all three processes exited zero. These
are live Task Programs using the existing FR3/parallel hand, home pose, verified
policy and normal physics. No prerecorded robot actions or object teleports
were used. Measured attach/release effects and the final settled post-policy
passed, alongside independent trajectory/capture-state checks.

| Scene | Actions | Paired states | RGB images | Lift (m) | Final error (m) | Final tilt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RoboSplatter office | 284 | 285 | 570 | 0.129430 | 0.007269 | 0 degrees |
| DISCOVERSE lab3 | 284 | 285 | 570 | 0.129431 | 0.007237 | 0 degrees |
| RoboSimGS tabletop | 283 | 284 | 568 | 0.129429 | 0.007267 | 0 degrees |

Total: **851 actions, 854 paired observations and 1,708 original RGB images**.
Both cameras include the initial and terminal states. All records passed UUID,
native frame identity, completed Gaussian-capture, calibration, actual clock
and 0.04 s cadence checks. Every encoded camera-video frame was decoded and
compared against its raw RGB source with maximum mean absolute uint8 error <=8.
The MP4 files are lossy previews; NPZ RGB remains the image-consistency source.

The wrist camera moved about 0.4252 m and maintained its rigid hand-relative
extrinsic throughout all three episodes; maximum translation disagreement was
under 0.2 micrometres. The hand-axis convention and projected gripper position
were checked: the gripper is below the working point and the jaws close
horizontally. This correction is in the camera extrinsic, not video processing.

## Recording cost and checks

| Scene | Simulation duration (s) | Wall time before video (s) | Observer work (s) |
| --- | ---: | ---: | ---: |
| Office | 11.36 | 23.49 | 4.57 |
| Lab3 | 11.36 | 34.65 | 16.32 |
| RoboSimGS | 11.32 | 22.11 | 4.61 |

These are single samples on a shared host; Office and Lab3 ran concurrently.
Wall time includes World creation/reset, execution, writing and the physical
artifact audit, and excludes video export. Observer time includes image copying,
metadata and NPZ spooling; it does not measure native rendering. The maximum
observed MediaSpool retained-image buffer was 65,318,400 bytes per run, below its
64 MiB budget. Temporary copies, compression workspace and native GPU memory
are outside that retained-buffer figure. No rendering speed or regression claim
is made from these samples.

Focused validation for this addition:

- 51 asset/download tests passed, including immutable descriptors, the shared
  registry, versioned cache selection and custom mirror fallback.
- 352 renderer/configuration tests passed, one skipped; the display-transfer
  override, serialization and default behavior are covered by extended cases.
- 42 context-map/tool tests and 11 API-doc checker tests passed.
- Black 26.10.1 (`black .`), syntax checks, `git diff --check`, context validation
  and public API coverage (4916/4916 exports) passed.
- No tutorial/example-import unit-test harness was added. The three real native
  runs and raw/video audit exercise the example's observable behavior.
- The real `embodichain run-example ... --prepare-only` entry passed. A moved
  Office recording passed the offline CLI; disposable copies with a missing
  completion marker, incorrect artifact hash or missing media chunk were rejected.

The local environment has no Sphinx installation, so a full Sphinx build was
not run. Native-window preview was not exercised on the headless host. This is
one attempt per scene, not a multi-seed or ten-episode reliability qualification.

## Offline verifier regression qualification — 2026-11-16

A further Office episode was recorded on the same Python 3.12 / DexSim rc2
environment, using physical GPU 2 with the existing Hybrid/8 spp/OptiX settings.
It completed 284 actions, 285 paired observations and 570 RGB images. Measured
lift was 0.129430 m, final position error 0.007269 m and final tilt 0 degrees.
Run identity: `office-20261116-135519-a327e447`; episode:
`a6ca9055-3311-4361-a9c2-e7066cfabbf0`.

The real standalone verifier CLI was then exercised against that native
recording and isolated copies, with `CUDA_VISIBLE_DEVICES` empty. The original
verifier from `a90785d3c` incorrectly accepted all seven damaged variants. The
corrected verifier rejected every variant with its expected diagnostic:

| Input | Original verifier exit | Corrected verifier exit |
| --- | ---: | ---: |
| Original native recording | 0 | 0 |
| Empty episode file inventory | 0 | 1 |
| Missing wrist video and its digest | 0 | 1 |
| NaN action, trajectory digest updated | 0 | 1 |
| Infinite action, trajectory digest updated | 0 | 1 |
| Changed environment configuration | 0 | 1 |
| Empty configuration hash inventory | 0 | 1 |
| NaN target, configuration digest updated | 0 | 1 |
| Original recording after all checks | 0 | 0 |

The original recording's fingerprints were unchanged. All nine corrected
checks passed; this includes seven expected nonzero exits and two valid-run
checks. A separate relocated copy also passed the standalone CLI without a
visible GPU. `_negative_checks.py` retains per-case CLI logs, verifier SHA256 and
`acceptance.json`. Its `--verifier` option can select a saved older verifier
and its sibling helpers for the before/after control. No unit-test harness
imports the example: these checks exercise the real CLI and retained native
artifacts rather than mocked sensor or storage data.

Completion inventory is shared with the writer; the standalone CLI and video
collection exporter use the same run verification path. Configuration checks
also apply before an episode's completion marker is published. Offline failure
no longer overwrites `physical_metrics.json`. These changes affect artifact
admission, not simulation, rendering, control or metric GS-depth support.

Black 26.10.1 (`black .`), changed-source syntax/header checks and public API
coverage (4916/4916) passed. Context impact was reviewed; existing recording and
ownership guidance remains accurate, with example-specific checks documented
in the example README. Earlier asset/rendering test counts above were not
rerun for this verifier-only correction.

## Main synchronization qualification — 2026-11-16

The candidate merging `main@8170fca184da253c617fe79dd0f635598587c46f`
into `a6bcf6c1a` was requalified after the sensor-context and Task Program
updates. Conflict resolution retains the upstream display-transfer CLI guidance,
the PR's USD/ZIP visual-subtree isolation cases, and both numeric/string rejection
cases for the display-transfer boolean.

- All three deployment inspections passed without World construction.
- 664 focused rendering/configuration, USD admission, Task Program composition,
  preview, visualization and asset tests passed; 15 native cases were deselected
  from that lane. One unrelated Dash/pytest plugin deprecation warning remains.
- All three fresh seed-42 native attempts passed, with process exits zero:
  851 actions, 854 paired observations and 1,708 RGB images. Measured lift was
  0.129429–0.129431 m, final position error 7.237–7.269 mm and final tilt 0 degrees.
  Every camera video was decoded against its raw RGB source; the largest
  per-frame mean absolute uint8 compression error was below 1.640.
- The complete native collection is identified by
  `franka-gs-20261116-183539-1e2537d1`. Its bundle audit includes measured
  attach/release, settled placement, T+1 camera alignment and rigid wrist mounting.
- The offline original-recording check and all nine negative-acceptance checks
  passed with no visible GPU, preserving the earlier three P2 fixes and the
  original artifact fingerprints.
- Black 26.10.1 (`black .`), context-map validation, `git diff --check` and
  public API coverage (4961/4961 exports) passed.

The first GPU-hidden deployment-inspector attempt encountered this host's
missing `open3d.cpu` binding. Inspection and the focused tests passed in the
qualified GPU-visible native environment; recording verification and negative
acceptance separately passed with GPU visibility disabled. No dependency
fallback or production-code workaround was added. These are one attempt per
scene at one seed, not a multi-seed stability qualification.

## Task-extension migration qualification — 2026-11-16

The three deployments now belong to the official `gaussian_pick_place` task
under `embodichain_tasks`; embodiment and execution-policy components belong to
the shared component directories. No task Python subclass was introduced.
Examples retain capture, portable snapshot preparation, video export and
offline/negative acceptance. All 14 pre-migration resolved YAML declarations
were compared by value and retained, including the existing task/camera IDs,
calibration, policy and physical settings.

66 focused catalog, package-data, official-layout, example-packaging and
configuration-path tests passed. The actual wheel-install test was rerun after
the recorder's asset identity lookup was changed to derive its reference from
the official environment instead of another example-owned asset mapping.
42 context tests passed; the new task query routes to `env-framework`.

The native-qualified wheel, `embodichain-0.3.2-py3-none-any.whl`, has SHA256
`944dc8eba0912513370571f9d78855990dd8096d250abd69539177727dc08254`.
It was installed without editable source into an isolated virtual environment
reusing the qualified native dependency environment. Commands ran from an
unrelated `/tmp` directory with `PYTHONPATH` empty. Core, task and robot module
paths and all configuration paths were checked to belong to that installation.

| Official deployment | Consecutive accepted episodes | World constructions | Observed resets | Maximum reset joint delta |
| --- | ---: | ---: | ---: | ---: |
| Office | 2 | 1 | 6 | 0 |
| DISCOVERSE lab3 | 2 | 1 | 6 | 0 |
| RoboSimGS tabletop | 2 | 1 | 6 | 0 |

The standard installed `run-task --max_episodes 2 --record_trajectory` path
performed the resets; no example execution loop replaced it. Qualification
instrumentation only counted actual construction/reset calls and sampled
state. World identity remained stable, cubes returned to their configured
initial support positions, camera UIDs stayed `cam_third`/`cam_wrist`, and both
saved episodes had distinct UUIDs and successful first attempts. All six
episodes passed lift and final-placement bounds, along with the original
native action-contract checks.

The final installed `run-example .../run_all` also completed all three full
recordings: 851 actions, 854 paired observations and 1,708 RGB images, with
physical and decoded-video audits passing. Collection identity:
`franka-gs-20261116-220002-db809168`. Comparisons with pre-migration recordings
passed for root/metadata keys, action dtype/dimensions and joint ordering,
state/terminal-state leaf schemas, `rgb_recording` keys, raw RGB feature schemas,
camera intrinsics and the wrist mount transform.

Installed default `show-task`/`list-task` discovery passed for all three original
Gym IDs. This host's CUDA-only Open3D installation still prevents default
runtime-package discovery when all GPUs are hidden; static `--config-root`
catalog inspection passed without a visible GPU. Installed offline verification
accepted old original/relocated recordings, and all nine acceptance checks
passed against the new recordings with no visible GPU. Original recording
fingerprints remained unchanged.

The release wheel adds a preparation guard against output nested inside the
official task directory and updated example documentation. Its SHA256 is
`7c574a1a41e8a4f75e50a2e06a4f122f5db8abcbd56744bb41f9e196b3c8870f`.
All 1,654 core/task/robot package files are byte-identical to the native-qualified
wheel, retaining the runtime and reset qualification above. After installing
the release wheel, all three installed `--prepare-only` commands passed and
again preserved the 14 resolved declarations; a self-nested output was rejected
before copying or constructing a World. The wheel-install unit case was rerun
successfully for this release.

Earlier installed candidate checks also passed (one target-install Office run
with two episodes, then a three-scene six-episode virtual-environment check and
three full recordings). Their logs are retained separately; the table above
and final collection refer to the native-qualified wheel. The first listing probe assumed
long IDs appeared on one line; checking the formatter's wrapped ID column and
the complete `show-task` output corrected that measurement without changing CLI
behavior. These are reset/packaging checks at seed 42, not a multi-seed
reliability claim. Metric GS depth remains outside this qualification.

## Shared-embodiment calibration regression — 2026-11-16

The three deployments now select one `franka_gs_dual_camera` component.
Each retains its original `cam_third` calibration through the closed
`embodiment.camera_overrides` selector. The component retains the original
robot/hand/home, wrist calibration, sensor suite and skill profile. Expanded
physical declarations and skill profiles match commit `5380f6283` for all three
scenes; Gym IDs, both camera UIDs and native trajectory schemas are unchanged.

562 focused Gym/configuration, Task Program, catalog, packaging and path tests
passed. After covering the camera's nullable default `up`, all 216 relevant
configuration/wheel cases passed again. Coverage exercises independent and
mounted cameras, file-free snapshots, partial calibration and representation
switching, unchanged later loads, and rejection of unknown/duplicate UIDs,
non-camera targets, forbidden fields, malformed/nonfinite numbers, nonpositive
focal lengths, invalid quaternions and degenerate/mixed look-at poses.
The three static deployment inspections and 42 context tests passed.

The final tested wheel SHA256 is
`63de7f24f5cd85d61f80bcc3003f81f395ca08ba269cca56bc16a0eb85e5d2ce`.
It was installed in a separate virtual environment with the qualified native
dependencies. From an unrelated working directory, with `PYTHONPATH` empty,
installed discovery and all three preparation commands passed. The public
`run-task --max_episodes 2 --record_trajectory` path produced:

| Deployment | Accepted episodes | Worlds | Observed resets | Maximum reset joint delta |
| --- | ---: | ---: | ---: | ---: |
| Office | 2 | 1 | 6 | 0 |
| DISCOVERSE lab3 | 2 | 1 | 6 | 0 |
| RoboSimGS tabletop | 2 | 1 | 6 | 0 |

Every reset checked the native intrinsic matrices, third-person position/viewing
direction, wrist-local position/orientation and qualified parent binding against
the original calibration. The World identity stayed fixed, cube/home state was
restored, and all six unique episodes succeeded on attempt zero. Each lifted
the cube at least 0.1294 m and finished within 7.3 mm of its target.

The installed recorder also completed all three scenes under the shared
component: collection `franka-gs-20261116-231414-4869ada0`, 851 actions,
854 paired observations and 1,708 RGB images. T+1 clocks/identity, measured
pickup/release/stability and decoded-video/raw-RGB audits passed. Comparing with
the previous migration collection confirmed equal root/metadata keys,
action dtype/dimensions and ordered joints, state/terminal-state leaf schemas,
`rgb_recording` fields, media feature declarations, intrinsic matrices and
initial camera/wrist transforms. These recordings used the preceding wheel;
the final wheel adds only nullable-default validation handling to the resolver
and repeated all six native episodes with the same effective calibration.

With no GPU visible, the final installed offline verifier accepted the relocated
recording, and negative acceptance passed 9/9 checks (baseline, seven isolated
corruptions, unchanged original). No recording format or example execution code
changed. Black 26.10.1, API coverage 4961/4961, syntax/header checks and the
context/skill validators passed.

Local evidence is retained under `outputs/pr1210-camera-overrides/`: expanded
configuration and trajectory-schema comparisons, both tested wheels, installed
preparation/native logs, reset samples, full recordings, and offline checks.
An earlier instrumentation probe stopped at reset because it compared the
runtime-qualified `FrankaPanda/fr3_hand` parent with the authored `fr3_hand`.
Correcting that audit assertion changed no production code or configuration;
both subsequent six-episode batches passed. These remain seed-42 regression
checks, rather than a multi-seed reliability claim or metric GS-depth audit.

## Native USD multi-arena qualification

Dependency: [DexSim MR !1628](http://192.168.3.16/Engine/dexverse/-/merge_requests/1628)
provides native companion cloning, owner visibility and the resource-ID query.
The submitted SDK commit is `abed3d05b`; its final formatting correction preserves
the qualified Python AST. An SDK build containing that implementation is required
for the four-arena path.

The current official deployments again reference the packages' `asset.usda`
visual subtrees with `source_pose_mode: preserve`. Office's rotation is authored
only in USD. No explicit `GaussianSplatCfg` replacement is used in these runs.

DexSim's candidate Python Spawn/Scene implementation clones each actor's
Gaussian companions through the existing native Arena cloning API, binds them
to the owning handle and cleans new records/cameras/lights before parents when
birth fails. Immutable resource IDs are exposed by the public
`SpawnedRigidBody.get_gaussian_resource_ids()` query. Owner visibility includes
its Gaussians; EmbodiChain restores selected native-Gaussian visibility on reset.
The existing qualified C++/CUDA artifact was unchanged. These results require
the candidate SDK patch; unpatched SDKs retain singleton support and reject
multi-arena USD rendering before declaration.

Final seed-42 runs used four arenas, Default/CPU physics, Hybrid/8 spp, OptiX,
960x540 RGB, 25 Hz control and `apply_display_srgb=False`:

| Scene | Accepted rows | Actions per row | Frames per camera | RGB images |
| --- | ---: | ---: | ---: | ---: |
| Office | 4/4 | 284 | 285 | 2,280 |
| DISCOVERSE lab3 | 4/4 | 284 | 285 | 2,280 |
| RoboSimGS tabletop | 4/4 | 284 | 285 | 2,280 |

All 12 measured Pick/Place rows passed. Minimum lift was 0.129421 m and maximum
final position error was 7.271 mm. All 6,840 RGB images and 24 encoded camera
videos passed the existing boundary and decoded-video/raw-RGB audits; maximum
per-frame mean absolute compression error remained below 1.640 uint8 levels.
Every capture names USD prim `/Asset/Visual` and has one resource identity shared
by its four rows. Hiding arena 1 changed only its two views; all pixel differences
in arenas 0, 2 and 3 were exactly zero. Captured cube poses matched all native
trajectory rows exactly, including terminal states.

The final SDK regression has 68 passing cases, covering both supported thread
modes, prepared/finalized/incremental creation, native transforms, shared
resource identities, visibility, deletion and failure cleanup. The longest
recorded test call was under 10 seconds. EmbodiChain's candidate-SDK native USD
matrix passed 11 cases; an unpatched SDK separately passed five singleton and
early-rejection compatibility cases. New feature success cases skip on older
SDKs, while the negative capability test remains enabled. Fast configuration
tests stub GS resources with local paths and make no archive requests.

Affected regressions completed in separate process groups: 497 passed and eight
opt-in/resource cases skipped across 505 collected cases, including a fresh
installed-wheel check. An earlier combined invocation received SIGTERM during
the wheel check and is recorded as incomplete, not a passing result. The wheel
case passed independently, and all other groups have complete XML receipts.

The existing registered `spawn_load_dexsim_64` scenario passed on both the base
SDK and candidate in serial runs with identical settings. The single samples
measured 48.04 ms and 45.23 ms for finalization of 129 objects in 65 arenas.
These shared-host smoke measurements are not a statistical speedup claim.

See the [native USD measured summary](evidence/usd_parallel_qualification.json)
and [four-arena RGB preview](evidence/usd_parallel_rgb_preview.png). Full frozen
candidate recordings are retained in `outputs/usd-gaussian-parallel-validation/`.
This is a seed-42 four-arena qualification, not a multi-seed stability or metric
Gaussian-depth qualification.

## Limits

Only the independent task support/cube physics and RGB streams are qualified.
Captured chairs, shelves and tables have no scan-derived collision geometry.
Captured lighting is baked and is not relit with the robot. These recordings
do not establish metric Gaussian depth; the existing Library depth failure
documented in [its qualification](../library_pick_place/QUALIFICATION.md) remains
unresolved.

Data-license declarations are retained per package: RoboSplatter's dataset
card says Apache-2.0; RoboSimGS's says MIT. DISCOVERSE's model repository does
not separately declare an asset license, so its metadata retains `license: null`.
Its project's MIT software license is not inferred to cover the model bytes.

[assets]: https://huggingface.co/datasets/DexForceAI/embodichain_data/tree/3ec5b1fb7d320599c54f07bc20d4c49d570a2e2b/scene_assets/gaussian_backgrounds_v1
[evidence]: https://huggingface.co/datasets/DexForceAI/embodichain_data/tree/c819d09cd3aaf7c9fc25068288cd4d2e36fbb6ab/scene_assets/gaussian_backgrounds_v1/validation/tasks
[video]: https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/c819d09cd3aaf7c9fc25068288cd4d2e36fbb6ab/scene_assets/gaussian_backgrounds_v1/validation/tasks/complete_tasks.mp4
[comparison]: https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/c819d09cd3aaf7c9fc25068288cd4d2e36fbb6ab/scene_assets/gaussian_backgrounds_v1/validation/tasks/comparison.mp4
[preview]: https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/c819d09cd3aaf7c9fc25068288cd4d2e36fbb6ab/scene_assets/gaussian_backgrounds_v1/validation/tasks/preview.png
[report]: https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/c819d09cd3aaf7c9fc25068288cd4d2e36fbb6ab/scene_assets/gaussian_backgrounds_v1/validation/tasks/qualification.json
