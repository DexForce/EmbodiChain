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
