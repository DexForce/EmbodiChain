# Library Pick/Place with complete RGB-D recording

Native execution and integrity evidence: [qualification](QUALIFICATION.md).

Run the configured Dual UR5 verified Pick/Place task inside the existing
2,000,000-point Library Gaussian scan. One managed Camera records every control
boundary, including the post-reset initial and terminal frames. This composes
the existing Task Program executor, native trajectory writer and `MediaSpool`.

## Run

Use Python 3.12, an installed EmbodiChain checkout and a qualified DexSim build
supporting Gaussian USD references and whole-scene Gym import (#1157). The first
Library download requires the corporate asset server. Its archive and four
dependencies are verified against the existing
[capture manifest](../../sim/assets/gaussian_scenes.json) and cached under
`EMBODICHAIN_DATA_ROOT`. Scan binaries are not distributed with this example.

From the repository root:

```bash
PYTHONPATH=. python examples/gym/library_pick_place/run.py --prepare-only
PYTHONPATH=. python examples/gym/library_pick_place/run.py --episodes 1 --seed 42
PYTHONPATH=. python examples/gym/library_pick_place/run.py --episodes 10 --seed 42
```

Or use `embodichain run-example gym/library_pick_place/run`. The default is
headless; `--viewer` opens a native window while the Camera continues recording.
Headless mode defaults SDL to its dummy driver. If the host requires an XDG
runtime directory, create a private directory and set `XDG_RUNTIME_DIR`, as in
the project's native CI configuration. `--output` selects the output **parent**;
each invocation creates a fresh run UUID directory. `--data-root` selects the
asset cache. No retry hides a failed episode.

Pass the printed run directory to the CPU-only offline verifier:

```bash
PYTHONPATH=. python examples/gym/library_pick_place/verify.py /path/to/run_uuid
```

Verification streams local files without a World, asset server or GPU. Moving
the complete run directory preserves artifact references. Source-cache paths in
SDK provenance remain diagnostic; the verifier does not dereference them. Raw
NPZ media is the image-integrity reference: later rendering need not reproduce
Monte Carlo RGB noise bit-for-bit.

## Scene and configuration

The Library download is an acceptance ZIP, not an AssetRevision ZIP. Preparation
reuses `gaussian_scenes.py --download-only`, then passes the extracted
`asset.usda` through `simulation.usd_scene`, selecting `/Asset/Visual` only. Its
room-sized collision sibling is excluded. Source bytes remain unchanged.
`Rx(-90 degrees)` and translation `(0.8, 3.0, -0.05)` map the original camera-up
direction into world `+Z`.

The environment owns an independent static tabletop and dynamic 50 mm, 50 g
cube. Explicit cube/support bindings select the existing left-arm profile and
`motion_gen_verified` policy. Acceptance requires at least 4 cm lift, release,
settling, final position within 2 cm and tilt at most 15 degrees. An orange cube
material improves its visibility in single-sample, undenoised mesh rendering;
the scan retains captured radiance.

Preparation snapshots the SDK embodiment and replaces its independent sensor
list with one 960x540 RGB-D `cam_high`; robot/hand calibration is retained.
Runnable YAML and preflight evidence are written to `resolved/`. Templates need
this preparation before use. The standard source-checkout inspector also works:

```bash
PYTHONPATH=. python .agents/skills/add-task-program/scripts/inspect_deployment.py \
  /path/to/run_uuid/resolved/task.dual_ur5.yaml --json
```

## Recording contract

Physics runs at 100 Hz, with four physics steps per control command: 25 Hz
recording in simulation time. `T` actions produce `T+1` states, from post-settle
`s0` through terminal `sT`. The observer copies acknowledged normal and safe-stop
steps without another render or physics update. The initial frame is held
independently until the executor assigns its authoritative episode UUID.

- RGB: original first three color channels, uint8, 540x960x3.
- Depth: original float32 view-Z in metres, 540x960. Positive infinity preserves
  the native invalid-background meaning; depth is not quantized to video.
- JSONL: matching camera/frame/group identity, calibration, actual simulation
  time, Gaussian instance metadata and output contract for each observation.
- Media: a 64 MiB retained-frame budget and streamed metadata. Flush stacking
  has transient copies outside that budget. Full image rollout storage and the
  dataset manager are disabled.
- Trajectory: the existing SDK expert action schema, physical and terminal
  states, segment outcomes and canonical UUID. Added `rgbd_recording` metadata
  contains relative, hashed media references.

```text
run.json                         # versions, hashes, all attempts and run status
resolved/                        # environment/embodiment/program/policy snapshots
episodes/<episode_uuid>/
  captures.jsonl
  execution_result.json
  trajectory.pt
  media/manifest.json
  media/chunk-*.npz
  preview.png                    # initial / lifted / release / terminal
  episode.json                   # completion marker, written last
```

Completion follows physical acceptance, media flush, trajectory persistence and
offline verification. Any failure returns nonzero, retains available diagnostics
and stops. Partial episodes have no complete marker. A capture failure disables
further writes while the executor performs its ordinary safe stop. Preparation
failures are also recorded in `run.json`.

The verifier checks file/chunk hashes, all `T+1` identities and timestamps,
fixed calibration, action lengths and independently measured lift/final pose.
It rejects missing files, changed bytes and missing completion markers. Metrics
include wall time, effective control-step throughput, capture time, peak retained
media bytes and process peak RSS. No real-time rendering rate is assumed.

## Replay and qualification

Use the generated deployment with the existing native trajectory replay:

```bash
PYTHONPATH=. python -m embodichain run-task \
  --gym-config /path/to/run_uuid/resolved/task.dual_ur5.yaml \
  --renderer hybrid --replay --replay_mode control \
  --replay_trajectory /path/to/run_uuid/episodes/episode_uuid/trajectory.pt
```

Replay needs the verified Library cache and SDK robot assets. For a moved cache,
prepare again with `--data-root` and use the new deployment. Normal USD admission
checks the dependency bytes. Gaussian appearance uses one Arena, Default physics
and Hybrid rendering; use the native window, since Viser does not export splats.

One `--episodes 10` invocation exercises repeated reset in a shared World.
Every requested attempt must complete and persist. Inspect the saved previews
for scene/robot/object visibility and occlusion, then verify after moving the
run directory. On disposable copies, remove a chunk, change an artifact or
remove `episode.json`; verification must return nonzero. Production regressions
cover Gaussian subtree selection, normal/safe-stop observers and exact
float32/invalid-depth media ownership. The example exports raw recordings and
trajectories; LeRobot conversion is outside this example.
