# Replay the first five document tasks

This standalone launcher reproduces the five accepted DROID reconstructions from
[the task document](https://dexforce.feishu.cn/wiki/ZFkXwAdbZiSXLkkpeF2cjXaXnsh)
(document revision 133). It preserves the frozen runners, physics parameters,
recorded timestamps and acceptance thresholds used in the local qualification.
It is a historical reproduction tool, with a pinned runtime checkout; it does
not migrate these tasks to the current SDK APIs.

| Document order | Task | Episode | Frozen attempt | Frames |
| --- | --- | --- | --- | ---: |
| 1 | Organize desktop objects and close the drawer | 002 | `articulated_004` | 307 |
| 2 | Put the blue object into the tool cabinet and close the drawer | 003 | `articulated_005` | 131 |
| 3 | Put the pen into the yellow cup | 010 | `cup_018_final` | 374 |
| 4 | Close the storage box containing thin objects | 016 | `articulated_002` | 163 |
| 5 | Stack the red block on the green block | 019 | `frozen_angle_001` | 108 |

## External resources

The repository contains only this launcher and its instructions. The locked
`tasks.json`, validators, meshes, Blender scenes, recorded trajectories, original
videos and results belong to the separate asset delivery. No assets are fetched
or substituted automatically.

Set `DROID_REPLAY_RESOURCES` to the directory containing:

```text
resources/
├── tasks.json
├── validators/
│   ├── 002.py
│   ├── 003.py
│   ├── 010.py
│   ├── 016.py
│   └── 019.py
└── test_replay_document_tasks.py   # optional local regression checks
```

The original local resource directory is
`/media/dex/extra/data/droid/real2sim_80/document_first5`.
The delivered asset archive is
`/home/dex/Downloads/DROID_document_first5_complete_assets_20261105.zip`;
its extracted `replay_tool/` directory also provides the resource files above.
These are local delivery locations, not public download URLs.

The archive contains actual files, including all five frozen input sets, robot
resources and verified replay results. Its `PACKAGE_MANIFEST.json` maps original
paths to paths inside the archive. The replay manifest deliberately retains the
original absolute input, runtime and SDK paths. Extracting the archive on another
machine is sufficient for inspecting assets and results, but running the replay
there requires provisioning the pinned environment and remapping the manifest
and frozen runner paths. This launcher does not perform that migration.

## Run on the qualified machine

From the repository root:

```bash
export DROID_REPLAY_RESOURCES=/media/dex/extra/data/droid/real2sim_80/document_first5
python tools/droid_document_replay/replay_document_tasks.py --list
python tools/droid_document_replay/replay_document_tasks.py --check
python tools/droid_document_replay/replay_document_tasks.py --task 1
python tools/droid_document_replay/replay_document_tasks.py --task all
python tools/droid_document_replay/replay_document_tasks.py --task all --output /media/dex/extra/data/droid/first5_runs
```

Task numbers are document order 1–5, not episode IDs. Without a mode, all five
run. Without the environment variable, resources are read beside the launcher,
which also supports the original colocated delivery layout. The output default
is `runs/` under the resource directory. That directory must be writable: the
launcher owns its `.replay.lock` and updates its `latest` symlink after a batch.
Every batch uses a new directory and runs tasks serially. One task failure is
reported and the batch continues; a global preflight failure stops execution.

The entry process uses the Python standard library. Simulation and validation
subprocesses use the Python executable and SDK checkout locked in `tasks.json`:

- Python 3.10.19 at `/home/dex/miniconda3/envs/py310/bin/python`.
- SDK commit `1afcd308f4301d67c5a65298353dc2a7ebefd86c` at
  `/home/dex/.codex/worktrees/c284/EmbodiChain`.
- DexSim 0.5.1rc1, Torch 2.7.1+cu128, NumPy 2.2.6, SciPy 1.15.3,
  Trimesh 4.11.5, plus the existing OpenCV, Pillow and USD dependencies.
- NVIDIA GPU/driver, `nvidia-smi`, FFmpeg and FFprobe.

Keep the pinned runtime checkout separate from the checkout hosting this tool.
`--check` verifies its commit, locked SDK/input/robot file hashes, structured
asset references, imports and output writability. It does not install packages,
upgrade the SDK or download replacement resources.

## Preserved replay behavior

Each task gets an isolated staging tree built from its frozen `inputs/`, rather
than mutable episode-level assets. The derived runner changes only its root
path; AST comparison checks that the remaining code is unchanged. Inputs,
original and derived runner hashes, commands and environment settings are
recorded. Each batch also archives the executing repository launcher, this
README, the external manifest and validators. Validation subprocesses use that
archived resource directory.

The configuration uses the default physics backend, CPU physics at 240 Hz and
Hybrid rendering. Physics advances using recorded timestamps. Videos retain
30 FPS encoding, so presentation duration differs from acquisition duration.
Each task produces two 640×360 simulation views and a 1280×720 comparison video.

The exact historical robot resource is a Franka **FR3** with **Robotiq 2F-85**.
Its legacy library key is `Franka/Panda/Panda.urdf`; the document's Panda title
does not override the actual asset. The gripper's historical `joint_angle_rad`
field represents a model reference derived from an observation proxy, not a
measured finger joint angle or an exact original actuator command.

Gripper closure compensation is fixed to 0.020, 0.020, 0.045, 0.000 and 0.050 rad
for the five tasks. Episode 010 additionally fixes mimic frequency 5000,
damping ratio 1 and the frozen contact parameters. Geometry, appearance, mass
and friction retain the approximations disclosed in the frozen task evidence.

The launcher cleans up its own simulation/encoder process groups after each
run, timeout or interruption. Historical inputs and the original 40-task
acceptance count remain untouched.

## Outputs and acceptance

```text
runs/<batch>/
├── index.html
├── summary.json
├── tool/                     # exact launcher/resource snapshot
└── 01/                       # document task number; likewise 02–05
    ├── inputs_manifest.json
    ├── command.json
    ├── run.log
    ├── validation.log
    ├── replay_result.json
    ├── validation.json
    ├── sim/
    ├── comparison/
    ├── preview/              # source first frames, keyframes, settled state
    ├── source_assets/
    └── w/                    # isolated working tree and actual outputs
```

Normal run outputs share source resources through symlinks. Preserve their
source directories and `w/`; use the separately delivered complete asset archive
when actual independent copies are needed.

Acceptance is recomputed against the new rollout using the frozen validators.
Episode 003's simulated projections are recomputed before its original
thresholds are applied; Episode 019's historical publishing tail is disabled.
The worker never substitutes an old attempt's validation result.

Every frame and terminal stability sample is compared with the frozen baseline:
robot joints within `1e-5 rad`, positions within `0.1 mm`, orientations within
`1e-3 rad`, and articulation joints within `1e-4` in their native units. Fixed
articulations with no joints are supported; nonfinite states are rejected.
Videos are frame-counted, fully decoded and hashed. Task checks include objects
remaining in closed drawers, a released pen in the cup, the closed lid, and the
released stable block stack. A threshold violation retains results and reports
`not_consistent`; it never tunes the physics to force a pass.

The original complete rerun passed all five tasks (307, 131, 374, 163 and 108
frames per video). Recorded robot/object/articulation numerical differences were
zero apart from a quaternion comparison residual below `4.3e-8 rad`. This does
not claim byte-identical rendered video or qualification against current main.
Reports remain in the external delivery; no generated evidence is committed.

Exit codes: `0` all selected tasks reproduced, `1` a task failed or differed,
`2` preflight/global error, `130` interrupted.

## Local regression checks

The external delivery's four existing checks cover frozen-input isolation,
root-only source rewriting, zero-joint articulations and nonfinite states. They
can be reused against the repository launcher without committing data or test
fixtures. Run its CLI `--help`, `--list` and `--check` before starting a replay;
inspect the batch's `validation.json` files after completion.
