# Pour Water

`task.cobotmagic.yaml` runs the existing right-arm pick, tilt and return Task
Program. The scene uses a RoboCasa open-neck bottle and red cup, a wood tabletop
with a Poly Haven material, and fixed area lights. Materials survive resets and
illumination stays constant throughout an episode.

The program commands `right_arm` and `right_eef`, while the environment retains
its full dual-arm action layout. The bridge preserves issued joint-position
targets for uncommanded joints on active rows, so the idle left arm does not
repeatedly accept gravity deflection as a new target. Explicit safe stops and
inactive rows still hold their measured positions.

## Run locally

From the repository root, using the project's simulation Python environment:

```bash
python -m embodichain run-env \
  --gym_config embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic.yaml
```

The registered `PourWaterAssets` bundle downloads and extracts automatically
when the task first resolves its models. To fetch it ahead of time:

```bash
python -m embodichain data download --name PourWaterAssets
```

To rebuild the adapted assets from their pinned upstream sources instead:

```bash
python scripts/tools/prepare_pour_water_assets.py \
  --archive outputs/pour_water/PourWaterAssets.zip
```

The preparation script needs the existing `numpy`, `open3d`, `trimesh` and Pillow
dependencies. It selectively downloads two objects from a pinned RoboCasa
archive, verifies ZIP CRCs and texture checksums, simplifies dense solid-colour
bottle parts, removes the bottle's removable flip-cap assembly for pouring, and
embeds materials into GLB files. It writes
`${EMBODICHAIN_DATA_ROOT:-~/.cache/embodichain_data}/PourWaterAssets/` so the
ordinary asset resolver finds the task's `PourWaterAssets/*.glb` paths.
`--output-dir` can build elsewhere; copy the resulting `PourWaterAssets`
directory into the chosen data root before launching.

The bottle retains the previous local Z bounds and grasp frame. The cup retains
its height with a 62 mm outer diameter. Both start just above the table's 875 mm
support height, avoiding deep initial interpenetration. The cup uses 16 convex
parts, 30 g mass and increased friction; reset waits for both objects to settle.
The two area lights and modified Reinhard tone mapping are task-local settings.

## Sources and redistribution

- Bottle and cup: [RoboCasa assets](https://huggingface.co/datasets/robocasa/robocasa-assets),
  revision `1b92c3d02ca4354984fec961357db0bff7b32166`, objects
  `water_bottle/water_bottle_6` and `cup/cup_4`, CC BY 4.0.
- Table texture: [Wood Table 001](https://polyhaven.com/a/wood_table_001),
  Dimitrios Savva and Rico Cilliers, CC0 1.0. Table geometry: DexForce, Apache-2.0.

The ZIP contains the GLBs, original selected sources, license texts,
`ATTRIBUTION.md` and a SHA-256 manifest. Explicit directory entries support
Open3D extraction of nested source files. `PourWaterAssets` is registered in
`embodichain/data/assets/obj_assets.py`, targeting
`obj_assets/PourWaterAssets.zip` in `DexForceAI/embodichain_data`, with MD5
`7267053763ed8e1b84f3da3e49d39f01`. The configured hosting prefix is tried first,
with the official Hugging Face Hub as a fallback while mirrors synchronize.
The [published archive](https://huggingface.co/datasets/DexForceAI/embodichain_data/blob/main/obj_assets/PourWaterAssets.zip)
was uploaded in revision `48317bd8733182f36d306cc2a7a60ae5e47ed2c9`. A fresh
cache downloaded that revision through the production resolver and Open3D
extractor, verified the complete archive SHA-256 and all 20 manifest files,
and confirmed that the three GLBs match the locally qualified models.
The default mirror-based registry path passed the same empty-cache download,
extraction and checksum checks.
The preparation script itself does not publish anything.

The Task Program uses projected effects, with measured bottle-position and
cup-position validators. Water transfer is not measured; this scene does not
include fluid simulation.

## Grasp and pouring clearance

The deployment selects the standard `cobotmagic` embodiment. Both grippers use
the calibrated 40 N coupled drive budget from `CobotMagicCfg`; this task does
not duplicate the robot/sensor component or override its force settings.
Gripper stiffness/damping stay at `2000`/`70`. Position targets now update
at 100 Hz, matching the physics cadence instead of holding each target for four
substeps. The policy uses 480 samples to retain a 4.8 s arm-motion duration;
closing, opening and settling counts scale with that cadence. This reduces
acceleration impulses while preserving the 14-dimensional position-action schema
and leaving robot mass, inertia and gravity unchanged.

Force references must be kept separate. The [manufacturer's Cobot Magic page](https://www.agilex.ai/page/690aef2d5e78cfa260412cb5?mi=2&rn=COBOT+MAGIC)
describes two product generations and a custom gripper. AgileX's
[PiPER quick-start manual, page 12](https://static.generation-robots.com/media/agilex-piper-user-manual.pdf#page=12)
specifies 40 N rated and 50 N maximum clamping force for the optional 70 mm
gripper. The [official simulation URDF](https://github.com/agilexrobotics/mobile_aloha_sim/blob/f799ae0192c0c2bbd502ec4d9bbec8c2633acd1a/aloha_new_description/urdf/aloha_new.urdf#L648-L654)
authors `effort="10"` for the prismatic gripper joints, as does the local V100
asset, whose total opening is 100 mm. These are different specifications;
neither a product force rating nor a firmware command value directly identifies
the correct coupled-drive cap for this model. The 40 N default is a simulation
calibration with reserve below the earlier 50 N trial cap. The 10 N configuration
passed 8/9 load/seed trials, with 13 mm slip in one 250 g case. A 20 N candidate
passed the batched trials but slipped in a camera-enabled single-environment
run; both modes must pass qualification. This is not a real-device rating.

The pouring axis follows the wrist's roll axis in the bottle frame. The
pre-pour bottle origin is 180 mm above the cup origin, with horizontal offset
`[0.02764513585, -0.05356490290]` m. This offset cancels the bottle-mouth motion
at the 60-degree tilt, placing the opening over the cup while keeping both
the bottle body and robot clear.

The program first picks and positions the bottle, then checks its actual pose
within 10 mm of the pre-pour target before executing the pour/return segment.
A missed grasp stops the episode at that checkpoint. Final validators require
the bottle to return and the cup to remain within 2 mm of its rest target.
The placement release target is 7 mm above the final rest target, leaving
clearance while the hand opens; the bottle then settles naturally onto the table.
These checkpoints supplement projected effects; they are not continuous grasp
verification during every call.

## Validation

The corrected deployment was qualified locally on an RTX 5090 using the Default
physics backend, seeds 0, 1 and 2, and bottle masses 10 g, 100 g and 250 g.
All nine combinations completed two segments and four calls in 2108 control
steps (21.08 s). Focused tests include asset conversion, packaged resources,
grasp/collision metrics, calibrated mouth alignment and component isolation:

```bash
python -m pytest -q tests/scripts/tools/test_prepare_pour_water_assets.py \
  tests/test_task_program_package_data.py \
  tests/benchmark/test_cobotmagic_drives.py \
  tests/gym/envs/task_program/test_configured_integration.py

python .agents/skills/add-task-program/scripts/inspect_deployment.py \
  embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic.yaml
```

The idle-arm regression is covered by
`test_pour_water_commands_preserve_idle_left_arm_targets` in
`tests/gym/envs/task_program/test_configured_integration.py`, using the production
config and the full 14-joint controller layout. Left-arm TCP displacement stayed
below 0.22 mm over all nine trajectories with the per-axis PD defaults.

Context impact review: scene tuning preserves component ownership and asset
resolution. The Task Program execution context documents retained active-row
joint targets and the separate measured safe-stop/inactive-row holds. Robot
context points to measured grasp/contact qualification and the distinction
between drive budgets and hardware grip-force ratings. Existing simulation, Gym, RL and workspace guidance
remains accurate; no ownership or backend contracts change.

## Drive and load verification

CobotMagic retains the asset URDF's `100 Nm` arm effort cap. Arm speed uses
the lower of the asset limit and the
[standard PiPER manual reference](https://static.generation-robots.com/media/agilex-piper-user-manual.pdf#page=11); this reference
does not establish the V100 variant's hardware rating. Per-axis force-drive PD
gains decrease toward the wrist:

| Joint | Stiffness (Nm/rad) | Damping (Nm s/rad) | Speed cap (rad/s) |
| --- | --- | --- | --- |
| J1 | 30000 | 600 | 3.141593 |
| J2 | 30000 | 600 | 3.403392 |
| J3 | 20000 | 400 | 3.141593 |
| J4 | 12000 | 240 | 3.926991 |
| J5 | 8000 | 160 | 3.926991 |
| J6 | 4000 | 80 | 3.000000 |

Both arms use these defaults. The coupled gripper retains its calibrated
`40 N` budget and `2000`/`70` gains, with speed capped at `0.25 m/s`.
PD gains and gripper speed are qualified simulation settings, not measured
device parameters. These defaults belong to `CobotMagicCfg`, so the deployment
does not need a separate robot component.

```bash
python -m scripts.benchmark.robotics.cobotmagic_drives \
  --output-dir outputs/benchmarks/cobotmagic_drives

python -m scripts.benchmark.robotics.cobotmagic_drives \
  --task-config embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic.yaml \
  --bottle-masses 0.01 0.1 0.25 --seeds 0 1 2 \
  --renderer hybrid \
  --output-dir outputs/benchmarks/pour_water_per_joint_pd
```

The joint protocol compares configured defaults with pinned legacy and
experimental gains, using gravity holds, six per-joint steps and sinusoidal
tracking at 100 Hz physics and 25 Hz commands. It writes raw traces, an inertia
audit and one report with three benchmark tables.

Task qualification pins Hybrid to match camera-enabled runs, rather than
letting a camera-disabled launch select a different renderer automatically.
The task protocol uses the deployment's 100 Hz control cadence, scales bottle
mass and inertia together, and samples every physics substep. Alongside
projected program acceptance it requires measured
lift of at least 80 mm, tilt of at least 45 degrees, return error below 50 mm,
final tilt below 10 degrees, idle-arm movement below 1 mm and actuator speeds
within 5% of their limits. From lift until release it also requires bottle pose
drift relative to the gripper below 3 mm / 3 degrees, conservative bottle/cup
clearance above 10 mm, zero cup contacts against the robot or bottle, cup motion
below 2 mm and zero dropped contact records. Collision geometry is enclosed by
a bottle capsule and cup sphere; their separation is a conservative lower bound.
`physical_checks.json` contains the independent checks; `resolved_drives.json`
records effective native properties. Speed checks read those native caps in
the final independent-actuator order, including configuration overrides, rather
than assuming the earlier `5`/`3`/`1` limits. The protocol does not measure transferred
water or actual actuator torque.

The raw traces and report retain per-trial drift, tilt, return error and
conservative bottle/cup clearance; a camera-enabled 250 g recording is checked
using the same criteria as the batched load/seed trials.
The speed-only change with uniform `70000`/`1000` arm gains passed all nine
trials. The final per-axis PD defaults also passed all nine trials: grasp drift
stayed below 1.65 mm / 1.29 degrees, conservative bottle/cup clearance stayed
above 26.05 mm, return error stayed below 7.32 mm, and cup contacts were zero.
Idle-arm displacement increased from 0.097 mm to 0.217 mm, within the 1 mm
qualification threshold. The two stages are recorded separately under
`outputs/benchmarks/pour_water_velocity_caps` and
`outputs/benchmarks/pour_water_per_joint_pd`.
An isolated-branch recheck after batching the bridge's retained targets passed
all nine combinations again, with 1.63 mm / 1.29 degrees maximum grasp drift,
26.05 mm minimum conservative clearance and zero cup contacts. Its evidence is
under `outputs/pr/pour_water/live_task`.

Independent gravity-hold, per-joint step and sinusoidal tests passed 9/9 with
both gain sets. With the final defaults, extended-arm TCP error was 0.522 mm
and sinusoidal worst-axis RMSE was 0.183 degrees (previously 0.315 mm and
0.130 degrees); both remain within the protocol's thresholds. The estimated
peak PD demand during sinusoidal tracking fell from 74.74 to 22.93 Nm; this
diagnostic is not measured actuator torque. Raw results are under
`outputs/benchmarks/cobotmagic_velocity_uniform_pd` and
`outputs/benchmarks/cobotmagic_per_joint_pd`.
The separate camera-enabled 250 g run also passed, with grasp drift below
0.42 mm / 0.35 degrees, conservative bottle/cup clearance above 26.06 mm,
zero cup contacts and 0.217 mm idle-arm displacement. Its full 21.12 s video
and measured checks are saved under `outputs/pour_water/per_joint_pd_video`.

Fault injection with a 1 N right-gripper budget stops with
`segment_validation_failed` before `simulation.pour`; the cup is untouched.
The force budget is calibrated for these simulated checks and is not a measured
real-device payload rating.
