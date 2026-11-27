# Physical grasp validation

From the repository root, with the simulation dependencies and CUDA available:

```bash
d=embodichain/extensions/benchmarks/grasp_pose_generator
embodichain benchmark grasp-pose-generator validate --case "$d/block.yaml" --poses "$d/candidates.jsonl" --num-envs 3 --output outputs/benchmarks/grasp-validation
python -m embodichain benchmark grasp-pose-generator run --suite "$d/suite.yaml" --num-envs 10 --output outputs/benchmarks/grasp-desk --resume
```

To watch the suite execute grasps, use `run --no-headless`. The terminal
shows scene preparation and candidate generation. The window opens when
the first candidate is in its pre-grasp pose. A case with zero candidates
prints a message and skips playback. The camera starts on arena 0 and the
terminal shows the object, gripper, candidate rank and result. Each candidate
gets one trial from approach through hold. Multiple arenas run different
candidate ranks in parallel. Use `--num-envs 1` to watch every candidate
in sequence.

```bash
python -m embodichain benchmark grasp-pose-generator run --suite "$d/suite.yaml" --no-headless --num-envs 3 --output outputs/benchmarks/grasp-desk-view
```

For a quick visual check of all seven hands, use the single-arena preset.

```bash
python -m embodichain benchmark grasp-pose-generator run --suite "$d/visual_check.yaml" --output outputs/benchmarks/grasp-visual-check
```

The preset references `suite.yaml` for hand calibration, object physics and
placement. Its `objects` mapping selects one model ID per category. Each of
the 35 hand/category pairs uses the first configured seed, method and placement.
The antipodal generator samples 2,000 rays per pair; `sample_count` controls
this budget. The native window shows the top-ranked candidate's lift,
optional disturbance and three-second hold. A pair with zero candidates skips playback and
counts as a failed check with the reason `no candidate`.
The terminal shows progress, and `report.md` lists each pair's outcome and
failure stage. Use `--headless` to run the same checks without a window.

### Run different hands and objects together

`run --grid COUNT` places up to `COUNT` cases in one PhysX scene. Each
side of the square grid has `ceil(sqrt(COUNT))` cells. For example,
`--grid 35` fills 35 cells of a 6 × 6 grid in row-major order and leaves
the last cell empty. The runner mixes hands and objects across the grid, then
loads the next group until it has covered the suite. Every active hand sends
its command before the scene advances one physics step. Each cell has its own
PhysX collision group, contact sensor and 4 m × 4 m support centered on its
object workspace. The window draws a single tabletop for the grid.
`--grid-spacing` sets the distance between neighboring centers in metres
(default 1.0). The camera frames the entire grid.

```bash
d=embodichain/extensions/benchmarks/grasp_pose_generator
python -m embodichain benchmark grasp-pose-generator run --suite "$d/visual_check.yaml" --grid 35 --close-wait-steps 260 --no-headless --output outputs/benchmarks/grasp-grid-view
python -m embodichain benchmark grasp-pose-generator run --suite "$d/suite.yaml" --grid 16 --close-wait-steps 260 --headless --resume --output outputs/benchmarks/grasp-grid
```

The examples hold the final closing command for 2.6 simulated seconds before
lift. Omit `--close-wait-steps` to use the YAML value.
The override also works for the ordinary sequential runner.

The grid uses the default PhysX backend and one native arena. Candidate
generation runs per pair. Candidate rank, grasp outcome and checkpoint are
recorded separately for each pair. Between candidate ranks, the grid parks
the hands and lets the targets settle again at the configured placement.
Reuse the same suite, grid count, spacing and output directory when resuming.
Grid runs report wall-clock time for the
whole run because the pairs share their physics steps. Contact-sensitive grasps
can change outcome between layouts even with the same object-local candidate.
For comparisons across revisions, keep the physics backend, layout and dwell fixed.

To watch a completed case again, omit `--resume` and use a fresh output
directory. Use the default headless mode when comparing execution times.

Both commands accept `--physics default` (the default GPU backend) and
`--physics newton` (Newton with the DexUni solver). Each invocation evaluates
one backend. Use distinct output directories to keep both aggregate reports.
Checkpoint identities include the backend and solver, so a resumed run
reuses results only from the matching physics configuration. Default uses a
stationary kinematic table for GPU contact actor identification. DexUni uses
two-way coupling. Both paths use positive impulses for measured contact rows.
For geometry-only DexUni rows, a finite signed separation at or below zero
qualifies contact. Reports record the contact policy alongside the solver.

Use the same YAML/JSONL inputs with the installed CLI. Each JSON line contains a
4×4 rigid transform `T_object_tcp` in metres, ordered best first. A case selects
a native parallel-jaw gripper by catalog ID and a target through an ordinary
dynamic RigidObjectCfg. The flat support surface is at arena Z=0. The stand
sets the requested wrist orientation at reset and executes the fixed-orientation
trajectory with three force-limited prismatic drives and native finger drives.
The single-case example uses two lateral misses and one centered grasp on
the 12 cm tall block. All three TCPs sit 3.5 cm above the block center to
allow the fingers to close with the palm clear of its top.

The suite crosses object instances, grippers, placements and seeds. The
bundled suite discovers the unpacked SimReady models for `book`, `bottle`,
`bowl`, `cup` and `spoon` under `dataset_root`. Each object has one root
USD layer (`.usd`, `.usda` or `.usdc`) in `<category>/<object_id>/` with
its supporting files beside it. The loader also accepts the extracted
DexTrove layout at `<category>/<source_id>/<model>/simready_usd/`.
To prepare a separate dataset root from a DexTrove instance or replay a saved
object shortlist, see the [DexTrove subset tool](../../../../scripts/tools/README.md).
Models with backend variants use the physics engine selected on the CLI.
The shared `rigid.attrs.mass_props.mass` setting in `suite.yaml` defaults to
0.035 kg. Adjust it for all categories or give one category its own rigid
settings. Inertia is recomputed for the configured mass.
Each category's `rotation_shift_deg` gives the initial object rotation about
the fixed arena X, Y and Z axes in degrees, applied in that order to the
source orientation. The book entry uses `[0, 90, 0]` to stand books on edge.
A nonzero rotation preserves the object's XY origin and places its lowest
collision point on the table before settling. Zero or omission uses
the configured source placement. A single-case YAML accepts the same field.
Custom suites can also supply explicit `objects` with `id`, `category`
and a dynamic `rigid` config. Optional named `placements` override top-level
rigid config fields.

`grippers` accepts a list of catalog IDs or a mapping from each ID to its
settings. The bundled suite evaluates the Franka FR3 hand, DH PGI-140-80,
Robotiq 2F-85 and 2F-140, CobotMagic V100, Pika, and the left Tianji Marvin
hand. Each entry specifies its TCP, drive stiffness and maximum drive effort.
Omitted settings use the corresponding catalog defaults.

Set `gripper_root` to the directory containing
`<catalog_id>/<catalog_id>.urdf` and each hand's relative mesh and material
files. Both candidate generation and simulation load this standalone hand.
The URDF must match the configured catalog ID and variant. Omitting
`gripper_root` uses the catalog asset. The bundled suite and its visual
preset use `GraspBenchmark/objects_subset` and `GraspBenchmark/grippers`.
EmbodiChain resolves these registered paths in its data cache and downloads
the archive on first use. Both roots also accept absolute paths or existing
directories relative to the suite YAML. For single-case validation, set
`gripper_urdf` to the standalone URDF path in the case YAML.

- `variant` selects a catalog model for both candidate generation and physical
  execution. For `TianjiMarvinGripper`, specify `left` or `right`. Single-case
  validation accepts the same field alongside `gripper`.
- `root_from_tcp` is the 4×4 tool pose in the native hand root, with translation
  in metres. It defines the TCP used in saved candidates and external
  `T_object_tcp` inputs. The built-in generator uses the hand's catalog contact
  frame and expresses the resulting grasps in the selected TCP.
- `stiffness` is the nonnegative scalar position gain for the hand drives,
  in N/m for prismatic joints or N·m/rad for revolute joints. On PhysX the
  Robotiq 2F-85 drives `finger_joint` and its five followers track it through
  mimic constraints.
- `max_effort` sets the drive force limit (N for prismatic joints) or torque
  limit (N·m for revolute joints) on each active hand joint. Fingertip
  contact forces also depend on the linkage and contact location. The bundled
  suite sets `max_effort` to 30 for each hand. The translation stage uses
  its own drive limits. Changing either hand setting creates a distinct
  trial checkpoint. Single-case validation accepts both fields.
- `open_qpos` sets the joint positions used to sample the open-hand collision
  geometry and to reset and approach in simulation. `close_qpos` sets the
  targets for the closing motion. Each map names every moving hand joint,
  including mimic followers, in metres for prismatic joints or radians for
  revolute joints. Omitting either map uses its catalog command. The bundled
  suite lists both maps for every hand, including the DH PGI-140-80's
  0.040 m inward travel per finger.

On PhysX (`--physics default`), both `run` and `validate` use hard coupling
for every gripper's URDF mimic joints. They set natural frequency and damping
ratio to zero and preserve each follower's multiplier and offset. Newton uses
its own mimic constraints.

`algorithms` maps method names to ranked JSONL filename templates, supporting
`{object}`, `{category}`, `{gripper}`, `{placement}` and `{seed}`. A null value runs the
built-in antipodal generator on the object's semantic mesh. Use an empty file
to record an algorithm's abstention. Missing files are input errors.

The suite-wide `collision_threshold` sets the open-gripper clearance in metres
for built-in antipodal candidates. The bundled suite and its visual preset use
0.005 m. A candidate is rejected if any hand surface sample has a signed
distance of 5 mm or less to the object's convex hulls, or lies less than 5 mm
above the inferred ground plane. Omitting the setting uses zero. Changing it
selects a new checkpoint for generated candidates. External JSONL poses go
directly to physical validation.

After one simulated second, the validator checks the target pose every 0.1 s.
It takes a snapshot once every arena stays within 1 mm and 0.005 rad of its
starting pose for a 0.5 s window. A rocking target can settle for up to 5 s.
Each scene checkpoint stores the frozen candidates and measured starting pose.
Each candidate starts from the same settled target pose with cleared velocities.
The gripper approaches 8 cm in 60 physics steps and ramps the fingers from
open to closed over the next 40 steps. The suite-wide `close_wait_steps` then
holds the closed command at the grasp location before lifting. Each step is
0.01 s. The bundled suite sets 60 extra steps, giving the fingers 1.00 s from
the start of closing until lift. Set it to 0 for an immediate lift after the
ramp. Files that omit the key use 0. Single-case validation accepts the same
key in the case YAML. The lift travels 10 cm in 80 steps.

`enable_disturbance` controls the motion immediately after lift. It defaults
to `true` in suite and single-case YAML. The visual preset inherits the suite
value. With the fingers closed at lift height, the gripper moves 3 cm along
world +X, returns to the lift position, moves 3 cm along world +Y and returns
again. Each minimum-jerk leg takes 0.30 s, for 1.20 s of motion in one grasp.
The gripper then holds still for 3 s. Setting `enable_disturbance: false`
starts the same hold immediately after lift.

The protocol and the disturbance setting are part of the checkpoint identity.
`--resume` reuses trials from the matching protocol. Changing the wait or
target mass also selects a new checkpoint.
Labels use collision clearance, support contact, continuous retention and
terminal bilateral pad contact. Relative COM motion and orientation are
diagnostics. A window records whether the target entered airborne, and whether its drop time was censored at the observation horizon.
Missing contact evidence is retried from checkpoints. Native worker failure
does not remove its scene from the intended population.

The Markdown and JSON reports show Success@1, Hit@5, Hit@10 and Precision@10
by gripper, by object category and by gripper/category pair. Success requires
the candidate to pass the lift, the disturbance when enabled, and the final hold.
Each object instance and gripper pair has equal weight after averaging its
seeds and placements. Missing candidate slots count as failures for
Precision@10. Missing observations leave the affected group's score unavailable.
Conditional disturbance retention uses the trials that entered the combined
X/Y motion. The JSON includes case inventory, stage diagnostics, bootstrap
intervals over object instances and paired method differences.

Each Parquet trial in `trials/v6/` records one candidate rank, its gripper,
object instance, category and disturbance setting. Use a
fresh output directory when switching datasets or replacing referenced USD
layers. Checkpoints retain user-supplied candidate metadata such as score
and opening.
