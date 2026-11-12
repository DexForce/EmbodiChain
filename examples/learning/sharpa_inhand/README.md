# Sharpa online model-based APG

## Upstream baseline reproduction

Algorithm comparisons are paused until the physical task and policy information
match the pinned upstream baseline. The Newton/URDF workflow below is an
approximate experiment and its existing checkpoints are not equivalent-baseline
results.

First reproduce the original task at commit
`9816885a3b1fbdb947db71e72c27ed75effcbf9f`, using an independent checkout:

```bash
git clone https://github.com/Mingrui-Yu/sharpa_rl_unilab /path/to/sharpa-reference
git -C /path/to/sharpa-reference checkout 9816885a3b1fbdb947db71e72c27ed75effcbf9f
cd /path/to/sharpa-reference
uv sync --frozen
cd /path/to/EmbodiChain
python examples/learning/sharpa_inhand/reference.py \
  --upstream /path/to/sharpa-reference \
  --output outputs/sharpa_reference_teacher_seed1
```

The runner requires clean pinned sources and uses the upstream locked Python
environment. It saves source/asset hashes, installed packages, effective config,
test logs, original checkpoints and the original evaluator's result. A fresh
asset-cache directory prevents locally generated grasp caches from replacing
bundled data. `--verify-only` runs the checks without training.

The training path calls the original SuperMimic teacher implementation with
MJWarp, 2048 environments, seed 1 and 3000 iterations. Its only Hydra overrides
select the output directory, disable automatic playback and suppress console
progress. Physics, randomization, rewards, action processing, models and learning
parameters retain their upstream defaults. Evaluation uses the checkpoint-owned
task and the original 240-scene, 20-second validation protocol. This is a
reference reproduction; it does not validate the approximate implementation or
establish an APG comparison.

| Contract | Upstream baseline | Existing approximate APG path |
| --- | --- | --- |
| Physics/assets | Original MJCF and MJWarp | URDF imported into Newton, analytic cylinder |
| Control | Original gains and their randomization; actuator ranges scaled by 0.9 | Fixed PD gains and URDF limits |
| Policy inputs | Actor 147 with tactile history, privilege 9, critic 174; HORA teacher | Actor 109 with privileged physical state |
| Reset/task variation | Eight scales, matching bundled caches; mass, COM, friction, gains and gravity randomization; persistent forces | One scale and fixed physical parameters |
| Rewards | Six original terms with randomized PD torque and control-period finite differences | Six related terms with fixed PD; historical runs may add shaping |
| Episode boundary | Original height band, timeout and terminal observations; timeout bootstrap | Separate model and collector handling needs verification |
| Evaluation | Fixed 240-scene manifest, three seeds, 20 seconds, final checkpoint | Different scene/seeding and acceptance rules |

Before replacing the upstream learner with EmbodiChain APG, use the same
original task runtime and compare identical seeded initializations/actions:
joint targets, complete observation groups and history, reward components,
termination/timeout flags, terminal observations, reset randomization and the
evaluation manifest. The actor must use the same teacher or direct stage in
each comparison. Reusing existing 109-feature policies does not satisfy that
condition. After task parity, compare equal real-transition budgets and report
world-model and policy changes separately from algorithm changes.

The private `_aligned_task.py` boundary uses the original environment and state
capture, preserving all four observation groups and copying physical transitions
before selected-row resets. It records the original 89-value state and 55-value
randomization/task context. The private `_aligned_apg.py` experiment connects
this boundary to EmbodiChain's APG algorithm. Qualify it with the same locked
interpreter, from the EmbodiChain root:

```bash
/path/to/sharpa-reference/.venv/bin/python \
  examples/learning/sharpa_inhand/_verify_reference.py \
  --backend mujoco --output outputs/sharpa_parity_mujoco
/path/to/sharpa-reference/.venv/bin/python \
  examples/learning/sharpa_inhand/_verify_reference.py \
  --backend mjwarp --shared-native-step --output outputs/sharpa_parity_mjwarp
```

The CPU check compares seeded 8-scale, 401-step independent trajectories against
the original environment, including an original-versus-original repeat. It
asserts exact agreement of targets, observations/history, state/context,
per-term rewards, end flags, terminal observations and observations after
resets. It also checks analytic versus measured rewards and persists traces.
MJWarp's original environment is not bitwise deterministic across independent
GPU trajectories. Its check instead intercepts native step/reset receipts on
one trajectory and requires exact adapter input, pre-step state, output and
post-reset observation agreement. This verifies the interface without treating
numerical trajectory divergence as an adapter error or loosening tolerances.
Independent GPU repeats can be investigated by omitting `--shared-native-step`;
failures preserve their reports and traces. Run GPU qualification separately
from timed training when reporting performance.

## Controlled APG comparison

The aligned experiment uses the original HORA teacher and original SuperMimic
world model, replay, sampling, Muon/Adam optimizer, EMA and gradient clipping.
EmbodiChain APG optimizes the discounted 32-step return with the original soft
survival. It omits the learned terminal value bootstrap and critic fitting.
These differences are saved explicitly in `comparison_spec.json`; this is a
controlled actor-objective comparison, not a comparison of the old 109-feature
Newton/RWM experiment against the full upstream stack.

```bash
PYTHONPATH=/path/to/EmbodiChain \
SHARPA_RL_UNILAB_ASSET_CACHE=/path/to/EmbodiChain/outputs/sharpa_apg_assets \
/path/to/sharpa-reference/.venv/bin/python \
  examples/learning/sharpa_inhand/_aligned_apg.py \
  --reference-run outputs/sharpa_reference_teacher_seed1 \
  --parity-report outputs/sharpa_reference_teacher_seed1/parity_mjwarp/report.json \
  --output outputs/sharpa_aligned_apg_teacher_seed1
```

The runner requires a completed original reference, qualified GPU interface,
unchanged upstream source/asset hashes and matching task/model/protocol config.
Full runs also require identical replay and learning budgets. It verifies exact
initial actor/critic parameter equality, retains the original 3000-iteration
collection schedule and evaluates the final checkpoint using the reference's
scene manifest. APG checkpoints are explicitly labelled `embodichain_apg`.
Only their deserialization is adapted; the original evaluation RNG, task,
inference and scoring remain in use. `--smoke` reduces learner sizes/budgets for
a two-update wiring check and skips evaluation; it is not a comparable result.

Add `--value-bootstrap` to retain the original terminal value objective and
critic lambda-return fitting while routing actor gradients through EmbodiChain
APG. In eager mode (including `--smoke`), its first actor update checks un-clipped
gradients against an independent reduction of the pinned upstream objective.
Compiled runs retain the reference single-backward execution because AOTAutograd
donated buffers do not support this double-backward validation. The terminal value remains graph
connected through predicted states while critic parameters are frozen for the
actor update. The task's measured rewards remain unchanged. `--seed N` selects
an independent training replicate; the task and learning budgets still match.
The checkpoint records this mode in `comparison_spec.json` and stores the critic
optimizer alongside APG and world-model state.

For a shared GPU, `--num-envs 512` reduces physical parallelism while multiplying
warmup, replay depth per environment and collected steps per update by four.
It retains the original 131072 warmup transitions, 2048000 replay capacity,
3971072 total real transitions and 3000 optimizer iterations; world-model and
actor learning batches stay unchanged. Acquisition order and temporal
correlations differ, so record the parallelism when interpreting comparisons.
`--release-unused-cuda-cache` releases unused Torch allocator blocks before
Warp collection; its memory counters distinguish Torch reservation from global
free memory. This resource tradeoff does not change numerical precision or
per-environment task terms.

The private `_evaluate_rotation.py` wraps the original physical evaluator with
read-only angular-velocity traces. It checks survival, a full revolution and
continued net rotation in the final five seconds, with fixed criteria declared
before evaluation. Use `--manifest PATH` for the original scenes or
`--seeds 20001 20002 20003` to generate a fresh 240-scene manifest. It stores
`stability.json` and `physical_traces.pt`; imagined rewards and one video do not
qualify a learned policy.

One training seed is a preliminary comparison. MJWarp trajectories can diverge
even with identical seeds, and evaluation episodes are not independent training
replicates. Use multiple training seeds before claiming algorithm superiority.
The comparison does not claim that the old approximate model-only config has
been migrated to the original task or that it implements the full SuperMimic
teacher/student pipeline.

## Native EmbodiChain training

`native.py` collects and evaluates physics through EmbodiChain
`SimulationManager`, its Newton backend, Spawn robot/rigid-body facades and
`ContactSensor`. It never constructs `SharpaTeacherEnv`. It reuses the pinned
reference HORA teacher, world model, replay and fitting code, with actor updates
through EmbodiChain APG and terminal value bootstrap.

The initial native profile is **fixed scale 1.0 and fixed physical parameters**.
It is a native learning qualification, not full eight-scale/domain-randomized
baseline parity. A reference policy working in this scene is only a transfer
diagnostic; successful native training requires a fresh actor and independent
physical evaluation of its resulting checkpoint.

The reference MJCF is compiled only for asset translation into a native USD
articulation. The USD preserves body and mesh frames, inertias, hinge limits
and collision exclusions. Native configuration supplies the source per-joint
PD gains, armature and passive friction. Zero world gravity plus cylinder
weight applied at every 240 Hz physics step reproduces hand gravity
compensation. Control runs at20 Hz with integrated action targets and the
source 0.9 actuator-range scaling. All 10,000 scale-1 grasp rows are sampled.
In the first fixed-scale run, the default articulation setter projected cached
coordinates beyond hinge limits onto those limits. That run therefore does
not preserve raw upstream cached poses; the full source profile opts out of
this projection explicitly.
The older URDF approximation below keeps its original filtering behavior.

Native observations use three49D HORA frames,9 privileged channels and the
clean critic history. Tactile channels use native contact-force magnitudes;
the reference uses elastomer-geometry contact net-force sensors. Native contact impedance and dependency
versions also differ. The run's `native_identity.json` and saved config own the
precise scope. The eight-scale workflow below qualifies full randomization
separately from this initial fixed-scale calibration.

Run using a Python with native EmbodiChain/DexSim dependencies and the pinned
upstream learning dependencies. Put the pinned upstream `src/` on `PYTHONPATH`
for its policy/world-model imports:

```bash
python examples/learning/sharpa_inhand/native.py \
  --upstream /absolute/path/to/qualified/sharpa_rl_unilab \
  --reference-config /absolute/path/to/reference/training/config.yaml \
  --output outputs/sharpa_native/train_seed1 \
  --num-envs 512 --iterations 3000 --seed 1 --resource-profile compact
```

The compact profile reduces world-model width,
learning batches and replay capacity while keeping the32-step bootstrapped
objective. It is recorded explicitly and is unsuitable for an unchanged-budget
algorithm comparison. `--smoke --iterations 4 --num-envs 16` checks collection,
world-model fitting and an independent APG actor-gradient reduction.

For a fresh physical evaluation, supply `--checkpoint PATH` and a new output
directory, with `--num-envs 64 --eval-seeds 10001 10002 10003`. It measures400
physical control steps per episode, stores measured rotation traces and reports
drops, total angle, final-five-second angle and positive-speed fractions. A
pass requires at least90% stable scenes and at most5% drops; each stable scene
survives20 seconds, turns at least2π and continues rotating in the last5 seconds.
The aggregate positive-speed fraction must exceed80% overall and in that tail.
A fixed-scale pass does not qualify the full randomized upstream task.

## Native eight-scale workflow

`native_full.py` builds all eight fixed cylinder sizes in one native manager,
uses their corresponding raw grasp caches, and randomizes PD gains, friction,
mass, COM, gravity direction and persistent object forces. Geometry and
constant-mass inertias are shrunk from the largest template; enclosing broad-
phase bounds stay conservative. Render geometry retains that template, so
videos must use the measured-state renderer with matching scale models.
Geometry-pair sensing includes only elastomer/object contacts. Reset batches
write gains once and explicitly preserve cached joint-limit excursions.

This remains a native task variant: hand self-contacts use `condim=6`, metal
base torsional/rolling friction `0.1/0.02` and impedance `0.95/0.99`; upstream
hand geoms use `condim=3`, metal friction `0.005/0.0001` and impedance
`0.9/0.95`. Hand-object contacts use six dimensions in both because the object
has `condim=6`. Native summed joint drive torques are capped at1000Nm;
the source has no actuator or summed-joint force limit. Joint limit ranges and
their soft-constraint parameters match the source. Dependency versions and
learner resource sizes also differ.
Do not interpret native learning success as strict upstream physics parity or
an original-budget algorithm comparison.

```bash
python examples/learning/sharpa_inhand/native_full.py \
  --upstream /absolute/path/to/qualified/sharpa_rl_unilab \
  --reference-config /absolute/path/to/reference/training/config.yaml \
  --output outputs/sharpa_native/full_seed1 \
  --num-envs 256 --iterations 3000 --seed 1 --resource-profile reference_host
```

The `reference_eager` profile retains the source 2048-wide world model,
2048-sample world-model batches,1024-sample policy batches and2048000 total replay
transitions, with eager execution and compensated physical parallelism. It waits
for6GiB free CUDA memory at training boundaries; `--min-free-gpu-gib` adjusts
that shared-device headroom. This wait does not reserve memory against other
processes. The default `reference_host` keeps those same dimensions, batches
and total capacity while storing replay on CPU and recomputing the world-model
MLP activations during backward. Samples retain source CPU-sampler window masks
and history clamping; sampling uses CPU RNG rather than CUDA RNG. Transfers are
synchronous, preserve field dtypes, and return owned device batches. Checkpoint
keys and optimizer parameter identity stay unchanged. This mode defaults to
4GiB free-memory headroom, checked before each control period and training
boundary. It trades host RAM, transfers and recomputation for
lower peak VRAM; neither guard reserves memory against other processes.
The shared profile uses a 1024-wide world model, 256-sample world-model batches,
128-sample policy batches and 262144 replay transitions, with the reference
32-step bootstrapped objective. It saves every100 iterations. `--smoke
--iterations 4 --num-envs 16` exercises collection, fitting and actor-gradient
verification with a small model. Memory on a shared GPU still depends on
concurrent jobs.

Host-mode checkpoints also save `native_replay_latest.pt` atomically, paired
with the model by SHA256. Resume into a fresh output directory with the saved
environment count and resource profile:

```bash
python examples/learning/sharpa_inhand/native_full.py \
  --upstream /absolute/path/to/qualified/sharpa_rl_unilab \
  --reference-config /absolute/path/to/reference/training/config.yaml \
  --resume-native RUN/native_teacher_03000.pt \
  --resume-replay RUN/native_replay_latest.pt \
  --output outputs/sharpa_native/continued \
  --num-envs 256 --iterations 5000 --resource-profile reference_host
```

The target iteration count is cumulative. Resume uses the saved learning
configuration and seed; omit `--smoke` even when recovering a smoke checkpoint.
Without `--resume-replay`, models and optimizers recover but replay is refilled.
Restored replay retains its history and masks; the final old sampler row closes
the interrupted sequence before new simulator-reset transitions are appended.
Physical state and its local RNG restart, so this is not an exact trajectory
resume. Changed physical source or adapter hashes are rejected.

Independently evaluate a checkpoint from this entry point using
`native_full_evaluate.py --checkpoint PATH --upstream UPSTREAM --output NEW_PATH`.
The evaluator checks physical source hashes and native adapter snapshots before
constructing a fresh scene. The same rotation/drop criteria apply to the
aggregate and each scale. It waits for 4GiB free CUDA memory before each control
period; `--min-free-gpu-gib` adjusts this non-reserving guard.
`--allow-reference-diagnostic` permits a source-policy
transfer check; `--allow-implementation-transfer` permits a native-policy port
check after refactoring. Neither diagnostic qualifies unchanged native training
and evaluation physics. Render an accepted run with
`native_multiscale_video.py --evaluation EVALUATION_PATH --upstream UPSTREAM
--output VIDEO_PATH`; it selects the first seed and first row of every scale,
independently of outcomes, and never integrates a trajectory.

Training snapshots own reproducibility. Earlier experiments used frozen
prototype source copies; their evaluation receipts and scope corrections remain
in their output directories. Do not evaluate an old snapshot against changed
current files and silently treat its physical-source hashes as unchanged.
The contact-rich zero-action rollout also diverges between repeated instances
of the frozen prototype, despite identical physical state, random-generator
state and parameters. Refactor checks establish those initial contracts; they
do not establish deterministic trajectory equality. Qualify each implementation
with independent physical policy evaluations.

Each evaluation reset seeds Torch and NumPy observation-noise streams as well
as the task-owned physical generator. Use disjoint seeds for validation and
holdout; different physical scenes alone do not make reused observation-noise
streams independent.

### Measured native qualification

The native seed-2 training lineage reached its rotation acceptance gates at
iteration4500. Its actor started from random initialization, independently
checked against the seeded source model factory; no trained reference policy
was loaded. Training and evaluation used native `SimulationManager` physics,
with EmbodiChain APG actor updates, the source HORA teacher and learned world
model. Initial states came from the bundled grasp caches: this qualifies
in-hand manipulation from cached grasps, rather than learning to grasp an
object from a table. The teacher uses privileged physical information; student
distillation and real-hardware deployment are separate work.

An unseen test of1536 episodes (192 per scale,20 seconds each) obtained98.57%
stable rotation and1.43% drops, with mean net world-Z rotation12.24rad. All eight
sizes passed separately, including the unchanged5% maximum drop rate. A fresh
1536-episode evaluation through the consolidated `native_full_evaluate.py`
implementation obtained98.31% stable rotation and1.69% drops, again passing
every size. This latter result is explicitly an implementation-transfer
qualification, not a claim of identical deterministic physics trajectories.
Stored poses, termination masks, rewards and raw-cache membership were checked
independently of the task's metric/reward helpers.

The accepted model is
`outputs/sharpa_native/full_seed2_host_replay_extend5000/native_teacher_04500.pt`.
The full result receipt is
`outputs/sharpa_native/native_eight_scale_learning_result.json`; the20-second,
eight-panel measured-state video is
`outputs/sharpa_native/native_learned_eight_scales_04500.mp4`.
Its fixed first episode per size is selected independently of outcome.

Earlier full validation at2900 and the first unseen test at3900 failed; those
negative results remain recorded. The3900 test then became known validation,
and fresh test/implementation-transfer seeds were declared before use. The
successful lineage required extra optimization beyond the source3000-update
budget and recovery from two allocation failures, including replay rebuilds.
Training continued normally through5000, while4500 remains the accepted model.
These are results from one training seed in the native physics variant above,
not a controlled comparison of algorithm quality or sample efficiency.

## Approximate Newton experiment

The workflow alternates real Newton/MuJoCo-Warp transitions, multi-step RWM
fitting and analytic policy-gradient updates. The hand/cylinder are physical
simulation entities; gradients flow through the fitted model. An independent
simulation instance evaluates the policy from fixed seeded grasp selections.

Use a simulation-capable EmbodiChain installation with the official task/model
extensions. SharpaWave assets are provided by the separate asset integration.
Until that asset is published, pass the prepared right-hand URDF explicitly.

```bash
embodichain run-example learning/sharpa_inhand -- \
  --hand-urdf /absolute/path/to/right_sharpa_wave.urdf \
  --output outputs/sharpa_inhand
```

The output directory must be new. Omit `--hand-urdf` after the registered
`SharpaWave` asset is available. The pinned scale-1 grasp cache downloads on
first use; `--grasp-cache` may select an already downloaded verified copy.

For a bounded wiring check:

```bash
python examples/learning/sharpa_inhand \
  --hand-urdf /absolute/path/to/right_sharpa_wave.urdf \
  --output outputs/sharpa_inhand_smoke \
  --num-envs 2 --imagination-envs 4 --warmup-steps 16 \
  --collection-steps 8 --world-updates 3 --policy-updates 1 \
  --horizon 4 --world-horizon 2 --batch-size 8 \
  --start-bank-size 32 --iterations 1 --eval-steps 20 --world-hidden 32 32
```

The short check uses a one-second evaluation window and does not establish
task learning. The default physical evaluation uses 20 seconds.

Outputs include `metrics.jsonl`, `report.json`, a standard APG checkpoint,
`world_model.pt` for the common train-rl entry point and `policy.pt`.
`initial_evaluation.pt` and `final_evaluation.pt` contain canonical physical
states, incremental policy actions and pre-step survival masks in `[time, env]`
order. They are evaluation records, not model-generated demonstrations.
Reports separate real transitions, held-out real transitions, imagined policy
updates and measured evaluation. World-model normalization sees only new
training transitions; held-out states do not update statistics. Policy-model
weights remain frozen during each complete rollout and backward pass.
Input/output normalization is fixed after cold-start acquisition, or retained
from a warm-start checkpoint. Later fitting uses new real transitions without
moving the learned predictor merely by changing its normalization coordinates.
The fixed palm breaks the canonical RWM's yaw invariance: model context includes
current predicted object XYZ and its full rotation matrix. Version-1 models
that supplied only XYZ are incompatible with the version-2 model identity.
`--servo-context` additionally supplies executed target minus current joint
position, exposing small servo commands without subtracting large absolute
angles inside the MLP. This uses a distinct model identity and export metadata.
The actor observes 109 features: fixed-palm object pose/velocity, hand state,
current targets and original object/hand reset references. Reset-relative
object position alone would hide different physical contact configurations.

The reward uses the pinned source task's fixed default object position and
22-joint hand pose. Sampled reset references separately own the drop-height
band and optional imagined height penalty. Reward velocities are differences
over each completed control period, including world-frame SO(3) rotation
increments. Evaluation integrates these measured orientation increments and
uses the same rates for positive-rotation fractions. Instantaneous simulator
angular velocities alone can differ appreciably from actual angle changes.
Exports record the reward contract separately from physical geometry identity.

For the native analytic cylinder, use
`--world-frame --axial-symmetry --servo-context` to predict world increments
and encode only cylinder position and world axis, plus the servo displacement.
Rotation about the cylinder's own axis does not change its geometry or inertia.
This representation enforces that physical symmetry and keeps full orientation
in the returned state for policy observations and rotation rewards. A distinct
checkpoint identity separates it from body-frame and full-pose world models.
Task fitting weights root state errors and uses2 mm position/1 mm height scales;
world-frame models also compute supervised residuals in world axes.

`--joint-limit-aware-policy` uses a smooth feasible-target parameterization
inside the actor while preserving the simulation's original action/rate and
hard joint limits. It avoids flat outward-command regions at an already
saturated target; its gradients are derivatives of the actual emitted action.
The online policy export records this actor type for saved-policy evaluation
and warm starts. The standalone model-training YAML continues to use raw deltas.

`--axis-policy` additionally removes the cylinder's own-axis phase from the
actor's rotation features while preserving its world axis and all other inputs.
It enables the smooth joint-limit actor and records both modes in the export.
This symmetry is appropriate for the native analytic cylinder used here.

To evaluate an exported online policy without training, use a new output path:

```bash
embodichain run-example learning/sharpa_inhand -- \
  --hand-urdf /absolute/path/to/right_sharpa_wave.urdf \
  --evaluate-policy outputs/sharpa_inhand/policy.pt \
  --output outputs/sharpa_inhand_evaluation
```

This checks the saved physical identity and writes `report.json` and
`evaluation.pt`. Use the same `--num-envs`, `--seed` and `--eval-steps` as the
training run to repeat its final evaluation. Training can continue through the
common `train-rl` entry point using the exported world model; that entry point
optimizes the frozen model and does not collect additional physical data.

For longer online experiments, `--world-pretrain-updates` fits the initial
measured corpus before APG starts. Exploration uses temporally correlated
noise; `--grasp-exploration` additionally applies early flexion perturbations
to half the collection scenes. These perturbations are absent in evaluation.

Each physical evaluation creates `snapshots/iteration_XXXXX/` with world-model,
policy, learner and APG checkpoints plus its measured `evaluation.pt` trace.
`best/` retains the best measured survival
fraction, with rotation breaking ties. To initialize a new run from exported
weights, use `--initialize-from RUN/best`; add `--reset-policy` to retain only
the model and create a fresh actor. This refills physical replay and creates
new optimizers, so it is a warm start rather than exact experiment recovery.

Optional imagined-objective shaping is explicit: `--height-penalty W` adds
`-W * (height_excursion / 0.02)^2` before control-period scaling, supplying a
restoring gradient near the drop boundary. `--rotation-warmup-iterations K`
ramps the imagined rotation coefficient from zero toward 2.5. These options
change the policy training objective; physical evaluation retains the reference
reward weights and the same 20 mm failure threshold.

`--randomize-training-yaw` broadens the training cylinder's initial phase using
32 discrete phases of its native axial symmetry. Composition is about the cylinder's
own axis, preserving that axis and the collision shape. Every training state is
still acquired from the simulator. Held-out collection and policy evaluation
retain the original cache orientations; initial phase changes never count as
rotation. Reset distributions are recorded separately and may differ when
loading a policy, while physical geometry/control and actor-feature contracts
must match.

Evaluation reports include per-scene rotation/survival, positive-speed fraction
and five-second rotation windows. `stable_scene_fraction` requires no drop,
at least one positive revolution and speed above 0.1 rad/s for at least 80% of
the configured window and its final five-second window. Use the full 20-second
window and fresh seeds for outcome
claims; a one-second smoke window cannot establish stable rotation.

Replay is a fixed-capacity device ring with an explicit byte ceiling. The
default 8-environment, 1024-step training ring is about 6.5 MB; the model rollout
graph is additionally bounded by imagination batch and horizon. No comparative
throughput or hardware-generalization claim is implied.

## Resources and limits

- Hand: official Sharpa Wave 01 URDF/mesh bundle from
  [sharpa-robotics/sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml),
  through the separate `SharpaWave` asset integration.
- Grasp cache: `sharpa_grasp_linspace_1.npy` from
  [Mingrui-Yu/sharpa_rl_unilab](https://github.com/Mingrui-Yu/sharpa_rl_unilab/tree/9816885a3b1fbdb947db71e72c27ed75effcbf9f),
  SHA256 `22a2dac7f6ecdf73764d06c74f3810b21ec9af43c609d283d61b788b8383c842`.
  The upstream package attributes caches to `unilabsim/unilab-caches`;
  downloading does not relicense that data.
- Cache poses use XYZ+WXYZ; conversion to EmbodiChain XYZ+XYZW occurs once at
  reset. Joint mapping uses names, never native traversal order.
- Rows outside the selected URDF's limits are rejected before sampling.
  For the prepared official revision, 9,232 of 10,000 rows satisfy those limits;
  the physical identity records accepted/rejected counts and source hashes.
- This is a single-scale, privileged-state baseline. Reference joint armature
  and passive friction are overlaid by name; contact dimension is 6 with
  sliding/torsional/rolling friction 2.0/0.10/0.02. Contact stiffness/damping
  approximate the source time constants. Native URDF contact geometry, hand
  gravity, fixed PD settings and disabled hand self-collision
  differ from the reference MuJoCo task. Tactile observations, HORA distillation
  and the 240-scene multi-scale reference protocol are not implemented here.
- The object is a native analytic cylinder, authored through the existing USD
  source route, with radius20 mm, height32 mm and mass50 g. The native-cylinder
  physical profile is version3; older version2 runs used a32-section convex
  mesh. Those older experiments also used sampled reward references and
  instantaneous velocities and are retained as historical diagnostics.
  Version3 does not silently load their mismatched physical checkpoints.
- Control targets respect the selected official URDF limits. The source's
  additional0.9 actuator-range scaling is not applied to those URDF limits.
- Model prediction error, nonzero actor updates and positive physical rotation
  are separate evidence requirements. A passing smoke run proves the pipeline,
  not a trained manipulation policy.
