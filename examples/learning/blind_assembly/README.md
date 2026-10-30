# Native proprioceptive assembly

The three registered tasks use native Gym managers, the temporal actor-critic,
the native collector/rollout/PPO, and checkpoint evaluation. Contact labels and
privileged state are separate training inputs; neither enters the actor.

The peg study nominal configuration and its development variant explicitly
select `architecture_version: official_v1`, using the author's now-public
[policy layout](https://younghyopark.me/blind-proprioceptive-assembly/static/policies/peg.json)
and [inference reference](https://younghyopark.me/blind-proprioceptive-assembly/static/js/policy_net.js).
Their input is the current frame followed by 30 history frames including the
current frame. Per-frame projection precedes the temporal convolutions, and
the actor applies the exported variance/epsilon/clipping normalization formula.
The environment's `history_layout: current_history` and
`proprioception_convention: official_v1` must match that policy selection.

Configurations without a version retain `legacy_v1`: the original migration's
encoder with 30 previous frames followed by the current frame. Other formal,
full-range and gear/nut configurations currently retain that legacy version;
the public reference alignment does not qualify those deployments. Existing
legacy checkpoint module layouts remain compatible. The native population-moment
statistics update and privileged critic remain engineering choices because the
public actor export does not specify their training implementation.

Install the optional assembly generator and the SDK capabilities described in
`scripts/patches/README.md` in a source checkout. Wheels include the patch files
under the installation prefix's `share/embodichain/assembly_patches/`.
The Menagerie reference sources and both licenses are pinned by Git blob hashes.

The default embodiment selects the bare `FlexivRizon4S` catalog robot, attaches
`Robotiq2F85/Robotiq2F85.urdf` through `urdf_cfg.components`, and owns the mount,
TCP, hand drives and contact overrides in
`configs/components/embodiments/flexiv_rizon4s_robotiq2f85.yaml`.
There is no combined arm/gripper robot preset. Grasp reset geometry and model
qualification requirements belong to each task's physical environment config.
The assembly components explicitly select `PytorchSolver` for the recorded
profile; standalone bare-arm configurations retain the catalog's SSIK default.
The canonical study/fullrange environments require identified dynamics and
therefore reject the public engineering model; development environments remain
explicitly unqualified.

Historical `.nominal` deployments select a prepared parallel-jaw reference
artifact. Prepare it once before loading those configs:

```bash
python -m embodichain_benchmarks.assembly.robot_reference --gripper-model parallel_jaw
```

This optional preparation is separate from default-asset assembly and training.
Old model bundles that name `FlexivRizon4SRobotiq` require configuration migration
to `FlexivRizon4S` and their frozen local `fpath`; use the bundle's recorded code
revision for an unchanged historical replay. The measured results below predate
this ownership refactor and do not certify its native dynamics.
The current default-asset [model descriptor](validation/default_retrain_20261026/model_descriptor.json)
points to a migrated local bundle. Its checkpoint and physical resources are
unchanged; the included eight-episode inference smoke completes six successes
with finite sampled state and no capacity loss. Iteration-budget and SDK teardown
warnings remain recorded. This smoke does not replace the 8192-episode assessment.

For a training attempt from random initialization, the peg study development
configuration uses 4,096 environments, 32-step rollouts and at most 2,000 PPO
updates (262,144,000 transitions). It retains 1 kHz physics, a 50 Hz policy, sparse capture/success rewards,
and training-only contact supervision. This is a fixed-geometry nominal task,
with separate evaluation episodes; it does not establish the paper's geometry,
workspace or contact randomization coverage.

```bash
python scripts/patches/apply_mujoco_warp_epa_horizon.py --output /tmp/embodichain-mjw-epa128
PYTHONPATH=/tmp/embodichain-mjw-epa128:$PYTHONPATH python -m scripts.benchmark.assembly.train_peg \
  --config extensions/embodichain_tasks/configs/tasks/manipulation/assembly/peg_insertion/agents/ppo.study.nominal.dev.yaml \
  --output outputs/peg_training_health.json --raw-log outputs/peg_training.log > outputs/peg_training.log 2>&1
```

Retain the SDK overlay or qualified SDK installation required by the ordinary
assembly examples. The wrapper calls the native training entry point and adds
read-only solver diagnostics through `on_trainer_ready`; it does not implement
another trainer. Both train and validation check native capacity flags before
auto-reset. The EPA allocation overlay is version/hash guarded and leaves the
shared installation unchanged. Uninstrumented runs still use `embodichain
train-rl`, but contact-query overflow checks alone cannot qualify native
collision-buffer health.

Fresh processes using the current strict observer also check complete native
`qpos`, `qvel` and `qacc` arrays before auto-reset. Reports identify the sampled
scope and first nonfinite worlds; missing native fields fail strict assessment.
The V4 training process loaded the earlier observer before these checks were
added. Its capacity guards and finite PPO loss logs do not establish raw native
state finiteness. Existing V4 training snapshots retain that limitation; use a
fresh, independently instrumented held-out evaluation to establish this check.

The profile explicitly selects an elliptic friction cone and `impratio: 10`,
and a plate-attached 2 mm support floor through mjfit's public `bottom_thickness`,
matching those features of the public author's model export. The raw mjfit
generator and passive examples retain their independent defaults. The nominal robot, contact material,
100/50 solver iteration budgets and 50 Hz encoder gravity feedforward remain
distinct from that exported model. These settings do not establish full
physical equivalence.

The [reference record](../../sim/assembly/validation/rl_attempt/policy_reference.json)
reports fixed-input numerical alignment only. Public author weights are used
for that independent comparison and do not initialize this training attempt.
Learning outcomes require completed training and checkpoint evaluation.

Reproduce the CPU comparison with Node.js and the ordinary RL dependencies:

```bash
python -m scripts.benchmark.assembly.policy_reference --cache-dir outputs/policy_reference_cache --output outputs/policy_reference
```

The runner verifies every downloaded and cached source file against its pinned
SHA-256 and reconstructs eight fixed inputs from the small checked-in frame
fixture. It writes JSON and a Markdown report; it does not run a simulator or
change the installed MuJoCo version.

```bash
embodichain run-example learning/blind_assembly/run_suite --phase prepare --suite-dir outputs/assembly_suite --nominal
embodichain run-example learning/blind_assembly/run_suite --phase train --suite-dir outputs/assembly_suite --nominal
embodichain run-example learning/blind_assembly/run_suite --phase evaluate --suite-dir outputs/assembly_suite --nominal
```

Preparation writes 54 trials: three tasks, study/full-range profiles, three
seeds and full/no-history/no-contact-supervision variants. Defaults preserve
12,288 environments and 5,000 updates, with validation every 100 updates.
Preparation does not run those experiments. Evaluation requires an actual
validation-selected best checkpoint; it cannot silently fall back to latest.
Final in-distribution evaluation uses fixed per-environment episode quotas so
shorter episodes do not receive disproportionate weight.
OOD axes remain pending until separately held-out geometry configurations are
provided. The suite never substitutes training geometry for an OOD evaluation.

Formal profiles fail before scene construction when arbitrary-topology
heterogeneous assets or qualified source robot dynamics are unavailable.
`--nominal` explicitly selects the development approximation: the Rizon4S arm,
parallel-jaw carriages carrying source finger mass and pad geometry, encoder
gravity feedforward, and one geometry per run. This retains physical gripping
without a weld, but does not reproduce the Robotiq four-bar/tendon dynamics,
the author's identified model, workspace-wide reset distribution or geometry
training distribution. Nut grasp stability is not fully qualified.

Native checkpoint snapshots freeze component paths. These tasks additionally
bundle the exact assembly manifest, prepared robot meshes/URDF, fidelity record
and source licenses. The relative resources can be replayed after relocating
the run, without the generator or original asset cache.

```bash
embodichain eval-policy RUN --checkpoint latest --episodes 2 --num-envs 2 --renderer no-render
embodichain run-example learning/blind_assembly/export_actor CHECKPOINT --config RUN/configs/train.yaml --output RUN/exports/actor.pt
```

The actor export is checked against eager inference for multiple batch sizes.
It contains normalization and deterministic action means; actuator scaling and
limits remain owned by the task/controller.

Run the validation-selected checkpoint through the production deterministic
episode evaluator with fresh strict native-state and capacity checks:

```bash
python -m scripts.benchmark.assembly.eval_peg RUN --checkpoint best \
  --episodes 8192 --num-envs 1024 --seed 20000 --output outputs/peg_heldout
```

Use a held-out seed only after selecting the checkpoint from validation. This
runner preserves the bundled task geometry and reset support; it does not
substitute a different distribution or silently select the latest checkpoint.
Final assessment assigns fixed per-arena episode quotas before stepping:
8,192 episodes over 1,024 arenas scores eight episodes from every arena.
Rows finishing early keep stepping/resetting, but their extra episodes are
unscored. This avoids favoring fast successes over slow failures. Historical
training validation retains its loaded first-completed default; its curve is
separate from this balanced final assessment.

Preview three fixed seeded outcomes in an actual DexSim rendering window:

```bash
DISPLAY=:1 embodichain run-example learning/blind_assembly/preview RUN \
  --checkpoint best --episodes 3 --seed 30000 --output outputs/peg_preview
```

The preview uses the same production checkpoint loader and deterministic policy
inference. It writes native-camera MP4/GIF recordings, an initial/terminal
contact sheet and a trace with checkpoint SHA-256, actions, joint positions and
outcomes. All requested episodes are included. Terminal frames are captured
before the next reset; camera reads never advance the physics. This small visual
preview is separate from the held-out success-rate measurement. A display,
native rendering dependencies and `ffmpeg` are required.

To transfer the same actor to the registered default Rizon4S and Robotiq2F85
assets, use the explicit replacement deployment:

```bash
DISPLAY=:1 embodichain run-example learning/blind_assembly/preview RUN \
  --gym-config extensions/embodichain_tasks/configs/tasks/manipulation/assembly/peg_insertion/task.study.default_assets.yaml \
  --checkpoint best --episodes 3 --seed 30000 --output outputs/peg_default_assets
```

This composes the existing URDFs and retains their meshes, materials, collision
geometry, inertials and source mimic relations. It changes the hand dynamics
and TCP relative to the trained nominal parallel-jaw model; actual new FK is
used for proprioception. The previous nominal success rate does not qualify
this transfer. The first recorded default-asset replay completed all three
episodes with **0/3 successes**, without retraining or forcing poses.

For fresh training with those registered assets, select
`agents/ppo.study.default_assets.dev.yaml` in the peg task directory. It uses
4096 arenas and 1000 updates, with the default hand's full mimic tree and an
explicit linkage `solref: [0.002, 1.0]`. The MuJoCo/Warp 3.15 profile uses the
matching version-guarded EPA helper and source-contact SDK patch; it is a
separate experiment from the earlier 3.12 nominal result.

The trainer's first-completed validation remains diagnostic. Rank every saved
checkpoint before independent testing with balanced development episodes:

```bash
python -m scripts.benchmark.assembly.select_peg --run RUN \
  --output outputs/default_asset_selection --episodes 512 --num-envs 64 --seed 10000
```

Pass the selected checkpoint's explicit path to the strict evaluator with a
different seed. `rl_report --selection-report` verifies that final assessment
used those weights and keeps its curve separate from training validation.

The [measured default-asset result](validation/default_retrain_20261026/RESULTS.md)
records 7240/8192 balanced independent successes (88.38%), below the 95% target.
Its random-initialized lineage reaches 1000 updates after checkpoint recovery;
validation selects update 850. The [experiment review](validation/default_retrain_20261026/EXPERIMENT_REVIEW.md)
separates the unchanged fixture/arm settings from the changed hand mechanism,
grasp drift, native execution sensitivity and remaining causal experiments.

See [CAPACITY.md](CAPACITY.md) for measured synthetic 12,288-environment policy
capacity, and [the implementation report](../../sim/assembly/IMPLEMENTATION.md)
for actual task smoke checkpoints, evaluation results and outstanding gates.
Scientific precision/performance runners under `scripts/benchmark/` require a
source checkout.

The [measured nominal peg result](../../sim/assembly/validation/rl_attempt/RESULTS.md) records the frozen model's 8006/8192 balanced held-out successes, all raw scored episodes and native previews. Training later stopped on a nonfinite native-state guard; the result is a single-geometry research candidate with failed training-stability qualification. It is not a completed three-task paper reproduction.
