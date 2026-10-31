# LeWM visual dynamics

The current migration target is the official LeWM OGBench single-Cube task in
EmbodiChain/DexSim, with the pinned MuJoCo task as the reference. The previous
UR10 push-cube experiment remains available below as a diagnostic corpus.
Its results do not establish alignment with the official task. Model code and the supervised
factory are in `embodichain_models.dynamics.lewm`; collection, reference
comparison and simulator ownership remain here. Install the `lewm` extra.

The migration is in progress. The verified official queries now run through both
engines: MuJoCo succeeds on 34/50, native DexSim with the original planner on
34/50 after correcting the source equality-compliance mapping, and the migrated
model/planner on 32/50. The preceding named-pad checkpoint reached 33/50 and
31/50. Equal aggregate success does not establish physical alignment. Contact and neural
CEM arithmetic differences remain measurable. See [the alignment plan](OFFICIAL_CUBE.md)
for the exact protocol, remaining work and acceptance evidence.

```bash
embodichain run-example learning/lewm official-reference --swm-root /path/to/pinned/stable-worldmodel --ogbench-root /path/to/pinned/ogbench --output outputs/lewm/official-cube/alignment --compare-cem
embodichain run-example learning/lewm reference --resource-root outputs/lewm/reference --cube-weights /path/to/lewm-cube/weights.pt --cube-swm-root /path/to/pinned/stable-worldmodel --output outputs/lewm/official-cube/alignment/model-reference.json
embodichain run-example learning/lewm official-kinematics --reference outputs/lewm/official-cube/alignment --device cuda:0
```

Use a reference environment with `mujoco==3.11.0`, `dm_control==1.0.41`,
`pygame==2.6.1` and `pymunk==7.2.0`, plus the ordinary LeWM dependencies.
The pinned stable-worldmodel environment extra requires MuJoCo below 3.12.
The reference tool verifies clean upstream revisions before importing them.
It exports compiled mesh coordinates, inertias, the physical parameter sidecar,
and 100 source poses; the native check imports these through `SimulationManager`.
Explicit pre-build link overlays retain the source's small inertia tensors and
collisionless-link masses. The native check does not step dynamics or certify
the added zero-mass frames. Its report records these limits.

The following commands operate on the historical UR push-cube experiment:

```bash
embodichain run-example learning/lewm prepare --resource-root outputs/lewm/reference
embodichain run-example learning/lewm reference --resource-root outputs/lewm/reference --output outputs/lewm/reference.json
embodichain run-example learning/lewm collect --split train --output outputs/lewm/train.h5 --device cuda:0
embodichain run-example learning/lewm collect --split validation --output outputs/lewm/validation.h5 --device cuda:0
embodichain run-example learning/lewm train --train-data outputs/lewm/train.h5 --validation-data outputs/lewm/validation.h5 --output outputs/lewm/run --device cuda:0 --bf16
embodichain run-example learning/lewm collect --split test --output outputs/lewm/test.h5 --device cuda:0
embodichain run-example learning/lewm eval --model outputs/lewm/run/export --data outputs/lewm/test.h5 --device cuda:0 --output outputs/lewm/offline.json
embodichain run-example learning/lewm sim-eval --model outputs/lewm/run/export --device cuda:0 --trace outputs/lewm/sim-trace.h5 --output outputs/lewm/sim.json
embodichain run-example learning/lewm check-trace --model outputs/lewm/run/export --trace outputs/lewm/sim-trace.h5 --device cuda:0 --output outputs/lewm/trace-check.json
```

Collection defaults to 512, 64 and 50 episodes for train, validation and test.
Each file has disjoint seeds, uint8 RGB frames and raw bounded relative-joint
actions; training uses complete windows without crossing episode boundaries.
The first four of every five episodes use scripted geometry-guided collection;
the fifth uses random actions. Failures are retained. Geometry is not a model
or MPC input. A target image is rendered from the physical goal and the scene
state is restored before either controller runs.

Training uses frozen train-only action statistics, the shared SL checkpoint
format and a validation-selected weights-only export. `recipe.json` can also
be passed to `embodichain train-sl --config`. The example's `--resume` option
accepts trusted SL checkpoints from the same recipe and run. Model exports have
a checksum and bind the camera, action and control-period contract.

For action-block training, pass `train --action-block 5`. The corpus still stores
every raw frame and action; the source selects frames five timesteps apart and
concatenates all five intervening actions. Six raw joint controls therefore become
30 model inputs, and a five-step model rollout spans 25 raw controls. Existing
single-step exports retain `action_block=1`; changing the block requires training
a new model. Offline `eval` reports model-step errors and the corresponding model
control period.

`train --cache-mib 8192` optionally caches each corpus's raw uint8 images and
controls on the training device. The limit covers persistent storage per corpus;
model activations and gathered batches require additional memory. The cache
loads bounded chunks and returns independent batches. For CPU caching, use
`data.cache_device: cpu` and an explicit `data.cache_max_bytes` in the SL recipe.
Disk and cache sampling share the same source identity and sampling RNG.

Measured reachable goals provide a separate development protocol:

```bash
embodichain run-example learning/lewm prepare-goals --split validation --start-index 100 --episodes 20 --device cuda:0 --output outputs/lewm/goals.h5
embodichain run-example learning/lewm reachable-eval --models single=outputs/lewm/run/export block=outputs/lewm/block-run/export --pairs outputs/lewm/goals.h5 --episodes 20 --receding-horizon 5 --sampling-distribution training --action-selection mean --device cuda:0 --output outputs/lewm/reachable
embodichain run-example learning/lewm check-trace --model outputs/lewm/block-run/export --trace outputs/lewm/reachable/block.h5 --device cuda:0 --output outputs/lewm/reachable/block-check.json
```

`prepare-goals` records a guided reference, its numeric physical evidence and an
actual future RGB goal 25 raw steps later. Queries exclude starts already within
the success threshold and require admissible history. `reachable-eval` replays
the reference to recheck reachability in its current world, then restores an
owned snapshot for each controller. Serialized snapshot ownership is never
transplanted across simulations. All models receive only RGB goals and observed
histories with their executed raw actions. Geometry selects queries and checks
physical results; it is not a policy input.

`--receding-horizon` counts model blocks, so a value of five executes up to 25 raw
controls for a five-action-block model. Evaluation still stops at success or its
raw-step budget. Each model is paired with random and zero-action controls.
Traces retain the real prefix history and recheck the entire executed portion
of each selected plan, including a final partial block. This protocol is not
the existing rendered-goal acceptance experiment, and reports remain development
evidence. `sim-eval`, candidate-bank diagnostics and frozen-predictor ablations
currently retain their single-step execution route.

The reference check is CPU FP32 and uses fixed source/checkpoint revisions
and SHA-256 hashes. It checks preprocessing, image and action embeddings,
prediction, rollout and goal costs at `atol=1e-5, rtol=1e-4`. It does not
claim that Push-T pretrained weights transfer to the UR environment.

Offline evaluation reports one-, three- and five-step prediction error against
a persistence baseline and embedding spread. Simulation evaluation compares
LeWM and random actions on identical held-out seeds and reports success,
distance, planning latency and peak Torch GPU memory. A smoke run can reduce
`--episodes`, `--samples`, `--iterations` and `--topk`; a smoke run is not the
50-episode acceptance experiment. Repeated development tuning must use
validation data rather than the test seeds. Reports distinguish `beats_random`
from `passed`: the acceptance gate additionally requires 50 test episodes, the
default CEM budget and a verified rendering contract.

Training, offline evaluation and simulator evaluation must use the same recorded
rendering contract. `default` uses the project rendering settings (currently
Hybrid with offscreen DLSS); `legacy` is an alias for that default. `stable` is
an explicit diagnostic mode that disables offscreen denoising. Older models
with an unknown rendering contract, or deliberate
distribution-shift experiments, require `--allow-render-mismatch` and are marked
as unverified. A trace stores actual observations, actions, physical states,
planner inputs and RNG states. `check-trace` recomputes saved MPC decisions; it
also checks recorded distances against the saved cube state, but does not promise
that a fresh simulator run produces identical images or physics.
Planning and cycle latency exclude trace writing, though tracing can still affect
the runtime workload. Use `--start-index` to reserve fresh seeds and pass the same
starting index to `merge-sim` when aggregating disjoint shards.

`sim-eval --sampling-distribution training` initializes CEM with the frozen
training action mean and standard deviation. The default `bounds` distribution
uses the action-box midpoint and half-width. Candidates and returned actions
remain in raw controller units; the model applies its normalizer once.
`--action-selection mean` returns the first action of the final elite mean
sequence and recomputes that sequence's cost. The default `best` selects the best
sampled sequence across iterations. These options leave raw bounds, the raw
`min_std`, planning horizon and per-step execution unchanged. They isolate search
and selection behavior rather than reproducing the upstream action-block MPC
protocol. Both options are recorded in traces and reports; nondefault options
are development ablations and do not pass the existing acceptance gate.

Validation diagnostics separate prediction accuracy from action selection:

```bash
embodichain run-example learning/lewm diagnose-candidates --model outputs/lewm/run/export --data outputs/lewm/validation.h5 --seeds 200000 200001 200003 --device cuda:0 --trace outputs/lewm/candidates.h5 --output outputs/lewm/candidates.json
embodichain run-example learning/lewm finetune-predictor --model outputs/lewm/run/export --train-data outputs/lewm/train.h5 --validation-data outputs/lewm/validation.h5 --horizon 10 --output outputs/lewm/predictor-h10 --device cuda:0
embodichain run-example learning/lewm rescore-candidates --model outputs/lewm/predictor-h10/export --reference-model outputs/lewm/run/export --trace outputs/lewm/candidates.h5 --device cuda:0 --output outputs/lewm/candidates-h10.json
```

Candidate diagnosis starts near measured contact in recorded validation episodes
and executes a fixed bank of guided, perturbed and random continuations. It
restores numeric physical state and repeats a zero-action control to measure
remaining branch variation. It reports prediction/observation/distance rankings
and physical selection regret. Privileged geometry is used only for diagnosis.

Predictor fine-tuning freezes the image encoder, projector and action statistics,
caches bounded FP32 embeddings tied to the exact source and base weights, and
selects an export by validation rollout loss, including the unchanged baseline.
`--successful-tail-fraction` optionally changes training sampling while validation
remains uniform. Re-scoring verifies the frozen representation and statistics
before comparing the same recorded physical candidates. Reduced latent error or
better candidate ranking does not establish closed-loop success; run `sim-eval`
on validation seeds before freezing a model for fresh test evaluation.
