# Franka ODS PickPlace showcase

The entry point trains a small state policy from scratch, with a live expert
producer, an independent student evaluator, optional Viser/TensorBoard, and
explicit offline-data controls. See [implementation evidence](IMPLEMENTATION.md),
[design](REDESIGN.md), [review](REVIEW.md), and [efficiency protocol](BENCHMARK.md).
The [v2 comparison design](BENCHMARK.md) specifies four cold-start arms, five
paired seeds and fresh final tests; its [protocol manifest](BENCHMARK_PROTOCOL.json)
is a design artifact, not yet an executable experiment configuration.
Engineering smoke and training-reset qualification are separate from the final
80% success target; inspect each run's `summary.json`.
The [LeRobot offline experiment](LEROBOT_OFFLINE.md) records actual dataset
round-trip checks, offline learning results and the frozen-data comparison.

The [production/reuse tuning study](TUNING.md) describes the optional usage-budget
scheduler, throughput diagnostics and stability validation before formal comparison.

## Run the working recipe

```bash
python examples/data_pipeline/ods_pick_place_showcase.py \
  --sim-device cuda:0 --train-device cuda:0 --output /tmp/pick-place
```

The default recipe is `chunk-pose` with DLS, a 32-episode ODS cache, two rounds
of 5,000 updates and batch size 256. It predicts 64 absolute TCP/gripper targets
from two measured observations and executes 16 before replanning. Open Viser at
`http://127.0.0.1:8080` and TensorBoard at `http://127.0.0.1:6006` while the run is
active. Use `--no-viewer --no-video` for headless learning experiments. Both
services close when the run ends; checkpoints, curves, reports and recorded
initial/final test videos remain in the output directory.

The first invocation qualifies the physical expert, generates initial training
data, evaluates initialization, trains/evaluates each round and performs final
reserved evaluation. Reuse a matching `--qualification-report` for subsequent
experiments. Generation and physical evaluation dominate this small learner's
runtime; GPU speed alone does not determine time to completion.

The validated development-selected model achieved **99/100** reserved-test
successes. The later checkpoint regressed to **65/100**, so load the model named
by `summary.json.best_checkpoint`. See [failure analysis and working validation](WORKING_SHOWCASE.md). The GRU and
MLP results in older experiment reports are historical baselines, including the
previous teacher orientation; they are not claims about this default recipe.

## Physical task and policy

The unchanged task physically grasps a 5 cm cube, lifts it at least 5 cm, places
it within 3 cm of the fixed goal at table height, opens the fingers, withdraws
the TCP by 10 cm, and remains stable for 0.3 seconds. Initial cube XY varies by
±4 cm. Physics/control frequencies are 240/30 Hz, with a 600-step horizon.
There is no execution-time object attachment, teleport or student expert takeover.

Measured geometry policies use 51 current features: qpos/qvel, TCP/cube poses,
goal, relative vectors and cube velocities. Rotation features use two matrix
columns. Normalization is frozen from initial training data. Seven normalized
requests control arena-frame TCP translation/rotation increments plus the active
finger; mimic handling belongs to the robot.

`--policy chunk-pose` selects a two-layer MLP-512 with an action-chunk head
(approximately 644k trainable parameters). Targets are decoded from causal
pre-action TCP observations and recorded commands; measured next states are
never substituted for commands. Its pose predictions are converted to the same
bounded relative requests as the expert. A 0.01 normalized observation-noise
augmentation is used only during training. Input and target normalization are
frozen from initial data. No task phase or episode clock is an input.

`--policy mlp-pose` selects a two-layer MLP-256. `--policy gru-pose` selects the
small geometry GRU-128 fallback with full causal-prefix training and 24 supervised
steps per sequence. Hidden state resets only at episode reset. No expert stage,
clock or success flag is an input. `--policy mlp-joint` retains the geometric
joint-residual diagnostic; `--policy gru` retains the older 28-feature learner.

`--pose-ik solver` uses the robot's configured IK. The default `--pose-ik dls` selects a local
Jacobian step with damping 0.05, maximum joint increment 0.10, joint-limit clipping
and exact solver-root frame conversion. Its finite-step flag does not certify
pose convergence; complete physical task acceptance remains authoritative. The
expert executes through the same selected controller and records actual requests.
Its top-down grasp uses the cube-symmetric orientation closest to the fixed Panda
start, avoiding an unnecessary near-180-degree wrist turn.

## Qualification and small-data diagnostics

From the repository root, with compatible DexSim runtime and robot assets:

```bash
# Data/control/process smoke, with two generations and ten updates.
python examples/data_pipeline/ods_pick_place_showcase.py --smoke \
  --pose-ik dls --no-viewer --no-video --output /tmp/pick-place-smoke

# Qualify the selected physical teacher on 40 independent scenes.
python examples/data_pipeline/ods_pick_place_showcase.py --pose-ik dls \
  --qualification-only --output /tmp/pick-place-qualified

# One training reset, then ten independent student rollouts of that reset.
python examples/data_pipeline/ods_pick_place_showcase.py --policy chunk-pose \
  --pose-ik dls --stage overfit-one --updates-per-round 10000 --eval-every 10000 \
  --qualification-report /tmp/pick-place-qualified/expert_qualification.json \
  --no-viewer --no-video --output /tmp/pick-place-one

# Four training resets, then five student rollouts per reset.
python examples/data_pipeline/ods_pick_place_showcase.py --policy chunk-pose \
  --pose-ik dls --stage overfit-four --updates-per-round 10000 --eval-every 10000 \
  --qualification-report /tmp/pick-place-qualified/expert_qualification.json \
  --no-viewer --no-video --output /tmp/pick-place-four
```

The one/four diagnostic thresholds are 9/10 and 18/20 complete successes. Periodic
checks evaluate each training reset once; the dev-selected checkpoint receives
the complete repeatability panel at the end. These are training-reset diagnostics,
not held-out generalization results. Reusing a qualification checks task/controller
source hashes, physical configuration and at least 40 independent measured scenes.
The common qualification cost is explicitly excluded when reusing it.

The working recipe uses 32 initial expert episodes. A failed one-episode
diagnostic remains a reported limitation; it is not relabeled as a pass when the
32-episode policy succeeds. Default development evaluation uses 20 fixed scenes;
final acceptance uses reserved seeds 50000–50099 and the unchanged complete-task
80/100 target. Smoke uses a separate engineering seed range. Retain failures and
record any budget change rather than changing the physical success predicate.

## Online, offline and matched delivery

```bash
# Live ODS; omit no-viewer/no-video for Viser and TensorBoard.
python examples/data_pipeline/ods_pick_place_showcase.py --policy chunk-pose \
  --pose-ik dls --mode online --sim-device cuda:0 --train-device cuda:0 \
  --qualification-report /tmp/pick-place-qualified/expert_qualification.json \
  --output /tmp/pick-place-online

# Strong cold offline baseline: collect/save both configured groups, then train.
python examples/data_pipeline/ods_pick_place_showcase.py --policy chunk-pose \
  --pose-ik dls --mode offline-full --no-viewer --no-video \
  --export-data /tmp/pick-place-offline/corpus.pt \
  --output /tmp/pick-place-offline

# Small offline baseline: save G1 and train on it for the full update budget.
python examples/data_pipeline/ods_pick_place_showcase.py --policy chunk-pose \
  --pose-ik dls --mode offline-32 --no-viewer --no-video \
  --output /tmp/pick-place-offline32
```

`--mode frozen` keeps live G1 without refill; `offline-32` additionally saves,
closes the producer, loads G1 and trains. `--offline-data PATH` explicitly measures
a data-ready scenario. It must not be presented as an acquisition-inclusive run.
Offline format uses numeric CPU tensors, restricted loading, schema/content hashes
and episode/frame identity. The loader may retain the small state corpus in RAM.

An exported numeric corpus can also be persisted and loaded through the official
LeRobot SDK (validated with the installed LeRobot 0.4.4):

```bash
python examples/data_pipeline/_ods_pick_place_lerobot.py \
  --source /tmp/pick-place-four/corpus.pt --output /tmp/pick-place-lerobot
python examples/data_pipeline/ods_pick_place_showcase.py --policy chunk-pose \
  --pose-ik dls --stage overfit-four --offline-data /tmp/pick-place-lerobot \
  --updates-per-round 10000 --eval-every 10000 \
  --qualification-report /tmp/pick-place-qualified/expert_qualification.json \
  --no-viewer --no-video --output /tmp/pick-place-offline-four
```

To create the source corpus, add `--export-data /tmp/pick-place-four/corpus.pt`
to the four-reset command above. The LeRobot path stores `observation.state`,
`action` and numeric provenance in Parquet; the showcase sidecar preserves ordered
controller semantics, original episode boundaries and content hashes. Loading
reads actual SDK rows once, checks exact numeric equality, then samples resident
CPU tensors. This tests offline learning and initial loading cost, not native
LeRobot per-batch I/O. It uses the same locally generated expert demonstrations;
it does not introduce external demonstration data or a pretrained policy.
Small-data offline stages recover reset positions from valid frame-zero measured
observations and apply the same physical verification as live data stages.

For exact **MLP** batch replay, run online with `--export-data PATH
--record-batches`, then use `--mode offline-matched --offline-data PATH
--batch-manifest RUN/batch_ids.pt` with matching update/batch budgets. Archival
memory/I/O belongs to that diagnostic variant. Recurrent and chunk offline-full/frozen
causal sampling are supported; exact sequence batch-manifest replay is rejected.
`--mode serial` is an explicitly unmatched refresh diagnostic and cannot isolate
causal overlap speedup.

Use the same policy, data/target budgets and fixed hardware for a comparison.
Predeclare `--wall-time-budget SECONDS` from pilot timings. A timed-out run records
right censoring and cannot select post-deadline checkpoints. Cleanup and final
reserved verification time are reported separately. Never omit failed runs.

```bash
python examples/data_pipeline/ods_pick_place_compare.py \
  /tmp/pick-place-online /tmp/pick-place-offline /tmp/pick-place-offline32 \
  --output /tmp/pick-place-comparison
```

The command writes JSON/Markdown and quality/time, GPU-hour and measured resource
figures. It withholds unsupported speedup claims when attainment, panel sizes,
resource budgets or pairing are incomplete. See BENCHMARK.md for amortized reuse
and the distinction between delivery speed and learning efficiency.

## Artifacts and recovery

Use a fresh output directory. Each run saves configuration/source hashes, split
manifest, immutable checkpoints, per-scene reports, provenance, TensorBoard events,
efficiency/timeline records and cleanup status. Training-reset stages additionally
save complete student state/action traces. Missing device-wide peaks/actual kernel
overlap and other unsupported measurements remain null.

`--resume CHECKPOINT` restores the model, normalization, optimizer and random
states while starting a fresh ODS source; it does not restore shared cache contents.
Checkpoint evaluation validates the controller/observation schema. Viser defaults
to port 8080 and TensorBoard to 6006; select unused ports for concurrent runs.
Cameras remain display-only. No DAgger, feedback-driven collection, pretrained
model, aggregate replay framework or RLInf integration is included.
