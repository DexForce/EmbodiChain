# LeRobot offline PickPlace experiment

Date: 2026-10-17. The actual LeRobot offline run completed 3,000 BC updates.
Its selected step-1,000 policy achieved **3/20 complete successes and
15/20 lifts** on repeated training initial states. It
**did not pass the 18/20 repeatability gate**. The
same-source frozen ODS run obtained 3/20 complete successes and 15/20 lifts.
On the separate 20-state development panel, the selected policy achieved
**0/20 complete successes and 2/20 lifts**; initialization achieved 0/20 for both.
The small-data student therefore did not demonstrate reliable generalization.
This experiment does not establish an ODS training-speed advantage.

[Machine-readable results](LEROBOT_OFFLINE_RESULTS.json) retain every evaluation
record, failures, loss samples, audit hashes, checkpoint comparisons and cleanup
results. The prior evidence is described in [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Data and controlled settings

The source is the existing four accepted expert episodes in
`/tmp/ods-v2-dls-gru-four-diagnostic/corpus.pt`, exported with the official
**LeRobot 0.4.4 SDK** to `/tmp/ods-v3-lerobot-four/dataset`. It contains **4 × 332 =
1,328 valid frames**, written as four LeRobot episodes at 30 Hz. These are the
same simulator-generated demonstrations as the frozen comparison, not newly
collected human or external data. LeRobot `observation.state` has 51 features,
`action` has seven raw policy-request features, and packed `ods.provenance`
preserves source run, episode, frame and generation IDs.

A full SDK read-back recovered observations, actions, validity masks, frame IDs
and generations **bitwise identically**, including padded data. The schema hash is
`2bd9a129ff929fc0f47ac56e5a571d5191cfbc2f0f5042534fe643cfc2f74b59` and content ID is
`712dc54494ac1db686874bc5b8219224c9b120eefb61ad07751f38161e816c15`. Initial model and normalization hashes match the
source frozen run. The CPU audit is retained at
`/tmp/ods-v3-lerobot-four/data_audit.json`.

Both runs use a one-layer **GRU-128**, measured 51-feature geometry observations,
six normalized relative EEF pose requests plus one parallel-gripper request, and
bounded local DLS control. The task, success predicate, ±4 cm reset range, 30 Hz
control and 600-step horizon remain unchanged. There is no expert takeover or
privileged phase input during student execution.

Training uses seed 42, 3,000 updates, batch size 64, 24 supervised steps per
sequence with full causal prefixes, start probability 0.25, Adam at 3e-4 and
gradient norm limit 1. Statistics remain frozen from the four source episodes.
The learner receives **4,608,000 valid supervised targets**. No ODS producer is
created for the offline training run; expert qualification reuses the existing
38/40 report with matching source/controller checks.

## Training-state repeatability

Periodic evaluation executes each of the four training initial positions once
(seeds 21000–21003). These positions come directly from valid causal frame-zero
cube coordinates, equal to the source reset snapshots. Checkpoint selection uses
success rate, lift rate and negative mean final goal error in that order, with
the initialization included. The selected checkpoint is then evaluated five
times per training position (seeds 22000–22019). This is a fitting/repeatability
diagnostic, not unseen-state success.

| Updates | Complete success | Lift | Mean final goal error (cm) |
| --- | --- | --- | --- |
| 0 | 0/4 | 0/4 | 38.90 |
| 500 | 0/4 | 2/4 | 27.17 |
| 1,000 | 1/4 | 4/4 | 17.06 |
| 1,500 | 0/4 | 2/4 | 39.06 |
| 2,000 | 0/4 | 0/4 | 36.87 |
| 2,500 | 0/4 | 2/4 | 52.29 |
| 3,000 | 1/4 | 4/4 | 23.30 |

The selected step-1,000 checkpoint has a repeated-panel mean final goal error
of **37.24 cm** and mean episode length
**558.95 steps**. Per-position success counts are
2/5, 1/5, 0/5, 0/5. Every failed episode is retained.

## Comparison with the same-source frozen run

Saved model state dictionaries, including normalization/control buffers, were
compared on CPU at updates 0, 500, 1000, 1500, 2000, 2500, 3000. All model tensors were
**bitwise identical** at every compared checkpoint;
the maximum absolute difference was
0.0. The audit is in
`/tmp/ods-v3-lerobot-four/checkpoint_audit.json`. This is an empirical result of
this run. No recurrent minibatch manifest was recorded or replayed, so exact
minibatch identity is unverified and is not claimed.

The periodic panel records matched the source run, and the selected repeated-panel
records matched it. Logged loss values
also matched exactly:

| Updates | Offline arm loss | Offline gripper loss | Frozen arm loss | Frozen gripper loss |
| --- | --- | --- | --- | --- |
| 50 | 0.02484425 | 0.08248561 | 0.02484425 | 0.08248561 |
| 1,000 | 0.00049794 | 0.00023884 | 0.00049794 | 0.00023884 |
| 3,000 | 0.00013479 | 0.00003642 | 0.00013479 | 0.00003642 |

Regression loss fell while physical task success remained low. Reading the same
supervision through LeRobot did not remove this learning/control limitation.
This rules out a different persisted observation/action payload as the cause of
the observed difference between low training loss and low success in this run;
it does not identify the ultimate source of rollout failure or attribute the
low success to ODS. The same small-data learner fails with persisted offline
supervision.

The [exported learning curve](/tmp/ods-v3-lerobot-four/learning_curve.png) overlays
periodic training-reset success/lift and logged losses for both runs. Each
periodic checkpoint uses only four training initial states; the curves do not
represent held-out generalization or a speed comparison.

## Development-20 evaluation

The selected step-1,000 checkpoint was frozen before
looking at this panel and evaluated on seeds 20000–20019. The pre-evaluation
selection is retained in `/tmp/ods-v3-lerobot-four/development_selection.json`. These states were not
used for this run's training, normalization or checkpoint selection. This remains
a development panel, not the reserved final test or a multi-seed acceptance result.

| Policy | Complete success | Lift | Mean final goal error (cm) | Mean episode length |
| --- | --- | --- | --- | --- |
| Initialization | 0/20 | 0/20 | 35.35 | 600.00 |
| Selected step 1,000 | 0/20 | 2/20 | 34.84 | 600.00 |

The frozen pre-evaluation selection matches the completed training summary.
The evaluator cleanup result is retained in `development/cleanup.json`; it
reported exit code 0 and
closed without forced shutdown. Every failure and all twenty records for both policies are retained in the JSON report and
`/tmp/ods-v3-lerobot-four/development`. No reserved final-100 scene was evaluated.

## Timing interpretation and limits

The official LeRobot SDK reads each row once into resident CPU tensors before
training. The existing causal-history sampler then supplies batches from memory.
Therefore this tests LeRobot persistence/loading integrity and offline learning;
it **does not measure native LeRobot per-minibatch loader throughput**.

The separate development evaluation uses `cuda:3` on the same GPU model while
training-reset verification uses `cuda:1`. Only the device and GPU index differ
in that evaluator configuration. These diagnostics overlap in wall time and
therefore do not support a single-GPU timing or efficiency comparison.

The input dataset was already ready. Its collection/export costs are excluded
from this data-ready run. Runtime metrics include simulator evaluation, setup,
loading and training according to the retained timeline; they cannot be directly
compared with a cold ODS run as an end-to-end speedup. With only one seed and four
episodes, no success-rate efficiency, T80, native I/O throughput or ODS advantage
is established. The 32/160-episode sweep and reserved final 100-state evaluation
were not run. Failures and unmet qualification remain visible.

## Validation

The focused change validation passed 131 tests. Public API coverage remained
2604/2604 and the project-context check passed. The training evaluator exited 0
with no engine cleanup error. Native DexSim still emitted the existing entity
balance warning during teardown; it was retained in `train.log` and was not
treated as a new learning result or silently removed. The separate development
evaluator also exited 0 without forced shutdown. Both owned GPUs (1 and 3)
returned to 0 MiB allocated after completion. The full repository suite and full
Sphinx build were not run.

## Reproduction and retained artifacts

The commands below use this host's Python 3.11 / PyTorch 2.7 / LeRobot 0.4.4
runtime and the existing temporary compatibility overlay for stale native Newton
facade exports. The task still uses the default physics backend. Choose an idle
GPU explicitly; the recorded run used `cuda:1`. Use fresh output directories:
the exporter and runner refuse to overwrite existing artifacts.

```bash
cd /root/sources/EmbodiChain-ods-pick-place-showcase
PYTHONPATH=.:extensions:extensions/embodichain_tasks:/tmp/ods-dexsim-runtime \
EMBODICHAIN_SIM_EXIT_PROCESS=0 \
/root/miniconda3/envs/py311/bin/python \
  examples/data_pipeline/_ods_pick_place_lerobot.py \
  --source /tmp/ods-v2-dls-gru-four-diagnostic/corpus.pt \
  --output /tmp/ods-v3-lerobot-four/dataset

PYTHONPATH=.:extensions:extensions/embodichain_tasks:/tmp/ods-dexsim-runtime \
EMBODICHAIN_SIM_EXIT_PROCESS=0 \
/root/miniconda3/envs/py311/bin/python \
  examples/data_pipeline/ods_pick_place_showcase.py \
  --policy gru-pose --pose-ik dls --stage overfit-four \
  --offline-data /tmp/ods-v3-lerobot-four/dataset \
  --updates-per-round 3000 --eval-every 500 --batch-size 64 --seed 42 \
  --qualification-report /tmp/ods-v2-dls-qualified40/expert_qualification.json \
  --sim-device cuda:1 --train-device cuda:1 --no-viewer --no-video \
  --output /tmp/ods-v3-lerobot-four/gru
```

- Export log and SDK dataset: `/tmp/ods-v3-lerobot-four/export.log`, `dataset/`.
- Full training output: `/tmp/ods-v3-lerobot-four/train.log`, `gru/`.
- Source comparison: `/tmp/ods-v2-dls-gru-four-diagnostic`.
- Integrity/model audits: `/tmp/ods-v3-lerobot-four/data_audit.json`, `checkpoint_audit.json`.
- Separate development driver: `/tmp/ods-v3-lerobot-four/evaluate_development.py`;
  log `/tmp/ods-v3-lerobot-four/development.log` and results under `development/`.

These `/tmp` paths identify retained artifacts on the experiment host and are
not packaged datasets. The JSON report retains compact numeric evidence in the
repository. No failed result was excluded and no reserved test scene was used.
