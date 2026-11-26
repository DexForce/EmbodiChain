# Visual ACT with grouped ODS publication

The optional ODS mode publishes one vector rollout at a time. This PushCube
recipe uses 10 producer environments, 20 shared CPU slots, and at least 192
committed anchor draws per episode before its group can be recycled. The
learner gathers one current RGB/state observation and 30 future actions into
an owned minibatch. The camera publishes 224×224 frames. ACT receives 56×56
RGB after area pooling, nine measured joint positions, and one normalized
elapsed-control feature. Calibration copies the joint and action statistics
from the first publication.

```python
with engine.group_batch(32, 30, anchor_fields=[rgb_key, state_key]) as batch:
    optimizer_update(batch)
    synchronize_training_device()
next_generation = engine.try_refill_group()
```

An empty destination may be filled immediately. A completed group with one
under-sampled trajectory stays readable. A raised exception in an optimizer
context discards that batch's credits. Repeating a request while the writer is
active returns the pending generation. The bounded ledger stores one counter
per physical slot and the last authorization. Ordinary Dataset samples do not earn recycling credit. A grouped run
uses the explicit optimizer context for credited training.
The experiment logs release
receipts and the actual physical evaluation panels alongside each checkpoint.

## Reproduce

Run the commands from the repository root using the [ACT environment](README.md#environment-and-control-contract).
Each seed uses one physical GPU for the producer, learner and evaluation.
Use a fresh output directory. Expert qualification can be reused when the task
identity matches and its scenes are disjoint from the protocol.

```bash
export CUDA_VISIBLE_DEVICES=0 EMBODICHAIN_SIM_EXIT_PROCESS=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export ACT_ROOT="$(mktemp -d)"
export PROTOCOL="$PWD/examples/learning/act/grouped_elapsed_protocol.json"
xvfb-run -a python -m examples.learning.act.collect qualify \
    --episodes 100 --seed-base 8100000 --output "$ACT_ROOT/qualification"
for seed in 207101 207102 207103 207104 207105; do
    python -m examples.learning.act.timed_grouped --protocol "$PROTOCOL" --seed "$seed" \
        --qualification "$ACT_ROOT/qualification/report.json" --output "$ACT_ROOT/$seed" || break
done
python -m examples.learning.act.audit_grouped --protocol "$PROTOCOL" --root "$ACT_ROOT"
```

The clock starts before Xvfb. Two consecutive development checks must each
score at least 40/50. A second score of 50/50 allows early submission.
At 295 seconds the launcher’s training process locks the best eligible policy
by success count, goal distance and earlier update. The decision must be
published by 300 seconds. Each submitted policy then runs the same 100 reserved
scenes in independent single arenas with full-rate RGB capture.

The audit prints publication time, reserved score and the lowest draw count at
release for each seed. It writes checkpoint fingerprints, development results
and release receipts to `$ACT_ROOT/acceptance.json`. It exits successfully when
all five policies meet the deadline and each reaches 95/100.

## Single-GPU RAM replay

Set `"retired_history": true` in a copy of the protocol and pass that file to
both the cold launcher and the auditor. The control can set
`"record_group_draws": true` to produce the same journal with all-live batches.
The default protocol and dual-GPU recipes use their existing live batch path.

With a global batch of 32, each successful optimizer step uses up to eight
distinct retired trajectories and fills the remaining positions from the
active CPU slots. At an occupied-group release the engine copies 32 valid
frame anchors per outgoing trajectory with its action targets, continuity
boundaries and padding mask. The learner keeps the most recent 96 displaced
trajectories in process RAM. With this protocol's 100-trajectory budget and
20 active slots, at most 80 displaced trajectories are available.

The per-step journal in `sampling.json` records live IDs, history IDs and
the step boundary of each refill authorization. The auditor reconstructs
live credits at each release and reports history draws and mean distinct
trajectory counts over consecutive 10- and 100-update windows. An optimizer
failure discards the pending live credits and the journal entry.
The calibration, development selection, 300-second publication budget and
independent reserved evaluation use the same protocol for both arms.

## Compare dual-GPU supply layouts

Both layouts train one ACT policy using two local DDP ranks and a global batch
of 32. With `per-rank`, each rank owns a five-environment producer and a
ten-slot cache on its GPU. With `shared`, rank 0 owns one ten-environment
producer and a twenty-slot cache, and sends half of each CPU batch to rank 1.
Each layout budgets 100 accepted trajectories and uses the same 192-draw
floor. Rank 0 selects and evaluates the checkpoint on the first GPU.

Use the elapsed-input protocol and a qualified expert report from the
commands above. Select an idle pair of identical GPUs. Keep the CUDA,
LeRobot and DexSim environment and the ResNet-18 weight cache fixed across
the paired runs.

```bash
python -m examples.learning.act.paired_grouped --gpus 1,2 \
    --qualification "$ACT_ROOT/qualification/report.json" \
    --root "$ACT_ROOT/pair"
```

The paired launcher runs a recycling and checkpoint smoke test in each layout,
then alternates their order for the five seeds. It saves GPU telemetry,
per-run supply evidence, `comparison.json` and `comparison.md`. The report
uses committed updates 101 through 600 for steady-state rates. A partial
window is recorded with its completed step count. Deadline and reserved-scene
outcomes are recorded for all ten planned runs.

To run a single layout with the same cold clock, set `CUDA_VISIBLE_DEVICES`
to the ordered GPU pair and pass `--nproc-per-node 2` with
`--producer-layout per-rank` or `--producer-layout shared` to
`examples.learning.act.timed_grouped`. The first visible GPU owns evaluation
in both layouts. The single-GPU command uses the default launcher settings.

## Validation

A pilot on two NVIDIA A100-SXM4-80GB GPUs used the same observation layout and
300-second budget. Each selected policy for seed 207101 came
from step 1,500 and ran on the same 100 reserved scenes.

| Layout | Learner seed | Published (s) | Reserved successes | Minimum committed draws at release |
| --- | ---: | ---: | ---: | ---: |
| Per-rank | 207101 | 295.001 | 94/100 | 254 |
| Shared | 207101 | 295.001 | 94/100 | 318 |
| Shared | 207102 | — | — | 262 |

Both published policies fell short of the 95/100 gate. The shared run for
207102 ended without a checkpoint eligible for publication. The audit keeps
those outcomes separate from the draw-floor check.
The [historical report](GROUPED_REPAIR.md#acceptance) covers the earlier
landmark-assisted observation layout.
