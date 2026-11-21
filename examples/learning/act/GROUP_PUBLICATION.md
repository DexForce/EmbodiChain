# Single-GPU visual ACT with group publication

The optional ODS mode publishes one vector rollout at a time. This PushCube
recipe uses 10 producer environments, 20 shared CPU slots, and at least 192
committed anchor draws per episode before its group can be recycled. The
learner gathers one current RGB/state observation and 30 future actions into
an owned minibatch. It retains the first group temporarily to freeze the
training normalization statistics. There is no second RGB training replay.

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

The [core acceptance results](results/GROUPED_CORE.md) include the five reserved panels and the archived double-layer comparison.

## Reproduce

Use the example's Python 3.12 environment, LeRobot 0.6.1, matching native
DexSim runtime, Franka assets and cached ImageNet ResNet-18 weights. Each seed
uses one GPU for its producer, learner, development panels and subsequent
single-arena verification. Run with a new output directory per seed.

```bash
export CUDA_VISIBLE_DEVICES=4 EMBODICHAIN_SIM_EXIT_PROCESS=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/extensions:$PWD/extensions/embodichain_tasks"
python -m examples.learning.act.collect qualify --episodes 100 --seed-base 8100000 --output "$ACT_ROOT/qualification"
for seed in 147101 147102 147103 147104 147105; do
    xvfb-run -a python -m examples.learning.act.grouped --seed "$seed" \
        --qualification "$ACT_ROOT/qualification/report.json" --output "$ACT_ROOT/$seed"
done
```

The protocol uses fresh learner seeds and fresh 50-scene development and
100-scene verification panels. The image landmark features, compact native ACT,
arena spacing, batch size, training budget and checkpoint schedule follow the
previous isolated group experiment. Two consecutive development panels must
each score at least 40/50 and return within 300 seconds. The timer seals the
best eligible checkpoint by success, goal distance and earlier update. Only
that checkpoint can be verified on the reserved panel. The all-seed gate
requires at least 95/100 for each of five policies. The independent reserved
rollouts are outside the training deadline.

Run the acceptance check after all seeds finish:

```bash
python -m examples.learning.act.audit_grouped --root "$ACT_ROOT"
```

It recounts the physical outcomes and consecutive development scores, checks
native policy hashes, validates all release receipts and writes
`acceptance.json`. An incomplete set, an audit issue or any seed missing its
deadline or quality target produces a nonzero exit status.
