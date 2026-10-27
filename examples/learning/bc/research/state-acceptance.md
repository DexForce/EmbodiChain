# State BC PickPlace: protocol and acceptance record

The first comparison has completed: neither state recipe passed its development
gate. A fresh 256-demonstration fallback is in progress. The independent
100-scene test remains sealed. State observations include simulator-provided cube
and goal information; these results do not establish vision or real-world
performance. PR #861 remains draft until the engineering and task gates pass;
the maintainer's review determines whether it is merged.

## Task and physical evidence

The task is Franka PickPlace on the default physics backend, with 240 Hz physics,
30 Hz control and a maximum of 600 control steps. The seven-component action is
the EmbodiChain relative end-effector pose and parallel-gripper request, with
0.02 m translation and 0.1 rad rotation scales and action clipping. It is not
the ManiSkill controller. Observations are 51-dimensional pre-action states;
each action is recorded at the same control step.

Complete task success requires a recorded lift of at least 5 cm, then the cube
within 3 cm of the goal and 1.2 cm in height, fingers open by at least 0.035 m,
TCP at least 10 cm from the cube, finite state, and linear/angular speeds below
0.05 m/s and 0.2 rad/s. All placement, opening, withdrawal and stability
conditions must hold continuously for 0.3 seconds. A lift or arrival at the
goal alone is not success.

For each scene the evaluator resets the environment and policy queue, runs no
expert, and stops on the first termination or truncation without auto-reset.
It records before/after observations, model targets, executed requests,
conditions, replans and termination. The reported strict success requires
success to have occurred before any failure (a simultaneous first failure does
not pass). `success_once` and `success_at_end` are separately recorded at the
actual episode boundary. This differs from ManiSkill's fixed-horizon
ignore-termination convention.

The qualified expert achieved **40/40** on seed base `20700000`, above the
required 38/40. The first 32 accepted historical demonstrations replayed
**32/32** with matching visible initial states; this does not verify hidden
simulator/controller state. These physical runs actually used source
`ec5aa1984eaeb196152f4a7e343e243f25ed86e3`. A separate source-equivalence
check compared the relevant physical code and configuration with the later
training source; it does not change the runs' recorded source identity.

## Recipes and frozen populations

Both use `embodichain_models.policy.bc` and the common SL trainer. The
[ManiSkill BC implementation](https://github.com/mani-skill/ManiSkill/blob/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/examples/baselines/bc/bc.py)
is a low-dimensional actor/training reference, not an expected task score.

| Recipe | Inputs, targets and execution | Optimizer and schedule |
| --- | --- | --- |
| A `maniskill_action` | One state → current seven-component request; predict every step; MLP 256×256; identity normalization | Adam 3e-4, batch 1024, 10,000 updates; evaluate 8,000 and 10,000. Frame-uniform sampling with replacement. |
| B `legacy_chunk` | Two states → 64 absolute TCP pose/rotation6D/gripper targets; execute 16 and replan; MLP 512×512; first-32 training statistics | AdamW 3e-4, weight decay 1e-6, gradient norm 1, batch 256, 5,000 updates; evaluate 4,000 and 5,000; training-only observation noise 0.01 in normalized units. |

B fills missing initial history with the first state and masks incomplete future
targets; the evaluator decodes absolute targets to the task's relative request.
A's sampler draws with replacement, unlike the reference's epoch-wise frame
shuffle. Both use a single training process and the specified global batches.
The state showcase CLI rejects multi-process launch; this does not remove DDP
support from the shared trainer.

The first round selected the first 32 physically successful episodes from the
original visual corpus, with no image loading. Its parent manifest SHA256 was
`6b757b0f2d1343ead3d9812abf8652757825981a64fe9a84042dcd8e9c558093`;
the executed state-view manifest SHA256 was
`30ee6ea9e56acba073a95a667630706b7c60250268ab17536ab4edb713c676d7`.
The view binds the original manifest, every episode hash and the pre-action
contract. Its preparation report binds the training-only normalization. It has
16,770 valid frames.

Both recipes used seeds `20400011`, `20400012`, and `20400013`, and the
same 20 development scenes from seed base `20500000`. Each seed must reach
**at least 18/20 at both** fixed development checkpoints. If all three seeds
pass A, select A; otherwise, if all three pass B, select B. For each seed,
select the higher development success, then lower goal error, then earlier
checkpoint; freeze the choice before testing. The comparison changes several
training and model choices and does not isolate one factor's effect.

Only after a recipe passes the full development gate and checkpoint selection is
frozen may the independent 100-scene panel at seed base `20600000` be opened.
Each selected seed must reach **at least 80/100**. Neither the already-used
historical 100-scene panel nor the sealed original visual test is a fresh test
for this experiment. The command-line evaluator does not enforce this study
gate: the experiment operator must enforce it.

## Completed first-round results

The six trainings and twelve fixed development evaluations used source
`1b565b5aea83936df04f1532ce772a85ed022597`. Each cell is complete task
successes out of the same 20 scenes; these are repeated evaluations on that
panel, not 240 independent scenes.

| Training seed | A, 8,000 | A, 10,000 | B, 4,000 | B, 5,000 |
| --- | ---: | ---: | ---: | ---: |
| 20400011 | 0/20 | 0/20 | 4/20 | 2/20 |
| 20400012 | 0/20 | 0/20 | 0/20 | 1/20 |
| 20400013 | 0/20 | 0/20 | 0/20 | 1/20 |

Both recipes fail the gate. All 120 A scene-runs reached the 600-step limit;
B lifted more often but still did not consistently complete release and
withdrawal. This observation does not identify a unique cause. The A and B
training losses have different targets and must not be compared as equivalent
closed-loop performance.

The completed reports, per-scene traces, checkpoint and dataset identities,
and read-only analysis are retained outside Git under
`/root/outputs/bc-state-redesign-20261027/execution/development-v1/` and
`/root/outputs/bc-state-redesign-20261027/review/all-development-analysis.md`
on the execution host. These paths are provenance, not portable download
locations. Preserve the original reports and their hashes when publishing the
artifacts; do not attribute them to a later source or regenerated dataset.

## Fresh-256 fallback

The next predeclared comparison keeps both recipes, seeds and checkpoints, with
256 demonstrations and a new 20-scene development panel at seed base
`20500100`. Additional historical demonstrations replayed only 221/224
successfully, so the historical 256-episode corpus was not admitted. A new
current-source corpus collected **256/256** physically successful demonstrations
in 256 attempts; its first 32 define its own normalization. Its parent manifest
SHA256 is
`e65906ef2936632b513b24fc3aa6f4838d10525b61514d328a36704949670be1`
and state-view manifest SHA256 is
`5e4d1538af97ecbdd525f48cfbd9955a63deddc328208f6856cd46fd17c7cf75`.
These differ from the first-round inputs. The fallback's complete development
result and gate are **in progress**; no 100-scene result is claimed here.
Keep original and fallback outcomes separate.

## Running the state workflow

Use `py311`, the compatible simulation runtime/assets and an explicitly
allocated idle physical GPU. Replace the corpus and output paths and GPU UUID
below. Each physical operation creates fresh artifacts; reuse or regenerate
data only after checking its manifest and episode hashes. A new collection is a
new data identity and is not a reproduction of the archived result by itself.
Record source revision, environment, runtime, GPU, seed, configuration, input
hashes, checkpoints and reports. The executable commands do not enforce the
study's total per-seed six-hour budget or compare the gates automatically.

```bash
set -e
conda activate py311
export CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-ALLOCATED-UUID
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
embodichain run-example learning/bc environment \
  --output outputs/bc-state/environment.json --device cuda:0
embodichain run-example learning/bc qualify \
  --output-dir outputs/bc-state/qualification --seed-base 20700000 \
  --scenes 40 --device cuda:0
python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["scenes"] == 40 and d["successes"] >= 38 and d["qualification_passed"]' \
  outputs/bc-state/qualification/report.json

# For the first-round population, supply the retained original manifest.
DATASET=/path/to/original/corpus-v3/manifest.json
embodichain run-example learning/bc replay \
  --manifest "$DATASET" --output-dir outputs/bc-state/replay-32 \
  --episodes 32 --device cuda:0
python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["episodes"] == 32 and d["successes"] == 32 and d["reuse_qualified"]' \
  outputs/bc-state/replay-32/report.json
embodichain run-example learning/bc prepare-state \
  --manifest "$DATASET" --output-dir outputs/bc-state/view-32 --episodes 32
```

If the historical source is unavailable or fails qualification/replay, preserve
that evidence and collect a separate current-source corpus. The command below
illustrates the fresh-256 fallback population and attempt bound; it does not
overwrite the historical view:

```bash
embodichain run-example learning/bc collect \
  --output-dir outputs/bc-state/fresh-256 --seed-base 20800000 \
  --episodes 256 --max-attempts 768 --device cuda:0
embodichain run-example learning/bc prepare-state \
  --manifest outputs/bc-state/fresh-256/manifest.json \
  --output-dir outputs/bc-state/view-fresh-256 --episodes 256
```

Set `VIEW`, `RUN_ROOT` and `DEV_BASE` to the first-round view and
`20500000`, or to the fresh-256 view and `20500100`. The following prepares
and trains all six runs before evaluating either recipe; use distinct fresh
output directories. Run each command under the declared time/resource budget.

```bash
VIEW=outputs/bc-state/view-32/manifest.json
RUN_ROOT=outputs/bc-state/run-32
DEV_BASE=20500000
for recipe in maniskill_action legacy_chunk; do
  for seed in 20400011 20400012 20400013; do
    run="$RUN_ROOT/$recipe-$seed"
    embodichain run-example learning/bc train \
      --recipe "$recipe" --view "$VIEW" \
      --environment outputs/bc-state/environment.json --output-dir "$run" \
      --seed "$seed" --device cuda:0 --prepare-only
    embodichain run-example learning/bc train \
      --recipe "$recipe" --view "$VIEW" \
      --environment outputs/bc-state/environment.json --output-dir "$run" \
      --seed "$seed" --device cuda:0 --train
  done
done
for recipe in maniskill_action legacy_chunk; do
  if [ "$recipe" = maniskill_action ]; then steps="8000 10000"; else steps="4000 5000"; fi
  for seed in 20400011 20400012 20400013; do
    for step in $steps; do
      embodichain run-example learning/bc evaluate \
        --checkpoint "$RUN_ROOT/$recipe-$seed/training/step_$step.pt" \
        --output "$RUN_ROOT/development/$recipe-$seed-$step.json" \
        --seed-base "$DEV_BASE" --scenes 20 --device cuda:0
    done
  done
done
```

Do not use these development commands to open the reserved test. To continue a
compatible interrupted offline run, `train --resume FILE --train` requires
the matching prepared recipe and source. Load only trusted checkpoints:
training and physical evaluation deserialize checkpoint objects.

## Engineering checks and historical comparisons

The BC/SL component checks covered the old sampler/RNG identity, valid-window
masks, noise train/evaluation behavior, Adam update, prefetch and exact
2→4-update recovery, including rejection of incompatible source, noise or
optimizer identities. Actual two-rank CPU/Gloo checks exercised the shared
trainer; they are engineering checks, not state-task DDP acceptance. The
ManiSkill actor was independently compared at batch 1024 for forward, loss,
gradients and an Adam update; B was checked against the historical state
model's noisy loss/gradients and target decoding. The B recipe checks ran at
source `659be8be7f4340ac42a2b7029c57cb953bcbd57c`. A real ODS producer smoke
at source `d6f66a9b03c9ec5341f42d7c1d0df7a76e2bd12c` supplied samples to both
recipes for two updates and snapshot saves; it was not a task-success test.
Mutable live ODS
and quota-managed delivery cannot be resumed exactly because their delivery
state is not restored; only a declared immutable, unbudgeted resident source
with matching content and semantics supports exact continuation. The ODS host
owns producer lifecycle and waiting.

The [historical state record](../../../ods/simple_bc_demo/docs/WORKING_SHOWCASE.md)
reports 99/100 for one selected seed using an earlier ODS training pipeline,
privileged simulator state and already-consumed test scenes. It establishes
task feasibility, not the present multi-seed acceptance. The
[PR828 visual record](acceptance.md) and
[later visual diagnostics](improvement-v2.md) retain their own source, data,
negative development results and sealed test. The visual model and its results
are separate from both state recipes; visual policy acceptance is future work.
