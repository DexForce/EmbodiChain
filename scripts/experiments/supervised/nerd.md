# NeRD supervised-learning acceptance

The production NeRD trainer delegates optimizer updates to
`SupervisedLearner`. Its existing data, codec, numerical execution, learning-rate
schedule, export and checkpoint contracts remain authoritative. The separate
`train-sl` recipe builds the common offline trainer and uses the common SL
checkpoint format.

Run a bounded standalone SL recipe from the repository root after downloading
the published dataset into `outputs/datasets`:

```bash
embodichain train-sl --config extensions/embodichain_tasks/configs/tasks/classic_control/cartpole/sl.train.yaml
embodichain train-sl --config extensions/embodichain_tasks/configs/tasks/classic_control/ant/sl.train.yaml
```

These 1,000-update recipes demonstrate the public entry point; they do not claim
full-budget convergence. Configure the output directory explicitly for each run.
The existing `embodichain dynamics` CLI retains its historical training/export
formats. New common-format checkpoints resume through `train-sl` with the same
recipe and data identity.

## Frozen comparison

Use PR #855 revision `a58de92a82b4e1c1c260083b5b1930173322aceb` as the
independent reference source. Use published dataset revision
`b9ec58f36da99cc99a93f97adfe74ebc9fd66586` and model revision
`78e581526570069410bf269b38c0c52c800b24a7`. Downloaded roots must contain
`cartpole/v1` and `ant/v1`. The runner verifies all data-shard checksums and
records model/dataset manifest hashes, weights hashes and source identities.

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/experiments/supervised/nerd_acceptance.py \
  --reference-source /path/to/pr855-worktree \
  --datasets /path/to/downloaded/neural_dynamics/nerd \
  --models /path/to/downloaded/neural_dynamics/nerd \
  --device cuda:0 --output /path/to/fresh/results
```

The default matrix performs:

- Full test-split offline evaluation of the same published Cartpole/Ant weights
  through independent reference and candidate processes.
- Five paired seeds (`0..4`) per robot, 1,000 updates each, batch size 512. Model
  architecture and train-only normalization match the published artifacts;
  weights start from scratch. The schedule retains the original 4,995,000-update
  horizon, 5,000-update interval and offset rather than shortening it to the
  acceptance interruption point.
- First-update comparison of predictions, loss, raw/clipped gradients and
  updated parameters using `atol=1e-5`, `rtol=1e-4`.
- Exact final-weight comparison and a five-seed, one-sided 95% paired bootstrap
  gate allowing at most 5% normalized-validation-MSE degradation. Near-zero
  reference metrics use the established absolute-tolerance mode.
- A process-restarted 500+500 update candidate run for seed zero of each robot,
  compared with uninterrupted weights, optimizer, sampler, counters and RNG.

`summary.json` is the machine-readable gate; the command exits nonzero if a gate
fails or fewer than five paired seeds are supplied. Seed lists must contain
distinct integers; duplicate or malformed lists are rejected before output
creation or worker startup. `report.md` contains time,
memory, quality and trainer-ranking tables. Per-process requests, logs,
checkpoints, curves and first-update captures remain under the supplied output
directory. Use a fresh directory for every matrix.

Time includes worker setup, fitting and validation. Process peak RSS is a Linux
high-water measurement; GPU peaks cover the Torch allocator. The reference
model uses its original physical data signature. No Scene-generated model is
relabeled as a newly generated direct-Newton domain, and these offline checks
do not establish native solver rollout accuracy or paper-scale reproduction.
