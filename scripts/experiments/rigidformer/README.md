# RigidFormer reference acceptance

This runner compares the shared SL learner with the unmodified optimization
recipe from `lucidrains/rigidformer` revision
`4ab4edac7271b49484748d6bf8086a497f13272d`. The external model and Box2D
generator retain their MIT license; the runner verifies the checkout revision
and installed model source bytes before training. No upstream source or
generated dataset/checkpoint is vendored here.

Use the `py311` environment with the optional pinned RigidFormer dependency,
`box2d-py` and its build prerequisite SWIG. Install optional dependencies in an
isolated environment inheriting `py311` if the main environment is shared.

```bash
git clone https://github.com/lucidrains/rigidformer.git /tmp/rigidformer-reference
git -C /tmp/rigidformer-reference checkout 4ab4edac7271b49484748d6bf8086a497f13272d
python scripts/experiments/rigidformer/acceptance.py \
  --reference /tmp/rigidformer-reference \
  --output /tmp/rigidformer-acceptance --threads 4
```

The full default qualification trains two modes (friction and collision), five
seeds (42–46), and two engines (reference and shared SL), each for 1,000 updates.
It keeps the reference's 64 training trajectories, 16 held-out trajectories,
three objects, 48 material points per object, 1/6-second model interval, B8,
dim128 model and AdamW settings. Both engines consume identical initialization,
sampling and FPS randomness. Periodic evaluation remains at updates 500/1,000
and preserves the reference RNG consumption.

The report includes a loss/gradient/update comparison, the reference's 32
single-step samples, its first held-out trajectory and all 16 held-out
trajectories. Rollouts contain two initial frames followed by 11 predicted
frames. The single-step metric is mean Euclidean point distance, despite the
upstream example labeling it RMSE. Constant-velocity and stationary baselines,
point/center errors and intra-object rigidity errors are retained. The one-sided
95% seed-paired bootstrap upper bound allows at most 5% relative degradation;
near-zero references use an absolute tolerance of 1e-5. Numerical comparisons
use atol1e-5/rtol1e-4. Fewer than five seeds or fewer than 1,000 updates report
pending acceptance and exit nonzero.

Outputs include a frozen manifest, source/data hashes, dependency versions,
per-seed immutable results/checkpoints, curves and `summary.json`. Any failure
is retained in `failure.json`. A completed reference comparison establishes
adapter/trainer agreement, not paper accuracy, action gradients or physics
solver qualification. The reference's collision limitations remain visible.
`training_seconds` includes the two periodic reference-trajectory evaluations;
the 32-window and full-panel final evaluations are outside that interval.

For a user-supplied fixed geometry corpus, the generic training entry point uses
`factory: embodichain.learning.sl.rigidformer:build_rigidformer_trainer`. Its
tensor-only archive contains `positions[E,T,O,P,3]`, `properties[E,O,A]` (or
`[E,O,P,A]`) and `step_dt`; optional valid lengths retain the source field
names. This regular episode-uniform sampler is separate from the reference's
event-biased Box2D sampling. The source clones under an explicit resident byte
budget, returns owned bounded batches and never crosses episode boundaries.
Reference geometry is supplied explicitly and point identities remain aligned
across time.

Importing the adapter does not load the optional model or simulation. The
current upstream nearest-neighbour feature uses a dense pairwise distance
tensor, so the adapter checks its estimated byte size before forwarding. This
bound is for that tensor alone, not total process/accelerator memory.
