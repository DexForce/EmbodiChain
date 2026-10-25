# RigidFormer reference acceptance

This runner compares the shared SL learner with the unmodified optimization
recipe from `lucidrains/rigidformer` revision
`4ab4edac7271b49484748d6bf8086a497f13272d`. The external model and Box2D
generator retain their MIT license; the runner verifies the checkout revision,
the Box2D dataset and training recipe, and every model-package Python file
against the pinned Git blobs. Installed package Python files must match those
same blobs before the optional package is imported. A dirty checkout with the
correct HEAD is rejected. No upstream source or
generated dataset/checkpoint is vendored here.

Use the `py311` environment with the optional pinned RigidFormer dependency,
`box2d-py` and its build prerequisite SWIG. Install optional dependencies in an
isolated environment inheriting `py311` if the main environment is shared.
The ordinary `embodichain[rigidformer]` extra resolves PyPI `rigidformer==0.2.0`;
its four runtime Python files match the frozen commit. The exact Git-source
requirement remains example-owned in `requirements-reference.txt`, outside
published dependency metadata. From a source checkout, install the ordinary
dependencies first, then replace only the optional upstream package at the
exact commit. The force reinstall prevents a same-version index installation
from standing in for this explicit Git-source installation:

```bash
conda run -n py311 python -m venv --system-site-packages /tmp/rigidformer-reference-env
/tmp/rigidformer-reference-env/bin/python -m pip install 'embodichain[rigidformer]'
/tmp/rigidformer-reference-env/bin/python -m pip install --force-reinstall --no-deps \
  -r examples/learning/rigidformer/requirements-reference.txt
git clone https://github.com/lucidrains/rigidformer.git /tmp/rigidformer-reference
git -C /tmp/rigidformer-reference checkout 4ab4edac7271b49484748d6bf8086a497f13272d
/tmp/rigidformer-reference-env/bin/python -m embodichain run-example learning/rigidformer \
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
use atol1e-5/rtol1e-4. Every paired training loss must also satisfy
`abs(sl_loss - reference_loss) <= 1e-5 + 1e-4 * abs(reference_loss)` at every
update; both complete curves must contain the requested number of finite
values. A matching final loss alone cannot qualify a run. Fewer than five
seeds or fewer than 1,000 updates report pending acceptance and exit nonzero.

Outputs include a frozen manifest, source/data hashes, dependency versions,
per-seed immutable results/checkpoints, curves and `summary.json`. Once a fresh
output directory is admitted, setup, dependency, revision, training and failed
or pending qualification failures are retained in `failure.json`. An existing
output directory is rejected without modifying its artifacts. A completed
reference comparison establishes
adapter/trainer agreement, not paper accuracy, action gradients or physics
solver qualification. The reference's collision limitations remain visible.
`training_seconds` includes the two periodic reference-trajectory evaluations;
the 32-window and full-panel final evaluations are outside that interval.

For a user-supplied fixed geometry corpus, the generic training entry point uses
`factory: embodichain_models.dynamics.rigidformer.sl_recipe:build_rigidformer_trainer`. Its
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

Finite padding is supported with unequal object counts and unequal point
counts when the padded point axis is at most 128. Each supplied point count
must be at least `num_anchors` (four by default), including placeholder objects
outside `object_lens`; fixed anchors must index valid points. Nonfinite padding
is unsupported. The pinned PointNet hierarchy uses unmasked gathered KNN
neighbors when the padded point axis exceeds 128, so unequal point counts in
that layout are explicitly rejected by both the objective path and rollout.
Full point lengths above 128 remain available with unchanged upstream math.
The 48-point Box2D reference does not enter the restricted layout. Focused
optional-model tests change finite padded geometry, attributes and targets and
check objective loss/gradients and valid rollout points at 48 and 128 points.

## Source-bound evidence

The relocation preserves the pinned upstream model, recipe math, parameter keys,
geometry schema, fixed seeds, budgets and acceptance gates above. New runs bind
`candidate_sources_sha256` in `manifest.json` to the executing model, adapters,
shared trainer and private runner. Existing reports are immutable historical
evidence for their recorded source, data and runtime; they are not relabeled as
results for a newer commit. A short paired run verifies relocation parity but
continues to report pending full acceptance under the unchanged 5-seed/1,000-step
threshold.
