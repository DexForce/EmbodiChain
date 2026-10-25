# RigidFormer geometry prediction

The installable model family is `embodichain_models.dynamics.rigidformer`. It
wraps `lucidrains/rigidformer` at the pinned revision
`4ab4edac7271b49484748d6bf8086a497f13272d` and trains through the shared
`embodichain.learning.sl` contracts. Install `embodichain[rigidformer]` only when
using this model. The upstream model and Box2D reference recipe retain the
complete MIT license in the model package's `THIRD_PARTY_LICENSES`.

The ordinary optional extra uses the public-index pin `rigidformer==0.2.0`,
without a Git dependency in EmbodiChain's published metadata:

```bash
python -m pip install 'embodichain[rigidformer]'
```

All four runtime Python files in the [PyPI 0.2.0 wheel](https://pypi.org/project/rigidformer/0.2.0/)
were verified byte-for-byte against the frozen commit above. This establishes
that distribution's runtime-source equivalence; the version number alone does
not qualify a reference environment. For exact Git installation in an isolated
`py311` environment, use [requirements-reference.txt](requirements-reference.txt)
and the [reference setup](research/README.md). The acceptance runner still
checks the checkout and installed Python bytes against the pinned Git blobs.

For a fixed tensor-only PyTorch geometry archive, use the installable factory:

```bash
embodichain train-sl --config examples/learning/rigidformer/configs/sl.train.yaml
embodichain train-sl --config my-recipe.yaml --resume outputs/rigidformer/sl/last.pt
```

Set `data.path` in the configuration to your archive. Paths resolve from the
caller's working directory. Its fields are `positions[E,T,O,P,3]`,
`properties[E,O,A]` (or `[E,O,P,A]`) and positive `step_dt` in seconds. Optional
`episode_lengths`, `object_lens` and `object_point_lens` describe finite padding.
Point identities remain aligned across frames, and the owned previous-frame
geometry is the reference for each sampled window. Sampling preserves episode
boundaries and explicit resident/batch byte budgets.

`RigidFormerModel.action_conditioned` is `False`. The adapter exposes future
geometry prediction and supervised parameter training. Action-gradient/APG and
native physics-solver deployment are outside its qualified capabilities. The
dense nearest-neighbour buffer is checked against `max_pairwise_bytes`; this
is a bound on that intermediate alone. Unequal point lengths require padded
`P <= 128`, including at rollout. Supplied point counts must be at least
`num_anchors`, including placeholder objects; fixed anchors index valid points.
The objective weights each scene contribution by its number of valid objects.

The reference experiment is an example, with private support outside the
installable model package:

```bash
embodichain run-example learning/rigidformer --help
embodichain run-example learning/rigidformer \
  --reference /tmp/rigidformer-reference \
  --output /tmp/rigidformer-acceptance --threads 4
```

See the [frozen reference protocol](research/README.md) for source/data checks,
friction/collision modes, seeds, budgets and acceptance thresholds. Every new
manifest records the executing adapter, SL and runner source hashes. Historical
reports remain bound to their original source and data; relocation does not
change or requalify that evidence. Downloaded data, upstream checkouts, weights
and generated reports remain outside Git and the wheel.
