# NeRD reference experiments

The [showcase](../README.md) owns resource preparation, default SL training,
export and offline evaluation. This directory documents controlled comparisons
with the original NeRD implementation and historical training workflow. Private
runners live in `../_support/`; logs, checkpoints, reports and machine-specific
plans belong in caller-owned output directories.

Run a support tool through the example launcher so source and installed-wheel
example packages receive the same bootstrap:

```bash
embodichain run-example learning/nerd/_support/_comparison --help
embodichain run-example learning/nerd/_support/_reference --help
embodichain run-example learning/nerd/_support/_official_math_rollout --help
```

These research comparisons require explicit source/runtime paths. Source-bound
acceptance uses source checkouts even though the model package and ordinary
showcase commands are installable. The reference workers use their selected
interpreter and the private stdlib bootstrap; an installed example directory
is not a globally importable `examples` distribution.

## Environments and frozen sources

Run unchanged upstream source in its supported Python/Torch/Warp environment.
Use a separate EmbodiChain environment for current Newton collection and model
training. Original-runtime helpers do not import the installed simulation SDK.
Keep virtualenv executable paths intact rather than resolving their symlinks
to a different base interpreter. Trusted external source/checkpoint execution
requires each tool's explicit trust option.

The [acceptance protocol](acceptance.md) pins reference source
`a58de92a82b4e1c1c260083b5b1930173322aceb`, dataset revision
`b9ec58f36da99cc99a93f97adfe74ebc9fd66586` and model revision
`78e581526570069410bf269b38c0c52c800b24a7`. Reference asset revision
`97e202e41fe7fb504656a6b163291305e400780d` is prepared by the showcase; raw
URDF/XML and license/attribution are external Hub payloads. Keep every existing
report bound to its actual source commit and environment. The extension/example
migration does not constitute a rerun of a historical passing matrix.

`_portable_reference.py` and `_historical_physics.py` adapt only the isolated
original runtime. Preserve the original numerical implementation separately from
compatibility transport; do not call a migrated implementation an upstream oracle.

## Comparison contracts

- Separate model/trainer changes, runtime kernels, coordinate transport and
  analytical-solver versions when interpreting differences.
- Preserve sampling, normalization, loss and model selection for trainer-port
  parity. Equal seeds alone do not prove the same consumed batches.
- Replay identical realized open-loop efforts for predictive accuracy. A
  controller using each model's own state measures closed-loop behavior.
- Keep current Newton and historical Warp in their recorded physical domains;
  do not replace signatures or normalizers to make incompatible exports load.
- Retain every requested trajectory and report nonfinite outputs and failures.
  Distinguish endpoint error from trajectory-time average error.
- Use the matching analytical solver as the accuracy reference. Distance to
  another neural model measures similarity, not ground-truth error.
- State dataset size, update budget, population and horizon independently.
  A complete bounded budget or README example does not reproduce paper-scale
  training or establish live solver rollout accuracy.

Tools write to fresh caller-supplied directories. They do not change the model's
production numerical policy or provide a blanket accuracy acceptance.
