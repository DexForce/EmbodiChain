# Paused technical-validation checkpoint

The user paused migration work on 2026-10-31 and requested that the current
EmbodiChain work be saved to draft PR #895. `STATUS.json` is the status of this
snapshot. **Full physical alignment is not verified.**

The production task retains the last accepted native-drive recipe. Frozen
50-query measurements are MuJoCo 34 successes, native with the original planner
33, and native with the migrated planner 35. The fresh default source-action
replay gives 32 successes and 48 matching outcomes; its state/action/goal arrays
exactly match the preceding accepted provider. Native-only or recovered successes
are not evidence of physical parity.

`reports/` preserves the query-level reports, isolated-stage summaries, rejected
candidate comparisons, source parameter manifest and prior machine checkpoint.
`EXECUTION_HISTORY.md` preserves the detailed local execution history. Historical
entries saying ACTIVE refer to the run before the pause; they do not resume it.

The last qualified isolated provider was engine SHA
`57dcde3308c4c49d9aaccf93d0d2ee04dbb42b9c88b3019cf87a7cf545230f22` and GPU SHA
`d499a135a2e37a328f349ae39df9bf1c5d4d932c987862759e6c11d84c8b36db`.
It passed 102 affected SDK cases, 299 related Default-scope project cases and
42 context cases. Those results precede the latest soft-mimic-velocity WIP.
The known Newton `AutoSolverCfg` failures remain unresolved.

The newest code adds an optional per-articulation PGS soft mimic velocity policy.
Its C++ and CUDA builds completed, but post-build physical, benchmark and
whole-task validation were interrupted by the pause. Missing-API baselines failed
as expected before implementation. Do not treat that code as qualified or enable
it in the task by default. Resume with the mixed-articulation, hard-row,
CPU/TGS/old-provider, distant-body and real14 checks in the execution history.

`experiments/` contains quarantined diagnostic scripts; `rejected-source-servo/`
contains the rejected project-owned controller/integration prototype they may
load. They are not active task or package code. Scripts retain the original local
resource layout under `outputs/lewm/official-cube`; restoring that layout and the
pinned dependencies is required to rerun them. Copyright headers and formatting
were normalized for this source snapshot; `SOURCE_MANIFEST.json` records original
and snapshot hashes.

`physx-patches/` contains NVIDIA-licensed source patches. Apply the soft-velocity
WIP patch after the three previously qualified patches. It requires corresponding
SDK changes in the local isolated SDK worktree. **The private SDK implementation
is not published in this EmbodiChain PR.** Its location and dependency status are
recorded in `STATUS.json`; no SDK remote push is part of this update.

The 186 GiB local resource tree, official training data/checkpoints, third-party
checkouts, environments and compiled libraries remain outside Git. This snapshot
does not redistribute those artifacts or claim to reproduce paper training.
