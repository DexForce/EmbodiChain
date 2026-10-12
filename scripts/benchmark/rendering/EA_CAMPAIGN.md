# RTX 5090 × Isaac Lab 3.0.0-EA campaign

This campaign continues formal definition 2.0: 82 declared cells, including two
unsupported external-deformation cells, with 16 prespecified publication points.
The implementation keeps smoke, three-repeat coverage and independent ten-repeat
publication data separate. Image-quality failures remain measurements; only
verified matching quality evidence permits equal-quality ratios.

## Frozen baseline and prerequisites

Use clean Isaac Lab tag `v3.0.0-EA`, commit
`ae37b028ea415c91ea2bc32609efcd759ed2b974`, and Isaac Sim 6.1.0.0.
The local EA environment uses Python 3.12, Torch 2.11/CUDA 12.8 and NumPy 2.3.1;
record its package inventory instead of identifying release compatibility from
Isaac Lab's internal distribution version. Both worker environments need
`nvidia-ml-py==13.595.45`. Keep external source clean; dependency constraints
belong to the runtime environment rather than source patches.

EA camera settings use `IsaacRtxRendererCfg.global_settings`, with explicit
render mode and disabled frame generation. Canonical environment roots are
`/World/envs/env_{i}`; root and camera partition tokens are checked. State
publication invalidates EA's scene-state cadence. Preview images and actual
settings must be checked again after a runtime/profile change. Profile 1.8-ea
has a new hash; prior 1.7 results and PRO 5000 evidence remain historical groups.

Prepare the three common scenes with `rendering-preview`. Review
`scene_comparison.png`, `franka_camera_comparison.png` and `review.json`, then
save an explicit hash-bound layout approval using the format in [README](README.md).
Approval concerns scene layout and cameras; it does not certify image quality.
`--num-envs 16` also produces a batch preview, retaining all RGB/depth/label arrays.
The contact sheet and pose agreement cover env0; batch intrinsics and partition
metadata have their own explicitly recorded populations.

Commit and freeze benchmark source before timing. The source tree must remain
unchanged while acquiring original-history and routing proofs.
If the orchestration or reporting driver needs a correction, keep the measurement
checkout frozen and run the corrected driver from a separate clean checkout with
`--source-worktree "$FROZEN_SOURCE"`. Formal launchers and GPU adapters still import
the original source, whose commit and file hashes must match the campaign. Each
new attempt records both the execution checkout and the corrected driver identity;
the campaign definition and prior timing records remain unchanged.

## Plan and performance execution

The generated campaign records package/source identity, profiles, the full case
universe, phase slots and resource budgets. Generated evidence stays outside Git.

```bash
python -S -m scripts.benchmark.rendering.campaign --root "$CAMPAIGN_ROOT" --list
python -m scripts.benchmark.rendering.campaign --root "$CAMPAIGN_ROOT" --initialize \
  --scene-root "$SCENE_ROOT" \
  --embodichain-python /home/dex/miniconda3/envs/open/bin/python \
  --isaaclab-root /home/dex/workspace/sources/IsaacLab \
  --isaaclab-python /home/dex/workspace/sources/IsaacLab-v3.0.0-EA/.venv/bin/python
python -m scripts.benchmark.rendering.campaign --root "$CAMPAIGN_ROOT" --phase smoke
python -m scripts.benchmark.rendering.campaign --root "$CAMPAIGN_ROOT" --phase coverage
python -m scripts.benchmark.rendering.campaign --root "$CAMPAIGN_ROOT" --phase publication
```

Core points are Franka N=1 RGB at 256²/FHD, static/moving and both profiles (8),
R-04/R-05 N=1/16/64 (6), and ordinary R-09 device/host delivery (2).
Formal timings use 30 warmups and 300 measured batch captures. Backend order
alternates within repeat; workers remain isolated and serial on GPU 0.

`--case-id` is repeatable for explicit staged selection. Capacity stops apply only
within the same track/backend and require OOM or verified renderer-limit evidence;
a generic crash or freshness failure is not a capacity boundary. A failed point
can use `--retry-failed --case-id ID`, creating another immutable invocation.
Initial and retried worker reservations share one fixed budget, with 20% of total
capacity reserved for additional work. Original failed attempts remain in the ledger.

The evidence limit is 100 GiB with at least 20 GiB filesystem headroom. Before a
new operation, the campaign counts all registered evidence locations and checks
a conservative reservation for every backend/repeat in the admitted job, on the
actual destination filesystem. Blocked work remains not-run with a
reason. Histories/routing outputs use a sibling `evidence/` directory outside the
frozen source tree; its bytes are included. Shared hardlinked references count
once physically and retain independent per-experiment checksum bindings.

## Untimed proofs and independent evaluation

Create an isolated CPU reference environment using the optional reference
requirements. Cache and verify official learned-metric weights before evaluation.
The experiment evaluator performs no model downloads. Run these commands with
that environment, after primary GPU timing has finished:

```bash
python -m scripts.benchmark.rendering.campaign_proofs --root "$CAMPAIGN_ROOT" --kind history
python -m scripts.benchmark.rendering.campaign_proofs --root "$CAMPAIGN_ROOT" --kind routing
python -m scripts.benchmark.rendering.campaign_proofs --root "$CAMPAIGN_ROOT" --kind quality
python -m scripts.benchmark.rendering.campaign_proofs --root "$CAMPAIGN_ROOT" --kind calibration
python -m scripts.benchmark.rendering.campaign_proofs --root "$CAMPAIGN_ROOT" --kind quality --reevaluate
```

`--list` inspects source runs without starting acquisition; `--run-id` selects a
source explicitly. A failed/interrupted history or routing proof can use
`--retry-failed --run-id ID`; a new attempt directory and worker reservation preserve
the original proof. The proof driver also accepts `--source-worktree` for the clean
original measurement checkout. History covers every original publication moving run's
construction/freshness prefix, warmup, all 300 measured exposures and its final
warmup guard. The other moving captures remain diagnostic and cannot certify
original timed history. History/readback has no throughput claim.

Routing uses representative repeat0 R-04/R-05 N=16/64 runs. It perturbs first,
central and last environments plus first/central/last 32-view boundaries, retaining
all unaffected outputs and restoration failures. Isaac pose-buffer snapshots are
GPU buffers, but shader consumption is not separately exposed. Native Camera's
public pose getter reads scene-graph transforms and uploads them to a tensor;
this is recorded as authored pose evidence, with native renderer GPU uniforms
explicitly missing. It must not be relabelled as observed GPU renderer state.

Independent references use the fixed 4096→16384 spp noise-only precision policy,
seeds 0/1 and eight CPU threads. Shared cache keys include scene, camera, object
state, resolution, settings, implementation and reference-runtime identity.
Corrupt/incomplete/conflicting entries fail closed before reuse. Each importing
experiment registers the complete checked source closure.

Calibration uses the calibration scene's first ten views for training and last
ten for heldout validation, with frozen actual controls. The owner recomputes
heldout RGB/noise evidence and may reject the receipt. A passing receipt is bound
to target references through the existing verified binder; targets still require
their own image and temporal checks. Reevaluation appends evidence and preserves
original execution/timing fields. Normals remain quality-unassessed.

## Offline reporting and acceptance

```bash
python -S -m scripts.benchmark.rendering.campaign --root "$CAMPAIGN_ROOT" --report-only
```

`campaign-report.md`, coverage CSV, figure-data CSV, summary JSON and SVG/PDF
curves link revalidated per-experiment reports and source run IDs. Separate plots
retain phase, attempt, hardware, profile, scene, modalities and delivery scope.
Missing points are not zero; statistics use independent process medians and MAD.

Completion requires explicit states for all 82 cells, three scheduled independent
trials per runnable performance point, ten for each prespecified publication point,
and proof-linked outputs or retained failure/blocked reasons. Execution completion,
quality qualification and measurement coverage are independent. A filtered command,
preview or smoke does not complete the campaign. Original PRO 5000 results cannot
be pooled with this new 5090/EA baseline. Scene approval and storage availability
remain real execution gates; the campaign does not create approval receipts.
