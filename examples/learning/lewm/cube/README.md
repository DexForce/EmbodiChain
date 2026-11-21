# Reproduce LeWM on OGBench Cube

This example starts from the published LeWM Single-Cube checkpoint, adapts its
visual encoder and projector to the EmbodiChain camera, and evaluates it on
the 50 fixed OGBench queries in DexUni. Set up the Python and GPU environment
with the [runtime guide](DEPLOYMENT.md). See the controller, camera, and
simulator settings in the [Cube task contract](../OFFICIAL_CUBE.md).

## Reproduce the native camera benchmark

<!-- cube-reproduction-start -->
Run from the repository root with the Cube environment active. Keep the
resource directory for repeated runs and use a fresh run directory for each
collection, training, and evaluation.

```bash
CUBE_PROFILE=reproduce
CUBE_RESOURCES=${CUBE_RESOURCES:-outputs/lewm-cube/resources}
CUBE_RUN=outputs/lewm-cube/$CUBE_PROFILE

# 1. Prepare the pinned dataset, weights, and simulator assets.
embodichain run-example learning/lewm/cube prepare --resource-root "$CUBE_RESOURCES"

# 2. Render native images from the archived poses.
embodichain run-example learning/lewm/cube collect \
  --resource-root "$CUBE_RESOURCES" --output "$CUBE_RUN/pairs" \
  --profile "$CUBE_PROFILE" --gpu-ids 0 --workers 8

# 3. Fit the visual encoder and projector and export the selected checkpoint.
embodichain run-example learning/lewm/cube train \
  --resource-root "$CUBE_RESOURCES" --data "$CUBE_RUN/pairs" \
  --output "$CUBE_RUN/vision" --profile "$CUBE_PROFILE" --device cuda:0

# 4. Evaluate the fixed queries and save frames for an optional preview.
embodichain run-example learning/lewm/cube eval \
  --resource-root "$CUBE_RESOURCES" --checkpoint "$CUBE_RUN/vision/export" \
  --output "$CUBE_RUN/evaluation" --device cuda:0 --gpu-id 0 --preview-query 12

# 5. Optionally render the saved frames as an MP4.
embodichain run-example learning/lewm/cube demo \
  --evaluation "$CUBE_RUN/evaluation" --query 12 --output "$CUBE_RUN/query-12.mp4"
```

1. `prepare` fetches and verifies the public LeWM checkpoint and expert corpus,
   checks out pinned upstream revisions, and builds the native assets and
   seed-42 evaluation queries. It writes `resources.json` with the input digests.
2. `collect` renders 40 poses per episode and saves the images and episode
   splits under `pairs/`. The `reproduce` preset uses 224 training episodes and
   48 validation episodes, with the fixed evaluation episodes held out.
3. `train` aligns native frame embeddings to embeddings from the upstream
   encoder at the same poses. It saves the checkpoint with the lowest
   held-out paired error as `vision/export/weights.pt` alongside `manifest.json`.
4. `eval` runs the pinned upstream CEM on native initial, goal, and replanning
   images. Each plan covers 25 controller steps. If the cube has yet to reach
   the goal, the evaluator replans and executes up to 50 steps in total.
   Success means the cube center comes within 4 cm of the target after a
   control step. Check `num_eval` (50), `success_count`, and `success_rate`
   in `evaluation/report.json`. The evaluator also writes `trace.npz` and
   `observations.npz`.
5. `demo` uses the frames saved by `eval --preview-query 12` to write
   `query-12.mp4`, with the live camera on the left and the goal on the right.

For a shorter collection and training run, change `CUBE_PROFILE` to `smoke`
and use `--workers 1` in `collect`. This preset uses 8 training episodes,
2 validation episodes, and 40 updates. Evaluation still runs all 50 queries.
Omit `--checkpoint` from `eval` to use the published weights on native images.
<!-- cube-reproduction-end -->

## Fine-tune on native expert trajectories

After `prepare`, use `--kind expert` to collect new successful trajectories in
DexUni and fine-tune both the encoder and the dynamics predictor.

```bash
embodichain run-example learning/lewm/cube collect --kind expert \
  --resource-root "$CUBE_RESOURCES" --output "$CUBE_RUN/expert" --gpu-ids 0 --workers 8
embodichain run-example learning/lewm/cube train --kind expert \
  --resource-root "$CUBE_RESOURCES" --data "$CUBE_RUN/expert" \
  --output "$CUBE_RUN/expert-model" --device cuda:0
embodichain run-example learning/lewm/cube eval \
  --resource-root "$CUBE_RESOURCES" --checkpoint "$CUBE_RUN/expert-model/export" \
  --output "$CUBE_RUN/expert-evaluation" --device cuda:0 --gpu-id 0
```

The expert path uses the upstream plan oracle and a fixed target per episode.
The `reproduce` preset collects 384 successful training and 64 successful
validation episodes. Each contains 201 native RGB frames, physical state,
and 200 outgoing controls. Expert training starts from the published weights
and updates LeWM with next-embedding prediction and SIGReg.
`--cache-mib 20480` enables a bounded per-split device cache for this
profile on GPUs with sufficient free memory.
