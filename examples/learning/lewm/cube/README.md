# LeWM Cube showcase

Prepare the pinned OGBench single-Cube resources, collect EmbodiChain camera
observations, train LeWM, and evaluate the same 50 queries as the upstream
example. The closed-loop rollout constructs the example's DexUni environment
from `_env.py`.

Start with the [runtime setup](DEPLOYMENT.md). The [validation record](VALIDATION.md)
gives measured success rates and the [task contract](../OFFICIAL_CUBE.md)
describes the control and evaluation settings.

## Reproduce the native camera benchmark

From a source checkout or an installation with the Cube dependencies, choose
caller-owned resource and result directories.

For the 35/50 run, use the [Python 3.11 runtime and MuJoCo 3.11 asset compiler](DEPLOYMENT.md#mujoco-311-asset-compiler),
and supply that interpreter through `prepare --reference-python`. The
single-environment Python 3.12 run is also recorded in the validation table.

```bash
CUBE_RESOURCES=outputs/lewm-cube/resources
CUBE_RUN=outputs/lewm-cube/reproduction

embodichain run-example learning/lewm/cube prepare --resource-root "$CUBE_RESOURCES"
embodichain run-example learning/lewm/cube collect \
  --resource-root "$CUBE_RESOURCES" --output "$CUBE_RUN/pairs" --gpu-ids 0 --workers 8
embodichain run-example learning/lewm/cube train \
  --resource-root "$CUBE_RESOURCES" --data "$CUBE_RUN/pairs" \
  --output "$CUBE_RUN/vision" --device cuda:0
embodichain run-example learning/lewm/cube eval \
  --resource-root "$CUBE_RESOURCES" --checkpoint "$CUBE_RUN/vision/export" \
  --output "$CUBE_RUN/evaluation" --device cuda:0 --gpu-id 0 --preview-query 12
embodichain run-example learning/lewm/cube demo \
  --evaluation "$CUBE_RUN/evaluation" --query 12 --output "$CUBE_RUN/query-12.mp4"
```

The default [reproduce preset](configs/reproduce.json) renders 224 training
episodes and 48 validation episodes at 40 recorded poses per episode. The
encoder receives native images. The published LeWM encoder supplies target
features from the paired upstream frames. The dynamics and action statistics
come from the official weights. The trainer selects the visual checkpoint
by held-out paired error and writes a portable weights-only export to
`vision/export`.

`eval` uses native start, goal and replanning images with the pinned upstream
CEM. It writes `report.json`, `trace.npz` and `observations.npz`.
The MP4 uses recorded online frames, with the live camera at left and the
native goal image at right. Run `eval` without `--checkpoint` to use the
published Cube weights with the same native images.

Use [`--profile smoke`](configs/smoke.json) for both `collect` and `train`
to check the workflow at a smaller budget. The smoke checkpoint can run the
same evaluator. Its optimization budget is for integration checks.

## Generate successful native expert trajectories

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

The expert path uses the upstream plan oracle with online state readback,
the archived initial distribution and a fixed target per episode. It saves
384 successful training and 64 successful validation episodes in the
reproduction profile. Each contains 201 native RGB frames, physical state
and 200 outgoing controls. Expert training warm-starts the official
weights and updates LeWM using next-embedding prediction and SIGReg.
`--cache-mib 20480` enables a bounded per-split device cache for this
profile on GPUs with sufficient free memory.
