# LeWM visual dynamics

This example trains a single-GPU visual world model from the UR push-cube
environment and evaluates target-image MPC. Model code and the supervised
factory are in `embodichain_models.dynamics.lewm`; collection, reference
comparison and simulator ownership remain here. Install the `lewm` extra.

```bash
embodichain run-example learning/lewm prepare --resource-root outputs/lewm/reference
embodichain run-example learning/lewm reference --resource-root outputs/lewm/reference --output outputs/lewm/reference.json
embodichain run-example learning/lewm collect --split train --output outputs/lewm/train.h5 --device cuda:0
embodichain run-example learning/lewm collect --split validation --output outputs/lewm/validation.h5 --device cuda:0
embodichain run-example learning/lewm train --train-data outputs/lewm/train.h5 --validation-data outputs/lewm/validation.h5 --output outputs/lewm/run --device cuda:0 --bf16
embodichain run-example learning/lewm collect --split test --output outputs/lewm/test.h5 --device cuda:0
embodichain run-example learning/lewm eval --model outputs/lewm/run/export --data outputs/lewm/test.h5 --device cuda:0 --output outputs/lewm/offline.json
embodichain run-example learning/lewm sim-eval --model outputs/lewm/run/export --device cuda:0 --output outputs/lewm/sim.json
```

Collection defaults to 512, 64 and 50 episodes for train, validation and test.
Each file has disjoint seeds, uint8 RGB frames and raw bounded relative-joint
actions; training uses complete windows without crossing episode boundaries.
The first four of every five episodes use scripted geometry-guided collection;
the fifth uses random actions. Failures are retained. Geometry is not a model
or MPC input. A target image is rendered from the physical goal and the scene
state is restored before either controller runs.

Training uses frozen train-only action statistics, the shared SL checkpoint
format and a validation-selected weights-only export. `recipe.json` can also
be passed to `embodichain train-sl --config`. The example's `--resume` option
accepts trusted SL checkpoints from the same recipe and run. Model exports have
a checksum and bind the camera, action and control-period contract.

The reference check is CPU FP32 and uses fixed source/checkpoint revisions
and SHA-256 hashes. It checks preprocessing, image and action embeddings,
prediction, rollout and goal costs at `atol=1e-5, rtol=1e-4`. It does not
claim that Push-T pretrained weights transfer to the UR environment.

Offline evaluation reports one-, three- and five-step prediction error against
a persistence baseline and embedding spread. Simulation evaluation compares
LeWM and random actions on identical held-out seeds and reports success,
distance, planning latency and peak Torch GPU memory. A smoke run can reduce
`--episodes`, `--samples`, `--iterations` and `--topk`; a smoke run is not the
50-episode acceptance experiment. Repeated development tuning must use
validation data rather than the test seeds.
