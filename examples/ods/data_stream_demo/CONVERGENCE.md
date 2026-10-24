# Dual-camera PickPlace learning example

This recipe adds learning acceptance to the ODS visual example. It reuses the
current single-cycle Franka PickPlace physics, handwritten expert, bounded
publication handoff, visual policy and standard LeRobot recording. Review the
camera preview and resolved recipe before launching a training run.

## Preview and review

```bash
python -m examples.ods.data_stream_demo.preview \
  --device cuda:0 --camera-profile pick_place --output outputs/pick-place-preview
python -m examples.ods.data_stream_demo.converge review \
  --output outputs/pick-place-review.json
```

Mask an idle GPU before preview, for example `CUDA_VISIBLE_DEVICES=3`. Preview
creates an ODS producer, acquires one physically accepted expert episode, saves
actual RGB images and a side-by-side video, then closes the producer. It does
not construct a policy or optimizer. Preview data are not training or evaluation
data. The resolved `environment-config.json` records camera calibration.

`pick-place-v2` uses 256×256 third-person RGB with eye[-0.60,-0.45,0.50] and
target[-0.42,0.10,0.08], 60° FOV. The wrist camera is attached to
`FrankaPanda/fr3_hand_tcp`, with local eye[0.12,0,-0.04], target[0,0,0.05],
up[0,1,0] and 90° FOV. This avoids the initial palm occlusion in the historical
mount. The benchmark camera profile remains v3. Training and evaluation load
the same named calibration; mismatched checkpoint calibration is rejected.

## Candidate training recipe

The YAML at [configs/pick_place_dual_convergence.yaml](configs/pick_place_dual_convergence.yaml)
is the source of truth. `review` writes a resolved JSON packet and its input SHA.

| Setting | Candidate |
| --- | --- |
| Task | Current real-contact Franka single PickPlace; unchanged 600-step/30Hz limit and physical success predicate |
| Model | From-scratch shared ResNet10/GroupNorm/spatial embeddings; 256D/view; 512/512 action MLP |
| Input | Two causal frames per view and two 27D robot states; no cube pose or expert phase in policy forward |
| Prediction / execution | 32 absolute TCP/rotation/gripper targets; execute 8 before replanning |
| Work budget | 20,000 updates ×64 chunks =1,280,000 chunks /40,960,000 action targets |
| Optimizer | FP32 AdamW; weight decay1e-6; gradient norm limit1 |
| Learning rate | 500-update linear warmup to3e-4, cosine decay to3e-5 |
| Augmentation | Disabled for the first learning qualification |
| ODS | C32 /P4 /K2048; publication handoff; budget-aware refill; two encoder threads |
| Normalization | Initial32 training episodes only; frozen |
| Development | Initial policy and every2,000 updates; same20 fixed states |
| Development gate | Last two updated checkpoints both reach at least18/20 successes |
| Checkpoint selection | Highest development success, then smallest mean goal error; freeze selection before testing |
| Final test | One independent100-state panel, opened only after the development gate passes; target≥80/100 |
| Runtime bounds | Six-hour whole worker deadline; one-hour bound per evaluation |

K2048 is a provisional engineering cap, not a learning-qualified optimum. The
declared work requires at least625 trained sources if all deliveries reach
updates. Earlier retirement can require more. Keep all source exposure and
unused-tail records; do not increase K automatically to keep the GPU occupied.

Training, development, test and preview use distinct seeds20261025,20271025,
20281025 and20291025. Development data never fit normalization or enter a
training batch. The final panel must remain unopened during tuning; after its
outcomes are used to change a recipe, use a new sealed panel for a new claim.

The success criterion is practical task acceptance, not proof of mathematical
optimizer convergence. The candidate budget is not a promise of80% success.
Retain failed seeds/checkpoints and report an unmet target explicitly.

## Train after recipe review

First recheck the unchanged physical expert on the current installed stack.
The existing qualification-only path performs no policy learning and requires
at least38/40 measured successes. It uses no camera input for the expert; the
visual profile changes sensors only, with the same physics and action decoder.

```bash
CUDA_VISIBLE_DEVICES=3 python -m examples.ods.simple_bc_demo.train \
  --qualification-only --qualification-scenes 40 --no-viewer --no-video \
  --sim-device cuda:0 --train-device cuda:0 \
  --output outputs/pick-place-expert-qualification
```

Proceed after that command succeeds and the candidate learning recipe is
reviewed:

```bash
python -m examples.ods.data_stream_demo.converge train \
  --gpu-ids 3 4 --output outputs/pick-place-dual-learning
tensorboard --logdir outputs/pick-place-dual-learning/training/tensorboard
```

GPU indices are examples, not reservations. The supervisor admits two idle
GPUs, masks their UUIDs before model/simulation imports, binds eight CPU cores
and checks owned contexts. An optional `--aux-graphics-gpu-id` retains the
existing bounded passive-graphics allowance. The supervisor retains process
return codes and rejects leftover worker groups. Generation/evaluation use the
simulation device and policy training/inference use the second device.

The learning driver reuses `_Experiment` for acquisition, protected sampling,
actual updates, recording and failure cleanup. It adds a declared LR schedule,
TensorBoard losses, independent periodic development processes and immutable
checkpoint selection. Evaluation drains lifecycle events and recorder staging
while the producer remains alive. These pauses are part of this learning run;
it is not a fixed-budget speedup measurement or an RLInf adapter.

The full update budget runs even when an intermediate policy performs well,
so the final-two-development gate can reveal later degradation. If that gate
fails, final testing is skipped and the command exits nonzero. A failed final
test also exits nonzero; the model is never taken over by the expert.

## Outputs and limits

`training/` contains `initial.pt`, immutable periodic checkpoints, `final.pt`,
`development.json`, evaluator logs/outcomes, `selection.json`, TensorBoard events,
the existing source/anchor ledger and training trace, and the complete standard
LeRobot dataset. `selection.json` is written before `evaluation/final-test.json`.
`summary.json` keeps engineering completion separate from `learning_target_met`.
Loss, lifted rate, goal error and task success remain separate measures.

Checkpoints include model, optimizer, normalization, RNG and recipe. This entry
point currently evaluates saved checkpoints through `training_eval`; it does
not implement exact replay/source recovery or a convergence-run resume command.
Do not restart an interrupted run while claiming the original quota or source
cache was restored. Keep the saved outcomes and use a fresh output directory.

Runtime estimates depend on current generation/recording and evaluation service,
not optimizer time alone. The historical128-update timings cannot be scaled
into a reliable convergence-time estimate. Recheck hardware admission at launch
and retain all waits, recording drain and independent evaluation cost.
