# Using EmbodiChain with LeRobot

EmbodiChain integrates LeRobot for demonstration datasets and official ACT
imitation learning. The bundled visual PushCube workflow collects Franka
demonstrations, invokes LeRobot's trainer and evaluates native policies in
EmbodiChain simulation.

## Select the dependency profile

The repository's `pyproject.toml` selects dependencies by Python version:

| Workflow | Python | LeRobot |
| --- | --- | --- |
| Core collection with the older dependency profile | 3.10 / 3.11 | 0.4.4 |
| Official ACT collection, training and simulation evaluation | 3.12 workflow | `lerobot[training]==0.6.1` |
| Standalone offline export without EmbodiChain | >=3.12 | `lerobot[dataset]==0.6.1` |

Follow {doc}`/quick_start/install` for the simulator installation and the
[ACT example](https://github.com/DexForce/EmbodiChain/blob/main/examples/learning/act/README.md)
for its CUDA/PyTorch profile. Do not upgrade only LeRobot inside the older
profile: Python, Torch, TorchVision and TorchCodec constraints must agree.
Dataset storage version (v3.0) is separate from the library version (0.6.1).

## Collect, train and evaluate ACT

Run the example modules from the repository root after installing the Python
3.12 environment. The complete commands and qualification criteria live in the
[ACT walkthrough](https://github.com/DexForce/EmbodiChain/blob/main/examples/learning/act/README.md).

1. Run `python -m examples.learning.act.collect qualify` to qualify the expert.
2. Use its qualification report with `collect collect` to create independent
   training and validation datasets. Preserve each dataset's `report.json`.
3. Run `python -m examples.learning.act.train` with those two reports. The
   wrapper invokes `lerobot.scripts.lerobot_train`, records the recipe and
   fingerprints, and uses native LeRobot training state for recovery.
4. Run `python -m examples.learning.act.evaluate` on the development panel to
   select a checkpoint, then evaluate that policy on held-out test scenes.

These module names identify workflow stages; use the walkthrough's full
arguments, seeds and fresh output directories. Smoke collection produces an
unqualified corpus and is not evidence of learning acceptance.

The policy observes RGB and measured joints and predicts absolute arm targets
plus a normalized gripper command. The adapter in `embodichain_models.policy.act`
validates ordered data/control contracts. Loading uses native weights and
processors together with `embodichain_act.json`; keep the complete
`pretrained_model/` directory or portable development `best/` directory.
Normalization, temporal ensembling and episode resets follow the official
runtime. Historical custom ACT `.pt` checkpoints require retraining.

## Use existing demonstration datasets

For finalized expert recordings, first follow {doc}`offline_expert_datasets`.
Use {doc}`lerobot_export` when preparing the supported source format for the
official 0.6.1 consumer, including timestamped subtask language and native depth.
The exporter runs in a separate environment, preserves source evidence and
checks the official reader before publishing its output.

Export does not choose a policy, convert action semantics or automatically apply
the generated language recipe. Inspect feature names, units, action ordering,
image dimensions and cadence before using a different upstream training recipe.
The bundled ACT workflow uses its own collection reports and provenance checks;
an arbitrary exported dataset is not automatically admitted by that wrapper.

## Ownership and troubleshooting

| Symptom | Check |
| --- | --- |
| LeRobot import or binary codec failure | Install the complete matching Python/Torch/TorchCodec profile |
| ACT report or schema rejected | Use qualified collection reports with the matching task/action contract |
| Resume rejected | Preserve output ownership, source contents and recipe arguments |
| Policy loads but actions are wrong | Retain processors and manifest; check units, ordering and normalization |
| Dataset reads but language is missing from training | Explicitly apply the exported language recipe in the consumer |

EmbodiChain owns collection, provenance and simulation integration. LeRobot owns
its ACT model, optimizer, processors and native checkpoint state. See
{doc}`training_frameworks` for native RL/SL and {doc}`rlinf` for external RL.
