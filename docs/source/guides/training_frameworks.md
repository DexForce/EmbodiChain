# Choose a training framework

EmbodiChain supports RLinf and LeRobot integrations as well as native training.
Choose the workflow by its environment/data contract and checkpoint format.

| Workflow | Training owner | Start here |
| --- | --- | --- |
| Native RL with PPO, GRPO or differentiable APG | EmbodiChain RL | {doc}`/overview/rl/index` |
| External RL workers consuming EmbodiChain Gym tasks | RLinf | {doc}`rlinf` |
| Visual imitation learning with official ACT | LeRobot 0.6.1 | {doc}`lerobot` |
| Model-owned supervised objectives such as dynamics learning | EmbodiChain SL | {doc}`/overview/learning/supervised_learning` |

## Shared simulation and data, separate training runtimes

EmbodiChain owns task registration, scene/robot configuration, sensors,
observations, action semantics, rewards and physical evaluation. RLinf owns its
distributed actors, rollouts and trainer. LeRobot owns ACT training, processors,
normalization and native checkpoints. Installing EmbodiChain does not install
RLinf; follow its linked integration recipe in the worker environment.

`embodichain train-rl` and `embodichain train-sl` launch native trainers.
The external workflows use their own launchers; their checkpoints are not
interchangeable with native trainer checkpoints. Use the ACT example's evaluator
for its portable LeRobot policies and the RLinf workflow for RLinf artifacts.

## Before running an experiment

1. Choose a runnable task deployment and preserve its file-owned physics backend.
2. Match observation features, units, control-part ordering, action semantics and
   control cadence to the policy. A matching tensor shape alone is insufficient.
3. Install the selected dependency profile in every process that consumes the
   environment or data. See {doc}`/quick_start/install` and the framework guide.
4. Run a small collection/training/evaluation smoke workflow before scaling.
5. Preserve task configuration, dependency versions, dataset provenance and
   checkpoint processors. Keep training, development and test scenes separate.

LeRobot dataset compatibility does not imply support for every LeRobot policy.
Likewise, RLinf task discovery does not prove that a particular VLA observation
or chunked-action mapping is implemented. The linked guides describe the
concrete workflows and adaptation boundaries.
