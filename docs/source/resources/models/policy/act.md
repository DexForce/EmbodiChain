# Action Chunking Transformer (ACT)

The visual PushCube example uses a Franka arm to push a cube to a goal.
At each control step, the policy observes a 224 × 224 RGB image and nine
measured joint positions. ACT predicts a chunk of future actions. Its
temporal ensemble combines overlapping predictions into a command with
seven arm joint targets and one parallel-gripper command.

## Using a trained policy

Use the Python 3.12 environment with LeRobot 0.6.1 and the native camera
renderer. Load the exported `pretrained_model` directory together with its
`embodichain_act.json` manifest through `load_act_policy`. The adapter applies
the saved processors and returns physical commands. Call `reset()` after
every environment reset to clear the ensemble's action history.

## Training and evaluation

The example collects successful demonstrations, trains the official LeRobot
ACT model, then evaluates checkpoints on held-out simulation scenes. Follow
the [PushCube workflow](https://github.com/DexForce/EmbodiChain/tree/main/examples/learning/act)
for collection, training and evaluation commands. The
[ACT API reference](../../../api_reference/visual_act.rst) documents the
loader, adapter and training configuration.
