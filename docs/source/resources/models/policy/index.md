# Policy Models: Generating Robot Actions

| Model | Inputs | Prediction | Example |
| --- | --- | --- | --- |
| [NMG](nmg.md) | Arm joints, tool pose, ordered goals, previous action | One action per joint toward the active goal | [Franka and PiPER waypoints](https://github.com/DexForce/EmbodiChain/tree/main/examples/sim/motion/planners/nmg) |
| [ACT](act.md) | Camera images and measured joint positions | A chunk of future actions learned from demonstrations | [Visual PushCube](https://github.com/DexForce/EmbodiChain/tree/main/examples/learning/act) |

```{toctree}
:hidden:

nmg
act
```

## Original paper

Zhao et al. (2023) introduce ACT and the ALOHA system used to collect its
bimanual demonstrations. [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://www.roboticsproceedings.org/rss19/p016.html)
