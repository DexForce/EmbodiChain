Reinforcement Learning
======================

This section introduces the overall architecture and submodules of the embodychain RL (Reinforcement Learning) module. The RL framework supports mainstream algorithms (such as PPO) and provides flexible components for policy, buffer, trainer, etc., making it easy to extend and customize.

.. contents:: Table of contents
    :local:
    :depth: 2

Overview
--------

The embodychain RL module is used to train agents to accomplish tasks in simulation environments. It mainly includes algorithm implementations, policy networks, data buffers, training processes, and utility tools.

Architecture Diagram Example
---------------------------

.. code-block:: text

    +-------------------+
    |   train.py        |
    +-------------------+
             |
             v
    +-------------------+
    | Trainer           |
    +-------------------+
      |    |      |   |
      v    v      v   v
  Algo  Policy Buffer Env

- train.py is responsible for entry, config parsing, and module initialization.
- Trainer coordinates algorithm, policy, buffer, and environment.
- Algo/Policy/Buffer/Env are independent, making extension easy.

Module Categories
-----------------

- Algorithm (`algo/`): RL algorithm implementations, including `BaseAlgorithm`, `PPO`, etc.
- Buffer (`buffer/`): Trajectory data buffer, such as `RolloutBuffer`.
- Models (`models/`): Policy network modules, including `Policy`, `ActorCritic`, `MLP`.
- Trainer (`utils/trainer.py`): Main training loop and logging management.
- Config (`utils/config.py`): Algorithm config class definitions.
- Train Script (`train.py`): RL training entry script.

Extension and Customization
---------------------------

- Users can customize algorithms (by inheriting `BaseAlgorithm`), policies (by inheriting `Policy`), buffers, etc.
- Supports multi-environment parallelism, event-driven extension, and flexible config management.
- It is recommended to manage all parameters via config files for reproducibility and batch experiments.

Common Issues and Best Practices
-------------------------------
- Config files are recommended to use JSON for easy management and reproducibility.
- Parallel environment sampling can significantly improve training efficiency.
- The event-driven mechanism allows flexible insertion of custom logic (such as evaluation, saving, callbacks).
- It is recommended to use WandB/TensorBoard for training process visualization.

Example
-------

.. code-block:: bash

    python train.py --config configs/agents/rl/push_cube/train_config.json

For more details, please refer to the source code and API documentation of each submodule.

See also
--------

.. toctree::
    :maxdepth: 1

    algorithm.md
    buffer.md
    models.md
    trainer.md
    config.md
    train_script.md
