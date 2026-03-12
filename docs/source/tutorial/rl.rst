.. _tutorial_rl:

Reinforcement Learning Training
================================

.. currentmodule:: embodichain.agents.rl

This tutorial shows you how to train reinforcement learning agents using EmbodiChain's RL framework. You'll learn how to configure training via JSON, set up environments, policies, and algorithms, and launch training sessions.

Overview
~~~~~~~~

The RL framework provides a modular, extensible stack for robotics tasks:

- **Trainer**: Orchestrates the training loop (calls algorithm for data collection and updates, handles logging/eval/save)
- **Algorithm**: Controls data collection process (interacts with environment, fills buffer, computes advantages/returns) and updates the policy (e.g., PPO)
- **Policy**: Neural network models implementing a unified interface (get_action/get_value/evaluate_actions)
- **Buffer**: On-policy rollout storage and minibatch iterator (managed by algorithm)
- **Env Factory**: Build environments from a JSON config via registry

Architecture
~~~~~~~~~~~~

The framework follows a clean separation of concerns:

- **Trainer**: Orchestrates the training loop (calls algorithm for data collection and updates, handles logging/eval/save)
- **Algorithm**: Controls data collection process (interacts with environment, fills buffer, computes advantages/returns) and updates the policy (e.g., PPO)
- **Policy**: Neural network models implementing a unified interface
- **Buffer**: On-policy rollout storage and minibatch iterator (managed by algorithm)
- **Env Factory**: Build environments from a JSON config via registry

The core components and their relationships:

- Trainer → Policy, Env, Algorithm (via callbacks for statistics)
- Algorithm → Policy, RolloutBuffer (algorithm manages its own buffer)

Configuration via JSON
~~~~~~~~~~~~~~~~~~~~~~

Training is configured via a JSON file that defines runtime settings, environment, policy, and algorithm parameters.

Example Configuration
---------------------   

The configuration file (e.g., ``train_config.json``) is located in ``configs/agents/rl/push_cube``:

.. dropdown:: Example: train_config.json
   :icon: code

   .. literalinclude:: ../../../configs/agents/rl/push_cube/train_config.json
      :language: json
      :linenos:

Configuration Sections
---------------------

Runtime Settings
^^^^^^^^^^^^^^^^

The ``trainer`` section controls experiment setup:

- **exp_name**: Experiment name (used for output directories)
- **seed**: Random seed for reproducibility
- **device**: Runtime device string, e.g. ``"cpu"`` or ``"cuda:0"``
- **headless**: Whether to run simulation in headless mode
- **iterations**: Number of training iterations
- **buffer_size**: Steps collected per rollout (e.g., 1024)
- **eval_freq**: Frequency of evaluation (in steps)
- **save_freq**: Frequency of checkpoint saving (in steps)
- **use_wandb**: Whether to enable Weights & Biases logging (set in JSON config)
- **wandb_project_name**: Weights & Biases project name

Environment Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``env`` section defines the task environment:

- **id**: Environment registry ID (e.g., "PushCubeRL")
- **cfg**: Environment-specific configuration parameters

For RL environments, use the ``actions`` field for action preprocessing and ``extensions`` for task-specific parameters:

- **actions**: Action Manager config (e.g., DeltaQposTerm with scale)
- **extensions**: Task-specific parameters (e.g., success_threshold)

Example:

.. code-block:: json

   "env": {
     "id": "PushCubeRL",
     "cfg": {
       "num_envs": 4,
       "actions": {
         "delta_qpos": {
           "func": "DeltaQposTerm",
           "params": { "scale": 0.1 }
         }
       },
       "extensions": {
         "success_threshold": 0.1
       }
     }
   }

Policy Configuration
^^^^^^^^^^^^^^^^^^^

The ``policy`` section defines the neural network policy:

- **name**: Policy name (e.g., "actor_critic", "vla")
- **action_dim**: Optional policy output action dimension. If omitted, it is inferred from ``env.action_space``.
- **actor**: Actor network configuration (required for actor_critic)
- **critic**: Critic network configuration (required for actor_critic)

Example:

.. code-block:: json

   "policy": {
     "name": "actor_critic",
     "actor": {
       "type": "mlp",
       "network_cfg": {
         "hidden_sizes": [256, 256],
         "activation": "relu"
       }
     },
     "critic": {
       "type": "mlp",
       "network_cfg": {
         "hidden_sizes": [256, 256],
         "activation": "relu"
       }
     }
   }

Algorithm Configuration
^^^^^^^^^^^^^^^^^^^^^^^

The ``algorithm`` section defines the RL algorithm:

- **name**: Algorithm name (e.g., "ppo", "grpo")
- **cfg**: Algorithm-specific hyperparameters

PPO example:

.. code-block:: json

   "algorithm": {
     "name": "ppo",
     "cfg": {
       "learning_rate": 0.0001,
       "n_epochs": 10,
       "batch_size": 64,
       "gamma": 0.99,
       "gae_lambda": 0.95,
       "clip_coef": 0.2,
       "ent_coef": 0.01,
       "vf_coef": 0.5,
       "max_grad_norm": 0.5
     }
   }

GRPO example (for Embodied AI / from-scratch training, e.g. CartPole):

.. code-block:: json

   "algorithm": {
     "name": "grpo",
     "cfg": {
       "learning_rate": 0.0001,
       "n_epochs": 10,
       "batch_size": 8192,
       "gamma": 0.99,
       "clip_coef": 0.2,
       "ent_coef": 0.001,
       "kl_coef": 0,
       "group_size": 4,
       "eps": 1e-8,
       "reset_every_rollout": true,
       "max_grad_norm": 0.5,
       "truncate_at_first_done": true
     }
   }

For GRPO: use ``actor_only`` policy. Set ``kl_coef=0`` for from-scratch training; ``kl_coef=0.02`` for VLA/LLM fine-tuning.

Training Script
~~~~~~~~~~~~~~~

The training script (``train.py``) is located in ``embodichain/agents/rl/``:

.. dropdown:: Code for train.py
   :icon: code

   .. literalinclude:: ../../../embodichain/agents/rl/train.py
      :language: python
      :linenos:

The Script Explained
--------------------

The training script performs the following steps:

1. **Parse Configuration**: Loads JSON config and extracts runtime/env/policy/algorithm blocks
2. **Setup**: Initializes device, seeds, output directories, TensorBoard, and Weights & Biases
3. **Build Components**:
   - Environment via ``build_env()`` factory
   - Policy via ``build_policy()`` registry
   - Algorithm via ``build_algo()`` factory
4. **Create Trainer**: Instantiates the ``Trainer`` with all components
5. **Train**: Runs the training loop until completion

Launching Training
------------------

To start training, run:

.. code-block:: bash

   python -m embodichain.agents.rl.train --config configs/agents/rl/push_cube/train_config.json

Outputs
-------

All outputs are written to ``./outputs/<exp_name>_<timestamp>/``:

- **logs/**: TensorBoard logs
- **checkpoints/**: Model checkpoints

Training Process
~~~~~~~~~~~~~~~

The training process follows this sequence:

1. **Rollout Phase**: ``SyncCollector`` interacts with the environment and writes policy-side fields into a shared rollout ``TensorDict``. ``EmbodiedEnv`` writes environment-side step fields such as ``next.reward``, ``next.done``, ``next.terminated``, and ``next.truncated`` into the same rollout via ``set_rollout_buffer()``.
2. **Advantage/Return Computation**: Algorithm computes advantages and returns from the collected rollout (e.g. GAE for PPO, step-wise group normalization for GRPO)
3. **Update Phase**: Algorithm updates the policy with ``update(rollout)``
4. **Logging**: Trainer logs training losses and aggregated metrics to TensorBoard and Weights & Biases
5. **Evaluation** (periodic): Trainer evaluates the current policy
6. **Checkpointing** (periodic): Trainer saves model checkpoints

Policy Interface
~~~~~~~~~~~~~~~~

All policies must inherit from the ``Policy`` abstract base class:

.. code-block:: python

   from abc import ABC, abstractmethod
   import torch.nn as nn
   
   class Policy(nn.Module, ABC):
       device: torch.device
       
       def get_action(self, tensordict, deterministic: bool = False):
           """Samples action, sample_log_prob, and value into the TensorDict."""
           ...
       
       @abstractmethod
       def forward(self, tensordict, deterministic: bool = False):
           """Writes action, sample_log_prob, and value into the TensorDict."""
           raise NotImplementedError
       
       @abstractmethod
       def get_value(self, tensordict):
           """Writes value estimate into the TensorDict."""
           raise NotImplementedError
       
       @abstractmethod
       def evaluate_actions(self, tensordict):
           """Returns a new TensorDict with log_prob, entropy, and value."""
           raise NotImplementedError

Available Policies
------------------

- **ActorCritic**: MLP-based Gaussian policy with learnable log_std. Requires external ``actor`` and ``critic`` modules to be provided (defined in JSON config). Used with PPO.
- **ActorOnly**: Actor-only policy without Critic. Used with GRPO (group-relative advantage estimation).
- **VLAPlaceholderPolicy**: Placeholder for Vision-Language-Action policies

Algorithms
~~~~~~~~~~

Available Algorithms
--------------------

- **PPO**: Proximal Policy Optimization with GAE
- **GRPO**: Group Relative Policy Optimization (no Critic, step-wise returns, masked group normalization). Use ``actor_only`` policy. Set ``kl_coef=0`` for from-scratch training (CartPole, dense reward); ``kl_coef=0.02`` for VLA/LLM fine-tuning.

Adding a New Algorithm
---------------------

To add a new algorithm:

1. Create a new algorithm class in ``embodichain/agents/rl/algo/``
2. Implement ``update(rollout)`` and consume the shared rollout ``TensorDict``
3. Register in ``algo/__init__.py``:

.. code-block:: python

   from tensordict import TensorDict
   from embodichain.agents.rl.algo import BaseAlgorithm, register_algo
   
   @register_algo("my_algo")
   class MyAlgorithm(BaseAlgorithm):
       def __init__(self, cfg, policy):
           self.cfg = cfg
           self.policy = policy
           self.device = torch.device(cfg.device)
       
       def update(self, rollout: TensorDict):
           """Update the policy using a collected rollout."""
           # compute advantages / returns from rollout
           # optimize policy parameters
           return {"loss": 0.0}

Adding a New Policy
--------------------

To add a new policy:

1. Create a new policy class inheriting from the ``Policy`` abstract base class
2. Register in ``models/__init__.py``:

.. code-block:: python

   from embodichain.agents.rl.models import register_policy, Policy
   
   @register_policy("my_policy")
   class MyPolicy(Policy):
       def __init__(self, obs_dim, action_dim, device, config):
           super().__init__()
           self.device = device
           # Initialize your networks here
       
       def get_action(self, tensordict, deterministic=False):
           ...
       def forward(self, tensordict, deterministic=False):
           ...
       def get_value(self, tensordict):
           ...
       def evaluate_actions(self, tensordict):
           ...

Current built-in MLP policies use flattened observations in the training path. If your policy requires structured or multi-modal inputs, keep the richer ``obs_space`` interface and define a matching rollout/collector schema.

Adding a New Environment
------------------------

To add a new RL environment:

1. Create an environment class inheriting from ``EmbodiedEnv`` (with Action Manager configured for action preprocessing and standardized info structure):

.. code-block:: python

   from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
   from embodichain.lab.gym.utils.registration import register_env
   import torch
   
   @register_env("MyTaskRL", override=True)
   class MyTaskEnv(EmbodiedEnv):
       def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
           super().__init__(cfg, **kwargs)
       
       def compute_task_state(self, **kwargs):
           """Compute success/failure conditions and metrics."""
           is_success = ...  # Define success condition
           is_fail = torch.zeros_like(is_success)
           metrics = {"distance": ..., "error": ...}
           return is_success, is_fail, metrics


2. Configure the environment in your JSON config with ``actions`` and ``extensions``:

.. code-block:: json

   "env": {
     "id": "MyTaskRL",
     "cfg": {
       "num_envs": 4,
       "actions": {
         "delta_qpos": {
           "func": "DeltaQposTerm",
           "params": { "scale": 0.1 }
         }
       },
       "extensions": {
         "success_threshold": 0.05
       }
     }
   }

The ``EmbodiedEnv`` with Action Manager provides:

- **Action Preprocessing**: Configurable via ``actions`` (DeltaQposTerm, QposTerm, EefPoseTerm, etc.)
- **Standardized Info**: Implements ``get_info()`` using ``compute_task_state()`` template method

Best Practices
~~~~~~~~~~~~~~

- **Use EmbodiedEnv with Action Manager for RL Tasks**: Inherit from ``EmbodiedEnv`` and configure ``actions`` in your config. The Action Manager handles action preprocessing (delta_qpos, qpos, qvel, qf, eef_pose) in a modular way.

- **Action Configuration**: Use the ``actions`` field in your JSON config. Example: ``"delta_qpos": {"func": "DeltaQposTerm", "params": {"scale": 0.1}}``.

- **Device Management**: Device is single-sourced from ``runtime.cuda``. All components (trainer/algorithm/policy/env) share the same device.

- **Observation Format**: Environments should provide consistent observation shape/types (torch.float32) and a single ``done = terminated | truncated``.

- **Algorithm Interface**: Algorithms implement ``update(rollout)`` and consume a shared rollout ``TensorDict``. Collection is handled by ``SyncCollector`` plus environment-side rollout writes in ``EmbodiedEnv``.

- **Reward Configuration**: Use the ``RewardManager`` in your environment config to define reward components. Organize reward components in ``info["rewards"]`` dictionary and metrics in ``info["metrics"]`` dictionary. The trainer performs dense per-step logging directly from environment info.

- **Template Methods**: Override ``compute_task_state()`` to define success/failure conditions and metrics. Override ``check_truncated()`` for custom truncation logic.

- **Configuration**: Use JSON for all hyperparameters. This makes experiments reproducible and easy to track.

- **Logging**: Metrics are automatically logged to TensorBoard and Weights & Biases. Check ``outputs/<exp_name>/logs/`` for TensorBoard logs.

- **Checkpoints**: Regular checkpoints are saved to ``outputs/<exp_name>/checkpoints/``. Use these to resume training or evaluate policies.

