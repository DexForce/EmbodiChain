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

The ``runtime`` section controls experiment setup:

- **exp_name**: Experiment name (used for output directories)
- **seed**: Random seed for reproducibility
- **cuda**: Whether to use GPU (default: true)
- **headless**: Whether to run simulation in headless mode
- **iterations**: Number of training iterations
- **rollout_steps**: Steps per rollout (e.g., 1024)
- **eval_freq**: Frequency of evaluation (in steps)
- **save_freq**: Frequency of checkpoint saving (in steps)
- **use_wandb**: Whether to enable Weights & Biases logging (set in JSON config)
- **wandb_project_name**: Weights & Biases project name

Environment Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``env`` section defines the task environment:

- **id**: Environment registry ID (e.g., "PushCubeRL")
- **cfg**: Environment-specific configuration parameters

Example:

.. code-block:: json

   "env": {
     "id": "PushCubeRL",
     "cfg": {
       "num_envs": 4,
       "obs_mode": "state",
       "episode_length": 100,
       "action_scale": 0.1,
       "success_threshold": 0.1
     }
   }

Policy Configuration
^^^^^^^^^^^^^^^^^^^

The ``policy`` section defines the neural network policy:

- **name**: Policy name (e.g., "actor_critic", "vla")
- **cfg**: Policy-specific hyperparameters (empty for actor_critic)
- **actor**: Actor network configuration (required for actor_critic)
- **critic**: Critic network configuration (required for actor_critic)

Example:

.. code-block:: json

   "policy": {
     "name": "actor_critic",
     "cfg": {},
     "actor": {
       "type": "mlp",
       "hidden_sizes": [256, 256],
       "activation": "relu"
     },
     "critic": {
       "type": "mlp",
       "hidden_sizes": [256, 256],
       "activation": "relu"
     }
   }

Algorithm Configuration
^^^^^^^^^^^^^^^^^^^^^^^

The ``algorithm`` section defines the RL algorithm:

- **name**: Algorithm name (e.g., "ppo")
- **cfg**: Algorithm-specific hyperparameters

Example:

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

1. **Rollout Phase**: Algorithm collects trajectories by interacting with the environment (via ``collect_rollout``). During this phase, the trainer performs dense per-step logging of rewards and metrics from environment info.
2. **GAE Computation**: Algorithm computes advantages and returns using Generalized Advantage Estimation (internal to algorithm, stored in buffer extras)
3. **Update Phase**: Algorithm updates the policy using collected data (e.g., PPO)
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
       
       @abstractmethod
       def get_action(
           self, obs: torch.Tensor, deterministic: bool = False
       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
           """Returns (action, log_prob, value)"""
           raise NotImplementedError
       
       @abstractmethod
       def get_value(self, obs: torch.Tensor) -> torch.Tensor:
           """Returns value estimate"""
           raise NotImplementedError
       
       @abstractmethod
       def evaluate_actions(
           self, obs: torch.Tensor, actions: torch.Tensor
       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
           """Returns (log_prob, entropy, value)"""
           raise NotImplementedError

Available Policies
------------------

- **ActorCritic**: MLP-based Gaussian policy with learnable log_std. Requires external ``actor`` and ``critic`` modules to be provided (defined in JSON config).
- **VLAPlaceholderPolicy**: Placeholder for Vision-Language-Action policies

Algorithms
~~~~~~~~~~

Available Algorithms
--------------------

- **PPO**: Proximal Policy Optimization with GAE

Adding a New Algorithm
---------------------

To add a new algorithm:

1. Create a new algorithm class in ``embodichain/agents/rl/algo/``
2. Implement ``initialize_buffer()``, ``collect_rollout()``, and ``update()`` methods
3. Register in ``algo/__init__.py``:

.. code-block:: python

   from embodichain.agents.rl.algo import BaseAlgorithm, register_algo
   from embodichain.agents.rl.buffer import RolloutBuffer
   
   @register_algo("my_algo")
   class MyAlgorithm(BaseAlgorithm):
       def __init__(self, cfg, policy):
           self.cfg = cfg
           self.policy = policy
           self.device = torch.device(cfg.device)
           self.buffer = None
       
       def initialize_buffer(self, num_steps, num_envs, obs_dim, action_dim):
           """Initialize the algorithm's buffer."""
           self.buffer = RolloutBuffer(num_steps, num_envs, obs_dim, action_dim, self.device)
       
       def collect_rollout(self, env, policy, obs, num_steps, on_step_callback=None):
           """Control data collection process (interact with env, fill buffer, compute advantages/returns)."""
           # Collect trajectories
           # Compute advantages/returns (e.g., GAE for on-policy algorithms)
           # Attach extras to buffer: self.buffer.set_extras({"advantages": adv, "returns": ret})
           # Return empty dict (dense logging handled in trainer)
           return {}
       
       def update(self):
           """Update the policy using collected data."""
           # Access extras from buffer: self.buffer._extras.get("advantages")
           # Use self.buffer to update policy
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
       def __init__(self, obs_space, action_space, device, config):
           super().__init__()
           self.device = device
           # Initialize your networks here
       
       def get_action(self, obs, deterministic=False):
           ...
       def get_value(self, obs):
           ...
       def evaluate_actions(self, obs, actions):
           ...

Adding a New Environment
------------------------

To add a new RL environment:

1. Create an environment class inheriting from ``EmbodiedEnv``
2. Register it with the Gymnasium registry:

.. code-block:: python

   from embodichain.lab.gym.utils.registration import register_env
   
   @register_env("MyTaskRL", max_episode_steps=100, override=True)
   class MyTaskEnv(EmbodiedEnv):
       cfg: MyTaskEnvCfg
       ...

3. Use the environment ID in your JSON config:

.. code-block:: json

   "env": {
     "id": "MyTaskRL",
     "cfg": {
       ...
     }
   }

Best Practices
~~~~~~~~~~~~~~

- **Device Management**: Device is single-sourced from ``runtime.cuda``. All components (trainer/algorithm/policy/env) share the same device.

- **Action Scaling**: Keep action scaling in the environment, not in the policy.

- **Observation Format**: Environments should provide consistent observation shape/types (torch.float32) and a single ``done = terminated | truncated``.

- **Algorithm Interface**: Algorithms must implement ``initialize_buffer()``, ``collect_rollout()``, and ``update()`` methods. The algorithm completely controls data collection and buffer management.

- **Reward Components**: Organize reward components in ``info["rewards"]`` dictionary and metrics in ``info["metrics"]`` dictionary. The trainer performs dense per-step logging directly from environment info.

- **Configuration**: Use JSON for all hyperparameters. This makes experiments reproducible and easy to track.

- **Logging**: Metrics are automatically logged to TensorBoard and Weights & Biases. Check ``outputs/<exp_name>/logs/`` for TensorBoard logs.

- **Checkpoints**: Regular checkpoints are saved to ``outputs/<exp_name>/checkpoints/``. Use these to resume training or evaluate policies.

