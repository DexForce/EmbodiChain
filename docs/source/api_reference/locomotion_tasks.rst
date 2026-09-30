Locomotion tasks
================

Task definitions, tensor contracts and downloadable robot assets for the locomotion environments. See :doc:`/overview/rl/locomotion` for configuration, data preparation and controls.

Package entry points
--------------------

The Humanoid package exports its effort-controlled running environment. The
task manager package exports actor/critic observation builders and reward
functions. Joint-position actions and root-velocity disturbances are standard
EmbodiChain manager components configured by these tasks.

.. autosummary::

   embodichain_tasks.classic_control.humanoid.HumanoidRunEnv
   embodichain.lab.gym.envs.managers.actions.DefaultJointPositionAction
   embodichain.lab.gym.envs.managers.randomization.physics.push_articulation_by_setting_velocity
   embodichain_tasks.locomotion.managers.velocity_locomotion_observation
   embodichain_tasks.locomotion.managers.velocity_locomotion_reward
   embodichain_tasks.locomotion.managers.velocity_locomotion_total_reward

Robot configuration loaders
---------------------------

Each velocity package exports a configuration type and ``load_config()`` for
its packaged ``task.json``. These loaders preserve the robot's joint order,
action scale, observation dimensions and control timing.

.. autosummary::

   embodichain_tasks.locomotion.velocity.contracts.anymal_c.ANYmalCVelocityConfig
   embodichain_tasks.locomotion.velocity.contracts.anymal_c.load_config
   embodichain_tasks.locomotion.velocity.contracts.g1.G1VelocityConfig
   embodichain_tasks.locomotion.velocity.contracts.g1.load_config
   embodichain_tasks.locomotion.velocity.contracts.go1.Go1VelocityConfig
   embodichain_tasks.locomotion.velocity.contracts.go1.load_config
   embodichain_tasks.locomotion.velocity.contracts.go2.Go2VelocityConfig
   embodichain_tasks.locomotion.velocity.contracts.go2.load_config
   embodichain_tasks.locomotion.velocity.contracts.h1_2.H12VelocityConfig
   embodichain_tasks.locomotion.velocity.contracts.h1_2.load_config
   embodichain_tasks.locomotion.velocity.contracts.microduck.MicroDuckVelocityConfig
   embodichain_tasks.locomotion.velocity.contracts.microduck.load_config

Implementation reference
------------------------

.. automodule:: embodichain_tasks.locomotion.managers.observations
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.managers.rewards
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.anymal_c_flat
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.contracts.anymal_c.config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.contracts.anymal_c.mdp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.contracts.g1.config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.contracts.g1.mdp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.contracts.go1.config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.contracts.go1.mdp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.contracts.go2.config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.contracts.go2.mdp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.contracts.h1_2.config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.contracts.h1_2.mdp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.contracts.microduck.config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.contracts.microduck.mdp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.g1_flat
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.go1_flat
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.go2_flat
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.h1_2_flat
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.locomotion.velocity.microduck_flat
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.classic_control.humanoid.humanoid_run
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain_tasks.classic_control.humanoid.mdp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain.data.assets.locomotion_assets
   :members:
   :undoc-members:
   :show-inheritance:
