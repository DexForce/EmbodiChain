Locomotion robot configurations
===============================

These presets construct the bundled locomotion models through the ordinary
``RobotCfg`` and ``Robot`` interfaces. Each provides a floating base, its
standing pose, the ``locomotion`` control group and the existing simulation PD
drive defaults. They do not provide limb IK solvers or external actuator networks.

.. currentmodule:: embodichain_robots

.. autosummary::

   UnitreeGo1Cfg.from_dict
   UnitreeGo2Cfg.from_dict
   UnitreeG1Cfg.from_dict
   UnitreeH12Cfg.from_dict
   ANYmalCCfg.from_dict
   MicroDuckCfg.from_dict

.. py:class:: UnitreeGo1Cfg

   Twelve-joint Go1 preset with the bundled USD asset and standing pose.

.. py:class:: UnitreeGo2Cfg

   Twelve-joint Go2 preset with the bundled USD asset and standing pose.

.. py:class:: UnitreeG1Cfg

   Twenty-nine-joint G1 preset with the bundled USD asset and standing pose.

.. py:class:: UnitreeH12Cfg

   Twenty-seven-joint H1_2 preset with the bundled USD asset and standing pose.

.. py:class:: ANYmalCCfg

   Twelve-joint ANYmal-C preset with its bundled USD asset and joint limits.

.. py:class:: MicroDuckCfg

   Fourteen-joint MicroDuck preset with the bundled URDF asset and standing pose.

Each class inherits :class:`embodichain.lab.sim.cfg.RobotCfg` fields. Its
``from_dict`` method builds the model defaults before merging explicit field
overrides. Use catalog resolution or ``from_dict({})`` for standalone creation.

Model descriptions without a simulation backend
-----------------------------------------------

The profile reader supplies independent copies to standalone robot configs and
pure Torch task contracts. Asset initial positions use source joint order;
the description's standing positions use its explicit joint-name order.
Task-specific action scales, observations, rewards and fall rules remain in
the task package. Explicit deployment drive overrides take precedence.

.. currentmodule:: embodichain_robots.locomotion_profiles

.. autosummary::

   load_locomotion_robot_profile

.. autofunction:: load_locomotion_robot_profile

Implementation module imports
-----------------------------

The same preset classes are available through their defining modules:

.. currentmodule:: embodichain_robots.anymal_c

.. py:class:: ANYmalCCfg

   Defining-module import of :class:`embodichain_robots.ANYmalCCfg`.

.. currentmodule:: embodichain_robots.microduck

.. py:class:: MicroDuckCfg

   Defining-module import of :class:`embodichain_robots.MicroDuckCfg`.

.. currentmodule:: embodichain_robots.unitree_g1

.. py:class:: UnitreeG1Cfg

   Defining-module import of :class:`embodichain_robots.UnitreeG1Cfg`.

.. currentmodule:: embodichain_robots.unitree_go1

.. py:class:: UnitreeGo1Cfg

   Defining-module import of :class:`embodichain_robots.UnitreeGo1Cfg`.

.. currentmodule:: embodichain_robots.unitree_go2

.. py:class:: UnitreeGo2Cfg

   Defining-module import of :class:`embodichain_robots.UnitreeGo2Cfg`.

.. currentmodule:: embodichain_robots.unitree_h1_2

.. py:class:: UnitreeH12Cfg

   Defining-module import of :class:`embodichain_robots.UnitreeH12Cfg`.
