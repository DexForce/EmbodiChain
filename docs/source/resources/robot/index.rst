Supported Robots
================

EmbodiChain provides configuration classes for the robot families and composed
layouts below. Concrete robots follow their manufacturer or community project
namespaces in ``embodichain_robots``. Each model page documents construction,
control parts, kinematics, and the currently supported variants.

Run ``embodichain list-robot`` to discover installed families, variants,
independent control DOFs, default control parts, and mimic followers without
loading simulation or downloading assets. See :ref:`cli-list-robots` for the
command reference and :doc:`/guides/robot_catalog` for variant selection.

Manufacturers
-------------

.. list-table::
   :class: robot-catalog
   :header-rows: 1
   :widths: 16 20 24 40

   * - Manufacturer
     - Robot
     - Configuration class
     - Current coverage
   * - :doc:`AgileX <agilex/index>`
     - :doc:`CobotMagic <agilex/cobotmagic>`
     - ``CobotMagicCfg``
     - Fixed-base dual arms and grippers for bimanual tasks.
   * - :doc:`ANYbotics <anybotics/index>`
     - :doc:`ANYmal-C <anybotics/anymal_c>`
     - ``ANYmalCCfg``
     - Twelve-joint floating-base quadruped with simulation PD defaults.
   * - :doc:`ARX <arx/index>`
     - :doc:`ARX5 X5A <arx/arx5>`
     - ``ARX5Cfg``
     - Six-joint arm with two independent parallel fingers and numerical IK.
   * - :doc:`DexForce <dexforce/index>`
     - :doc:`Dexforce W1 <dexforce/w1>`
     - ``DexforceW1Cfg``
     - Dual 7-DOF arms, versioned assets, and optional hands and grippers.
   * - :doc:`Flexiv <flexiv/index>`
     - :doc:`Flexiv Rizon 4S <flexiv/rizon4s>`
     - ``FlexivRizon4SCfg``
     - Seven-axis bare arm with numerical IK and published Flexiv simulation dynamics.
   * - :doc:`Franka <franka/index>`
     - :doc:`Franka Panda <franka/panda>`
     - ``FrankaPandaCfg``
     - Panda arm with the stock Panda hand and numerical FK/IK.
   * - :doc:`Pollen Robotics <pollen_robotics/index>`
     - :doc:`MicroDuck <pollen_robotics/microduck>`
     - ``MicroDuckCfg``
     - Fourteen-joint floating-base biped with simulation PD defaults.
   * - :doc:`The Robot Studio <the_robot_studio/index>`
     - :doc:`SO101 Follower <the_robot_studio/so101>`
     - ``SO101FollowerCfg``
     - Five-axis follower arm and actuated jaw with official simulation dynamics.
   * - :doc:`Tianji <tianji/index>`
     - :doc:`Tianji Marvin <tianji/marvin>`
     - ``TianjiMarvinCfg``
     - Dual 7-DOF arms with optional parallel-jaw grippers.
   * - :doc:`Unitree <unitree/index>`
     - :doc:`Unitree G1 <unitree/g1>`
     - ``UnitreeG1Cfg``
     - Twenty-nine-joint humanoid with its standing pose and simulation PD defaults.
   * - :doc:`Unitree <unitree/index>`
     - :doc:`Unitree Go1 <unitree/go1>`
     - ``UnitreeGo1Cfg``
     - Twelve-joint floating-base quadruped with simulation PD defaults.
   * - :doc:`Unitree <unitree/index>`
     - :doc:`Unitree Go2 <unitree/go2>`
     - ``UnitreeGo2Cfg``
     - Twelve-joint floating-base quadruped with simulation PD defaults.
   * - :doc:`Unitree <unitree/index>`
     - :doc:`Unitree H1_2 <unitree/h1_2>`
     - ``UnitreeH12Cfg``
     - Twenty-seven-joint humanoid with its standing pose and simulation PD defaults.
   * - :doc:`Universal Robots <universal_robots/index>`
     - :doc:`UR family <universal_robots/ur_robot>`
     - ``URRobotCfg``
     - UR3, UR3e, UR5, UR5e, UR10, and UR10e with analytic IK.

.. toctree::
   :maxdepth: 2
   :caption: Manufacturers
   :hidden:

   agilex/index
   anybotics/index
   arx/index
   dexforce/index
   flexiv/index
   franka/index
   pollen_robotics/index
   the_robot_studio/index
   tianji/index
   unitree/index
   universal_robots/index

Community Projects
------------------

AlohaMini follows its community project's namespace. Its manufacturer is
currently unknown in the robot catalog.

.. list-table::
   :class: robot-catalog
   :header-rows: 1
   :widths: 16 20 24 40

   * - Project
     - Robot
     - Configuration class
     - Current coverage
   * - :doc:`AlohaMini <alohamini/index>`
     - :doc:`Aloha Mini 2 Pro <alohamini/aloha_mini>`
     - ``AlohaMiniCfg``
     - Mobile dual-arm platform with torso, arms, and grippers.

.. toctree::
   :maxdepth: 2
   :caption: Community Projects
   :hidden:

   alohamini/index

Robot Composition
-----------------

Reusable composition belongs to the core ``embodichain.lab.sim.robots``
package. It builds pairs from compatible single-arm presets and is shared
across manufacturers.

.. list-table::
   :class: robot-catalog
   :header-rows: 1
   :widths: 24 25 51

   * - Composition
     - Configuration class
     - Current coverage
   * - :doc:`Dual-arm composition <dual_arm>`
     - ``DualArmRobotCfg``
     - Registry-based composition of compatible single-arm configs in multiple layouts.

.. toctree::
   :maxdepth: 1
   :caption: Robot Composition
   :hidden:

   dual_arm

Locomotion Defaults
-------------------

See :doc:`locomotion` for shared standing poses, drive defaults, task integration,
and render settings. Individual robot pages are listed under their manufacturers.

.. toctree::
   :maxdepth: 1
   :caption: Locomotion Defaults
   :hidden:

   locomotion

To add another model, follow :doc:`/guides/add_robot` and include its page in
the owning manufacturer or project index and this overview.
