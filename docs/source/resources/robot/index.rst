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

Parallel-jaw grippers share physical defaults between standalone robots and
mounted embodiments. See :doc:`parallel_jaw_grippers` for catalog IDs, binding
syntax, command ownership and the TCP resolution contract.

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
   * - :doc:`AgiBot <agibot/index>`
     - :doc:`G1 <agibot/g1>`
     - ``AgibotG1Cfg``
     - Fixed-base USD robot with dual 7-DOF arms and closed-chain grippers on the Default backend.
   * - :doc:`AgiBot <agibot/index>`
     - :doc:`G2 <agibot/g2>`
     - ``AgibotG2Cfg``
     - Wheeled humanoid with dual 7-DOF CRS arms, OmniPicker grippers and Warp SSIK on CPU/CUDA.
   * - :doc:`AgileX <agilex/index>`
     - :doc:`CobotMagic <agilex/cobotmagic>`
     - ``CobotMagicCfg``
     - Fixed-base dual arms and grippers for bimanual tasks.
   * - :doc:`AgileX <agilex/index>`
     - :doc:`PiPER-X <agilex/piper_x>`
     - ``AgilexPiperXCfg``
     - Fixed-base six-axis arm with a separately mounted AGX tool.
   * - :doc:`ANYbotics <anybotics/index>`
     - :doc:`ANYmal-C <anybotics/anymal_c>`
     - ``ANYmalCCfg``
     - Twelve-joint floating-base quadruped with simulation PD defaults.
   * - :doc:`ARX <arx/index>`
     - :doc:`ARX5 X5A <arx/arx5>`
     - ``ARX5Cfg``
     - Six-joint arm with two independent parallel fingers, Warp near-parallel IK on CPU/CUDA and optional SSIK reference alignment.
   * - :doc:`BrainCo <brainco/index>`
     - :doc:`Revo 1 <brainco/revo1>`
     - ``BrainCoRevo1Cfg``
     - Standalone left/right hand with six joint commands and four source mimic followers.
   * - :doc:`Wuji Technology <wuji/index>`
     - :doc:`Wuji Hand <wuji/hand>`
     - ``WujiHandCfg``
     - Standalone left/right hand with 20 independent source joints.
   * - :doc:`Sharpa Robotics <sharpa/index>`
     - :doc:`Sharpa Wave 01 <sharpa/wave>`
     - ``SharpaWaveCfg``
     - Standalone left/right hand with 22 independent source joints.
   * - :doc:`DexForce <dexforce/index>`
     - :doc:`Dexforce W1 <dexforce/w1>`
     - ``DexforceW1Cfg``
     - Dual 7-DOF arms, versioned assets, and optional hands and grippers.
   * - :doc:`Flexiv <flexiv/index>`
     - :doc:`Flexiv Rizon 4S <flexiv/rizon4s>`
     - ``FlexivRizon4SCfg``
     - Seven-axis bare arm with Warp RR IK on CPU/CUDA, optional SSIK reference alignment and published Flexiv simulation dynamics.
   * - :doc:`Franka <franka/index>`
     - :doc:`Franka arms <franka/cfg>`
     - ``FrankaCfg``
     - Seven-DOF bare FR3 or original Panda arm; flange TCP and explicit tool assembly.
   * - :doc:`Franka <franka/index>`
     - :doc:`Franka Panda <franka/panda>`
     - ``FrankaPandaCfg``
     - Legacy FR3 bare-arm selector; native-hand deployments use the explicit ``franka_panda`` embodiment.
   * - :doc:`I2RT Robotics <i2rt/index>`
     - :doc:`YAM dual arms <i2rt/yam>`
     - ``YamI2rtCfg``
     - Two six-joint arms with four independent finger joints and per-arm numerical FK/IK.
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
     - Dual 7-DOF arms with optional parallel-jaw grippers and Warp SRS IK on CPU/CUDA.
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

   parallel_jaw_grippers
   :caption: Manufacturers
   :hidden:

   agibot/index
   agilex/index
   anybotics/index
   arx/index
   brainco/index
   sharpa/index
   wuji/index
   changingtek/index
   dh_robotics/index
   inspire/index
   robotiq/index
   wheeltec/index
   dexforce/index
   flexiv/index
   franka/index
   i2rt/index
   pollen_robotics/index
   the_robot_studio/index
   tianji/index
   unitree/index
   agibot/index
   universal_robots/index

End-effector Assets
-------------------

Hands and grippers share the manufacturer/model organization above.
BrainCo Revo 1, Sharpa Wave 01 and Wuji Hand provide standalone hand presets;
asset-only
entries provide source assets for caller-owned assembly. ``eef`` is a control-group
convention and does not identify grasping capabilities.

See :doc:`end_effectors` for the asset IDs, canonical imports, migration
instructions, provenance boundaries and rendered source-model illustrations.

.. list-table::
   :class: robot-catalog
   :header-rows: 1
   :widths: 25 40 35

   * - Manufacturer
     - Assets
     - Coverage
   * - :doc:`AgileX <agilex/index>`
     - :doc:`Pika Gripper <agilex/pika>`, :doc:`AGX Gripper <agilex/piper_x>`
     - Gripper asset; host-owned mounting.
   * - :doc:`BrainCo <brainco/index>`
     - :doc:`Revo 1 <brainco/revo1>`
     - Left/right assets and standalone preset.
   * - :doc:`Wuji Technology <wuji/index>`
     - :doc:`Wuji Hand <wuji/hand>`
     - Left/right assets and standalone preset.
   * - :doc:`Sharpa Robotics <sharpa/index>`
     - :doc:`Sharpa Wave 01 <sharpa/wave>`
     - Left/right assets and standalone preset.
   * - :doc:`ChangingTek <changingtek/index>`
     - :doc:`CTM2F110 <changingtek/ctm2f110>`
     - Gripper asset.
   * - :doc:`DH-Robotics <dh_robotics/index>`
     - :doc:`PGC-140-50 <dh_robotics/pgc_140_50>`, :doc:`PGI-140-80 <dh_robotics/pgi_140_80>`, :doc:`AG95 <dh_robotics/ag95>`
     - Four archives; modified assets documented separately.
   * - :doc:`INSPIRE ROBOTS <inspire/index>`
     - :doc:`Hand assets <inspire/hand>`
     - Left/right assets; exact product revision unverified.
   * - :doc:`Robotiq <robotiq/index>`
     - :doc:`Shared archive <robotiq/assets>`, :doc:`2F-85 archive <robotiq/robotiq_2f85>`
     - Three source models in two independent archives.
   * - :doc:`WHEELTEC <wheeltec/index>`
     - :doc:`FA2F asset <wheeltec/fa2f>`
     - Brand known; formal model identity unverified.

.. toctree::
   :maxdepth: 1
   :hidden:

   end_effectors

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
     - Mobile dual-arm platform with torso, grippers and Warp IK for both arms on CPU/CUDA.

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
