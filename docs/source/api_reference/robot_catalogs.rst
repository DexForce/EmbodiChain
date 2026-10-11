Robot catalogs and configurations
=================================

The main ``embodichain`` distribution bundles ``embodichain_robots`` with the
same release version. Concrete robot defaults and W1 hardware specifications
live in that import package; generic simulation, solver, and dual-arm APIs
remain in the core.

Gym robot selectors, workspace analysis, and dual-arm base selection share a
catalog resolver. Catalog entry points advertise robot IDs, configuration class
import paths, and optional defaults. Listing catalogs defers robot class imports
and simulation initialization; resolution validates the selected class and
isolates its default/override fields.

Official configurations provide the existing ``RobotCfg.from_dict()`` and
``build_pk_serial_chain()`` contracts. Legacy import aliases retain the same
class, enum, and module objects. Shared runtime fields remain documented on
``RobotCfg``; the class declarations below identify canonical imports without
re-expanding inherited configuration fields or generated constructors.


embodichain.robot_registry
--------------------------

Discover installed robot IDs or construct the selected configuration with caller overrides.

.. autofunction:: embodichain.robot_registry.list_robot_cfgs

.. autofunction:: embodichain.robot_registry.resolve_robot_cfg


embodichain_robots
------------------

Concrete robot configurations, exposed lazily from the bundled robot package.

.. py:class:: embodichain_robots.AlohaMiniCfg
   :no-index:

   Configure Aloha Mini 2 Pro from its complete, source-named URDF.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.

.. py:class:: embodichain_robots.TianjiMarvinCfg
   :no-index:

   Configure Tianji Marvin with or without its parallel-jaw grippers.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.

.. py:class:: embodichain_robots.DexforceW1Cfg
   :no-index:

   DexforceW1 specific configuration, inherits from RobotCfg and allows custom parameters.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.

.. py:class:: embodichain_robots.CobotMagicCfg
   :no-index:

   Dual-arm CobotMagic with source arm limits and calibrated gripper drives.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.

.. py:class:: embodichain_robots.FrankaPandaCfg
   :no-index:

   Configuration for the Franka Emika Panda robot with Panda hand.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.

.. py:class:: embodichain_robots.URRobotCfg
   :no-index:

   Configuration for the UR family of robots.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain_robots.aloha_mini
-----------------------------

Concrete robot configurations, exposed lazily from the bundled robot package.

.. py:class:: embodichain_robots.aloha_mini.AlohaMiniCfg
   :no-index:

   Configure Aloha Mini 2 Pro from its complete, source-named URDF.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain_robots.cobotmagic
-----------------------------

Concrete robot configurations, exposed lazily from the bundled robot package.

.. py:class:: embodichain_robots.cobotmagic.CobotMagicCfg
   :no-index:

   Dual-arm CobotMagic with source arm limits and calibrated gripper drives.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain_robots.dexforce_w1
------------------------------

W1 configuration, hardware revisions, calibrated transforms, hand specifications, and assembly helpers.

.. py:class:: embodichain_robots.dexforce_w1.DexforceW1Cfg
   :no-index:

   DexforceW1 specific configuration, inherits from RobotCfg and allows custom parameters.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain_robots.dexforce_w1.cfg
----------------------------------

W1 configuration, hardware revisions, calibrated transforms, hand specifications, and assembly helpers.

.. py:class:: embodichain_robots.dexforce_w1.cfg.DexforceW1Cfg
   :no-index:

   DexforceW1 specific configuration, inherits from RobotCfg and allows custom parameters.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain_robots.dexforce_w1.hand_specs
-----------------------------------------

W1 configuration, hardware revisions, calibrated transforms, hand specifications, and assembly helpers.

.. py:class:: embodichain_robots.dexforce_w1.hand_specs.W1HandSideSpec

   Side-specific asset metadata for one hand release.

.. py:class:: embodichain_robots.dexforce_w1.hand_specs.W1HandSpec

   A hand release independent of the W1 body/arm version.

.. autofunction:: embodichain_robots.dexforce_w1.hand_specs.get_default_w1_hand_version

.. autofunction:: embodichain_robots.dexforce_w1.hand_specs.get_w1_hand_spec

.. autofunction:: embodichain_robots.dexforce_w1.hand_specs.normalize_w1_hand_mappings


embodichain_robots.dexforce_w1.params
-------------------------------------

W1 configuration, hardware revisions, calibrated transforms, hand specifications, and assembly helpers.

.. py:class:: embodichain_robots.dexforce_w1.params.W1ArmKineParams

   Kinematics parameters for a W1 arm.


embodichain_robots.dexforce_w1.specs
------------------------------------

W1 configuration, hardware revisions, calibrated transforms, hand specifications, and assembly helpers.

.. py:class:: embodichain_robots.dexforce_w1.specs.W1VersionSpec

   Asset layout and calibrated defaults belonging to one W1 revision.

.. autofunction:: embodichain_robots.dexforce_w1.specs.get_w1_version_spec


embodichain_robots.dexforce_w1.types
------------------------------------

W1 configuration, hardware revisions, calibrated transforms, hand specifications, and assembly helpers.

.. py:class:: embodichain_robots.dexforce_w1.types.DexforceW1Version

   Released W1 body revision identifiers.

.. py:class:: embodichain_robots.dexforce_w1.types.DexforceW1HandVersion

   W1 end-effector revision identifiers, independent of the body revision.

.. py:class:: embodichain_robots.dexforce_w1.types.DexforceW1ArmSide

   Left/right side identifiers for W1 assembly and calibration.

.. py:class:: embodichain_robots.dexforce_w1.types.DexforceW1Type

   Physical W1 component identifiers used by the assembly helpers.

.. py:class:: embodichain_robots.dexforce_w1.types.DexforceW1HandBrand

   Supported W1 end-effector model identifiers.


embodichain_robots.dexforce_w1.utils
------------------------------------

W1 configuration, hardware revisions, calibrated transforms, hand specifications, and assembly helpers.

.. py:class:: embodichain_robots.dexforce_w1.utils.ChassisManager

   Robot hardware configuration and assembly helper.

.. py:class:: embodichain_robots.dexforce_w1.utils.TorsoManager

   Robot hardware configuration and assembly helper.

.. py:class:: embodichain_robots.dexforce_w1.utils.HeadManager

   Robot hardware configuration and assembly helper.

.. py:class:: embodichain_robots.dexforce_w1.utils.ArmManager

   Robot hardware configuration and assembly helper.

.. py:class:: embodichain_robots.dexforce_w1.utils.HandManager

   Robot hardware configuration and assembly helper.

.. py:class:: embodichain_robots.dexforce_w1.utils.EyesManager

   Robot hardware configuration and assembly helper.

.. autofunction:: embodichain_robots.dexforce_w1.utils.build_dexforce_w1_assembly_urdf_cfg


embodichain_robots.franka_panda
-------------------------------

Concrete robot configurations, exposed lazily from the bundled robot package.

.. py:class:: embodichain_robots.franka_panda.FrankaPandaCfg
   :no-index:

   Configuration for the Franka Emika Panda robot with Panda hand.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain_robots.tianji_marvin
--------------------------------

Concrete robot configurations, exposed lazily from the bundled robot package.

.. py:class:: embodichain_robots.tianji_marvin.TianjiMarvinCfg
   :no-index:

   Configure Tianji Marvin with or without its parallel-jaw grippers.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain_robots.ur_robot
---------------------------

Concrete robot configurations, exposed lazily from the bundled robot package.

.. py:class:: embodichain_robots.ur_robot.URRobotCfg
   :no-index:

   Configuration for the UR family of robots.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain.lab.sim.robots.dexforce_w1
--------------------------------------

Compatibility imports for existing W1 configuration and kinematic-parameter consumers.

.. py:class:: embodichain.lab.sim.robots.dexforce_w1.DexforceW1Cfg
   :no-index:

   DexforceW1 specific configuration, inherits from RobotCfg and allows custom parameters.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain.lab.sim.robots.dexforce_w1.cfg
------------------------------------------

Compatibility imports for existing W1 configuration and kinematic-parameter consumers.

.. py:class:: embodichain.lab.sim.robots.dexforce_w1.cfg.DexforceW1Cfg
   :no-index:

   DexforceW1 specific configuration, inherits from RobotCfg and allows custom parameters.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain.lab.sim.robots.dexforce_w1.params
---------------------------------------------

Compatibility imports for existing W1 configuration and kinematic-parameter consumers.

.. py:class:: embodichain.lab.sim.robots.dexforce_w1.params.W1ArmKineParams
   :no-index:

   Kinematics parameters for a W1 arm.
