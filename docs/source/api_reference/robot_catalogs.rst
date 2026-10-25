Robot catalogs and configurations
=================================

The main ``embodichain`` distribution bundles ``embodichain_robots`` with the
same release version. Concrete robot defaults and W1 hardware specifications
live in that import package; generic simulation, solver, and dual-arm APIs
remain in the core.

Gym robot selectors, workspace analysis, and dual-arm base selection share a
catalog resolver. Catalog entry points advertise robot IDs, configuration class
import paths, optional defaults, classification, and control/variant summaries.
``embodichain list-robot`` reads these summaries to display independent control
DOFs, default control parts, mimic followers, vendor, morphology, and arm count. Listing defers robot class
imports and simulation initialization; resolution validates the selected class
and isolates its default/override fields. See :doc:`/guides/robot_catalog` for
the CLI, variant selection, and provider registration.

Official configurations provide the existing ``RobotCfg.from_dict()`` and
``build_pk_serial_chain()`` contracts. Import concrete configurations from
``embodichain_robots`` and generic dual-arm helpers from
``embodichain.lab.sim.robots``. Concrete configuration modules follow vendor/model ownership. Former flat
robot configuration modules and historical selector aliases have been removed.
The stock Panda hand is the sole official Panda preset; Panda/PGI assemblies
belong to the examples and benchmark fixtures that use them. Selectors use exact
registered IDs such as ``FrankaPanda``, ``URRobot``, or ``ur5``.
Shared runtime fields remain documented on ``RobotCfg``; the class declarations
below identify canonical imports without re-expanding inherited configuration
fields or generated constructors.


embodichain.robot_registry
--------------------------

Discover installed robot IDs or construct the selected configuration with caller overrides.
An explicit ``variant`` selects its catalog profile defaults before caller
overrides are applied. Omitting it preserves existing resolution defaults.
The profile's display-only ``default_variant`` does not implicitly select a
variant. All merges preserve the catalog and caller mappings.

.. autofunction:: embodichain.robot_registry.list_robot_cfgs

.. autofunction:: embodichain.robot_registry.resolve_robot_cfg


embodichain_robots
------------------

Concrete robot configurations, exposed lazily from the bundled robot package.

.. py:class:: embodichain_robots.ARX5Cfg

   Configure the X5A arm and its two independently actuated parallel fingers.
   Construct with ``from_dict()``; ``build_pk_serial_chain()`` returns the
   six-joint flange chain. Its separate numerical-solver TCP aligns jaw
   opening and approach for parallel-jaw grasp generation.
   The default ready pose unfolds the elbow and wrist, and self-collision is
   enabled. Task-specific IK branch limits belong to the caller's scene.
   An explicit SSIK CPU reference can use the prepared actual-model artifact;
   see :doc:`/overview/sim/motion/solvers/ssik_solver`.

.. py:class:: embodichain_robots.AlohaMiniCfg
   :no-index:

   Configure Aloha Mini 2 Pro from its complete, source-named URDF.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.

.. py:class:: embodichain_robots.TianjiMarvinCfg

   Configure Tianji Marvin with or without its parallel-jaw grippers.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.

.. py:class:: embodichain_robots.DexforceW1Cfg

   DexforceW1 specific configuration, inherits from RobotCfg and allows custom parameters.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.

.. py:class:: embodichain_robots.CobotMagicCfg

   Dual-arm CobotMagic with source arm limits and calibrated gripper drives.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.

.. py:class:: embodichain_robots.FrankaPandaCfg

   Configuration for the Franka Emika Panda robot with Panda hand.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.

.. py:class:: embodichain_robots.URRobotCfg

   Configuration for the UR family of robots.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.

.. py:class:: embodichain_robots.SO101FollowerCfg

   SO101 follower with five arm joints, one actuated jaw, official CAD inertials
   and published STS3215 simulation drives. Both serial chains use the same
   source URDF as simulation; arm IK constrains position.
   See :doc:`/resources/robot/so101` for frames, units and source limitations.

.. automethod:: embodichain_robots.SO101FollowerCfg.from_dict

.. automethod:: embodichain_robots.SO101FollowerCfg.build_pk_serial_chain

.. py:class:: embodichain_robots.FlexivRizon4SCfg

   Seven-axis bare Rizon 4S arm with published Flexiv dynamics, tuned simulation
   damping and a numerical flange IK solver. Construct with ``from_dict()`` and use
   ``build_pk_serial_chain()`` for the matching differentiable flange chain.
   Source inertials are preserved; public dynamics are manufacturer-blurred.
   See :doc:`/resources/robot/flexiv_rizon4s` for units and pinned provenance.


embodichain_robots.alohamini.aloha_mini.cfg
-------------------------------------------

Concrete robot configurations, exposed lazily from the bundled robot package.

.. py:class:: embodichain_robots.alohamini.aloha_mini.cfg.AlohaMiniCfg
   :no-index:

   Configure Aloha Mini 2 Pro from its complete, source-named URDF.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain_robots.arx.arx5.cfg
-------------------------------

.. py:class:: embodichain_robots.arx.arx5.cfg.ARX5Cfg
   :no-index:

   Implementation of :class:`embodichain_robots.ARX5Cfg`, using the bundled
   X5A asset and a six-joint numerical kinematic chain.


embodichain_robots.agilex.cobotmagic.cfg
----------------------------------------

Concrete robot configurations, exposed lazily from the bundled robot package.

.. py:class:: embodichain_robots.agilex.cobotmagic.cfg.CobotMagicCfg
   :no-index:

   Dual-arm CobotMagic with source arm limits and calibrated gripper drives.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain_robots.dexforce.w1
------------------------------

W1 configuration, hardware revisions, calibrated transforms, hand specifications, and assembly helpers.

.. py:class:: embodichain_robots.dexforce.w1.DexforceW1Cfg
   :no-index:

   DexforceW1 specific configuration, inherits from RobotCfg and allows custom parameters.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain_robots.dexforce.w1.cfg
----------------------------------

W1 configuration, hardware revisions, calibrated transforms, hand specifications, and assembly helpers.

.. py:class:: embodichain_robots.dexforce.w1.cfg.DexforceW1Cfg
   :no-index:

   DexforceW1 specific configuration, inherits from RobotCfg and allows custom parameters.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain_robots.dexforce.w1.hand_specs
-----------------------------------------

W1 configuration, hardware revisions, calibrated transforms, hand specifications, and assembly helpers.

.. py:class:: embodichain_robots.dexforce.w1.hand_specs.W1HandSideSpec

   Side-specific asset metadata for one hand release.

.. py:class:: embodichain_robots.dexforce.w1.hand_specs.W1HandSpec

   A hand release independent of the W1 body/arm version.

.. autofunction:: embodichain_robots.dexforce.w1.hand_specs.get_default_w1_hand_version

.. autofunction:: embodichain_robots.dexforce.w1.hand_specs.get_w1_hand_spec

.. autofunction:: embodichain_robots.dexforce.w1.hand_specs.normalize_w1_hand_mappings


embodichain_robots.dexforce.w1.params
-------------------------------------

W1 configuration, hardware revisions, calibrated transforms, hand specifications, and assembly helpers.

.. py:class:: embodichain_robots.dexforce.w1.params.W1ArmKineParams

   Kinematics parameters for a W1 arm.


embodichain_robots.dexforce.w1.specs
------------------------------------

W1 configuration, hardware revisions, calibrated transforms, hand specifications, and assembly helpers.

.. py:class:: embodichain_robots.dexforce.w1.specs.W1VersionSpec

   Asset layout and calibrated defaults belonging to one W1 revision.

.. autofunction:: embodichain_robots.dexforce.w1.specs.get_w1_version_spec


embodichain_robots.dexforce.w1.types
------------------------------------

W1 configuration, hardware revisions, calibrated transforms, hand specifications, and assembly helpers.

.. py:class:: embodichain_robots.dexforce.w1.types.DexforceW1Version

   Released W1 body revision identifiers.

.. py:class:: embodichain_robots.dexforce.w1.types.DexforceW1HandVersion

   W1 end-effector revision identifiers, independent of the body revision.

.. py:class:: embodichain_robots.dexforce.w1.types.DexforceW1ArmSide

   Left/right side identifiers for W1 assembly and calibration.

.. py:class:: embodichain_robots.dexforce.w1.types.DexforceW1Type

   Physical W1 component identifiers used by the assembly helpers.

.. py:class:: embodichain_robots.dexforce.w1.types.DexforceW1HandBrand

   Supported W1 end-effector model identifiers.


embodichain_robots.dexforce.w1.utils
------------------------------------

W1 configuration, hardware revisions, calibrated transforms, hand specifications, and assembly helpers.

.. py:class:: embodichain_robots.dexforce.w1.utils.ChassisManager

   Robot hardware configuration and assembly helper.

.. py:class:: embodichain_robots.dexforce.w1.utils.TorsoManager

   Robot hardware configuration and assembly helper.

.. py:class:: embodichain_robots.dexforce.w1.utils.HeadManager

   Robot hardware configuration and assembly helper.

.. py:class:: embodichain_robots.dexforce.w1.utils.ArmManager

   Robot hardware configuration and assembly helper.

.. py:class:: embodichain_robots.dexforce.w1.utils.HandManager

   Robot hardware configuration and assembly helper.

.. py:class:: embodichain_robots.dexforce.w1.utils.EyesManager

   Robot hardware configuration and assembly helper.

.. autofunction:: embodichain_robots.dexforce.w1.utils.build_dexforce_w1_assembly_urdf_cfg


embodichain_robots.flexiv.rizon4s.cfg
-------------------------------------

.. py:class:: embodichain_robots.flexiv.rizon4s.cfg.FlexivRizon4SCfg
   :no-index:

   Implementation of :class:`embodichain_robots.FlexivRizon4SCfg`.
   An explicit URDF ``fpath`` is shared by simulation and serial-chain FK/IK.


embodichain_robots.franka.panda
-------------------------------

Concrete robot configurations, exposed lazily from the bundled robot package.

.. py:class:: embodichain_robots.franka.panda.FrankaPandaCfg
   :no-index:

   Configuration for the Franka Emika Panda robot with Panda hand.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain_robots.the_robot_studio.so101.cfg
---------------------------------------------

.. py:class:: embodichain_robots.the_robot_studio.so101.cfg.SO101FollowerCfg
   :no-index:

   Implementation of :class:`embodichain_robots.SO101FollowerCfg`.


embodichain_robots.tianji.marvin.cfg
------------------------------------

Concrete robot configurations, exposed lazily from the bundled robot package.

.. py:class:: embodichain_robots.tianji.marvin.cfg.TianjiMarvinCfg
   :no-index:

   Configure Tianji Marvin with or without its parallel-jaw grippers.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain_robots.universal_robots.ur_robot
--------------------------------------------

Concrete robot configurations, exposed lazily from the bundled robot package.

.. py:class:: embodichain_robots.universal_robots.ur_robot.URRobotCfg
   :no-index:

   Configuration for the UR family of robots.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


embodichain_robots.agilex.cobotmagic
------------------------------------

The model package exposes its configuration lazily; accessing asset definitions
does not import this configuration implementation.

.. py:class:: embodichain_robots.agilex.cobotmagic.CobotMagicCfg
   :no-index:

   Lazy model-level export of :class:`embodichain_robots.CobotMagicCfg`.


embodichain_robots.agilex.cobotmagic.assets
-------------------------------------------

Model-owned download definitions preserve their asset IDs, archive checksums
and cache prefixes while using the shared core download and locking services.

.. py:class:: embodichain_robots.agilex.cobotmagic.assets.CobotMagicArm
   :no-index:

   Dataset class for the Cobot Magic Arm robot.


embodichain_robots.alohamini.aloha_mini
---------------------------------------

The model package exposes its configuration lazily; accessing asset definitions
does not import this configuration implementation.

.. py:class:: embodichain_robots.alohamini.aloha_mini.AlohaMiniCfg
   :no-index:

   Lazy model-level export of :class:`embodichain_robots.AlohaMiniCfg`.


embodichain_robots.alohamini.aloha_mini.assets
----------------------------------------------

Model-owned download definitions preserve their asset IDs, archive checksums
and cache prefixes while using the shared core download and locking services.

.. py:class:: embodichain_robots.alohamini.aloha_mini.assets.AlohaMini
   :no-index:

   Dataset class for the Aloha Mini robot.


embodichain_robots.arx.arx5
---------------------------

The model package exposes its configuration lazily; accessing asset definitions
does not import this configuration implementation.

.. py:class:: embodichain_robots.arx.arx5.ARX5Cfg
   :no-index:

   Lazy model-level export of :class:`embodichain_robots.ARX5Cfg`.


embodichain_robots.arx.arx5.assets
----------------------------------

Model-owned download definitions preserve their asset IDs, archive checksums
and cache prefixes while using the shared core download and locking services.

.. py:class:: embodichain_robots.arx.arx5.assets.ARX5
   :no-index:

   Dataset class for the ARX5 robot.


embodichain_robots.dexforce.w1.assets
-------------------------------------

Model-owned download definitions preserve their asset IDs, archive checksums
and cache prefixes while using the shared core download and locking services.

.. py:class:: embodichain_robots.dexforce.w1.assets.DexforceW1V021
   :no-index:

   Dataset class for the Dexforce W1 V021.

.. py:class:: embodichain_robots.dexforce.w1.assets.DexforceW1V022
   :no-index:

   Dataset class for the unified Dexforce W1 V022 release archive.

.. py:class:: embodichain_robots.dexforce.w1.assets.DexforceW1V025
   :no-index:

   Dataset class for the unified Dexforce W1 V025 release archive.

.. py:class:: embodichain_robots.dexforce.w1.assets.DexforceW1V030
   :no-index:

   Standard V030 body archive, excluding detachable end effectors.

.. py:class:: embodichain_robots.dexforce.w1.assets.DexforceW1ChassisV021
   :no-index:

   Resolve the DexforceW1ChassisV021 resource bundle through the shared data cache.

.. py:class:: embodichain_robots.dexforce.w1.assets.DexforceW1TorsoV021
   :no-index:

   Resolve the DexforceW1TorsoV021 resource bundle through the shared data cache.

.. py:class:: embodichain_robots.dexforce.w1.assets.DexforceW1EyesV021
   :no-index:

   Resolve the DexforceW1EyesV021 resource bundle through the shared data cache.

.. py:class:: embodichain_robots.dexforce.w1.assets.DexforceW1HeadV021
   :no-index:

   Resolve the DexforceW1HeadV021 resource bundle through the shared data cache.

.. py:class:: embodichain_robots.dexforce.w1.assets.DexforceW1LeftArmV021
   :no-index:

   Resolve the DexforceW1LeftArmV021 resource bundle through the shared data cache.

.. py:class:: embodichain_robots.dexforce.w1.assets.DexforceW1RightArmV021
   :no-index:

   Resolve the DexforceW1RightArmV021 resource bundle through the shared data cache.


embodichain_robots.flexiv.rizon4s
---------------------------------

The model package exposes its configuration lazily; accessing asset definitions
does not import this configuration implementation.

.. py:class:: embodichain_robots.flexiv.rizon4s.FlexivRizon4SCfg
   :no-index:

   Lazy model-level export of :class:`embodichain_robots.FlexivRizon4SCfg`.


embodichain_robots.flexiv.rizon4s.assets
----------------------------------------

Model-owned download definitions preserve their asset IDs, archive checksums
and cache prefixes while using the shared core download and locking services.

.. py:class:: embodichain_robots.flexiv.rizon4s.assets.FlexivRizon4S
   :no-index:

   Download the Apache-2.0 Flexiv Rizon 4S description from Hugging Face.


embodichain_robots.franka.assets
--------------------------------

Model-owned download definitions preserve their asset IDs, archive checksums
and cache prefixes while using the shared core download and locking services.

.. py:class:: embodichain_robots.franka.assets.Franka
   :no-index:

   Dataset class for the Franka robots.


embodichain_robots.the_robot_studio.so101
-----------------------------------------

The model package exposes its configuration lazily; accessing asset definitions
does not import this configuration implementation.

.. py:class:: embodichain_robots.the_robot_studio.so101.SO101FollowerCfg
   :no-index:

   Lazy model-level export of :class:`embodichain_robots.SO101FollowerCfg`.


embodichain_robots.the_robot_studio.so101.assets
------------------------------------------------

Model-owned download definitions preserve their asset IDs, archive checksums
and cache prefixes while using the shared core download and locking services.

.. py:class:: embodichain_robots.the_robot_studio.so101.assets.SO101FollowerAssets
   :no-index:

   Download the official follower URDF and its dependencies at a fixed revision.


embodichain_robots.tianji.marvin
--------------------------------

The model package exposes its configuration lazily; accessing asset definitions
does not import this configuration implementation.

.. py:class:: embodichain_robots.tianji.marvin.TianjiMarvinCfg
   :no-index:

   Lazy model-level export of :class:`embodichain_robots.TianjiMarvinCfg`.


embodichain_robots.tianji.marvin.assets
---------------------------------------

Model-owned download definitions preserve their asset IDs, archive checksums
and cache prefixes while using the shared core download and locking services.

.. py:class:: embodichain_robots.tianji.marvin.assets.TianjiMarvin
   :no-index:

   Dataset class for the Tianji Marvin robot.


embodichain_robots.universal_robots.assets
------------------------------------------

Model-owned download definitions preserve their asset IDs, archive checksums
and cache prefixes while using the shared core download and locking services.

.. py:class:: embodichain_robots.universal_robots.assets.UniversalRobots
   :no-index:

   Dataset class for the Universal Robots.
