Robot catalogs and configurations
=================================

The main ``embodichain`` distribution bundles ``embodichain_robots`` with the
same release version. Concrete robot defaults and W1 hardware specifications
live in that import package; generic simulation, solver, and dual-arm APIs
remain in the core.

Gym robot selectors, workspace analysis, and dual-arm base selection share a
catalog resolver. Catalog entry points advertise robot IDs, configuration class
import paths, optional defaults, and optional control/variant summaries.
``embodichain list-robot`` reads these summaries to display independent control
DOFs, default control parts, and mimic followers. Listing defers robot class
imports and simulation initialization; resolution validates the selected class
and isolates its default/override fields. See :doc:`/guides/robot_catalog` for
the CLI, variant selection, and provider registration.

Official configurations provide the existing ``RobotCfg.from_dict()`` and
``build_pk_serial_chain()`` contracts. Import concrete configurations from
``embodichain_robots`` and generic dual-arm helpers from
``embodichain.lab.sim.robots``. Former concrete imports under the core robot
package and historical selector aliases have been removed. Selectors use exact
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

.. py:class:: embodichain_robots.FrankaPandaPgiCfg

   Panda arm with the DH Robotics PGI gripper, its TCP and finger contacts.

   Construct with ``from_dict()``. The ``arm`` control part contains seven
   joints and ``eef`` controls the PGI leader joint. ``build_pk_serial_chain()``
   returns the flange pose; the configured solver TCP locates the gripper tip.
   See :doc:`/resources/robot/franka_panda` for the assembly and frame contract.

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


embodichain_robots.flexiv_rizon4s
---------------------------------

.. py:class:: embodichain_robots.flexiv_rizon4s.FlexivRizon4SCfg
   :no-index:

   Implementation of :class:`embodichain_robots.FlexivRizon4SCfg`.
   An explicit URDF ``fpath`` is shared by simulation and serial-chain FK/IK.


embodichain_robots.franka_panda
-------------------------------

Concrete robot configurations, exposed lazily from the bundled robot package.

.. py:class:: embodichain_robots.franka_panda.FrankaPandaCfg
   :no-index:

   Configuration for the Franka Emika Panda robot with Panda hand.

   Construct with ``from_dict()``; ``build_pk_serial_chain()`` provides
   the robot-specific differentiable kinematic chains. Generic runtime
   fields and backend selection follow ``RobotCfg``.


.. py:class:: embodichain_robots.franka_panda.FrankaPandaPgiCfg
   :no-index:

   Panda arm with the PGI gripper; available as catalog ID ``FrankaPandaPgi``.
   The arm chain ends at ``fr3_link8`` and the solver applies the PGI TCP.
   See :class:`embodichain_robots.FrankaPandaPgiCfg` for the public entry point.


embodichain_robots.so101
------------------------

.. py:class:: embodichain_robots.so101.SO101FollowerCfg
   :no-index:

   Implementation of :class:`embodichain_robots.SO101FollowerCfg`.


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
