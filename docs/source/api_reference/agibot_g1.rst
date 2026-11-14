AgiBot G1
=========

The preset loads the original GenieSim G1_120s USD and exposes its joint
controls. It uses the source-authored mimic and gripper closure relationships
on the Default physics backend. See :doc:`../resources/robot/agibot/g1` for
preview and local source selection.

.. py:class:: embodichain_robots.AgibotG1Cfg

   Source-preserving USD robot configuration with torso, head, dual-arm and
   single-driver gripper control groups. Construct with ``from_dict()`` or
   resolve catalog ID ``AgibotG1`` to apply the preset defaults. Serial-chain
   construction is disabled; the preset supplies joint-space control.

.. automethod:: embodichain_robots.AgibotG1Cfg.from_dict

.. py:class:: embodichain_robots.agibot.g1.AgibotG1Cfg

   Model namespace for :class:`embodichain_robots.AgibotG1Cfg`.

.. py:class:: embodichain_robots.agibot.g1.cfg.AgibotG1Cfg

   Implementation of :class:`embodichain_robots.AgibotG1Cfg`.

.. autoclass:: embodichain_robots.agibot.g1.assets.AgibotG1
