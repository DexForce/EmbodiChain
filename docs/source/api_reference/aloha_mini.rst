:orphan:

Aloha Mini robot configuration
==============================

The preset configures the complete Aloha Mini 2 Pro asset, its five control
parts, arm and torso solvers, and local serial chains for all five parts.

Public import
-------------

.. currentmodule:: embodichain_robots

.. py:class:: AlohaMiniCfg

   Preset for the complete Aloha Mini 2 Pro asset. Shared configuration fields
   follow :class:`~embodichain.lab.sim.cfg.RobotCfg`.

.. automethod:: AlohaMiniCfg.from_dict

.. automethod:: AlohaMiniCfg.build_pk_serial_chain

Implementation module
---------------------

.. currentmodule:: embodichain_robots.aloha_mini

.. py:class:: AlohaMiniCfg
   :no-index:

   Implementation of :class:`embodichain_robots.AlohaMiniCfg`.
