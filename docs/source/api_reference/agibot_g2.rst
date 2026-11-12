:orphan:

AgiBot G2
=========

The ``AgibotG2`` catalog preset supplies the official CRS-arm G2 with
OmniPicker grippers, fine collision geometry and a pinned asset source.
It exposes two seven-joint arm chains, torso/head joint control and separate
steering/wheel groups. Gripper motion follows the source mimic approximation.
The default root is fixed; tool offsets and mobile-base control are explicit
application configuration.

Public configuration
--------------------

.. currentmodule:: embodichain_robots

.. autosummary::

   AgibotG2Cfg

.. autoclass:: AgibotG2Cfg
   :members:
   :show-inheritance:

Model namespace
---------------

.. currentmodule:: embodichain_robots.agibot.g2

.. autosummary::

   AgibotG2Cfg

.. autoclass:: AgibotG2Cfg
   :members:

Configuration implementation
----------------------------

.. currentmodule:: embodichain_robots.agibot.g2.cfg

.. autosummary::

   AgibotG2Cfg

.. autoclass:: AgibotG2Cfg
   :members:

Official asset cache
--------------------

.. currentmodule:: embodichain_robots.agibot.g2.assets

.. autosummary::

   AgibotG2

.. autoclass:: AgibotG2
   :members:
   :show-inheritance:

This asset class verifies the immutable upstream archive, checks extracted
dependencies, and installs a pre-expanded URDF without a ROS/Xacro runtime
dependency. The source license and derivation lock accompany the cached model.
