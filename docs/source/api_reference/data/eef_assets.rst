End-effector assets
===================

Concrete download classes are owned by manufacturer/model modules in the
bundled ``embodichain_robots`` package. The former core EEF module and
aggregate class exports have been removed. Asset IDs, URL/checksum identities
and cache prefixes remain stable. See :doc:`/resources/robot/end_effectors`
for the import migration table, provenance boundaries and rendered models.

Construct a dataset with optional ``data_root`` to select a cache, or use
``embodichain.data.get_data_path`` with its stable asset reference. Catalog
lookup alone does not construct datasets or download files. A download class
provides an asset; it does not imply a standalone robot or grasping controller.

embodichain_robots.dh_robotics.pgc_140_50.assets
------------------------------------------------

.. currentmodule:: embodichain_robots.dh_robotics.pgc_140_50.assets

.. autosummary::
   :nosignatures:

   DH_PGC_140_50
   DH_PGC_140_50_M

.. autoclass:: DH_PGC_140_50
   :members:

.. autoclass:: DH_PGC_140_50_M
   :members:

embodichain_robots.dh_robotics.pgi_140_80.assets
------------------------------------------------

.. currentmodule:: embodichain_robots.dh_robotics.pgi_140_80.assets

.. autosummary::
   :nosignatures:

   DH_PGI_140_80

.. autoclass:: DH_PGI_140_80
   :members:

embodichain_robots.dh_robotics.ag95.assets
------------------------------------------

.. currentmodule:: embodichain_robots.dh_robotics.ag95.assets

.. autosummary::
   :nosignatures:

   DH_AG95

.. autoclass:: DH_AG95
   :members:

embodichain_robots.changingtek.ctm2f110.assets
----------------------------------------------

.. currentmodule:: embodichain_robots.changingtek.ctm2f110.assets

.. autosummary::
   :nosignatures:

   ZH_CTM2F110

.. autoclass:: ZH_CTM2F110
   :members:

embodichain_robots.brainco.revo1.assets
---------------------------------------

.. currentmodule:: embodichain_robots.brainco.revo1.assets

.. autosummary::
   :nosignatures:

   BrainCoHandRevo1

.. autoclass:: BrainCoHandRevo1
   :members:

embodichain_robots.inspire.hand.assets
--------------------------------------

.. currentmodule:: embodichain_robots.inspire.hand.assets

.. autosummary::
   :nosignatures:

   InspireHand

.. autoclass:: InspireHand
   :members:

embodichain_robots.robotiq.assets
---------------------------------

.. currentmodule:: embodichain_robots.robotiq.assets

.. autosummary::
   :nosignatures:

   Robotiq

.. autoclass:: Robotiq
   :members:

embodichain_robots.robotiq.robotiq_2f85.assets
----------------------------------------------

.. currentmodule:: embodichain_robots.robotiq.robotiq_2f85.assets

.. autosummary::
   :nosignatures:

   Robotiq2F85

.. autoclass:: Robotiq2F85
   :members:

embodichain_robots.wheeltec.fa2f.assets
---------------------------------------

.. currentmodule:: embodichain_robots.wheeltec.fa2f.assets

.. autosummary::
   :nosignatures:

   WheelTecFA2F

.. autoclass:: WheelTecFA2F
   :members:

embodichain_robots.agilex.agx_gripper.assets
--------------------------------------------

Pinned AGX two-finger gripper with a single opening command in meters and
two source mimic joints. The asset provides a portable URDF and source
meshes for explicit mounting in a robot embodiment.

.. currentmodule:: embodichain_robots.agilex.agx_gripper.assets

.. autosummary::
   :nosignatures:

   AgxGripper

.. autoclass:: AgxGripper
   :members:

embodichain_robots.agilex.pika.assets
-------------------------------------

.. currentmodule:: embodichain_robots.agilex.pika.assets

.. autosummary::
   :nosignatures:

   PikaGripper

.. autoclass:: PikaGripper
   :members:

embodichain_robots.sharpa.wave.assets
-------------------------------------

Pinned official left/right Sharpa Wave 01 descriptions and meshes.
The asset includes 22 active joints per hand, upstream MJCF, licensing
and a source manifest. It supplies model resources without registering a
robot preset, tactile runtime or manipulation policy.

.. currentmodule:: embodichain_robots.sharpa.wave.assets

.. autosummary::
   :nosignatures:

   SharpaWave

.. autoclass:: SharpaWave
   :members:


embodichain_robots.wuji.hand.assets
-----------------------------------

Pinned official left/right Wuji Hand descriptions and meshes.
The asset includes 20 active joints per hand, upstream MJCF, licensing
and a source manifest. It supplies model resources without registering a
robot preset, tactile runtime or manipulation policy.

.. currentmodule:: embodichain_robots.wuji.hand.assets

.. autosummary::
   :nosignatures:

   WujiHand

.. autoclass:: WujiHand
   :members:
