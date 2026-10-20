:orphan:

End-effector assets
===================

.. currentmodule:: embodichain.data.assets.eef_assets

These dataset classes download and extract end-effector URDFs and their mesh
bundles. Pass a custom ``data_root`` to isolate the cache, or resolve a model
through ``embodichain.data.get_data_path`` using its dataset name and model path.

PikaGripper uses ``PikaGripper/PikaGripper.urdf``. Its root attachment frame is
``GRIPPER_ROOT_LINK``. ``FINGER1_JOINT`` drives the parallel jaws over 0–0.05 m;
``FINGER2_JOINT`` follows it through a URDF mimic relationship. The archive
preserves the DexEChain visual DAE and collision STL meshes.

.. autosummary::
   :nosignatures:

   DH_PGC_140_50
   DH_PGI_140_80
   DH_PGC_140_50_M
   DH_AG95
   ZH_CTM2F110
   BrainCoHandRevo1
   InspireHand
   Robotiq
   Robotiq2F85
   WheelTecFA2F
   PikaGripper

.. autoclass:: DH_PGC_140_50
   :members:

.. autoclass:: DH_PGI_140_80
   :members:

.. autoclass:: DH_PGC_140_50_M
   :members:

.. autoclass:: DH_AG95
   :members:

.. autoclass:: ZH_CTM2F110
   :members:

.. autoclass:: BrainCoHandRevo1
   :members:

.. autoclass:: InspireHand
   :members:

.. autoclass:: Robotiq
   :members:

.. autoclass:: Robotiq2F85
   :members:

.. autoclass:: WheelTecFA2F
   :members:

.. autoclass:: PikaGripper
   :members:
