embodichain.lab.sim.sensors
==========================================

.. automodule:: embodichain.lab.sim.sensors

Overview
--------

Sensors attached to the simulation scene. Every sensor derives from
:class:`BaseSensor` (itself a :class:`~embodichain.lab.sim.common.BatchEntity`),
is configured through a :class:`SensorCfg` subclass, and maintains a batched
``TensorDict`` data buffer of shape ``[num_envs]``. The built-ins are
:class:`Camera` (single RGB-D camera with configurable intrinsics/extrinsics),
:class:`StereoCamera` (a camera pair with a baseline transform and optional
disparity), and :class:`ContactSensor` (collision detection between rigid
bodies and articulation links via Warp kernels).

  .. rubric:: Classes

  .. autosummary::
    SensorCfg
    BaseSensor
    CameraCfg
    Camera
    StereoCameraCfg
    StereoCamera
    ContactSensorCfg
    ArticulationContactFilterCfg
    ContactSensor

.. currentmodule:: embodichain.lab.sim.sensors

Sensor
------
.. autoclass:: BaseSensor
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: SensorCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

Camera
------
.. autoclass:: Camera
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: CameraCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, copy, replace, to_dict, validate

Stereo Camera
-------------
.. autoclass:: StereoCamera
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: StereoCameraCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, copy, replace, to_dict, validate

Contact Sensor
--------------
.. autoclass:: ContactSensor
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: ContactSensorCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: ArticulationContactFilterCfg
    :members:
    :show-inheritance:
