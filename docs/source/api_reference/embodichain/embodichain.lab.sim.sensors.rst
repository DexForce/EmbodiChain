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

Attachment resolution
---------------------

``SimulationManager.add_sensor()`` delegates parent-name resolution to the
function below before creating camera views. It accepts a canonical link name
or ``"<asset_uid>/<link_name>"`` to distinguish links shared by several assets.
The resolver receives the scene asset mapping explicitly and queries
``Articulation.get_link_render_nodes()``; it does not look up a global manager.

``Camera.attach_to_parent_nodes()`` then attaches the resolved nodes, reapplies
parent-relative extrinsics, and updates ``is_attached``. Stereo cameras share
this path. Directly constructed cameras require an explicit attachment call.

.. autosummary::

    ~attachment.resolve_parent_nodes

.. autofunction:: embodichain.lab.sim.sensors.attachment.resolve_parent_nodes

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
