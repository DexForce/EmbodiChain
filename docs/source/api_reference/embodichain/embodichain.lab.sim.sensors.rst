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
    ~base_sensor.SensorCfg
    ~base_sensor.BaseSensor
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

Camera and stereo captures exclude native markers by default. Set
``include_visual_overlays=True`` on an explicit preview camera to include
both scene-lit markers and overlays.

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

Contact history
---------------

Contact filtering and substep timing are owned by the sensor layer. Tasks
select body and counterpart IDs, then consume interval events and completed
flight durations without advancing the sensor state.

.. automodule:: embodichain.lab.sim.sensors.contact.history
   :members:
   :undoc-members:

Creating a contact history automatically enables physics-substep sampling.
CUDA history reduction consumes compact contact rows and device counts through
sparse Warp kernels. The ``dropped_contacts`` diagnostic sums query loss and
scatter overflow across the current control interval; standalone updates
without histories or sampling observers report the most recent sample.

Newton force conversion uses the final internal solver substep's duration,
while history clocks advance by the full World-step duration. Standalone
``ContactHistory.update`` accepts an explicit ``impulse_dt`` for this conversion.
Physics-reset-invalidated reports yield invalid observation rows until a new
solve; a sensor-only reset can still reread a valid current report.

Contact sampling through ``update`` or ``update_physics_step`` must run eagerly
outside caller-owned CUDA Graph capture. Sample before each replay and establish
stream dependencies before consuming the stable ``get_data()`` buffers in the
graph. The sensor's internal reduction graph remains supported.

``latest_sample_dropped_contacts`` reports the latest fetch's query and scatter
loss even when observers retain interval counters. Evidence sampling rejects
either diagnostic and takes their maximum to avoid double counting. Full reset
clears latest loss; selective reset retains query-wide loss. Diagnostic reads
respect sampling-stream dependencies and may synchronize device work.
``evidence_dropped_contacts`` combines both diagnostics with one packed readback
and int64 host reduction. Latest query loss is retained by the existing CUDA scatter
kernel, including empty-contact samples and graph replay; CPU uses a scalar copy
to avoid a per-row branch. The standard
sampler uses this path; custom sources retain the separate-counter fallback.

Sampling observers
------------------

Named sampling observers belong to one ContactSensor instance. The empty default
adds no observer snapshots or callbacks. Begin and selective-reset notifications
follow the sensor's own history/counter updates; before/after tokens surround one
physics-step sample. A prior raw fetch in that same step is reused; repeated
substep calls do not repeat callbacks. Observation callbacks run outside the
internal CUDA capture, and all sampling in external capture is rejected before
native queries or callbacks begin. The observer owns retention limits and diagnostics and must not
step physics or issue commands.

.. currentmodule:: embodichain.lab.sim.sensors.contact.sensor

.. autosummary::

   ContactSamplingObserver

.. autoclass:: ContactSamplingObserver
   :members:


Contact package
---------------

``embodichain.lab.sim.sensors.contact`` is the canonical entry point for contact
sensor configuration, acquisition and history. Its ``sensor``, ``sampling``,
``history``, ``timeline`` and ``observation`` modules share one package; sensor-specific Warp
adaptation kernels live under its private ``_warp`` directory. Generic physical
evidence remains in ``lab.sim.evidence``; grasp interpretation and Task Program
bindings remain in their owning layers. Calibrated hand opening used for release
interpretation belongs to ``atomic_skills.grip_evidence``, not the sensor layer.

.. currentmodule:: embodichain.lab.sim.sensors.contact

.. autosummary::
   :nosignatures:

   ArticulationContactFilterCfg
   ContactSensor
   ContactSensorCfg
   ContactSamplingObserver
   CONTACT_ACTOR_IDS
   CONTACT_VALID_MASK
   CONTACT_NORMALS
   CONTACT_IMPULSES
   CONTACT_POSITIONS
   CONTACT_DISTANCES
   DEFAULT_CHANNELS
   ContactEvidenceSample
   ContactEvidenceSampler
   ContactSourceStamp
   native_contact_revision
   ContactTimeline
   ContactTimelineCfg
   ContactHistory
   JointContactObservation
   JointContactObserver
Use the public ``sensors.contact`` entry point for joint/contact windows.
``capture`` appends without publishing internal storage; ``snapshot(start=0)``
returns owned full or incremental history. Callers retain the window generation
when reusing an incremental offset.


Import migration
----------------

The sensor registry and ``from embodichain.lab.sim.sensors import ContactSensor``
keep their existing names. The old flat ``sensors.contact_sensor`` and
``sensors.contact_history`` modules have been removed. Import from
``sensors.contact`` or its ``sensor`` and ``history`` submodules instead.
Sampling and timeline APIs use ``contact.sampling`` and ``contact.timeline``.
Regenerate declarations that explicitly seal implementation-qualified type names;
old identities are not silently rewritten.


Stereo implementation exports
-----------------------------

.. currentmodule:: embodichain.lab.sim.sensors.stereo

The implementation module exposes the stereo sensor and its configuration under
these import paths. ``PairCameraView`` provides compatibility access to one
left/right pair: state and attachment operations require a live owning sensor
and publish through its Scene batch. Its center-pose reads return owned matrices.

.. autosummary::

    StereoCamera
    StereoCameraCfg
    PairCameraView

.. autoclass:: PairCameraView
    :members:
    :show-inheritance:
