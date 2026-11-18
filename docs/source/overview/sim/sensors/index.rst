Sensors
=======

Sensors provide batched observations of the simulation, including rendered
images, joint and frame state, inertial and contact measurements, geometric
scans, and tactile outputs.

- :doc:`backend_support` summarizes sensor and acquisition-mode support by physics backend.
- :doc:`camera` covers image modalities, intrinsics, extrinsics, and camera usage.
- :doc:`stereo_camera` covers the right camera, baseline, and disparity settings.
- :doc:`contact_sensor` covers contact filters, observation data, and backend support.

For a complete camera simulation example, see :doc:`/tutorial/sensor`.

.. toctree::
   :maxdepth: 1

   backend_support
   camera
   stereo_camera
   contact_sensor
