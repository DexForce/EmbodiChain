Sensors
=======

Sensors provide batched observations of the simulation. Choose a camera for
image data, a stereo camera for paired views and disparity, or a contact sensor
for contacts between rigid bodies and articulation links.

- :doc:`camera` covers image modalities, intrinsics, extrinsics, and camera usage.
- :doc:`stereo_camera` covers the right camera, baseline, and disparity settings.
- :doc:`contact_sensor` covers contact filters, observation data, and backend support.

For a complete camera simulation example, see :doc:`/tutorial/sensor`.

.. toctree::
   :maxdepth: 1

   camera
   stereo_camera
   contact_sensor
