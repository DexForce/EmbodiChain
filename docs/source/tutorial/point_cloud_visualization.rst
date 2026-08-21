.. _tutorial_point_cloud_visualization:

Visualizing a Point Cloud
=========================

This tutorial uses :meth:`SimulationManager.visualize_point_cloud` to display
a color-coded point cloud in the native DexSim viewer. It is useful for
inspecting sampled workspaces, sensor output, and other point-based data in the
same coordinate frame as a simulation scene.

The Code
~~~~~~~~

The tutorial corresponds to ``visualize_point_cloud.py`` in
``scripts/tutorials/sim``.

.. dropdown:: Code for visualize_point_cloud.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/sim/visualize_point_cloud.py
      :language: python
      :linenos:

Building a Verifiable Point Cloud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The example constructs three orthogonal point axes:

- red points along X;
- green points along Y;
- blue points along Z.

Each color is stored as ``uint8`` RGB. The manager accepts either normalized
``[0, 1]`` values or ``[0, 255]`` values, and normalizes the latter before
passing them to DexSim.

.. literalinclude:: ../../../scripts/tutorials/sim/visualize_point_cloud.py
   :language: python
   :start-at: def build_demo_point_cloud(
   :end-at:     return points, colors

Creating the Native Point Cloud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create the simulation headlessly, add the point cloud, then open the native
window after the scene is ready. The ``name`` identifies the native DexSim
object, and ``point_size`` is measured in renderer pixels.

.. literalinclude:: ../../../scripts/tutorials/sim/visualize_point_cloud.py
   :language: python
   :start-at:     sim = SimulationManager(
   :end-before:     try:
   :dedent: 4

.. literalinclude:: ../../../scripts/tutorials/sim/visualize_point_cloud.py
   :language: python
   :start-at:         points, colors = build_demo_point_cloud()
   :end-at:         )
   :dedent: 8

`visualize_point_cloud` accepts point positions with shape ``(N, 3)`` and
optional per-point RGB or RGBA colors with shape ``(N, 3)`` or ``(N, 4)``.
When colors are omitted, the manager renders all points in green. RGBA input is
accepted for compatibility, but the native manager currently renders RGB
colors with opaque alpha.

Running the Tutorial
~~~~~~~~~~~~~~~~~~~~

Run the tutorial from the repository root:

.. code-block:: bash

   python scripts/tutorials/sim/visualize_point_cloud.py

The native DexSim window should show a red horizontal X axis, a green
horizontal Y axis, and a blue vertical Z axis. This verifies both point
placement and per-point color handling. Press ``Ctrl+C`` in the terminal to
stop the tutorial.

The script explicitly uses ``destroy(exit_process=False)`` and then
``SimulationManager.flush_cleanup_queue()`` so its simulation resources are
released before Python exits.

Headless Rendering
~~~~~~~~~~~~~~~~~~

To save a single frame on a machine without a native display, pass
``--headless``. The tutorial creates an offscreen DexSim camera at the same
overview pose as the interactive viewer, renders once, and then exits:

.. code-block:: bash

   python scripts/tutorials/sim/visualize_point_cloud.py --headless \
       --output outputs/point_cloud_visualization.png

The resulting image preserves the red X, green Y, and blue Z axes, so it is a
portable visual check of both point placement and per-point colors.

.. figure:: /_static/tutorials/point_cloud_visualization.png
   :alt: An offscreen render of red, green, and blue point-cloud axes.
   :width: 100%

   One frame rendered by the tutorial's offscreen camera.

Next Steps
~~~~~~~~~~

- :doc:`create_scene` — Add rigid objects and sensors to the same scene.
- :doc:`sensor` — Capture and process camera data that can be visualized as
  point samples.
- :doc:`/overview/sim/sim_manager` — Learn about the simulation lifecycle and
  the full manager API.
