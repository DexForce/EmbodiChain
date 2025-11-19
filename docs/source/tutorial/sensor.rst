.. _tutorial_simulate_sensor:

Simulating a Camera Sensor
=========================

.. currentmodule:: embodichain.lab.sim

This tutorial demonstrates how to create and simulate a camera sensor attached to a robot using SimulationManager. You will learn how to configure a camera, attach it to the robot's end-effector, and visualize the sensor's output during simulation.

Source Code
~~~~~~~~~~~

The code for this tutorial is in ``scripts/tutorials/sim/create_sensor.py``.

.. dropdown:: Show code for create_sensor.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/sim/create_sensor.py
      :language: python
      :linenos:

Overview
~~~~~~~~

This tutorial builds on the basic robot simulation example. If you are not familiar with robot simulation in SimulationManager, please read the :doc:`robot` tutorial first.

1. **Sensor Creation and Attachment**
-------------------------------------

The camera sensor is created using :class:`CameraCfg` and can be attached to the robot's end-effector or placed freely in the scene. The attachment is controlled by the ``--attach_sensor`` argument.

.. literalinclude:: ../../../scripts/tutorials/sim/create_sensor.py
   :language: python
   :start-at: def create_sensor
   :end-at: return camera

- The camera's intrinsics (focal lengths and principal point) and resolution are set.
- The ``extrinsics`` specify the camera's pose relative to its parent (e.g., the robot's ``ee_link`` or the world).
- The camera is added to the simulation with :meth:`sim.add_sensor_v2`.

2. **Visualizing Sensor Output**
--------------------------------

The function ``get_sensor_image`` retrieves and visualizes the camera's color, depth, mask, and normal images. In GUI mode, images are shown in a 2x2 grid using OpenCV. In headless mode, images are saved to disk.

.. literalinclude:: ../../../scripts/tutorials/sim/create_sensor.py
   :language: python
   :start-at: def get_sensor_image
   :end-at: plt.close(fig)

- The camera is updated to capture the latest data.
- Four types of images are visualized: color, depth, mask, and normals.
- Images are displayed in a window or saved as PNG files depending on the mode.

3. **Simulation Loop**
----------------------

The simulation loop moves the robot through different arm poses and periodically updates and visualizes the sensor output.

.. literalinclude:: ../../../scripts/tutorials/sim/create_sensor.py
   :language: python
   :start-at: def run_simulation
   :end-at: sim.destroy()

- The robot alternates between two arm positions.
- After each movement, the sensor image is refreshed and visualized.

Running the Example
~~~~~~~~~~~~~~~~~~~

To run the sensor simulation script:

.. code-block:: bash

   cd /home/dex/projects/yuanhaonan/embodichain
   python scripts/tutorials/sim/create_sensor.py

You can customize the simulation with the following command-line options:

.. code-block:: bash

   # Use GPU physics
   python scripts/tutorials/sim/create_sensor.py --device cuda

   # Simulate multiple environments
   python scripts/tutorials/sim/create_sensor.py --num_envs 4

   # Run in headless mode (no GUI, images saved to disk)
   python scripts/tutorials/sim/create_sensor.py --headless

   # Enable ray tracing rendering
   python scripts/tutorials/sim/create_sensor.py --enable_rt

   # Attach the camera to the robot end-effector
   python scripts/tutorials/sim/create_sensor.py --attach_sensor

Key Features Demonstrated
~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial demonstrates:

1. **Camera sensor creation** using :class:`CameraCfg`
2. **Sensor attachment** to a robot link or placement in the scene
3. **Camera configuration** (intrinsics, extrinsics, clipping planes)
4. **Real-time visualization** of color, depth, mask, and normal images
5. **Robot-sensor integration** in a simulation loop

Next Steps
~~~~~~~~~~

After completing this tutorial, you can explore:

- Using other sensor types (e.g., stereo cameras, force sensors)
- Recording sensor data for offline analysis
- Integrating sensor feedback into robot control or learning algorithms

This tutorial provides a foundation for integrating perception into robotic simulation scenarios with SimulationManager.
This tutorial provides the foundation for integrating perception into robotic simulation scenarios with SimulationManager.