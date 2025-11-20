.. _tutorial_simulate_robot:

Simulating a Robot
================

.. currentmodule:: embodichain.lab.sim

This tutorial shows you how to create and simulate a robot using SimulationManager. You'll learn how to load a robot from URDF files, configure control systems, and run basic robot simulation with joint control.

The Code
~~~~~~~~

The tutorial corresponds to the ``create_robot.py`` script in the ``scripts/tutorials/sim`` directory.

.. dropdown:: Code for create_robot.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/sim/create_robot.py
      :language: python
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Similar to the previous tutorial on creating a simulation scene, we use the :class:`SimulationManager` class to set up the simulation environment. If you haven't read that tutorial yet, please refer to :doc:`create_scene` first.

Loading Robot URDF
-------------------

SimulationManager supports loading robots from URDF (Unified Robot Description Format) files. You can load either a single URDF file or compose multiple URDF components into a complete robot system.

For a simple two-component robot (arm + hand):

.. literalinclude:: ../../../scripts/tutorials/sim/create_robot.py
   :language: python
   :start-at: sr5_urdf_path = get_data_path("Rokae/SR5/SR5.urdf")
   :end-at: robot: Robot = sim.add_robot(cfg=cfg)


The :class:`cfg.URDFCfg` allows you to compose multiple URDF files with specific transformations, enabling complex robot assemblies.


Configuring Control Parts
--------------------------

Control parts define how the robot's joints are grouped for control purposes. This is useful for organizing complex robots with multiple subsystems.

.. literalinclude:: ../../../scripts/tutorials/sim/create_robot.py
   :language: python
   :start-at: # Define control parts for the robot
   :end-at: }

Joint names in control parts can use regex patterns for flexible matching. For example:

- ``"JOINT[1-6]"`` matches JOINT1, JOINT2, ..., JOINT6
- ``"L_.*"`` matches all joints starting with `"L_"`
  
Setting Drive Properties
------------------------

Drive properties control how the robot's joints behave during simulation, including stiffness, damping, and force limits.

.. literalinclude:: ../../../scripts/tutorials/sim/create_robot.py
   :language: python
   :start-at: drive_pros=JointDrivePropertiesCfg(
   :end-at: )

You can set different stiffness values for different joint groups using regex patterns. More details on drive properties can be found in :class:`cfg.JointDrivePropertiesCfg`.

For more robot configuration options, refer to :class:`cfg.RobotCfg`.

Robot Control
-------------

For the basic control of robot joints, you can set position targets using :meth:`objects.Robot.set_qpos`. The control action should be created as a torch.Tensor with shape (num_envs, num_joints), where `num_joints` is the total number of joints in the robot or the number of joints in a specific control part.

- If you can control all joints, use:

  .. code-block:: python

     robot.set_qpos(qpos=target_positions)

- If you want to control a subset of joints, specify the joint IDs:
  
   .. code-block:: python

      robot.set_qpos(qpos=target_positions, joint_ids=subset_joint_ids)

Getting Robot State
--------------------

You can query the robot's current joint positions and velocities via :meth:`objects.Robot.get_qpos` and :meth:`objects.Robot.get_qvel`. For more robot API details, see :class:`objects.Robot`.

The Code Execution
~~~~~~~~~~~~~~~~~~

To run the robot simulation script:

.. code-block:: bash

   cd /root/sources/embodichain
   python scripts/tutorials/sim/create_robot.py

You can customize the simulation with various command-line options:

.. code-block:: bash

   # Run with GPU physics
   python scripts/tutorials/sim/create_robot.py --device cuda
   
   # Run multiple environments
   python scripts/tutorials/sim/create_robot.py --num_envs 4
   
   # Run in headless mode
   python scripts/tutorials/sim/create_robot.py --headless
   
   # Enable ray tracing rendering
   python scripts/tutorials/sim/create_robot.py --enable_rt

The simulation will show the robot moving through different poses, demonstrating basic joint control capabilities.

Key Features Demonstrated
~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial demonstrates several key features of robot simulation in SimulationManager:

1. **URDF Loading**: Both single-file and multi-component robot loading
2. **Control Parts**: Organizing joints into logical control groups
3. **Drive Properties**: Configuring joint stiffness and control behavior
4. **Joint Control**: Setting position targets and reading joint states
5. **Multi-Environment**: Running multiple robot instances in parallel

Next Steps
~~~~~~~~~~

After mastering basic robot simulation, you can explore:

- End-effector control and inverse kinematics
- Sensor integration (cameras, force sensors)
- Robot-object interaction scenarios

This tutorial provides the foundation for creating sophisticated robotic simulation scenarios with SimulationManager.