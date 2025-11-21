.. _tutorial_modular_env:

Creating a Modular Environment
==============================

.. currentmodule:: embodichain.lab.gym

This tutorial demonstrates how to create sophisticated robotic environments using EmbodiChain's modular architecture. You'll learn how to use the advanced :class:`envs.EmbodiedEnv` class with configuration-driven setup, event managers, observation managers, and randomization systems.

The Code
~~~~~~~~

The tutorial corresponds to the ``modular_env.py`` script in the ``scripts/tutorials/gym`` directory.

.. dropdown:: Code for modular_env.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/gym/modular_env.py
      :language: python
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

This tutorial showcases EmbodiChain's most powerful environment creation approach using the :class:`envs.EmbodiedEnv` class. Unlike the basic environment tutorial, this approach uses declarative configuration classes and manager systems for maximum flexibility and reusability.

Event Configuration
-------------------

Events define automated behaviors that occur during simulation. There are three types of supported modes:

- `startup`: triggers once when the environment is initialized
- `reset`: triggers every time the environment is reset
- `interval`: triggers at fixed step intervals during simulation

The :class:`ExampleEventCfg` demonstrates three types of events:

.. literalinclude:: ../../../scripts/tutorials/gym/modular_env.py
   :language: python
   :lines: 36-76

**Asset Replacement Event**

The ``replace_obj`` event demonstrates dynamic asset swapping:

- **Function**: :func:`envs.managers.events.replace_assets_from_group`
- **Mode**: ``"reset"`` - triggers at environment reset
- **Purpose**: Randomly selects different fork models from a folder

**Light Randomization Event**

The ``randomize_light`` event creates dynamic lighting conditions:

- **Function**: :func:`envs.managers.randomization.rendering.randomize_light`
- **Mode**: ``"interval"`` - triggers every 5 steps
- **Parameters**: Randomizes position, color, and intensity within specified ranges

**Material Randomization Event**

The ``randomize_table_mat`` event varies visual appearance:

- **Function**: :func:`envs.managers.randomization.rendering.randomize_visual_material`
- **Mode**: ``"interval"`` - triggers every 10 steps
- **Features**: Random textures from COCO dataset and base color variations

for more randomization events, please refer

Observation Configuration
-------------------------

The default observation from :class:`envs.EmbodiedEnv` includes:
- `robot`: robot proprioceptive data (joint positions, velocities, efforts)
- `sensor`: all available sensor data (images, depth, segmentation, etc.)

However, users always need to define some custom observation for specified learning tasks. To handle this, the observation manager system allows users to declaratively specify additional observations.

.. literalinclude:: ../../../scripts/tutorials/gym/modular_env.py
   :language: python
   :lines: 79-87

This configuration:

- **Function**: :func:`envs.managers.observations.get_rigid_object_pose`
- **Mode**: ``"add"`` - appends data to observation dictionary
- **Name**: Custom key for the observation data
- **Target**: Tracks the fork object's pose in the scene

For details documentation, see :class:`envs.managers.cfg.ObservationCfg`.

Environment Configuration
-------------------------

The main environment configuration inherits from :class:`envs.EmbodiedEnvCfg` and defines all scene components:

**Robot Configuration**

.. currentmodule:: embodichain.lab.sim.robots

.. literalinclude:: ../../../scripts/tutorials/gym/modular_env.py
   :language: python
   :start-at: robot: RobotCfg = DexforceW1Cfg.from_dict(
   :end-at: )

Uses the pre-configured :class:`DexforceW1Cfg` with customizations:

- **Version**: Specific robot variant (v021)
- **Arm Type**: Anthropomorphic configuration
- **Position**: Initial placement in the scene

**Sensor Configuration**

.. literalinclude:: ../../../scripts/tutorials/gym/modular_env.py
   :language: python
   :lines: 104-118

.. currentmodule:: embodichain.lab.sim.sensors

Configures a stereo camera system using :class:`StereoCameraCfg`:

- **Resolution**: 960x540 pixels for realistic visual input
- **Features**: Depth sensing and segmentation masks enabled
- **Stereo Setup**: 6cm baseline between left and right cameras
- **Mounting**: Attached to robot's "eyes" frame

**Lighting Configuration**

.. literalinclude:: ../../../scripts/tutorials/gym/modular_env.py
   :language: python
   :lines: 120-130

Defines scene illumination with controllable point lights:

- **Type**: Point light for realistic shadows
- **Properties**: Configurable color, intensity, and position
- **UID**: Named reference for event system manipulation

**Rigid Objects**

.. literalinclude:: ../../../scripts/tutorials/gym/modular_env.py
   :language: python
   :lines: 132-157

Multiple objects demonstrate different physics properties:

*Table Configuration:*

- **Shape**: Custom PLY mesh with UV mapping
- **Physics**: Kinematic body (movable but not affected by forces)  
- **Material**: Friction and restitution properties for realistic contact

*Fork Configuration:*

- **Shape**: Detailed mesh from asset library
- **Scale**: Proportionally scaled for scene consistency
- **Physics**: Dynamic body affected by gravity and collisions

**Articulated Objects**

.. literalinclude:: ../../../scripts/tutorials/gym/modular_env.py
   :language: python
   :lines: 159-169

Demonstrates complex mechanisms with moving parts:

- **URDF**: Sliding drawer with joints and constraints
- **Positioning**: Placed on table surface for interaction

Environment Implementation
--------------------------

The actual environment class is remarkably simple due to the configuration-driven approach:

.. literalinclude:: ../../../scripts/tutorials/gym/modular_env.py
   :language: python
   :start-at: @register_env("ModularEnv-v1", max_episode_steps=100, override=True)
   :end-at: super().__init__(cfg, **kwargs)

The :class:`envs.EmbodiedEnv` base class automatically:

- Loads all configured scene components
- Sets up observation and action spaces
- Initializes event and observation managers
- Handles environment lifecycle (reset, step, etc.)

The Code Execution
~~~~~~~~~~~~~~~~~~

To run the modular environment:

.. code-block:: bash

   cd /path/to/embodichain
   python scripts/tutorials/gym/modular_env.py

The script demonstrates the complete workflow:

1. **Configuration**: Creates an instance of ``ExampleCfg``
2. **Registration**: Uses the registered environment ID
3. **Execution**: Runs episodes with zero actions to observe automatic behaviors


Manager System Benefits
~~~~~~~~~~~~~~~~~~~~~~~

The manager-based architecture provides several key advantages:

**Event Managers**

- **Modularity**: Reusable event functions across environments
- **Timing Control**: Flexible scheduling (reset, interval, condition-based)
- **Parameter Binding**: Type-safe configuration with validation
- **Extensibility**: Easy to add custom event behaviors

**Observation Managers**

- **Flexible Data**: Any simulation data can become an observation
- **Processing Pipeline**: Built-in normalization and transformation
- **Dynamic Composition**: Runtime observation space modification
- **Performance**: Efficient data collection and GPU acceleration


Key Features Demonstrated
~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial showcases the most advanced features of EmbodiChain environments:

1. **Configuration-Driven Design**: Declarative environment specification
2. **Manager Systems**: Modular event and observation handling
3. **Asset Management**: Dynamic loading and randomization
4. **Sensor Integration**: Realistic camera systems with stereo vision
5. **Physics Simulation**: Complex articulated and rigid body dynamics
6. **Visual Randomization**: Automated domain randomization
7. **Extensible Architecture**: Easy customization and extension points


This tutorial demonstrates the full power of EmbodiChain's modular environment system, providing the foundation for creating sophisticated robotic learning scenarios.
