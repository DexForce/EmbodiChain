.. _tutorial_add_robot:

Adding a New Robot
==================

.. currentmodule:: embodichain.lab.sim.robots

This tutorial guides you through adding a new robot to EmbodiChain. You'll learn the file structure, key components, and patterns used for robot definitions.

EmbodiChain supports two approaches for defining robots:

1. **Single-file approach**: For simpler robots (like ``CobotMagic``)
2. **Package approach**: For complex robots with multiple variants (like ``DexforceW1``)

Choose the approach based on your robot's complexity.

---

Prerequisites
~~~~~~~~~~~~~~

Before adding a new robot, ensure you have:

- URDF file(s) for your robot
- Robot's kinematic parameters (DH parameters or joint limits)
- Understanding of your robot's joint structure and control parts

---

Approach 1: Single-File Robot (Simple Robots)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this approach for robots with a single variant and straightforward configuration.

File: ``embodichain/lab/sim/robots/my_robot.py``

.. dropdown:: Complete Example: CobotMagic-style Robot
   :icon: code

   .. literalinclude:: ../../../embodichain/lab/sim/robots/cobotmagic.py
      :language: python
      :linenos:

Step-by-Step Guide
------------------

1. **Create the configuration class** inheriting from ``RobotCfg``:

   .. code-block:: python

      from __future__ import annotations

      from typing import Dict, List, Any
      import numpy as np

      from embodichain.lab.sim.cfg import (
          RobotCfg,
          URDFCfg,
          JointDrivePropertiesCfg,
          RigidBodyAttributesCfg,
      )
      from embodichain.lab.sim.solvers import SolverCfg, OPWSolverCfg
      from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
      from embodichain.data import get_data_path
      from embodichain.utils import configclass

      @configclass
      class MyRobotCfg(RobotCfg):
          urdf_cfg: URDFCfg = None
          control_parts: Dict[str, List[str]] | None = None
          solver_cfg: Dict[str, "SolverCfg"] | None = None

2. **Implement the ``from_dict`` class method** for flexible initialization:

   .. code-block:: python

      @classmethod
      def from_dict(cls, init_dict: Dict[str, Any]) -> "MyRobotCfg":
          cfg = cls()
          default_cfgs = cls()._build_default_cfgs()
          for key, value in default_cfgs.items():
              setattr(cfg, key, value)
          cfg = merge_robot_cfg(cfg, init_dict)
          return cfg

3. **Define ``_build_default_cfgs``** with your robot's defaults:

   .. code-block:: python

      @staticmethod
      def _build_default_cfgs() -> Dict[str, Any]:
          # URDF path
          urdf_path = get_data_path("MyRobot/my_robot.urdf")

          # URDF configuration (for multi-component robots)
          urdf_cfg = URDFCfg(
              components=[
                  {
                      "component_type": "arm",
                      "urdf_path": urdf_path,
                      "transform": np.eye(4),  # 4x4 transform matrix
                  },
              ]
          )

          # Control parts - group joints for control
          control_parts = {
              "arm": [
                  "JOINT1", "JOINT2", "JOINT3",
                  "JOINT4", "JOINT5", "JOINT6",
              ],
              "gripper": ["JOINT7", "JOINT8"],
          }

          # Solver configuration for IK
          solver_cfg = {
              "arm": OPWSolverCfg(
                  end_link_name="link6",
                  root_link_name="base_link",
                  tcp=np.array([...]),  # Tool center point transform
              ),
          }

          # Drive properties - joint physics parameters
          drive_pros = JointDrivePropertiesCfg(
              stiffness={
                  "JOINT[1-6]": 7e4,  # Regex pattern for joints 1-6
                  "JOINT[7-8]": 3e2,
              },
              damping={
                  "JOINT[1-6]": 1e3,
                  "JOINT[7-8]": 3e1,
              },
              max_effort={
                  "JOINT[1-6]": 3e6,
                  "JOINT[7-8]": 3e3,
              },
          )

          return {
              "uid": "MyRobot",
              "urdf_cfg": urdf_cfg,
              "control_parts": control_parts,
              "solver_cfg": solver_cfg,
              "drive_pros": drive_pros,
              "attrs": RigidBodyAttributesCfg(
                  mass=0.1,
                  static_friction=0.95,
                  dynamic_friction=0.9,
                  linear_damping=0.7,
                  angular_damping=0.7,
              ),
          }

4. **Implement ``build_pk_serial_chain``** for PyTorch-Kinematics:

   .. code-block:: python

      def build_pk_serial_chain(
          self, device: torch.device = torch.device("cpu"), **kwargs
      ) -> Dict[str, "pk.SerialChain"]:
          from embodichain.lab.sim.utility.solver_utils import (
              create_pk_chain,
              create_pk_serial_chain,
          )

          urdf_path = get_data_path("MyRobot/my_robot.urdf")
          chain = create_pk_chain(urdf_path, device)

          arm_chain = create_pk_serial_chain(
              chain=chain,
              end_link_name="link6",
              root_link_name="base_link"
          ).to(device=device)

          return {"arm": arm_chain}

5. **Register in** ``embodichain/lab/sim/robots/__init__.py``:

   .. code-block:: python

      from .my_robot import MyRobotCfg

---

Approach 2: Package-Based Robot (Complex Robots)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this approach for robots with multiple variants (e.g., different arm types, versions, or configurations).

File Structure
--------------

For complex robots, create a package directory:

.. code-block::

   robots/
   └── my_robot/
       ├── __init__.py      # Exports the main config class
       ├── types.py         # Enums for robot variants
       ├── params.py        # Kinematics parameters
       ├── utils.py         # Manager classes and builders
       └── cfg.py           # Main configuration class

Step-by-Step Guide
-----------------

1. **types.py** - Define enums for robot variants:

   .. code-block:: python

      from enum import Enum

      class MyRobotVersion(Enum):
          V010 = "v010"
          V020 = "v020"

      class MyRobotArmKind(Enum):
          STANDARD = "standard"
          EXTENDED = "extended"

      class MyRobotSide(Enum):
          LEFT = "left"
          RIGHT = "right"

2. **params.py** - Define kinematics parameters:

   .. code-block:: python

      from dataclasses import dataclass
      import numpy as np
      from typing import Optional

      @dataclass
      class MyRobotArmKineParams:
          arm_side: MyRobotSide
          arm_kind: MyRobotArmKind
          version: MyRobotVersion

          dh_params: np.ndarray = None  # DH parameters (N x 4)
          qpos_limits: np.ndarray = None  # Joint limits (N x 2)
          link_lengths: np.ndarray = None  # Link lengths
          T_b_ob: np.ndarray = None  # Base to origin transform
          T_e_oe: np.ndarray = None  # End-effector transform

3. **utils.py** - Manager classes and builder functions:

   .. code-block:: python

      class ArmManager:
          """Manages arm URDF and configuration."""
          pass

      def build_my_robot_assembly_urdf_cfg(...):
          """Build URDF assembly from components."""
          pass

      def build_my_robot_cfg(...):
          """Build complete robot configuration."""
          pass

4. **cfg.py** - Main configuration class:

   .. code-block:: python

      @configclass
      class MyRobotCfg(RobotCfg):
          version: MyRobotVersion = MyRobotVersion.V010
          arm_kind: MyRobotArmKind = MyRobotArmKind.STANDARD

          @classmethod
          def from_dict(cls, init_dict: Dict) -> "MyRobotCfg":
              # Implementation similar to single-file approach
              pass

5. **__init__.py** - Export the config:

   .. code-block:: python

      from .cfg import MyRobotCfg

6. **Register in** ``robots/__init__.py``:

   .. code-block:: python

      from .my_robot import *

---

Key Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regardless of the approach, your robot config needs these core parameters:

+---------------------+------------------------+----------------------------------+
| Parameter           | Type                   | Description                      |
+=====================+========================+==================================+
| ``uid``             | str                    | Unique robot identifier         |
+---------------------+------------------------+----------------------------------+
| ``urdf_cfg``        | URDFCfg                | URDF file and components        |
+---------------------+------------------------+----------------------------------+
| ``control_parts``   | Dict[str, List[str]]   | Joint groups for control        |
+---------------------+------------------------+----------------------------------+
| ``solver_cfg``      | Dict[str, SolverCfg]   | IK solver configurations        |
+---------------------+------------------------+----------------------------------+
| ``drive_pros``      | JointDrivePropertiesCfg | Joint stiffness, damping, force |
+---------------------+------------------------+----------------------------------+
| ``attrs``           | RigidBodyAttributesCfg | Mass, friction, damping         |
+---------------------+------------------------+----------------------------------+

URDF Configuration
-----------------

The ``URDFCfg`` allows composing robots from multiple URDF files:

.. code-block:: python

   urdf_cfg = URDFCfg(
       components=[
           {
               "component_type": "arm",
               "urdf_path": arm_urdf,
               "transform": np.eye(4),
           },
           {
               "component_type": "gripper",
               "urdf_path": gripper_urdf,
               "transform": gripper_transform,
           },
       ]
   )

Control Parts
-------------

Group joints logically for different control modes:

.. code-block:: python

   control_parts = {
       "arm": ["JOINT1", "JOINT2", "JOINT3", "JOINT4", "JOINT5", "JOINT6"],
       "gripper": ["JOINT7", "JOINT8"],
   }

Use regex patterns for flexible matching:
- ``"JOINT[1-6]"`` matches JOINT1 through JOINT6
- ``"(LEFT|RIGHT)_ARM.*"`` matches all arm joints

Drive Properties
----------------

Configure joint physics behavior:

.. code-block:: python

   drive_pros = JointDrivePropertiesCfg(
       stiffness={
           "ARM_JOINTS": 1e4,    # High stiffness for arm joints
           "GRIPPER_JOINTS": 3e2,  # Lower stiffness for gripper
       },
       damping={
           "ARM_JOINTS": 1e3,
           "GRIPPER_JOINTS": 3e1,
       },
       max_effort={
           "ARM_JOINTS": 1e5,
           "GRIPPER_JOINTS": 1e3,
       },
   )

IK Solver Configuration
-----------------------

Choose the appropriate solver for your robot:

- **OPWSolverCfg**: For 6-axis industrial arms (like CobotMagic)
- **SRSSolverCfg**: For robots with specific kinematics (like DexforceW1)
- **SolverCfg**: Generic solver configuration

.. code-block:: python

   solver_cfg = {
       "arm": OPWSolverCfg(
           end_link_name="link6",
           root_link_name="base_link",
           tcp=np.array([...]),  # Tool center point
       ),
   }

---

Using Your Robot
~~~~~~~~~~~~~~~~

After adding the robot, use it in your code:

.. code-block:: python

   from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
   from embodichain.lab.sim.robots import MyRobotCfg

   # Create simulation
   sim_cfg = SimulationManagerCfg(headless=False, num_envs=2)
   sim = SimulationManager(sim_cfg)

   # Create robot config
   robot_cfg = MyRobotCfg.from_dict({
       "uid": "my_robot",
   })

   # Add robot to simulation
   robot = sim.add_robot(cfg=robot_cfg)

---

Testing Your Robot
~~~~~~~~~~~~~~~~~~

Add a test block at the bottom of your robot config file:

.. code-block:: python

   if __name__ == "__main__":
       from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

       sim_cfg = SimulationManagerCfg(headless=True, num_envs=2)
       sim = SimulationManager(sim_cfg)

       robot_cfg = MyRobotCfg.from_dict({"uid": "my_robot"})
       robot = sim.add_robot(cfg=robot_cfg)

       print("Robot added successfully!")

---

Best Practices
~~~~~~~~~~~~~~

1. **Use the** ``@configclass`` **decorator** for all config classes
2. **Provide** ``from_dict`` **method** for flexible initialization
3. **Use regex patterns** for joint names in drive properties
4. **Keep kinematics parameters** separate in ``params.py`` for complex robots
5. **Include** ``build_pk_serial_chain`` **method** for IK support
6. **Add** ``to_dict`` **and** ``save_to_file`` **methods** for serialization
7. **Test with** ``__main__`` **block** before integrating
8. **Add robot documentation** in ``docs/source/resources/robot/`` for user reference

---

Adding Robot Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When adding a new robot, create documentation in ``docs/source/resources/robot/`` to help users understand and use your robot.

File Location
-------------

Create a markdown file: ``docs/source/resources/robot/my_robot.md``

Recommended Structure
---------------------

.. code-block:: markdown

   # MyRobot

   Brief description of the robot and its manufacturer.

   <div style="text-align: center;">
     <img src="../../_static/robots/my_robot.jpg" alt="MyRobot" style="height: 400px; width: auto;"/>
     <p><b>MyRobot</b></p>
   </div>

   ## Key Features

   - Feature 1
   - Feature 2
   - Feature 3

   ---

   ## Robot Parameters

   | Parameter | Description |
   |-----------|-------------|
   | Joints    | Number of joints |
   | DOF       | Degrees of freedom |
   | ...       | ... |

   ---

   ## Quick Initialization Example

   ```python
   from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
   from embodichain.lab.sim.robots import MyRobotCfg

   config = SimulationManagerCfg(headless=False, sim_device="cpu", num_envs=2)
   sim = SimulationManager(config)

   robot = sim.add_robot(cfg=MyRobotCfg.from_dict({}))
   ```

   ---

   ## Configuration Parameters

   ### Main Configuration Items

   - **uid**: Unique identifier
   - **urdf_cfg**: URDF configuration
   - **control_parts**: Control groups
   - **solver_cfg**: IK solver configuration
   - **drive_pros**: Joint drive properties
   - **attrs**: Physical attributes

   ### Custom Usage Example

   ```python
   custom_cfg = {
       "uid": "my_robot",
       # Add parameters
   }
   cfg = MyRobotCfg.from_dict(custom_cfg)
   robot = sim.add_robot(cfg=cfg)
   ```

   ---

   ## References

   - Manufacturer product page
   - URDF file paths
   - Related documentation

Register the Robot in Index
---------------------------

After creating the robot documentation, add it to the index file at ``docs/source/resources/robot/index.rst``:

.. code-block:: rst

   .. toctree::
      :maxdepth: 1

      Dexforce W1 <dexforce_w1.md>
      CobotMagic <cobotmagic.md>
      MyRobot <my_robot.md>  # Add your robot here

---

Next Steps
~~~~~~~~~~

After adding your robot:

- Add robot documentation in ``docs/source/resources/robot/``
- Update ``docs/source/resources/robot/index.rst`` to include the new robot
- Add task environments that use your robot
- Configure sensors (cameras, force sensors)
- Implement custom IK solvers if needed
- Add motion planning support
