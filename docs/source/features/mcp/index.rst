EmbodiChain MCP
===============

DexSim Runtime owns the simulator session and executes operations on the
World's owner thread. DexSim MCP exposes native scene, articulation and
simulation operations. EmbodiChain MCP adds robot configuration and
control-part semantics. Both servers can connect independently to the same
Runtime session. EmbodiChain MCP needs only that session for its queries.

A stdio tool call follows this path:

.. code-block:: text

   MCP client → EmbodiChain MCP server → DexSim Runtime Session.invoke
              → EmbodiChain host adapter → articulation in the current scene

The host resolves the articulation on each request. The stdio connection
can close and reopen during the lifetime of one World.
See :doc:`../../guides/mcp` for installation and a working client, or the
:doc:`architecture overview <../../overview/architecture/index>` for the project map.

Robot state
-----------

``embodichain_get_robot_state`` returns one measurement without stepping the
World or changing drive targets. This tool supports scalar angular and linear
joint positions on the DexSim backend and Newton MuJoCo-Warp.

.. code-block:: json

   {"object_ref": "ur5_b", "robot_cfg": "URRobot", "variant": "ur5", "control_part": "arm"}

``object_ref`` selects the live scene instance. ``robot_cfg`` selects the
registered configuration and ``variant`` optionally selects a declared model
variant. ``control_part`` supplies the configured joint selectors and their
order. These checks establish that the selected joints can be read.

Example structured result from the two-UR5 test with the released DexSim 0.6.0
wheel on Newton MuJoCo-Warp (positions rounded):

.. code-block:: json

   {
     "session_id": "dsws_8ffba19b8ff348adb0050345f77eed45",
     "object_ref": "ur5_b", "topology_revision": 0, "control_part": "arm",
     "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
     "dof_ids": [0, 1, 2, 3, 4, 5],
     "qpos": [-0.60, 0.50, -0.40, 0.30, -0.20, 0.10],
     "units": ["rad", "rad", "rad", "rad", "rad", "rad"],
     "step_index": 0, "simulation_time_s": 0.0
   }

At each array index, ``joint_names``, ``dof_ids``, ``qpos`` and ``units``
describe the same joint. DOF IDs are public state-buffer offsets and can be
nonconsecutive. Revolute joints use radians and prismatic joints use metres.
The topology revision identifies the scene layout. The native DexSim backend
returns JSON null for both clock fields.

An unknown object path, incompatible control part, overlapping joint
selectors, or a selected multi-DOF joint returns a tool error. Use the
session's current object paths and the selected model's control groups to
diagnose the mismatch.
