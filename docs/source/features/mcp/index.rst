EmbodiChain MCP
===============

DexSim Runtime owns the simulator session and executes operations on the
World's owner thread. DexSim MCP exposes native scene, articulation and
simulation operations. EmbodiChain MCP adds robot configuration and
control-part semantics. Both servers can connect independently to the same
Runtime session. EmbodiChain MCP needs only that session for its queries.

Follow :doc:`the MCP connection guide <../../guides/mcp>` to install the server
and read joint state from a running World.

The diagram follows a joint-state request through either server. Each server
connects using the same session ID and selects a host-side handler
through ``Session.invoke()``.

.. mermaid::

   flowchart TB
       client["MCP client process"]
       subgraph ec["EmbodiChain MCP server process"]
           ecTool["embodichain_get_robot_state"]
       end
       subgraph ds["DexSim MCP server process"]
           dsTool["dexsim_get_articulation_state"]
       end
       subgraph host["DexSim Runtime host process"]
           queue["Shared Runtime Session<br/>command queue"]
           subgraph owner["World owner thread"]
               ecRead["EmbodiChain host adapter<br/>Resolve control part and DOF order"]
               dsRead["DexSim native handler<br/>Resolve requested joints or DOF IDs"]
               joints["World / current scene / articulation<br/>get_current_qpos() and get_current_qvel()"]
           end
       end

       client -->|stdio| ecTool
       client -->|stdio| dsTool
       ecTool -->|Session.invoke| queue
       dsTool -->|Session.invoke| queue
       queue --> ecRead
       queue --> dsRead
       ecRead --> joints
       dsRead --> joints

The owner thread dispatches each request to its selected handler. For the
same articulation, both handlers read measured joint positions and velocities.
EmbodiChain returns them in configured control-part order. Each result goes
back through the requesting server; closing its stdio connection leaves the
World running.
See the :doc:`architecture overview <../../overview/architecture/index>` for
the project map.

Robot state
-----------

``embodichain_get_robot_state`` returns one measurement without stepping the
World or changing drive targets. This tool supports scalar angular and linear
joint positions and velocities on the DexSim backend and Newton MuJoCo-Warp.

.. code-block:: json

   {"object_ref": "ur5_b", "robot_cfg": "URRobot", "variant": "ur5", "control_part": "arm"}

``object_ref`` selects the live scene instance. ``robot_cfg`` selects the
registered configuration and ``variant`` optionally selects a declared model
variant. ``control_part`` supplies the configured joint selectors and their
order. These checks establish that the selected joints can be read.

Example structured result from the two-UR5 test with the released DexSim 0.6.0
wheel on Newton MuJoCo-Warp (measurements rounded):

.. code-block:: json

   {
     "session_id": "dsws_f02e5024064f4059a9ec76d117cc7519",
     "object_ref": "ur5_b", "topology_revision": 0, "control_part": "arm",
     "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
     "dof_ids": [0, 1, 2, 3, 4, 5],
     "qpos": [-0.60, 0.50, -0.40, 0.30, -0.20, 0.10],
     "qvel": [-0.21, 0.22, -0.23, 0.24, -0.25, 0.26],
     "units": ["rad", "rad", "rad", "rad", "rad", "rad"],
     "step_index": 0, "simulation_time_s": 0.0
   }

At each array index, ``joint_names``, ``dof_ids``, ``qpos``, ``qvel`` and ``units``
describe the same joint. DOF IDs are public state-buffer offsets and can be
nonconsecutive. The host reads current position and velocity in one call on the
World owner thread, without advancing physics between reads. For an angular
joint, ``units[i]`` is ``rad`` and velocity is in rad/s; for a prismatic joint
it is ``m`` and velocity is in m/s. An unavailable measurement fails the call.
The topology revision identifies the scene layout. The native DexSim backend
returns JSON null for both clock fields.

An unknown object path, incompatible control part, overlapping joint
selectors, or a selected multi-DOF joint returns a tool error. Use the
session's current object paths and the selected model's control groups to
diagnose the mismatch.
