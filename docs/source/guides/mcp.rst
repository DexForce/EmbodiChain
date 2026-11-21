Read a robot in an existing DexSim World
========================================

DexSim Runtime owns the World. EmbodiChain MCP connects a local stdio client
to that World and reads measured joints using a robot's configured control
parts. A DexSim MCP server is optional. It can attach to the same session to
inspect native scene objects. See :doc:`../features/mcp/index` for the
component boundaries and tool contract.

Locate the session and the robot
--------------------------------

Use a local DexSim Runtime host with a finalized scene and a loaded robot.
Install the same EmbodiChain revision in the World host's Python environment
and the stdio server environment. Install the MCP extra on the server.
From a checkout of that revision:

.. code-block:: bash

   # In the host environment
   python -m pip install -e .

   # In the environment that will run the MCP server and the example client
   python -m pip install -e '.[mcp]'

List local Runtime sessions and choose the ID for the running host. Replace
the example ID below. Copy the object's exact scene path from the listing:

.. code-block:: bash

   dexsim session list
   export SESSION_ID=dsws_example
   dexsim session status "$SESSION_ID"
   dexsim session exec "$SESSION_ID" 'scene.get_paths()'

If the scene contains two UR5s at paths ``ur5_a`` and ``ur5_b``, select B
with ``object_ref="ur5_b"``. The configuration ID ``URRobot`` and variant
``ur5`` describe the model; ``arm`` selects its six-joint control part.
For another scene, substitute its actual path. An empty session list means
there is no discoverable host in the selected directory; check the host's
session ID and discovery location. Finalize a pending scene on its owner
before querying joints.

Connect, query and reconnect
----------------------------

Run this Python example in the server environment. The client starts the
stdio server and handles the MCP handshake and tool discovery:

.. code-block:: python

   import asyncio
   import os
   import sys

   from mcp import Client, StdioServerParameters

   async def read_once():
       server = StdioServerParameters(
           command=sys.executable,
           args=["-m", "embodichain", "mcp", "--session", os.environ["SESSION_ID"]],
       )
       async with Client(server) as client:
           print([tool.name for tool in (await client.list_tools()).tools])
           reply = await client.call_tool("embodichain_get_robot_state", {
               "object_ref": "ur5_b", "robot_cfg": "URRobot",
               "variant": "ur5", "control_part": "arm",
           })
           if reply.is_error:
               raise RuntimeError(reply.content)
           return reply.structured_content

   print(asyncio.run(read_once()))

Rerun the example to reconnect. Exiting the client closes its server process;
the host retains the World. A robot moved by the host between reads will
return its latest measured positions and velocities.

For another stdio client, set its executable to ``embodichain`` and its arguments
to ``mcp --session <ID>``. If you launch that command directly, it waits for
MCP requests on standard input. For a custom discovery directory use
``dexsim session --directory DIR list`` and append ``--session-directory DIR``
to the EmbodiChain server arguments.
