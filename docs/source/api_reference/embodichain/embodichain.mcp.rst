MCP and DexSim session API
==========================

See :doc:`../../guides/mcp` to connect a client and
:doc:`../../features/mcp/index` for the tool request and response.

Server lifecycle
----------------

.. currentmodule:: embodichain.mcp.server

.. autosummary::

   create_server
   serve

.. autofunction:: create_server
.. autofunction:: serve

The command-line adapter is :func:`embodichain.cli.mcp.main`.

.. autofunction:: embodichain.cli.mcp.main

Host measurements
-----------------

A control-part observation contains aligned joint names, measured positions,
measured velocities, and position units. A DexSim Session result adds object
identity, topology-specific DOF indices, and the available clock to those
same fields.

.. autoclass:: embodichain.lab.observations.joints.ControlPartJointState

.. autoclass:: embodichain.lab.observations.joints.SessionJointState

.. currentmodule:: embodichain.lab.sim.dexsim_session.robot_state

.. autosummary::

   read_robot_state
   run

.. autofunction:: read_robot_state
.. autofunction:: run

The package-level aliases :func:`embodichain.lab.sim.dexsim_session.read_robot_state`
and :func:`embodichain.lab.sim.dexsim_session.run` preserve the original
Python and Runtime invocation paths.

.. currentmodule:: embodichain.lab.sim.dexsim_session

.. autosummary::

   read_robot_state
   run
