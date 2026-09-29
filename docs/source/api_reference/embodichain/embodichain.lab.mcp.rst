embodichain.lab.mcp
===================

The MCP adapter exposes phase-one, coarse-grained simulation and motion
operations.  The protocol layer is optional; install the ``mcp`` extra before
starting the command-line server.

.. currentmodule:: embodichain.lab.mcp

.. autosummary::

   EmbodiChainMCPService
   InMemorySimulationBackend
   SimulationBackend
   SimulationManagerBackend
   create_server
   serve
   cli

.. autoclass:: EmbodiChainMCPService
   :members:

.. autoclass:: InMemorySimulationBackend
   :members:

.. autoclass:: SimulationBackend

.. autoclass:: SimulationManagerBackend
   :members:

.. autofunction:: create_server

.. autofunction:: serve

.. autofunction:: cli

embodichain.lab.mcp.backend
===========================

.. automodule:: embodichain.lab.mcp.backend
   :members:

embodichain.lab.mcp.server
==========================

.. automodule:: embodichain.lab.mcp.server
   :members:

embodichain.lab.mcp.service
===========================

.. automodule:: embodichain.lab.mcp.service
   :members:
