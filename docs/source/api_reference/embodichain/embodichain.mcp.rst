embodichain.mcp
===============

The MCP adapter exposes phase-one, coarse-grained simulation and motion
operations.  The protocol layer is optional; install the ``mcp`` extra before
starting the command-line server.

.. currentmodule:: embodichain.mcp

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

embodichain.mcp.backend
=======================

.. automodule:: embodichain.mcp.backend
   :members:

embodichain.mcp.server
======================

.. automodule:: embodichain.mcp.server
   :members:

embodichain.mcp.service
=======================

.. automodule:: embodichain.mcp.service
   :members:

Compatibility imports
=====================

The original ``embodichain.lab.mcp`` import path remains available for
callers that adopted the initial phase-one preview.

.. automodule:: embodichain.lab.mcp
   :members:

.. automodule:: embodichain.lab.mcp.backend
   :members:

.. automodule:: embodichain.lab.mcp.server
   :members:

.. automodule:: embodichain.lab.mcp.service
   :members:
