embodichain.mcp
===============

The MCP package exposes phase-one, coarse-grained project operations.  The
protocol layer is optional; install the ``mcp`` extra before starting the
command-line server.  Simulation is one provider, while URDF assembly is a
provider-independent asset workflow with optional simulation verification.

.. currentmodule:: embodichain.mcp

.. autosummary::

   MCPAdapter
   MCPAdapterRegistry
   EmbodiChainMCPService
   InMemorySimulationBackend
   SimulationBackend
   SimulationManagerBackend
   SimulationMCPAdapter
   URDFAssemblyAdapter
   create_server
   serve
   cli

.. autoclass:: EmbodiChainMCPService
   :members:

.. autoclass:: MCPAdapter

.. autoclass:: MCPAdapterRegistry
   :members:

.. autoclass:: InMemorySimulationBackend
   :members:

.. autoclass:: SimulationBackend

.. autoclass:: SimulationManagerBackend
   :members:

.. autoclass:: SimulationMCPAdapter
   :members:

.. autoclass:: URDFAssemblyAdapter
   :members:

.. autofunction:: create_server

.. autofunction:: serve

.. autofunction:: cli

embodichain.mcp.backend
=======================

.. automodule:: embodichain.mcp.backend
   :members:

embodichain.mcp.adapters
========================

.. automodule:: embodichain.mcp.adapters
   :members:

embodichain.mcp.urdf
====================

.. automodule:: embodichain.mcp.urdf
   :members:

embodichain.mcp.simulation
==========================

.. automodule:: embodichain.mcp.simulation
   :members:

embodichain.mcp.urdf_simulation
===============================

.. automodule:: embodichain.mcp.urdf_simulation
   :members:

embodichain.mcp.server
======================

.. automodule:: embodichain.mcp.server
   :members:

embodichain.mcp.service
=======================

.. automodule:: embodichain.mcp.service
   :members:
