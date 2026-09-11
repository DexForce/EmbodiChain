Creating a cloth simulation
===========================

.. currentmodule:: embodichain.lab.sim

This tutorial shows how to create a cloth simulation using ``SimulationManager``. It covers procedurally generating a grid mesh, configuring a deformable cloth object, adding a rigid body for interaction, and running the simulation loop.

The Code
~~~~~~~~

The tutorial corresponds to the ``create_cloth.py`` script in the ``scripts/tutorials/sim`` directory.

.. dropdown:: Code for create_cloth.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/sim/create_cloth.py
      :language: python
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Generating the cloth mesh
--------------------------

Unlike the soft-body tutorial where a pre-existing mesh file is loaded, cloth objects are typically defined by a flat 2-D surface. The helper function ``create_2d_grid_mesh`` generates a rectangular grid mesh procedurally using PyTorch, then saves it to a temporary ``.ply`` file via Open3D so that the simulation can load it.

Loading a mesh from file also works for cloth objects, but generating a grid in code allows for easy customization of the cloth dimensions and resolution. 

The function accepts the physical dimensions (``width``, ``height``) and the number of subdivisions (``nx``, ``ny``). A finer grid gives more cloth-like wrinkle detail at the cost of simulation performance.

.. literalinclude:: ../../../scripts/tutorials/sim/create_cloth.py
   :language: python
   :start-at: def create_2d_grid_mesh
   :end-at:     return verts, faces

Configuring the simulation
--------------------------

The simulation environment is configured with :class:`SimulationManagerCfg`. For cloth simulation the device must be set to ``cuda``. The ``arena_space`` parameter controls the spacing between parallel environments so that objects in neighboring environments do not overlap.

.. literalinclude:: ../../../scripts/tutorials/sim/create_cloth.py
   :language: python
   :start-at: # Configure the simulation
   :end-at:     print("[INFO]: Scene setup complete!")

Adding a cloth object to the scene
------------------------------------

The grid mesh generated earlier is saved to disk and then passed to :meth:`SimulationManager.add_deformable_object`. The physical properties of the cloth are controlled through :class:`cfg.SurfaceDeformableObjectCfg` together with :class:`cfg.SurfaceDeformablePhysicsCfg`:

- :class:`cfg.MeshCfg` — references the ``.ply`` file written to the system temp directory
- :class:`cfg.SurfaceDeformablePhysicsCfg` — material parameters:

  - ``density`` — surface density (kg/m²)
  - ``surface_props.tri_ke`` / ``tri_ka`` — triangle elastic and area stiffness
  - ``surface_props.tri_kd`` — triangle damping
  - ``surface_props.edge_ke`` / ``edge_kd`` — bending stiffness and damping
  - ``add_springs`` / ``spring_ke`` / ``spring_kd`` — optional mesh springs

The object stores these properties in ``attrs``. Set ``particle_radius`` on
``SurfaceDeformableObjectCfg`` to control the particle contact radius.

.. literalinclude:: ../../../scripts/tutorials/sim/create_cloth.py
   :language: python
   :start-at:     cloth_verts, cloth_faces = create_2d_grid_mesh
   :end-at:         padding_box_cfg = RigidObjectCfg

Adding a rigid body for interaction
-------------------------------------

A small cubic rigid body (``padding_box``) is placed beneath the cloth so the cloth drapes over it. It is added with :meth:`SimulationManager.add_rigid_object` using :class:`cfg.RigidObjectCfg` and :class:`cfg.RigidBodyPhysicsCfg`:

- :class:`cfg.CubeCfg` — defines the box dimensions
- ``body_type="dynamic"`` — the box responds to physics; change to ``"static"`` for a fixed obstacle
- ``static_friction`` / ``dynamic_friction`` — surface friction keeps the cloth from sliding off too easily

.. literalinclude:: ../../../scripts/tutorials/sim/create_cloth.py
   :language: python
   :start-at:     padding_box_cfg = RigidObjectCfg(
   :end-at:     print("[INFO]: Add cloth object complete!")

The Code Execution
~~~~~~~~~~~~~~~~~~

To run the script and see the result, execute the following command:

.. code-block:: bash

   python scripts/tutorials/sim/create_cloth.py

A window should appear showing a cloth panel falling and draping over a small rigid box. To stop the simulation, close the window or press ``Ctrl+C`` in the terminal.

You can also pass arguments to customise the simulation. For example, to run in headless mode with ``n`` parallel environments:

.. code-block:: bash

   python scripts/tutorials/sim/create_cloth.py --headless --num_envs <n>

To view the simulated cloth surface through Viser:

.. code-block:: bash

   python scripts/tutorials/sim/create_cloth.py \
       --viser \
       --viser-soft-body-fps 5

The browser mesh uses the cloth's physical vertices and a welded mapping of
the source triangles, so its topology matches the simulated cloth surface.
Deformable updates are sampled separately from rigid-body poses; adjust
``--viser-soft-body-fps`` to balance smoothness and browser/upload cost.

See :doc:`/overview/sim/viser_visualization` for details.

Now that we have a basic understanding of how to create a cloth scene, let's move on to more advanced topics.
