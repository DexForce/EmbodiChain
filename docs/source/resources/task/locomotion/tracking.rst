Reference Motion Tracking
=========================

``UnitreeG1MotionRL-v1`` trains Unitree G1 to follow reference motion clips.
Use the ``g1_motion`` task key to discover its deployments and select the
environment for training.

.. code-block:: bash

   embodichain show-task embodichain.extensions.tasks:g1_motion

See :doc:`../deployments` for environment IDs in this release and
:doc:`/resources/models/neural_dynamics/rwm` for the model-based motion
tracking workflow. The `G1 motion example <https://github.com/DexForce/EmbodiChain/tree/main/examples/learning/motion_tracking>`_
contains the training and evaluation instructions.
