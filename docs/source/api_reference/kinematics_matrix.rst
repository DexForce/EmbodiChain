Matrix products for kinematics
==============================

The shared matrix helpers preserve Torch broadcasting and differentiation.
Under ``torch.compile``, FP32 and FP64 products with trailing 3-by-3 matrices
use pointwise reductions that can fuse with surrounding geometry. Eager
execution, other sizes and dtypes, and autocast regions use Torch matrix
multiplication. The dynamics model, motion-tracking task and tree kinematics
share these operations.

Package imports
---------------

.. currentmodule:: embodichain.compute.kinematics

.. autosummary::

   matrix_product
   matrix_vector_product

.. autofunction:: matrix_product
.. autofunction:: matrix_vector_product

Implementation module
---------------------

.. currentmodule:: embodichain.compute.kinematics.matrix

.. autofunction:: matrix_product
.. autofunction:: matrix_vector_product
