embodichain.lab.sim.solvers
==========================================

.. automodule:: embodichain.lab.sim.solvers

Overview
--------

Inverse-kinematics solvers for robot control parts. Every solver implements the
:class:`BaseSolver` interface (forward kinematics, IK, Jacobian, TCP, and joint
limits) and is constructed from a :class:`SolverCfg` subclass whose
``init_solver()`` factory produces the runtime instance inside
:class:`~embodichain.lab.sim.objects.RobotCfg`. A robot may carry one solver per
control part. All solvers share a ``pytorch_kinematics`` serial chain for FK and
Jacobian computation, with ``torch.compile`` applied to the FK path.

Available implementations: analytic/closed-form (``SRS``, ``OPW``, ``UR``),
numerical (``Pinocchio``, ``Pink`` with null-space posture tasks,
``Differential``), learning-based (``PytorchSolver``, ``NeuralIKSolver``).

  .. rubric:: Classes

  .. autosummary::
    SolverCfg
    BaseSolver
    SRSSolverCfg
    SRSSolver
    OPWSolverCfg
    OPWSolver
    URSolverCfg
    URSolver
    PytorchSolverCfg
    PytorchSolver
    PinocchioSolverCfg
    PinocchioSolver
    PinkSolverCfg
    PinkSolver
    DifferentialSolverCfg
    DifferentialSolver
    NeuralIKSolverCfg
    NeuralIKSolver

.. currentmodule:: embodichain.lab.sim.solvers

Base Solver
-----------

.. autoclass:: SolverCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: BaseSolver
    :members:
    :inherited-members:
    :show-inheritance:

PyTorch Solver
--------------

.. autoclass:: PytorchSolverCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: PytorchSolver
    :members:
    :inherited-members:
    :show-inheritance:

Pinocchio Solver
----------------

.. autoclass:: PinocchioSolverCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: PinocchioSolver
    :members:
    :inherited-members:
    :show-inheritance:

Pink Solver
-----------

.. autoclass:: PinkSolverCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: PinkSolver
    :members:
    :inherited-members:
    :show-inheritance:

Differential Solver
-------------------

.. autoclass:: DifferentialSolverCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: DifferentialSolver
    :members:
    :inherited-members:
    :show-inheritance:

OPW Solver
----------

.. autoclass:: OPWSolverCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: OPWSolver
    :members:
    :inherited-members:
    :show-inheritance:

SRS Solver
----------

.. autoclass:: SRSSolverCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: SRSSolver
    :members:
    :inherited-members:
    :show-inheritance:

UR Solver
---------

.. autoclass:: URSolverCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: URSolver
    :members:
    :inherited-members:
    :show-inheritance:

Neural IK Solver
----------------

.. autoclass:: NeuralIKSolverCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: NeuralIKSolver
    :members:
    :inherited-members:
    :show-inheritance:
