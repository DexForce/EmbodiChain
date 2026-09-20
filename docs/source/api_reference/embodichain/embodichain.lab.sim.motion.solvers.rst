embodichain.lab.sim.motion.solvers
==========================================

.. automodule:: embodichain.lab.sim.motion.solvers

Overview
--------

Inverse-kinematics solvers for robot control parts. Every solver implements the
:class:`BaseSolver` interface (forward kinematics, IK, Jacobian, TCP, and joint
limits) and is constructed from a :class:`SolverCfg` subclass whose
``init_solver()`` factory produces the runtime instance inside
:class:`~embodichain.lab.sim.objects.RobotCfg`. A robot may carry one solver per
control part. All solvers share a ``pytorch_kinematics`` serial chain for FK and
Jacobian computation, with ``torch.compile`` applied to the FK path.

Available implementations: analytic/closed-form (``SRS``, ``OPW``, ``UR``, ``FEP``),
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
    FEPSolverCfg
    FEPSolver
    PinocchioSolverCfg
    PinocchioSolver
    PinkSolverCfg
    PinkSolver
    DifferentialSolverCfg
    DifferentialSolver
    NeuralIKSolverCfg
    NeuralIKSolver

.. currentmodule:: embodichain.lab.sim.motion.solvers

Base Solver
-----------

``Robot.compute_batch_ik(..., continuous=True)`` checks
``BaseSolver.supports_continuous_batch_ik`` before generating any candidates.
The capability defaults to ``False``; OPW opts in and implements the protected
``_select_continuous_ik_path`` hook. Supporting solvers accept
``get_ik(return_all_solutions=True)`` and return validity ``(M, K)`` and joint
candidates ``(M, K, DOF)``. Selection consumes ``(B, N, K, DOF)`` candidates,
``(B, N, K)`` validity, and an initial ``(B, DOF)`` seed, then returns
``(B, N)`` validity and ``(B, N, DOF)`` joint positions. Other solvers retain
ordinary batch IK and reject continuous requests before solving.

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

FEP Solver
----------

Geometric IK for Franka-compatible screw-axis layouts, following the fixed-q7
construction in HolisticMotion/`GeoFIK <https://arxiv.org/abs/2503.03992>`_.
Dimensions, axis signs and fixed transforms are extracted from the URDF;
incompatible seven-axis chains raise an error. CPU and CUDA share one Warp
kernel. There is no DLS iteration or numerical fallback.

The seed fixes q7 by default and selects the nearest valid branch. Without a
seed, the joint-limit midpoint is used. ``return_all_solutions=True`` returns
up to eight branches with validity masks; invalid slots contain the clamped
seed. Every accepted solution satisfies the actual URDF FK, TCP and joint limits.

Enable ``redundancy_search`` to sample and refine q7 using seed distance,
arm-angle preference and joint-limit margin. ``max_joint_step`` remains a hard
per-joint displacement bound in radians. Pass the previous solution as the next
seed for sequential motion. Search returns the best sampled solutions, not an
exhaustive continuous family; failure does not prove global unreachability.

``get_arm_angle(qpos)`` measures the TCP-independent GeoFIK swivel angle in
radians, returning NaN at undefined planes. It differs from q7. During search,
``get_ik(..., arm_angle=...)`` overrides ``cfg.arm_angle``. With neither supplied,
the seed's angle is preferred when defined. This is a soft posture preference, not a
complete humanlike or collision model. See the API below for shapes and options.

Run ``python -m scripts.tutorials.sim.fep_solver --device cuda --radius 0.15``
from the repository root (or use ``--device cpu``). Add ``--redundancy-search``
and optionally ``--arm-angle 0.3`` for posture optimization, or
``--headless --max-steps 301`` for a finite run. The circle example draws target
and actual TCP paths and reports IK and physical tracking errors separately.

.. currentmodule:: embodichain.lab.sim.motion.solvers.fep_solver

.. autosummary::

    FEPSolverCfg
    FEPSolver

.. currentmodule:: embodichain.lab.sim.motion.solvers

.. autoclass:: FEPSolverCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: FEPSolver
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

``URSolverCfg`` selects the analytical DH parameters for UR3, UR5, UR10 and
their e-series variants. ``URSolver.get_ik()`` returns validity flags followed
by joint positions. By default, a dedicated Warp kernel selects the nearest
valid candidate using the joint seed and per-joint weights, returning shapes
``(N,)`` and ``(N, 6)`` without allocating the complete candidate tensor.

With ``return_all_solutions=True``, the solver returns shapes ``(N, 512)`` and
``(N, 512, 6)``. These candidates contain eight analytical branches expanded
over 64 combinations of periodic joint representatives, with validity flags
for FK agreement and joint limits. When no shifted representative fits a
joint's limits, its base value is repeated.

.. currentmodule:: embodichain.lab.sim.motion.solvers.ur_solver

.. autosummary::

    URSolverCfg
    URSolver

.. currentmodule:: embodichain.lab.sim.motion.solvers

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

Seed Selection
--------------

.. currentmodule:: embodichain.lab.sim.motion.solvers.qpos_seed_sel_sampler

.. autoclass:: QposSeedSelSampler
    :members:
    :show-inheritance:
