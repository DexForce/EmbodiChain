# UR Family (UR3 / UR5 / UR10 + e variants)

`URRobotCfg` is a single config class covering the Universal Robots UR family —
UR3, UR3e, UR5, UR5e, UR10, UR10e — selected via the ``robot_type`` field. The
kinematic (DH) parameters are owned by ``URSolverCfg``; the robot config owns the
URDF, control parts, drive properties, and rigid-body attributes.

## Key Features

- **One class, six variants** — switch with ``robot_type`` (``"ur3"`` / ``"ur3e"``
  / ``"ur5"`` / ``"ur5e"`` / ``"ur10"`` / ``"ur10e"``).
- **Analytic UR IK** via ``URSolverCfg`` (Warp GPU kernel, 6-DOF closed-form).
- **Scale-aware defaults** — ``max_effort`` is sized per variant (UR3 < UR5 < UR10).
- **Forward kinematics** through ``build_pk_serial_chain`` (pytorch-kinematics),
  routed via ``_pk_urdf_path`` so it cannot drift from the simulation URDF.
- **Round-trippable** — ``URRobotCfg.from_dict(cfg.to_dict())`` reproduces the cfg.

## Usage

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.robots import URRobotCfg

sim = SimulationManager(SimulationManagerCfg(headless=True, num_envs=4))
cfg = URRobotCfg.from_dict({"robot_type": "ur5"})
robot = sim.add_robot(cfg=cfg)
```

## Robot Parameters

| Parameter          | Description                                                       |
|--------------------|-------------------------------------------------------------------|
| ``robot_type``     | UR variant: ``ur3`` / ``ur3e`` / ``ur5`` / ``ur5e`` / ``ur10`` / ``ur10e`` |
| Number of joints   | 6 revolute + 1 fixed (``ee_link``)                               |
| Control parts      | ``arm`` (6 joints)                                               |
| Root / end link    | ``base_link`` / ``ee_link``                                      |
| Solver             | ``URSolverCfg`` (analytic UR IK, Warp kernel)                    |
| Drive ``max_effort`` | UR3/UR3e ≈ 56 N·m · UR5/UR5e ≈ 150 N·m · UR10/UR10e ≈ 330 N·m (sim defaults, not factory specs) |

.. note::

   The UR5 URDF uses lowercase joint names (``joint1``..``joint6``); every other
   variant uses ``Joint1``..``Joint6``. ``URRobotCfg._build_defaults`` selects the
   correct joint-name casing per ``robot_type`` automatically.

## Variants at a glance

| ``robot_type`` | URDF                          | Reach (m) | Payload (kg) |
|----------------|-------------------------------|-----------|--------------|
| ``ur3``        | ``UniversalRobots/UR3/UR3.urdf``   | ~0.5  | 3  |
| ``ur3e``       | ``UniversalRobots/UR3e/UR3e.urdf``  | ~0.5  | 3  |
| ``ur5``        | ``UniversalRobots/UR5/UR5.urdf``    | ~0.85 | 5  |
| ``ur5e``       | ``UniversalRobots/UR5e/UR5e.urdf``  | ~0.85 | 5  |
| ``ur10``       | ``UniversalRobots/UR10/UR10.urdf``  | ~1.3  | 10 |
| ``ur10e``      | ``UniversalRobots/UR10e/UR10e.urdf`` | ~1.3 | 10 |

See Also
--------

- :doc:`/guides/add_robot` — Adding a new robot (quick reference)
- :doc:`/tutorial/add_robot` — Adding a new robot (full tutorial)
- :doc:`/overview/sim/solvers/index` — IK solver reference
