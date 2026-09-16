# DexSim Unified Coupled Solver For Embodied AI

```{currentmodule} embodichain.lab.sim
```

Developed by the **DexForce Simulation Engine Team**, DexUni is the recommended
Newton solver configuration for EmbodiChain scenes that combine Articulated
Objects with cloth or a volume soft body. DexSim exposes it through
`DexUniSolverCfg`; EmbodiChain also accepts the short mapping name
`solver_type="dexuni"`.

DexUni is provided through the DexSim integration rather than Newton's public
native solver list. This page focuses on its user-facing capabilities. For
Newton-native solver details, use the
[official Newton Solver Guide](https://newton-physics.github.io/newton/stable/solvers/index.html).

## Capabilities

| Capability | Current support |
| :--- | :--- |
| Articulated Objects with cloth | Supported |
| Articulated Objects with volume soft bodies | Supported |
| Contact between Articulated Objects and deformables | Supported |
| Batched arenas, reset, state access, and rendering | Supported through `SimulationManager` |
| Fluid | Unsupported in the current version |
| Cable | Unsupported in the current version |

Use DexUni for tasks in which Articulated Objects manipulate cloth, garments,
cushions, or other supported deformable assets. If a scene has only deformables
or only rigid bodies, let AutoSolver select the simpler matching solver unless
the task needs an explicit choice.

The current version does not support fluid or cable simulation.

## Minimal configuration

AutoSolver selects DexUni when the complete scene contains both an articulation
and a supported deformable. To keep the choice explicit:

```python
from embodichain.lab.sim.cfg import NewtonPhysicsCfg

physics_cfg = NewtonPhysicsCfg(
    device="cuda:0",
    solver_cfg={"solver_type": "dexuni"},
    collision_cfg=None,
)
```

Add all Articulated Objects and deformables before
{meth}`SimulationManager.prepare`. DexUni requires GPU execution and
`requires_grad=False`. Set `collision_cfg=None` because this solver owns the
robot/deformable contact path it uses.

For prescribed motion, register the complete joint trajectory before
`prepare()`:

```python
sim.add_robot(robot_cfg)
sim.add_deformable_object(deformable_cfg)
sim.register_kinematic_joint_trajectory("robot", joint_positions)
sim.prepare()
```

The trajectory layout is `[num_envs, frames, dof]`, in the articulation's
public joint order.

## Current limitations

```{warning}
The current DexUni path does not solve or publish rigid-rigid or rigid-ground
contacts. `ContactSensor` is therefore unsupported on this path.
```

Full capability coverage, including fluid, cable, and the rigid-contact
limitations above, will be completed in the next release.

Qualify any task that mixes deformables with loose rigid objects against its
required contact pairs. See {doc}`runtime_qualification` for the complete
solver/device matrix and validation workflow.

## Examples and further reading

- [`pick_up_cloth.py`](https://github.com/DexForce/EmbodiChain/blob/main/examples/sim/demo/pick_up_cloth.py)
- [`press_softbody.py`](https://github.com/DexForce/EmbodiChain/blob/main/examples/sim/demo/press_softbody.py)
- [`fold_tshirt.py`](https://github.com/DexForce/EmbodiChain/blob/main/examples/sim/demo/fold_tshirt.py)
- [DexUni Solver reference](https://dexforce.feishu.cn/wiki/TrsWwJtGpiCqw4kZ8pGcXBDsngd)

The examples are scene-specific starting points rather than universal tuning
defaults. For deformable asset configuration and runtime data access, see
{doc}`../../../sim_soft_object` and {doc}`../../../sim_cloth`.
