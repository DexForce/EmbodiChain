# ANYmal-C

`ANYmalCCfg` configures the bundled 12-joint ANYmal-C quadruped,
including its floating base, standing pose and simulation PD drive defaults.

<div style="text-align: center;">
  <img src="../../_static/robots/anymal_c.png" alt="ANYmal-C robot" style="height: 400px; width: auto;"/>
  <p><b>ANYmal-C in the default joint pose</b></p>
</div>

## Usage

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain_robots import ANYmalCCfg

sim = SimulationManager(SimulationManagerCfg(headless=True, num_envs=1))
cfg = ANYmalCCfg.from_dict({})
robot = sim.add_robot(cfg=cfg)
```

The catalog ID is also accepted by the shared preview command:

```bash
embodichain preview-robot --robot ANYmalC
```

## Robot Configuration

| Item | Default |
| --- | --- |
| Catalog ID / UID | `ANYmalC` |
| Configuration class | `ANYmalCCfg` |
| Asset | `ANYmalCLocomotion/usd/anymal_c.usda` |
| Control parts | `locomotion`: 12 joints |
| Base | Floating |
| Initial root height | 0.6 m |
| Physics backends | Default and Newton |
| Limb IK solvers | None; `build_pk_chain=False` |

Standard `RobotCfg` overrides, including `joint_drive_props`, are accepted
by `from_dict`. Locomotion tasks preserve these explicit overrides.

## See Also

- [Shared locomotion defaults and render settings](locomotion.md)
- [Locomotion robot API reference](../../api_reference/locomotion_robots.rst)
