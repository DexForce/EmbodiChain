# Unitree Go2

`UnitreeGo2Cfg` configures the bundled 12-joint Unitree Go2 quadruped,
including its floating base, standing pose and simulation PD drive defaults.

<div style="text-align: center;">
  <img src="../../_static/robots/unitree_go2.png" alt="Unitree Go2 robot" style="height: 400px; width: auto;"/>
  <p><b>Unitree Go2 in the default joint pose</b></p>
</div>

## Usage

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain_robots import UnitreeGo2Cfg

sim = SimulationManager(SimulationManagerCfg(headless=True, num_envs=1))
cfg = UnitreeGo2Cfg.from_dict({})
robot = sim.add_robot(cfg=cfg)
```

The catalog ID is also accepted by the shared preview command:

```bash
embodichain preview-robot --robot UnitreeGo2
```

## Robot Configuration

| Item | Default |
| --- | --- |
| Catalog ID / UID | `UnitreeGo2` |
| Configuration class | `UnitreeGo2Cfg` |
| Canonical module | `embodichain_robots.unitree.go2` |
| Vendor | `unitree` |
| Morphology | `quadruped` |
| Arm count | 0 |
| Asset | `UnitreeGo2Locomotion/go2.usda` |
| Control parts | `locomotion`: 12 joints |
| Base | Floating |
| Initial root height | 0.32 m |
| Physics backends | Default and Newton |
| Limb IK solvers | None; `build_pk_chain=False` |

Standard `RobotCfg` overrides, including `joint_drive_props`, are accepted
by `from_dict`. Locomotion tasks preserve these explicit overrides.

## See Also

- [Shared locomotion defaults and render settings](locomotion.md)
- [Locomotion robot API reference](../../api_reference/locomotion_robots.rst)
