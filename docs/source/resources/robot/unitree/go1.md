# Unitree Go1

`UnitreeGo1Cfg` configures the bundled 12-joint Unitree Go1 quadruped,
including its floating base, standing pose and simulation PD drive defaults.

<div style="text-align: center;">
  <img src="../../../_static/robots/unitree_go1.png" alt="Unitree Go1 robot" style="height: 400px; width: auto;"/>
  <p><b>Unitree Go1 in the default joint pose</b></p>
</div>

## Usage

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain_robots import UnitreeGo1Cfg

sim = SimulationManager(SimulationManagerCfg(headless=True, num_envs=1))
cfg = UnitreeGo1Cfg.from_dict({})
robot = sim.add_robot(cfg=cfg)
```

The catalog ID is also accepted by the shared preview command:

```bash
embodichain preview-robot --robot UnitreeGo1
```

## Robot Configuration

| Item | Default |
| --- | --- |
| Catalog ID / UID | `UnitreeGo1` |
| Configuration class | `UnitreeGo1Cfg` |
| Canonical module | `embodichain_robots.unitree.go1` |
| Vendor | `unitree` |
| Morphology | `quadruped` |
| Arm count | 0 |
| Asset | `UnitreeGo1Locomotion/go1.usda` |
| Control parts | `locomotion`: 12 joints |
| Base | Floating |
| Initial root height | 0.278 m |
| Physics backends | Default and Newton |
| Limb IK solvers | None; `build_pk_chain=False` |

Standard `RobotCfg` overrides, including `joint_drive_props`, are accepted
by `from_dict`. Locomotion tasks preserve these explicit overrides.

## See Also

- {doc}`Shared locomotion defaults and render settings </resources/robot/locomotion>`
- {doc}`Locomotion robot API reference </api_reference/locomotion_robots>`
