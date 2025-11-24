# Dexforce W1

Dexforce W1 is a versatile robot developed by DexForce Technology Co., Ltd., supporting both industrial and anthropomorphic arm types. It is suitable for various simulation and real-world application scenarios.

## Key Features

- Supports multiple arm types (industrial, anthropomorphic)
- Configurable left/right hand brand and version
- Flexible URDF assembly and simulation configuration
- Compatible with SimulationManager simulation environment


## Usage in Simulation Environment

"""
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1HandBrand, DexforceW1ArmSide, DexforceW1ArmKind, DexforceW1Version
)
from embodichain.lab.sim.robots.dexforce_w1.utils import build_dexforce_w1_cfg

config = SimulationManagerCfg(headless=False, sim_device="cpu")
sim = SimulationManager(config)
sim.build_multiple_arenas(4)

hand_types = {
    DexforceW1ArmSide.LEFT: DexforceW1HandBrand.BRAINCO_HAND,
    DexforceW1ArmSide.RIGHT: DexforceW1HandBrand.BRAINCO_HAND,
}
hand_versions = {
    DexforceW1ArmSide.LEFT: DexforceW1Version.V021,
    DexforceW1ArmSide.RIGHT: DexforceW1Version.V021,
}

cfg = build_dexforce_w1_cfg(
    arm_kind=DexforceW1ArmKind.ANTHROPOMORPHIC,
    hand_types=hand_types,
    hand_versions=hand_versions,
)

robot = sim.add_robot(cfg=cfg)
print("DexforceW1 robot added to the simulation.")
```

## Type Descriptions


| Type                    | Options / Values                                      | Description                        |
|-------------------------|-------------------------------------------------------|------------------------------------|
| `DexforceW1ArmKind`     | `ANTHROPOMORPHIC`, `INDUSTRIAL`                       | Arm type                           |
| `DexforceW1HandBrand`   | `BRAINCO_HAND`, `DH_PGC_GRIPPER`, `DH_PGC_GRIPPER_M`  | Hand brand                         |
| `DexforceW1Version`     | `V021`                                                | Release version                    |
| `DexforceW1ArmSide`     | `LEFT`, `RIGHT`                                       | Left/right hand identifier         |
