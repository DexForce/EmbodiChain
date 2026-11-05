# Sharpa Wave 01

`embodichain_robots.SharpaWaveCfg` configures a standalone left or right
Sharpa Wave 01. The robot catalog ID is `SharpaWave`; `left` and `right` select the
native models, with right as the default.

<figure style="text-align: center; margin: 20px auto; max-width: 960px;">
  <img src="../../../_static/robots/eef/sharpa_wave_left.png" alt="Native Sharpa Wave 01 left hand at its initial joint pose" width="1600" height="1080" style="display: block; width: 100%; height: auto;"/>
  <figcaption><b>Sharpa Wave 01 · left hand · 22 independent joints</b></figcaption>
</figure>

[Render manifest](../../../_static/robots/eef/sharpa_wave_left.json).

<figure style="text-align: center; margin: 20px auto; max-width: 960px;">
  <img src="../../../_static/robots/eef/sharpa_wave_right.png" alt="Native Sharpa Wave 01 right hand at its initial joint pose" width="1600" height="1080" style="display: block; width: 100%; height: auto;"/>
  <figcaption><b>Sharpa Wave 01 · right hand · 22 independent joints</b></figcaption>
</figure>

[Render manifest](../../../_static/robots/eef/sharpa_wave_right.json).

The illustrations render the published visual geometry and authored materials
in native OfflineRT, at the preset's initial joint coordinates and a presentation
root transform. No physics steps or visual-mesh/material replacements are used.
Each manifest records the display lighting (environment intensity 40,
half-intensity studio lights placed outside the camera view), camera and
source/image hashes. See [EEF rendering conventions](../end_effectors.md#rendered-illustrations).

## Construct, discover and preview

Given an existing `SimulationManager` named `sim`:

```python
from embodichain.robot_registry import resolve_robot_cfg
from embodichain_robots import SharpaWaveCfg

cfg = resolve_robot_cfg("SharpaWave", variant="left")
# Equivalent model-local selection:
cfg = SharpaWaveCfg.from_dict({"side": "left"})
restored = SharpaWaveCfg.from_dict(cfg.to_dict())
robot = sim.add_robot(cfg=restored)
```

```bash
embodichain list-robot
embodichain preview-robot --robot SharpaWave --variant left --viser
embodichain preview-robot --robot SharpaWave --variant right --viser
embodichain preview-robot --robot SharpaWave --variant right --headless --device cpu --renderer no-render --physics default --max-steps 100
```

The model package also lazily exports `SharpaWaveCfg` from
`embodichain_robots.sharpa.wave`. Listing catalogs does not
import configurations or download assets. Unsupported sides fail before asset
lookup. `from_dict()` builds defaults and applies isolated overrides through the
standard `RobotCfg` path. An existing local `fpath` bypasses asset download.
Caller-defined control-group names are retained.

## Control contract

`eef` commands all **22 independent revolute joints**, in radians.
Source order is thumb (CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP), then index,
middle and ring (MCP_FE, MCP_AA, PIP, DIP each), then pinky
(CMC, MCP_FE, MCP_AA, PIP, DIP). Names retain their `left_` / `right_` prefix.
There are no mimic followers. All 22 joints start at zero, within their source position limits.

```python
q = robot.get_qpos(name="eef").clone()
robot.set_qpos(q + 0.08, name="eef")
# Advance the simulator to execute the position command.
sim.update(step=100)
robot.reset()
```

`solver_cfg` and `build_pk_serial_chain()` are empty. The hand is a branched
articulation: no finger IK, single tool TCP, open/close service, grasping policy,
retargeting or tactile simulation is implied. Catalog metadata declares
`dexterous_hand`, zero arms and no generic dual-arm base compatibility.

## Physical defaults

The palm is fixed to the world at `(0, 0, 0.3)` metres with the source orientation.
This root placement clears the standalone ground and is not a calibrated mount.
Self-collision is disabled by the generic root defaults; callers can explicitly
configure it for their application. Native joint/link names, position, effort
and velocity limits, masses, COMs and inertias remain source-owned.

Force-drive stiffness **3 N·m/rad** and damping **0.1 N·m·s/rad** are initial
simulation PD defaults, not measured motor gains or hardware calibration.
Source effort/velocity limits and link inertials are explicitly reapplied
through portable drive/link fields because native importer defaults can
otherwise replace them. These overlays reproduce source values without
retuning the physical model. Caller configuration can
replace the gains and physical defaults. Validation covers Default-backend CPU
spawn, joint commands and reset for both sides; it does not qualify contact-rich
manipulation or other physics backends.

## Assets and provenance

```python
from embodichain.data import get_data_path
from embodichain_robots.sharpa.wave.assets import SharpaWave

right_urdf = get_data_path("SharpaWave/right_sharpa_wave/right_sharpa_wave.urdf")
```

The [published V1 archive](https://huggingface.co/datasets/DexForceAI/embodichain_data/blob/541650b05674ead90b58e9edcac650e87d1a35a6/eef_assets/SharpaWaveV1.zip)
contains both sides and uses the `SharpaWaveV1` cache. Source:
[sharpa-robotics/sharpa-urdf-usd-xml at `0d19cac602f4`](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/tree/0d19cac602f46456b819e4b6a2c09a74982c9a3e).
The model-owned [provenance record](https://github.com/DexForce/EmbodiChain/blob/main/extensions/embodichain_robots/sharpa/wave/README.md)
records licensing, preparation, archive hashes and native MJCF resources.
Host assemblies own mount transforms, naming prefixes and task-specific interfaces.
