# State and frame sensors

From the repository root, with EmbodiChain and its declared DexSim SDK installed:

The scene panel is a DexSim Hybrid camera capture of the sampled state.
It requires an NVIDIA GPU with native camera rendering support, even when
physics runs on CPU. The demo does not open a simulation window.

```bash
python -m pip install matplotlib
python -m examples.sim.sensors.state.run --output /tmp/sensor-state
```

The example uses the Default physics backend on CPU and needs no window or native
renderer. Install `ffmpeg` for MP4 export or pass `--no-video` for PNG only.

The built-in `JointEncoder` and `FrameTransform` are constructed through
`SensorCfg.from_dict` and read after each completed physics step. The same revolute joint
spins in opposite directions in the two arenas, then reverses halfway through. The preview shows both
encoder traces and the pose of the tip frame relative to the base in arena 0.
The dashed RGB axes mark the base, and the solid axes mark the tip, with
red, green and blue denoting X, Y and Z. A vertical cursor shows which simulated
sample supplies the displayed frames.

The command writes `preview.png` and `preview.mp4` to the selected directory.

## PVA and IMU

```bash
python -m examples.sim.sensors.state.inertial --output /tmp/sensor-inertial
```

This uses Default CPU and finite differences. Three cubes are measured at rest,
in free fall, and spinning around Z with a sensor 0.18 m off the rotation axis.
PVA shows arena-frame vertical acceleration and IMU shows local angular
velocity and specific force. Specific force is near zero in free fall and near
9.81 m/s² along Z at rest. The spinning sensor measures centripetal specific
force along negative local X. Dots in the validity panel identify samples
where acceleration is unavailable, including startup and the mid-run reset.

### Acceleration source

```bash
python -m examples.sim.sensors.state.inertial --source finite_difference --output /tmp/inertial-fd
python -m examples.sim.sensors.state.inertial --backend newton --device cuda:0 --source native --output /tmp/inertial-native
python -m examples.sim.sensors.state.inertial --source auto --output /tmp/inertial-auto
```

Native measurements require a CUDA runtime and a DexSim SDK with Newton
MuJoCo Warp acceleration reports. The example selects
`solver_type="mujoco_warp"` and `rne_mode="last_substep"`. Native acquisition
requires a dynamic rigid body. Explicit `native` rejects unsupported targets
or backends. `auto` selects native when available and uses finite differences
on Default CPU. The plot displays the selected source and the free-fall and
centripetal references. For the offset sensor, the latter is
`-0.18 m × (6 rad/s)² = -6.48 m/s²` along local X.
