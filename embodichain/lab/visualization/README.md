# Native WebRTC window preview (experimental)

`--web` replaces the local simulation window with a browser observation view.
It uses the existing DexSim World and renderer, with no mesh export or Python
video encoding. Viser remains available through `--viser`; the two flags are
mutually exclusive.

```bash
embodichain preview-robot --robot ur5 --web --web-port 8765
embodichain preview-asset --asset_path /path/to/model.urdf --web
embodichain run-task --gym_config /path/to/task.yaml --web
```

Install this checkout in the project's Python environment with its pinned
DexSim `0.6.0rc1` dependency. The package exposes the required WebRTC APIs;
video playback also requires loadable native streaming libraries and NVIDIA
NVENC support. Development builds can be preserved with
`python -m pip install --no-deps -e .` after installing project dependencies.

Open `http://127.0.0.1:8765/`. Click the main video before using mouse/keyboard
navigation. The page supports resetting the view, pausing viewer rendering,
requesting one frame, saving the current video image, entity picking/dragging,
and reconnection. Pausing viewing does not pause physics.

## Configuration and lifecycle

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.visualization import DexSimWebCfg, VisualizationCfg

sim = SimulationManager(SimulationManagerCfg(
    visualization=VisualizationCfg(
        backend="dexsim-web",
        allow_commands=False,  # Navigation works; entity editing is disabled.
        web=DexSimWebCfg(width=1280, height=720, fps=15, port=8765),
    ),
))
# Declare assets, then sim.prepare(). sim.update() publishes due viewer frames.
# While physics is paused, pump on the World creator thread:
# sim.capture_visualization()
# Always destroy the manager and flush deferred cleanup when finished.
```

Gym YAML accepts the same `visualization.web` mapping. `--web-port` and
`--web-fps` override only supplied values. Auto renderer selection uses Hybrid
for this experimental route. Headless is forced before World creation; this
means no local window, not NoRender.

The observation camera uses its own explicit camera group, even when its size
matches a task camera. Task cameras are read-only video sources and show their
latest task-rendered images; opening the browser does not add sensor renders
or change sensor intrinsics/extrinsics. Initial sensor video can remain black
until the task renders it. Video screenshots are presentation images, not a
replacement for original sensor data or dataset recording.

`capture_visualization()` pumps bounded application requests and renders only
the observer. It does not prepare an empty scene or advance physics. Browser
commands carry the application run ID and scene revision. All application
operations and cleanup stay on the World creator thread.

`stop_visualization()` closes application routes, editing and the observation
camera. DexSim currently keeps the HTTP/WebRTC listener until World teardown;
only `destroy()` followed by deferred cleanup releases the port. The viewer
can recreate its application in that World. `visualization_health` reports
application readiness/errors; its client count is not available through the
native Python facade (reported as zero), not a connectivity measurement.

## Current boundaries

- Requires a Linux/NVIDIA WebRTC-enabled DexSim build exporting `WebApplication`
  and the `WorldConfig.webrtc_*` settings, present in the pinned `0.6.0rc1`.
  Initial GPU/browser acceptance used development commit `8b867e321`; native
  streaming-library compatibility must be checked when changing SDK builds.
- Single simulation environment and single browser viewer. The Python native
  service auto-streams all cameras. Supported manager sensors are at most 15
  RGB `Camera` instances plus the observer; dimensions must be even, 32–4096.
  Contact sensors remain valid. Stereo/depth-only cameras are rejected before
  allocation. Cameras created outside the manager are outside this contract.
- Robot IK, joint sliders, native recording hotkeys, and Viser-style overlays
  or per-environment visibility are not implemented in this first stage.
  Viser and native windows retain their existing behavior.
- The listener binds **all IPv4 interfaces**, not only loopback. Use a trusted
  network or an access-controlled deployment. Native viewer tokens are session
  controls, not user-account authentication. ICE configuration can be provided
  through `web.ice`; forwarding only the HTTP port does not ensure media
  connectivity.
- Default physics was initially verified end to end with `8b867e321`. The
  current Newton adapter uses DexSim 0.6's `DexUniSolverCfg` and scene dispatch;
  browser acceptance exercises both backends without the obsolete
  `AutoSolverCfg` requirement. End-to-end Newton browser behavior still needs
  qualification with loadable native streaming libraries.

## Validation

CPU contract coverage lives in `tests/visualization/test_web_viewer.py`, with
CLI/manager/Gym cases in their existing test modules. Real browser acceptance
uses a bounded subprocess with one object and a same-resolution task camera:

```bash
# Use a compatible DexSim environment, Playwright, Chrome and the host display.
python -m pytest tests/visualization/test_web_viewer_browser.py --run-gpu -q
```

Set `WEBRTC_TEST_CHROME` when Chrome is not on PATH. Tests cover decoded pixels,
view input, physical object dragging, read-only task video, unchanged sensor
pixels/poses/intrinsics, paused/manual rendering without physics advancement,
screenshot download, reconnect and port release. Tests run serially and record
screenshots and child logs under pytest's temporary directory.
