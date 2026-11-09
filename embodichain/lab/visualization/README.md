# Web presentation for native rendering

`--web` selects a browser viewport instead of a local window. It is a common
simulation presentation option, alongside `--headless`. It uses the existing
DexSim World, scene, materials, lights and renderer. Viser remains a separate
browser renderer, selected with `--viser`.

```bash
embodichain preview-robot --robot ur5 --web
embodichain preview-asset --asset_path /path/to/model.urdf --web --preview
embodichain preview-material --material GlassClear --web
embodichain run-task --gym_config /path/to/task.yaml --web
embodichain run-atomic-action --skill pickup --web
```

Open `http://127.0.0.1:8765/`. `--web-port` and `--web-fps` override the
HTTP/signaling port and observer rendering rate. Web and Viser are mutually
exclusive. `--headless --web` keeps local windows closed while presenting the
scene in the browser.

The page displays live video, connection status and the EmbodiChain
initialization table. It shares the terminal's structured summary rows rather
than parsing logs. Standalone runs show runtime, renderer, physics and scene
readiness; Gym runs publish their combined environment/manager initialization
snapshot. `startup_summary: compact|full|off` also controls these details.

The page is read-only: browser input is disabled and `/api/command` rejects
application commands even for older configurations with `allow_commands=True`.
Object editing, robot controls, keyboard shortcuts and replay sliders are owned
by the native/Viser frontends or Python host. The Web integration supplies an
observation view and initialization information, leaving general viewer GUI
functionality to DexSim.

## Programmatic configuration

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.visualization import DexSimWebCfg, VisualizationCfg

sim = SimulationManager(SimulationManagerCfg(
    web=True,
    num_envs=8,
    visualization=VisualizationCfg(
        allow_commands=False,
        web=DexSimWebCfg(width=1280, height=720, fps=15, port=8765),
    ),
))
# Declare assets and call sim.prepare().
sim.open_viewer()
sim.viewer.set_look_at([2.5, -3.5, 2], [0, 0, .6], [0, 0, 1])
# Normal sim.update() calls publish due frames.
# In paused/render-only loops, call sim.poll_viewer() on the World creator thread.
# Use sim.wait_for_input("Continue? ") instead of a blocking input() in UI loops.
# Finally destroy the manager and flush deferred cleanup.
```

Gym configuration accepts top-level `web: true`, plus the optional
`visualization.web` settings. The original `visualization.backend: dexsim-web`
spelling remains supported. Web uses the same renderer selection as the native window, including global
overrides and launcher preferences. Explicit renderer choices are preserved. The observer inherits the configured window
size (padding odd dimensions for the codec) unless Web dimensions are supplied.
The renderer and streaming service are reserved
before World construction. NoRender cannot serve a native viewport.

`open_viewer()`, `viewer`, `is_viewer_open`, `poll_viewer()`, `close_viewer()` and
`wait_for_input()` / `sleep()` are the shared local/Web presentation surface. Viewports
support Python look-at/pose and title operations; the Web page dispatches no
keyboard callbacks. Explicit
`open_window()` and `World.get_windows()` retain their native-window meaning;
custom callers using them must migrate to the shared surface. Viser retains
its scene-snapshot runtime rather than a native viewport.

## Information and lifecycle

The observer has a dedicated camera group, even when its size matches a task
camera. Native task-camera videos show existing rendered RGB buffers. Opening
the observer does not render task sensors again or alter their configuration.
Task cameras may remain black until their host renders them. Environment count
and task-sensor construction are independent of Web mode.

Application creation, information requests, observer rendering and closure
remain on the World creator thread. Reading initialization rows never prepares
a scene or advances physics. Paused callers keep the information page and video
responsive through `poll_viewer()` / `wait_for_input()` / `sleep()`.
`visualization_runtime` remains the Viser snapshot runtime; native Web
presentation is available through `viewer` and `visualization_health`.

`close_viewer()` closes the application and observer without changing the task.
It remains closed until explicitly reopened. The World-owned observer group is
reused on reopen to avoid repeated native-buffer allocation. Destroying the
World and flushing deferred cleanup releases its listener/port; closing the
application alone cannot stop the current SDK's listener. Health reports
application readiness/errors; native peer count is unavailable (reported zero).

## DexSim boundaries

- Requires Linux/NVIDIA, loadable WebRTC streaming libraries, and NVENC. The
  pinned SDK is `dexsim_engine==0.6.0rc1`. Its Python API exposes
  `WebApplication` and startup `WorldConfig.webrtc_*` settings.
- The SDK automatically enumerates scene cameras and has no Python video-source
  whitelist. Its video catalog is limited to 16 RGB streams, with even
  dimensions in 32–4096. Stereo/depth-only/large camera catalogs are allowed in
  the simulation, but an incompatible native catalog can prevent WebRTC media
  startup. The browser displays the native service error. A selective-source
  SDK API is still needed to lift this media limitation.
- Only one browser viewer is supported by this native Python path. The listener
  binds all IPv4 interfaces and should be accessed through a trusted network or
  access-controlled deployment. Native room tokens are session controls, not
  user-account authentication; forwarding HTTP alone does not ensure ICE/media
  connectivity. `visualization.web.ice` configures STUN/TURN services.
- `eval-policy --web` supports EmbodiChain simulator tasks. External Motion
  Policy Kit profiles own their own World/window factory and expose no WebRTC
  selection; that combination fails explicitly instead of opening a local
  window. Low-level native SDK input-view factories remain native APIs.

## Validation

CPU coverage includes configuration, CLI forwarding, shared viewport lifecycle,
read-only command rejection and shared terminal/Web initialization data. Real
browser acceptance runs bounded child scenes with default/Newton physics and
multiple environments, checking decoded pixels, information tables, disabled
input, unchanged scene/sensor data, reconnect and cleanup:

```bash
python -m pytest tests/visualization/test_web_viewer_browser.py --run-gpu -q
```

Install Playwright and Chrome and retain the host's display/SDL environment.
`WEBRTC_TEST_CHROME` selects a Chrome executable outside PATH. SDK-native
library failures are runtime qualification failures, not successful browser
validation; see the PR for the exact environment and any test-only repairs.
