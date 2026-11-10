# NVIDIA DLSS

EmbodiChain exposes the core NVIDIA DLSS controls through
{class}`~embodichain.lab.sim.cfg.DLSSCfg` and passes them to DexSim when the
{class}`~embodichain.lab.sim.sim_manager.SimulationManager` creates its world.
The integration applies to the `hybrid`, `fast-rt`, and `rt` renderers. DexSim
owns DLSS feature detection, initialization, temporal history, and fallback;
constructing a configuration object does not by itself initialize DLSS.

## Processing model

DexSim manages Ray Reconstruction (RR) for the public `dlss` path. The public
`nrd` path is configured separately through `RenderCfg.nrd` and maps to
standalone NRD RELAX.

## EmbodiChain configuration

The following fields are available under `RenderCfg.dlss`:

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `dlss_quality` | `2` | Quality preset: `-1` auto, `0` ultra performance, `1` performance, `2` balanced, `3` quality, `4` ultra quality, or `5` DLAA. |
| `render_width`, `render_height` | `0` | Optional internal dimensions for FastRT/OfflineRT windows. Zero derives the dimensions from the quality preset. |
| `target_width`, `target_height` | `0` | Compatibility fields. Set the actual output size on the window or camera instead. |
| `upsample_ratio` | `None` | Optional FastRT/OfflineRT ratio used to derive unset internal dimensions. |
| `exposure_compensation` | `1.0` | Positive exposure multiplier used by the RR bridge. |
`DLSSCfg.tiled_enabled` controls whether compatible multi-camera targets use a
single tiled atlas. Atlas gutter and dimension limits remain owned by DexSim
and use its native defaults.

## Quality and resolution

The quality preset determines the internal render resolution relative to the
requested output:

| Value | Mode | Approximate internal size |
| :---: | :--- | :---: |
| `-1` | Auto | Balanced-safe scale followed by NGX mode selection |
| `0` | Ultra Performance | 33% of output |
| `1` | Performance | 50% of output |
| `2` | Balanced | 58% of output |
| `3` | Quality | 67% of output |
| `4` | Ultra Quality | 77% of output |
| `5` | DLAA | 100% of output |

Set the window output size with `SimulationManagerCfg.width` and
`SimulationManagerCfg.height`, or set the output resolution in the camera
configuration. FastRT/OfflineRT windows may additionally use explicit
`render_width`/`render_height` values or `upsample_ratio`. Hybrid and offscreen
targets derive their internal resolution from the target output and quality
preset.

## Examples

Configure DLSS quality in a headless simulation:

```python
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import DenoisingCfg, DLSSCfg, RenderCfg

sim_config = SimulationManagerCfg(
    headless=True,
    render_cfg=RenderCfg(
        renderer="hybrid",
        denoising=DenoisingCfg(window="dlss", offscreen="dlss"),
        dlss=DLSSCfg(
            dlss_quality=3,
        ),
    ),
)
```

Configure an OfflineRT window with an explicit internal resolution:

```python
sim_config = SimulationManagerCfg(
    width=1920,
    height=1080,
    render_cfg=RenderCfg(
        renderer="rt",
        dlss=DLSSCfg(render_width=1280, render_height=720),
    ),
)
```

The same settings are available in task JSON/YAML under `render_cfg.denoising`
and `render_cfg.dlss` (decoded into the corresponding nested config objects).
Use `denoising.window: off` or `denoising.offscreen: off` to select the raw
ray-traced path for a scope.

## Availability and fallback

DLSS requires a Vulkan render device, a compatible NVIDIA GPU and driver, and
a DexSim build that includes the NGX runtime libraries. DexSim checks support
during renderer startup. If DLSS is unavailable, it falls back to the OptiX
denoiser and reports initialization or fallback in the engine log.

For offscreen rendering, each enabled camera group may allocate temporal
history and Vulkan exchange resources. Increasing the number or resolution of
camera groups therefore increases GPU memory usage. Validate DLSS by rendering
an eligible frame and checking the engine log; configuration conversion or
world construction alone is not sufficient evidence that DLSS initialized.
