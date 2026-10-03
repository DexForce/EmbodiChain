# Rendering Configuration

The {class}`~embodichain.lab.sim.cfg.RenderCfg` class controls the renderer,
ray-tracing sample count, scoped denoising/reconstruction, tone mapping, DLSS,
and NRD settings used by
{class}`~embodichain.lab.sim.sim_manager.SimulationManager`.

## Core options

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `renderer` | `str` | `"auto"` | Renderer backend: `auto`, `hybrid`, `fast-rt`, or `rt`. |
| `spp` | `int` | `1` | Samples per pixel for ray-traced rendering. Must be at least `1`. |
| `denoising` | `DenoisingCfg` | `DenoisingCfg()` | Independent `window`/`offscreen` choices: `off`, `optix`, `dlss`, or `nrd`. |
| `tone_mapping_enabled` | `bool` | `False` | Apply modified Reinhard tone mapping to RGB output. |
| `tone_mapping_exposure` | `float` | `1.0` | Fixed linear exposure multiplier used before tone mapping. |
| `dlss` | `DLSSCfg` | `DLSSCfg()` | NVIDIA DLSS settings. See {doc}`dlss`. |
| `nrd` | `NRDCfg` | `NRDCfg()` | NVIDIA NRD settings for the public `nrd` path. |

The public `dlss` path maps to DexSim `DLSS_RR`; the public `nrd` path maps to
standalone DexSim `NRD_RELAX`. Tone mapping affects RGB output only; depth,
segmentation masks, normals, and position buffers remain unchanged.

## Renderer selection

With `renderer="auto"`, EmbodiChain selects a backend from the GPU detected at
the configured `gpu_id` when the simulation manager is constructed:

| GPU class | Examples | Selected renderer |
| :--- | :--- | :--- |
| RTX-series consumer/workstation GPUs | RTX 4090, RTX 6000 Ada | `hybrid` |
| Datacenter accelerators | A100, A800, H100, H800, H200, H20 | `fast-rt` |
| No CUDA device or unknown GPU | — | `hybrid` |

An explicit renderer always takes precedence over automatic selection. The
process-wide default can also be changed through
`SimulationManager.set_default_renderer()`:

```python
from embodichain.lab.sim import SimulationManager

SimulationManager.set_default_renderer("auto")
SimulationManager.set_default_renderer("fast-rt")
```

## Configuration example

```python
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import DenoisingCfg, RenderCfg

sim_config = SimulationManagerCfg(
    render_cfg=RenderCfg(
        renderer="fast-rt",
        spp=4,
        denoising=DenoisingCfg(window="dlss", offscreen="nrd"),
        tone_mapping_enabled=True,
        tone_mapping_exposure=1.0,
    )
)
```

For DLSS-specific settings, see {doc}`dlss`.

## Native window capture

When a native window opens, `SimulationManager` registers the **C** hotkey by
default. It saves one PNG under `outputs/images` using the current viewer
pose. Configure the control next to the recording and camera-pose settings:

```python
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import WindowCaptureCfg

sim_config = SimulationManagerCfg(
    window_capture=WindowCaptureCfg(
        enable_hotkey=True,
        hotkey="c",
        save_path="outputs/images/debug_view.png",
    )
)
```

The programmatic `sim.capture_window(save_path=...)` method returns an owned
`uint8` image array. DexSim's native frame is used when available; Hybrid and
Fast-RT builds without native CPU readback use an offscreen camera at the
current window pose. Closed windows and empty readbacks return `None` and do
not create a file.
