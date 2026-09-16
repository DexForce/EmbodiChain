# Rendering Configuration

The {class}`~embodichain.lab.sim.cfg.RenderCfg` class controls the renderer,
ray-tracing sample count, tone mapping, and DLSS settings used by
{class}`~embodichain.lab.sim.sim_manager.SimulationManager`.

## Core options

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `renderer` | `str` | `"auto"` | Renderer backend: `auto`, `hybrid`, `fast-rt`, or `rt`. |
| `spp` | `int` | `1` | Samples per pixel for ray-traced rendering. Must be at least `1`. |
| `tone_mapping_enabled` | `bool` | `False` | Apply modified Reinhard tone mapping to RGB output. |
| `tone_mapping_exposure` | `float` | `1.0` | Fixed linear exposure multiplier used before tone mapping. |
| `dlss` | `DLSSCfg` | `DLSSCfg()` | NVIDIA DLSS settings. See {doc}`dlss`. |

Ray-traced output uses DexSim's OptiX denoiser. Tone mapping affects RGB
output only; depth, segmentation masks, normals, and position buffers remain
unchanged.

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
from embodichain.lab.sim.cfg import RenderCfg

sim_config = SimulationManagerCfg(
    render_cfg=RenderCfg(
        renderer="fast-rt",
        spp=4,
        tone_mapping_enabled=True,
        tone_mapping_exposure=1.0,
    )
)
```

For DLSS-specific settings, see {doc}`dlss`.
