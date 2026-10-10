# Rendering Configuration

The {class}`~embodichain.lab.sim.cfg.RenderCfg` class controls the renderer,
ray-tracing sample count, scoped denoising/reconstruction, tone mapping, DLSS,
path-depth limits, and NRD settings used by
{class}`~embodichain.lab.sim.sim_manager.SimulationManager`.

## Core options

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `renderer` | `str` | `"auto"` | Renderer backend: `auto`, `no-render`, `hybrid`, `fast-rt`, or `rt`. |
| `spp` | `int` | `1` | Samples per pixel for ray-traced rendering. Must be at least `1`. |
| `min_bounces` | `int` | `4` | Path depth at which Russian Roulette may start terminating rays. Must be non-negative. |
| `max_bounces` | `int` | `8` | Maximum traced path depth, counting the primary segment. Must be positive and at least `min_bounces`. |
| `denoising` | `DenoisingCfg` | `DenoisingCfg()` | Independent `window`/`offscreen` choices: `off`, `optix`, `dlss`, or `nrd`. |
| `tone_mapping_enabled` | `bool` | `False` | Apply modified Reinhard tone mapping to RGB output. |
| `tone_mapping_exposure` | `float` | `1.0` | Fixed linear exposure multiplier used before tone mapping. |
| `dlss` | `DLSSCfg` | `DLSSCfg()` | NVIDIA DLSS settings. See {doc}`dlss`. |
| `nrd` | `NRDCfg` | `NRDCfg()` | NVIDIA NRD settings for the public `nrd` path. |

The public `dlss` path maps to DexSim `DLSS_RR`; the public `nrd` path maps to
standalone DexSim `NRD_RELAX`. Tone mapping affects RGB output only; depth,
segmentation masks, normals, and position buffers remain unchanged.

Bounce limits are validated at construction and again before conversion to
DexSim's `WorldConfig`, including after mutable config edits. A ray can end on
a miss or absorption before `min_bounces`; that setting controls the start of
random path termination rather than guaranteeing a path length. The primary
segment has depth zero. The defaults preserve DexSim's existing path budget.

NRD's indirect-lighting budget remains controlled by
{attr}`~embodichain.lab.sim.cfg.NRDCfg.max_indirect_bounces`; `max_bounces`
also bounds NRD transparent continuations. Matching the two general bounce
limits across denoising modes does not make their light-transport budgets
identical.

## Renderer selection

For state-based RL, use `RenderCfg(renderer="no-render")` with
`SimulationManagerCfg(headless=True)`. This selects DexSim's `Renderer.NORENDER`
backend. EmbodiChain skips background, light and visual-material creation,
retains physical ground, and skips Newton render-state publication.
Native cameras and windows require `hybrid`, `fast-rt`, or `rt`.
`headless=True` with a native renderer keeps offscreen rendering available.

Policy evaluation accepts `--renderer no-render`. For visual evaluation of a
NoRender checkpoint, `--viewer` selects Hybrid by default.

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
        min_bounces=1,
        max_bounces=4,
        denoising=DenoisingCfg(window="dlss", offscreen="nrd"),
        tone_mapping_enabled=True,
        tone_mapping_exposure=1.0,
    )
)
```

For DLSS-specific settings, see {doc}`dlss`.
