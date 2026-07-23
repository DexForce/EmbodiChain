# Installation

EmbodiChain is a Python framework built on the [DexSim](https://github.com/DexForce) simulation engine (`dexsim_engine` on PyPI). This guide covers system requirements, package indexes, Docker and local install paths, optional generative-simulation dependencies, and verification.

After installation, continue with the [Quick Start Tutorial](../tutorial/index.rst).

## Choose your setup

| Path | Best for | Notes |
|------|----------|-------|
| **Docker** | First run, reproducible GPU sim | Pre-built image with CUDA 12.8, Vulkan, and Python 3.11 |
| **Local + [uv](https://github.com/astral-sh/uv)** | Day-to-day development | Fast installs; recommended with a virtual environment |
| **Local + pip** | Simple environments | Use a virtual environment |

## System requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux x86_64 (Ubuntu 20.04+ recommended) |
| **GPU** | NVIDIA GPU with compute capability 7.0+ |
| **NVIDIA driver** | ≥ 535 (tested on driver branches up to 580.x) |
| **CUDA** | 12.x (aligned with the Docker image and `dexsim_engine` wheels) |
| **Vulkan** | Host ICD/layer files for GPU rendering (see Docker notes) |
| **Python** | 3.10 or 3.11 |
| **Display** (optional) | X11 `DISPLAY` for interactive viewer windows |

> [!NOTE]
> **PyTorch:** EmbodiChain depends on PyTorch transitively (for example via `dexsim_engine` and `pytorch_kinematics`). If you install or upgrade PyTorch separately, match the wheel to your CUDA version using the [official PyTorch install selector](https://pytorch.org/get-started/locally/).

## Package indexes

EmbodiChain and its simulation backend are published on a DexForce package index. Generative-simulation extras also need Blender's index for the `bpy` wheel.

| Index | URL | Used for |
|-------|-----|----------|
| **DexForce (required)** | `http://pyp.open3dv.site:2345/simple/` | `embodichain`, `dexsim_engine`, and related wheels |
| **Blender (gensim only)** | `https://download.blender.org/pypi/` | `bpy` |

Reuse these flags on every `pip` / `uv pip` install command:

```bash
DEXFORCE_INDEX="http://pyp.open3dv.site:2345/simple/"
DEXFORCE_TRUSTED_HOST="pyp.open3dv.site"
BLENDER_INDEX="https://download.blender.org/pypi/"

PIP_EXTRA_ARGS="--extra-index-url ${DEXFORCE_INDEX} --trusted-host ${DEXFORCE_TRUSTED_HOST}"
GENSIM_EXTRA_ARGS="${PIP_EXTRA_ARGS} --extra-index-url ${BLENDER_INDEX}"
```

> [!TIP]
> To avoid repeating flags, you can configure pip once:  
> `pip config set global.extra-index-url "${DEXFORCE_INDEX}"` and  
> `pip config set global.trusted-host "${DEXFORCE_TRUSTED_HOST}"`.

## Docker (recommended for first run)

The pre-configured image includes CUDA 12.8, Vulkan-related mounts, and dependencies needed for GPU simulation and rendering.

### Prerequisites

- [Docker](https://docs.docker.com/engine/install/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA driver ≥ 535 on the host
- For **GUI** runs: working X11 forwarding (`DISPLAY`, `~/.Xauthority`, `/tmp/.X11-unix`)
- For **headless** servers: no display required; use `--headless` in tutorial scripts

### Pull and start a container

**1. Pull the image:**

```bash
docker pull dexforce/embodichain:ubuntu22.04-cuda12.8
```

**2. Start a container** using the repo script `docker/docker_run.sh` (mounts GPU drivers, Vulkan, shared memory, and your data directory):

```bash
git clone https://github.com/DexForce/EmbodiChain.git
cd EmbodiChain
./docker/docker_run.sh <container_name> <data_path>
```

| Argument | Meaning |
|----------|---------|
| `container_name` | Name for the new container |
| `data_path` | Host directory mounted at `/root/workspace` inside the container |

The script checks for Vulkan ICD/layer and EGL vendor JSON files on the host. Warnings usually mean reduced rendering support; the script exits only when required driver paths are missing entirely.

**3. Attach to the running container:**

```bash
docker exec -it <container_name> bash
```

Inside the container, install or update EmbodiChain with the [local installation](#local-installation) commands if needed, then [verify](#verify-installation).

> [!NOTE]
> The script uses `--network=host`, `--gpus all`, and a large `--shm-size` for simulation workloads. Adjust mounts in `docker/docker_run.sh` if your driver files live under `/etc` instead of `/usr/share`.

## Local installation

Use a dedicated virtual environment to avoid conflicts with system Python packages.

### 1. Create a virtual environment

**With uv (recommended):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 .venv
source .venv/bin/activate
```

**With pip:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install EmbodiChain

Set the index variables from [Package indexes](#package-indexes), then pick one row:

| Source | Tool | Command |
|--------|------|---------|
| PyPI | uv | `uv pip install embodichain ${PIP_EXTRA_ARGS}` |
| PyPI | pip | `pip install embodichain ${PIP_EXTRA_ARGS}` |
| Git clone | uv | `uv pip install -e . ${PIP_EXTRA_ARGS}` |
| Git clone | pip | `pip install -e . ${PIP_EXTRA_ARGS}` |

**Example — editable install from source with uv:**

```bash
git clone https://github.com/DexForce/EmbodiChain.git
cd EmbodiChain
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e . \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ \
  --trusted-host pyp.open3dv.site
```

**Example — install from PyPI with pip:**

```bash
pip install embodichain \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ \
  --trusted-host pyp.open3dv.site
```

This pulls in `dexsim_engine` (Python package `dexsim`) and the rest of the core dependencies declared in `pyproject.toml`.

## Optional: cuRobo V2 motion planning

Install a cuRobo extra to use EmbodiChain's CUDA-accelerated, collision-aware
motion planner. cuRobo is intentionally not part of the core dependency set:
select exactly one extra that matches the CUDA version reported by
`nvidia-smi`.

The normal EmbodiChain environment already provides PyTorch, so prefer one of
the non-`torch` extras:

| CUDA | Published package | Source checkout |
|------|-------------------|-----------------|
| 12.x | `uv pip install "embodichain[curobo-cu12]" ${PIP_EXTRA_ARGS}` | `uv pip install -e ".[curobo-cu12]" ${PIP_EXTRA_ARGS}` |
| 13.x | `uv pip install "embodichain[curobo-cu13]" ${PIP_EXTRA_ARGS}` | `uv pip install -e ".[curobo-cu13]" ${PIP_EXTRA_ARGS}` |

For a fresh environment that also needs cuRobo to select and install PyTorch,
use `curobo-cu12-torch` or `curobo-cu13-torch` instead. The same extras work
with `pip`; replace `uv pip install` with `pip install`.

**Recommended for the current CUDA 12.x EmbodiChain stack:**

```bash
uv pip install -e ".[curobo-cu12]" \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ \
  --trusted-host pyp.open3dv.site

python -c "import curobo; print(curobo.__version__)"
pytest --pyargs curobo.tests
```

The dependency is installed from NVIDIA's source repository and pinned to the
cuRobo V2 `v0.8.0` release. cuRobo has stricter requirements than the core
EmbodiChain installation: Linux, Python 3.10--3.13, a supported NVIDIA GPU with
at least 4 GB VRAM, and a driver that supports CUDA 12 or newer. See
[NVIDIA's official installation guide](https://nvlabs.github.io/curobo/latest/getting-started/installation.html)
for the current compatibility requirements, and see
[cuRobo V2 Planner](../overview/sim/planners/curobo_planner.md) for EmbodiChain
configuration and usage.

## Optional: generative simulation (`gensim`)

Install the `gensim` extra for SimReady asset pipelines, Blender-based mesh processing, and `pyrender`. The `bpy` wheel is hosted on Blender's index and must be included in the install command.

| Source | Tool | Command |
|--------|------|---------|
| PyPI | uv | `uv pip install "embodichain[gensim]" ${GENSIM_EXTRA_ARGS}` |
| PyPI | pip | `pip install "embodichain[gensim]" ${GENSIM_EXTRA_ARGS}` |
| Git clone | uv | `uv pip install -e ".[gensim]" ${GENSIM_EXTRA_ARGS}` |
| Git clone | pip | `pip install -e ".[gensim]" ${GENSIM_EXTRA_ARGS}` |

**Example:**

```bash
pip install -e ".[gensim]" \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ \
  --trusted-host pyp.open3dv.site \
  --extra-index-url https://download.blender.org/pypi/
```

> [!TIP]
> When using **uv** from a source checkout, `pyproject.toml` already defines the Blender index under `[tool.uv.index]` for the `bpy` source. You still need the DexForce index flags for `dexsim_engine`.

For SimReady pipeline usage and LLM configuration, see [SimReady Asset Pipeline](../features/generative_sim/simready_pipeline.md).

## Verify installation

### Quick check (all install methods)

```bash
python -c "import embodichain, dexsim; print('embodichain', embodichain.__version__); print('dexsim', dexsim.__version__)"
```

You should see version strings for both packages with no import errors.

### Simulation tutorial (source tree or Docker with repo)

The tutorial script `scripts/tutorials/sim/create_scene.py` ships with the repository. Run it from the **repository root**:

```bash
cd /path/to/EmbodiChain
python scripts/tutorials/sim/create_scene.py
```

- **With a display:** omit `--headless` to open the DexSim viewer after the scene is built.
- **Headless / SSH:** use `--headless` to run without a window (FPS logs in the terminal):

```bash
python scripts/tutorials/sim/create_scene.py --headless
```

Optional GPU smoke test:

```bash
python scripts/tutorials/sim/create_scene.py --headless --device cuda
```

Press `Ctrl+C` to stop; the script cleans up the simulation on exit.

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| `Could not find a version` / `No matching distribution` for `embodichain` or `dexsim_engine` | Add the DexForce index and `--trusted-host pyp.open3dv.site` (see [Package indexes](#package-indexes)). |
| `No module named 'dexsim'` after install | Reinstall with the DexForce index; `dexsim` is provided by the `dexsim_engine` package. |
| Docker Vulkan / EGL warnings from `docker_run.sh` | Install host NVIDIA drivers and Vulkan user-space packages; paths must be files under `/etc` or `/usr/share`, not directories. |
| Viewer does not open | Export `DISPLAY`, allow X11 access (`xhost +local:` on the host), and ensure `~/.Xauthority` is mounted (the run script does this by default). |
| PyTorch / CUDA errors at runtime | Reinstall a PyTorch build that matches your driver/CUDA from [pytorch.org](https://pytorch.org/get-started/locally/). |
| `No module named 'curobo'` | Install exactly one CUDA-matched cuRobo extra, such as `uv pip install -e ".[curobo-cu12]"`, from the EmbodiChain repository root. |
| `bpy` install fails | Include the Blender index (`https://download.blender.org/pypi/`) and use Python 3.10 or 3.11. |

## Next steps

- [Quick Start Tutorial](../tutorial/index.rst)
- [Simulation Manager](../overview/sim/sim_manager.md)
- [Build documentation](docs.md)
