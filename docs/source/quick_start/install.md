# Installation

EmbodiChain is a Python framework built on the [DexSim](https://github.com/DexForce) simulation engine (`dexsim_engine` on PyPI). This guide covers system requirements, package indexes, Docker and local install paths, optional policy deployment and generative-simulation dependencies, and verification.

After installation, continue with the [Quick Start Tutorial](../tutorial/index.rst).

## Choose your setup

| Path | Best for | Notes |
|------|----------|-------|
| **Docker** | First run, reproducible GPU sim | Pre-built Python 3.12 runtime with CUDA 12.8 and an isolated Blender worker |
| **Local + [uv](https://github.com/astral-sh/uv)** | Day-to-day development | Fast installs; recommended with a virtual environment |
| **Local + pip** | Simple environments | Use a virtual environment |

## System requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux x86_64 (Ubuntu 20.04+ recommended) |
| **GPU** | NVIDIA GPU with compute capability 7.0+ |
| **NVIDIA driver** | ≥ 535 (tested on driver branches up to 595.x) |
| **CUDA** | 12.x (aligned with the Docker image and `dexsim_engine` wheels) |
| **Vulkan** | Host ICD/layer files for GPU rendering (see Docker notes) |
| **Python** | Core / GenSim: 3.11 or 3.12; isolated Blender worker: 3.11 |
| **Display** (optional) | X11 `DISPLAY` for interactive viewer windows |

> [!IMPORTANT]
> GenSim runs in the main EmbodiChain environment, while Blender operations can
> run in an isolated Python 3.11 worker selected by `EMBODICHAIN_BLENDER_PYTHON`.
> The public Docker image configures this worker automatically. For local
> installs, see [generative simulation](#optional-generative-simulation-gensim).

NVIDIA drivers are backward compatible with applications built against older
CUDA toolkits. A 595-series host driver therefore works with the current CUDA
12.8 Docker image and wheels; installing a CUDA 13 toolkit on the host is not
required. See NVIDIA's [CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/latest/index.html)
for details.

## Python and LeRobot compatibility

Installation selects the LeRobot SDK by Python version:

| Python | LeRobot | Hugging Face Hub | Torch / torchvision / TorchCodec | TensorDict |
| --- | --- | --- | --- | --- |
| 3.11 | 0.4.4 | >=0.35.3, <0.36 | 2.7.x / 0.22.x / 0.5.x | >=0.13, <0.15 |
| 3.12 | 0.6.1, including training/dataset dependencies | >=1.6, <2 | 2.11.x / 0.26.x / >=0.11.1, <0.12 | >=0.15.3, <0.16 |

Torch remains a direct dependency because the framework uses it independently
of LeRobot. TorchCodec must match the Torch binary interface; LeRobot's wider
dependency ranges alone do not enforce that pairing. CI pins exact patch
versions within each supported profile; package metadata allows patch updates
within those version lines. Further qualification is tracked in
[#883](https://github.com/DexForce/EmbodiChain/issues/883).
Select matching CPU or CUDA wheels when installing the Torch stack. For the
Python 3.12 GPU profile, select the CUDA 12.8 wheels explicitly when using the
existing CUDA 12.8 runtime; PyPI's default Torch wheel may select CUDA 13:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 torchcodec==0.11.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

The SDK matrix CI checks a clean CPU installation, recording/loading, video
decoding, and an official ACT update/checkpoint/reset. GPU simulation and full policy learning acceptance
are separate checks. This release requires Python >=3.11 and pins DexSim
0.6.0rc1. The DexForce index provides Linux x86_64 wheels for CPython 3.11 and
3.12; use one of those versions for this release. Python 3.10 is unsupported,
and versions above 3.12 are outside the current wheel/validation matrix.

Recordings retain the existing task/subtask table and depth-sidecar schema on
both routes. LeRobot 0.6.1 exposes `subtask_index` but does not automatically
resolve `meta/subtasks.parquet`; the EmbodiChain preview resolves it explicitly.
Use the [standalone exporter](https://github.com/DexForce/EmbodiChain/blob/main/tools/lerobot_export/README.md) for native 0.6.1
persistent-language annotations and depth features. The [ACT example](https://github.com/DexForce/EmbodiChain/blob/main/examples/learning/act/README.md) uses the official LeRobot
0.6.1 trainer and policy on Python 3.12. Historical custom ACT checkpoints require
retraining.

## Package indexes

EmbodiChain and its simulation backend are published on a DexForce package index. The isolated Blender worker uses Blender's index for the `bpy` wheel.

| Index | URL | Used for |
|-------|-----|----------|
| **DexForce (required)** | `http://pyp.open3dv.site:2345/simple/` | `embodichain`, `dexsim_engine`, and related wheels |
| **Blender (worker only)** | `https://download.blender.org/pypi/` | `bpy` |

Use these flags when installing EmbodiChain or its extras:

```bash
DEXFORCE_INDEX="http://pyp.open3dv.site:2345/simple/"
DEXFORCE_TRUSTED_HOST="pyp.open3dv.site"

PIP_EXTRA_ARGS="--extra-index-url ${DEXFORCE_INDEX} --trusted-host ${DEXFORCE_TRUSTED_HOST}"
```

> [!TIP]
> To avoid repeating flags, you can configure pip once:  
> `pip config set global.extra-index-url "${DEXFORCE_INDEX}"` and  
> `pip config set global.trusted-host "${DEXFORCE_TRUSTED_HOST}"`.

## Docker (recommended for first run)

The public image provides a Python 3.12 runtime with GPU simulation, GenSim,
cuRobo, and ONNX CUDA deployment. Blender runs in an isolated Python 3.11
worker. The `ubuntu22.04-cuda12.8` and `ubuntu22.04-cuda12.8-py312` tags refer
to the same public image.

The internal image additionally provides a Python 3.11 business environment;
select `py311` or `py312` with `EMBODICHAIN_ENV`. See the
[container guide](https://github.com/DexForce/EmbodiChain/blob/main/docker/README.md)
for internal registry access, environment selection, source development, and
dependency locks.

The published image digests in the container guide's validation record belong
to the earlier DexSim 0.5.1rc1 runtime. For this checkout's DexSim 0.6.0rc1
requirements, build a candidate from the current source or follow the guide's
source-development instructions to update the selected environment.

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

The script mounts available Vulkan ICD/layer and EGL vendor JSON files from
`/etc` and `/usr/share`. Headless startup does not require X11 files.

**3. Attach to the running container:**

```bash
docker exec -it <container_name> bash
```

EmbodiChain is already installed in the container; continue with
[verification](#verify-installation). For editable source development, follow the
[container guide](https://github.com/DexForce/EmbodiChain/blob/main/docker/README.md#install-and-run)
to preserve the selected environment's dependency locks.

> [!NOTE]
> The script uses `--network=host`, `--gpus all`, and a large `--shm-size` for simulation workloads. Set `EMBODICHAIN_DOCKER_IMAGE` to select a different image.

## Local installation

Use a dedicated virtual environment to avoid conflicts with system Python packages.
For the core package, choose Python 3.11 or 3.12. The examples below use
Python 3.12. Check the [compatibility matrix](#python-and-lerobot-compatibility)
before choosing a version for an existing dataset or learning workflow.

### 1. Create a virtual environment

**With uv (recommended):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
source .venv/bin/activate
```

**With pip:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install EmbodiChain

Set the index variables from [Package indexes](#package-indexes), then pick one row.
For a Python 3.12 / CUDA 12.8 environment, install the matching Torch stack
from the [compatibility section](#python-and-lerobot-compatibility) first.
With uv, add `--torch-backend cu128` to the installation command to select that
CUDA wheel source explicitly.

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
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e . --torch-backend cu128 \
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

### 3. Official task environments

The main `embodichain` distribution includes the official task environments as
the `embodichain_tasks` import package, together with their JSON/YAML configs.
Both a published-wheel install and `pip install -e .` register the
`embodichain.tasks` entry point automatically; do not install
`embodichain_tasks/` separately.

If an older checkout was installed with `pip install -e embodichain_tasks/`,
remove that legacy editable distribution once before reinstalling the main
project:

```bash
pip uninstall -y embodichain_tasks
pip install -e .
```

Commands can continue to use repository-style paths such as
`embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic.yaml`.
EmbodiChain resolves these paths from the checkout when present and otherwise
from the installed wheel.

## Optional: policy deployment (`policy-deploy`)

Install the `policy-deploy` extra to run ONNX policies with NeuralPlanner/NMG
or DexSim Motion Policy Kit. It installs `onnxruntime-gpu>=1.20,<1.27` for the
CUDA 12.x stack; ORT 1.27 and later default to CUDA 13. The base EmbodiChain
installation does not request ONNX Runtime.

Use the index variables from [Package indexes](#package-indexes):

| Source | Tool | Command |
|--------|------|---------|
| PyPI | uv | `uv pip install "embodichain[policy-deploy]" ${PIP_EXTRA_ARGS}` |
| PyPI | pip | `pip install "embodichain[policy-deploy]" ${PIP_EXTRA_ARGS}` |
| Git clone | uv | `uv pip install -e ".[policy-deploy]" ${PIP_EXTRA_ARGS}` |
| Git clone | pip | `pip install -e ".[policy-deploy]" ${PIP_EXTRA_ARGS}` |

The GPU package also provides CPU execution. Install only one of
`onnxruntime` and `onnxruntime-gpu` in an environment; see the
[ONNX Runtime installation guide](https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime).
For GPU execution, match the installed ORT version to the environment's CUDA
and cuDNN libraries using the
[CUDA provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).
Installing the extra does not override a policy's execution-provider selection.

The former `nmg` extra has been replaced by `policy-deploy`. Update existing
install commands accordingly. If the old extra installed the CPU package,
remove it before installing the shared GPU runtime. If both ORT packages are
already installed, remove both first to avoid overlapping package files:

```bash
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install "embodichain[policy-deploy]" ${PIP_EXTRA_ARGS}
```

For an editable checkout, use `-e ".[policy-deploy]"` in the second command.

## Newton planning geometry

Native convex planning queries reuse the isolated Newton collision pipeline.
They require the supported Newton runtime and do not require `python-fcl` or
an additional FCL extra. Queries run on CPU independently of the live physics
backend; missing Newton entry points fail before native scene capture.

Distance diagnostics are capped contact-band estimates, not global exact
minimum distances. Precision intervals spanning a requested contact threshold
remain unavailable. CPU query tests do not establish GPU or physical-task
qualification.

## Optional: cuRobo V2 motion planning

Install cuRobo separately to use EmbodiChain's CUDA-accelerated,
collision-aware motion planner. cuRobo is intentionally not part of the core
dependency set, and its Git source requirement cannot be included in metadata
published to PyPI. Select exactly one command that matches the CUDA version
used by PyTorch in the active environment:

```bash
python -c "import torch; print(torch.version.cuda)"
```

`nvidia-smi` reports the newest CUDA version supported by the installed driver,
which can be newer than the CUDA runtime used by PyTorch and is therefore not
the value to use when choosing the cuRobo extra.

The normal EmbodiChain environment already provides PyTorch, so prefer one of
the non-`torch` variants:

```bash
# CUDA 12.x
uv pip install \
  "nvidia-curobo[cu12] @ git+https://github.com/NVlabs/curobo.git@v0.8.0"

# CUDA 13.x
uv pip install \
  "nvidia-curobo[cu13] @ git+https://github.com/NVlabs/curobo.git@v0.8.0"
```

For a fresh environment that also needs cuRobo to select and install PyTorch,
replace `cu12` or `cu13` with `cu12-torch` or `cu13-torch`. The same source
requirements work with `pip`; replace `uv pip install` with `pip install`.

**Recommended for the current CUDA 12.x EmbodiChain stack:**

```bash
uv pip install \
  "nvidia-curobo[cu12] @ git+https://github.com/NVlabs/curobo.git@v0.8.0"

python -c "import curobo; print(curobo.__version__)"
pytest --pyargs curobo.tests
```

The dependency is installed from NVIDIA's source repository and pinned to the
cuRobo V2 `v0.8.0` release. cuRobo has additional platform requirements:
Linux, upstream Python 3.10--3.13, a supported NVIDIA GPU with
at least 4 GB VRAM, and a driver that supports CUDA 12 or newer. See
[NVIDIA's official installation guide](https://nvlabs.github.io/curobo/latest/getting-started/installation.html)
for the current compatibility requirements, and see
[cuRobo V2 Planner](../overview/sim/motion/planners/curobo_planner.md) for EmbodiChain
configuration and usage. cuRobo planning always runs on CUDA, but the
SimulationManager physics device may be either CPU or CUDA.
cuRobo's upstream Python range does not expand this release's EmbodiChain
runtime matrix; use Python 3.11 or 3.12 when installing them together.

## Optional: generative simulation (`gensim`)

Install the `gensim` extra in your main EmbodiChain environment for the
Python-facing GenSim dependencies. Blender is supplied by a separate runtime;
the extra does not install `bpy`.

Use the index variables from [Package indexes](#package-indexes):

| Source | Tool | Command |
|--------|------|---------|
| Published package | uv | `uv pip install "embodichain[gensim]" ${PIP_EXTRA_ARGS}` |
| Published package | pip | `pip install "embodichain[gensim]" ${PIP_EXTRA_ARGS}` |
| Git clone | uv | `uv pip install -e ".[gensim]" ${PIP_EXTRA_ARGS}` |
| Git clone | pip | `pip install -e ".[gensim]" ${PIP_EXTRA_ARGS}` |

The Docker images already configure the isolated Blender worker. For a local
installation, run the following from a source checkout with `uv` installed:

```bash
uv venv --python 3.11 .venv-blender
EMBODICHAIN_CONSTRAINTS_DIR="$PWD/docker/locks" \
  bash docker/install-profile.sh blender311 "$PWD/.venv-blender"
export EMBODICHAIN_BLENDER_PYTHON="$PWD/.venv-blender/bin/python"
```

Keep your main EmbodiChain environment active when launching GenSim. Blender
mesh and rendering jobs are handed off to the selected worker automatically.
The installer applies the locked Blender dependencies and the wheel metadata
correction described in the
[Blender runtime notes](https://github.com/DexForce/EmbodiChain/blob/main/docker/README.md#blender-and-native-library-isolation).

For Asset Engine usage and LLM configuration, see [Asset Engine](../features/generative_sim/asset_engine.md).

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

For a headless browser view, enable Viser:

```bash
python scripts/tutorials/sim/create_scene.py --viser
```

Open the endpoint printed in the terminal, normally
`http://127.0.0.1:8080`. On a remote machine, keep the default loopback binding
and forward the port:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@worker-host
```

See [Browser visualization with Viser](../overview/sim/visualization/viser.md)
for supported objects, camera preview, update-rate options, and remote-access
guidance.

Press `Ctrl+C` to stop; the script cleans up the simulation on exit.

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| `Could not find a version` / `No matching distribution` for `embodichain` or `dexsim_engine` | Add the DexForce index and `--trusted-host pyp.open3dv.site` (see [Package indexes](#package-indexes)). |
| `No module named 'dexsim'` after install | Reinstall with the DexForce index; `dexsim` is provided by the `dexsim_engine` package. |
| Docker Vulkan / EGL rendering errors | Check NVIDIA Container Toolkit and host driver installation; the run script mounts available JSON files under `/etc` and `/usr/share`. |
| Viewer does not open | Export `DISPLAY`, allow X11 access (`xhost +local:` on the host), and ensure `~/.Xauthority` is mounted (the run script does this by default). |
| PyTorch / CUDA errors at runtime | Reinstall a PyTorch build that matches your driver/CUDA from [pytorch.org](https://pytorch.org/get-started/locally/). |
| `No module named 'curobo'` | Install the CUDA-matched cuRobo source requirement separately, such as `uv pip install "nvidia-curobo[cu12] @ git+https://github.com/NVlabs/curobo.git@v0.8.0"`. |
| Blender worker fails or `bpy` is missing | Install the isolated Python 3.11 worker and set `EMBODICHAIN_BLENDER_PYTHON` to its interpreter (see [GenSim setup](#optional-generative-simulation-gensim)). |

## Next steps

- [Quick Start Tutorial](../tutorial/index.rst)
- [Simulation Manager](../overview/sim/sim_manager.md)
- [Browser visualization with Viser](../overview/sim/visualization/viser.md)
- [Build documentation](docs.md)
