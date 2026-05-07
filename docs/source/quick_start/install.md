# Installation

## System Requirements

| Component | Requirement |
|-----------|------------|
| **OS** | Linux (x86_64): Ubuntu 20.04+ |
| **GPU** | NVIDIA with compute capability 7.0+ |
| **NVIDIA Driver** | 535 - 570 (580+ is untested and may be unstable) |
| **Python** | 3.10 or 3.11 |

> [!NOTE]
> Ensure your NVIDIA driver is compatible with your chosen PyTorch wheel. We recommend installing PyTorch from the [official PyTorch instructions](https://pytorch.org/get-started/locally/) for your CUDA version.

## Installation

### Docker (Recommended)

We strongly recommend using our pre-configured Docker environment, which contains all necessary dependencies including CUDA, Vulkan, and GPU rendering support.

**1. Pull the image:**

```bash
docker pull dexforce/embodichain:ubuntu22.04-cuda12.8
```

**2. Start a container:**

Use the provided run script ([`docker/docker_run.sh`](../../../docker/docker_run.sh)), which handles GPU driver and Vulkan mounting:

```bash
./docker/docker_run.sh <container_name> <data_path>
```

### pip (PyPI)

> [!TIP]
> We strongly recommend using a virtual environment to avoid dependency conflicts.

```bash
pip install embodichain --extra-index-url http://pyp.open3dv.site:2345/simple/ --trusted-host pyp.open3dv.site
```

### From Source

> [!TIP]
> We strongly recommend using a virtual environment to avoid dependency conflicts.

```bash
git clone https://github.com/DexForce/EmbodiChain.git
cd EmbodiChain
pip install -e . --extra-index-url http://pyp.open3dv.site:2345/simple/ --trusted-host pyp.open3dv.site
```

## Verify Installation

Run the demo script to confirm everything is set up correctly:

```bash
python scripts/tutorials/sim/create_scene.py
```

If the installation is successful, you will see a simulation window with a rendered scene. To run without a display:

```bash
python scripts/tutorials/sim/create_scene.py --headless
```
