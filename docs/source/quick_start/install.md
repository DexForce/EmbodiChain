# Installation

## System Requirements

The following minimum system requirements are recommended to run EmbodiChain reliably. These are the tested configurations during development — other Linux distributions and versions may work but are not officially supported.

- Operating System: Linux (x86_64)
    - Recommended distributions: Ubuntu 20.04 LTS or Ubuntu 22.04 LTS

- NVIDIA GPU and drivers:
    - Hardware: NVIDIA GPU with compute capability 7.0 or higher (e.g., RTX 20 series, RTX 30 series, A100, etc.)
    - NVIDIA driver: 535 or higher (recommended 570)
    - CUDA Toolkit: any of 11.8 — 12.8 (we test primarily with 11.8 and 12.x)

- Python:
    - Supported Python versions:
        - Python 3.9
        - Python 3.10
    - Use a virtual environment (venv, virtualenv, or conda) to isolate dependencies

Notes:

- Ensure your NVIDIA driver and CUDA toolkit versions are compatible with your chosen PyTorch wheel.
- We recommend installing PyTorch from the official PyTorch instructions for your CUDA version: https://pytorch.org/get-started/locally/

---

### Recommended: Install with Docker 

We strongly recommend using our pre-configured Docker environment, which contains all necessary dependencies.

```bash
docker pull dexforce/embodichain:ubuntu22.04-cuda12.8
```

After pulling the Docker image, you can run a container with the provided [scripts](../../../docker/docker_run.sh).

```bash
./docker_run.sh [container_name] [data_path]
```

---


### Install EmbodiChain

> **We strongly recommend using a virtual environment to avoid dependency conflicts.**

Clone the EmbodiChain repository:
```bash
git clone https://github.com/DexForce/EmbodiChain.git
```

Install the project in development mode:

```bash
pip install -e . 
```

### Verify Installation
To verify that EmbodiChain is installed correctly, run a simple demo script to create a simulation scene:

```bash
python scripts/tutorials/sim/create_scene.py

# Or run in headless mode.
python scripts/tutorials/sim/create_scene.py --headless
```
---

