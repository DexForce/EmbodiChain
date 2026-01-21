# Installation

## System Requirements

The following minimum system requirements are recommended to run EmbodiChain reliably. These are the tested configurations during development — other Linux distributions and versions may work but are not officially supported.

- Operating System: 
    - Linux (x86_64): Ubuntu 20.04+

- NVIDIA GPU and drivers:
    - Hardware: NVIDIA GPU with compute capability 7.0 or higher
    - NVIDIA Driver: 535 or higher (recommended 570)


- Python:
    - 3.10
    - 3.11

Notes:

- Ensure your NVIDIA driver is compatible with your chosen PyTorch wheel.
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
pip install -e . --extra-index-url http://pyp.open3dv.site:2345/simple/ --trusted-host pyp.open3dv.site

# Or install with the lerobot extras:
pip install -e .[lerobot] --extra-index-url http://pyp.open3dv.site:2345/simple/ --trusted-host pyp.open3dv.site
```

> [!NOTE]
> * [LeRobot](https://huggingface.co/docs/lerobot/installation) is an optional module for EmbodiChain that provides data saving and loading functionalities for robot learning tasks. Installing with the `lerobot` extras will include this module and its dependencies.

### Verify Installation
To verify that EmbodiChain is installed correctly, run a simple demo script to create a simulation scene:

```bash
python scripts/tutorials/sim/create_scene.py

# Or run in headless mode.
python scripts/tutorials/sim/create_scene.py --headless
```
---

