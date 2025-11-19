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

### Recommended: Install with *Dexsim* Docker Setup

We strongly recommend using our pre-configured Docker environment, which contains all necessary dependencies.

See: [*Dexsim* Docker setup](http://192.168.3.120/MixedAI/docs_dev/dexsim/markdown/docker.html)

---

### Manual Setup (If Not Using Docker)

PyTorch and its related packages should be installed from the **official PyTorch website**:  
👉 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

### Install EmbodiChain

Clone the EmbodiChain repository:
```bash
git clone http://69.235.177.182:8081/Engine/embodichain.git
```

Install the project in development mode:

```bash
pip install -e . --index-url http://192.168.3.43:8080/simple/ --trusted-host 192.168.3.43
```

Install the project in deploy mode:

```bash
pip install -e ".[deploy]" --index-url http://192.168.3.43:8080/simple/ --trusted-host 192.168.3.43
```


### Verify Installation
To verify that EmbodiChain is installed correctly, run:

```bash
python -c "import embodichain; print(embodichain.__version__)"
```
---

