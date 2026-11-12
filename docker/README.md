# EmbodiChain container environments

Both targets use Ubuntu 22.04, CUDA 12.8 and the Python-selected dependency
profiles introduced by [PR #911](https://github.com/DexForce/EmbodiChain/pull/911).

| Target | Business environments | Included features |
| --- | --- | --- |
| `public` | `py312` (default) | EmbodiChain, official extensions, DexSim, recording/training, GenSim, cuRobo and ONNX CUDA deployment |
| `internal` | `py312` (default), `py311` | All public features in both profiles; Node.js 22 and CI test tools |

RigidFormer extra dependencies and model weights are not preinstalled. Models,
task assets and remote service credentials remain runtime inputs.

This checkout's profiles select DexSim 0.6.0rc1 and Newton 1.7.0. The published
tags and image digests recorded in [VALIDATION.md](VALIDATION.md) describe the
earlier DexSim 0.5.1rc1 runtime. Build and qualify a source-commit candidate for
the current profiles before promoting a replacement; the historical image
qualification does not validate the new dependency set.

## Install and run

Use a Linux amd64 host with an NVIDIA driver **≥ 580.00**, Docker and NVIDIA
Container Toolkit. The current DLSS integration uses SDK and Super Resolution
(SR) **310.7.0** and the OVRTX-aligned Ray Reconstruction (RR) runtime
**310.7.128**, which require this minimum host driver version. Confirm GPU
access and the driver version with `nvidia-smi` on the host before starting
the container.

| Runtime | Published image |
| --- | --- |
| Public Python 3.12 | `dexforce/embodichain:ubuntu22.04-cuda12.8-py312` |
| Public compatibility alias, same image | `dexforce/embodichain:ubuntu22.04-cuda12.8` |
| Internal Python 3.11 / 3.12 | `192.168.3.13:5000/dexsdk:ubuntu22.04-cuda12.8.0-h5ffmpeg-v4-py311-py312` |

Pull the public image and mount a working directory:

```bash
docker pull dexforce/embodichain:ubuntu22.04-cuda12.8-py312
docker run --rm --gpus all -it --shm-size 8g \
  -v "$PWD":/workspace --workdir /workspace \
  dexforce/embodichain:ubuntu22.04-cuda12.8-py312 bash
```

EmbodiChain and its runtime dependencies are already installed. Check the
selected environment inside the container:

```bash
python -c 'import sys, torch, dexsim; print(sys.version); print(torch.__version__); assert torch.cuda.is_available()'
embodichain list-task
```

The internal registry requires access to the internal network. Add
`"insecure-registries": ["192.168.3.13:5000"]` to the existing Docker daemon
configuration (`/etc/docker/daemon.json`) and restart Docker to enable its HTTP
endpoint. Start its Python 3.11 environment with:

```bash
docker pull 192.168.3.13:5000/dexsdk:ubuntu22.04-cuda12.8.0-h5ffmpeg-v4-py311-py312
docker run --rm --gpus all -it --shm-size 8g -e EMBODICHAIN_ENV=py311 \
  -v "$PWD":/workspace --workdir /workspace \
  192.168.3.13:5000/dexsdk:ubuntu22.04-cuda12.8.0-h5ffmpeg-v4-py311-py312 bash
```

Use `EMBODICHAIN_ENV=py312` for the other environment; `py312` is the image
default. CI explicitly activates `py311`.

The images contain the EmbodiChain wheel built from the source revision in
[VALIDATION.md](VALIDATION.md). For source development, mount your checkout as
above and install it into the selected environment. Select the checkout's
dependency locks, which may be newer than the image's bundled locks:

```bash
export EMBODICHAIN_CONSTRAINTS_DIR="$PWD/docker/locks"
python -m pip install -e '.[gensim]' \
  -c "$EMBODICHAIN_CONSTRAINTS_DIR/$CONDA_DEFAULT_ENV.txt" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ \
  --trusted-host pyp.open3dv.site
```

Stable tags reuse the qualified image digests. The source-commit candidate
tags and previous public image remain available for reproducibility and rollback;
see [VALIDATION.md](VALIDATION.md) for digests and CI evidence.

## Build and qualify

Run on a Linux amd64 Docker host with NVIDIA Container Toolkit, an appropriate
NVIDIA driver, and access to PyPI, PyTorch's `cu128` index, conda-forge and the
public DexSim wheel server (`pyp.open3dv.site:2345`). BuildKit is required.
The Miniforge installer uses a fixed release and SHA256. If its versioned
download name fails, the build retries the official alias within that same
release and checks the same SHA256 before execution.

```bash
bash docker/build.sh --verify
```

This builds and serially tests both images. It runs the public Python 3.12 and
both internal business environments on one GPU; set `EMBODICHAIN_GPU_DEVICE`
to another host GPU index when necessary. Reports are written under
`output/container-validation/`. `PUBLIC_IMAGE` and `INTERNAL_IMAGE` override
the registry/repository; tags always include the source commit.

Reused behavior and native suites run from a temporary directory containing
only tests and supporting scripts. Isolated Python ignores inherited import
paths; checks before and after pytest require the core, tasks and robot
packages to resolve to the installed wheel. Reports include those locations,
and configuration checks read installed package resources.

Container build and publication workflows are maintained as a separate
automation change. The build/qualification scripts here remain runnable
manually; qualified public candidates can be staged in the internal registry
for inspection before Docker Hub publication.

## Runtime selection

```bash
# Public runtime; native driver/graphics libraries come from Container Toolkit.
docker run --rm --gpus all dexforce/embodichain:ubuntu22.04-cuda12.8-py312 \
  python -c 'import torch, dexsim; print(torch.__version__)'

# Internal image: the entrypoint selects the complete legacy environment.
docker run --rm --gpus all -e EMBODICHAIN_ENV=py311 \
  192.168.3.13:5000/dexsdk:ubuntu22.04-cuda12.8.0-h5ffmpeg-v4-py311-py312 \
  python -c 'import sys, torch; print(sys.version, torch.__version__)'

# Existing container: choose an environment explicitly.
docker exec CONTAINER conda run --no-capture-output -n py311 python --version
```

Interactive and login shells preserve the selected business environment.
`conda activate py311` / `conda activate py312` switch it inside a shell.
`docker/docker_run.sh NAME DATA_PATH` retains its two arguments and mounts data
at `/root/workspace`. Set `EMBODICHAIN_DOCKER_IMAGE` and `EMBODICHAIN_ENV` to
select an image/environment. Headless startup does not require X11 files.

## Blender and native library isolation

Blender 4.5 LTS uses Python 3.11 and NumPy 1.26, whereas LeRobot 0.6.1 uses
Python 3.12 and NumPy 2. Both images therefore contain a small `blender311`
worker environment in addition to the business environment(s). It has no
Torch or DexSim installation.

The official bpy 4.5.15 cp311 artifact has an upstream cp39 tag inside its
WHEEL metadata. Its SHA256 is pinned; the install step corrects that tag and
the corresponding RECORD checksum without modifying native binaries or the
download cache. The complete Blender render/mesh/baking checks qualify the
installed artifact.

`EMBODICHAIN_BLENDER_PYTHON` selects that interpreter. GenSim sends file-based
mesh/render jobs to the packaged worker automatically. The worker does not
import the main package and does not inherit business Python/native-library
paths. Blender stdout is diagnostic output; JSON response files carry results.
Timeout, exceptions and pipeline cancellation terminate and reap the worker.

For a non-container installation, create a separate Python 3.11 environment,
then run `docker/install-profile.sh blender311 ENVIRONMENT_PREFIX` with
`EMBODICHAIN_CONSTRAINTS_DIR` set to the repository's `docker/locks` directory.
This installs the locked dependencies and applies the metadata correction.
Set `EMBODICHAIN_BLENDER_PYTHON` to that environment's Python executable.
`.[gensim]` installs the Python-facing GenSim dependencies; Blender is supplied
by the independent runtime.

If Eevee cannot create a graphics context through the `bpy` module, select the
standalone Blender application with `EMBODICHAIN_BLENDER_EXECUTABLE`. This takes
precedence over `EMBODICHAIN_BLENDER_PYTHON` and runs the same worker with Blender's
background/factory startup, which initializes the graphics context before rendering.
Install the worker dependencies into the application's bundled Python, keeping
Torch and DexSim in the business environment:

```bash
uv pip install --python /opt/blender/4.5/python/bin/python3.11 \
  --constraint docker/locks/blender311.txt Pillow scipy
export EMBODICHAIN_BLENDER_EXECUTABLE=/opt/blender/blender
```

Adjust the paths to the installed Blender application. The selectors can also
be set in the shared GenSim `.env` or `EMBODICHAIN_ENV_FILE`; process values take
precedence. The launcher reads these settings without changing the process
environment. Both launch modes use the same file protocol, error propagation,
environment isolation and process cleanup.

Each business environment owns its CPython-specific DexSim wheel and Torch
libraries. Do not globally add either environment's `dexsim/lib` or
`site-packages` to `LD_LIBRARY_PATH`/`PYTHONPATH`. Shared FFmpeg/HDF5 libraries
live in `/opt/embodichain/media`. The wheel's engine libraries, shaders and
DLSS assets are retained alongside EGL/Vulkan/OptiX driver support.

## Dependency locks

`profiles/` fixes the qualified Torch/SDK pairings. `locks/` contains the
complete Python dependency sets and explicit conda package URLs/checksums.
The CUDA base, Node base, Miniforge installer and cuRobo commit are pinned.
Both DexSim wheel URLs carry SHA256 pins, which the installer verifies even
though the wheel server uses HTTP. These match the published index checksums
for DexSim 0.6.0rc1. Build and qualify both updated profiles before promoting
new image digests.
DexSim and LeRobot currently require different OpenCV distributions sharing
`cv2`. Python-selected dependencies pin both to the same release; qualification
checks the imported binding as well as distribution metadata.
Ubuntu security packages are installed from the base image's configured apt
repositories; full image reproducibility is identified by the built digest.

Regenerate intentionally with `uv==0.12.5` and conda:

```bash
for profile in py311 py312 blender311; do bash docker/lock.sh "$profile"; done
for profile in py311 py312 media; do python docker/lock-native.py "$profile"; done
```

Use the matching profile when installing additional business dependencies:

```bash
python -m pip install -c "$EMBODICHAIN_CONSTRAINTS_DIR/$CONDA_DEFAULT_ENV.txt" PACKAGE
```

Constraints are explicit rather than global: isolated CPU SDK jobs must remain
free to install their own Torch/codec profile.

## CI rollout and rollback

CI defaults to the published internal `v4-py311-py312` image
and explicitly activates `py311`. The image's interactive default remains
`py312`. The image reference and activation command are explicit
in the existing workflows/shared action; the existing job graph, triggers,
test selection and installation commands are preserved. The SDK compatibility
matrix selects Python 3.11 and 3.12 independently of this default.

Builds continue producing source-commit candidate tags. After qualification
and successful CI, promote the existing manifests to the public
`ubuntu22.04-cuda12.8-py312` / `ubuntu22.04-cuda12.8` tags and the internal
`ubuntu22.04-cuda12.8.0-h5ffmpeg-v4-py311-py312` tag. Promotion preserves the
image digest and does not rebuild the layers. Preserve the previous public
digest under a legacy tag before moving the alias; retain the old internal
`v3-py311` tag.

For rollback, restore the previous public digest from its legacy tag. CI can
select the retained internal source-commit candidate or `v3-py311` reference
and activate `py311`.
Qualification fails when required CUDA, native, Blender or codec checks skip.
