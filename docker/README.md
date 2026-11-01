# EmbodiChain container environments

Both targets use Ubuntu 22.04, CUDA 12.8 and the Python-selected dependency
profiles introduced by [PR #911](https://github.com/DexForce/EmbodiChain/pull/911).

| Target | Business environments | Included features |
| --- | --- | --- |
| `public` | `py312` (default) | EmbodiChain, official extensions, DexSim, recording/training, GenSim, cuRobo and ONNX CUDA deployment |
| `internal` | `py312` (default), `py311` | All public features in both profiles; Node.js 22 and CI test tools |

RigidFormer extra dependencies and model weights are not preinstalled. Models,
task assets and remote service credentials remain runtime inputs. No production
registry tag is changed by the build/qualification workflow.

## Build and qualify

Run on a Linux amd64 Docker host with NVIDIA Container Toolkit, an appropriate
NVIDIA driver, and access to PyPI, PyTorch's `cu128` index, conda-forge and the
public DexSim wheel server (`pyp.open3dv.site:2345`). BuildKit is required.

```bash
bash docker/build.sh --verify
```

This builds and serially tests both images. It runs the public Python 3.12 and
both internal business environments on one GPU; set `EMBODICHAIN_GPU_DEVICE`
to another host GPU index when necessary. Reports are written under
`output/container-validation/`. `PUBLIC_IMAGE` and `INTERNAL_IMAGE` override
the registry/repository; tags always include the source commit.

The `Container environments` workflow runs on `codex/container-*` pushes and
can also be dispatched manually. It pushes qualified SHA-tagged candidates to
the internal registry only after all three environments pass. The public
candidate is staged in that registry for inspection before Docker Hub publication.

## Runtime selection

```bash
# Public runtime; native driver/graphics libraries come from Container Toolkit.
docker run --rm --gpus all dexforce/embodichain:ubuntu22.04-cuda12.8-py312 \
  python -c 'import torch, dexsim; print(torch.__version__)'

# Internal image: the entrypoint selects the complete legacy environment.
docker run --rm --gpus all -e EMBODICHAIN_ENV=py311 INTERNAL_IMAGE \
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

Each business environment owns its CPython-specific DexSim wheel and Torch
libraries. Do not globally add either environment's `dexsim/lib` or
`site-packages` to `LD_LIBRARY_PATH`/`PYTHONPATH`. Shared FFmpeg/HDF5 libraries
live in `/opt/embodichain/media`. The wheel's engine libraries, shaders and
DLSS assets are retained alongside EGL/Vulkan/OptiX driver support.

## Dependency locks

`profiles/` fixes the qualified Torch/SDK pairings. `locks/` contains the
complete Python dependency sets and explicit conda package URLs/checksums.
The CUDA base, Node base, Miniforge installer and cuRobo commit are pinned.
Ubuntu security packages are installed from the base image's configured apt
repositories; full image reproducibility is identified by the built digest.

Regenerate intentionally with `uv==0.12.5` and conda:

```bash
for profile in py311 py312 blender311; do bash docker/lock.sh "$profile"; done
for profile in py311 py312 media; do python docker/lock-native.py "$profile"; done
```

Use the matching profile when installing additional business dependencies:

```bash
python -m pip install -c "$EMBODICHAIN_CONSTRAINTS_DIR/$EMBODICHAIN_ENV.txt" PACKAGE
```

Constraints are explicit rather than global: isolated CPU SDK jobs must remain
free to install their own Torch/codec profile.

## CI rollout and rollback

CI keeps the existing image and py311 until candidate qualification completes.
Set repository variables `EMBODICHAIN_CI_IMAGE` to the qualified internal image
digest and `EMBODICHAIN_CI_ENV` to the selected business environment. Moving the
main CI to py312 requires PR #911's dependency changes in the checked-out code.

After qualification, publish the public SHA-tagged candidate to Docker Hub,
then move `ubuntu22.04-cuda12.8-py312` and the existing
`ubuntu22.04-cuda12.8` alias to that digest. Preserve the previous public digest
under a legacy tag before promotion. Keep the old internal `v3-py311` tag.

Rollback restores the previous image digest and `EMBODICHAIN_CI_ENV=py311`.
Qualification fails when required CUDA, native, Blender or codec checks skip.
