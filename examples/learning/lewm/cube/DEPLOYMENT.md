# Cube runtime and resources

## Environment

Use Linux x86-64 with an NVIDIA GPU that supports Vulkan and OptiX and a
driver compatible with CUDA 12.8. Training uses CUDA bfloat16. Cube dynamics
use the DexUni CPU solver. Allow about 138 GiB for the downloaded archive and
extracted HDF5, plus space for run outputs.

From the repository root, create a Python 3.12 virtual environment and
install the CUDA wheels, EmbodiChain with geometry support, and the pinned
Cube dependencies.

```bash
python3.12 -m venv .venv-lewm-cube
source .venv-lewm-cube/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchvision==0.26.0
python -m pip install -e '.[geometry]' --index-url https://pypi.org/simple \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ --trusted-host pyp.open3dv.site
python -m pip install -r examples/learning/lewm/cube/requirements.txt
```

For headless execution, select SDL's offscreen driver and MuJoCo EGL. Create
a private runtime directory if the session does not provide one.

```bash
export SDL_VIDEO_DRIVER=offscreen SDL_VIDEODRIVER=offscreen MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ -z "${XDG_RUNTIME_DIR:-}" ]; then
  export XDG_RUNTIME_DIR="$(mktemp -d)"
fi
```

Each collection process owns one DexUni World. Tune `--workers` to CPU
and GPU capacity. `--gpu-ids 0 1` distributes render Worlds across the
listed GPUs. Eval uses `--gpu-id` for the camera and `--device` for CEM.

### MuJoCo 3.11 asset compiler

To run Cube under Python 3.11, use `python3.11` in the setup above and
install `torch==2.7.0` and `torchvision==0.22.0` from the same CUDA 12.8
index. The Cube requirements pin MuJoCo 3.17.0 for the runtime.

To compile assets with the original MuJoCo 3.11.0 version, repeat the Python
3.11 setup in a separate virtual environment named `.venv-cube-reference`.
Set its MuJoCo version, then use `--reference-python` from the Cube runtime
environment with a fresh resource directory.

```bash
.venv-cube-reference/bin/python -m pip install mujoco==3.11.0
CUBE_RESOURCES=outputs/lewm-cube/reference-311
embodichain run-example learning/lewm/cube prepare \
  --resource-root "$CUBE_RESOURCES" \
  --reference-python .venv-cube-reference/bin/python
```

Use this `CUBE_RESOURCES` value for the remaining reproduction commands.
`prepare` reuses existing compiled assets in the selected resource directory.

## Prepared resources

`prepare` clones the pinned stable-worldmodel and OGBench revisions,
downloads the public checkpoint and corpus, verifies their digests, and
compiles the native reference assets and fixed queries under `--resource-root`.

For cached preparation, the resource tree accepts existing verified inputs
in these locations.

| Input | Path below the resource root |
| --- | --- |
| Official extracted HDF5 | `data/cube_single_expert.h5` |
| Official weights | `weights/weights.pt` |
| Pinned stable-worldmodel checkout | `upstream/stable-worldmodel` |
| Pinned OGBench checkout | `upstream/ogbench` |

Copy or link the inputs before running `prepare --local-files-only`.
Preparation records their revisions and digests in `resources.json`.
For training and evaluation, keep each export's `manifest.json` next to
`weights.pt` so the loader can validate the checkpoint and render contract.
