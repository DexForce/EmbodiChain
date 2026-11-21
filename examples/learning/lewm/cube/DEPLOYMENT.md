# Cube runtime and resources

## Environment

Use Linux x86-64 with an NVIDIA GPU that supports Vulkan and OptiX and a
driver compatible with CUDA 12.8. The Python 3.12 profile uses DexSim Engine
0.6.0 from the EmbodiChain package index and Torch 2.11 with the CUDA 12.8
wheels. Cube dynamics use the SDK's DexUni CPU solver. The task authors
the gripper equalities on its own Newton scene builder through the
installed SDK.

From a checkout, set up a Python environment and follow the project's
[package installation instructions](../../../../README.md). These commands
show the pinned CUDA and additional Cube dependencies.

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchvision==0.26.0
python -m pip install -e '.[lewm,geometry]' --index-url https://pypi.org/simple \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ --trusted-host pyp.open3dv.site
python -m pip install -r examples/learning/lewm/cube/requirements.txt
```

For an offscreen Linux session, select SDL's offscreen video driver. Provide
a writable `XDG_RUNTIME_DIR` when the session has none. The upstream
reference asset export uses MuJoCo EGL.

```bash
export SDL_VIDEO_DRIVER=offscreen SDL_VIDEODRIVER=offscreen MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Each collection process owns one DexUni World. Tune `--workers` to CPU
and GPU capacity. `--gpu-ids 0 1` distributes render Worlds across the
listed GPUs. Eval uses `--gpu-id` for the camera and `--device` for CEM.

### MuJoCo 3.11 asset compiler

The original reference assets use MuJoCo 3.11.0. To select that compiler
profile, install the example in two Python 3.11 virtual environments, using
`torch==2.7.0` and `torchvision==0.22.0` from the same CUDA 12.8 index.
Install the Cube requirements in both environments, then set
`mujoco==3.11.0` in the asset compiler environment. The runtime environment
uses MuJoCo 3.15.0 as pinned in the requirements.

```bash
.venv-cube-reference/bin/python -m pip install mujoco==3.11.0
embodichain run-example learning/lewm/cube prepare \
  --resource-root outputs/lewm-cube/reference-311 \
  --reference-python .venv-cube-reference/bin/python
```

Run `collect`, `train`, `eval` and `demo` from the runtime environment.
Pass the resource root prepared by your selected compiler to each step.

## Prepared resources

`prepare` clones the pinned stable-worldmodel and OGBench revisions,
downloads the public checkpoint and corpus, verifies their digests, and
compiles the native reference assets and fixed queries under `--resource-root`.
The archive and extracted HDF5 together use about 138 GiB. Experiment
outputs are kept in the caller's output directories.

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
