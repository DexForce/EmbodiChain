# Rebuilding the Torch 2.11 operators

`runtime-manifest.torch211.json` identifies the locally tested binaries. They
are external artifacts, not bundled wheels or a published installation extra.
Rebuild in an isolated Linux x86_64 / Python 3.12 environment already using
Torch 2.11.0+cu128, Torchvision 0.26.0+cu128, NumPy 2.2.6 and Einops 0.8.2.
CUDA Toolkit 12.8, a compatible C++ compiler, CMake, Ninja, Git and the Python
`setuptools`/`wheel` build packages are required.
This profile targets SM120; it does not qualify other GPU architectures.

The following commands preserve the Torch installation. Set `WP_REPOSITORY` to
this checkout and choose a new scratch directory for `WP_BUILD`.

```bash
export WP_REPOSITORY="$PWD"
export WP_BUILD="$(mktemp -d)"
python -m pip install --no-deps --target "$WP_BUILD/yapf" 'yapf==0.30.0'
python -m pip install --no-deps --target "$WP_BUILD/patchelf" 'patchelf==0.17.2.4'

git clone https://github.com/isl-org/Open3D.git "$WP_BUILD/Open3D"
git -C "$WP_BUILD/Open3D" checkout 1e7b17438687a0b0c1e5a7187321ac7044afe275
git -C "$WP_BUILD/Open3D" config core.abbrev 8
git -C "$WP_BUILD/Open3D" apply "$WP_REPOSITORY/examples/learning/worldparticle/open3d-build.patch"
cmake -S "$WP_BUILD/Open3D" -B "$WP_BUILD/open3d-build" -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(command -v python)" \
  -DBUILD_CUDA_MODULE=ON -DCMAKE_CUDA_ARCHITECTURES=120-real \
  -DBUILD_COMMON_CUDA_ARCHS=OFF -DBUILD_PYTORCH_OPS=ON \
  -DBUILD_TENSORFLOW_OPS=OFF -DBUILD_GUI=OFF -DBUILD_WEBRTC=OFF \
  -DBUILD_ISPC_MODULE=OFF -DBUILD_EXAMPLES=OFF -DBUILD_UNIT_TESTS=OFF \
  -DWITH_IPP=OFF -DCMAKE_INSTALL_PREFIX="$WP_BUILD/open3d-install"
PYTHONPATH="$WP_BUILD/yapf${PYTHONPATH:+:$PYTHONPATH}" \
  cmake --build "$WP_BUILD/open3d-build" --target pip-package --parallel 8
python "$WP_REPOSITORY/examples/learning/worldparticle/relocate_open3d_wheel.py" \
  "$WP_BUILD/open3d-build/lib/python_package/pip_package/open3d-0.19.0+1e7b1743-cp312-cp312-manylinux_2_35_x86_64.whl" \
  --output "$WP_BUILD/wheels" --build-tag 1torch2110cu128sm120 \
  --patchelf "$WP_BUILD/patchelf/bin/patchelf"

git clone https://github.com/Dao-AILab/flash-attention.git "$WP_BUILD/flash-attention"
git -C "$WP_BUILD/flash-attention" checkout 060c9188beec3a8b62b33a3bfa6d5d2d44975fab
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=120 \
  MAX_JOBS=12 NVCC_THREADS=2 CUDA_HOME=/usr/local/cuda \
  python -m pip wheel --no-build-isolation --no-deps \
  "$WP_BUILD/flash-attention" -w "$WP_BUILD/wheels"
python -m wheel unpack "$WP_BUILD/wheels/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl" \
  -d "$WP_BUILD/flash-unpacked"
python -m wheel pack "$WP_BUILD/flash-unpacked/flash_attn-2.8.3" \
  --build-number 1torch2110cu128sm120 -d "$WP_BUILD/wheels"
```

Wheel timestamps and build paths may change the resulting file hashes. Record
new hashes and versions when qualifying a rebuild; do not reuse another
binary's hash or relabel a Torch 2.7 wheel as Torch 2.11. The Open3D relocation
utility writes an adjacent JSON report of each patched ELF runpath.

Install the two tagged wheels in the isolated environment, or extract them to
a separate import directory. Main's SDK pin remains `0.6.0rc1`, which does not
contain `ParticleInferenceSolver`. The model transition can run independently;
native deployment was checked using SDK `0.6.0rc2` in a source checkout. A
default-install SDK upgrade is a separate compatibility change, not hidden in
an operator dependency extra.

Validation covered CUDA fixed-radius search, BF16 FlashAttention against SDPA,
all four published models for two native steps, and standalone execution on
a nondefault stream without Newton/DexSim imports or input mutation. This is an
execution check, not paper-scale accuracy or historical-runtime equivalence.
Repeated BF16 rollouts are not bitwise deterministic: the unchanged source
uses CUDA reductions and hash-neighbor construction with no determinism
guarantee. No claim of native/standalone numerical parity is made for the real
network; adapter transport/reset contracts are checked separately with a
deterministic predictor.
