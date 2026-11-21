#!/usr/bin/env bash
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd. SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
public_image=${1:?Usage: docker/validate.sh public-image internal-image [report-directory]}
internal_image=${2:?Provide the internal image}
reports=$(realpath -m "${3:-$root/output/container-validation}")
mkdir -p "$reports"
common=(--rm --gpus "device=${EMBODICHAIN_GPU_DEVICE:-0}" --shm-size 4g
  -e NVIDIA_DRIVER_CAPABILITIES=all -e HF_HUB_OFFLINE=1
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2
  -e PYTHONDONTWRITEBYTECODE=1
  -v "$root:/repo:ro" -v "$reports:/reports"
  -v "${EMBODICHAIN_CONTAINER_CACHE:-/cache}:/cache"
  -e EMBODICHAIN_DATA_ROOT=/cache/embodichain_data)
for path in /usr/share/vulkan/icd.d /usr/share/vulkan/implicit_layer.d /usr/share/glvnd/egl_vendor.d /usr/share/nvidia; do
  [[ ! -d "$path" ]] || common+=(-v "$path:$path:ro")
done
docker run "${common[@]}" "$internal_image" bash -lc \
  'set -euo pipefail
   conda activate py311
   test "$EMBODICHAIN_ENV" = py311
   test "$(python -c "import sys; print(sys.version_info.minor)")" = 11
   conda activate py312
   test "$EMBODICHAIN_ENV" = py312
   test "$(python -c "import sys; print(sys.version_info.minor)")" = 12'
docker run "${common[@]}" "$public_image" bash -c \
  'test ! -d /opt/conda/envs/py311 && test -x /opt/conda/envs/blender311/bin/python'
for pair in public:py312 internal:py311 internal:py312; do
  target=${pair%:*}; environment=${pair#*:}
  if [[ $target == public ]]; then image=$public_image; else image=$internal_image; fi
  docker run "${common[@]}" -e "EMBODICHAIN_ENV=$environment" "$image" \
    python /opt/embodichain/verify.py --suite all --repo /repo \
    --report "/reports/$target-$environment.json"
  docker run "${common[@]}" -e "EMBODICHAIN_ENV=$environment" "$image" \
    bash -lc 'test "$(python -c "import sys; print(sys.version_info.minor)")" = "${EMBODICHAIN_ENV#py3}"'
done
