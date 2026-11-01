#!/usr/bin/env bash
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd. SPDX-License-Identifier: Apache-2.0
set -euo pipefail
profile=${1:?Usage: install-profile.sh py311|py312|blender311 [environment-prefix]}
case "$profile" in py311|py312|blender311) ;; *) exit 2 ;; esac
prefix=${2:-/opt/conda/envs/$profile}
lock_dir=${EMBODICHAIN_CONSTRAINTS_DIR:-/opt/embodichain/locks}
uv pip sync --python "$prefix/bin/python" --torch-backend cu128 \
  --default-index https://pypi.org/simple --allow-insecure-host pyp.open3dv.site:2345 \
  "$lock_dir/$profile.txt"
if [[ "$profile" == blender311 ]]; then
  "$prefix/bin/python" "$(dirname "$0")/repair-bpy-wheel.py"
fi
uv pip check --python "$prefix/bin/python"
if [[ "$profile" == py311 || "$profile" == py312 ]]; then
  mkdir -p "$prefix/etc/conda/activate.d" "$prefix/etc/conda/deactivate.d"
  printf '%s\n' "export EMBODICHAIN_ENV=$profile" \
    > "$prefix/etc/conda/activate.d/embodichain.sh"
  printf '%s\n' 'unset EMBODICHAIN_ENV' \
    > "$prefix/etc/conda/deactivate.d/embodichain.sh"
fi
