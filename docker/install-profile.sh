#!/usr/bin/env bash
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd. SPDX-License-Identifier: Apache-2.0
set -euo pipefail
profile=${1:?Usage: install-profile.sh py311|py312|blender311 [environment-prefix]}
case "$profile" in py311|py312|blender311) ;; *) exit 2 ;; esac
prefix=${2:-/opt/conda/envs/$profile}
lock_dir=${EMBODICHAIN_CONSTRAINTS_DIR:-/opt/embodichain/locks}
lock_file="$lock_dir/$profile.txt"
# uv 0.12.5 cannot read the ZIP64 field in the DexSim 0.6.1rc1 CPython 3.12 wheel.
if [[ "$profile" == py312 ]] && dexsim_wheel=$(grep -F 'dexsim_engine-0.6.1rc1-cp312-' "$lock_file"); then
  "$prefix/bin/python" -m pip --isolated install --no-deps --no-cache-dir \
    --force-reinstall --trusted-host pyp.open3dv.site "$dexsim_wheel"
  lock_file=$(mktemp)
  trap 'rm -f "$lock_file"' EXIT
  sed 's/^dexsim-engine @ .*/dexsim-engine==0.6.1rc1/' "$lock_dir/$profile.txt" > "$lock_file"
fi
uv pip sync --python "$prefix/bin/python" --torch-backend cu128 \
  --default-index https://pypi.org/simple --allow-insecure-host pyp.open3dv.site:2345 \
  "$lock_file"
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
