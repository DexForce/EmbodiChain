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
uv pip check --python "$prefix/bin/python"
