#!/usr/bin/env bash
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd. SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/.."
profile=${1:?Usage: docker/lock.sh py311|py312|blender311}
case "$profile" in
  py311) python_version=3.11; extras=(pyproject.toml --extra gensim --extra policy-deploy) ;;
  py312) python_version=3.12; extras=(pyproject.toml --extra gensim --extra policy-deploy) ;;
  blender311) python_version=3.11; extras=() ;;
  *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac
uv pip compile "${extras[@]}" "docker/profiles/$profile.in" \
  --python-version "$python_version" --python-platform x86_64-manylinux_2_35 \
  --torch-backend cu128 --default-index https://pypi.org/simple \
  --index-strategy unsafe-best-match --allow-insecure-host pyp.open3dv.site:2345 \
  --no-emit-package embodichain --no-annotate \
  --custom-compile-command "docker/lock.sh $profile" \
  --output-file "docker/locks/$profile.txt"
