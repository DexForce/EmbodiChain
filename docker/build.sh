#!/usr/bin/env bash
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd. SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/.."
revision=$(git rev-parse HEAD)
suffix=${revision:0:12}
public_image=${PUBLIC_IMAGE:-dexforce/embodichain}:ubuntu22.04-cuda12.8-py312-$suffix
internal_image=${INTERNAL_IMAGE:-192.168.3.13:5000/dexsdk}:ubuntu22.04-cuda12.8.0-h5ffmpeg-v4-py311-py312-$suffix
export DOCKER_BUILDKIT=1
for target in public internal; do
  if [[ $target == public ]]; then image=$public_image; else image=$internal_image; fi
  docker build --platform linux/amd64 --progress plain --file docker/Dockerfile \
    --target "$target" --build-arg "SOURCE_REVISION=$revision" --tag "$image" .
done
echo "PUBLIC_IMAGE=$public_image"
echo "INTERNAL_IMAGE=$internal_image"
if [[ ${1:-} == --verify ]]; then
  bash docker/validate.sh "$public_image" "$internal_image"
fi
