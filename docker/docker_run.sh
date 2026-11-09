#!/usr/bin/env bash
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd. SPDX-License-Identifier: Apache-2.0
# Usage: EMBODICHAIN_DOCKER_IMAGE=... ./docker_run.sh container_name data_path
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 [container_name] [data_path]" >&2
  exit 2
fi
container_name=$1
data_path=$(realpath "$2")
image_name=${EMBODICHAIN_DOCKER_IMAGE:-dexforce/embodichain:ubuntu22.04-cuda12.8-py312}
command=(docker run -itd --gpus all
  -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all
  -e "EMBODICHAIN_ENV=${EMBODICHAIN_ENV:-py312}"
  --name "$container_name" --shm-size 50G --network host
  -v "$data_path:/root/workspace" --workdir /root/workspace)
[[ ! -d /dev/dri ]] || command+=(--device /dev/dri)
for prefix in /usr/share /etc; do
  for resource in vulkan/icd.d/nvidia_icd.json vulkan/implicit_layer.d/nvidia_layers.json glvnd/egl_vendor.d/10_nvidia.json; do
    path="$prefix/$resource"
    [[ ! -f "$path" ]] || command+=(-v "$path:$path:ro")
  done
done
[[ ! -d /usr/share/nvidia ]] || command+=(-v /usr/share/nvidia:/usr/share/nvidia:ro)
if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
  command+=(-v /tmp/.X11-unix:/tmp/.X11-unix -e "DISPLAY=$DISPLAY")
  [[ ! -f "$HOME/.Xauthority" ]] || command+=(-v "$HOME/.Xauthority:/root/.Xauthority:ro")
fi
command+=("$image_name" /bin/bash)
"${command[@]}"
echo "Started $container_name with $image_name (${EMBODICHAIN_ENV:-py312})."
