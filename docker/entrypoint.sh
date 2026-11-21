#!/usr/bin/env bash
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd. SPDX-License-Identifier: Apache-2.0
set -euo pipefail
environment=${EMBODICHAIN_ENV:-py312}
case "$environment" in py311|py312) ;; *) echo "Unknown business environment: $environment" >&2; exit 2 ;; esac
source /opt/conda/etc/profile.d/conda.sh
conda activate "$environment"
exec /opt/nvidia/nvidia_entrypoint.sh "$@"
