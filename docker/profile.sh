# Copyright (c) 2021-2026 DexForce Technology Co., Ltd. SPDX-License-Identifier: Apache-2.0
# Sourced by login and interactive shells; retain an explicitly selected env.
. /opt/conda/etc/profile.d/conda.sh
case "${CONDA_DEFAULT_ENV:-${EMBODICHAIN_ENV:-py312}}" in
  py311|py312)
    export EMBODICHAIN_ENV="${CONDA_DEFAULT_ENV:-${EMBODICHAIN_ENV:-py312}}"
    conda activate "$EMBODICHAIN_ENV"
    ;;
esac
