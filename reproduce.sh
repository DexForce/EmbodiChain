#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# EmbodiChain Agent Atomic Action Reproduction Script
# Branch: ljd/agentchoard_dexsim040_migration
# DexSim: 0.3.11
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="embodichain"
GRASP_CACHE_SRC="${SCRIPT_DIR}/cache/grasp_annotator_cache"
GRASP_CACHE_DST="${HOME}/.cache/embodichain/grasp_annotator_cache"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
info()  { echo -e "\033[1;36m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

run_in_env() {
    conda run --no-capture-output -n "${CONDA_ENV}" "$@"
}

# ---------------------------------------------------------------------------
# 1. Environment check
# ---------------------------------------------------------------------------
info "Checking conda environment '${CONDA_ENV}'..."
if ! conda env list | grep -qw "${CONDA_ENV}"; then
    error "Conda environment '${CONDA_ENV}' not found. Please create it first."
fi

info "Verifying DexSim version..."
DEXSIM_VER=$(run_in_env python -c "import dexsim; print(dexsim.__version__)")
if [[ "${DEXSIM_VER}" != "0.3.11" ]]; then
    warn "Expected DexSim 0.3.11, got ${DEXSIM_VER}. Results may differ."
fi

info "Verifying key dependencies..."
run_in_env python -c "import gymnasium, torch, cv2; print('Dependencies OK')"

# ---------------------------------------------------------------------------
# 2. Install grasp cache
# ---------------------------------------------------------------------------
info "Installing grasp annotator cache..."
mkdir -p "${GRASP_CACHE_DST}"
if [[ -d "${GRASP_CACHE_SRC}" ]]; then
    cp -n "${GRASP_CACHE_SRC}"/*.npy "${GRASP_CACHE_DST}/" 2>/dev/null || true
    info "Grasp cache installed to ${GRASP_CACHE_DST}"
else
    warn "No grasp cache found in repo. grasp_cup.py will generate candidates at runtime."
fi

# ---------------------------------------------------------------------------
# 3. Verify agent artifacts
# ---------------------------------------------------------------------------
info "Checking agent graph artifacts..."
ARTIFACTS_OK=true
for dir in single_normal single_recovery dual_normal dual_recovery; do
    if [[ ! -f "${SCRIPT_DIR}/outputs/agent_repro_compare/${dir}/artifacts/agent_task_graph.json" ]]; then
        warn "Missing artifacts for ${dir}"
        ARTIFACTS_OK=false
    fi
done

if [[ "${ARTIFACTS_OK}" == "false" ]]; then
    error "Agent artifacts incomplete. Cannot run pour_water demo in offline mode."
fi
info "Agent artifacts OK."

# ---------------------------------------------------------------------------
# 4. Run unit tests
# ---------------------------------------------------------------------------
info "Running unit tests..."
cd "${SCRIPT_DIR}"
run_in_env pytest -q \
    tests/sim/agent \
    tests/sim/atomic_actions \
    tests/sim/utility/test_solver_utils.py

# ---------------------------------------------------------------------------
# 5. Demo 1: grasp_cup
# ---------------------------------------------------------------------------
info "Running Demo 1: grasp_cup.py..."
TS="$(date +%Y%m%d_%H%M)"
OUT_GRASP="outputs/grasp_cup_repro_${TS}"

run_in_env python scripts/tutorials/gym/grasp_cup.py \
    --output_root "${OUT_GRASP}" \
    2>&1 | tee "${OUT_GRASP}.runner.log" || {
    warn "grasp_cup.py exited with error. Check ${OUT_GRASP}.runner.log"
}

# ---------------------------------------------------------------------------
# 6. Demo 2: pour_water_recovery_compare (full matrix)
# ---------------------------------------------------------------------------
info "Running Demo 2: pour_water_recovery_compare.py --case all..."
OUT_POUR="outputs/pour_water_recovery_compare_repro_${TS}"

run_in_env python scripts/tutorials/gym/pour_water_recovery_compare.py \
    --case all \
    --continue_on_case_failure \
    --output_root "${OUT_POUR}" \
    2>&1 | tee "${OUT_POUR}.runner.log" || {
    warn "pour_water_recovery_compare.py exited with error. Check ${OUT_POUR}.runner.log"
}

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
echo ""
info "=========================================="
info "Reproduction finished."
info "=========================================="
info "Unit tests: see pytest output above"
info "grasp_cup:  ${OUT_GRASP}/summary.tsv"
info "pour_water: ${OUT_POUR}/logs/summary.tsv"
echo ""
info "Check summary.tsv for expectation_matched=True on all cases."
info "Check videos in outputs/*/outputs/videos/ for visual verification."
