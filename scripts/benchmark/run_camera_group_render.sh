#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out_dir="${repo_root}/benchmark-results/camera-groups-$(date +%Y%m%d_%H%M%S)"
python_bin="${PYTHON_BIN:-python}"
benchmark="${repo_root}/scripts/benchmark/camera_group_render.py"

mkdir -p "${out_dir}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

run_case() {
    local name="$1"
    shift
    "${python_bin}" -u "${benchmark}" \
        --envs 4 --warmup 5 --measure 25 --gpu 0 "$@" \
        > "${out_dir}/${name}.log" 2>&1
    tr "\000" "\n" < "${out_dir}/${name}.log" | grep '^RESULT' \
        | tee "${out_dir}/${name}.result"
}

run_case fast_rt --renderer fast-rt
run_case hybrid --renderer hybrid
run_case hybrid_batched --renderer hybrid --batch-camera-group-render

{
    printf 'Camera-group benchmark results\n'
    printf 'topology: 4 environments; head, right wrist and left wrist are separate groups\n'
    cat "${out_dir}"/*.result
} | tee "${out_dir}/summary.txt"

printf 'Saved logs and summary to %s\n' "${out_dir}"
