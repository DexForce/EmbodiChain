#!/usr/bin/env bash
# Development-only matrix. This is NOT the v2 qualified matrix runner.
# Each worker uses a separate interpreter; failures are retained and exit nonzero.
set -uo pipefail
if [[ ${DEXUNI_ALLOW_PILOT:-} != 1 ]]; then
    echo 'Blocked: read PLAN.md; set DEXUNI_ALLOW_PILOT=1 only for unqualified pilot work.' >&2
    exit 2
fi
REPO=$(cd "$(dirname "$0")/../../../../../../../.." && pwd)
cd "$REPO"
P=scripts/benchmark/tech_reports/physics/platforms/dexuni
read -r -a PLATFORM_LIST <<< "${PLATFORMS:-embodichain isaaclab}"
OUT="${OUT_ROOT:-$REPO/outputs/benchmarks/tech_reports/dexuni/development/s01t/matrix_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
mkdir -p "$OUT/raw_data/_logs"
JOURNAL="$OUT/attempts.tsv"
# Do not reuse an old matrix directory or silently overwrite its attempt log.
(set -o noclobber; printf 'utc\ttag\tplatform\tevent\texit_code\n' > "$JOURNAL") || exit 2
failed=0
both() {
    local tag=$1 platform python rc
    shift
    for platform in "${PLATFORM_LIST[@]}"; do
        case "$platform" in
            dexsim) python="${DEXUNI_DEXSIM_PYTHON:-/home/oem/code/engine/dexsim/.env-3.12/bin/python}" ;;
            embodichain) python="${DEXUNI_EMBODICHAIN_PYTHON:-$REPO/.venv/bin/python}" ;;
            isaaclab) python="${DEXUNI_ISAACLAB_PYTHON:-$HOME/app/IsaacLab/.venv/bin/python}" ;;
            *) echo "Unknown platform: $platform" >&2; return 2 ;;
        esac
        printf '%s\t%s\t%s\tstarted\t\n' "$(date -u +%FT%TZ)" "$tag" "$platform" >> "$JOURNAL"
        "$python" "$P/$platform/s01t.py" "$@" --out-root "$OUT" > "$OUT/raw_data/_logs/${tag}_${platform}.log" 2>&1
        rc=$?
        printf '%s\t%s\t%s\tfinished\t%s\n' "$(date -u +%FT%TZ)" "$tag" "$platform" "$rc" >> "$JOURNAL"
        printf '%s %s rc=%s\n' "$platform" "$tag" "$rc"
        if ((rc != 0)); then failed=1; fi
    done
}
groups=("$@")
[ ${#groups[@]} -eq 0 ] && groups=(nominal)
# The default is a smoke pair, not a full supposedly-qualified experiment.
for group in "${groups[@]}"; do
    case "$group" in
        nominal) both s00_nominal --state s00 --images full ;;
        states) for i in {1..9}; do both "s0${i}_nominal" --state "s0$i" --images keyframes; done ;;
        variants) for v in substeps5 substeps20 direct; do both "s00_$v" --state s00 --variant "$v" --images keyframes; done ;;
        timing) for r in 1 2 3; do both "s00_timing_r$r" --state s00 --kind timing --repeat "$r"; done ;;
        *) echo "Unknown group: $group" >&2; exit 2 ;;
    esac
    rc=$?
    if ((rc != 0)); then failed=1; fi
done
echo "Unqualified pilot outputs: $OUT"
exit "$failed"
