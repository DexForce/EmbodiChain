echo "exec python ..."

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

run_python_script() {
    local script_name="$1"
    local script_path="$2"

    log_info "Running: ${script_name}"

    if python "$script_path"; then
        log_success "${script_name} succeeded"
        return 0
    else
        log_error "${script_name} failed"
        return 1
    fi
}

run_pytest() {
    local test_path="$1"
    log_info "Running pytest: ${test_path}"

    local pytest_args=(
        "--durations=1000"      # record the slowest 1000 tests
        "--tb=long"             # long traceback
        "-vv"                   # verbose
        "--disable-warnings"    # disable warnings
        "--color=yes"           # enable color output
    )

    pytest "${pytest_args[@]}" "$test_path"
    local status=$?

    # Check if no tests were collected
    if pytest --collect-only "$test_path" | grep -q "collected 0 items"; then
        log_warn "No tests collected: ${test_path}"
        return 0
    fi

    # Check pytest return code
    if [ $status -ne 0 ]; then
        log_error "pytest failed: ${test_path}"
        exit 1
    fi
}

main() {
    echo "Starting scripts..."

    run_python_script "pourwater_offline_test" "tests/datasets/run_pourwater_env_offline.py" || exit 1

    echo "Starting pytest unit tests..."

    for test_dir in tests/*/; do
        # Check whether the directory contains any recursive test_*.py files; do not skip if top-level has none (supports subdirectory structures)
        if ! find "$test_dir" -type f -name 'test_*.py' | grep -q .; then
            log_warn "Skipping empty directory or no test files: ${test_dir}"
            continue
        fi
        run_pytest "$test_dir"
    done

    log_success "All test scripts and unit tests completed!"
}

main "$@"

# # demo
# ./demo/origin_demo.sh CI
# echo -e "\e[32morigin_demo executed successfully\e[0m"
