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

    log_info "执行: ${script_name}"

    # 设置环境变量
    # export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

    if python "$script_path"; then
        log_success "${script_name} 执行成功"
        return 0
    else
        log_error "${script_name} 执行失败"
        return 1
    fi
}

run_pytest() {
    local test_path="$1"
    log_info "执行pytest: ${test_path}"

    local pytest_args=(
        "--durations=1000"      # 记录最慢的1000个测试用例的执行时间
        "--tb=long"             # 详细回溯信息
        "-vv"                   # 详细输出
        "--disable-warnings"    # 禁用警告
        "--color=yes"           # 启用颜色输出
    )

    pytest "${pytest_args[@]}" "$test_path"
    local status=$?

    # 检查是否没有测试用例被收集
    if pytest --collect-only "$test_path" | grep -q "collected 0 items"; then
        log_warn "没有测试用例被收集: ${test_path}"
        return 0
    fi

    # 检查 pytest 返回码
    if [ $status -ne 0 ]; then
        log_error "pytest测试失败: ${test_path}"
        exit 1
    fi
}

main() {
    echo "开始执行脚本..."

    # 执行 Python 测试脚本
    # run_python_script "pourwater环境离线测试" "tests/datasets/run_pourwater_env_offline.py" || exit 1
    # run_python_script "通用环境离线测试" "tests/datasets/run_env_offline.py" || exit 1
    # run_python_script "环境在线测试" "tests/datasets/run_env_online.py" || exit 1
    run_python_script "real2sim环境测试" "tests/datasets/run_real2sim_offline.py" || exit 1
    run_python_script "real2sim v3环境测试" "tests/datasets/run_real2sim_offline_v3.py" || exit 1

    # TODO: Open new env tests
    run_python_script "pourwater_v3_offline_test" "tests/datasets/run_pourwater_v3_env_offline.py" || exit 1

    echo "开始执行pytest单元测试..."

    for test_dir in tests/*/; do
        # 检查目录中是否含有任何递归 test_*.py 文件；顶层没有也不要跳过（支持子目录结构）
        if ! find "$test_dir" -type f -name 'test_*.py' | grep -q .; then
            log_warn "跳过空目录或无测试文件: ${test_dir}"
            continue
        fi
        run_pytest "$test_dir"
    done

    log_success "所有测试脚本和单元测试执行完成！"
}

main "$@"

# # demo
# ./demo/origin_demo.sh CI
# echo -e "\e[32morigin_demo执行成功\e[0m"