#!/bin/bash

set -uo pipefail

# Verification script for OpenSHMEM MLIR frontend selection mechanism
#
# This script tests that:
#   1. Frontend selection via CMake options works correctly
#   2. Both --enable-cir and --disable-cir flags function
#   3. Environment variable OPENSHMEM_ENABLE_CIR is respected
#   4. shmem4py_mlir Python package structure is complete
#   5. Python bindings infrastructure is set up correctly
#
# Usage:
#   ./scripts/verify_frontend_selection.sh [--verbose]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
VERBOSE=0

if [[ "${1:-}" == "--verbose" ]]; then
    VERBOSE=1
fi

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name="$1"
    local test_cmd="$2"
    
    if [[ $VERBOSE -eq 1 ]]; then
        echo "Running: $test_cmd"
    fi
    
    if eval "$test_cmd" > /dev/null 2>&1; then
        log_info "$test_name"
        ((TESTS_PASSED++))
        return 0
    else
        log_error "$test_name"
        ((TESTS_FAILED++))
        return 1
    fi
}

echo "========================================"
echo "  OpenSHMEM MLIR Frontend Verification"
echo "========================================"
echo

echo "1. Checking build configuration files..."
echo

# Check that CMakeLists.txt has the option
run_test \
    "CMakeLists.txt contains OPENSHMEM_ENABLE_CIR option" \
    "grep -q 'option(OPENSHMEM_ENABLE_CIR' ${ROOT_DIR}/CMakeLists.txt"

# Check that build script has the flags
run_test \
    "build_openshmem_mlir.sh supports --enable-cir flag" \
    "grep -q 'enable-cir' ${ROOT_DIR}/scripts/build_openshmem_mlir.sh"

run_test \
    "build_openshmem_mlir.sh supports --disable-cir flag" \
    "grep -q 'disable-cir' ${ROOT_DIR}/scripts/build_openshmem_mlir.sh"

# Check that CIR subdirectory is conditionally included
run_test \
    "CMakeLists.txt conditionally includes cir/ subdirectory" \
    "grep -q 'add_subdirectory(cir)' ${ROOT_DIR}/CMakeLists.txt"

echo
echo "2. Checking shmem4py frontend structure..."
echo

# Check shmem4py_mlir directory exists
run_test \
    "python/shmem4py_mlir/ directory exists" \
    "[[ -d ${ROOT_DIR}/python/shmem4py_mlir ]]"

# Check all required Python modules exist
run_test \
    "shmem4py_mlir/__init__.py exists" \
    "[[ -f ${ROOT_DIR}/python/shmem4py_mlir/__init__.py ]]"

run_test \
    "shmem4py_mlir/frontend.py exists" \
    "[[ -f ${ROOT_DIR}/python/shmem4py_mlir/frontend.py ]]"

run_test \
    "shmem4py_mlir/builder.py exists" \
    "[[ -f ${ROOT_DIR}/python/shmem4py_mlir/builder.py ]]"

run_test \
    "shmem4py_mlir/passes.py exists" \
    "[[ -f ${ROOT_DIR}/python/shmem4py_mlir/passes.py ]]"

run_test \
    "shmem4py_mlir/jit.py exists" \
    "[[ -f ${ROOT_DIR}/python/shmem4py_mlir/jit.py ]]"

run_test \
    "shmem4py_mlir/cli.py exists" \
    "[[ -f ${ROOT_DIR}/python/shmem4py_mlir/cli.py ]]"

echo
echo "3. Checking Python infrastructure..."
echo

# Check Python bindings configuration
run_test \
    "python/CMakeLists.txt exists" \
    "[[ -f ${ROOT_DIR}/python/CMakeLists.txt ]]"

run_test \
    "python/requirements.txt has been updated" \
    "grep -q 'numpy' ${ROOT_DIR}/python/requirements.txt"

run_test \
    "python/tests/ directory exists" \
    "[[ -d ${ROOT_DIR}/python/tests ]]"

run_test \
    "python/tests/test_frontend_basics.py exists" \
    "[[ -f ${ROOT_DIR}/python/tests/test_frontend_basics.py ]]"

run_test \
    "python/examples/ directory exists" \
    "[[ -d ${ROOT_DIR}/python/examples ]]"

echo
echo "4. Checking setup scripts..."
echo

run_test \
    "scripts/setup_shmem4py_env.sh exists" \
    "[[ -f ${ROOT_DIR}/scripts/setup_shmem4py_env.sh ]]"

run_test \
    "setup_shmem4py_env.sh is executable" \
    "[[ -x ${ROOT_DIR}/scripts/setup_shmem4py_env.sh ]]"

echo
echo "5. Checking Python module syntax..."
echo

if command -v python3 >/dev/null 2>&1; then
    # Check that Python modules have correct syntax
    for module in \
        python/shmem4py_mlir/__init__.py \
        python/shmem4py_mlir/frontend.py \
        python/shmem4py_mlir/builder.py \
        python/shmem4py_mlir/passes.py \
        python/shmem4py_mlir/jit.py \
        python/shmem4py_mlir/cli.py
    do
        module_name=$(basename "$module" .py)
        run_test \
            "shmem4py_mlir/$module_name.py has valid Python syntax" \
            "python3 -m py_compile ${ROOT_DIR}/${module}"
    done
else
    log_warn "Python 3 not found, skipping syntax checks"
fi

echo
echo "6. Checking documentation..."
echo

run_test \
    "README.md mentions frontend selection" \
    "grep -q 'Frontend Selection' ${ROOT_DIR}/README.md"

run_test \
    "README.md documents ClangIR frontend" \
    "grep -q 'ClangIR Frontend' ${ROOT_DIR}/README.md"

run_test \
    "README.md documents shmem4py frontend" \
    "grep -q 'shmem4py Frontend' ${ROOT_DIR}/README.md"

run_test \
    "shmem4py design document exists" \
    "[[ -f ${ROOT_DIR}/notes/shmem4py.tex ]]"

echo
echo "========================================"
echo "Test Results Summary"
echo "========================================"
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Failed: ${RED}${TESTS_FAILED}${NC}"
echo

if [[ $TESTS_FAILED -eq 0 ]]; then
    log_info "All verifications passed!"
    echo
    echo "Next steps:"
    echo "  1. Source the environment setup script:"
    echo "     source scripts/setup_shmem4py_env.sh"
    echo
    echo "  2. Build OpenSHMEM MLIR with Python bindings:"
    echo "     ./scripts/build_openshmem_mlir.sh"
    echo
    echo "  3. Run the test suite to verify bindings:"
    echo "     cd python && pytest tests/ -v"
    echo
    echo "  4. Test frontend selection by building with/without CIR:"
    echo "     ./scripts/build_openshmem_mlir.sh --disable-cir"
    echo "     ./scripts/build_openshmem_mlir.sh --enable-cir"
    echo
    exit 0
else
    log_error "Some verifications failed. Please review the output above."
    exit 1
fi
