#!/bin/bash

# shmem4py to OpenSHMEM MLIR Conversion Test Script
# Tests the complete pipeline: shmem4py Python → MLIR IR

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse command line arguments
usage() {
        local script_name
        script_name="$(basename "$0")"
        cat <<EOF
Usage: ./${script_name} [options]

Options:
    --verbose, -v     Enable verbose conversion output
    --help, -h        Show this help message

Environment overrides:
    TEST_DIR          Directory containing shmem4py test files (defaults to python/tests)
EOF
}

VERBOSE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --*)
            echo "ERROR: Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            echo "ERROR: Unexpected positional argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="${TEST_DIR:-${PROJECT_ROOT}/python/tests}"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   shmem4py → OpenSHMEM MLIR Conversion Test Suite         ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

if [[ ${VERBOSE} -eq 1 ]]; then
    echo -e "${CYAN}Verbose mode: ON${NC}"
    echo ""
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found in PATH${NC}"
    exit 1
fi

# Test counters
total_tests=0
passed_tests=0
failed_tests=0

# Function to run a Python test
run_python_test() {
    local test_file="$1"
    local test_name="$(basename "${test_file}")"
    
    total_tests=$((total_tests + 1))
    
    # Run the test
    local output
    if output=$(python3 "${test_file}" 2>&1); then
        echo -e "  ${GREEN}✓${NC} ${test_name}"
        
        # If verbose mode, show output
        if [[ ${VERBOSE} -eq 1 ]]; then
            echo ""
            echo -e "    ${CYAN}━━━ Test Output ━━━${NC}"
            echo "${output}" | sed 's/^/    /'
            echo ""
        fi
        
        passed_tests=$((passed_tests + 1))
        return 0
    else
        echo -e "  ${RED}✗${NC} ${test_name}"
        
        # If verbose mode, show error
        if [[ ${VERBOSE} -eq 1 ]]; then
            echo ""
            echo -e "    ${RED}━━━ Error Output ━━━${NC}"
            echo "${output}" | sed 's/^/    /'
            echo ""
        fi
        
        failed_tests=$((failed_tests + 1))
        return 1
    fi
}

# Stage 1: Frontend Tests (AST Recognition)
echo -e "${YELLOW}═══ Stage 1: shmem4py AST Recognition ═══${NC}"
echo ""

# Run inline Python tests instead of pytest
python3 << 'PYTHON_TESTS'
import sys
sys.path.insert(0, '/home/mbeebe/lanl/llvm/openshmem-mlir')

from python.shmem4py_mlir.frontend import ASTVisitor, Shmem4PyFrontend
import ast

# Test 1: Recognize shmem.init()
code = "import shmem4py as shmem\nshmem.init()"
tree = ast.parse(code)
visitor = ASTVisitor()
visitor.visit(tree)
assert len(visitor.shmem_calls) == 1
assert visitor.shmem_calls[0]['name'] == 'init'
print("  ✓ test_recognize_init")

# Test 2: Recognize all operations
code = """
import shmem4py as shmem
shmem.init()
me = shmem.my_pe()
npes = shmem.n_pes()
shmem.barrier_all()
shmem.finalize()
"""
tree = ast.parse(code)
visitor = ASTVisitor()
visitor.visit(tree)
assert len(visitor.shmem_calls) == 5
names = [c['name'] for c in visitor.shmem_calls]
assert names == ['init', 'my_pe', 'n_pes', 'barrier_all', 'finalize']
print("  ✓ test_recognize_all_operations")

# Test 3: Frontend mapping
frontend = Shmem4PyFrontend()
assert 'init' in frontend.shmem_functions
assert 'finalize' in frontend.shmem_functions
print("  ✓ test_frontend_mapping")

PYTHON_TESTS

echo ""

# Stage 2: End-to-End Compilation Tests
echo -e "${YELLOW}═══ Stage 2: End-to-End Compilation ═══${NC}"
echo ""

if [[ -f "${TEST_DIR}/test_e2e_compilation.py" ]]; then
    run_python_test "${TEST_DIR}/test_e2e_compilation.py"
else
    echo -e "${YELLOW}  No E2E compilation tests found${NC}"
fi

if [[ -f "${TEST_DIR}/test_e2e_hello_shmem.py" ]]; then
    run_python_test "${TEST_DIR}/test_e2e_hello_shmem.py"
else
    echo -e "${YELLOW}  No hello_shmem tests found${NC}"
fi

echo ""

# Summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                    Test Summary                           ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "  Total tests:  ${total_tests}"
echo -e "  ${GREEN}Passed:${NC}       ${passed_tests}"
echo -e "  ${RED}Failed:${NC}       ${failed_tests}"
echo ""

if [[ ${failed_tests} -eq 0 ]]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    if [[ ${VERBOSE} -eq 0 ]]; then
        echo -e "  ${CYAN}Tip: Run with --verbose to see detailed output${NC}"
    fi
    echo ""
    echo -e "${CYAN}Next steps:${NC}"
    echo "  1. Run: ./scripts/test_conversion.sh"
    echo "     to test CIR → OpenSHMEM MLIR → LLVM lowering"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
