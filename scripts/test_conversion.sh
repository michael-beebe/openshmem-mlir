#!/bin/bash

# Comprehensive OpenSHMEM MLIR Conversion Test Script
# Tests the complete pipeline: CIR → OpenSHMEM MLIR → LLVM MLIR

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse command line arguments
VERBOSE=0
if [[ "$1" == "--verbose" || "$1" == "-v" ]]; then
    VERBOSE=1
fi

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
TEST_DIR="${PROJECT_ROOT}/test"
SHMEM_MLIR_OPT="${BUILD_DIR}/tools/shmem-mlir-opt/shmem-mlir-opt"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   OpenSHMEM MLIR Comprehensive Conversion Test Suite       ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo ""

if [[ ${VERBOSE} -eq 1 ]]; then
    echo -e "${CYAN}Verbose mode: ON${NC}"
    echo ""
fi

# Check if shmem-mlir-opt exists
if [[ ! -f "${SHMEM_MLIR_OPT}" ]]; then
    echo -e "${RED}Error: shmem-mlir-opt not found at ${SHMEM_MLIR_OPT}${NC}"
    echo "Please build the project first: ./scripts/build_openshmem_mlir.sh"
    exit 1
fi

# Test counters
total_tests=0
passed_tests=0
failed_tests=0

# Function to run a test
run_test() {
    local test_file="$1"
    local pass_flags="$2"
    local test_name="$(basename "${test_file}")"
    
    total_tests=$((total_tests + 1))
    
    # Run the test
    local output
    if output=$("${SHMEM_MLIR_OPT}" "${test_file}" ${pass_flags} 2>&1); then
        echo -e "  ${GREEN}✓${NC} ${test_name}"
        
        # If verbose mode, show before and after
        if [[ ${VERBOSE} -eq 1 ]]; then
            echo ""
            echo -e "    ${CYAN}━━━ Input MLIR ━━━${NC}"
            cat "${test_file}" | head -30
            if [[ $(wc -l < "${test_file}") -gt 30 ]]; then
                echo -e "    ${CYAN}... ($(wc -l < "${test_file}") total lines, showing first 30) ...${NC}"
            fi
            echo ""
            echo -e "    ${CYAN}━━━ Converted MLIR ━━━${NC}"
            echo "${output}" | head -30
            if [[ $(echo "${output}" | wc -l) -gt 30 ]]; then
                echo -e "    ${CYAN}... ($(echo "${output}" | wc -l) total lines, showing first 30) ...${NC}"
            fi
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
            echo "${output}"
            echo ""
        fi
        
        failed_tests=$((failed_tests + 1))
        return 1
    fi
}

# Test Stage 1: CIR → OpenSHMEM MLIR Conversion
echo -e "${YELLOW}═══ Stage 1: CIR → OpenSHMEM MLIR Conversion ═══${NC}"
echo ""

CIR_TO_OPENSHMEM_DIR="${TEST_DIR}/Conversion/CIRToOpenSHMEM"
if [[ -d "${CIR_TO_OPENSHMEM_DIR}" ]]; then
    for test_file in "${CIR_TO_OPENSHMEM_DIR}"/*.mlir; do
        [[ -f "${test_file}" ]] || continue
        run_test "${test_file}" "--convert-cir-to-openshmem"
    done
else
    echo -e "${YELLOW}  No CIR conversion tests found${NC}"
fi

echo ""

# Test Stage 2: OpenSHMEM MLIR → LLVM MLIR Lowering
echo -e "${YELLOW}═══ Stage 2: OpenSHMEM MLIR → LLVM MLIR Lowering ═══${NC}"
echo ""

OPENSHMEM_TO_LLVM_DIR="${TEST_DIR}/Conversion/OpenSHMEMToLLVM"
if [[ -d "${OPENSHMEM_TO_LLVM_DIR}" ]]; then
    for test_file in "${OPENSHMEM_TO_LLVM_DIR}"/*.mlir; do
        [[ -f "${test_file}" ]] || continue
        run_test "${test_file}" "--convert-openshmem-to-llvm"
    done
else
    echo -e "${YELLOW}  No OpenSHMEM lowering tests found${NC}"
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
        echo -e "  ${CYAN}Tip: Run with --verbose to see conversion output${NC}"
    fi
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
