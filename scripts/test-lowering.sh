#!/bin/bash

# OpenSHMEM MLIR Test Lowering Script
# This script automates testing of OpenSHMEM dialect operations and their lowering to LLVM

shopt -s globstar  # Enable ** globbing

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
TEST_DIR="${PROJECT_ROOT}/test"
OPENSHMEM_OPT="${BUILD_DIR}/tools/openshmem-opt/openshmem-opt"

echo -e "${BLUE}OpenSHMEM MLIR Test Lowering Script${NC}"
echo "Project root: ${PROJECT_ROOT}"
echo "Build dir: ${BUILD_DIR}"
echo "Test dir: ${TEST_DIR}"
echo ""

# Check if openshmem-opt exists
if [[ ! -f "${OPENSHMEM_OPT}" ]]; then
    echo -e "${RED}Error: openshmem-opt not found at ${OPENSHMEM_OPT}${NC}"
    echo "Please run ./scripts/build-openshmem-mlir.sh first"
    exit 1
fi

# Function to run a test file
run_test() {
    local test_file="$1"
    local test_name="$(basename "${test_file}" .mlir)"
    
    echo -e "${YELLOW}Testing: ${test_name}${NC}"
    
    # Test 1: Parse and print (basic dialect functionality)
    echo "  - Parsing and printing..."
    if "${OPENSHMEM_OPT}" "${test_file}" > /dev/null 2>&1; then
        echo -e "    ${GREEN}✓ Parse/print successful${NC}"
    else
        echo -e "    ${RED}✗ Parse/print failed${NC}"
        return 1
    fi
    
    # Test 2: Verify the operations can be parsed correctly
    echo "  - Verifying operations..."
    if "${OPENSHMEM_OPT}" "${test_file}" --verify-each > /dev/null 2>&1; then
        echo -e "    ${GREEN}✓ Verification successful${NC}"
    else
        echo -e "    ${RED}✗ Verification failed${NC}"
        return 1
    fi
    
    echo ""
    return 0
}

# Function to test lowering (when conversion pass is available)
test_lowering() {
    local test_file="$1"
    local test_name="$(basename "${test_file}" .mlir)"
    
    echo -e "${YELLOW}Testing lowering: ${test_name}${NC}"
    
    # Test lowering to LLVM dialect
    echo "  - Attempting lowering to LLVM..."
    
    # Check if the pass is available
    if "${OPENSHMEM_OPT}" --help | grep -q "convert-openshmem-to-llvm"; then
        echo "  - Running conversion pass..."
        
        # Show MLIR before lowering
        echo ""
        echo -e "${BLUE}=== MLIR BEFORE LOWERING ===${NC}"
        "${OPENSHMEM_OPT}" "${test_file}"
        
        echo ""
        echo -e "${BLUE}=== MLIR AFTER LOWERING ===${NC}"
        
        # Test the lowering and show output
        if "${OPENSHMEM_OPT}" "${test_file}" --convert-openshmem-to-llvm; then
            echo ""
            echo -e "    ${GREEN}✓ Lowering successful${NC}"
            return 0
        else
            echo ""
            echo -e "    ${RED}✗ Lowering failed${NC}"
            return 1
        fi
    else
        echo -e "    ${YELLOW}⚠ Lowering pass not yet registered${NC}"
        return 2  # Different return code for "not implemented yet"
    fi
    
    echo ""
}

# Main test execution
echo -e "${BLUE}Running OpenSHMEM dialect tests...${NC}"
echo ""

# Track test results
total_tests=0
passed_tests=0
failed_tests=0
lowering_not_implemented=0

# Test the main operations file
if [[ -f "${TEST_DIR}/Dialect/openshmemops.mlir" ]]; then
    total_tests=$((total_tests + 1))
    if run_test "${TEST_DIR}/Dialect/openshmemops.mlir"; then
        passed_tests=$((passed_tests + 1))
        
        # Test lowering
        test_lowering "${TEST_DIR}/Dialect/openshmemops.mlir"
        lowering_result=$?
        if [[ $lowering_result -eq 2 ]]; then
            lowering_not_implemented=1
        elif [[ $lowering_result -ne 0 ]]; then
            failed_tests=$((failed_tests + 1))
        fi
    else
        failed_tests=$((failed_tests + 1))
    fi
else
    echo -e "${YELLOW}Warning: ${TEST_DIR}/Dialect/openshmemops.mlir not found${NC}"
fi

# Test any other MLIR files in the test directory
echo -e "${BLUE}Looking for additional test files...${NC}"
additional_files_found=0

for test_file in "${TEST_DIR}"/**/*.mlir; do
    if [[ -f "${test_file}" ]] && [[ "${test_file}" != "${TEST_DIR}/Dialect/openshmemops.mlir" ]]; then
        echo "Found: ${test_file}"
        additional_files_found=$((additional_files_found + 1))
        total_tests=$((total_tests + 1))
        
        if run_test "${test_file}"; then
            passed_tests=$((passed_tests + 1))
        else
            failed_tests=$((failed_tests + 1))
        fi
    fi
done

if [[ ${additional_files_found} -eq 0 ]]; then
    echo "No additional test files found"
fi

echo ""
echo -e "${BLUE}Test Summary${NC}"
echo "Total tests: ${total_tests}"
echo -e "Passed: ${GREEN}${passed_tests}${NC}"
if [[ ${failed_tests} -gt 0 ]]; then
    echo -e "Failed: ${RED}${failed_tests}${NC}"
fi

if [[ ${lowering_not_implemented} -eq 1 ]]; then
    echo -e "Note: ${YELLOW}Lowering pass not yet implemented${NC}"
fi

echo ""

# Exit with appropriate code
if [[ ${failed_tests} -gt 0 ]]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
elif [[ ${passed_tests} -eq 0 ]]; then
    echo -e "${RED}No tests were run!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi