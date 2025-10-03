#!/bin/bash

# Test script for OpenSHMEM MLIR parsing/verification and lowering to LLVM

shopt -s globstar

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
TEST_DIR="${PROJECT_ROOT}/test"
OPENSHMEM_OPT="${BUILD_DIR}/tools/openshmem-opt/openshmem-opt"

echo "OpenSHMEM MLIR -> LLVM lowering tests"

if [[ ! -f "${OPENSHMEM_OPT}" ]]; then
  echo "Error: openshmem-opt not found at ${OPENSHMEM_OPT}"
  exit 1
fi

total_tests=0
passed_tests=0
failed_tests=0

# Consider both Dialect and Conversion test trees
for test_file in "${TEST_DIR}"/**/*.mlir; do
  if [[ ! -f "${test_file}" ]]; then
    continue
  fi
  total_tests=$((total_tests+1))
  name=$(basename "${test_file}")
  echo
  echo "=== Running test: ${name} ==="

  # Basic parse/print
  if ! "${OPENSHMEM_OPT}" "${test_file}" > /dev/null 2>&1; then
  echo "  ✗ parse/print failed"
  failed_tests=$((failed_tests+1))
  continue
  fi

  # Verify
  if ! "${OPENSHMEM_OPT}" "${test_file}" --verify-each > /dev/null 2>&1; then
  echo "  ✗ verification failed"
  failed_tests=$((failed_tests+1))
  continue
  fi

  # Lowering (if pass available)
  if "${OPENSHMEM_OPT}" --help | grep -q "convert-openshmem-to-llvm"; then
    echo "  - Running convert-openshmem-to-llvm"
    if ! "${OPENSHMEM_OPT}" "${test_file}" --convert-openshmem-to-llvm > /dev/null 2>&1; then
    echo "  ✗ lowering failed"
    failed_tests=$((failed_tests+1))
    continue
    fi
  else
    echo "  ⚠ convert-openshmem-to-llvm not available; skipping lowering"
  fi

  echo "  ✓ ok"
  passed_tests=$((passed_tests+1))
done

if [[ ${total_tests} -eq 0 ]]; then
  echo "No OpenSHMEM tests found under ${TEST_DIR}"
  exit 2
fi

echo
echo "OpenSHMEM lowering test summary:"
echo "  Total: ${total_tests}"
echo "  Passed: ${passed_tests}"
if [[ ${failed_tests} -gt 0 ]]; then
  echo "  Failed: ${failed_tests}"
  exit 1
fi

exit 0
