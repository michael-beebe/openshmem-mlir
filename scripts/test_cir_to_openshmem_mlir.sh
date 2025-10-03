#!/bin/bash

# Test script for CIR -> OpenSHMEM conversion

shopt -s globstar

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
CIR_TEST_DIR="${PROJECT_ROOT}/cir/test/OpenSHMEM"
OPENSHMEM_OPT="${BUILD_DIR}/tools/openshmem-opt/openshmem-opt"

echo "CIR -> OpenSHMEM conversion tests"

if [[ ! -f "${OPENSHMEM_OPT}" ]]; then
  echo "Error: openshmem-opt not found at ${OPENSHMEM_OPT}"
  exit 1
fi

total_tests=0
passed_tests=0
failed_tests=0

for test_file in "${CIR_TEST_DIR}"/*.mlir; do
  if [[ ! -f "${test_file}" ]]; then
    continue
  fi
  total_tests=$((total_tests+1))
  name=$(basename "${test_file}")
  echo
  echo "=== Running CIR conversion test: ${name} ==="

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

  # Conversion (run recognition + conversion)
  echo "  - Converting (recognition + convert-cir-to-openshmem)"
  if ! "${OPENSHMEM_OPT}" "${test_file}" --openshmem-recognition --convert-cir-to-openshmem > /dev/null 2>&1; then
  echo "  ✗ conversion failed"
  failed_tests=$((failed_tests+1))
  continue
  fi

  echo "  ✓ ok"
  passed_tests=$((passed_tests+1))
done
echo
if [[ ${total_tests} -eq 0 ]]; then
  echo "No CIR tests found in ${CIR_TEST_DIR}"
  exit 2
fi

echo "CIR conversion test summary:"
echo "  Total: ${total_tests}"
echo "  Passed: ${passed_tests}"
if [[ ${failed_tests} -gt 0 ]]; then
  echo "  Failed: ${failed_tests}"
  exit 1
fi

exit 0
