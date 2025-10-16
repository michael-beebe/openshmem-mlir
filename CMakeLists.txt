cmake_minimum_required(VERSION 3.20)

project(openshmem-mlir LANGUAGES C CXX)

find_package(MLIR REQUIRED CONFIG)

set(Clang_DIR "${MLIR_CMAKE_DIR}/../clang")
find_package(Clang REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Make headers visible to both C++ and mlir-tblgen, and add MLIR/LLVM include roots.
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  "${CMAKE_BINARY_DIR}/include"
)

# Check if ClangIR (CIR) is available by looking for the clangCIR target
if (TARGET clangCIR)
  set(HAVE_CLANGIR ON)
  message(STATUS "ClangIR detected - enabling CIR support")
else()
  set(HAVE_CLANGIR OFF)
  message(STATUS "ClangIR not detected - CIR support disabled")
endif()

# If ClangIR is available, add CIR include paths
if (HAVE_CLANGIR)
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/cir/include")
  include_directories("${CMAKE_BINARY_DIR}/cir/include")

  set(_clangir_include_candidates)

  if (DEFINED CLANGIR_BUILD_DIR)
    list(APPEND _clangir_include_candidates
      "${CLANGIR_BUILD_DIR}/tools/clang/include"
      "${CLANGIR_BUILD_DIR}/include")
  endif()

  if (DEFINED CLANGIR_SRC_DIR)
    list(APPEND _clangir_include_candidates
      "${CLANGIR_SRC_DIR}/clang/include"
      "${CLANGIR_SRC_DIR}/llvm/include")
  endif()

  if (DEFINED LLVM_PROJECT_BUILD_DIR)
    list(APPEND _clangir_include_candidates
      "${LLVM_PROJECT_BUILD_DIR}/tools/clang/include")
  endif()

  if (DEFINED LLVM_PROJECT_SOURCE_DIR)
    list(APPEND _clangir_include_candidates
      "${LLVM_PROJECT_SOURCE_DIR}/clang/include")
  endif()

  list(REMOVE_DUPLICATES _clangir_include_candidates)

  foreach(dir IN LISTS _clangir_include_candidates)
    if (EXISTS "${dir}")
      include_directories("${dir}")
      message(STATUS "  Added ClangIR include: ${dir}")
    endif()
  endforeach()
endif()

set(MLIR_TABLEGEN_FLAGS "${MLIR_TABLEGEN_FLAGS} \
  -I${CMAKE_CURRENT_SOURCE_DIR}/include \
  -I${MLIR_MAIN_INCLUDE_DIR} -I${MLIR_INCLUDE_DIRS} \
  -I${LLVM_MAIN_INCLUDE_DIR} -I${LLVM_INCLUDE_DIRS}")

add_subdirectory(include)
add_subdirectory(lib)

# Build CIR passes when ClangIR is available (upstream or incubator)
if (HAVE_CLANGIR)
  message(STATUS "Building ClangIR-specific passes (cir/ subdirectory)")
  add_subdirectory(cir)
else()
  message(STATUS "Skipping ClangIR-specific passes (cir/ subdirectory) - ClangIR not available")
endif()

add_subdirectory(tools)

