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
  "${CMAKE_CURRENT_SOURCE_DIR}/cir/include"
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  "${CMAKE_BINARY_DIR}/include"
  "${CMAKE_BINARY_DIR}/cir/include"
)

set(MLIR_TABLEGEN_FLAGS "${MLIR_TABLEGEN_FLAGS} \
  -I${CMAKE_CURRENT_SOURCE_DIR}/include \
  -I${MLIR_MAIN_INCLUDE_DIR} -I${MLIR_INCLUDE_DIRS} \
  -I${LLVM_MAIN_INCLUDE_DIR} -I${LLVM_INCLUDE_DIRS}")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(cir)
add_subdirectory(tools)

