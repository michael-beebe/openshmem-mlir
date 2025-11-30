include(AddMLIRPython)

# Place generated modules under the standard `mlir` package namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir.")

declare_mlir_python_sources(OpenSHMEMPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT OpenSHMEMPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE "${PROJECT_SOURCE_DIR}/include/mlir/Dialect/OpenSHMEM/IR/OpenSHMEM.td"
  SOURCES
    dialects/openshmem.py
  DIALECT_NAME openshmem)

add_mlir_python_common_capi_library(OpenSHMEMPythonCAPI
  INSTALL_COMPONENT OpenSHMEMPythonModules
  INSTALL_DESTINATION python_packages/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_packages/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES
    OpenSHMEMPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core)

add_mlir_python_modules(OpenSHMEMPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/mlir"
  INSTALL_PREFIX "python_packages/mlir"
  DECLARED_SOURCES
    OpenSHMEMPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    OpenSHMEMPythonCAPI)
