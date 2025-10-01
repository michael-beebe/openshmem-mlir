add_mlir_library(MLIROpenSHMEMCIRPasses
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/cir/include/OpenSHMEMCIR
  ${PROJECT_BINARY_DIR}/cir/include/OpenSHMEMCIR

  DEPENDS
  MLIROpenSHMEMCIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIROpenSHMEMDialect
)

add_subdirectory(Rewriters)
