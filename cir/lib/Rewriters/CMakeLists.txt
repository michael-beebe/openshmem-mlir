add_mlir_library(MLIROpenSHMEMCIRRewriters
  RewriteOpenSHMEMCalls.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/cir/include/OpenSHMEMCIR

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIROpenSHMEMDialect
)