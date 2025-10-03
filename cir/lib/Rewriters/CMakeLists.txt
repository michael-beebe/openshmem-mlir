add_mlir_library(MLIROpenSHMEMCIRRewriters
  RewriteOpenSHMEMCalls.cpp
  RewriteSetup.cpp
  RewriteMemory.cpp
  RewriteSync.cpp
  RewriteTeams.cpp
  RewriteContexts.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/cir/include/OpenSHMEMCIR

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIROpenSHMEMDialect
  MLIRCIR
)