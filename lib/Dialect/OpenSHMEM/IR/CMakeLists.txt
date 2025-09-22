add_mlir_library(MLIROpenSHMEM
  OpenSHMEM.cpp
  OpenSHMEMOps.cpp

  DEPENDS
  MLIROpenSHMEMOpsIncGen
  MLIROpenSHMEMTypesIncGen
  MLIROpenSHMEMAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
