add_mlir_library(MLIROpenSHMEMTransforms
  Passes.cpp

  DEPENDS
  MLIROpenSHMEMPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIROpenSHMEM
)
