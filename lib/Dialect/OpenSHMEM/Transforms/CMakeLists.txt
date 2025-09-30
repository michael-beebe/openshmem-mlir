add_mlir_library(MLIROpenSHMEMTransforms
  # TODO: add passes here
  Passes.cpp

  DEPENDS
  MLIROpenSHMEMTransformsIncGen

  # FIXME: I'm not sure about the below
  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRMemRefDialect
  MLIROpenSHMEMDialect
  MLIRPass
  MLIRTransforms
)
