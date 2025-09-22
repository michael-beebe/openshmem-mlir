set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(OpenSHMEMConvertPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIROpenSHMEMConvertPassesIncGen)

