set(LLVM_TARGET_DEFINITIONS OpenSHMEMCIR/Passes.td)
mlir_tablegen(OpenSHMEMCIR/Passes.h.inc -gen-pass-decls -name OpenSHMEMCIR)
mlir_tablegen(OpenSHMEMCIR/Passes.capi.h.inc -gen-pass-capi-header --prefix OpenSHMEMCIR)
mlir_tablegen(OpenSHMEMCIR/Passes.capi.cpp.inc -gen-pass-capi-impl --prefix OpenSHMEMCIR)
add_public_tablegen_target(MLIROpenSHMEMCIRPassIncGen)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/OpenSHMEMCIR
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.def"
)