add_mlir_dialect(OpenSHMEM openshmem)
add_mlir_doc(OpenSHMEMDialect OpenSHMEM Dialects/ -gen-dialect-doc)

# Ops
set(LLVM_TARGET_DEFINITIONS OpenSHMEMOps.td)
mlir_tablegen(OpenSHMEMOps.h.inc -gen-op-decls)
mlir_tablegen(OpenSHMEMOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIROpenSHMEMOpsIncGen)

# Types
set(LLVM_TARGET_DEFINITIONS OpenSHMEMTypes.td)
mlir_tablegen(OpenSHMEMTypesGen.h.inc -gen-typedef-decls)
mlir_tablegen(OpenSHMEMTypesGen.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIROpenSHMEMTypesIncGen)

# Enums and Attrs
set(LLVM_TARGET_DEFINITIONS OpenSHMEM.td)
mlir_tablegen(OpenSHMEMEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpenSHMEMEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(OpenSHMEMAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(OpenSHMEMAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIROpenSHMEMAttrsIncGen)

