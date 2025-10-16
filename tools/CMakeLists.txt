add_subdirectory(shmem-mlir-opt)

# shmem-cir-opt requires ClangIR incubator and has API compatibility issues
# Skip for now - use clangir's cir-opt directly instead
# TODO: Fix API changes (createCIRFlattenCFGPass, populateCIRToLLVMPasses signature)
# if (CLANGIR_BUILD_DIR)
#   add_subdirectory(shmem-cir-opt)
# endif()



