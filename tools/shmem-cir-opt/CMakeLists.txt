get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(shmem-cir-opt
  shmem-cir-opt.cpp
)

llvm_update_compile_flags(shmem-cir-opt)

target_link_libraries(shmem-cir-opt
  PRIVATE
  # ClangIR libraries
  clangCIR
  clangCIRLoweringDirectToLLVM
  MLIRCIR
  MLIRCIRTransforms
  
  # OpenSHMEM libraries
  MLIROpenSHMEMDialect
  MLIROpenSHMEMToLLVM
  MLIROpenSHMEMCIRPasses
  MLIROpenSHMEMCIRRewriters
  
  # MLIR core libraries
  ${dialect_libs}
  ${conversion_libs}
  MLIRAnalysis
  MLIRDialect
  MLIRIR
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRTransformUtils
  MLIRReconcileUnrealizedCasts
)

# GCC warning suppression (same as cir-opt)
if (LLVM_COMPILER_IS_GCC_COMPATIBLE AND (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
  target_compile_options(shmem-cir-opt PRIVATE -Wno-overloaded-virtual)
endif()
