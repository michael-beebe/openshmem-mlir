// RUN: openshmem-opt %s --convert-cir-to-openshmem --convert-openshmem-to-llvm | FileCheck %s
// Test complete lowering pipeline: ClangIR -> OpenSHMEM -> LLVM IR

!s32i = !cir.int<s, 32>
!s64i = !cir.int<s, 64>
!void = !cir.void

module {
  // Function declarations for all supported OpenSHMEM operations
  cir.func private @shmem_init()
  cir.func private @shmem_finalize()
  cir.func private @shmem_my_pe() -> !s32i
  cir.func private @shmem_n_pes() -> !s32i
  cir.func private @shmem_malloc(!s64i) -> !cir.ptr<!void>
  cir.func private @shmem_free(!cir.ptr<!void>)
  cir.func private @shmem_barrier_all()
  cir.func private @shmem_quiet()

  // Test function demonstrating complete ClangIR to LLVM lowering
  cir.func @openshmem_end_to_end_test() {
    // Setup phase
    cir.call @shmem_init() : () -> ()
    
    // Query PE information
    %my_pe = cir.call @shmem_my_pe() : () -> !s32i
    %n_pes = cir.call @shmem_n_pes() : () -> !s32i
    
    // Memory management
    %size = cir.const #cir.int<1024> : !s64i
    %ptr = cir.call @shmem_malloc(%size) : (!s64i) -> !cir.ptr<!void>
    
    // Synchronization
    cir.call @shmem_barrier_all() : () -> ()
    cir.call @shmem_quiet() : () -> ()
    
    // Cleanup
    cir.call @shmem_free(%ptr) : (!cir.ptr<!void>) -> ()
    cir.call @shmem_finalize() : () -> ()
    
    cir.return
  }
}