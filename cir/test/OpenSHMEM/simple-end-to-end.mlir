// Test complete lowering pipeline: ClangIR -> OpenSHMEM -> LLVM IR
// This version focuses on operations that convert cleanly to LLVM

!s32i = !cir.int<s, 32>

module {
  // Function declarations for cleanly converting OpenSHMEM operations
  cir.func private @shmem_init()
  cir.func private @shmem_finalize()
  cir.func private @shmem_my_pe() -> !s32i
  cir.func private @shmem_n_pes() -> !s32i
  cir.func private @shmem_barrier_all()
  cir.func private @shmem_quiet()

  // Test function demonstrating ClangIR to LLVM lowering (without malloc/free)
  cir.func @openshmem_simple_end_to_end() {
    // Setup phase
    cir.call @shmem_init() : () -> ()
    
    // Query PE information
    %my_pe = cir.call @shmem_my_pe() : () -> !s32i
    %n_pes = cir.call @shmem_n_pes() : () -> !s32i
    
    // Synchronization operations
    cir.call @shmem_barrier_all() : () -> ()
    cir.call @shmem_quiet() : () -> ()
    
    // Cleanup
    cir.call @shmem_finalize() : () -> ()
    
    cir.return
  }
}