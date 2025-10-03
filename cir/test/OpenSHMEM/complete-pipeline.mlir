// Test complete lowering pipeline: ClangIR -> OpenSHMEM -> LLVM IR

!s32i = !cir.int<s, 32>

module {
  // Test function demonstrating ClangIR to LLVM lowering
  cir.func @openshmem_complete_pipeline() {
    // Setup phase - will convert to openshmem.init
    cir.call @shmem_init() : () -> ()
    
    // Query PE information - will convert to openshmem.my_pe/n_pes
    %my_pe = cir.call @shmem_my_pe() : () -> !s32i
    %n_pes = cir.call @shmem_n_pes() : () -> !s32i
    
    // Synchronization operations - will convert to openshmem.barrier_all/quiet
    cir.call @shmem_barrier_all() : () -> ()
    cir.call @shmem_quiet() : () -> ()
    
    // Cleanup - will convert to openshmem.finalize
    cir.call @shmem_finalize() : () -> ()
    
    cir.return
  }
  
  // Function declarations (these get removed during conversion)
  cir.func private @shmem_init()
  cir.func private @shmem_finalize()
  cir.func private @shmem_my_pe() -> !s32i
  cir.func private @shmem_n_pes() -> !s32i
  cir.func private @shmem_barrier_all()
  cir.func private @shmem_quiet()
}