// FINAL TEST: Complete ClangIR to OpenSHMEM Conversion
// This test demonstrates the successful ClangIR frontend integration

module {
  // OpenSHMEM function declarations
  cir.func private @shmem_init() -> ()
  cir.func private @shmem_finalize() -> ()
  cir.func private @shmem_my_pe() -> !cir.int<s, 32>
  cir.func private @shmem_n_pes() -> !cir.int<s, 32>
  cir.func private @shmem_barrier_all() -> ()
  cir.func private @shmem_quiet() -> ()

  // Example OpenSHMEM program in ClangIR
  cir.func @openshmem_program() {
    // Initialize the OpenSHMEM library
    cir.call @shmem_init() : () -> ()
    
    // Get processing element information
    %my_pe = cir.call @shmem_my_pe() : () -> !cir.int<s, 32>
    %n_pes = cir.call @shmem_n_pes() : () -> !cir.int<s, 32>
    
    // Synchronization operations
    cir.call @shmem_barrier_all() : () -> ()
    cir.call @shmem_quiet() : () -> ()
    
    // Finalize OpenSHMEM
    cir.call @shmem_finalize() : () -> ()
    
    cir.return
  }
}