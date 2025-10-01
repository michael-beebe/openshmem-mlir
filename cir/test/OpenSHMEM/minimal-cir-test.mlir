// ClangIR OpenSHMEM test using proper CIR types

module {
  // Function declarations
  cir.func private @shmem_init() -> ()
  cir.func private @shmem_barrier_all() -> ()
  cir.func private @shmem_quiet() -> ()
  cir.func private @shmem_finalize() -> ()

  cir.func @shmem_example() {
    // shmem_init() call
    cir.call @shmem_init() : () -> ()
    
    // shmem_barrier_all() call
    cir.call @shmem_barrier_all() : () -> ()
    
    // shmem_quiet() call
    cir.call @shmem_quiet() : () -> ()
    
    // shmem_finalize() call
    cir.call @shmem_finalize() : () -> ()
    
    cir.return
  }
}