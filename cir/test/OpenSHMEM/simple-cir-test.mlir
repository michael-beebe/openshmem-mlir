// Simple ClangIR OpenSHMEM test using CIR types

module {
  // Function declarations
  cir.func private @shmem_init() -> ()
  cir.func private @shmem_my_pe() -> !cir.int<s, 32>
  cir.func private @shmem_n_pes() -> !cir.int<s, 32>
  cir.func private @shmem_barrier_all() -> ()
  cir.func private @shmem_quiet() -> ()
  cir.func private @shmem_finalize() -> ()

  cir.func @shmem_example() {
    // shmem_init() call
    cir.call @shmem_init() : () -> ()
    
    // shmem_my_pe() call
    %pe = cir.call @shmem_my_pe() : () -> !cir.int<s, 32>
    
    // shmem_n_pes() call  
    %npes = cir.call @shmem_n_pes() : () -> !cir.int<s, 32>
    
    // shmem_barrier_all() call
    cir.call @shmem_barrier_all() : () -> ()
    
    // shmem_quiet() call
    cir.call @shmem_quiet() : () -> ()
    
    // shmem_finalize() call
    cir.call @shmem_finalize() : () -> ()
    
    cir.return
  }
}