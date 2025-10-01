// Simple ClangIR OpenSHMEM test using standard types

module {
  cir.func @shmem_example() {
    // shmem_init() call
    cir.call @shmem_init() : () -> ()
    
    // shmem_my_pe() call
    %pe = cir.call @shmem_my_pe() : () -> i32
    
    // shmem_n_pes() call  
    %npes = cir.call @shmem_n_pes() : () -> i32
    
    // shmem_barrier_all() call
    cir.call @shmem_barrier_all() : () -> ()
    
    // shmem_quiet() call
    cir.call @shmem_quiet() : () -> ()
    
    // shmem_finalize() call
    cir.call @shmem_finalize() : () -> ()
    
    cir.return
  }
}