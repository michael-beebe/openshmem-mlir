// Test ClangIR OpenSHMEM conversion
// This demonstrates cir.call operations being converted to openshmem dialect operations

module {
  cir.func @shmem_example() {
    // shmem_init() call
    cir.call @shmem_init() : () -> ()
    
    // shmem_my_pe() call
    %pe = cir.call @shmem_my_pe() : () -> !s32i
    
    // shmem_n_pes() call
    %npes = cir.call @shmem_n_pes() : () -> !s32i
    
    // shmem_malloc() call
    %size = cir.const #cir.int<1024> : !s32i
    %ptr = cir.call @shmem_malloc(%size) : (!s32i) -> !cir.ptr<!u8i>
    
    // shmem_free() call
    cir.call @shmem_free(%ptr) : (!cir.ptr<!u8i>) -> ()
    
    // shmem_barrier_all() call
    cir.call @shmem_barrier_all() : () -> ()
    
    // shmem_quiet() call  
    cir.call @shmem_quiet() : () -> ()
    
    // shmem_finalize() call
    cir.call @shmem_finalize() : () -> ()
    
    cir.return
  }
}