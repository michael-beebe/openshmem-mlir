// Test ClangIR OpenSHMEM conversion
// This demonstrates cir.call operations being converted to openshmem dialect operations

module {
  // Function declarations
  cir.func private @shmem_init() -> ()
  cir.func private @shmem_my_pe() -> !cir.int<s, 32>
  cir.func private @shmem_n_pes() -> !cir.int<s, 32>
  cir.func private @shmem_malloc(!cir.int<s, 64>) -> !cir.ptr<!cir.void>
  cir.func private @shmem_free(!cir.ptr<!cir.void>) -> ()
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
    
    // shmem_malloc() call
    %size = cir.const #cir.int<1024> : !cir.int<s, 64>
    %ptr = cir.call @shmem_malloc(%size) : (!cir.int<s, 64>) -> !cir.ptr<!cir.void>
    
    // shmem_free() call
    cir.call @shmem_free(%ptr) : (!cir.ptr<!cir.void>) -> ()
    
    // shmem_barrier_all() call
    cir.call @shmem_barrier_all() : () -> ()
    
    // shmem_quiet() call  
    cir.call @shmem_quiet() : () -> ()
    
    // shmem_finalize() call
    cir.call @shmem_finalize() : () -> ()
    
    cir.return
  }
}