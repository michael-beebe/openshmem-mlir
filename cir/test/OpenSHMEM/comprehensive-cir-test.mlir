// Comprehensive ClangIR OpenSHMEM test with all function types

module {
  // Function declarations
  cir.func private @shmem_init() -> ()
  cir.func private @shmem_finalize() -> ()
  cir.func private @shmem_my_pe() -> !cir.int<s, 32>
  cir.func private @shmem_n_pes() -> !cir.int<s, 32>
  cir.func private @shmem_malloc(!cir.int<s, 64>) -> !cir.ptr<!cir.void>
  cir.func private @shmem_free(!cir.ptr<!cir.void>) -> ()
  cir.func private @shmem_barrier_all() -> ()
  cir.func private @shmem_quiet() -> ()

  cir.func @comprehensive_shmem_test() {
    // Initialize OpenSHMEM
    cir.call @shmem_init() : () -> ()
    
    // Get PE info
    %pe = cir.call @shmem_my_pe() : () -> !cir.int<s, 32>
    %npes = cir.call @shmem_n_pes() : () -> !cir.int<s, 32>
    
    // Allocate symmetric memory
    %size = cir.const #cir.int<1024> : !cir.int<s, 64>
    %ptr = cir.call @shmem_malloc(%size) : (!cir.int<s, 64>) -> !cir.ptr<!cir.void>
    
    // Synchronization
    cir.call @shmem_barrier_all() : () -> ()
    cir.call @shmem_quiet() : () -> ()
    
    // Clean up
    cir.call @shmem_free(%ptr) : (!cir.ptr<!cir.void>) -> ()
    cir.call @shmem_finalize() : () -> ()
    
    cir.return
  }
}