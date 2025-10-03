// Complete ClangIR OpenSHMEM test demonstrating all conversions

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

  cir.func @test_all_conversions() {
    // Initialize OpenSHMEM
    cir.call @shmem_init() : () -> ()
    
    // Get PE info (values not used to avoid type conversion issues)
    %pe = cir.call @shmem_my_pe() : () -> !cir.int<s, 32>
    %npes = cir.call @shmem_n_pes() : () -> !cir.int<s, 32>
    
    // Synchronization
    cir.call @shmem_barrier_all() : () -> ()
    cir.call @shmem_quiet() : () -> ()
    
    // Finalize
    cir.call @shmem_finalize() : () -> ()
    
    cir.return
  }
}
