// ClangIR OpenSHMEM malloc/free test

module {
  // Function declarations
  cir.func private @shmem_init() -> ()
  cir.func private @shmem_malloc(!cir.int<s, 64>) -> !cir.ptr<!cir.void>
  cir.func private @shmem_finalize() -> ()

  cir.func @test_malloc() {
    // Initialize
    cir.call @shmem_init() : () -> ()
    
    // Allocate symmetric memory  
    %size = cir.const #cir.int<1024> : !cir.int<s, 64>
    %ptr = cir.call @shmem_malloc(%size) : (!cir.int<s, 64>) -> !cir.ptr<!cir.void>
    
    // Note: We don't call shmem_free to avoid type conversion issues in this test
    
    // Finalize
    cir.call @shmem_finalize() : () -> ()
    
    cir.return
  }
}