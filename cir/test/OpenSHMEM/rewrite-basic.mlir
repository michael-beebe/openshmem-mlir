// RUN: openshmem-opt %s --openshmem-recognition | FileCheck %s

// This test verifies basic OpenSHMEM API recognition and annotation.

module {
  func.func @test_basic_openshmem() {
    // CHECK: func.call @shmem_init() {openshmem.api_call = "shmem_init", openshmem.category = {{.*}}} : () -> ()
    func.call @shmem_init() : () -> ()
    
    // CHECK: func.call @shmem_my_pe() {openshmem.api_call = "shmem_my_pe", openshmem.category = {{.*}}} : () -> i32
    %pe = func.call @shmem_my_pe() : () -> i32
    
    // CHECK: func.call @shmem_n_pes() {openshmem.api_call = "shmem_n_pes", openshmem.category = {{.*}}} : () -> i32
    %npes = func.call @shmem_n_pes() : () -> i32
    
    // CHECK: func.call @shmem_barrier_all() {openshmem.api_call = "shmem_barrier_all", openshmem.category = {{.*}}} : () -> ()
    func.call @shmem_barrier_all() : () -> ()
    
    // CHECK: func.call @shmem_finalize() {openshmem.api_call = "shmem_finalize", openshmem.category = {{.*}}} : () -> ()
    func.call @shmem_finalize() : () -> ()
    
    return
  }
  
  func.func @test_memory_ops() {
    %size = arith.constant 1024 : i64
    
    // CHECK: func.call @shmem_malloc(%{{.*}}) {openshmem.api_call = "shmem_malloc", openshmem.category = {{.*}}} : (i64) -> !llvm.ptr
    %ptr = func.call @shmem_malloc(%size) : (i64) -> !llvm.ptr
    
    // CHECK: func.call @shmem_free(%{{.*}}) {openshmem.api_call = "shmem_free", openshmem.category = {{.*}}} : (!llvm.ptr) -> ()
    func.call @shmem_free(%ptr) : (!llvm.ptr) -> ()
    
    return
  }
  
  func.func @non_openshmem_call() {
    // CHECK-NOT: openshmem.api_call
    func.call @regular_function() : () -> ()
    return
  }
  
  func.func private @shmem_init() -> ()
  func.func private @shmem_my_pe() -> i32
  func.func private @shmem_n_pes() -> i32
  func.func private @shmem_barrier_all() -> ()
  func.func private @shmem_finalize() -> ()
  func.func private @shmem_malloc(i64) -> !llvm.ptr
  func.func private @shmem_free(!llvm.ptr) -> ()
  func.func private @regular_function() -> ()
}
