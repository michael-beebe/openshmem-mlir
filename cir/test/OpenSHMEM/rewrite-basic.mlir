// RUN: openshmem-opt %s --openshmem-recognition | FileCheck %s --check-prefix=RECOGNITION
// RUN: openshmem-opt %s --convert-cir-to-openshmem | FileCheck %s --check-prefix=CONVERT

// This test verifies OpenSHMEM API recognition and conversion to OpenSHMEM dialect.

module {
  func.func @test_basic_openshmem() {
    // RECOGNITION: func.call @shmem_init() {openshmem.api_call = "shmem_init", openshmem.category = {{.*}}} : () -> ()
    // CONVERT: openshmem.init
    func.call @shmem_init() : () -> ()
    
    // RECOGNITION: func.call @shmem_my_pe() {openshmem.api_call = "shmem_my_pe", openshmem.category = {{.*}}} : () -> i32
    // CONVERT: %{{.*}} = openshmem.my_pe : () -> i32
    %pe = func.call @shmem_my_pe() : () -> i32
    
    // RECOGNITION: func.call @shmem_n_pes() {openshmem.api_call = "shmem_n_pes", openshmem.category = {{.*}}} : () -> i32
    // CONVERT: %{{.*}} = openshmem.n_pes : () -> i32
    %npes = func.call @shmem_n_pes() : () -> i32
    
    // RECOGNITION: func.call @shmem_barrier_all() {openshmem.api_call = "shmem_barrier_all", openshmem.category = {{.*}}} : () -> ()
    // CONVERT: openshmem.barrier_all
    func.call @shmem_barrier_all() : () -> ()
    
    // RECOGNITION: func.call @shmem_finalize() {openshmem.api_call = "shmem_finalize", openshmem.category = {{.*}}} : () -> ()
    // CONVERT: openshmem.finalize
    func.call @shmem_finalize() : () -> ()
    
    return
  }
  
  func.func @test_memory_ops() {
    %size = arith.constant 1024 : index
    
    // RECOGNITION: func.call @shmem_malloc(%{{.*}}) {openshmem.api_call = "shmem_malloc", openshmem.category = {{.*}}} : (index) -> memref<?xi8, #openshmem.symmetric_memory>
    // CONVERT: %{{.*}} = openshmem.malloc %{{.*}} : (index) -> memref<?xi8, #openshmem.symmetric_memory>
    %ptr = func.call @shmem_malloc(%size) : (index) -> memref<?xi8, #openshmem.symmetric_memory>
    
    // RECOGNITION: func.call @shmem_free(%{{.*}}) {openshmem.api_call = "shmem_free", openshmem.category = {{.*}}} : (memref<?xi8, #openshmem.symmetric_memory>) -> ()
    // CONVERT: openshmem.free %{{.*}} : memref<?xi8, #openshmem.symmetric_memory>
    func.call @shmem_free(%ptr) : (memref<?xi8, #openshmem.symmetric_memory>) -> ()
    
    return
  }
  
  func.func @non_openshmem_call() {
    // RECOGNITION-NOT: openshmem.api_call
    // CONVERT: func.call @regular_function() : () -> ()
    func.call @regular_function() : () -> ()
    return
  }
  
  func.func private @shmem_init() -> ()
  func.func private @shmem_my_pe() -> i32
  func.func private @shmem_n_pes() -> i32
  func.func private @shmem_barrier_all() -> ()
  func.func private @shmem_finalize() -> ()
  func.func private @shmem_malloc(index) -> memref<?xi8, #openshmem.symmetric_memory>
  func.func private @shmem_free(memref<?xi8, #openshmem.symmetric_memory>) -> ()
  func.func private @regular_function() -> ()
}
