// RUN: openshmem-opt %s --convert-cir-to-openshmem | FileCheck %s

// This test verifies conversion from CIR-style function calls to OpenSHMEM dialect operations.

module {
  func.func @test_openshmem_conversion() {
    // CHECK: openshmem.init
    func.call @shmem_init() : () -> ()
    
    // CHECK: %[[PE:.*]] = openshmem.my_pe : () -> i32
    %pe = func.call @shmem_my_pe() : () -> i32
    
    // CHECK: %[[NPES:.*]] = openshmem.n_pes : () -> i32  
    %npes = func.call @shmem_n_pes() : () -> i32
    
    %size = arith.constant 1024 : index
    // CHECK: %[[PTR:.*]] = openshmem.malloc %{{.*}} : (index) -> memref<?xi8, #openshmem.symmetric_memory>
    %ptr = func.call @shmem_malloc(%size) : (index) -> !llvm.ptr
    
    // CHECK: openshmem.barrier_all
    func.call @shmem_barrier_all() : () -> ()
    
    // CHECK: openshmem.free %{{.*}} : memref<?xi8, #openshmem.symmetric_memory>
    func.call @shmem_free(%ptr) : (!llvm.ptr) -> ()
    
    // CHECK: openshmem.finalize
    func.call @shmem_finalize() : () -> ()
    
    return
  }
  
  func.func @test_rma_operations() {
    %size = arith.constant 64 : index
    %pe = arith.constant 1 : i32
    
    %src = func.call @shmem_malloc(%size) : (index) -> !llvm.ptr  
    %dest = func.call @shmem_malloc(%size) : (index) -> !llvm.ptr
    
    // CHECK: openshmem.put %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<?xi8, #openshmem.symmetric_memory>, memref<?xi8, #openshmem.symmetric_memory>, index, i32
    func.call @shmem_put(%dest, %src, %size, %pe) : (!llvm.ptr, !llvm.ptr, index, i32) -> ()
    
    // CHECK: openshmem.get %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : memref<?xi8, #openshmem.symmetric_memory>, memref<?xi8, #openshmem.symmetric_memory>, index, i32  
    func.call @shmem_get(%src, %dest, %size, %pe) : (!llvm.ptr, !llvm.ptr, index, i32) -> ()
    
    func.call @shmem_free(%src) : (!llvm.ptr) -> ()
    func.call @shmem_free(%dest) : (!llvm.ptr) -> ()
    
    return
  }
  
  // Function declarations (these would come from ClangIR)
  func.func private @shmem_init() -> ()
  func.func private @shmem_my_pe() -> i32
  func.func private @shmem_n_pes() -> i32
  func.func private @shmem_malloc(index) -> !llvm.ptr
  func.func private @shmem_free(!llvm.ptr) -> ()
  func.func private @shmem_barrier_all() -> ()
  func.func private @shmem_finalize() -> ()
  func.func private @shmem_put(!llvm.ptr, !llvm.ptr, index, i32) -> ()
  func.func private @shmem_get(!llvm.ptr, !llvm.ptr, index, i32) -> ()
}