// RUN: openshmem-opt %s --convert-openshmem-to-llvm | FileCheck %s

// Test OpenSHMEM memory management operations lowering to LLVM

module {
  // CHECK-LABEL: @test_malloc_free
  func.func @test_malloc_free() {
    openshmem.region {
      %size = arith.constant 1024 : index
      
      // CHECK: %[[SIZE:.*]] = arith.constant 1024
      // CHECK: call @shmem_malloc(%[[SIZE]]) : (index) -> !llvm.ptr
      %ptr = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      
      // CHECK: call @shmem_free(%{{.*}}) : (!llvm.ptr) -> ()
      openshmem.free(%ptr) : memref<i32, #openshmem.symmetric_memory>
    }
    return
  }

  // CHECK-LABEL: @test_symmetric_allocation
  func.func @test_symmetric_allocation() {
    openshmem.region {
      %size = arith.constant 64 : index
      
      // Allocate multiple symmetric memory regions
      %ptr1 = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      %ptr2 = openshmem.malloc(%size) : index -> memref<i64, #openshmem.symmetric_memory>
      
      // Free in reverse order
      openshmem.free(%ptr2) : memref<i64, #openshmem.symmetric_memory>
      openshmem.free(%ptr1) : memref<i32, #openshmem.symmetric_memory>
    }
    return
  }
}
