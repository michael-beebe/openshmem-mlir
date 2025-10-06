// RUN: openshmem-opt %s --convert-openshmem-to-llvm | FileCheck %s

// Test OpenSHMEM atomic operations lowering to LLVM

module {
  // CHECK-LABEL: @test_atomic_add
  func.func @test_atomic_add(%pe: i32) {
    openshmem.region {
      %size = arith.constant 4 : index
      %ptr = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      %value = arith.constant 42 : i32
      
      // CHECK: call @shmem_atomic_add(%{{.*}}, %{{.*}}, %{{.*}})
      openshmem.atomic_add(%ptr, %value, %pe) : memref<i32, #openshmem.symmetric_memory>, i32, i32
      
      openshmem.free(%ptr) : memref<i32, #openshmem.symmetric_memory>
    }
    return
  }

  // CHECK-LABEL: @test_atomic_fetch_add
  func.func @test_atomic_fetch_add(%pe: i32) {
    openshmem.region {
      %size = arith.constant 4 : index
      %ptr = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      %value = arith.constant 10 : i32
      
      // CHECK: call @shmem_atomic_fetch_add(%{{.*}}, %{{.*}}, %{{.*}}) : {{.*}} -> i32
      %old_val = openshmem.atomic_fetch_add(%ptr, %value, %pe) : memref<i32, #openshmem.symmetric_memory>, i32, i32 -> i32
      
      openshmem.free(%ptr) : memref<i32, #openshmem.symmetric_memory>
    }
    return
  }

  // CHECK-LABEL: @test_atomic_fetch
  func.func @test_atomic_fetch(%pe: i32) {
    openshmem.region {
      %size = arith.constant 4 : index
      %ptr = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      
      // CHECK: call @shmem_atomic_fetch(%{{.*}}, %{{.*}}) : {{.*}} -> i32
      %val = openshmem.atomic_fetch(%ptr, %pe) : memref<i32, #openshmem.symmetric_memory>, i32 -> i32
      
      openshmem.free(%ptr) : memref<i32, #openshmem.symmetric_memory>
    }
    return
  }

  // CHECK-LABEL: @test_atomic_set
  func.func @test_atomic_set(%pe: i32) {
    openshmem.region {
      %size = arith.constant 4 : index
      %ptr = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      %value = arith.constant 99 : i32
      
      // CHECK: call @shmem_atomic_set(%{{.*}}, %{{.*}}, %{{.*}})
      openshmem.atomic_set(%ptr, %value, %pe) : memref<i32, #openshmem.symmetric_memory>, i32, i32
      
      openshmem.free(%ptr) : memref<i32, #openshmem.symmetric_memory>
    }
    return
  }
}
