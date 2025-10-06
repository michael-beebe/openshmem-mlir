// RUN: openshmem-opt %s --convert-openshmem-to-llvm | FileCheck %s

// Test OpenSHMEM RMA (Remote Memory Access) operations lowering to LLVM

module {
  // CHECK-LABEL: @test_put_operations
  func.func @test_put_operations(%arg0: memref<10xi32>, %size: index, %pe: i32) {
    openshmem.region {
      %symmetric_ptr = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      
      // CHECK: call @shmem_putmem(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
      openshmem.putmem(%symmetric_ptr, %arg0, %size, %pe) : memref<i32, #openshmem.symmetric_memory>, memref<10xi32>, index, i32
      
      openshmem.free(%symmetric_ptr) : memref<i32, #openshmem.symmetric_memory>
    }
    return
  }

  // CHECK-LABEL: @test_get_operations
  func.func @test_get_operations(%arg0: memref<10xi32>, %size: index, %pe: i32) {
    openshmem.region {
      %symmetric_ptr = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      
      // CHECK: call @shmem_getmem(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
      openshmem.getmem(%arg0, %symmetric_ptr, %size, %pe) : memref<10xi32>, memref<i32, #openshmem.symmetric_memory>, index, i32
      
      openshmem.free(%symmetric_ptr) : memref<i32, #openshmem.symmetric_memory>
    }
    return
  }

  // CHECK-LABEL: @test_rma_combined
  func.func @test_rma_combined(%local: memref<16xi32>, %size: index, %pe: i32) {
    openshmem.region {
      %remote = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      
      // Put data to remote PE
      openshmem.putmem(%remote, %local, %size, %pe) : memref<i32, #openshmem.symmetric_memory>, memref<16xi32>, index, i32
      
      // Get data from remote PE
      openshmem.getmem(%local, %remote, %size, %pe) : memref<16xi32>, memref<i32, #openshmem.symmetric_memory>, index, i32
      
      openshmem.free(%remote) : memref<i32, #openshmem.symmetric_memory>
    }
    return
  }
}
