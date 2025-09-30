// Test file for OpenSHMEM dialect operations
// To test conversion: openshmem-opt openshmemtollvm.mlir -convert-openshmem-to-llvm

module {
  // Test basic setup and teardown operations
  func.func @test_setup() {
    openshmem.region {
      openshmem.init
      %my_pe = openshmem.my_pe : i32
      %n_pes = openshmem.n_pes : i32
      openshmem.finalize
    }
    return
  }

  // Test memory management operations
  func.func @test_memory() {
    openshmem.region {
      %size = arith.constant 1024 : index
      
      // Allocate symmetric memory
      %ptr = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      
      // Free symmetric memory
      openshmem.free(%ptr) : memref<i32, #openshmem.symmetric_memory>
    }
    return
  }

  // Test RMA (Remote Memory Access) operations
  func.func @test_rma(%arg0: memref<10xi32>, %size: index, %pe: i32) {
    openshmem.region {
      %symmetric_ptr = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      
      // Test put operations
      openshmem.putmem(%symmetric_ptr, %arg0, %size, %pe) : memref<i32, #openshmem.symmetric_memory>, memref<10xi32>, index, i32
      
      // Test get operations  
      openshmem.getmem(%arg0, %symmetric_ptr, %size, %pe) : memref<10xi32>, memref<i32, #openshmem.symmetric_memory>, index, i32
      
      openshmem.free(%symmetric_ptr) : memref<i32, #openshmem.symmetric_memory>
    }
    return
  }

  // Test collective operations
  func.func @test_collectives(%team: !openshmem.team) {
    openshmem.region {
      %size = arith.constant 64 : index
      %src = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      %dest = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      %nelems = arith.constant 16 : index
      
      // Broadcast operation
      openshmem.broadcast(%dest, %src, %nelems, %team) : memref<i32, #openshmem.symmetric_memory>, memref<i32, #openshmem.symmetric_memory>, index, !openshmem.team
      
      openshmem.free(%src) : memref<i32, #openshmem.symmetric_memory>
      openshmem.free(%dest) : memref<i32, #openshmem.symmetric_memory>
    }
    return
  }

  // Test atomic operations
  func.func @test_atomics(%pe: i32) {
    openshmem.region {
      %size = arith.constant 4 : index
      %ptr = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      %value = arith.constant 42 : i32
      
      // Atomic add
      openshmem.atomic_add(%ptr, %value, %pe) : memref<i32, #openshmem.symmetric_memory>, i32, i32
      
      // Atomic fetch and add
      %old_val = openshmem.atomic_fetch_add(%ptr, %value, %pe) : memref<i32, #openshmem.symmetric_memory>, i32, i32 -> i32
      
      openshmem.free(%ptr) : memref<i32, #openshmem.symmetric_memory>
    }
    return
  }

  // Test synchronization operations
  func.func @test_sync() {
    openshmem.region {
      // Barrier synchronization
      openshmem.barrier_all
      
      // Fence operation
      openshmem.fence
      
      // Quiet operation
      openshmem.quiet
    }
    return
  }

  // Test team operations
  func.func @test_teams() -> !openshmem.team {
    openshmem.region {
      %my_pe = openshmem.my_pe : i32
      %n_pes = openshmem.n_pes : i32
      
      // Create a team (simplified - would normally have more parameters)
      %team = openshmem.team_split_strided(%my_pe, %n_pes) : i32, i32 -> !openshmem.team
      
      openshmem.region : !openshmem.team %team {
        // Operations within the team context
        openshmem.barrier(%team) : !openshmem.team
      }
      
      return %team : !openshmem.team
    }
  }

  // Test context operations  
  func.func @test_contexts() -> !openshmem.ctx {
    openshmem.region {
      // Create a context
      %ctx = openshmem.ctx_create : !openshmem.ctx
      
      openshmem.region : !openshmem.ctx %ctx {
        // Operations within the context
        openshmem.fence(%ctx) : !openshmem.ctx
        openshmem.quiet(%ctx) : !openshmem.ctx
      }
      
      // Destroy the context
      openshmem.ctx_destroy(%ctx) : !openshmem.ctx
      
      return %ctx : !openshmem.ctx
    }
  }
}
