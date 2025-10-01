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
      %root_pe = arith.constant 0 : i32
      
      // Broadcast operation
      %broadcast_ret = openshmem.broadcast(%team, %dest, %src, %nelems, %root_pe) : !openshmem.team, memref<i32, #openshmem.symmetric_memory>, memref<i32, #openshmem.symmetric_memory>, index, i32 -> i32
      
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
      
      // Quiet operation
      openshmem.quiet
    }
    return
  }

  // Test team operations
  func.func @test_teams() -> !openshmem.team {
    %my_pe = openshmem.my_pe : i32
    %n_pes = openshmem.n_pes : i32
    %world_team = openshmem.team_world -> !openshmem.team
    %start = arith.constant 0 : i32
    %stride = arith.constant 1 : i32
    
    // Create a team  
    %team, %retval = openshmem.team_split_strided(%world_team, %start, %stride, %n_pes) : !openshmem.team, i32, i32, i32 -> !openshmem.team, i32
    
    openshmem.region {
      // Operations using the team can go here
    }
    
    func.return %team : !openshmem.team
  }

  // Test context operations  
  func.func @test_contexts() -> !openshmem.ctx {
    %options = arith.constant 0 : i64
    %ctx, %status = openshmem.ctx_create(%options) : i64 -> !openshmem.ctx, i32
    
    openshmem.region {
      // Operations using the context can go here
    }
    
    func.return %ctx : !openshmem.ctx
  }
}
