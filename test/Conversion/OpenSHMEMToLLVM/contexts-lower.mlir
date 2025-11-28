// RUN: openshmem-opt %s --convert-openshmem-to-llvm | FileCheck %s

// Test OpenSHMEM context operations lowering to LLVM

module {
  // CHECK-LABEL: @test_ctx_create
  func.func @test_ctx_create() -> !openshmem.ctx {
    %options = arith.constant 0 : i64
    
    // CHECK: call @shmem_ctx_create(%{{.*}}) : (i64) -> !openshmem.ctx
    %ctx, %status = openshmem.ctx_create(%options) : i64 -> !openshmem.ctx, i32
    
    func.return %ctx : !openshmem.ctx
  }

  // CHECK-LABEL: @test_ctx_destroy
  func.func @test_ctx_destroy(%ctx: !openshmem.ctx) {
    // CHECK: call @shmem_ctx_destroy(%{{.*}})
    openshmem.ctx_destroy(%ctx) : !openshmem.ctx
    return
  }

  // CHECK-LABEL: @test_ctx_get_team
  func.func @test_ctx_get_team(%ctx: !openshmem.ctx) -> !openshmem.team {
    // CHECK: call @shmem_ctx_get_team(%{{.*}}) : (!openshmem.ctx) -> !openshmem.team, i32
    %team, %status = openshmem.ctx_get_team(%ctx) : !openshmem.ctx -> !openshmem.team, i32
    func.return %team : !openshmem.team
  }

  // CHECK-LABEL: @test_team_create_ctx
  func.func @test_team_create_ctx(%team: !openshmem.team) -> !openshmem.ctx {
    %options = arith.constant 0 : i64
    
    // CHECK: call @shmem_team_create_ctx(%{{.*}}, %{{.*}}) : (!openshmem.team, i64) -> !openshmem.ctx
    %ctx, %status = openshmem.team_create_ctx(%team, %options) : !openshmem.team, i64 -> !openshmem.ctx, i32
    
    func.return %ctx : !openshmem.ctx
  }

  // CHECK-LABEL: @test_ctx_lifecycle
  func.func @test_ctx_lifecycle() {
    %options = arith.constant 0 : i64
    %ctx, %status = openshmem.ctx_create(%options) : i64 -> !openshmem.ctx, i32
    
    openshmem.region {
      // Context can be used for communication operations
      %size = arith.constant 64 : index
      %ptr = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      openshmem.free(%ptr) : memref<i32, #openshmem.symmetric_memory>
      openshmem.yield
    }
    
    openshmem.ctx_destroy(%ctx) : !openshmem.ctx
    return
  }
}
