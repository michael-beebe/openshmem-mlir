// RUN: openshmem-opt %s --openshmem-recognition --convert-cir-to-openshmem | FileCheck %s --check-prefix=CONVERT

// Test conversion of context-related CIR calls to OpenSHMEM dialect ops

module {
  func.func @test_contexts_conversion() {
    %options = arith.constant 0 : i64

    // CONVERT: [[voptions:%.*]] = arith.constant 0 : i64
    // CONVERT: [[vctx:%.*]], [[vstatus:%.*]] = openshmem.ctx_create([[voptions]]) : i64 -> !openshmem.ctx, i32
    %ctx, %status = func.call @shmem_ctx_create(%options) : (i64) -> (!openshmem.ctx, i32)

    // CONVERT: [[vteam_ctx:%.*]], [[vstatus2:%.*]] = openshmem.team_create_ctx([[vworld_team]], [[voptions]]) : !openshmem.team, i64 -> !openshmem.ctx, i32
    %world_team = func.call @shmem_team_world() : () -> !openshmem.team
    %team_ctx, %status2 = func.call @shmem_team_create_ctx(%world_team, %options) : (!openshmem.team, i64) -> (!openshmem.ctx, i32)

    // CONVERT: [[vctx_team:%.*]], [[vstatus3:%.*]] = openshmem.ctx_get_team([[vctx]]) : !openshmem.ctx -> !openshmem.team, i32
    %ctx_team, %status3 = func.call @shmem_ctx_get_team(%ctx) : (!openshmem.ctx) -> (!openshmem.team, i32)

    // CONVERT: openshmem.ctx_destroy([[vctx]]) : !openshmem.ctx
    func.call @shmem_ctx_destroy(%ctx) : (!openshmem.ctx) -> ()

    // CONVERT: openshmem.ctx_destroy([[vteam_ctx]]) : !openshmem.ctx
    func.call @shmem_ctx_destroy(%team_ctx) : (!openshmem.ctx) -> ()

    return
  }

  // Private CIR functions representing the original APIs
  func.func private @shmem_ctx_create(i64) -> (!openshmem.ctx, i32)
  func.func private @shmem_team_create_ctx(!openshmem.team, i64) -> (!openshmem.ctx, i32)
  func.func private @shmem_ctx_get_team(!openshmem.ctx) -> (!openshmem.team, i32)
  func.func private @shmem_ctx_destroy(!openshmem.ctx) -> ()
  func.func private @shmem_team_world() -> !openshmem.team
}
