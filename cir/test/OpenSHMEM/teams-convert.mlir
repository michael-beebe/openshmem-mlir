// RUN: openshmem-opt %s --openshmem-recognition --convert-cir-to-openshmem | FileCheck %s --check-prefix=CONVERT

// Test conversion of team-related CIR calls to OpenSHMEM dialect ops

module {
  func.func @test_teams_conversion() {
    // Create placeholders for arguments
    %world_team = func.call @shmem_team_world() : () -> !openshmem.team
    %start = arith.constant 0 : i32
    %stride = arith.constant 1 : i32
    %size = arith.constant 4 : i32
    %xrange = arith.constant 2 : i32

    // CONVERT: [[vworld_team:%.*]] = openshmem.team_world -> !openshmem.team
  // The private function returns a team that should be recognized and folded

    // CONVERT: [[vnew_team:%.*]], [[vretval:%.*]] = openshmem.team_split_strided([[vworld_team]], [[vstart]], [[vstride]], [[vsize]]) : !openshmem.team, i32, i32, i32 -> !openshmem.team, i32
    %new_team, %retval = func.call @shmem_team_split_strided(%world_team, %start, %stride, %size) : (!openshmem.team, i32, i32, i32) -> (!openshmem.team, i32)

    // CONVERT: [[vxaxis_team:%.*]], [[vyaxis_team:%.*]], [[vretval2:%.*]] = openshmem.team_split_2d([[vworld_team]], [[vxrange]]) : !openshmem.team, i32 -> !openshmem.team, !openshmem.team, i32
    %xaxis_team, %yaxis_team, %retval2 = func.call @shmem_team_split_2d(%world_team, %xrange) : (!openshmem.team, i32) -> (!openshmem.team, !openshmem.team, i32)

    // CONVERT: openshmem.team_sync([[vnew_team]]) : !openshmem.team
    func.call @shmem_team_sync(%new_team) : (!openshmem.team) -> ()

    // CONVERT: openshmem.team_destroy([[vnew_team]]) : !openshmem.team
    func.call @shmem_team_destroy(%new_team) : (!openshmem.team) -> ()

    // CONVERT: openshmem.team_destroy([[vxaxis_team]]) : !openshmem.team
    func.call @shmem_team_destroy(%xaxis_team) : (!openshmem.team) -> ()

    // CONVERT: openshmem.team_destroy([[vyaxis_team]]) : !openshmem.team
    func.call @shmem_team_destroy(%yaxis_team) : (!openshmem.team) -> ()

    return
  }

  // Private CIR functions representing the original APIs
  func.func private @shmem_team_world() -> !openshmem.team
  func.func private @shmem_team_split_strided(!openshmem.team, i32, i32, i32) -> (!openshmem.team, i32)
  func.func private @shmem_team_split_2d(!openshmem.team, i32) -> (!openshmem.team, !openshmem.team, i32)
  func.func private @shmem_team_sync(!openshmem.team) -> ()
  func.func private @shmem_team_destroy(!openshmem.team) -> ()
}
