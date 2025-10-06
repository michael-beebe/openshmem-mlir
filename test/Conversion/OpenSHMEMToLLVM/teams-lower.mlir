// RUN: openshmem-opt %s --convert-openshmem-to-llvm | FileCheck %s

// Test OpenSHMEM team operations lowering to LLVM

module {
  // CHECK-LABEL: @test_team_world
  func.func @test_team_world() -> !openshmem.team {
    // CHECK: call @shmem_team_world() : () -> !openshmem.team
    %world_team = openshmem.team_world -> !openshmem.team
    func.return %world_team : !openshmem.team
  }

  // CHECK-LABEL: @test_team_split_strided
  func.func @test_team_split_strided() -> !openshmem.team {
    %my_pe = openshmem.my_pe : i32
    %n_pes = openshmem.n_pes : i32
    %world_team = openshmem.team_world -> !openshmem.team
    %start = arith.constant 0 : i32
    %stride = arith.constant 1 : i32
    
    // CHECK: call @shmem_team_split_strided(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
    %team, %retval = openshmem.team_split_strided(%world_team, %start, %stride, %n_pes) : !openshmem.team, i32, i32, i32 -> !openshmem.team, i32
    
    func.return %team : !openshmem.team
  }

  // CHECK-LABEL: @test_team_split_2d
  func.func @test_team_split_2d() -> !openshmem.team {
    %world_team = openshmem.team_world -> !openshmem.team
    %num_x = arith.constant 2 : i32
    
    // CHECK: call @shmem_team_split_2d(%{{.*}}, %{{.*}})
    %team_x, %team_y, %retval = openshmem.team_split_2d(%world_team, %num_x) : !openshmem.team, i32 -> !openshmem.team, !openshmem.team, i32
    
    func.return %team_x : !openshmem.team
  }

  // CHECK-LABEL: @test_team_destroy
  func.func @test_team_destroy(%team: !openshmem.team) {
    // CHECK: call @shmem_team_destroy(%{{.*}})
    openshmem.team_destroy(%team) : !openshmem.team
    return
  }

  // CHECK-LABEL: @test_team_my_pe
  func.func @test_team_my_pe(%team: !openshmem.team) -> i32 {
    // CHECK: call @shmem_team_my_pe(%{{.*}}) : (!openshmem.team) -> i32
    %pe = openshmem.team_my_pe(%team) : !openshmem.team -> i32
    func.return %pe : i32
  }

  // CHECK-LABEL: @test_team_n_pes
  func.func @test_team_n_pes(%team: !openshmem.team) -> i32 {
    // CHECK: call @shmem_team_n_pes(%{{.*}}) : (!openshmem.team) -> i32
    %n_pes = openshmem.team_n_pes(%team) : !openshmem.team -> i32
    func.return %n_pes : i32
  }
}
