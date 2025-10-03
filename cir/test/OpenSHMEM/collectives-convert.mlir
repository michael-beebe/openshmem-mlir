// RUN: openshmem-opt %s --openshmem-recognition --convert-cir-to-openshmem | FileCheck %s --check-prefix=CONVERT

// Test conversion of collective-related CIR calls to OpenSHMEM dialect ops

module {
  func.func @test_collectives_conversion() {
    %team = func.call @shmem_team_world() : () -> !openshmem.team
    %size = arith.constant 64 : index
    %root = arith.constant 0 : i32
    %dst_stride = arith.constant 1 : index
    %src_stride = arith.constant 1 : index
    %nreduce = arith.constant 10 : index

    %dest = func.call @shmem_malloc(%size) : (index) -> memref<?xi32, #openshmem.symmetric_memory>
    %source = func.call @shmem_malloc(%size) : (index) -> memref<?xi32, #openshmem.symmetric_memory>

    // CONVERT: [[vret1:%.*]] = openshmem.broadcastmem([[vteam]], [[vdest]], [[vsource]], [[vsize]], [[vroot]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index, i32 -> i32
    %ret1 = func.call @shmem_broadcast(%team, %dest, %source, %size, %root) : (!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index, i32) -> i32

    // CONVERT: [[vret2:%.*]] = openshmem.collectmem([[vteam]], [[vdest]], [[vsource]], [[vsize]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
    %ret2 = func.call @shmem_collect(%team, %dest, %source, %size) : (!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32

    // CONVERT: [[vret3:%.*]] = openshmem.fcollectmem([[vteam]], [[vdest]], [[vsource]], [[vsize]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
    %ret3 = func.call @shmem_fcollect(%team, %dest, %source, %size) : (!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32

    // CONVERT: [[vret4:%.*]] = openshmem.alltoallmem([[vteam]], [[vdest]], [[vsource]], [[vsize]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
    %ret4 = func.call @shmem_alltoall(%team, %dest, %source, %size) : (!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32

    // CONVERT: [[vret5:%.*]] = openshmem.alltoallsmem([[vteam]], [[vdest]], [[vsource]], [[vdst_stride]], [[vsrc_stride]], [[vsize]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index, index, index -> i32
    %ret5 = func.call @shmem_alltoalls(%team, %dest, %source, %dst_stride, %src_stride, %size) : (!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index, index, index) -> i32

    // CONVERT: [[vret6:%.*]] = openshmem.sumreduce([[vteam]], [[vdest]], [[vsource]], [[vnreduce]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
    %ret6 = func.call @shmem_sum_reduce(%team, %dest, %source, %nreduce) : (!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32

    // CONVERT: [[vret7:%.*]] = openshmem.maxreduce([[vteam]], [[vdest]], [[vsource]], [[vnreduce]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
    %ret7 = func.call @shmem_max_reduce(%team, %dest, %source, %nreduce) : (!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32

    // CONVERT: [[vret8:%.*]] = openshmem.minreduce([[vteam]], [[vdest]], [[vsource]], [[vnreduce]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
    %ret8 = func.call @shmem_min_reduce(%team, %dest, %source, %nreduce) : (!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32

    // CONVERT: [[vret9:%.*]] = openshmem.prodreduce([[vteam]], [[vdest]], [[vsource]], [[vnreduce]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
    %ret9 = func.call @shmem_prod_reduce(%team, %dest, %source, %nreduce) : (!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32

    // CONVERT: [[vret10:%.*]] = openshmem.andreduce([[vteam]], [[vdest]], [[vsource]], [[vnreduce]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
    %ret10 = func.call @shmem_and_reduce(%team, %dest, %source, %nreduce) : (!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32

    // CONVERT: [[vret11:%.*]] = openshmem.orreduce([[vteam]], [[vdest]], [[vsource]], [[vnreduce]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
    %ret11 = func.call @shmem_or_reduce(%team, %dest, %source, %nreduce) : (!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32

    // CONVERT: [[vret12:%.*]] = openshmem.xorreduce([[vteam]], [[vdest]], [[vsource]], [[vnreduce]]) : !openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index -> i32
    %ret12 = func.call @shmem_xor_reduce(%team, %dest, %source, %nreduce) : (!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32

    func.call @shmem_free(%dest) : (memref<?xi32, #openshmem.symmetric_memory>) -> ()
    func.call @shmem_free(%source) : (memref<?xi32, #openshmem.symmetric_memory>) -> ()

    return
  }

  // Private CIR functions representing the original APIs
  func.func private @shmem_team_world() -> !openshmem.team
  func.func private @shmem_malloc(index) -> memref<?xi32, #openshmem.symmetric_memory>
  func.func private @shmem_free(memref<?xi32, #openshmem.symmetric_memory>) -> ()
  func.func private @shmem_broadcast(!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index, i32) -> i32
  func.func private @shmem_collect(!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32
  func.func private @shmem_fcollect(!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32
  func.func private @shmem_alltoall(!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32
  func.func private @shmem_alltoalls(!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index, index, index) -> i32
  func.func private @shmem_sum_reduce(!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32
  func.func private @shmem_max_reduce(!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32
  func.func private @shmem_min_reduce(!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32
  func.func private @shmem_prod_reduce(!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32
  func.func private @shmem_and_reduce(!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32
  func.func private @shmem_or_reduce(!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32
  func.func private @shmem_xor_reduce(!openshmem.team, memref<?xi32, #openshmem.symmetric_memory>, memref<?xi32, #openshmem.symmetric_memory>, index) -> i32
}