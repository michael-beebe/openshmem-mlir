// RUN: openshmem-opt %s --convert-openshmem-to-llvm | FileCheck %s

// Test OpenSHMEM collective operations lowering to LLVM

module {
  // CHECK-LABEL: @test_broadcast
  func.func @test_broadcast(%team: !openshmem.team) {
    openshmem.region {
      %size = arith.constant 64 : index
      %src = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      %dest = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      %nelems = arith.constant 16 : index
      %root_pe = arith.constant 0 : i32
      
      // CHECK: call @shmem_broadcast(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
      %ret = openshmem.broadcast(%team, %dest, %src, %nelems, %root_pe) : !openshmem.team, memref<i32, #openshmem.symmetric_memory>, memref<i32, #openshmem.symmetric_memory>, index, i32 -> i32
      
      openshmem.free(%src) : memref<i32, #openshmem.symmetric_memory>
      openshmem.free(%dest) : memref<i32, #openshmem.symmetric_memory>
      openshmem.yield
    }
    return
  }

  // CHECK-LABEL: @test_collect
  func.func @test_collect(%team: !openshmem.team) {
    openshmem.region {
      %size = arith.constant 256 : index
      %src = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      %dest = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      %nelems = arith.constant 64 : index
      
      // CHECK: call @shmem_collect(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
      %ret = openshmem.collect(%team, %dest, %src, %nelems) : !openshmem.team, memref<i32, #openshmem.symmetric_memory>, memref<i32, #openshmem.symmetric_memory>, index -> i32
      
      openshmem.free(%src) : memref<i32, #openshmem.symmetric_memory>
      openshmem.free(%dest) : memref<i32, #openshmem.symmetric_memory>
      openshmem.yield
    }
    return
  }

  // CHECK-LABEL: @test_sumreduce
  func.func @test_sumreduce(%team: !openshmem.team) {
    openshmem.region {
      %size = arith.constant 128 : index
      %src = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      %dest = openshmem.malloc(%size) : index -> memref<i32, #openshmem.symmetric_memory>
      %nelems = arith.constant 32 : index
      
      // CHECK: call @shmem_sumreduce(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
      %ret = openshmem.sumreduce(%team, %dest, %src, %nelems) : !openshmem.team, memref<i32, #openshmem.symmetric_memory>, memref<i32, #openshmem.symmetric_memory>, index -> i32
      
      openshmem.free(%src) : memref<i32, #openshmem.symmetric_memory>
      openshmem.free(%dest) : memref<i32, #openshmem.symmetric_memory>
      openshmem.yield
    }
    return
  }
}
