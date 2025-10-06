// RUN: openshmem-opt %s --convert-openshmem-to-llvm | FileCheck %s

// Test OpenSHMEM setup and query operations lowering to LLVM

module {
  // CHECK-LABEL: @test_init_finalize
  func.func @test_init_finalize() {
    // CHECK: openshmem.region
    openshmem.region {
      // CHECK: call @shmem_init()
      openshmem.init
      
      // CHECK: call @shmem_finalize()
      openshmem.finalize
    }
    return
  }

  // CHECK-LABEL: @test_query_functions
  func.func @test_query_functions() {
    openshmem.region {
      // CHECK: call @shmem_my_pe() : () -> i32
      %my_pe = openshmem.my_pe : i32
      
      // CHECK: call @shmem_n_pes() : () -> i32
      %n_pes = openshmem.n_pes : i32
    }
    return
  }
}
