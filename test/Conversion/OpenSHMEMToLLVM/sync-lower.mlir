// RUN: openshmem-opt %s --convert-openshmem-to-llvm | FileCheck %s

// Test OpenSHMEM synchronization operations lowering to LLVM

module {
  // CHECK-LABEL: @test_barrier_all
  func.func @test_barrier_all() {
    openshmem.region {
      // CHECK: call @shmem_barrier_all()
      openshmem.barrier_all
    }
    return
  }

  // CHECK-LABEL: @test_barrier
  func.func @test_barrier(%pe_start: i32, %log_pe_stride: i32, %pe_size: i32, %p_sync: memref<?xi64, #openshmem.symmetric_memory>) {
    openshmem.region {
      // CHECK: call @shmem_barrier(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
      openshmem.barrier(%pe_start, %log_pe_stride, %pe_size, %p_sync) : i32, i32, i32, memref<?xi64, #openshmem.symmetric_memory>
    }
    return
  }

  // CHECK-LABEL: @test_quiet
  func.func @test_quiet() {
    openshmem.region {
      // CHECK: call @shmem_quiet()
      openshmem.quiet
    }
    return
  }

  // CHECK-LABEL: @test_combined_sync
  func.func @test_combined_sync() {
    openshmem.region {
      // Typical synchronization pattern
      openshmem.barrier_all
      openshmem.quiet
    }
    return
  }
}
