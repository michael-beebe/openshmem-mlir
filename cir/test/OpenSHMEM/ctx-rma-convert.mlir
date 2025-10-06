// RUN: openshmem-opt %s --convert-cir-to-openshmem | FileCheck %s

// Test CIR -> OpenSHMEM context-aware RMA operations conversion

!s32i = !cir.int<s, 32>
!s64i = !cir.int<s, 64>
!void = !cir.void

module {
  cir.func @test_ctx_rma_operations(%ctx: !cir.ptr<!void>) {
    %size = cir.const #cir.int<256> : !s64i
    %pe = cir.const #cir.int<1> : !s32i
    
    // Allocate symmetric memory for testing
    %src = cir.call @shmem_malloc(%size) : (!s64i) -> !cir.ptr<!void>
    %dst = cir.call @shmem_malloc(%size) : (!s64i) -> !cir.ptr<!void>
    
    // Test context-aware blocking put
    // CHECK: openshmem.ctx_put
    cir.call @shmem_ctx_put(%ctx, %dst, %src, %size, %pe) : (!cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i) -> ()
    
    // Test context-aware non-blocking put
    // CHECK: openshmem.ctx_put_nbi
    cir.call @shmem_ctx_put_nbi(%ctx, %dst, %src, %size, %pe) : (!cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i) -> ()
    
    // Test context-aware blocking get
    // CHECK: openshmem.ctx_get
    cir.call @shmem_ctx_get(%ctx, %src, %dst, %size, %pe) : (!cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i) -> ()
    
    // Test context-aware non-blocking get
    // CHECK: openshmem.ctx_get_nbi
    cir.call @shmem_ctx_get_nbi(%ctx, %src, %dst, %size, %pe) : (!cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i) -> ()
    
    // Cleanup
    cir.call @shmem_free(%src) : (!cir.ptr<!void>) -> ()
    cir.call @shmem_free(%dst) : (!cir.ptr<!void>) -> ()
    
    cir.return
  }
  
  // Function declarations
  cir.func private @shmem_malloc(!s64i) -> !cir.ptr<!void>
  cir.func private @shmem_free(!cir.ptr<!void>)
  
  // Context-aware RMA function declarations
  cir.func private @shmem_ctx_put(!cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i)
  cir.func private @shmem_ctx_put_nbi(!cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i)
  cir.func private @shmem_ctx_get(!cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i)
  cir.func private @shmem_ctx_get_nbi(!cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i)
}
