// RUN: openshmem-opt %s --convert-cir-to-openshmem | FileCheck %s

// Test CIR -> OpenSHMEM RMA operations conversion

!s32i = !cir.int<s, 32>
!s64i = !cir.int<s, 64>
!void = !cir.void

module {
  cir.func @test_rma_operations() {
    %size = cir.const #cir.int<256> : !s64i
    %pe = cir.const #cir.int<1> : !s32i
    
    // Allocate symmetric memory for testing
    %src = cir.call @shmem_malloc(%size) : (!s64i) -> !cir.ptr<!void>
    %dst = cir.call @shmem_malloc(%size) : (!s64i) -> !cir.ptr<!void>
    
    // Test blocking put
    // CHECK: openshmem.put
    cir.call @shmem_put(%dst, %src, %size, %pe) : (!cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i) -> ()
    
    // Test non-blocking put
    // CHECK: openshmem.put_nbi
    cir.call @shmem_put_nbi(%dst, %src, %size, %pe) : (!cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i) -> ()
    
    // Test blocking get
    // CHECK: openshmem.get
    cir.call @shmem_get(%src, %dst, %size, %pe) : (!cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i) -> ()
    
    // Test non-blocking get
    // CHECK: openshmem.get_nbi
    cir.call @shmem_get_nbi(%src, %dst, %size, %pe) : (!cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i) -> ()
    
    // Test blocking putmem (byte-level)
    // CHECK: openshmem.putmem
    cir.call @shmem_putmem(%dst, %src, %size, %pe) : (!cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i) -> ()
    
    // Test non-blocking putmem
    // CHECK: openshmem.putmem_nbi
    cir.call @shmem_putmem_nbi(%dst, %src, %size, %pe) : (!cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i) -> ()
    
    // Test blocking getmem (byte-level)
    // CHECK: openshmem.getmem
    cir.call @shmem_getmem(%src, %dst, %size, %pe) : (!cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i) -> ()
    
    // Test non-blocking getmem
    // CHECK: openshmem.getmem_nbi
    cir.call @shmem_getmem_nbi(%src, %dst, %size, %pe) : (!cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i) -> ()
    
    // Cleanup
    cir.call @shmem_free(%src) : (!cir.ptr<!void>) -> ()
    cir.call @shmem_free(%dst) : (!cir.ptr<!void>) -> ()
    
    cir.return
  }
  
  // Function declarations
  cir.func private @shmem_malloc(!s64i) -> !cir.ptr<!void>
  cir.func private @shmem_free(!cir.ptr<!void>)
  
  // RMA function declarations
  cir.func private @shmem_put(!cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i)
  cir.func private @shmem_put_nbi(!cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i)
  cir.func private @shmem_get(!cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i)
  cir.func private @shmem_get_nbi(!cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i)
  cir.func private @shmem_putmem(!cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i)
  cir.func private @shmem_putmem_nbi(!cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i)
  cir.func private @shmem_getmem(!cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i)
  cir.func private @shmem_getmem_nbi(!cir.ptr<!void>, !cir.ptr<!void>, !s64i, !s32i)
}
