// RUN: openshmem-opt %s --convert-cir-to-openshmem | FileCheck %s

!s32i = !cir.int<s, 32>
!u64i = !cir.int<u, 64>
!void = !cir.void
!ptrU8 = !cir.ptr<!cir.int<u, 8>>

//===----------------------------------------------------------------------===//
// Basic Atomic Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_atomic_fetch
cir.func @test_atomic_fetch() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["source", init]
  %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %2 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %3 = cir.load %1 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.atomic.fetch
  %4 = cir.call @shmem_atomic_fetch(%2, %3) : (!ptrU8, !s32i) -> !s32i
  cir.return
}

// CHECK-LABEL: @test_atomic_set
cir.func @test_atomic_set() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %3 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %4 = cir.load %1 : !cir.ptr<!s32i>, !s32i
  %5 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.atomic.set
  cir.call @shmem_atomic_set(%3, %4, %5) : (!ptrU8, !s32i, !s32i) -> ()
  cir.return
}

// CHECK-LABEL: @test_atomic_compare_swap
cir.func @test_atomic_compare_swap() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["cond", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %3 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %4 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %5 = cir.load %1 : !cir.ptr<!s32i>, !s32i
  %6 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  %7 = cir.load %3 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.atomic.compare_swap
  %8 = cir.call @shmem_atomic_compare_swap(%4, %5, %6, %7) : (!ptrU8, !s32i, !s32i, !s32i) -> !s32i
  cir.return
}

// CHECK-LABEL: @test_atomic_swap
cir.func @test_atomic_swap() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %3 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %4 = cir.load %1 : !cir.ptr<!s32i>, !s32i
  %5 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.atomic.swap
  %6 = cir.call @shmem_atomic_swap(%3, %4, %5) : (!ptrU8, !s32i, !s32i) -> !s32i
  cir.return
}

// CHECK-LABEL: @test_atomic_fetch_inc
cir.func @test_atomic_fetch_inc() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %2 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %3 = cir.load %1 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.atomic.fetch_inc
  %4 = cir.call @shmem_atomic_fetch_inc(%2, %3) : (!ptrU8, !s32i) -> !s32i
  cir.return
}

// CHECK-LABEL: @test_atomic_inc
cir.func @test_atomic_inc() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %2 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %3 = cir.load %1 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.atomic.inc
  cir.call @shmem_atomic_inc(%2, %3) : (!ptrU8, !s32i) -> ()
  cir.return
}

// CHECK-LABEL: @test_atomic_fetch_add
cir.func @test_atomic_fetch_add() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %3 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %4 = cir.load %1 : !cir.ptr<!s32i>, !s32i
  %5 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.atomic.fetch_add
  %6 = cir.call @shmem_atomic_fetch_add(%3, %4, %5) : (!ptrU8, !s32i, !s32i) -> !s32i
  cir.return
}

// CHECK-LABEL: @test_atomic_add
cir.func @test_atomic_add() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %3 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %4 = cir.load %1 : !cir.ptr<!s32i>, !s32i
  %5 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.atomic.add
  cir.call @shmem_atomic_add(%3, %4, %5) : (!ptrU8, !s32i, !s32i) -> ()
  cir.return
}

// CHECK-LABEL: @test_atomic_fetch_and
cir.func @test_atomic_fetch_and() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %3 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %4 = cir.load %1 : !cir.ptr<!s32i>, !s32i
  %5 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.atomic.fetch_and
  %6 = cir.call @shmem_atomic_fetch_and(%3, %4, %5) : (!ptrU8, !s32i, !s32i) -> !s32i
  cir.return
}

// CHECK-LABEL: @test_atomic_fetch_or
cir.func @test_atomic_fetch_or() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %3 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %4 = cir.load %1 : !cir.ptr<!s32i>, !s32i
  %5 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.atomic.fetch_or
  %6 = cir.call @shmem_atomic_fetch_or(%3, %4, %5) : (!ptrU8, !s32i, !s32i) -> !s32i
  cir.return
}

// CHECK-LABEL: @test_atomic_or
cir.func @test_atomic_or() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %3 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %4 = cir.load %1 : !cir.ptr<!s32i>, !s32i
  %5 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.atomic.or
  cir.call @shmem_atomic_or(%3, %4, %5) : (!ptrU8, !s32i, !s32i) -> ()
  cir.return
}

// CHECK-LABEL: @test_atomic_fetch_xor
cir.func @test_atomic_fetch_xor() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %3 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %4 = cir.load %1 : !cir.ptr<!s32i>, !s32i
  %5 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.atomic.fetch_xor
  %6 = cir.call @shmem_atomic_fetch_xor(%3, %4, %5) : (!ptrU8, !s32i, !s32i) -> !s32i
  cir.return
}

// CHECK-LABEL: @test_atomic_xor
cir.func @test_atomic_xor() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %3 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %4 = cir.load %1 : !cir.ptr<!s32i>, !s32i
  %5 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.atomic.xor
  cir.call @shmem_atomic_xor(%3, %4, %5) : (!ptrU8, !s32i, !s32i) -> ()
  cir.return
}

//===----------------------------------------------------------------------===//
// Context-Aware Atomic Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_ctx_atomic_fetch
cir.func @test_ctx_atomic_fetch() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["ctx", init]
  %1 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["source", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %3 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %4 = cir.load %1 : !cir.ptr<!ptrU8>, !ptrU8
  %5 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.ctx.atomic.fetch
  %6 = cir.call @shmem_ctx_atomic_fetch(%3, %4, %5) : (!ptrU8, !ptrU8, !s32i) -> !s32i
  cir.return
}

// CHECK-LABEL: @test_ctx_atomic_set
cir.func @test_ctx_atomic_set() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["ctx", init]
  %1 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %3 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %4 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %5 = cir.load %1 : !cir.ptr<!ptrU8>, !ptrU8
  %6 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  %7 = cir.load %3 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.ctx.atomic.set
  cir.call @shmem_ctx_atomic_set(%4, %5, %6, %7) : (!ptrU8, !ptrU8, !s32i, !s32i) -> ()
  cir.return
}

// CHECK-LABEL: @test_ctx_atomic_compare_swap
cir.func @test_ctx_atomic_compare_swap() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["ctx", init]
  %1 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["cond", init]
  %3 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %4 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %5 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %6 = cir.load %1 : !cir.ptr<!ptrU8>, !ptrU8
  %7 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  %8 = cir.load %3 : !cir.ptr<!s32i>, !s32i
  %9 = cir.load %4 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.ctx.atomic.compare_swap
  %10 = cir.call @shmem_ctx_atomic_compare_swap(%5, %6, %7, %8, %9) : (!ptrU8, !ptrU8, !s32i, !s32i, !s32i) -> !s32i
  cir.return
}

// CHECK-LABEL: @test_ctx_atomic_swap
cir.func @test_ctx_atomic_swap() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["ctx", init]
  %1 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %3 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %4 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %5 = cir.load %1 : !cir.ptr<!ptrU8>, !ptrU8
  %6 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  %7 = cir.load %3 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.ctx.atomic.swap
  %8 = cir.call @shmem_ctx_atomic_swap(%4, %5, %6, %7) : (!ptrU8, !ptrU8, !s32i, !s32i) -> !s32i
  cir.return
}

// CHECK-LABEL: @test_ctx_atomic_fetch_inc
cir.func @test_ctx_atomic_fetch_inc() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["ctx", init]
  %1 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %3 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %4 = cir.load %1 : !cir.ptr<!ptrU8>, !ptrU8
  %5 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.ctx.atomic.fetch_inc
  %6 = cir.call @shmem_ctx_atomic_fetch_inc(%3, %4, %5) : (!ptrU8, !ptrU8, !s32i) -> !s32i
  cir.return
}

// CHECK-LABEL: @test_ctx_atomic_inc
cir.func @test_ctx_atomic_inc() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["ctx", init]
  %1 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %3 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %4 = cir.load %1 : !cir.ptr<!ptrU8>, !ptrU8
  %5 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.ctx.atomic.inc
  cir.call @shmem_ctx_atomic_inc(%3, %4, %5) : (!ptrU8, !ptrU8, !s32i) -> ()
  cir.return
}

// CHECK-LABEL: @test_ctx_atomic_fetch_add
cir.func @test_ctx_atomic_fetch_add() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["ctx", init]
  %1 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %3 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %4 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %5 = cir.load %1 : !cir.ptr<!ptrU8>, !ptrU8
  %6 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  %7 = cir.load %3 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.ctx.atomic.fetch_add
  %8 = cir.call @shmem_ctx_atomic_fetch_add(%4, %5, %6, %7) : (!ptrU8, !ptrU8, !s32i, !s32i) -> !s32i
  cir.return
}

// CHECK-LABEL: @test_ctx_atomic_add
cir.func @test_ctx_atomic_add() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["ctx", init]
  %1 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %3 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %4 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %5 = cir.load %1 : !cir.ptr<!ptrU8>, !ptrU8
  %6 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  %7 = cir.load %3 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.ctx.atomic.add
  cir.call @shmem_ctx_atomic_add(%4, %5, %6, %7) : (!ptrU8, !ptrU8, !s32i, !s32i) -> ()
  cir.return
}

// CHECK-LABEL: @test_ctx_atomic_fetch_and
cir.func @test_ctx_atomic_fetch_and() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["ctx", init]
  %1 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %3 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %4 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %5 = cir.load %1 : !cir.ptr<!ptrU8>, !ptrU8
  %6 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  %7 = cir.load %3 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.ctx.atomic.fetch_and
  %8 = cir.call @shmem_ctx_atomic_fetch_and(%4, %5, %6, %7) : (!ptrU8, !ptrU8, !s32i, !s32i) -> !s32i
  cir.return
}

// CHECK-LABEL: @test_ctx_atomic_fetch_or
cir.func @test_ctx_atomic_fetch_or() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["ctx", init]
  %1 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %3 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %4 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %5 = cir.load %1 : !cir.ptr<!ptrU8>, !ptrU8
  %6 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  %7 = cir.load %3 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.ctx.atomic.fetch_or
  %8 = cir.call @shmem_ctx_atomic_fetch_or(%4, %5, %6, %7) : (!ptrU8, !ptrU8, !s32i, !s32i) -> !s32i
  cir.return
}

// CHECK-LABEL: @test_ctx_atomic_or
cir.func @test_ctx_atomic_or() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["ctx", init]
  %1 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %3 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %4 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %5 = cir.load %1 : !cir.ptr<!ptrU8>, !ptrU8
  %6 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  %7 = cir.load %3 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.ctx.atomic.or
  cir.call @shmem_ctx_atomic_or(%4, %5, %6, %7) : (!ptrU8, !ptrU8, !s32i, !s32i) -> ()
  cir.return
}

// CHECK-LABEL: @test_ctx_atomic_fetch_xor
cir.func @test_ctx_atomic_fetch_xor() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["ctx", init]
  %1 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %3 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %4 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %5 = cir.load %1 : !cir.ptr<!ptrU8>, !ptrU8
  %6 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  %7 = cir.load %3 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.ctx.atomic.fetch_xor
  %8 = cir.call @shmem_ctx_atomic_fetch_xor(%4, %5, %6, %7) : (!ptrU8, !ptrU8, !s32i, !s32i) -> !s32i
  cir.return
}

// CHECK-LABEL: @test_ctx_atomic_xor
cir.func @test_ctx_atomic_xor() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["ctx", init]
  %1 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["dest", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["value", init]
  %3 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %4 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %5 = cir.load %1 : !cir.ptr<!ptrU8>, !ptrU8
  %6 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  %7 = cir.load %3 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.ctx.atomic.xor
  cir.call @shmem_ctx_atomic_xor(%4, %5, %6, %7) : (!ptrU8, !ptrU8, !s32i, !s32i) -> ()
  cir.return
}

//===----------------------------------------------------------------------===//
// Non-Blocking Atomic Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_atomic_fetch_nbi
cir.func @test_atomic_fetch_nbi() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["fetch_ptr", init]
  %1 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["source", init]
  %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %3 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %4 = cir.load %1 : !cir.ptr<!ptrU8>, !ptrU8
  %5 = cir.load %2 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.atomic.fetch_nbi
  cir.call @shmem_atomic_fetch_nbi(%3, %4, %5) : (!ptrU8, !ptrU8, !s32i) -> ()
  cir.return
}

// CHECK-LABEL: @test_ctx_atomic_fetch_nbi
cir.func @test_ctx_atomic_fetch_nbi() {
  %0 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["ctx", init]
  %1 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["fetch_ptr", init]
  %2 = cir.alloca !ptrU8, !cir.ptr<!ptrU8>, ["source", init]
  %3 = cir.alloca !s32i, !cir.ptr<!s32i>, ["pe", init]
  %4 = cir.load %0 : !cir.ptr<!ptrU8>, !ptrU8
  %5 = cir.load %1 : !cir.ptr<!ptrU8>, !ptrU8
  %6 = cir.load %2 : !cir.ptr<!ptrU8>, !ptrU8
  %7 = cir.load %3 : !cir.ptr<!s32i>, !s32i
  // CHECK: openshmem.ctx.atomic.fetch_nbi
  cir.call @shmem_ctx_atomic_fetch_nbi(%4, %5, %6, %7) : (!ptrU8, !ptrU8, !ptrU8, !s32i) -> ()
  cir.return
}

cir.func private @shmem_atomic_fetch(!ptrU8, !s32i) -> !s32i
cir.func private @shmem_atomic_set(!ptrU8, !s32i, !s32i) -> !void
cir.func private @shmem_atomic_compare_swap(!ptrU8, !s32i, !s32i, !s32i) -> !s32i
cir.func private @shmem_atomic_swap(!ptrU8, !s32i, !s32i) -> !s32i
cir.func private @shmem_atomic_fetch_inc(!ptrU8, !s32i) -> !s32i
cir.func private @shmem_atomic_inc(!ptrU8, !s32i) -> !void
cir.func private @shmem_atomic_fetch_add(!ptrU8, !s32i, !s32i) -> !s32i
cir.func private @shmem_atomic_add(!ptrU8, !s32i, !s32i) -> !void
cir.func private @shmem_atomic_fetch_and(!ptrU8, !s32i, !s32i) -> !s32i
cir.func private @shmem_atomic_fetch_or(!ptrU8, !s32i, !s32i) -> !s32i
cir.func private @shmem_atomic_or(!ptrU8, !s32i, !s32i) -> !void
cir.func private @shmem_atomic_fetch_xor(!ptrU8, !s32i, !s32i) -> !s32i
cir.func private @shmem_atomic_xor(!ptrU8, !s32i, !s32i) -> !void

cir.func private @shmem_ctx_atomic_fetch(!ptrU8, !ptrU8, !s32i) -> !s32i
cir.func private @shmem_ctx_atomic_set(!ptrU8, !ptrU8, !s32i, !s32i) -> !void
cir.func private @shmem_ctx_atomic_compare_swap(!ptrU8, !ptrU8, !s32i, !s32i, !s32i) -> !s32i
cir.func private @shmem_ctx_atomic_swap(!ptrU8, !ptrU8, !s32i, !s32i) -> !s32i
cir.func private @shmem_ctx_atomic_fetch_inc(!ptrU8, !ptrU8, !s32i) -> !s32i
cir.func private @shmem_ctx_atomic_inc(!ptrU8, !ptrU8, !s32i) -> !void
cir.func private @shmem_ctx_atomic_fetch_add(!ptrU8, !ptrU8, !s32i, !s32i) -> !s32i
cir.func private @shmem_ctx_atomic_add(!ptrU8, !ptrU8, !s32i, !s32i) -> !void
cir.func private @shmem_ctx_atomic_fetch_and(!ptrU8, !ptrU8, !s32i, !s32i) -> !s32i
cir.func private @shmem_ctx_atomic_fetch_or(!ptrU8, !ptrU8, !s32i, !s32i) -> !s32i
cir.func private @shmem_ctx_atomic_or(!ptrU8, !ptrU8, !s32i, !s32i) -> !void
cir.func private @shmem_ctx_atomic_fetch_xor(!ptrU8, !ptrU8, !s32i, !s32i) -> !s32i
cir.func private @shmem_ctx_atomic_xor(!ptrU8, !ptrU8, !s32i, !s32i) -> !void

cir.func private @shmem_atomic_fetch_nbi(!ptrU8, !ptrU8, !s32i) -> !void
cir.func private @shmem_ctx_atomic_fetch_nbi(!ptrU8, !ptrU8, !ptrU8, !s32i) -> !void
