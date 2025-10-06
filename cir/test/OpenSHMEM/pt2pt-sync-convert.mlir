// RUN: openshmem-opt %s --convert-cir-to-openshmem | FileCheck %s

!s32i = !cir.int<s, 32>
!u64i = !cir.int<u, 64>
!void = !cir.void

module {
  // Test shmem_wait_until
  cir.func @test_wait_until(%ivar: !cir.ptr<!void>, %cmp: !s32i, %cmp_value: !s32i) {
    cir.call @shmem_wait_until(%ivar, %cmp, %cmp_value) : (!cir.ptr<!void>, !s32i, !s32i) -> ()
    cir.return
  }
  cir.func private @shmem_wait_until(!cir.ptr<!void>, !s32i, !s32i)

  // CHECK: @test_wait_until
  // CHECK: openshmem.wait_until

  // Test shmem_wait_until_all
  cir.func @test_wait_until_all(%ivars: !cir.ptr<!void>, %nelems: !u64i, %status: !cir.ptr<!void>, %cmp: !s32i, %cmp_value: !s32i) {
    cir.call @shmem_wait_until_all(%ivars, %nelems, %status, %cmp, %cmp_value) : (!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !s32i, !s32i) -> ()
    cir.return
  }
  cir.func private @shmem_wait_until_all(!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !s32i, !s32i)

  // CHECK: @test_wait_until_all
  // CHECK: openshmem.wait_until_all

  // Test shmem_wait_until_any
  cir.func @test_wait_until_any(%ivars: !cir.ptr<!void>, %nelems: !u64i, %status: !cir.ptr<!void>, %cmp: !s32i, %cmp_value: !s32i) {
    cir.call @shmem_wait_until_any(%ivars, %nelems, %status, %cmp, %cmp_value) : (!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !s32i, !s32i) -> ()
    cir.return
  }
  cir.func private @shmem_wait_until_any(!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !s32i, !s32i)

  // CHECK: @test_wait_until_any
  // CHECK: openshmem.wait_until_any

  // Test shmem_wait_until_some
  cir.func @test_wait_until_some(%ivars: !cir.ptr<!void>, %nelems: !u64i, %indices: !cir.ptr<!void>, %status: !cir.ptr<!void>, %cmp: !s32i, %cmp_value: !s32i) {
    %result = cir.call @shmem_wait_until_some(%ivars, %nelems, %indices, %status, %cmp, %cmp_value) : (!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !cir.ptr<!void>, !s32i, !s32i) -> !u64i
    cir.return
  }
  cir.func private @shmem_wait_until_some(!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !cir.ptr<!void>, !s32i, !s32i) -> !u64i

  // CHECK: @test_wait_until_some
  // CHECK: openshmem.wait_until_some

  // Test shmem_wait_until_all_vector
  cir.func @test_wait_until_all_vector(%ivars: !cir.ptr<!void>, %nelems: !u64i, %status: !cir.ptr<!void>, %cmp: !s32i, %cmp_values: !cir.ptr<!void>) {
    cir.call @shmem_wait_until_all_vector(%ivars, %nelems, %status, %cmp, %cmp_values) : (!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !s32i, !cir.ptr<!void>) -> ()
    cir.return
  }
  cir.func private @shmem_wait_until_all_vector(!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !s32i, !cir.ptr<!void>)

  // CHECK: @test_wait_until_all_vector
  // CHECK: openshmem.wait_until_all_vector

  // Test shmem_wait_until_any_vector
  cir.func @test_wait_until_any_vector(%ivars: !cir.ptr<!void>, %nelems: !u64i, %status: !cir.ptr<!void>, %cmp: !s32i, %cmp_values: !cir.ptr<!void>) {
    cir.call @shmem_wait_until_any_vector(%ivars, %nelems, %status, %cmp, %cmp_values) : (!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !s32i, !cir.ptr<!void>) -> ()
    cir.return
  }
  cir.func private @shmem_wait_until_any_vector(!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !s32i, !cir.ptr<!void>)

  // CHECK: @test_wait_until_any_vector
  // CHECK: openshmem.wait_until_any_vector

  // Test shmem_wait_until_some_vector
  cir.func @test_wait_until_some_vector(%ivars: !cir.ptr<!void>, %nelems: !u64i, %indices: !cir.ptr<!void>, %status: !cir.ptr<!void>, %cmp: !s32i, %cmp_values: !cir.ptr<!void>) {
    %result = cir.call @shmem_wait_until_some_vector(%ivars, %nelems, %indices, %status, %cmp, %cmp_values) : (!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !cir.ptr<!void>, !s32i, !cir.ptr<!void>) -> !u64i
    cir.return
  }
  cir.func private @shmem_wait_until_some_vector(!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !cir.ptr<!void>, !s32i, !cir.ptr<!void>) -> !u64i

  // CHECK: @test_wait_until_some_vector
  // CHECK: openshmem.wait_until_some_vector

  // Test shmem_test
  cir.func @test_test(%ivar: !cir.ptr<!void>, %cmp: !s32i, %cmp_value: !s32i) {
    %result = cir.call @shmem_test(%ivar, %cmp, %cmp_value) : (!cir.ptr<!void>, !s32i, !s32i) -> !s32i
    cir.return
  }
  cir.func private @shmem_test(!cir.ptr<!void>, !s32i, !s32i) -> !s32i

  // CHECK: @test_test
  // CHECK: openshmem.test

  // Test shmem_test_all
  cir.func @test_test_all(%ivars: !cir.ptr<!void>, %nelems: !u64i, %status: !cir.ptr<!void>, %cmp: !s32i, %cmp_value: !s32i) {
    %result = cir.call @shmem_test_all(%ivars, %nelems, %status, %cmp, %cmp_value) : (!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !s32i, !s32i) -> !s32i
    cir.return
  }
  cir.func private @shmem_test_all(!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !s32i, !s32i) -> !s32i

  // CHECK: @test_test_all
  // CHECK: openshmem.test_all

  // Test shmem_test_any
  cir.func @test_test_any(%ivars: !cir.ptr<!void>, %nelems: !u64i, %status: !cir.ptr<!void>, %cmp: !s32i, %cmp_value: !s32i) {
    %result = cir.call @shmem_test_any(%ivars, %nelems, %status, %cmp, %cmp_value) : (!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !s32i, !s32i) -> !u64i
    cir.return
  }
  cir.func private @shmem_test_any(!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !s32i, !s32i) -> !u64i

  // CHECK: @test_test_any
  // CHECK: openshmem.test_any

  // Test shmem_test_some
  cir.func @test_test_some(%ivars: !cir.ptr<!void>, %nelems: !u64i, %indices: !cir.ptr<!void>, %status: !cir.ptr<!void>, %cmp: !s32i, %cmp_value: !s32i) {
    %result = cir.call @shmem_test_some(%ivars, %nelems, %indices, %status, %cmp, %cmp_value) : (!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !cir.ptr<!void>, !s32i, !s32i) -> !u64i
    cir.return
  }
  cir.func private @shmem_test_some(!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !cir.ptr<!void>, !s32i, !s32i) -> !u64i

  // CHECK: @test_test_some
  // CHECK: openshmem.test_some

  // Test shmem_test_all_vector
  cir.func @test_test_all_vector(%ivars: !cir.ptr<!void>, %nelems: !u64i, %status: !cir.ptr<!void>, %cmp: !s32i, %cmp_values: !cir.ptr<!void>) {
    %result = cir.call @shmem_test_all_vector(%ivars, %nelems, %status, %cmp, %cmp_values) : (!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !s32i, !cir.ptr<!void>) -> !s32i
    cir.return
  }
  cir.func private @shmem_test_all_vector(!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !s32i, !cir.ptr<!void>) -> !s32i

  // CHECK: @test_test_all_vector
  // CHECK: openshmem.test_all_vector

  // Test shmem_test_any_vector
  cir.func @test_test_any_vector(%ivars: !cir.ptr<!void>, %nelems: !u64i, %status: !cir.ptr<!void>, %cmp: !s32i, %cmp_values: !cir.ptr<!void>) {
    %result = cir.call @shmem_test_any_vector(%ivars, %nelems, %status, %cmp, %cmp_values) : (!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !s32i, !cir.ptr<!void>) -> !u64i
    cir.return
  }
  cir.func private @shmem_test_any_vector(!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !s32i, !cir.ptr<!void>) -> !u64i

  // CHECK: @test_test_any_vector
  // CHECK: openshmem.test_any_vector

  // Test shmem_test_some_vector
  cir.func @test_test_some_vector(%ivars: !cir.ptr<!void>, %nelems: !u64i, %indices: !cir.ptr<!void>, %status: !cir.ptr<!void>, %cmp: !s32i, %cmp_values: !cir.ptr<!void>) {
    %result = cir.call @shmem_test_some_vector(%ivars, %nelems, %indices, %status, %cmp, %cmp_values) : (!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !cir.ptr<!void>, !s32i, !cir.ptr<!void>) -> !u64i
    cir.return
  }
  cir.func private @shmem_test_some_vector(!cir.ptr<!void>, !u64i, !cir.ptr<!void>, !cir.ptr<!void>, !s32i, !cir.ptr<!void>) -> !u64i

  // CHECK: @test_test_some_vector
  // CHECK: openshmem.test_some_vector

  // Test shmem_signal_wait_until
  cir.func @test_signal_wait_until(%sig_addr: !cir.ptr<!void>, %cmp: !s32i, %cmp_value: !u64i) {
    %result = cir.call @shmem_signal_wait_until(%sig_addr, %cmp, %cmp_value) : (!cir.ptr<!void>, !s32i, !u64i) -> !u64i
    cir.return
  }
  cir.func private @shmem_signal_wait_until(!cir.ptr<!void>, !s32i, !u64i) -> !u64i

  // CHECK: @test_signal_wait_until
  // CHECK: openshmem.signal_wait_until
}
