; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-n32:64-S128-Fn32-p270:32:32:32:32-p271:32:32:32:32-p272:64:64:64:64-i8:8:32-i16:16:32-i64:64-i128:128-p0:64:64:64:64-i1:8-i32:32-f16:16-f64:64-f128:128"
target triple = "aarch64-unknown-linux-gnu"

@.str = private global [24 x i8] c"Hello from PE %d of %d\0A\00", align 1

declare void @shmem_barrier_all()

declare i32 @shmem_n_pes()

declare i32 @shmem_my_pe()

declare void @shmem_finalize()

declare void @shmem_init()

declare i32 @printf(ptr, ...)

define dso_local i32 @main() {
  %1 = alloca i32, i64 1, align 4
  %2 = alloca i32, i64 1, align 4
  %3 = alloca i32, i64 1, align 4
  call void @shmem_init()
  %4 = call i32 @shmem_my_pe()
  store i32 %4, ptr %2, align 4
  %5 = call i32 @shmem_n_pes()
  store i32 %5, ptr %3, align 4
  %6 = load i32, ptr %2, align 4
  %7 = load i32, ptr %3, align 4
  %8 = call i32 (ptr, ...) @printf(ptr @.str, i32 %6, i32 %7)
  call void @shmem_barrier_all()
  call void @shmem_finalize()
  store i32 0, ptr %1, align 4
  %9 = load i32, ptr %1, align 4
  ret i32 %9
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
