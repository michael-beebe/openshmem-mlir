!s32i = !cir.int<s, 32>
!u8i = !cir.int<u, 8>
module attributes {cir.triple = "aarch64-unknown-linux-gnu", dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i8 = dense<[8, 32]> : vector<2xi64>, i16 = dense<[16, 32]> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 32, 64>, "dlti.stack_alignment" = 128 : i64, "dlti.function_pointer_alignment" = #dlti.function_pointer_alignment<32, function_dependent = true>>} {
  llvm.func @shmem_finalize()
  llvm.func @shmem_barrier_all()
  llvm.func @shmem_n_pes() -> i32
  llvm.func @shmem_my_pe() -> i32
  llvm.func @shmem_init()
  cir.global "private" cir_private dso_local @".str" = #cir.const_array<"Hello from PE %d of %d\0A\00" : !cir.array<!u8i x 24>> : !cir.array<!u8i x 24> {alignment = 1 : i64}
  cir.func private @printf(!cir.ptr<!u8i>, ...) -> !s32i
  cir.func dso_local @main() -> !s32i {
    %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
    %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["me", init] {alignment = 4 : i64}
    %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["npes", init] {alignment = 4 : i64}
    llvm.call @shmem_init() : () -> ()
    %3 = llvm.call @shmem_my_pe() : () -> i32
    %4 = builtin.unrealized_conversion_cast %3 : i32 to !s32i
    cir.store align(4) %4, %1 : !s32i, !cir.ptr<!s32i>
    %5 = llvm.call @shmem_n_pes() : () -> i32
    %6 = builtin.unrealized_conversion_cast %5 : i32 to !s32i
    cir.store align(4) %6, %2 : !s32i, !cir.ptr<!s32i>
    %7 = cir.get_global @".str" : !cir.ptr<!cir.array<!u8i x 24>>
    %8 = cir.cast(array_to_ptrdecay, %7 : !cir.ptr<!cir.array<!u8i x 24>>), !cir.ptr<!u8i>
    %9 = cir.load align(4) %1 : !cir.ptr<!s32i>, !s32i
    %10 = cir.load align(4) %2 : !cir.ptr<!s32i>, !s32i
    %11 = cir.call @printf(%8, %9, %10) : (!cir.ptr<!u8i>, !s32i, !s32i) -> !s32i
    llvm.call @shmem_barrier_all() : () -> ()
    llvm.call @shmem_finalize() : () -> ()
    %12 = cir.const #cir.int<0> : !s32i
    cir.store %12, %0 : !s32i, !cir.ptr<!s32i>
    %13 = cir.load %0 : !cir.ptr<!s32i>, !s32i
    cir.return %13 : !s32i
  }
}

