!s32i = !cir.int<s, 32>
module attributes {cir.triple = "aarch64-unknown-linux-gnu", dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i8 = dense<[8, 32]> : vector<2xi64>, i16 = dense<[16, 32]> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 32, 64>, "dlti.stack_alignment" = 128 : i64, "dlti.function_pointer_alignment" = #dlti.function_pointer_alignment<32, function_dependent = true>>, llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  llvm.mlir.global private @".str"() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : !llvm.array<24 x i8> {
    %0 = llvm.mlir.undef : !llvm.array<24 x i8>
    %1 = llvm.mlir.constant(72 : i8) : i8
    %2 = llvm.insertvalue %1, %0[0] : !llvm.array<24 x i8> 
    %3 = llvm.mlir.constant(101 : i8) : i8
    %4 = llvm.insertvalue %3, %2[1] : !llvm.array<24 x i8> 
    %5 = llvm.mlir.constant(108 : i8) : i8
    %6 = llvm.insertvalue %5, %4[2] : !llvm.array<24 x i8> 
    %7 = llvm.mlir.constant(108 : i8) : i8
    %8 = llvm.insertvalue %7, %6[3] : !llvm.array<24 x i8> 
    %9 = llvm.mlir.constant(111 : i8) : i8
    %10 = llvm.insertvalue %9, %8[4] : !llvm.array<24 x i8> 
    %11 = llvm.mlir.constant(32 : i8) : i8
    %12 = llvm.insertvalue %11, %10[5] : !llvm.array<24 x i8> 
    %13 = llvm.mlir.constant(102 : i8) : i8
    %14 = llvm.insertvalue %13, %12[6] : !llvm.array<24 x i8> 
    %15 = llvm.mlir.constant(114 : i8) : i8
    %16 = llvm.insertvalue %15, %14[7] : !llvm.array<24 x i8> 
    %17 = llvm.mlir.constant(111 : i8) : i8
    %18 = llvm.insertvalue %17, %16[8] : !llvm.array<24 x i8> 
    %19 = llvm.mlir.constant(109 : i8) : i8
    %20 = llvm.insertvalue %19, %18[9] : !llvm.array<24 x i8> 
    %21 = llvm.mlir.constant(32 : i8) : i8
    %22 = llvm.insertvalue %21, %20[10] : !llvm.array<24 x i8> 
    %23 = llvm.mlir.constant(80 : i8) : i8
    %24 = llvm.insertvalue %23, %22[11] : !llvm.array<24 x i8> 
    %25 = llvm.mlir.constant(69 : i8) : i8
    %26 = llvm.insertvalue %25, %24[12] : !llvm.array<24 x i8> 
    %27 = llvm.mlir.constant(32 : i8) : i8
    %28 = llvm.insertvalue %27, %26[13] : !llvm.array<24 x i8> 
    %29 = llvm.mlir.constant(37 : i8) : i8
    %30 = llvm.insertvalue %29, %28[14] : !llvm.array<24 x i8> 
    %31 = llvm.mlir.constant(100 : i8) : i8
    %32 = llvm.insertvalue %31, %30[15] : !llvm.array<24 x i8> 
    %33 = llvm.mlir.constant(32 : i8) : i8
    %34 = llvm.insertvalue %33, %32[16] : !llvm.array<24 x i8> 
    %35 = llvm.mlir.constant(111 : i8) : i8
    %36 = llvm.insertvalue %35, %34[17] : !llvm.array<24 x i8> 
    %37 = llvm.mlir.constant(102 : i8) : i8
    %38 = llvm.insertvalue %37, %36[18] : !llvm.array<24 x i8> 
    %39 = llvm.mlir.constant(32 : i8) : i8
    %40 = llvm.insertvalue %39, %38[19] : !llvm.array<24 x i8> 
    %41 = llvm.mlir.constant(37 : i8) : i8
    %42 = llvm.insertvalue %41, %40[20] : !llvm.array<24 x i8> 
    %43 = llvm.mlir.constant(100 : i8) : i8
    %44 = llvm.insertvalue %43, %42[21] : !llvm.array<24 x i8> 
    %45 = llvm.mlir.constant(10 : i8) : i8
    %46 = llvm.insertvalue %45, %44[22] : !llvm.array<24 x i8> 
    %47 = llvm.mlir.constant(0 : i8) : i8
    %48 = llvm.insertvalue %47, %46[23] : !llvm.array<24 x i8> 
    llvm.return %48 : !llvm.array<24 x i8>
  }
  llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {sym_visibility = "private"}
  llvm.func @main() -> i32 attributes {dso_local} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
    openshmem.init
    %6 = openshmem.my_pe : !s32i
    %7 = builtin.unrealized_conversion_cast %6 : !s32i to i32
    llvm.store %7, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    %8 = openshmem.n_pes : !s32i
    %9 = builtin.unrealized_conversion_cast %8 : !s32i to i32
    llvm.store %9, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    %10 = llvm.mlir.addressof @".str" : !llvm.ptr
    %11 = llvm.getelementptr %10[0] : (!llvm.ptr) -> !llvm.ptr, i8
    %12 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %13 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
    %14 = llvm.call @printf(%11, %12, %13) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32
    openshmem.barrier_all
    openshmem.finalize
    %15 = llvm.mlir.constant(0 : i32) : i32
    llvm.store %15, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    %16 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %16 : i32
  }
}

