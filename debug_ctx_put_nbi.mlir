func.func @test_ctx_put_nbi() {
  %options = arith.constant 0 : i64
  %ctx, %status = openshmem.ctx_create(%options) : i64 -> !openshmem.ctx, i32
  %pe = arith.constant 0 : i32
  %nelems = arith.constant 10 : index
  %symm_mem = openshmem.malloc(%nelems) : (index) -> memref<?xi32, #openshmem.symmetric_memory>
  %local_mem = memref.alloc() : memref<100xi32>
  
  openshmem.ctx_put_nbi(%ctx, %symm_mem, %local_mem, %nelems, %pe) : !openshmem.ctx, memref<?xi32, #openshmem.symmetric_memory>, memref<100xi32>, index, i32
  
  return
}