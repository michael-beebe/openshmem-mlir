// Simplified RMA Operations Analysis
//
// Current: 36 RMA operations
// Proposed: 12 RMA operations (67% reduction)
//
// CURRENT REDUNDANT STRUCTURE:
// ============================
// 1. Generic typed: put, get, put_nbi, get_nbi
// 2. Context typed: ctx_put, ctx_get, ctx_put_nbi, ctx_get_nbi  
// 3. Sized variants: put8, put16, put32, put64, put128 (×2 for get)
// 4. Context sized: ctx_put8, ctx_put16, ctx_put32, ctx_put64, ctx_put128 (×2 for get)
// 5. Memory: putmem, getmem, putmem_nbi, getmem_nbi
// 6. Single element: p, g, ctx_p, ctx_g
//
// PROPOSED SIMPLIFIED STRUCTURE:
// ===============================
// 1. Generic operations (handles all types via MLIR type system):
//    - openshmem.put(dest, source, nelems, pe) 
//    - openshmem.get(dest, source, nelems, pe)
//    - openshmem.put_nbi(dest, source, nelems, pe)
//    - openshmem.get_nbi(dest, source, nelems, pe)
//
// 2. Context operations:
//    - openshmem.ctx_put(ctx, dest, source, nelems, pe)
//    - openshmem.ctx_get(ctx, dest, source, nelems, pe) 
//    - openshmem.ctx_put_nbi(ctx, dest, source, nelems, pe)
//    - openshmem.ctx_get_nbi(ctx, dest, source, nelems, pe)
//
// 3. Memory operations (byte-level):
//    - openshmem.putmem(dest, source, nelems, pe)
//    - openshmem.getmem(dest, source, nelems, pe)
//    - openshmem.putmem_nbi(dest, source, nelems, pe) 
//    - openshmem.getmem_nbi(dest, source, nelems, pe)
//
// LOWERING STRATEGY:
// ==================
// The LLVM lowering patterns choose the appropriate C function based on:
// 1. Element type of the memref (i8->put8, i32->int_put, f64->double_put, etc.)
// 2. Context presence (ctx variants)
// 3. Memory space (symmetric -> RMA, regular -> error)
//
// BENEFITS:
// =========
// 1. 67% reduction in operations (36 -> 12)
// 2. Cleaner MLIR representation
// 3. Type safety through MLIR type system
// 4. Easier to maintain and extend
// 5. Matches OpenSHMEM spec's conceptual model
// 6. Single-element p/g operations can be expressed as nelems=1