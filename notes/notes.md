# OpenSHMEM MLIR Dialect - Development Notes

**Date**: October 10, 2024  
**Status**: HelloWorld ✅ | 2DHeatStencil ❌ Blocked by ClangIR bug

---

## Executive Summary

We've been working to get the 2DHeatStencil test to compile through the full pipeline. The HelloWorld test works perfectly, but 2DHeatStencil fails due to an upstream ClangIR bug with boolean type conversions. In attempting to fix this, we explored several solutions including changing type constraints to `AnyType`, but this feels like the wrong approach. This document describes the original state of the code, the problems we encountered, and the various solutions we've tried.

---

## 2025-10-17 - Incubator Pipeline Notes

- Observed `cir.cast` parse failures when running `test_end_to_end.sh --toolchain incubator`; `shmem-mlir-opt` (and `cir-opt`) could not read CIR emitted via `oshcc` because the wrapper defaulted to the upstream 21.x clang.
- Root cause: SOS `oshcc` hard-coded its compiler to `${repo}/llvm-project/build-release-21.x/bin/clang`, which emits legacy CIR syntax (`cir.cast(array_to_ptrdecay, ...)`) that the incubator parser rejects.
- Fix: Prefix `oshcc` invocations with `SHMEM_CC`/`SHMEM_CXX` pointing at the resolved toolchain clang binaries so the wrapper uses the same toolchain we requested.
- Updated `scripts/test_end_to_end.sh` to set those environment variables for both the initial CIR emission and final link step; incubator runs now see consistent CIR syntax.
- Updated `shmem-mlir-opt` so both incubator and upstream builds register the `cir-to-llvm` pipeline; the end-to-end harness now probes for that support and prefers the project driver, falling back to the toolchain `cir-opt` only if necessary (adding `--allow-unregistered-dialect` in that case).

---

## Original Code State (Before Modifications)

### Type System Design

The OpenSHMEM dialect was originally designed with **specific type constraints** to enforce type safety:

#### RMA Operations (`OpenSHMEMRMAOps.td`)
```tablegen
// Original put operation signature
def OpenSHMEM_Put : OpenSHMEM_Op<"put"> {
  let arguments = (ins 
    SymmetricMemRef:$dest,    // Destination in symmetric memory
    AnyMemRef:$source,        // Source can be any memref
    Index:$nelems,            // Number of elements (index type)
    I32:$pe                   // PE number (32-bit integer)
  );
}

// Similar for get, put_nbi, get_nbi, and context-aware variants
```

#### Memory Operations (`OpenSHMEMMemory.td`)
```tablegen
def OpenSHMEM_Malloc : OpenSHMEM_Op<"malloc"> {
  let arguments = (ins Index:$size);
  let results = (outs SymmetricMemRef:$ptr);
}

def OpenSHMEM_Free : OpenSHMEM_Op<"free"> {
  let arguments = (ins SymmetricMemRef:$ptr);
}
```

#### Setup Operations (`OpenSHMEMSetup.td`)
```tablegen
def OpenSHMEM_Region : OpenSHMEM_Op<"region", [
  SingleBlockImplicitTerminator<"openshmem::YieldOp">
]> {
  let regions = (region SizedRegion<1>:$body);
}
```

**Key Design Principles:**
- `SymmetricMemRef`: Enforces symmetric memory semantics
- `Index`: Standard MLIR index type for sizes/counts
- `I32`: 32-bit integers for PE numbers (matches OpenSHMEM spec)
- `SingleBlockImplicitTerminator`: Assumes simple linear control flow

---

## Problems Encountered

### Problem 1: Typed RMA Variants Not Recognized ✅ FIXED

**Issue**: The ClangIR rewriters couldn't recognize typed RMA variants like `shmem_double_get`, `shmem_float_put`, etc.

**Root Cause**: Pattern matching in `RewriteRMA.cpp` only looked for exact matches like `"shmem_get"` or `"shmem_put"`, not the 23 standard typed variants (`shmem_<TYPE>_<OP>` pattern).

**Impact**: Any real OpenSHMEM C code failed immediately at Step 2 (CIR → OpenSHMEM).

**Solution**: ✅ **FIXED** - Added `matchesTypedRMA()` helper function:
```cpp
static bool matchesTypedRMA(StringRef funcName, StringRef baseOp) {
  // Check exact match first
  if (funcName == ("shmem_" + baseOp).str())
    return true;
    
  // Check all 23 typed variants
  const char *types[] = {"float", "double", "longdouble", "char", "schar",
                         "short", "int", "long", "longlong", "uchar", 
                         "ushort", "uint", "ulong", "ulonglong", "int8",
                         "int16", "int32", "int64", "uint8", "uint16",
                         "uint32", "uint64", "size"};
  
  for (auto *type : types) {
    if (funcName == ("shmem_" + type + "_" + baseOp).str())
      return true;
  }
  return false;
}
```

**Status**: This fix is solid and should be kept.

---

### Problem 2: Type Mismatches Between CIR and MLIR Types ⚠️ CORE ISSUE

**Issue**: CIR types don't match the MLIR types expected by OpenSHMEM operations.

**Example**:
```mlir
// CIR generates:
%size = ... : !cir.int<u, 64>          // ClangIR's unsigned 64-bit int
%ptr = ... : !cir.ptr<!cir.double>     // ClangIR's pointer type

// But OpenSHMEM operations expect:
openshmem.malloc(%size : Index) -> (SymmetricMemRef)
                       ^^^^^ Type mismatch!
```

**Specific Mismatches:**
- `!cir.int<u, 64>` vs `Index` (for sizes)
- `!cir.int<s, 32>` vs `I32` (for PE numbers)
- `!cir.ptr<...>` vs `MemRef` types (for pointers)

**Why This Happens**: We're in a **three-dialect pipeline**:
1. CIR dialect (from ClangIR frontend)
2. OpenSHMEM dialect (our custom dialect)
3. LLVM dialect (final target)

Each has its own type system, and we need to interoperate with all three.

---

### Problem 3: Multi-Block Regions ✅ FIXED

**Issue**: The `openshmem.region` operation had `SingleBlockImplicitTerminator` trait, which only allows single-block regions.

**Impact**: Any control flow inside the OpenSHMEM region (loops, conditionals) caused verification failure:
```
error: 'openshmem.region' op expects region #0 to have 0 or 1 blocks
```

**Why 2DHeatStencil Has Multiple Blocks**: The generated CIR has 32 blocks due to:
- Loop structures (`for` loops)
- Conditional statements (`if` statements)
- Short-circuit boolean evaluation (`&&`, `||` operators)

**Solution**: ✅ **FIXED** - Changed trait:
```tablegen
// Before:
def OpenSHMEM_Region : OpenSHMEM_Op<"region", [
  SingleBlockImplicitTerminator<"openshmem::YieldOp">
]> {

// After:
def OpenSHMEM_Region : OpenSHMEM_Op<"region", [
  NoRegionArguments
]> {
```

And added conditional `openshmem.yield` terminator in the rewriters.

**Status**: This fix is correct and should be kept.

---

### Problem 4: Unrealized Conversion Casts ⚠️ THE BIG PROBLEM

**Issue**: When we try to convert CIR types to MLIR types, we create `builtin.unrealized_conversion_cast` operations. However, the CIR-to-LLVM pipeline **explicitly marks these as illegal**.

**What We Tried Initially**:
```cpp
// In RewriteRMA.cpp - ORIGINAL APPROACH (didn't work)
static Value convertPtrToMemRef(OpBuilder &builder, Location loc, Value ptr) {
  auto ptrType = ptr.getType();
  
  // Try to create a memref type
  auto memrefType = MemRefType::get({ShapedType::kDynamic}, 
                                    builder.getF64Type(),
                                    /* layout */ {},
                                    /* memorySpace */ builder.getI32IntegerAttr(1));
  
  // This creates an unrealized_conversion_cast
  return builder.create<UnrealizedConversionCastOp>(loc, memrefType, ptr).getResult(0);
}
```

**Result**: 
```
error: 'builtin.unrealized_conversion_cast' op requires target to be legal for given typeConverter
```

**Why This Failed**: The CIR-to-LLVM conversion pass (Step 3) explicitly marks unrealized casts as illegal. From LLVM's CIR codebase:
```cpp
// In cir-to-llvm pass
target.addIllegalOp<mlir::UnrealizedConversionCastOp>();
```

This is intentional - the CIR-to-LLVM pass wants ALL types to be properly converted, with no intermediate casts left over.

---

### Problem 5: ClangIR Boolean Bug ⚠️ UPSTREAM BLOCKER

**Issue**: ClangIR's `cir.ternary` operation fails to convert `!cir.bool` to `i1` during CIR-to-LLVM lowering.

**Error**:
```
error: type mismatch for bb argument #0 of successor #0
cir.yield %31 : !cir.bool
note: see current operation: "llvm.br"(%68)[^bb16] : (i1) -> ()
```

**Root Cause**: When C code has `&&` or `||` operators:
```c
if (me == 0 && npes > 1) {
    shmem_double_get(remote_data, local_data, 10, 1);
}
```

ClangIR generates `cir.ternary` with `cir.yield %bool : !cir.bool`. During lowering, the branch instruction passes `i1` but the block argument remains `!cir.bool` - type mismatch.

**Why We Can't Fix This**: 
- This happens in Step 3 (CIR-to-LLVM), before our OpenSHMEM-to-LLVM (Step 4)
- The `cir.ternary` lowering is in ClangIR's code, not ours
- Our dialect never sees these operations

**Current Status**: Blocks 2DHeatStencil at Step 3. See `clangir_bug.md` for details.

---

## Solutions We've Tried

### Solution 1: Use `--allow-unregistered-dialect` Flag ❌ FAILED

**Idea**: Allow unrealized conversion casts by permitting unregistered operations.

**What We Did**:
```bash
mlir-opt --allow-unregistered-dialect \
         -cir-to-llvm \
         input.mlir
```

**Result**: ❌ **FAILED** - The CIR-to-LLVM pass still marks unrealized casts as illegal, regardless of this flag. The flag only affects dialect registration, not operation legality.

---

### Solution 2: Create Proper MemRef Types with Symmetric Memory Space ❌ FAILED

**Idea**: Convert CIR pointers to proper MemRef types with a custom memory space attribute for symmetric memory.

**What We Did**:
```cpp
// Try to create "proper" memref with symmetric memory space
auto memrefType = MemRefType::get(
  {ShapedType::kDynamic},           // Dynamic shape
  builder.getF64Type(),             // Element type
  /* layout */ {},                  // Default layout
  builder.getI32IntegerAttr(1)      // Memory space = 1 (symmetric)
);

Value memref = builder.create<UnrealizedConversionCastOp>(
  loc, memrefType, cirPtr
).getResult(0);
```

**Result**: ❌ **FAILED** - Still creates unrealized conversion casts, which are illegal. The problem isn't the memref type itself, it's the cast operation.

---

### Solution 3: Change All Types to `AnyType` ⚠️ WORKS BUT FEELS WRONG

**Idea**: Make OpenSHMEM operations accept any type, passing CIR types through without conversion.

**What We Did**:
```tablegen
// Changed RMA operations to:
def OpenSHMEM_Put : OpenSHMEM_Op<"put"> {
  let arguments = (ins 
    AnyType:$dest,      // Was: SymmetricMemRef
    AnyType:$source,    // Was: AnyMemRef
    AnyType:$nelems,    // Was: Index
    AnyType:$pe         // Was: I32
  );
}

// Changed memory operations to:
def OpenSHMEM_Malloc : OpenSHMEM_Op<"malloc"> {
  let arguments = (ins AnyType:$size);  // Was: Index
  let results = (outs AnyType:$ptr);    // Was: SymmetricMemRef
}
```

**Result**: ✅ **WORKS** - No more unrealized conversion casts! Types flow through:
- Step 1: C → CIR (CIR types)
- Step 2: CIR → OpenSHMEM (CIR types preserved)
- Step 3: CIR-to-LLVM (CIR types converted to LLVM)
- Step 4: OpenSHMEM-to-LLVM (LLVM types preserved)

**Why This Works**: By accepting `AnyType`, we avoid creating any conversion casts. The types just "flow through" our dialect, and get converted by the CIR-to-LLVM pass later.

**Why This Feels Wrong**:
- ❌ Loses type safety - we can't enforce symmetric memory semantics at compile time
- ❌ Operations can accept ANY type, even nonsensical ones
- ❌ Defeats the purpose of having a typed intermediate representation
- ❌ Makes verification much harder

**Comparison to Other Dialects**:
- `arith` dialect: Uses `AnyType` for generic arithmetic (but that's appropriate there)
- `scf` dialect: Uses `AnyType` for control flow values (also appropriate)
- `memref` dialect: Uses **specific** memref types with memory spaces
- `gpu` dialect: Uses **specific** types for device memory vs host memory

The `AnyType` approach is common for **generic** dialects, but OpenSHMEM has **specific** semantics (symmetric memory, PE-based addressing) that should be enforced.

---

### Solution 4: Custom Type Constraints ⚠️ ATTEMPTED BUT INCOMPLETE

**Idea**: Create custom type constraints that accept either CIR, MLIR, or LLVM types.

**What We Tried**:
```tablegen
// Attempted custom type constraint
def AnyPointerLike : Type<
  Or<[CIR_AnyPtr, AnyMemRef, LLVM_AnyPointer]>,
  "any pointer-like type"
>;

def AnyIntegerLike : Type<
  Or<[CIR_AnyInt, AnyInteger, Index]>,
  "any integer-like type"
>;
```

**Problems**:
- ❌ TableGen doesn't have `Or` for type constraints
- ❌ Would need complex C++ predicates
- ❌ CIR types aren't registered in our dialect (can't check for them in TableGen)
- ❌ Would require separate operation definitions for each type system (explosion of ops)

**Status**: Abandoned this approach due to TableGen limitations.

---

### Solution 5: Create Separate Conversion Operations ⚠️ THEORETICAL

**Idea**: Add explicit conversion operations in the OpenSHMEM dialect to convert between type systems.

**Hypothetical Design**:
```tablegen
def OpenSHMEM_CIRToMemRef : OpenSHMEM_Op<"cir_to_memref"> {
  let arguments = (ins CIR_AnyPtr:$cir_ptr);
  let results = (outs SymmetricMemRef:$memref);
}
```

Then in the rewriters:
```cpp
Value memref = builder.create<OpenSHMEM_CIRToMemRefOp>(loc, resultType, cirPtr);
```

**Why This Might Work**:
- ✅ Makes conversions explicit
- ✅ Can be lowered properly in OpenSHMEM-to-LLVM pass
- ✅ Preserves type safety in OpenSHMEM operations

**Why We Haven't Tried This Yet**:
- ⚠️ Requires defining conversion ops for all type pairs
- ⚠️ Need to implement lowering for these conversion ops
- ⚠️ Unclear if this would bypass the "illegal unrealized cast" problem
- ⚠️ More complex, but might be the "right" solution

---

## Current State Summary

### What's Working
- ✅ HelloWorld: Compiles and runs successfully with `AnyType` approach
- ✅ Typed RMA variants: All 23 types recognized
- ✅ Multi-block regions: Control flow supported
- ✅ Pipeline Steps 1-2: C → CIR → OpenSHMEM MLIR

### What's Blocked
- ❌ 2DHeatStencil: Blocked by ClangIR boolean bug at Step 3
- ⚠️ Type safety: Lost due to `AnyType` workaround

### Code State
- **Modified** (with `AnyType`): RMA ops, Memory ops accept any type
- **Fixed** (good changes): Typed RMA recognition, multi-block regions
- **Upstream blocked**: ClangIR boolean conversion bug

---

## Recommendations

### Option 1: Keep `AnyType` for Now (Pragmatic) ⚠️

**Pros**:
- ✅ Unblocks development
- ✅ HelloWorld works
- ✅ Can implement other features (atomics, collectives)

**Cons**:
- ❌ Loses type safety
- ❌ Not a "correct" solution
- ❌ Technical debt

**When to Use**: If you want to make progress on other features while waiting for ClangIR fixes.

---

### Option 2: Implement Explicit Conversion Operations (Proper Solution) ✅ RECOMMENDED

**Approach**:
1. Revert RMA/Memory ops back to specific types (`SymmetricMemRef`, `Index`, `I32`)
2. Create conversion operations in OpenSHMEM dialect:
   - `openshmem.cir_ptr_to_memref`
   - `openshmem.cir_int_to_index`
   - `openshmem.cir_int_to_i32`
3. Use these in the CIR rewriters instead of creating unrealized casts
4. Implement lowering for these conversion ops in OpenSHMEM-to-LLVM pass

**Example**:
```tablegen
// OpenSHMEMConversions.td
def OpenSHMEM_CIRPtrToMemRef : OpenSHMEM_Op<"cir_ptr_to_memref"> {
  let summary = "Convert CIR pointer to symmetric memref";
  let arguments = (ins AnyType:$cir_ptr);
  let results = (outs SymmetricMemRef:$memref);
  let hasVerifier = 1;
}
```

```cpp
// In RewriteRMA.cpp
Value memref = builder.create<openshmem::CIRPtrToMemRefOp>(
  loc, symmetricMemRefType, cirPtr
);
openshmem::PutOp putOp = builder.create<openshmem::PutOp>(
  loc, memref, source, nelems, pe
);
```

```cpp
// In OpenSHMEMToLLVM.cpp - lowering
class ConvertCIRPtrToMemRef : public OpConversionPattern<openshmem::CIRPtrToMemRefOp> {
  LogicalResult matchAndRewrite(
    openshmem::CIRPtrToMemRefOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const override {
    
    // At this point, CIR types should already be converted to LLVM by CIR-to-LLVM pass
    // So we just need to reinterpret LLVM pointer as memref descriptor
    Value llvmPtr = adaptor.getCirPtr();
    
    // Create memref descriptor from LLVM pointer
    Value memref = createMemRefDescriptor(rewriter, op.getLoc(), llvmPtr);
    
    rewriter.replaceOp(op, memref);
    return success();
  }
};
```

**Pros**:
- ✅ Preserves type safety
- ✅ Makes conversions explicit and controllable
- ✅ Proper separation of concerns
- ✅ Can add verification/validation

**Cons**:
- ⚠️ More work upfront
- ⚠️ Need to implement lowering for conversion ops

**Why This Is Better**: This is how MLIR is designed to work - explicit operations for conversions, with proper lowering passes.

---

### Option 3: Wait for ClangIR Fix and Use Standard MLIR Import ⏳

**Approach**:
1. Report ClangIR boolean bug upstream
2. Wait for fix
3. Or switch to: C → LLVM IR → MLIR import path (bypass ClangIR entirely)

**Pros**:
- ✅ Might be simpler if ClangIR fixes the bug
- ✅ Standard LLVM IR import is well-tested

**Cons**:
- ❌ Waiting on upstream (unknown timeline)
- ❌ LLVM IR import loses high-level structure from C

---

## My Recommendation

I recommend **Option 2: Implement Explicit Conversion Operations**.

**Reasoning**:
1. **Type safety matters**: OpenSHMEM has specific semantics (symmetric memory) that should be enforced
2. **This is the MLIR way**: Explicit conversion ops are how MLIR dialects interoperate
3. **Future-proof**: Works regardless of ClangIR state
4. **Learning opportunity**: Proper MLIR dialect design pattern

**Implementation Plan**:
1. Create `OpenSHMEMConversions.td` with conversion ops
2. Revert RMA/Memory ops to specific types
3. Update rewriters to use conversion ops instead of direct casts
4. Implement lowering in `OpenSHMEMToLLVM/ConversionOpsToLLVM.cpp`
5. Test with HelloWorld first, then 2DHeatStencil (will still hit ClangIR bug, but our dialect will be correct)

**This way**:
- Your dialect is "correct" and type-safe
- The ClangIR bug is isolated (not your problem)
- When ClangIR is fixed, 2DHeatStencil will just work

Would you like me to implement this approach?
