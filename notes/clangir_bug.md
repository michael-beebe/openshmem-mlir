# ClangIR Boolean Type Conversion Bug

**Date**: October 10, 2024  
**Status**: Blocking 2DHeatStencil test at Step 3 (CIR-to-LLVM conversion)

---

## The Bug

ClangIR's `cir.ternary` operation fails to properly convert `!cir.bool` types to LLVM's `i1` type during the CIR-to-LLVM lowering pass. This causes a type mismatch error when boolean values are yielded from ternary operation regions.

### Error Message

```
/tmp/2DHeatStencil/2.2d_stencil.openshmem.mlir:90:11: error: type mismatch for bb argument #0 of successor #0
cir.yield %31 : !cir.bool
          ^
note: see current operation: "llvm.br"(%68)[^bb16] : (i1) -> ()
```

### What's Happening

1. **CIR Stage**: The `cir.ternary` operation creates regions with `cir.yield` that return `!cir.bool` values
2. **During CIR-to-LLVM**: The ternary gets lowered to LLVM control flow with `llvm.br` branches
3. **Type Conversion Fails**: The `!cir.bool` value should be converted to `i1`, but it's not
4. **Result**: The `llvm.br` expects `i1` but receives `!cir.bool`, causing verification failure

---

## Example: Where This Shows Up

### C Source Code (2d_stencil.c)

```c
// This innocent-looking conditional:
if (me == 0 && npes > 1) {
    shmem_double_get(remote_data, local_data, 10, 1);
}
```

### Generated CIR (After Step 1)

```mlir
%25 = cir.load align(4) %1 : !cir.ptr<!s32i>, !s32i
%26 = cir.const #cir.int<0> : !s32i
%27 = cir.cmp(eq, %25, %26) : !s32i, !cir.bool          // Comparison produces !cir.bool

%28 = cir.ternary(%27, true {
  %29 = cir.load align(4) %2 : !cir.ptr<!s32i>, !s32i
  %30 = cir.const #cir.int<1> : !s32i
  %31 = cir.cmp(gt, %29, %30) : !s32i, !cir.bool        // Another !cir.bool
  cir.yield %31 : !cir.bool                             // ⚠️ This yield has !cir.bool type
}, false {
  %29 = cir.const #false
  cir.yield %29 : !cir.bool                             // ⚠️ This yield also has !cir.bool
}) : (!cir.bool) -> !cir.bool

cir.if %28 {
  // ... actual work
}
```

### What Should Happen During CIR-to-LLVM

The `cir.ternary` should be lowered to something like:

```mlir
llvm.cond_br %condition, ^bb_true, ^bb_false

^bb_true:
  %result = ... // compute true value
  llvm.br ^bb_merge(%result : i1)    // ✅ Should pass i1

^bb_false:
  %result = ... // compute false value
  llvm.br ^bb_merge(%result : i1)    // ✅ Should pass i1

^bb_merge(%merged : i1):              // ✅ Should accept i1
  // continue
```

### What Actually Happens

```mlir
llvm.cond_br %condition, ^bb_true, ^bb_false

^bb_true:
  %68 = ... // some value with type i1 (after conversion)
  llvm.br ^bb16(%68) : (i1) -> ()     // ✅ Branch passes i1

^bb_merge(%arg : !cir.bool):          // ❌ But block argument is still !cir.bool!
  // Type mismatch error!
```

The problem is that the block argument in the merge block is not being converted from `!cir.bool` to `i1`.

---

## Why This Matters for OpenSHMEM

### Impact

- **Simple programs (HelloWorld)**: ✅ Work fine - no complex conditionals
- **Real programs (2DHeatStencil)**: ❌ Fail - use `&&` operator which ClangIR represents as `cir.ternary`
- **Production code**: ❌ Almost all real OpenSHMEM programs use conditionals like:
  ```c
  if (me == 0 && npes > 1) { ... }
  if (i > 0 || j > 0) { ... }
  if (condition1 && condition2) { ... }
  ```

### Current Workarounds

1. **Avoid short-circuit evaluation**: 
   ```c
   // Instead of:
   if (me == 0 && npes > 1) { ... }
   
   // Use nested ifs:
   if (me == 0) {
       if (npes > 1) {
           // ...
       }
   }
   ```

2. **Use separate boolean variables**:
   ```c
   // Instead of:
   if (x && y) { ... }
   
   // Use:
   int cond1 = x;
   int cond2 = y;
   if (cond1) {
       if (cond2) {
           // ...
       }
   }
   ```

But these workarounds make the code unnatural and defeat the purpose of using a high-level language.

---

## Why We Can't Fix This in OpenSHMEM Dialect

This is **not an OpenSHMEM dialect issue** - it's a ClangIR type conversion bug. Here's why:

1. **The bug occurs before OpenSHMEM operations**: The error happens in Step 3 (CIR-to-LLVM), but our OpenSHMEM operations only exist in Step 2
2. **We don't control CIR-to-LLVM lowering**: This is a built-in LLVM pass (`-cir-to-llvm`) that we don't implement
3. **The `cir.ternary` operation is the problem**: This is a ClangIR operation, not ours
4. **Our dialect works correctly**: When we bypass the ClangIR frontend (HelloWorld), everything works

### Evidence

After fixing all OpenSHMEM dialect issues:
- ✅ Step 1 (C → CIR): Success - ClangIR frontend works
- ✅ Step 2 (CIR → OpenSHMEM): Success - Our rewriters work correctly
- ❌ Step 3 (CIR → LLVM): **Failure** - ClangIR's own lowering pass fails
- Step 4-9: Never reached due to Step 3 failure

The OpenSHMEM dialect never even sees the `!cir.bool` values - they're internal to the CIR dialect.

---

## Next Steps

### Immediate Actions

1. **Report to ClangIR**: Create minimal reproducer and file GitHub issue
2. **Document limitation**: Update OpenSHMEM dialect docs with known ClangIR issues
3. **Build test suite**: Focus on programs that avoid `&&`/`||` operators

### Long-term Solutions

1. **Wait for ClangIR fix**: This is the cleanest solution
2. **Alternative frontend**: Consider using standard Clang → LLVM IR → MLIR import
3. **Custom ClangIR patches**: Fork ClangIR and fix the boolean conversion (high maintenance cost)

### What Works Now

The OpenSHMEM dialect successfully handles:
- ✅ All RMA operations (put, get, nbi variants)
- ✅ Typed variants (shmem_double_get, shmem_float_put, etc.)
- ✅ Setup operations (init, finalize, my_pe, n_pes)
- ✅ Memory operations (malloc, free)
- ✅ Multi-block regions with control flow
- ✅ Simple conditionals (single comparisons, nested ifs)

What's blocked by ClangIR bug:
- ❌ Short-circuit boolean operators (`&&`, `||`)
- ❌ Complex conditional expressions in single if-statements
- ❌ Ternary operator (`condition ? true_val : false_val`)

---

## Minimal Reproducer

To verify this is a ClangIR bug (not OpenSHMEM):

```c
// test_bool_bug.c
int main() {
    int x = 1;
    int y = 2;
    
    // This triggers cir.ternary with !cir.bool
    if (x > 0 && y > 1) {
        return 0;
    }
    return 1;
}
```

Compile with ClangIR:
```bash
clang -fclangir -emit-mlir test_bool_bug.c -o test.cir.mlir
mlir-opt --cir-to-llvm test.cir.mlir -o test.llvm.mlir  # ❌ Should fail with same error
```

This reproducer has **no OpenSHMEM code** and should still trigger the bug.
