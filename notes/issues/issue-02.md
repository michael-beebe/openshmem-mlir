# ClangIR → LLVM rejects OpenSHMEM bridge casts in 2DHeatStencil pipeline

## Summary

Running `./scripts/test_end_to_end.sh --test 2DHeatStencil --toolchain incubator` completes the CIR → OpenSHMEM step but fails during CIR → LLVM. The conversion emits `builtin.unrealized_conversion_cast` ops to coerce CIR types into the shapes our OpenSHMEM ops expect, and the CIR → LLVM pipeline marks those casts illegal, stopping the pipeline immediately after `2.2d_stencil.openshmem.mlir` is produced.

## Environment

- Toolchain: `incubator` (`clangir/build-main`)
- Project build: `build-incubator` (rebuilt after typed RMA rewrite updates)
- Runtime: `openshmem-runtime/SOS-v1.5.2`
- Script: `./scripts/test_end_to_end.sh --test 2DHeatStencil --toolchain incubator`
- Step 3 forced to use our driver: `CIR_TO_LLVM_TOOL=build-incubator/bin/shmem-mlir-opt`

## Reproduction

1. `ninja -C build-incubator shmem-mlir-opt`
2. `./scripts/test_end_to_end.sh --test 2DHeatStencil --toolchain incubator`
3. Step 3 fails with:

   ```text
   %13 = builtin.unrealized_conversion_cast %12 : !u64i to index
          ^
   error: failed to legalize operation 'builtin.unrealized_conversion_cast' that was explicitly marked illegal
   ```

## Expected Result

- CIR → LLVM conversion produces `3.2d_stencil.partial-llvm.mlir`, leaving OpenSHMEM ops intact for the next pass.

## Actual Result

- Conversion aborts as soon as the illegal `builtin.unrealized_conversion_cast` is encountered; no partial LLVM MLIR file is generated.

## Discussion

- The cast originates when we map `!cir.int<u,64>` (and other CIR types) to the types expected by OpenSHMEM operations. Typed RMA rewrites made this path hot for real inputs.
- Rather than bolting explicit conversion ops into our dialect, we should coordinate with the ClangIR team to decide the right inter-dialect interface.
- Separate upstream blocker: ClangIR’s boolean lowering bug (`!cir.bool` vs `i1`) still prevents 2DHeatStencil completion even after this issue is resolved.

## References

- `notes/journal.md` entry dated 2025-10-17
- Failing artifacts under `tmp/2DHeatStencil-incubator-YYYYMMDD-HHMMSS/`
