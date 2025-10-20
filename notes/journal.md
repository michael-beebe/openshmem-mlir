# OpenSHMEM MLIR Journal

## 2025-10-17

### Progress

- Switched project scripts (`build_toolchain.sh`, `build_openshmem_mlir.sh`, `test_conversion.sh`, `test_end_to_end.sh`) to default to the incubator toolchain and refreshed their help text so incubator is the first-run path.
- Regenerated build targets so `shmem-mlir-opt` now installs under `build-<toolchain>/bin`; updated the end-to-end harness to look there first while retaining a fallback for pre-existing trees.
- Ensured `test_end_to_end.sh` exports `SHMEM_CC`/`SHMEM_CXX` when calling the SOS launcher so it emits CIR compatible with the active toolchain.
- Confirmed `shmem-mlir-opt` always registers the `cir-to-llvm` pipeline and taught the harness to prefer it, dropping to toolchain `cir-opt` only when necessary with `--allow-unregistered-dialect`.
- Extended the CIR RMA rewriters to recognize all typed `shmem_<type>_{get,put}` and context variants so 2DHeatStencil lowers past Step 2.
- Default pipeline still selects the toolchain `cir-opt`, so Step 3 lacks the OpenSHMEM dialect; rerunning with `CIR_TO_LLVM_TOOL=build-incubator/bin/shmem-mlir-opt` loads our dialect but then fails when the illegal `builtin.unrealized_conversion_cast` (%13 = !cir.int<u,64> → index) hits the CIR→LLVM legality check, confirming the typed rewrites work but the cast bridge remains unresolved.

### Working

- HelloWorld end-to-end flow under both upstream and incubator toolchains.

### Known Issues

- **2DHeatStencil unrealized cast failure (Step 3):** `shmem-mlir-opt` conversion emits `builtin.unrealized_conversion_cast` when mapping CIR integers/pointers to OpenSHMEM types; CIR→LLVM marks the op illegal, halting the pipeline. Rather than adding custom conversion ops in our dialect, investigate handing this back to the ClangIR team.
- **ClangIR boolean lowering bug:** Upstream issue where `cir.ternary` produces `!cir.bool` block arguments that mismatch the expected `i1` after lowering, blocking 2DHeatStencil even if the cast issue is resolved.

### Immediate Plan

- Engage the ClangIR maintainers with the unrealized cast blocker and boolean lowering bug rather than papering over them locally.

## 2025-10-19

### Status

- Step 3 of the 2DHeatStencil pipeline still fails inside `openshmem.region`: the CIR→LLVM branch rewrite produces `i1` operands, but the successor block arguments remain `!cir.bool`, triggering the verifier.
- Initial conversion pattern meant to call `convertRegionTypes` never fires, so region blocks keep their CIR types and the mismatch persists.

### Next Steps

- Implement an explicit region argument rewrite (e.g., per-block signature conversion) so every `openshmem.region` block adopts LLVM-legal argument types before the branch lowering runs.
- Re-test the end-to-end flow after the region rewrite and chase whatever the next failure is.

### Updates

- Added a per-block signature conversion for `openshmem.region`; Step 3 now completes under the incubator toolchain without verifier trips.
- Step 4 currently fails because `openshmem.wrap_value` still receives CIR integer types (`!u64i`), so the OpenSHMEM→LLVM conversion flags it as illegal.

### Follow-up Work

- Teach the conversion pipeline to drop the remaining `!cir.*` artifacts (starting with `openshmem.wrap_value`) before the OpenSHMEM→LLVM stage runs.

## 2025-10-19 (evening)

### Recent Actions

- Implemented CIR→OpenSHMEM RMA rewriters and centralized helper utilities so typed and generic shmem_put/get variants lower into `openshmem.*` ops.
- Added multiple bridge-op lowering strategies in `lib/Conversion/OpenSHMEMToLLVM/BridgeOpsToLLVM.cpp`: materialization to `UnrealizedConversionCastOp`, targeted ConvertOpToLLVMPatterns for pointer/memref cases, and a conservative fallback rewrite to remove remaining `openshmem.wrap_value` ops.
- Tried a few pre-lowering approaches in `ConvertOpenSHMEMToLLVMPass` (greedy pattern application and a safe two-phase walk/replace). Iterated on these to avoid triggering memref materialization crashes.

### Current Result

- Build/iteration: `shmem-mlir-opt` rebuilt successfully with the changes.
- End-to-end harness still fails at Step 4 (OpenSHMEM→LLVM): a `wrap_value` op remains that carries a memref operand and produces a CIR pointer type and is not being legalized by the current set of patterns. Example failing op seen in `3.2d_stencil.partial-llvm.mlir`:
	- `%27 = openshmem.wrap_value %26 : memref<?xi8, #openshmem.symmetric_memory> -> !cir.ptr<!void>`

### Next Steps

- Add a narrowly scoped pre-lowering or pattern that handles the remaining memref->cir.ptr `wrap_value` variant (or instrument pattern matching to discover why the conversion pattern is not firing for this case).
- If this is a systemic mismatch (CIR types leaking into OpenSHMEM lowering), consider opening a brief design issue to determine whether CIR types should be removed earlier in the pipeline or if the OpenSHMEM->LLVM conversion should accept and materialize these bridges.

### Note

- I avoided broad pre-lowering that previously caused crashes by restricting replacements to specific type shapes; the remaining failing pattern appears to need a narrowly targeted fix.
