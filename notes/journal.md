# OpenSHMEM MLIR Journal

## 2025-10-17

- Switched project scripts (`build_toolchain.sh`, `build_openshmem_mlir.sh`, `test_conversion.sh`, `test_end_to_end.sh`) to default to the incubator toolchain and refreshed their help text so incubator is the first-run path.
- Regenerated build targets so `shmem-mlir-opt` now installs under `build-<toolchain>/bin`; updated the end-to-end harness to look there first while retaining a fallback for pre-existing trees.
- Ensured `test_end_to_end.sh` exports `SHMEM_CC`/`SHMEM_CXX` when calling the SOS launcher so it emits CIR compatible with the active toolchain.
- Confirmed `shmem-mlir-opt` always registers the `cir-to-llvm` pipeline and taught the harness to prefer it, dropping to toolchain `cir-opt` only when necessary with `--allow-unregistered-dialect`.
- Extended the CIR RMA rewriters to recognize all typed `shmem_<type>_{get,put}` and context variants so 2DHeatStencil lowers past Step 2; pipeline now halts in Step 3 on the expected unrealized-cast issue that needs dedicated conversion ops.

**Working:** HelloWorld end-to-end flow under both upstream and incubator toolchains.

**Blocked:** 2DHeatStencil remains gated by the upstream ClangIR boolean lowering bug (mismatch between `!cir.bool` and `i1`).

**Next Steps:** Monitor upstream fix for the boolean bug; plan explicit conversion ops to recover type safety once the pipeline is stable.
