# OpenSHMEM MLIR

Out-of-tree MLIR dialect, conversion passes, and tooling for the OpenSHMEM
programming model. This repository provides a staging ground for the OpenSHMEM
dialect outside of the upstream LLVM monorepo together with helper scripts that
build the required LLVM/Clang toolchains, exercise the lowering pipeline, and
keep example artifacts up to date.

---

## Overview

- Defines the OpenSHMEM MLIR dialect (`include/` + `lib/`) and the
	OpenSHMEM→LLVM lowering pipeline (`lib/Conversion`).
- Supplies reproducible build scripts for LLVM toolchains (upstream and
	experimental branches) and for the OpenSHMEM MLIR project itself.
- Captures example artifacts that document each stage of the transformation
	pipeline from C source through to an executable linked with an OpenSHMEM
	runtime.

## Highlights

- **Structured Dialect Definition** – TableGen ODS definitions for setup,
	memory, team, synchronization, and atomic operations with the accompanying
	C++ dialect registration code.
- **Lowering Passes** – `convert-openshmem-to-llvm` pass bundles
	pattern-based rewrites for all OpenSHMEM operations.
- **Automation** – Scripts for bootstrapping toolchains, building this
	project, running conversion tests, and performing demonstrations.
- **Reference Outputs** – The `examples/` directory contains sequential
	snapshots of the pipeline for a `hello_shmem` program.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `include/mlir/Dialect/OpenSHMEM` | Public headers and TableGen definitions for the dialect. |
| `lib/Dialect/OpenSHMEM` | C++ dialect implementation, traits, and transformations. |
| `lib/Conversion/OpenSHMEMToLLVM` | Conversion utilities and the `convert-openshmem-to-llvm` pass. |
| `llvm-project/` | Optional source/build tree for a local LLVM toolchain. |
| `scripts/` | Build/test automation (`build_toolchain.sh`, `build_openshmem_mlir.sh`, `test_conversion.sh`, `test_end_to_end.sh` *experimental*, `format.sh`). |
| `examples/` | Numbered artifacts showing each stage of the pipeline for `hello_shmem`. |
| `test/` | Unit and integration suites that stress conversion pipelines. |
| `openshmem-runtime/` | Sandia OpenSHMEM (SOS) runtime sources used for end-to-end runs. |
| `tools/` | Standalone utilities built alongside the main project. |

---

## Prerequisites

- Git, CMake ≥ 3.24, and a recent Ninja build (default generator).
- A C++17 compiler capable of building LLVM/MLIR.
- Python 3 (for auxiliary tooling and virtualenv setups if desired).
- Optional: OpenSHMEM runtime (SOS) for linking/running the end-to-end example.

> **Note:** Scripts default to the `incubator` toolchain identifier, which
> tracks fast-moving LLVM development branches. Use `--toolchain upstream` to
> build against vanilla llvm-project sources.

## Build the LLVM/Clang Toolchain

```bash
./scripts/build_toolchain.sh --toolchain incubator --jobs 16
```

- `--toolchain incubator` (default) downloads an LLVM development branch.
- `--toolchain upstream` pulls the corresponding branch of llvm-project.
- `CORES`, `GENERATOR`, `CMAKE_ARGS`, and related environment variables allow
	deeper customization.

The helper prints the resolved source/build/install directories and caches them
under toolchain-specific folders inside the repository.

## Build OpenSHMEM MLIR

```bash
./scripts/build_openshmem_mlir.sh --toolchain incubator --jobs 16
```

- Reuses the previously built toolchain via `toolchain.sh` helpers.
- Configures CMake in `build-incubator/` (or `build-upstream/`) and prepares
	the OpenSHMEM MLIR libraries/tools.
- Generates the `shmem-mlir-opt` driver and installs dialect libraries.

### Optional: Build the SOS Runtime

```bash
./scripts/build_sos.sh
```

Produces a local OpenSHMEM runtime under `openshmem-runtime/` that provides the
headers and libraries required for linking the end-to-end executable. The
pipeline scripts automatically adjust `PATH`/`LD_LIBRARY_PATH` when SOS is
available.

---

## Running the Pipeline

### Lower OpenSHMEM MLIR to LLVM Dialect

Use `shmem-mlir-opt` to convert OpenSHMEM dialect IR into the LLVM dialect:

```bash
${BUILD_DIR}/bin/shmem-mlir-opt input.openshmem.mlir \
	--convert-openshmem-to-llvm |
${LLVM_BIN_DIR}/mlir-opt --reconcile-unrealized-casts > output.llvm.mlir
```

Once converted, generate LLVM IR with `mlir-translate` and lower to objects via
`llc` or `clang` as needed.

### Conversion Test Suite

```bash
./scripts/test_conversion.sh --toolchain incubator --verbose
```

The helper validates OpenSHMEM→LLVM lowering against the patterns in
`test/Conversion/OpenSHMEMToLLVM`. Verbose mode prints representative input and
output snippets for easier comparison when developing new patterns.

### Example Artifacts

The `examples/` directory contains reference MLIR/LLVM snapshots for a
`hello_shmem` program. Regenerate them by rerunning the conversion pipeline and
copying refreshed artifacts into the directory.

---

## Development Tips

- **Format:** `./scripts/format.sh` applies the project formatting policy to
	C++ sources and TableGen files.
- **Build Trees:** `build-incubator/` and `build-upstream/` are generated by
	the helper scripts. Remove them or run `cmake --build <dir> --target clean`
	to reset the workspace.
- **Custom Toolchains:** Export `LLVM_DIR`, `MLIR_DIR`, or
	`PROJECT_BUILD_DIR` if you maintain out-of-tree builds.
- **Troubleshooting:** Check `${BUILD_DIR}/bin/shmem-mlir-opt` existence when
	scripts report missing drivers. Re-run `build_openshmem_mlir.sh` if needed.

---

## Roadmap & Status

- Continue expanding OpenSHMEM operation coverage and lowering tests.
- Track upstream LLVM/MLIR changes and periodically refresh toolchain builds.
- Integrate additional runtime validation once SOS support stabilizes.

Contributions via issues and pull requests are welcome. Please follow the LLVM
Project coding standards and include unit tests when adding new functionality.

---

## Citation

If you leverage this project in academic or industrial work, please cite:

```
@inproceedings{10.1145/3731599.3767483,
author = {Beebe, Michael and Michalowicz, Benjamin and McNamara, Andrew and Kumar, Yash and Panda, Dhabaleswar K. and Chen, Yong and Poole, Wendy and Poole, Steve},
title = {OpenSHMEM MLIR: A Dialect for Compile-Time Optimization of One-Sided Communications},
year = {2025},
isbn = {9798400718717},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3731599.3767483},
doi = {10.1145/3731599.3767483},
abstract = {Communication increasingly limits performance in high-performance computing (HPC), yet mainstream compilers focus on computation because communication intent is lost early in compilation. OpenSHMEM offers a one-sided Partitioned Global Address Space (PGAS) model with symmetric memory and explicit synchronization, but lowering to opaque runtime calls hides these semantics from analysis. We present an OpenSHMEM dialect for Multi-Level Intermediate Representation (MLIR) that preserves one-sided communication, symmetric memory, and team/context structure as first-class intermediate representation (IR) constructs. Retaining these semantics prior to lowering enables precise, correctness-preserving optimizations that are difficult to recover from LLVM IR. The dialect integrates with existing MLIR/LLVM passes while directly representing communication and synchronization intent. We demonstrate four transformations: recording the number of processing elements, fusing compatible atomics, converting blocking operations to non-blocking forms when safe, and aggregating small messages. These examples show how explicit OpenSHMEM semantics enable communication-aware optimization and lay the groundwork for richer cross-layer analyses.},
booktitle = {Proceedings of the SC '25 Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis},
pages = {1075--1087},
numpages = {13},
keywords = {OpenSHMEM, MLIR, LLVM, compiler optimizations, PGAS, one-sided communication, symmetric memory, high-performance computing},
location = {},
series = {SC Workshops '25}
}
```

---

## License

This project is distributed under the Apache License v2.0 with LLVM exceptions
(`LICENSE`).


