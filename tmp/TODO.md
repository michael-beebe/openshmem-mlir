OpenSHMEM MLIR - TODO

- [x] Port OpenSHMEM TableGen: Memory, RMA, Atomics, Collectives, Sync
- [x] Port TableGen: Pt2ptSync, Teams, Contexts, Ops aggregator
- [ ] Implement verifiers/constraints for Region, types, attrs
- [x] Finish OpenSHMEM.cpp dialect init (types, attrs, ops)
- [ ] Fill OpenSHMEMOps.cpp custom logic (if any)
- [ ] Write Transforms passes: AtomicFusion
- [ ] Write Transforms passes: MessageAggregation (options, analysis)
- [ ] Write Transforms passes: AsyncConversion
- [ ] Write Transforms passes: InjectNumPEs
- [x] Implement OpenSHMEMToLLVM: setup utils and pass skeleton
- [x] Implement lowering: SetupOpsToLLVM
- [x] Implement lowering: MemoryOpsToLLVM and RMAOpsToLLVM (initial)
- [x] Implement lowering: AtomicOpsToLLVM and Sync/Pt2ptSync (initial)
- [x] Implement lowering: CollectiveOpsToLLVM and TeamOpsToLLVM (initial)
- [x] Add conversion registry stub (ConvertToLLVMPatternInterface)
- [ ] Add mlir-opt plugin (optional) and pass registration
- [ ] Create shmem-opt tool (optional) linking dialect and passes
- [ ] Add lit tests: IR parsing/printing for core ops
- [ ] Add lit tests: conversion to LLVM for core ops
- [ ] Add end-to-end lowering test driver and FileCheck baselines
- [ ] Add docs: Dialects/OpenSHMEM.md with op references
- [ ] Scaffold CIR rewriter pass to map shmem_* calls

