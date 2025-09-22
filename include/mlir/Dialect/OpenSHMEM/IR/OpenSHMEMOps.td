//===- OpenSHMEMOps.td - OpenSHMEM dialect operations ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMOPS_TD
#define MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMBase.td"
include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMSetup.td"
include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMMemory.td"
include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMRMAOps.td"
include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMCollectives.td"
include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMTeams.td"
include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMContexts.td"
include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMAtomics.td"
include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMSync.td"
include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMPt2ptSync.td"

// TODO: remaining v1.5 operations

#endif // MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMOPS_TD

