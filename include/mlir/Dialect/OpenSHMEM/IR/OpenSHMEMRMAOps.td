//===- OpenSHMEMRMAOps.td - OpenSHMEM RMA operations -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMRMAOPS_TD
#define MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMRMAOPS_TD

include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMBase.td"
include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMTypes.td"

//===----------------------------------------------------------------------===//
// Simplified RMA Operations
//
// This file implements a smaller set of RMA operations following the
// simplification strategy in tmp/rma_simplification_analysis.md. The goal is
// to keep a concise MLIR surface (generic typed ops + context variants +
// byte-level mem ops) and let the lowering select the appropriate backend C
// APIs based on element types and presence of context.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Generic Typed RMA Ops
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PutOp
//===----------------------------------------------------------------------===//
def OpenSHMEM_PutOp : OpenSHMEM_Op<"put", []> {
  let summary = "Typed blocking put operation (generic)";
  let description = [{
    Copies `nelems` elements of type T from the local `source` memref to the
    symmetric `dest` memref on the target PE. The element type is inferred
    from the memref types.

    Signature (conceptual):
    ```c
    void shmem_put(TYPE *dest, const TYPE *source, size_t nelems, int pe);
    ```
  }];

  let arguments = (ins
    SymmetricMemRef:$dest,
    AnyMemRef:$source,
    Index:$nelems,
    I32:$pe
  );

  let assemblyFormat = "`(` $dest `,` $source `,` $nelems `,` $pe `)` "
                       "attr-dict `:` "
                       "type($dest) `,` type($source) `,` type($nelems) `,`"
                       " type($pe)";
}

//===----------------------------------------------------------------------===//
// PutNbiOp
//===----------------------------------------------------------------------===//
def OpenSHMEM_PutNbiOp : OpenSHMEM_Op<"put_nbi", []> {
  let summary = "Typed non-blocking put operation (generic)";
  let description = [{
    Non-blocking variant of `put`. Returns once the data has been copied
    out of the local source buffer.
  }];

  let arguments = (ins
    SymmetricMemRef:$dest,
    AnyMemRef:$source,
    Index:$nelems,
    I32:$pe
  );

  let assemblyFormat = "`(` $dest `,` $source `,` $nelems `,` $pe `)` "
                       "attr-dict `:` "
                       "type($dest) `,` type($source) `,` type($nelems) `,`"
                       " type($pe)";
}

//===----------------------------------------------------------------------===//
// GetOp
//===----------------------------------------------------------------------===//
def OpenSHMEM_GetOp : OpenSHMEM_Op<"get", []> {
  let summary = "Typed blocking get operation (generic)";
  let description = [{
    Copies `nelems` elements of type T from the symmetric `source` memref on
    the target PE into the local `dest` memref.
  }];

  let arguments = (ins
    AnyMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nelems,
    I32:$pe
  );

  let assemblyFormat = "`(` $dest `,` $source `,` $nelems `,` $pe `)` "
                       "attr-dict `:` "
                       "type($dest) `,` type($source) `,` type($nelems) `,`"
                       " type($pe)";
}

//===----------------------------------------------------------------------===//
// GetNbiOp
//===----------------------------------------------------------------------===//
def OpenSHMEM_GetNbiOp : OpenSHMEM_Op<"get_nbi", []> {
  let summary = "Typed non-blocking get operation (generic)";
  let description = [{
    Non-blocking variant of `get`. Returns once the data has been delivered
    to the destination buffer on the local PE.
  }];

  let arguments = (ins
    AnyMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nelems,
    I32:$pe
  );

  let assemblyFormat = "`(` $dest `,` $source `,` $nelems `,` $pe `)` "
                       "attr-dict `:` "
                       "type($dest) `,` type($source) `,` type($nelems) `,`"
                       " type($pe)";
}

//===----------------------------------------------------------------------===//
// Context-aware variants (ctx_put, ctx_put_nbi, ctx_get, ctx_get_nbi)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CtxPutOp
//===----------------------------------------------------------------------===//
def OpenSHMEM_CtxPutOp : OpenSHMEM_Op<"ctx_put", []> {
  let summary = "Typed blocking put operation with context";
  let arguments = (ins
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$dest,
    AnyMemRef:$source,
    Index:$nelems,
    I32:$pe
  );
  let assemblyFormat = "`(` $ctx `,` $dest `,` $source `,` $nelems `,` $pe `)` "
                       "attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($source) `,`"
                       " type($nelems) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// CtxPutNbiOp
//===----------------------------------------------------------------------===//
def OpenSHMEM_CtxPutNbiOp : OpenSHMEM_Op<"ctx_put_nbi", []> {
  let summary = "Typed non-blocking put operation with context";
  let arguments = (ins
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$dest,
    AnyMemRef:$source,
    Index:$nelems,
    I32:$pe
  );
  let assemblyFormat = "`(` $ctx `,` $dest `,` $source `,` $nelems `,` $pe `)` "
                       "attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($source) `,`"
                       " type($nelems) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// CtxGetOp
//===----------------------------------------------------------------------===//
def OpenSHMEM_CtxGetOp : OpenSHMEM_Op<"ctx_get", []> {
  let summary = "Typed blocking get operation with context";
  let arguments = (ins
    OpenSHMEM_Ctx:$ctx,
    AnyMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nelems,
    I32:$pe
  );
  let assemblyFormat = "`(` $ctx `,` $dest `,` $source `,` $nelems `,` $pe `)` "
                       "attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($source) `,`"
                       " type($nelems) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// CtxGetNbiOp
//===----------------------------------------------------------------------===//
def OpenSHMEM_CtxGetNbiOp : OpenSHMEM_Op<"ctx_get_nbi", []> {
  let summary = "Typed non-blocking get operation with context";
  let arguments = (ins
    OpenSHMEM_Ctx:$ctx,
    AnyMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nelems,
    I32:$pe
  );
  let assemblyFormat = "`(` $ctx `,` $dest `,` $source `,` $nelems `,` $pe `)` "
                       "attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($source) `,`"
                       " type($nelems) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// Byte-level memory operations (putmem/getmem and non-blocking variants)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PutmemOp
//===----------------------------------------------------------------------===//
def OpenSHMEM_PutmemOp : OpenSHMEM_Op<"putmem", []> {
  let summary = "Blocking putmem operation (matches shmem_putmem C API)";
  let arguments = (ins
    SymmetricMemRef:$dest,
    AnyType:$src,
    Index:$size,
    I32:$pe
  );
  let assemblyFormat = "`(` $dest `,` $src `,` $size `,` $pe `)` "
                       "attr-dict `:` "
                       "type($dest) `,` type($src) `,` type($size) `,`"
                       " type($pe)";
}

//===----------------------------------------------------------------------===//
// PutmemNbiOp
//===----------------------------------------------------------------------===//
def OpenSHMEM_PutmemNbiOp : OpenSHMEM_Op<"putmem_nbi", []> {
  let summary = "Non-blocking putmem operation (matches shmem_putmem_nbi C API)";
  let arguments = (ins
    SymmetricMemRef:$dest,
    AnyType:$src,
    Index:$nelems,
    I32:$pe
  );
  let assemblyFormat = "`(` $dest `,` $src `,` $nelems `,` $pe `)` "
                       "attr-dict `:` "
                       "type($dest) `,` type($src) `,` type($nelems) `,`"
                       " type($pe)";
}

//===----------------------------------------------------------------------===//
// GetmemOp
//===----------------------------------------------------------------------===//
def OpenSHMEM_GetmemOp : OpenSHMEM_Op<"getmem", []> {
  let summary = "Blocking getmem operation (matches shmem_getmem C API)";
  let arguments = (ins
    AnyType:$dest,
    SymmetricMemRef:$src,
    Index:$size,
    I32:$pe
  );
  let assemblyFormat = "`(` $dest `,` $src `,` $size `,` $pe `)` "
                       "attr-dict `:` "
                       "type($dest) `,` type($src) `,` type($size) `,`"
                       " type($pe)";
}

//===----------------------------------------------------------------------===//
// GetmemNbiOp
//===----------------------------------------------------------------------===//
def OpenSHMEM_GetmemNbiOp : OpenSHMEM_Op<"getmem_nbi", []> {
  let summary = "Non-blocking getmem operation (matches shmem_getmem_nbi C API)";
  let arguments = (ins
    AnyType:$dest,
    SymmetricMemRef:$src,
    Index:$nelems,
    I32:$pe
  );
  let assemblyFormat = "`(` $dest `,` $src `,` $nelems `,` $pe `)` "
                       "attr-dict `:` "
                       "type($dest) `,` type($src) `,` type($nelems) `,`"
                       " type($pe)";
}

//===----------------------------------------------------------------------===//
// Notes:
// - Sized (byte-width) variants and single-element p/g ops were removed to
//   reduce redundancy. Use the generic ops with appropriate memref element
//   types and nelems == 1 for single-element transfers. Lowering selects the
//   concrete C API (put8/put32/... or shmem_p/shmem_g) as needed.
//===----------------------------------------------------------------------===//

#endif // MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMRMAOPS_TD
