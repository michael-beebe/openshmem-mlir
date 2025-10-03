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

//===----------------------------------------------------------------------===//
// Typed RMA Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PutOp (Typed - Generic)
//===----------------------------------------------------------------------===//

def OpenSHMEM_PutOp : OpenSHMEM_Op<"put", []> {
  let summary = "Typed blocking put operation (generic)";
  let description = [{
    This operation performs a typed blocking put, copying data from the
    source memory region on the calling PE to the destination symmetric
    memory region on the target PE. The operation copies `nelems` elements
    of the specified type.
    
    This is the generic typed equivalent of putmem, providing type safety
    and better performance for known data types. The element type is
    inferred from the memref type with symmetric memory space.
    
    Arguments:
    - dest: Destination symmetric memory reference on the target PE
      (memref<T, #openshmem.symmetric_memory> where T is a supported RMA type)
    - source: Source local memory reference on the calling PE (memref<T>)
    - nelems: Number of elements to transfer (index)
    - pe: Target PE number (i32)
    
    The element type T must be one of the supported OpenSHMEM RMA types.
    
    Signature:
    ```c
    void shmem_put(TYPE *dest, const TYPE *source, size_t nelems, int pe);
    ```
    where TYPE is one of the standard RMA types from Table 5.
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
// CtxPutOp (Typed - Context-aware)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxPutOp : OpenSHMEM_Op<"ctx_put", []> {
  let summary = "Typed blocking put operation with context";
  let description = [{
    This operation performs a typed blocking put with a specific context,
    copying data from the source memory region on the calling PE to the
    destination symmetric memory region on the target PE. The operation
    copies `nelems` elements of the specified type.
    
    Arguments:
    - ctx: OpenSHMEM context handle (!openshmem.ctx)
    - dest: Destination symmetric memory reference on the target PE
      (memref<T, #openshmem.symmetric_memory> where T is a supported RMA type)
    - source: Source local memory reference on the calling PE (memref<T>)
    - nelems: Number of elements to transfer (index)
    - pe: Target PE number (i32)
    
    The element type T must be one of the supported OpenSHMEM RMA types.
    
    Signature:
    ```c
    void shmem_put(shmem_ctx_t ctx, TYPE *dest, const TYPE *source,
                   size_t nelems, int pe);
    ```
    where TYPE is one of the standard RMA types from Table 5.
  }];

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
// PutNbiOp (Typed - Non-blocking)
//===----------------------------------------------------------------------===//

def OpenSHMEM_PutNbiOp : OpenSHMEM_Op<"put_nbi", []> {
  let summary = "Typed non-blocking put operation (generic)";
  let description = [{
    This operation performs a typed non-blocking put, copying data from the
    source memory region on the calling PE to the destination symmetric
    memory region on the target PE. The operation copies `nelems` elements
    and returns immediately after the data has been copied out of the source
    array.
    
    This is the generic typed equivalent of putmem_nbi, providing type
    safety and better performance for known data types.
    
    Arguments:
    - dest: Destination symmetric memory reference on the target PE
      (memref<T, #openshmem.symmetric_memory> where T is a supported RMA type)
    - source: Source local memory reference on the calling PE (memref<T>)
    - nelems: Number of elements to transfer (index)
    - pe: Target PE number (i32)
    
    The element type T must be one of the supported OpenSHMEM RMA types.
    
    Signature:
    ```c
    void shmem_put_n(TYPE *dest, const TYPE *source, size_t nelems, int pe);
    ```
    where TYPE is one of the standard RMA types from Table 5.
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
// CtxPutNbiOp (Typed - Context-aware Non-blocking)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxPutNbiOp : OpenSHMEM_Op<"ctx_put_nbi", []> {
  let summary = "Typed non-blocking put operation with context";
  let description = [{
    This operation performs a typed non-blocking put with a specific context,
    copying data from the source memory region on the calling PE to the
    destination symmetric memory region on the target PE. The operation
    copies `nelems` elements and returns immediately after the data has been
    copied out of the source array.
    
    Arguments:
    - ctx: OpenSHMEM context handle (!openshmem.ctx)
    - dest: Destination symmetric memory reference on the target PE
      (memref<T, #openshmem.symmetric_memory> where T is a supported RMA type)
    - source: Source local memory reference on the calling PE (memref<T>)
    - nelems: Number of elements to transfer (index)
    - pe: Target PE number (i32)
    
    The element type T must be one of the supported OpenSHMEM RMA types.
    
    Signature:
    ```c
    void shmem_put_n(shmem_ctx_t ctx, TYPE *dest, const TYPE *source,
                     size_t nelems, int pe);
    ```
    where TYPE is one of the standard RMA types from Table 5.
  }];

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
// Sized Put Operations (8, 16, 32, 64, 128 bit)
//===----------------------------------------------------------------------===//

def OpenSHMEM_Put8Op : OpenSHMEM_Op<"put8", []> {
  let summary = "8-bit sized put operation";
  let description = [{
    This operation performs an 8-bit sized put operation, copying data from
    the source memory region on the calling PE to the destination symmetric
    memory region on the target PE. The operation copies `nelems` bytes.
    
    Arguments:
    - dest: Destination symmetric memory reference on the target PE
      (memref<i8, #openshmem.symmetric_memory>)
    - source: Source local memory reference on the calling PE (memref<i8>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_put8(void *dest, const void *source, size_t nelems, int pe);
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

def OpenSHMEM_Put16Op : OpenSHMEM_Op<"put16", []> {
  let summary = "16-bit sized put operation";
  let description = [{
    This operation performs a 16-bit sized put operation, copying data from
    the source memory region on the calling PE to the destination symmetric
    memory region on the target PE. The operation copies `nelems` bytes.
    
    Arguments:
    - dest: Destination symmetric memory reference on the target PE
      (memref<i16, #openshmem.symmetric_memory>)
    - source: Source local memory reference on the calling PE (memref<i16>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_put16(void *dest, const void *source, size_t nelems, int pe);
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

def OpenSHMEM_Put32Op : OpenSHMEM_Op<"put32", []> {
  let summary = "32-bit sized put operation";
  let description = [{
    This operation performs a 32-bit sized put operation, copying data from
    the source memory region on the calling PE to the destination symmetric
    memory region on the target PE. The operation copies `nelems` bytes.
    
    Arguments:
    - dest: Destination symmetric memory reference on the target PE
      (memref<i32, #openshmem.symmetric_memory>)
    - source: Source local memory reference on the calling PE (memref<i32>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_put32(void *dest, const void *source, size_t nelems, int pe);
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

def OpenSHMEM_Put64Op : OpenSHMEM_Op<"put64", []> {
  let summary = "64-bit sized put operation";
  let description = [{
    This operation performs a 64-bit sized put operation, copying data from
    the source memory region on the calling PE to the destination symmetric
    memory region on the target PE. The operation copies `nelems` bytes.
    
    Arguments:
    - dest: Destination symmetric memory reference on the target PE
      (memref<i64, #openshmem.symmetric_memory>)
    - source: Source local memory reference on the calling PE (memref<i64>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_put64(void *dest, const void *source, size_t nelems, int pe);
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

def OpenSHMEM_Put128Op : OpenSHMEM_Op<"put128", []> {
  let summary = "128-bit sized put operation";
  let description = [{
    This operation performs a 128-bit sized put operation, copying data from
    the source memory region on the calling PE to the destination symmetric
    memory region on the target PE. The operation copies `nelems` bytes.
    
    Arguments:
    - dest: Destination symmetric memory reference on the target PE
      (memref<f128, #openshmem.symmetric_memory> or vector type)
    - source: Source local memory reference on the calling PE (memref<f128>
      or vector type)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_put128(void *dest, const void *source, size_t nelems, int pe);
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
// Context-aware Sized Put Operations
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxPut8Op : OpenSHMEM_Op<"ctx_put8", []> {
  let summary = "8-bit sized put operation with context";
  let description = [{
    This operation performs an 8-bit sized put operation with a context,
    copying data from the source local memory region on the calling PE to
    the destination symmetric memory region on the target PE. The operation
    copies `nelems` bytes.

    Arguments:
    - ctx: OpenSHMEM context handle (!openshmem.ctx)
    - dest: Destination symmetric memory reference on the target PE
      (memref<i8, #openshmem.symmetric_memory>)
    - source: Source local memory reference on the calling PE (memref<i8>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)

    Signature:
    ```c
    void shmem_ctx_put8(shmem_ctx_t ctx, void *dest, const void *source,
                        size_t nelems, int pe);
    ```
  }];

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

def OpenSHMEM_CtxPut16Op : OpenSHMEM_Op<"ctx_put16", []> {
  let summary = "16-bit sized put operation with context";
  let description = [{
    This operation performs a 16-bit sized put operation with a context,
    copying data from the source local memory region on the calling PE to
    the destination symmetric memory region on the target PE. The operation
    copies `nelems` bytes.

    Arguments:
    - ctx: OpenSHMEM context handle (!openshmem.ctx)
    - dest: Destination symmetric memory reference on the target PE
      (memref<i16, #openshmem.symmetric_memory>)
    - source: Source local memory reference on the calling PE (memref<i16>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)

    Signature:
    ```c
    void shmem_ctx_put16(shmem_ctx_t ctx, void *dest, const void *source,
                         size_t nelems, int pe);
    ```
  }];

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

def OpenSHMEM_CtxPut32Op : OpenSHMEM_Op<"ctx_put32", []> {
  let summary = "32-bit sized put operation with context";
  let description = [{
    This operation performs a 32-bit sized put operation with a context,
    copying data from the source local memory region on the calling PE to
    the destination symmetric memory region on the target PE. The operation
    copies `nelems` bytes.

    Arguments:
    - ctx: OpenSHMEM context handle (!openshmem.ctx)
    - dest: Destination symmetric memory reference on the target PE
      (memref<i32, #openshmem.symmetric_memory>)
    - source: Source local memory reference on the calling PE (memref<i32>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)

    Signature:
    ```c
    void shmem_ctx_put32(shmem_ctx_t ctx, void *dest, const void *source,
                         size_t nelems, int pe);
    ```
  }];

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

def OpenSHMEM_CtxPut64Op : OpenSHMEM_Op<"ctx_put64", []> {
  let summary = "64-bit sized put operation with context";
  let description = [{
    This operation performs a 64-bit sized put operation with a context,
    copying data from the source local memory region on the calling PE to
    the destination symmetric memory region on the target PE. The operation
    copies `nelems` bytes.

    Arguments:
    - ctx: OpenSHMEM context handle (!openshmem.ctx)
    - dest: Destination symmetric memory reference on the target PE
      (memref<i64, #openshmem.symmetric_memory>)
    - source: Source local memory reference on the calling PE (memref<i64>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)

    Signature:
    ```c
    void shmem_ctx_put64(shmem_ctx_t ctx, void *dest, const void *source,
                         size_t nelems, int pe);
    ```
  }];

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

def OpenSHMEM_CtxPut128Op : OpenSHMEM_Op<"ctx_put128", []> {
  let summary = "128-bit sized put operation with context";
  let description = [{
    This operation performs a 128-bit sized put operation with a context,
    copying data from the source local memory region on the calling PE to
    the destination symmetric memory region on the target PE. The operation
    copies `nelems` bytes.

    Arguments:
    - ctx: OpenSHMEM context handle (!openshmem.ctx)
    - dest: Destination symmetric memory reference on the target PE
      (memref<f128, #openshmem.symmetric_memory> or vector type)
    - source: Source local memory reference on the calling PE (memref<f128>
      or vector type)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)

    Signature:
    ```c
    void shmem_ctx_put128(shmem_ctx_t ctx, void *dest, const void *source,
                          size_t nelems, int pe);
    ```
  }];

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
// Putmem Operations (byte-level)
//===----------------------------------------------------------------------===//

def OpenSHMEM_PutmemOp : OpenSHMEM_Op<"putmem", []> {
  let summary = "Blocking putmem operation (matches shmem_putmem C API)";
  let description = [{
    This operation performs a blocking putmem, copying raw memory from the
    source memory region on the calling PE to the destination symmetric
    memory region on the target PE. The operation copies `size` bytes and
    returns only after the data has been copied out of the source array on
    the local PE.
    
    Arguments:
    - dest: Destination symmetric memory reference on the target 
      PE (memref<T, #openshmem.symmetric_memory>)
    - src: Source local memory reference on the calling PE (any type)
    - size: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_putmem(void *dest, const void *src, size_t nelems, int pe);
    ```
    
    Note: For putmem operations, nelems represents the number of bytes to
    transfer.
  }];

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

def OpenSHMEM_PutmemNbiOp : OpenSHMEM_Op<"putmem_nbi", []> {
  let summary = "Non-blocking putmem operation (matches shmem_putmem_nbi C API)";
  let description = [{
    This operation performs a non-blocking putmem, copying raw memory from
    the source memory region on the calling PE to the destination symmetric
    memory region on the target PE. The operation copies `nelems` bytes and
    returns immediately after the data has been copied out of the source
    array on the local PE.

    Arguments:
    - dest: Destination symmetric memory reference on the target
      PE (memref<T, #openshmem.symmetric_memory>)
    - src: Source local memory reference on the calling PE (any type)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)

    Signature:
    ```c
    void shmem_putmem_nbi(void *dest, const void *src, size_t nelems, int pe);
    ```

    Note: For putmem_nbi operations, nelems represents the number of bytes
    to transfer.
  }];

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
// Get Operations
//===----------------------------------------------------------------------===//

def OpenSHMEM_GetOp : OpenSHMEM_Op<"get", []> {
  let summary = "Typed blocking get operation (generic)";
  let description = [{
    This operation performs a typed blocking get, copying data from the
    source symmetric memory region on the target PE to the destination local
    memory region on the calling PE. The operation copies `nelems` elements
    of the specified type.
    
    This is the generic typed equivalent of getmem, providing type safety
    and better performance for known data types. The element type is
    inferred from the symmetric memref type.
    
    Arguments:
    - dest: Destination local memory reference on the calling PE (memref<T>)
    - source: Source symmetric memory reference on the target PE
      (memref<T, #openshmem.symmetric_memory> where T is a supported RMA type)
    - nelems: Number of elements to transfer (index)
    - pe: Target PE number (i32)
    
    The element type T must be one of the supported OpenSHMEM RMA types.
    
    Signature:
    ```c
    void shmem_get(TYPE *dest, const TYPE *source, size_t nelems, int pe);
    ```
    where TYPE is one of the standard RMA types from Table 5.
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

def OpenSHMEM_CtxGetOp : OpenSHMEM_Op<"ctx_get", []> {
  let summary = "Typed blocking get operation with context";
  let description = [{
    This operation performs a typed blocking get with a specific context,
    copying data from the source symmetric memory region on the target PE
    to the destination local memory region on the calling PE. The operation
    copies `nelems` elements of the specified type.
    
    Arguments:
    - ctx: OpenSHMEM context handle (!openshmem.ctx)
    - dest: Destination local memory reference on the calling PE (memref<T>)
    - source: Source symmetric memory reference on the target PE
      (memref<T, #openshmem.symmetric_memory> where T is a supported RMA type)
    - nelems: Number of elements to transfer (index)
    - pe: Target PE number (i32)
    
    The element type T must be one of the supported OpenSHMEM RMA types.
    
    Signature:
    ```c
    void shmem_get(shmem_ctx_t ctx, TYPE *dest, const TYPE *source,
                   size_t nelems, int pe);
    ```
    where TYPE is one of the standard RMA types from Table 5.
  }];

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

def OpenSHMEM_GetNbiOp : OpenSHMEM_Op<"get_nbi", []> {
  let summary = "Typed non-blocking get operation (generic)";
  let description = [{
    This operation performs a typed non-blocking get, copying data from the
    source symmetric memory region on the target PE to the destination local
    memory region on the calling PE. The operation copies `nelems` elements
    and returns immediately after the data has been delivered to the
    destination array.
    
    Arguments:
    - dest: Destination local memory reference on the calling PE (memref<T>)
    - source: Source symmetric memory reference on the target PE
      (memref<T, #openshmem.symmetric_memory> where T is a supported RMA type)
    - nelems: Number of elements to transfer (index)
    - pe: Target PE number (i32)
    
    The element type T must be one of the supported OpenSHMEM RMA types.
    
    Signature:
    ```c
    void shmem_get_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe);
    ```
    where TYPE is one of the standard RMA types from Table 5.
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

def OpenSHMEM_CtxGetNbiOp : OpenSHMEM_Op<"ctx_get_nbi", []> {
  let summary = "Typed non-blocking get operation with context";
  let description = [{
    This operation performs a typed non-blocking get with a specific
    context, copying data from the source symmetric memory region on the
    target PE to the destination local memory region on the calling PE. The
    operation copies `nelems` elements and returns immediately after the
    data has been delivered to the destination array.
    
    Arguments:
    - ctx: OpenSHMEM context handle (!openshmem.ctx)
    - dest: Destination local memory reference on the calling PE (memref<T>)
    - source: Source symmetric memory reference on the target PE
      (memref<T, #openshmem.symmetric_memory> where T is a supported RMA type)
    - nelems: Number of elements to transfer (index)
    - pe: Target PE number (i32)
    
    The element type T must be one of the supported OpenSHMEM RMA types.
    
    Signature:
    ```c
    void shmem_get_nbi(shmem_ctx_t ctx, TYPE *dest, const TYPE *source,
                       size_t nelems, int pe);
    ```
    where TYPE is one of the standard RMA types from Table 5.
  }];

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
// Sized Get Operations (8, 16, 32, 64, 128 bit)
//===----------------------------------------------------------------------===//

def OpenSHMEM_Get8Op : OpenSHMEM_Op<"get8", []> {
  let summary = "8-bit sized get operation";
  let description = [{
    This operation performs an 8-bit sized get operation, copying data from
    the source symmetric memory region on the target PE to the destination
    local memory region on the calling PE. The operation copies `nelems`
    bytes.
    
    Arguments:
    - dest: Destination local memory reference on the calling PE (memref<i8>)
    - source: Source symmetric memory reference on the target PE
      (memref<i8, #openshmem.symmetric_memory>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_get8(void *dest, const void *source, size_t nelems, int pe);
    ```
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

def OpenSHMEM_Get16Op : OpenSHMEM_Op<"get16", []> {
  let summary = "16-bit sized get operation";
  let description = [{
    This operation performs a 16-bit sized get operation, copying data from
    the source symmetric memory region on the target PE to the destination
    local memory region on the calling PE. The operation copies `nelems`
    bytes.
    
    Arguments:
    - dest: Destination local memory reference on the calling PE (memref<i16>)
    - source: Source symmetric memory reference on the target PE
      (memref<i16, #openshmem.symmetric_memory>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_get16(void *dest, const void *source, size_t nelems, int pe);
    ```
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

def OpenSHMEM_Get32Op : OpenSHMEM_Op<"get32", []> {
  let summary = "32-bit sized get operation";
  let description = [{
    This operation performs a 32-bit sized get operation, copying data from
    the source symmetric memory region on the target PE to the destination
    local memory region on the calling PE. The operation copies `nelems`
    bytes.
    
    Arguments:
    - dest: Destination local memory reference on the calling PE (memref<i32>)
    - source: Source symmetric memory reference on the target PE
      (memref<i32, #openshmem.symmetric_memory>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_get32(void *dest, const void *source, size_t nelems, int pe);
    ```
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

def OpenSHMEM_Get64Op : OpenSHMEM_Op<"get64", []> {
  let summary = "64-bit sized get operation";
  let description = [{
    This operation performs a 64-bit sized get operation, copying data from
    the source symmetric memory region on the target PE to the destination
    local memory region on the calling PE. The operation copies `nelems`
    bytes.
    
    Arguments:
    - dest: Destination local memory reference on the calling PE (memref<i64>)
    - source: Source symmetric memory reference on the target PE
      (memref<i64, #openshmem.symmetric_memory>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_get64(void *dest, const void *source, size_t nelems, int pe);
    ```
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

def OpenSHMEM_Get128Op : OpenSHMEM_Op<"get128", []> {
  let summary = "128-bit sized get operation";
  let description = [{
    This operation performs a 128-bit sized get operation, copying data from
    the source symmetric memory region on the target PE to the destination
    local memory region on the calling PE. The operation copies `nelems`
    bytes.
    
    Arguments:
    - dest: Destination local memory reference on the calling PE (memref<f128>
      or vector type)
    - source: Source symmetric memory reference on the target PE
      (memref<f128, #openshmem.symmetric_memory> or vector type)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_get128(void *dest, const void *source, size_t nelems, int pe);
    ```
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
// Context-aware Sized Get Operations
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxGet8Op : OpenSHMEM_Op<"ctx_get8", []> {
  let summary = "8-bit sized get operation with context";
  let description = [{
    This operation performs an 8-bit sized get operation with a context,
    copying data from the source symmetric memory region on the target PE
    to the destination local memory region on the calling PE. The operation
    copies `nelems` bytes.
    
    Arguments:
    - ctx: OpenSHMEM context handle (!openshmem.ctx)
    - dest: Destination local memory reference on the calling PE (memref<i8>)
    - source: Source symmetric memory reference on the target PE
      (memref<i8, #openshmem.symmetric_memory>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_ctx_get8(shmem_ctx_t ctx, void *dest, const void *source,
                        size_t nelems, int pe);
    ```
  }];

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

def OpenSHMEM_CtxGet16Op : OpenSHMEM_Op<"ctx_get16", []> {
  let summary = "16-bit sized get operation with context";
  let description = [{
    This operation performs a 16-bit sized get operation with a context,
    copying data from the source symmetric memory region on the target PE
    to the destination local memory region on the calling PE. The operation
    copies `nelems` bytes.
    
    Arguments:
    - ctx: OpenSHMEM context handle (!openshmem.ctx)
    - dest: Destination local memory reference on the calling PE (memref<i16>)
    - source: Source symmetric memory reference on the target PE
      (memref<i16, #openshmem.symmetric_memory>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_ctx_get16(shmem_ctx_t ctx, void *dest, const void *source,
                         size_t nelems, int pe);
    ```
  }];

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

def OpenSHMEM_CtxGet32Op : OpenSHMEM_Op<"ctx_get32", []> {
  let summary = "32-bit sized get operation with context";
  let description = [{
    This operation performs a 32-bit sized get operation with a context,
    copying data from the source symmetric memory region on the target PE
    to the destination local memory region on the calling PE. The operation
    copies `nelems` bytes.
    
    Arguments:
    - ctx: OpenSHMEM context handle (!openshmem.ctx)
    - dest: Destination local memory reference on the calling PE (memref<i32>)
    - source: Source symmetric memory reference on the target PE
      (memref<i32, #openshmem.symmetric_memory>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_ctx_get32(shmem_ctx_t ctx, void *dest, const void *source,
                         size_t nelems, int pe);
    ```
  }];

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

def OpenSHMEM_CtxGet64Op : OpenSHMEM_Op<"ctx_get64", []> {
  let summary = "64-bit sized get operation with context";
  let description = [{
    This operation performs a 64-bit sized get operation with a context,
    copying data from the source symmetric memory region on the target PE
    to the destination local memory region on the calling PE. The operation
    copies `nelems` bytes.
    
    Arguments:
    - ctx: OpenSHMEM context handle (!openshmem.ctx)
    - dest: Destination local memory reference on the calling PE (memref<i64>)
    - source: Source symmetric memory reference on the target PE
      (memref<i64, #openshmem.symmetric_memory>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_ctx_get64(shmem_ctx_t ctx, void *dest, const void *source,
                         size_t nelems, int pe);
    ```
  }];

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

def OpenSHMEM_CtxGet128Op : OpenSHMEM_Op<"ctx_get128", []> {
  let summary = "128-bit sized get operation with context";
  let description = [{
    This operation performs a 128-bit sized get operation with a context,
    copying data from the source symmetric memory region on the target PE
    to the destination local memory region on the calling PE. The operation
    copies `nelems` bytes.
    
    Arguments:
    - ctx: OpenSHMEM context handle (!openshmem.ctx)
    - dest: Destination local memory reference on the calling PE (memref<f128>
      or vector type)
    - source: Source symmetric memory reference on the target PE
      (memref<f128, #openshmem.symmetric_memory> or vector type)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_ctx_get128(shmem_ctx_t ctx, void *dest, const void *source,
                          size_t nelems, int pe);
    ```
  }];

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
// Getmem Operations (byte-level)
//===----------------------------------------------------------------------===//

def OpenSHMEM_GetmemOp : OpenSHMEM_Op<"getmem", []> {
  let summary = "Blocking getmem operation (matches shmem_getmem C API)";
  let description = [{
    This operation performs a blocking getmem, copying raw memory from the
    source symmetric memory region on the target PE to the destination local
    memory region on the calling PE. The operation copies `size` bytes and
    returns only after the data has been delivered to the destination array
    on the local PE.
    
    Arguments:
    - dest: Destination local memory reference on the calling PE (any type)
    - src: Source symmetric memory reference on the target PE
      (memref<T, #openshmem.symmetric_memory>)
    - size: Number of bytes to transfer (index)
    - pe: Target PE number (i32)
    
    Signature:
    ```c
    void shmem_getmem(void *dest, const void *src, size_t nelems, int pe);
    ```
    
    Note: For getmem operations, nelems represents the number of bytes to
    transfer.
  }];

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

def OpenSHMEM_GetmemNbiOp : OpenSHMEM_Op<"getmem_nbi", []> {
  let summary = "Non-blocking getmem operation (matches shmem_getmem_nbi C API)";
  let description = [{
    This operation performs a non-blocking getmem, copying raw memory from
    the source symmetric memory region on the target PE to the destination
    local memory region on the calling PE. The operation copies `size` bytes
    and returns immediately after the data has been delivered to the
    destination array on the local PE.

    Arguments:
    - dest: Destination local memory reference on the calling PE (any type)
    - src: Source symmetric memory reference on the target PE
      (memref<T, #openshmem.symmetric_memory>)
    - nelems: Number of bytes to transfer (index)
    - pe: Target PE number (i32)

    Signature:
    ```c
    void shmem_getmem_nbi(void *dest, const void *src, size_t nelems, int pe);
    ```

    Note: For getmem_nbi operations, nelems represents the number of bytes
    to transfer.
  }];

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
// POp
//===----------------------------------------------------------------------===//

def OpenSHMEM_POp : OpenSHMEM_Op<"p", []> {
  let summary = "Point-to-point put operation";
  let description = [{
    Copies one data item to a remote PE.

    Arguments:
    - dest: Symmetric address of the destination data object.
    - value: Value to be transferred to dest.
    - pe: Target PE number

    Signature:
    ```c
    void shmem_p(void *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $dest `,` $value `,` $pe `)` "
                       "attr-dict `:` "
                       "type($dest) `,` type($value) `,` type($pe)";
}

//===----------------------------------------------------------------------===//
// CtxPOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxPOp : OpenSHMEM_Op<"ctx_p", []> {
  let summary = "Point-to-point put operation with context";
  let description = [{
    Copies one data item to a remote PE.

    Arguments:
    - ctx: OpenSHMEM context handle (!openshmem.ctx)
    - dest: Symmetric address of the destination data object.
    - value: Value to be transferred to dest.
    - pe: Target PE number

    Signature:
    ```c
    void shmem_ctx_p(shmem_ctx_t ctx, void *dest, TYPE value, int pe);
    ```
  }];

  let arguments = (ins
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$dest,
    AnyType:$value,
    I32:$pe
  );

  let assemblyFormat = "`(` $ctx `,` $dest `,` $value `,` $pe `)` "
                       "attr-dict `:` "
                       "type($ctx) `,` type($dest) `,` type($value) `,`"
                       " type($pe)";
}

//===----------------------------------------------------------------------===//
// GOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_GOp : OpenSHMEM_Op<"g", []> {
  let summary = "Point-to-point get operation";
  let description = [{
    Copies one data item from a remote PE.

    Arguments:
    - source: Source symmetric memory reference on the target PE
      (memref<T, #openshmem.symmetric_memory>)
    - pe: Target PE number

    Signature:
    ```c
    void shmem_g(const void *source, int pe);
    ```
  }];

  let arguments = (ins
    SymmetricMemRef:$source,
    I32:$pe
  );

  let results = (outs
    AnyType:$value
  );

  let assemblyFormat = "`(` $source `,` $pe `)` "
                       "attr-dict `:` "
                       "type($source) `,` type($pe) `->` type($value)";
}

//===----------------------------------------------------------------------===//
// CtxGOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxGOp : OpenSHMEM_Op<"ctx_g", []> {
  let summary = "Point-to-point get operation with context";
  let description = [{
    Copies one data item from a remote PE.

    Arguments:
    - ctx: OpenSHMEM context handle (!openshmem.ctx)


    Signature:
    ```c
    void shmem_ctx_g(shmem_ctx_t ctx, TYPE *source, int pe);
    ```
  }];

  let arguments = (ins
    OpenSHMEM_Ctx:$ctx,
    SymmetricMemRef:$source,
    I32:$pe
  );

  let results = (outs
    AnyType:$value
  );

  let assemblyFormat = "`(` $ctx `,` $source `,` $pe `)` "
                       "attr-dict `:` "
                       "type($ctx) `,` type($source) `,` type($pe) `->`"
                       " type($value)";
}

//===----------------------------------------------------------------------===//
// TODO: IputOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: CtxIputOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: Iput8
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: CtxIput8
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: Iput16
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: CtxIput16
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: Iput32
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: CtxIput32
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: Iput64
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: CtxIput64
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: Iput128
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: CtxIput128
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: IGetOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: CtxIGetOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: IGet8
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: CtxIGet8
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: IGet16
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: CtxIGet16
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: IGet32
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: CtxIGet32
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: IGet64
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: CtxIGet64
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: IGet128
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TODO: CtxIGet128
//===----------------------------------------------------------------------===//


#endif // MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMRMAOPS_TD
