//===- OpenSHMEMCollectives.td - OpenSHMEM collective operations -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMCOLLECTIVES_TD
#define MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMCOLLECTIVES_TD

include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMBase.td"

//===----------------------------------------------------------------------===//
// AlltoallmemOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_AlltoallmemOp : OpenSHMEM_Op<"alltoallmem", []> {
  let summary = "All-to-all communication";
  let description = [{
    This operation performs an all-to-all communication operation.
    
    Exchanges a fixed amount of contiguous data blocks between all pairs
    of PEs participating in the collective routine.

    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - nelems: Number of elements to transfer

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_alltoallmem(shmem_team_t team, void *dest,
                          const void *source, size_t nelems);
    ```
  }];

  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nelems
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $nelems `)` "
                       "attr-dict `:` type($team) `,` type($dest) `,` "
                       "type($source) `,` type($nelems) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// AlltoallOp (Typed)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AlltoallOp : OpenSHMEM_Op<"alltoall", []> {
  let summary = "Typed all-to-all communication";
  let description = [{
    This operation performs a typed all-to-all communication operation.
    
    Exchanges a fixed amount of contiguous data blocks between all pairs
    of PEs participating in the collective routine.

    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - nelems: Number of elements to transfer

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_<TYPE>_alltoall(shmem_team_t team,
                              TYPE *dest, const TYPE *source, size_t nelems);
    ```
    where TYPE is one of the standard types (int, float, double, etc.)
  }];

  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nelems
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $nelems `)` "
                       "attr-dict `:` type($team) `,` type($dest) `,` "
                       "type($source) `,` type($nelems) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// AlltoallsmemOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_AlltoallsmemOp : OpenSHMEM_Op<"alltoallsmem", []> {
  let summary = "All-to-all communication";
  let description = [{
    This operation performs a strided all-to-all communication operation.

    Exchanges a fixed amount of strided data blocks between all pairs of
    PEs participating in the collective routine.

    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - nelems: Number of elements to transfer
    - dst: Stride for destination memory
    - sst: Stride for source memory

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_alltoallsmem(shmem_team_t team, void *dest, const void *source,
                        ptrdiff_t dst, ptrdiff_t sst, size_t nelems);
    ```
  }];

  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$dst,
    Index:$sst,
    Index:$nelems
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $dst `,` $sst `,` "
                       "$nelems `)` attr-dict `:` type($team) `,` type($dest) "
                       "`,` type($source) `,` type($dst) `,` type($sst) `,` "
                       "type($nelems) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// AlltoallsOp (Typed Strided)
//===----------------------------------------------------------------------===//

def OpenSHMEM_AlltoallsOp : OpenSHMEM_Op<"alltoalls", []> {
  let summary = "Typed strided all-to-all communication";
  let description = [{
    This operation performs a typed strided all-to-all communication
    operation.

    Exchanges a fixed amount of strided data blocks between all pairs of
    PEs participating in the collective routine.

    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - dst: Stride for destination memory
    - sst: Stride for source memory
    - nelems: Number of elements to transfer

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_<TYPE>_alltoalls(shmem_team_t team, TYPE *dest, const TYPE
                               *source, ptrdiff_t dst, ptrdiff_t sst, size_t
                               nelems);
    ```
    where TYPE is one of the standard types (int, float, double, etc.)
  }];

  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$dst,
    Index:$sst,
    Index:$nelems
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $dst `,` $sst `,` "
                       "$nelems `)` attr-dict `:` type($team) `,` type($dest) "
                       "`,` type($source) `,` type($dst) `,` type($sst) `,` "
                       "type($nelems) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// BroadcastmemOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_BroadcastmemOp : OpenSHMEM_Op<"broadcastmem", []> {
  let summary = "Broadcast a value to all PEs in a team";
  let description = [{
    This operation performs a broadcast operation on a value to all PEs in a
    team.

    Broadcasts a block of data from one PE to one or more destination PEs.

    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - nelems: Number of elements to transfer
    - PE_root: Root PE number

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_broadcastmem(shmem_team_t team, void *dest, const void *source,
                           size_t nelems, int PE_root);
    ```
  }];

  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nelems,
    I32:$PE_root
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $nelems `,` "
                       "$PE_root `)` attr-dict `:` type($team) `,` type($dest) "
                       "`,` type($source) `,` type($nelems) `,` type($PE_root) "
                       "`->` type($retval)";
}

//===----------------------------------------------------------------------===//
// BroadcastOp (Typed)
//===----------------------------------------------------------------------===//

def OpenSHMEM_BroadcastOp : OpenSHMEM_Op<"broadcast", []> {
  let summary = "Typed broadcast a value to all PEs in a team";
  let description = [{
    This operation performs a typed broadcast operation on a value to all PEs
    in a team.

    Broadcasts a block of data from one PE to one or more destination PEs.

    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - nelems: Number of elements to transfer
    - PE_root: Root PE number

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_<TYPE>_broadcast(shmem_team_t team, TYPE *dest, const TYPE
                               *source, size_t nelems, int PE_root);
    ```
    where TYPE is one of the standard types (int, float, double, etc.)
  }];

  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nelems,
    I32:$PE_root
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $nelems `,` "
                       "$PE_root `)` attr-dict `:` type($team) `,` type($dest) "
                       "`,` type($source) `,` type($nelems) `,` type($PE_root) "
                       "`->` type($retval)";
}

//===----------------------------------------------------------------------===//
// CollectmemOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_CollectmemOp : OpenSHMEM_Op<"collectmem", []> {
  let summary = "Collect a value from all PEs in a team";
  let description = [{
    This operation performs a collect operation on a value from all PEs in a
    team.

    Concatenates blocks of data from multiple PEs to an array in every PE
    participating in the collective routine.

    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - nelems: Number of elements to transfer

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_collectmem(shmem_team_t team, void *dest, const void *source,
                         size_t nelems);
    ```
  }];

  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nelems
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $nelems `)` "
                       "attr-dict `:` type($team) `,` type($dest) `,` "
                       "type($source) `,` type($nelems) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// CollectOp (Typed)
//===----------------------------------------------------------------------===//

def OpenSHMEM_CollectOp : OpenSHMEM_Op<"collect", []> {
  let summary = "Typed collect a value from all PEs in a team";
  let description = [{
    This operation performs a typed collect operation on a value from all PEs
    in a team.

    Concatenates blocks of data from multiple PEs to an array in every PE
    participating in the collective routine.

    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - nelems: Number of elements to transfer

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_<TYPE>_collect(shmem_team_t team, TYPE *dest, const TYPE
                             *source, size_t nelems);
    ```
    where TYPE is one of the standard types (int, float, double, etc.)
  }];

  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nelems
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $nelems `)` "
                       "attr-dict `:` type($team) `,` type($dest) `,` "
                       "type($source) `,` type($nelems) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// FCollectmemOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_FCollectmemOp : OpenSHMEM_Op<"fcollectmem", []> {
  let summary = "Collect a value from all PEs in a team";
  let description = [{
    This operation performs a fixed-size collect operation on a value from
    all PEs in a team.

    Concatenates blocks of data from multiple PEs to an array in every PE
    participating in the collective routine.

    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - nelems: Number of elements to transfer

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_fcollectmem(shmem_team_t team, void *dest, const void *source,
                          size_t nelems);
    ```
  }];

  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nelems
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $nelems `)` "
                       "attr-dict `:` type($team) `,` type($dest) `,` "
                       "type($source) `,` type($nelems) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// FCollectOp (Typed)
//===----------------------------------------------------------------------===//

def OpenSHMEM_FCollectOp : OpenSHMEM_Op<"fcollect", []> {
  let summary = "Typed fixed-size collect a value from all PEs in a team";
  let description = [{
    This operation performs a typed fixed-size collect operation on a value
    from all PEs in a team.

    Concatenates blocks of data from multiple PEs to an array in every PE
    participating in the collective routine.

    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - nelems: Number of elements to transfer

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_<TYPE>_fcollect(shmem_team_t team, TYPE *dest, const TYPE
                              *source, size_t nelems);
    ```
    where TYPE is one of the standard types (int, float, double, etc.)
  }];

  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nelems
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $nelems `)` "
                       "attr-dict `:` type($team) `,` type($dest) `,` "
                       "type($source) `,` type($nelems) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// AndReduceOP
//===----------------------------------------------------------------------===//

def OpenSHMEM_AndReduceOp : OpenSHMEM_Op<"andreduce", []> {
  let summary = "Perform a bitwise AND reduction";
  let description = [{
    This operation performs a bitwise AND reduction on a value from all PEs
    in a team.

    Performs a bitwise AND reduction on a value from all PEs in a team.

    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - nreduce: Number of elements in the source and dest arrays

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_and_reduce(shmem_team_t team, void *dest, const void *source,
                         size_t nreduce);
    ```
  }];

  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nreduce
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $nreduce `)` "
                       "attr-dict `:` type($team) `,` type($dest) `,` "
                       "type($source) `,` type($nreduce) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// OrReduceOP
//===----------------------------------------------------------------------===//

def OpenSHMEM_OrReduceOp : OpenSHMEM_Op<"orreduce", []> {
  let summary = "Perform a bitwise OR reduction";
  let description = [{
    This operation performs a bitwise OR reduction on a value from all PEs
    in a team.

    Performs a bitwise OR reduction on a value from all PEs in a team.

    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - nreduce: Number of elements in the source and dest arrays

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_or_reduce(shmem_team_t team, void *dest, const void *source,
                        size_t nreduce);
    ```
  }];


  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nreduce
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $nreduce `)` "
                       "attr-dict `:` type($team) `,` type($dest) `,` "
                       "type($source) `,` type($nreduce) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// XorReduceOP
//===----------------------------------------------------------------------===//

def OpenSHMEM_XorReduceOp : OpenSHMEM_Op<"xorreduce", []> {
  let summary = "Perform a bitwise XOR reduction";
  let description = [{
    This operation performs a bitwise XOR reduction on a value from all PEs
    in a team.

    Performs a bitwise XOR reduction on a value from all PEs in a team.

    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - nreduce: Number of elements in the source and dest arrays

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_xor_reduce(shmem_team_t team, void *dest, const void *source,
                         size_t nreduce);
    ```
  }];

  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nreduce
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $nreduce `)` "
                       "attr-dict `:` type($team) `,` type($dest) `,` "
                       "type($source) `,` type($nreduce) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// MaxReduceOP
//===----------------------------------------------------------------------===//

def OpenSHMEM_MaxReduceOp : OpenSHMEM_Op<"maxreduce", []> {
  let summary = "Perform a maximum reduction";
  let description = [{
    This operation performs a maximum reduction on a value from all PEs in a
    team.

    Performs a maximum reduction on a value from all PEs in a team.

    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - nreduce: Number of elements in the source and dest arrays

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_max_reduce(shmem_team_t team, void *dest, const void *source,
                         size_t nreduce);
    ```
  }];

  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nreduce
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $nreduce `)` "
                       "attr-dict `:` type($team) `,` type($dest) `,` "
                       "type($source) `,` type($nreduce) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// MinReduceOP
//===----------------------------------------------------------------------===//

def OpenSHMEM_MinReduceOp : OpenSHMEM_Op<"minreduce", []> {
  let summary = "Perform a minimum reduction";
  let description = [{
    This operation performs a minimum reduction on a value from all PEs in a
    team.

    Performs a minimum reduction on a value from all PEs in a team.


    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - nreduce: Number of elements in the source and dest arrays

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_min_reduce(shmem_team_t team, void *dest, const void *source,
                         size_t nreduce);
    ```
  }];

  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nreduce
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $nreduce `)` "
                       "attr-dict `:` type($team) `,` type($dest) `,` "
                       "type($source) `,` type($nreduce) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// SumReduceOP
//===----------------------------------------------------------------------===//

def OpenSHMEM_SumReduceOp : OpenSHMEM_Op<"sumreduce", []> {
  let summary = "Perform a sum reduction";
  let description = [{
    This operation performs a sum reduction on a value from all PEs in a
    team.

    Performs a sum reduction on a value from all PEs in a team.

    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - nreduce: Number of elements in the source and dest arrays

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_sum_reduce(shmem_team_t team, void *dest, const void *source,
                         size_t nreduce);
    ```
  }];

  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nreduce
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $nreduce `)` "
                       "attr-dict `:` type($team) `,` type($dest) `,` "
                       "type($source) `,` type($nreduce) `->` type($retval)";
}

//===----------------------------------------------------------------------===//
// ProdReduceOP
//===----------------------------------------------------------------------===//

def OpenSHMEM_ProdReduceOp : OpenSHMEM_Op<"prodreduce", []> {
  let summary = "Perform a product reduction";
  let description = [{
    This operation performs a product reduction on a value from all PEs in a
    team.

    Performs a product reduction on a value from all PEs in a team.

    Arguments:
    - team: The team to participate in
    - dest: Destination symmetric memory reference
    - source: Source symmetric memory reference
    - nreduce: Number of elements in the source and dest arrays

    Results:
    - retval: Return value for error checking (0 for success, non-zero for
              failure)

    Signature:
    ```c
    int shmem_prod_reduce(shmem_team_t team, void *dest, const void *source,
                          size_t nreduce);
    ```
  }];

  let arguments = (ins
    OpenSHMEM_Team:$team,
    SymmetricMemRef:$dest,
    SymmetricMemRef:$source,
    Index:$nreduce
  );

  let results = (outs
    I32:$retval
  );

  let assemblyFormat = "`(` $team `,` $dest `,` $source `,` $nreduce `)` "
                       "attr-dict `:` type($team) `,` type($dest) `,` "
                       "type($source) `,` type($nreduce) `->` type($retval)";
}

#endif // MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMCOLLECTIVES_TD 
