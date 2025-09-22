//===- OpenSHMEMTeams.td - OpenSHMEM team operations -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMTEAMS_TD
#define MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMTEAMS_TD

include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Team Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// shmem_team_config_t (dialect attribute)
//===----------------------------------------------------------------------===//

def OpenSHMEM_TeamConfigAttr : AttrDef<OpenSHMEM_Dialect, "TeamConfig"> {
  let mnemonic = "team_config";
  let summary = "Configuration for team creation (subset of shmem_team_config_t)";
  let description = [{
    Represents configuration parameters for team creation and split operations.
    Currently models the "num_contexts" field from the OpenSHMEM spec. This
    attribute can be attached to team split/create operations to guide runtime
    behavior and compiler heuristics.
  }];

  let parameters = (ins
    "int32_t":$numContexts
  );

  let assemblyFormat = "`<` `num_contexts` `=` $numContexts `>`";
}

//===----------------------------------------------------------------------===//
// TeamSplitStridedOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_TeamSplitStridedOp : OpenSHMEM_Op<"team_split_strided", []> {
  let summary = "Equivalent to the shmem_team_split_strided C API";
  let description = [{
    Splits a parent team into multiple subteams based on a strided pattern.
    PEs are assigned to subteams using the specified start, stride, and size
    parameters.

    Arguments:
    - parent_team: The parent team to split
    - start: Starting PE number in the parent team
    - stride: Stride for PE assignment
    - size: Number of PEs in each subteam

    Results:
    - new_team: The new team that the calling PE belongs to
    - retval: Return value (0 for success, non-zero for failure)

    ```c
    int shmem_team_split_strided(shmem_team_t parent_team, int start,
                                 int stride, int size,
                                 const shmem_team_config_t *config,
                                 long config_mask, shmem_team_t *new_team);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Team:$parent_team,
    I32:$start,
    I32:$stride, 
    I32:$size
  );
  
  let results = (outs 
    OpenSHMEM_Team:$new_team,
    I32:$retval
  );
  
  let assemblyFormat = "`(` $parent_team `,` $start `,` $stride `,` $size `)`"
                       " attr-dict `:` "
                       "type($parent_team) `,` type($start) `,` type($stride) `,`"
                       " type($size) `->` type($new_team) `,` type($retval)";
}

//===----------------------------------------------------------------------===//
// TeamSplit2dOp  
//===----------------------------------------------------------------------===//

def OpenSHMEM_TeamSplit2dOp : OpenSHMEM_Op<"team_split_2d", []> {
  let summary = "Equivalent to the shmem_team_split_2d C API";
  let description = [{
    Splits a parent team into two subteams organized in a 2D grid pattern.
    Creates one team containing PEs in the same row and another team
    containing PEs in the same column.

    Arguments:
    - parent_team: The parent team to split
    - xrange: Number of teams in the x dimension

    Results:
    - xaxis_team: Team containing PEs in the same row
    - yaxis_team: Team containing PEs in the same column
    - retval: Return value (0 for success, non-zero for failure)

    ```c
    int shmem_team_split_2d(shmem_team_t parent_team, int xrange,
                            const shmem_team_config_t *xaxis_config,
                            long xaxis_mask, shmem_team_t *xaxis_team,
                            const shmem_team_config_t *yaxis_config,
                            long yaxis_mask, shmem_team_t *yaxis_team);
    ```
  }];

  let arguments = (ins 
    OpenSHMEM_Team:$parent_team,
    I32:$xrange
  );
  
  let results = (outs 
    OpenSHMEM_Team:$xaxis_team,
    OpenSHMEM_Team:$yaxis_team,
    I32:$retval
  );
  
  let assemblyFormat = "`(` $parent_team `,` $xrange `)` attr-dict `:` "
                       "type($parent_team) `,` type($xrange) `->` "
                       "type($xaxis_team) `,` type($yaxis_team) `,` type($retval)";
}

//===----------------------------------------------------------------------===//
// TeamMyPeOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_TeamMyPeOp : OpenSHMEM_Op<"team_my_pe", []> {
  let summary = "Equivalent to the shmem_team_my_pe C API";
  let description = [{
    Returns the PE number of the calling PE within the specified team.
    PE numbers within a team range from 0 to team_n_pes() - 1.

    Arguments:
    - team: The team to query

    Results:
    - pe: PE number within the team

    ```c
    int shmem_team_my_pe(shmem_team_t team);
    ```
  }];

  let arguments = (ins OpenSHMEM_Team:$team);
  let results = (outs I32:$pe);
  let assemblyFormat = "`(` $team `)` attr-dict `:` type($team) `->` type($pe)";
}

//===----------------------------------------------------------------------===//
// TeamNPesOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_TeamNPesOp : OpenSHMEM_Op<"team_n_pes", []> {
  let summary = "Equivalent to the shmem_team_n_pes C API";
  let description = [{
    Returns the total number of PEs in the specified team.

    Arguments:
    - team: The team to query

    Results:
    - n_pes: Number of PEs in the team

    ```c
    int shmem_team_n_pes(shmem_team_t team);
    ```
  }];

  let arguments = (ins OpenSHMEM_Team:$team);
  let results = (outs I32:$n_pes);
  let assemblyFormat = "`(` $team `)` attr-dict `:` type($team) `->` type($n_pes)";
}

//===----------------------------------------------------------------------===//
// TeamSyncOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_TeamSyncOp : OpenSHMEM_Op<"team_sync", []> {
  let summary = "Equivalent to the shmem_team_sync C API";
  let description = [{
    Performs a barrier synchronization on all PEs in the specified team.
    All PEs in the team must call this operation before any PE can continue.

    Arguments:
    - team: The team to synchronize

    ```c
    void shmem_team_sync(shmem_team_t team);
    ```
  }];

  let arguments = (ins OpenSHMEM_Team:$team);
  let assemblyFormat = "`(` $team `)` attr-dict `:` type($team)";
}

//===----------------------------------------------------------------------===//
// TeamDestroyOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_TeamDestroyOp : OpenSHMEM_Op<"team_destroy", []> {
  let summary = "Equivalent to the shmem_team_destroy C API";
  let description = [{
    Destroys a team that was previously created by a team split operation.
    After destruction, the team handle becomes invalid and should not be
    used.

    Arguments:
    - team: The team to destroy

    ```c
    void shmem_team_destroy(shmem_team_t team);
    ```
  }];

  let arguments = (ins OpenSHMEM_Team:$team);
  let assemblyFormat = "`(` $team `)` attr-dict `:` type($team)";
}

//===----------------------------------------------------------------------===//
// Predefined Teams
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TeamWorldOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_TeamWorldOp : OpenSHMEM_Op<"team_world", []> {
  let summary = "Equivalent to the SHMEM_TEAM_WORLD constant";
  let description = [{
    Returns a reference to the predefined world team that contains all PEs
    in the OpenSHMEM program.

    Arguments:
    - (none)

    Results:
    - team: The world team

    ```c
    shmem_team_t team = SHMEM_TEAM_WORLD;
    ```
  }];

  let results = (outs OpenSHMEM_Team:$team);
  let assemblyFormat = "attr-dict `->` type($team)";
}

//===----------------------------------------------------------------------===//
// TeamSharedOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_TeamSharedOp : OpenSHMEM_Op<"team_shared", []> {
  let summary = "Equivalent to the SHMEM_TEAM_SHARED constant";
  let description = [{
    Returns a reference to the predefined shared memory team. This team
    contains PEs that share memory and can perform optimized communication.

    Arguments:
    - (none)

    Results:
    - team: The shared memory team

    ```c
    shmem_team_t team = SHMEM_TEAM_SHARED;
    ```
  }];

  let results = (outs OpenSHMEM_Team:$team);
  let assemblyFormat = "attr-dict `->` type($team)";
}

#endif // MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMTEAMS_TD 
