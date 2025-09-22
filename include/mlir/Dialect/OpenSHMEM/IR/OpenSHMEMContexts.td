//===- OpenSHMEMContexts.td - OpenSHMEM context operations -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMCONTEXTS_TD
#define MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMCONTEXTS_TD

include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMBase.td"

// TODO: probably need an op to initialize the default context

//===----------------------------------------------------------------------===//
// CreateCtxOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxCreateOp : OpenSHMEM_Op<"ctx_create", []> {
  let summary = "Equivalent to the shmem_ctx_create C API";
  let description = [{
    Creates a new OpenSHMEM communication context with the specified
    options. Contexts provide ordered and independent communication
    operations.

    Arguments:
    - options: Set of options for the context (bitwise OR combination)

    Results:
    - ctx: Handle to the newly created context
    - status: Status code (0 for success, non-zero for failure)

    ```c
    int shmem_ctx_create(long options, shmem_ctx_t *ctx);
    ```
  }];
  let arguments = (ins I64:$options);
  let results = (outs OpenSHMEM_Ctx:$ctx, I32:$status);
  let assemblyFormat = "`(` $options `)` attr-dict `:` "
                       "type($options) `->` type($ctx) `,` type($status)";
}

//===----------------------------------------------------------------------===//
// TeamCreateCtxOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_TeamCreateCtxOp : OpenSHMEM_Op<"team_create_ctx", []> {
  let summary = "Equivalent to the shmem_team_create_ctx C API";
  let description = [{
    Creates a new OpenSHMEM communication context associated with the
    specified team and options.

    Arguments:
    - team: Handle to the specified PE team
    - options: Set of options for the context (bitwise OR combination)

    Results:
    - ctx: Handle to the newly created context
    - status: Status code (0 for success, non-zero for failure)

    ```c
    int shmem_team_create_ctx(shmem_team_t team, long options,
                              shmem_ctx_t *ctx);
    ```
  }];
  let arguments = (ins OpenSHMEM_Team:$team, I64:$options);
  let results = (outs OpenSHMEM_Ctx:$ctx, I32:$status);
  let assemblyFormat = "`(` $team `,` $options `)` attr-dict `:` "
                       "type($team) `,` type($options) `->` type($ctx) `,`"
                       " type($status)";
}

//===----------------------------------------------------------------------===//
// DestroyCtxOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxDestroyOp : OpenSHMEM_Op<"ctx_destroy", []> {
  let summary = "Equivalent to the shmem_ctx_destroy C API";
  let description = [{
    Destroys an OpenSHMEM communication context that was previously
    created. After destruction, the context handle becomes invalid.

    Arguments:
    - ctx: Handle to the context to destroy

    ```c
    void shmem_ctx_destroy(shmem_ctx_t ctx);
    ```
  }];
  let arguments = (ins OpenSHMEM_Ctx:$ctx);
  let assemblyFormat = "`(` $ctx `)` attr-dict `:` type($ctx)";
}

//===----------------------------------------------------------------------===//
// CtxGetTeamOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_CtxGetTeamOp : OpenSHMEM_Op<"ctx_get_team", []> {
  let summary = "Equivalent to the shmem_ctx_get_team C API";
  let description = [{
    Retrieves the team associated with the specified communication context.

    Arguments:
    - ctx: Handle to the communication context

    Results:
    - team: Handle to the associated PE team
    - status: Status code (0 for success, non-zero for failure)

    ```c
    int shmem_ctx_get_team(shmem_ctx_t ctx, shmem_team_t *team);
    ```
  }];
  let arguments = (ins OpenSHMEM_Ctx:$ctx);
  let results = (outs OpenSHMEM_Team:$team, I32:$status);
  let assemblyFormat = "`(` $ctx `)` attr-dict `:` "
                       "type($ctx) `->` type($team) `,` type($status)";
}

#endif // MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMCONTEXTS_TD 
