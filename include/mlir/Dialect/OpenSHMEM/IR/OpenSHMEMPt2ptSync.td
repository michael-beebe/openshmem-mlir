//===- OpenSHMEMPt2ptSync.td - OpenSHMEM point-to-point synchronization operations -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMPT2PTSYNC_TD
#define MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMPT2PTSYNC_TD

include "mlir/Dialect/OpenSHMEM/IR/OpenSHMEMBase.td"

//===----------------------------------------------------------------------===//
// WaitUntilOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_WaitUntilOp : OpenSHMEM_Op<"wait_until", []> {
  let summary = "Equivalent to the shmem_wait_until C API";

  let description = [{
    Blocks until a symmetric memory variable on the local PE satisfies the
    specified wait condition.

    Arguments:
    - ivar: Symmetric memory variable to wait on
    - cmp: Comparison operator
    - cmp_value: Value to compare against

    ```c
    void shmem_wait_until(TYPE *ivar, int cmp, TYPE cmp_value);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$ivar,
    I32:$cmp,
    AnyType:$cmp_value
  );

  let assemblyFormat = "`(` $ivar `,` $cmp `,` $cmp_value `)` attr-dict `:` "
                       "type($ivar) `,` type($cmp) `,` type($cmp_value)";
}

//===----------------------------------------------------------------------===//
// WaitUntilAllOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_WaitUntilAllOp : OpenSHMEM_Op<"wait_until_all", []> {
  let summary = "Equivalent to the shmem_wait_until_all C API";

  let description = [{
    Blocks until all variables in an array of symmetric memory variables
    on the local PE satisfy the specified wait condition.

    Arguments:
    - ivars: Array of symmetric memory variables to wait on
    - nelems: Number of elements in the array
    - status: Status array indicating which elements to check
    - cmp: Comparison operator
    - cmp_value: Value to compare against

    ```c
    void shmem_wait_until_all(TYPE *ivars, size_t nelems, const int *status,
                              int cmp, TYPE cmp_value);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$ivars,
    Index:$nelems,
    AnyMemRef:$status,
    I32:$cmp,
    AnyType:$cmp_value
  );

  let assemblyFormat = "`(` $ivars `,` $nelems `,` $status `,` $cmp `,`"
                       " $cmp_value `)` attr-dict `:` "
                       "type($ivars) `,` type($nelems) `,` type($status) `,`"
                       " type($cmp) `,` type($cmp_value)";
}

//===----------------------------------------------------------------------===//
// WaitUntilAnyOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_WaitUntilAnyOp : OpenSHMEM_Op<"wait_until_any", []> {
  let summary = "Equivalent to the shmem_wait_until_any C API";

  let description = [{
    Blocks until any variable in an array of symmetric memory variables
    on the local PE satisfies the specified wait condition.

    Arguments:
    - ivars: Array of symmetric memory variables to wait on
    - nelems: Number of elements in the array
    - status: Status array indicating which elements to check
    - cmp: Comparison operator
    - cmp_value: Value to compare against

    ```c
    size_t shmem_wait_until_any(TYPE *ivars, size_t nelems,
                                const int *status, int cmp, TYPE cmp_value);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$ivars,
    Index:$nelems,
    AnyMemRef:$status,
    I32:$cmp,
    AnyType:$cmp_value
  );

  let assemblyFormat = "`(` $ivars `,` $nelems `,` $status `,` $cmp `,`"
                       " $cmp_value `)` attr-dict `:` "
                       "type($ivars) `,` type($nelems) `,` type($status) `,`"
                       " type($cmp) `,` type($cmp_value)";
}

//===----------------------------------------------------------------------===//
// WaitUntilSomeOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_WaitUntilSomeOp : OpenSHMEM_Op<"wait_until_some", []> {
  let summary = "Equivalent to the shmem_wait_until_some C API";

  let description = [{
    Blocks until at least one variable in an array of symmetric memory
    variables on the local PE satisfies the specified wait condition.

    Arguments:
    - ivars: Array of symmetric memory variables to wait on
    - nelems: Number of elements in the array
    - indices: Array to store indices of satisfied variables
    - status: Status array indicating which elements to check
    - cmp: Comparison operator
    - cmp_value: Value to compare against

    Results:
    - result: Number of variables that satisfied the condition

    ```c
    size_t shmem_wait_until_some(TYPE *ivars, size_t nelems, size_t *indices,
                                 const int *status, int cmp, TYPE cmp_value);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$ivars,
    Index:$nelems,
    AnyMemRef:$indices,
    AnyMemRef:$status,
    I32:$cmp,
    AnyType:$cmp_value
  );

  let results = (outs Index:$result);

  let assemblyFormat = "`(` $ivars `,` $nelems `,` $indices `,` $status `,`"
                       " $cmp `,` $cmp_value `)` attr-dict `:` "
                       "type($ivars) `,` type($nelems) `,` type($indices) `,`"
                       " type($status) `,` type($cmp) `,` type($cmp_value) `->`"
                       " type($result)";
}

//===----------------------------------------------------------------------===//
// WaitUntilAllVectorOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_WaitUntilAllVectorOp : OpenSHMEM_Op<"wait_until_all_vector", []> {
  let summary = "Equivalent to the shmem_wait_until_all_vector C API";

  let description = [{
    Blocks until all variables in an array of symmetric memory variables
    on the local PE satisfy their corresponding wait conditions.

    Arguments:
    - ivars: Array of symmetric memory variables to wait on
    - nelems: Number of elements in the array
    - status: Status array indicating which elements to check
    - cmp: Comparison operator
    - cmp_values: Array of values to compare against

    ```c
    void shmem_wait_until_all_vector(TYPE *ivars, size_t nelems,
                                     const int *status, int cmp,
                                     TYPE *cmp_values);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$ivars,
    Index:$nelems,
    AnyMemRef:$status,
    I32:$cmp,
    AnyMemRef:$cmp_values
  );

  let assemblyFormat = "`(` $ivars `,` $nelems `,` $status `,` $cmp `,`"
                       " $cmp_values `)` attr-dict `:` "
                       "type($ivars) `,` type($nelems) `,` type($status) `,`"
                       " type($cmp) `,` type($cmp_values)";
}

//===----------------------------------------------------------------------===//
// WaitUntilAnyVectorOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_WaitUntilAnyVectorOp : OpenSHMEM_Op<"wait_until_any_vector", []> {
  let summary = "Equivalent to the shmem_wait_until_any_vector C API";

  let description = [{
    Blocks until any variable in an array of symmetric memory variables
    on the local PE satisfies its corresponding wait condition.

    Arguments:
    - ivars: Array of symmetric memory variables to wait on
    - nelems: Number of elements in the array
    - status: Status array indicating which elements to check
    - cmp: Comparison operator
    - cmp_values: Array of values to compare against

    ```c
    size_t shmem_wait_until_any_vector(TYPE *ivars, size_t nelems,
                                       const int *status, int cmp,
                                       TYPE *cmp_values);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$ivars,
    Index:$nelems,
    AnyMemRef:$status,
    I32:$cmp,
    AnyMemRef:$cmp_values
  );

  let assemblyFormat = "`(` $ivars `,` $nelems `,` $status `,` $cmp `,`"
                       " $cmp_values `)` attr-dict `:` "
                       "type($ivars) `,` type($nelems) `,` type($status) `,`"
                       " type($cmp) `,` type($cmp_values)";
}

//===----------------------------------------------------------------------===//
// WaitUntilSomeVectorOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_WaitUntilSomeVectorOp : OpenSHMEM_Op<"wait_until_some_vector", []> {
  let summary = "Equivalent to the shmem_wait_until_some_vector C API";

  let description = [{
    Blocks until at least one variable in an array of symmetric memory
    variables on the local PE satisfies its corresponding wait condition.

    Arguments:
    - ivars: Array of symmetric memory variables to wait on
    - nelems: Number of elements in the array
    - indices: Array to store indices of satisfied variables
    - status: Status array indicating which elements to check
    - cmp: Comparison operator
    - cmp_values: Array of values to compare against

    Results:
    - result: Number of variables that satisfied the condition

    ```c
    size_t shmem_wait_until_some_vector(TYPE *ivars, size_t nelems, size_t *indices,
                                        const int *status, int cmp, TYPE *cmp_values);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$ivars,
    Index:$nelems,
    AnyMemRef:$indices,
    AnyMemRef:$status,
    I32:$cmp,
    AnyMemRef:$cmp_values
  );

  let results = (outs Index:$result);

  let assemblyFormat = "`(` $ivars `,` $nelems `,` $indices `,` $status `,`"
                       " $cmp `,` $cmp_values `)` attr-dict `:` "
                       "type($ivars) `,` type($nelems) `,` type($indices) `,`"
                       " type($status) `,` type($cmp) `,` type($cmp_values) `->`"
                       " type($result)";
}

//===----------------------------------------------------------------------===//
// TestOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_TestOp : OpenSHMEM_Op<"test", []> {
  let summary = "Equivalent to the shmem_test C API";

  let description = [{
    Tests whether a symmetric memory variable on the local PE satisfies
    the specified condition without blocking.

    Arguments:
    - ivar: Symmetric memory variable to test
    - cmp: Comparison operator
    - cmp_value: Value to compare against

    Results:
    - result: Non-zero if condition is satisfied, zero otherwise

    ```c
    int shmem_test(TYPE *ivar, int cmp, TYPE cmp_value);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$ivar,
    I32:$cmp,
    AnyType:$cmp_value
  );

  let results = (outs I32:$result);

  let assemblyFormat = "`(` $ivar `,` $cmp `,` $cmp_value `)` attr-dict `:` "
                       "type($ivar) `,` type($cmp) `,` type($cmp_value) `->`"
                       " type($result)";
}

//===----------------------------------------------------------------------===//
// TestAllOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_TestAllOp : OpenSHMEM_Op<"test_all", []> {
  let summary = "Equivalent to the shmem_test_all C API";

  let description = [{
    Tests whether all variables in an array of symmetric memory variables
    on the local PE satisfy the specified condition without blocking.

    Arguments:
    - ivars: Array of symmetric memory variables to test
    - nelems: Number of elements in the array
    - status: Status array indicating which elements to check
    - cmp: Comparison operator
    - cmp_value: Value to compare against

    Results:
    - result: Non-zero if all conditions are satisfied, zero otherwise

    ```c
    int shmem_test_all(TYPE *ivars, size_t nelems, const int *status,
                       int cmp, TYPE cmp_value);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$ivars,
    Index:$nelems,
    AnyMemRef:$status,
    I32:$cmp,
    AnyType:$cmp_value
  );

  let results = (outs I32:$result);

  let assemblyFormat = "`(` $ivars `,` $nelems `,` $status `,` $cmp `,`"
                       " $cmp_value `)` attr-dict `:` "
                       "type($ivars) `,` type($nelems) `,` type($status) `,`"
                       " type($cmp) `,` type($cmp_value) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// TestAnyOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_TestAnyOp : OpenSHMEM_Op<"test_any", []> {
  let summary = "Equivalent to the shmem_test_any C API";

  let description = [{
    Tests whether any variable in an array of symmetric memory variables
    on the local PE satisfies the specified condition without blocking.

    Arguments:
    - ivars: Array of symmetric memory variables to test
    - nelems: Number of elements in the array
    - status: Status array indicating which elements to check
    - cmp: Comparison operator
    - cmp_value: Value to compare against

    Results:
    - result: Index of first satisfied variable, SIZE_MAX if none

    ```c
    size_t shmem_test_any(TYPE *ivars, size_t nelems, const int *status,
                          int cmp, TYPE cmp_value);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$ivars,
    Index:$nelems,
    AnyMemRef:$status,
    I32:$cmp,
    AnyType:$cmp_value
  );

  let results = (outs Index:$result);

  let assemblyFormat = "`(` $ivars `,` $nelems `,` $status `,` $cmp `,`"
                       " $cmp_value `)` attr-dict `:` "
                       "type($ivars) `,` type($nelems) `,` type($status) `,`"
                       " type($cmp) `,` type($cmp_value) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// TestSomeOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_TestSomeOp : OpenSHMEM_Op<"test_some", []> {
  let summary = "Equivalent to the shmem_test_some C API";

  let description = [{
    Tests whether at least one variable in an array of symmetric memory
    variables on the local PE satisfies the specified condition without
    blocking.

    Arguments:
    - ivars: Array of symmetric memory variables to test
    - nelems: Number of elements in the array
    - indices: Array to store indices of satisfied variables
    - status: Status array indicating which elements to check
    - cmp: Comparison operator
    - cmp_value: Value to compare against

    Results:
    - result: Number of variables that satisfied the condition

    ```c
    size_t shmem_test_some(TYPE *ivars, size_t nelems, size_t *indices,
                           const int *status, int cmp, TYPE cmp_value);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$ivars,
    Index:$nelems,
    AnyMemRef:$indices,
    AnyMemRef:$status,
    I32:$cmp,
    AnyType:$cmp_value
  );

  let results = (outs Index:$result);

  let assemblyFormat = "`(` $ivars `,` $nelems `,` $indices `,` $status `,`"
                       " $cmp `,` $cmp_value `)` attr-dict `:` "
                       "type($ivars) `,` type($nelems) `,` type($indices) `,`"
                       " type($status) `,` type($cmp) `,` type($cmp_value) `->`"
                       " type($result)";
}

//===----------------------------------------------------------------------===//
// TestAllVectorOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_TestAllVectorOp : OpenSHMEM_Op<"test_all_vector", []> {
  let summary = "Equivalent to the shmem_test_all_vector C API";

  let description = [{
    Tests whether all variables in an array of symmetric memory variables
    on the local PE satisfy their corresponding conditions without blocking.

    Arguments:
    - ivars: Array of symmetric memory variables to test
    - nelems: Number of elements in the array
    - status: Status array indicating which elements to check
    - cmp: Comparison operator
    - cmp_values: Array of values to compare against

    Results:
    - result: Non-zero if all conditions are satisfied, zero otherwise

    ```c
    int shmem_test_all_vector(TYPE *ivars, size_t nelems, const int *status,
                              int cmp, TYPE *cmp_values);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$ivars,
    Index:$nelems,
    AnyMemRef:$status,
    I32:$cmp,
    AnyMemRef:$cmp_values
  );

  let results = (outs I32:$result);

  let assemblyFormat = "`(` $ivars `,` $nelems `,` $status `,` $cmp `,`"
                       " $cmp_values `)` attr-dict `:` "
                       "type($ivars) `,` type($nelems) `,` type($status) `,`"
                       " type($cmp) `,` type($cmp_values) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// TestAnyVectorOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_TestAnyVectorOp : OpenSHMEM_Op<"test_any_vector", []> {
  let summary = "Equivalent to the shmem_test_any_vector C API";

  let description = [{
    Tests whether any variable in an array of symmetric memory variables
    on the local PE satisfies its corresponding condition without blocking.

    Arguments:
    - ivars: Array of symmetric memory variables to test
    - nelems: Number of elements in the array
    - status: Status array indicating which elements to check
    - cmp: Comparison operator
    - cmp_values: Array of values to compare against

    Results:
    - result: Index of first satisfied variable, SIZE_MAX if none

    ```c
    size_t shmem_test_any_vector(TYPE *ivars, size_t nelems,
                                 const int *status, int cmp,
                                 TYPE *cmp_values);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$ivars,
    Index:$nelems,
    AnyMemRef:$status,
    I32:$cmp,
    AnyMemRef:$cmp_values
  );

  let results = (outs Index:$result);

  let assemblyFormat = "`(` $ivars `,` $nelems `,` $status `,` $cmp `,`"
                       " $cmp_values `)` attr-dict `:` "
                       "type($ivars) `,` type($nelems) `,` type($status) `,`"
                       " type($cmp) `,` type($cmp_values) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// TestSomeVectorOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_TestSomeVectorOp : OpenSHMEM_Op<"test_some_vector", []> {
  let summary = "Equivalent to the shmem_test_some_vector C API";

  let description = [{
    Tests whether at least one variable in an array of symmetric memory
    variables on the local PE satisfies its corresponding condition without
    blocking.

    Arguments:
    - ivars: Array of symmetric memory variables to test
    - nelems: Number of elements in the array
    - indices: Array to store indices of satisfied variables
    - status: Status array indicating which elements to check
    - cmp: Comparison operator
    - cmp_values: Array of values to compare against

    Results:
    - result: Number of variables that satisfied the condition

    ```c
    size_t shmem_test_some_vector(TYPE *ivars, size_t nelems, size_t *indices,
                                  const int *status, int cmp,
                                  TYPE *cmp_values);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$ivars,
    Index:$nelems,
    AnyMemRef:$indices,
    AnyMemRef:$status,
    I32:$cmp,
    AnyMemRef:$cmp_values
  );

  let results = (outs Index:$result);

  let assemblyFormat = "`(` $ivars `,` $nelems `,` $indices `,` $status `,`"
                       " $cmp `,` $cmp_values `)` attr-dict `:` "
                       "type($ivars) `,` type($nelems) `,` type($indices) `,`"
                       " type($status) `,` type($cmp) `,` type($cmp_values) `->`"
                       " type($result)";
}

//===----------------------------------------------------------------------===//
// SignalWaitUntilOp
//===----------------------------------------------------------------------===//

def OpenSHMEM_SignalWaitUntilOp : OpenSHMEM_Op<"signal_wait_until", []> {
  let summary = "Equivalent to the shmem_signal_wait_until C API";

  let description = [{
    Blocks until a signal on the local PE satisfies the specified wait
    condition and returns the signal value.

    Arguments:
    - sig_addr: Symmetric memory address of the signal
    - cmp: Comparison operator
    - cmp_value: Value to compare against

    Results:
    - result: Value of the signal when condition was satisfied

    ```c
    uint64_t shmem_signal_wait_until(uint64_t *sig_addr, int cmp,
                                     uint64_t cmp_value);
    ```
  }];

  let arguments = (ins 
    SymmetricMemRef:$sig_addr,
    I32:$cmp,
    I64:$cmp_value
  );

  let results = (outs I64:$result);

  let assemblyFormat = "`(` $sig_addr `,` $cmp `,` $cmp_value `)` attr-dict `:` "
                       "type($sig_addr) `,` type($cmp) `,` type($cmp_value) `->`"
                       " type($result)";
}

#endif // MLIR_DIALECT_OPENSHMEM_IR_OPENSHMEMPT2PTSYNC_TD
