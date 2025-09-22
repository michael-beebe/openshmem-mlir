//===- Passes.td - OpenSHMEM dialect passes ----------------*- tablegen -*-===//

include "mlir/Pass/PassBase.td"

// Define a minimal placeholder pass so pass declarations can be generated.
def OpenSHMEMNoOpPass : Pass<"openshmem-noop", "ModuleOp"> {
  let summary = "No-op pass placeholder for OpenSHMEM dialect";
  let constructor = "mlir::openshmem::createOpenSHMEMNoOpPass()";
}

