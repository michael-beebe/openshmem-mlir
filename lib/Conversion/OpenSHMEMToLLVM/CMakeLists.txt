add_mlir_conversion_library(MLIROpenSHMEMToLLVM
  OpenSHMEMToLLVM.cpp
  OpenSHMEMConversionUtils.cpp
  SetupOpsToLLVM.cpp
  MemoryOpsToLLVM.cpp
  RMAOpsToLLVM.cpp
  CollectiveOpsToLLVM.cpp
  TeamOpsToLLVM.cpp
  ContextOpsToLLVM.cpp
  AtomicOpsToLLVM.cpp
  SyncOpsToLLVM.cpp
  Pt2ptSyncOpsToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/OpenSHMEMToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIROpenSHMEMDialect
  MLIRPass
  MLIRTransforms
) 
