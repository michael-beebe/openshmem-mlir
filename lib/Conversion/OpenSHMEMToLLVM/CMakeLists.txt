add_mlir_library(MLIROpenSHMEMToLLVM
  OpenSHMEMToLLVM.cpp
  OpenSHMEMConversionUtils.cpp
  SetupOpsToLLVM.cpp
  MemoryOpsToLLVM.cpp
  RMAOpsToLLVM.cpp
  AtomicOpsToLLVM.cpp
  SyncOpsToLLVM.cpp
  Pt2ptSyncOpsToLLVM.cpp
  CollectiveOpsToLLVM.cpp
  TeamOpsToLLVM.cpp
  ContextOpsToLLVM.cpp

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  MLIRIR
  MLIROpenSHMEM

  DEPENDS
  MLIROpenSHMEMConvertPassesIncGen
)

target_include_directories(MLIROpenSHMEMToLLVM
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(MLIROpenSHMEMToLLVM
  PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

