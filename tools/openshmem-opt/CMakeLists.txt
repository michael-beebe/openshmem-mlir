add_executable(openshmem-opt
  openshmem-opt.cpp
)

# Link against MLIR driver components and the built OpenSHMEM libraries so
# the dialect and passes are available inside the tool.
mlir_check_all_link_libraries(openshmem-opt)

target_link_libraries(openshmem-opt
  PRIVATE
    MLIRAsmParser
    MLIRParser
    MLIRPass
    MLIRMlirOptMain
    MLIRSupport
    MLIRTransforms
    MLIRIR
    MLIRBytecodeReader
    MLIRBytecodeWriter
    # OpenSHMEM libraries need to come after MLIR libraries so the linker
    # will pull in archive members that define TypeID symbols and other
    # dialect/pass implementations referenced by the tool's objects.
)

# Link the OpenSHMEM dialect and conversion static libraries last to ensure
# their object files are pulled in to resolve references from other objects.


# Some symbols (TypeID initializers and generated helpers) may not be
# pulled in by the linker from static archives unless object files are
# referenced; force inclusion of the OpenSHMEM static archives to avoid
# unresolved references when building the test driver.
# Link the OpenSHMEM libraries (use the CMake targets so transitive
# dependencies and link ordering are handled by CMake). Place them after
# MLIR libraries so their object files are available to resolve symbols.
target_link_libraries(openshmem-opt
  PRIVATE
    MLIROpenSHMEMToLLVM
    MLIROpenSHMEMDialect
    MLIROpenSHMEMCIRPasses
    MLIROpenSHMEMCIRRewriters
    MLIRCIR
)

# Make sure the include path contains generated headers
target_include_directories(openshmem-opt PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cir/include
  ${CMAKE_CURRENT_BINARY_DIR}/../../include
  ${CMAKE_CURRENT_BINARY_DIR}/../../cir/include
)

# Export symbols for plugin capability if needed
if(MLIR_EXPORT_SYMBOLS_FOR_PLUGINS)
  export_executable_symbols_for_plugins(openshmem-opt)
endif()
