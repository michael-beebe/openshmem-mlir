# OpenSHMEM ClangIR Frontend

add_subdirectory(include)
add_subdirectory(lib)
if(MLIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(test)
endif()
