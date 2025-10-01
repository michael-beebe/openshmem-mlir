if(MLIR_ENABLE_BINDINGS_PYTHON)
  add_lit_testsuite(check-openshmem-cir "Running the OpenSHMEM CIR regression tests"
    ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS openshmem-opt
  )
  set_target_properties(check-openshmem-cir PROPERTIES FOLDER "OpenSHMEM CIR tests")
endif()
