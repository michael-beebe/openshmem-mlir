# For now, just add the subdirectories but don't set up lit tests
# since the LLVM test tools (FileCheck, etc.) are not available in out-of-tree builds
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(EndToEnd)