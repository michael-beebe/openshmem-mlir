#!/bin/bash

# Find all .h and .cpp files in the project directory, excluding the llvm-project directory, and run clang-format on them
find . -path ./llvm-project -prune -o \( -name "*.h" -o -name "*.cpp" \) -print | while read -r file; do
  clang-format -i -style=LLVM "$file"
done
