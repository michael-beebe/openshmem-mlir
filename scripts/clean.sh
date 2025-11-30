#!/bin/bash

clean-incubator() {
    if [ -d build-incubator ]; then
        rm -rf build-incubator
        echo "Removed build-incubator directory."
    else
        echo "build-incubator directory does not exist."
    fi
}

clean-upstream() {
    if [ -d build-upstream ]; then
        rm -rf build-upstream
        echo "Removed build-upstream directory."
    else
        echo "build-upstream directory does not exist."
    fi
}

clean-clangir() {
    if [ -d build-clangir ]; then
        rm -rf build-clangir
        echo "Removed build-clangir directory."
    else
        echo "build-clangir directory does not exist."
    fi
}

clean-llvm-project() {
    if [ -d build-llvm-project ]; then
        rm -rf build-llvm-project
        echo "Removed build-llvm-project directory."
    else
        echo "build-llvm-project directory does not exist."
    fi
}

clean-openshmem-runtime() {
    if [ -d build-openshmem-runtime ]; then
        rm -rf build-openshmem-runtime
        echo "Removed build-openshmem-runtime directory."
    else
        echo "build-openshmem-runtime directory does not exist."
    fi
}

clean-all() {
    clean-incubator
    clean-upstream
    clean-clangir
    clean-llvm-project
    clean-openshmem-runtime
    echo "All build directories have been cleaned."
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 [incubator|upstream|clangir|llvm-project|openshmem-runtime|all]"
    echo "  incubator          - Clean build-incubator directory"
    echo "  upstream           - Clean build-upstream directory"
    echo "  clangir            - Clean build-clangir directory"
    echo "  llvm-project       - Clean build-llvm-project directory"
    echo "  openshmem-runtime  - Clean build-openshmem-runtime directory"
    echo "  all                - Clean all build directories"
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Error: You must specify an option."
    echo "Usage: $0 [incubator|upstream|clangir|llvm-project|openshmem-runtime|all]"
    echo "  incubator          - Clean build-incubator directory"
    echo "  upstream           - Clean build-upstream directory"
    echo "  clangir            - Clean build-clangir directory"
    echo "  llvm-project       - Clean build-llvm-project directory"
    echo "  openshmem-runtime  - Clean build-openshmem-runtime directory"
    echo "  all                - Clean all build directories"
    exit 1
fi

case "$1" in
    incubator)
        clean-incubator
        ;;
    upstream)
        clean-upstream
        ;;
    clangir)
        clean-clangir
        ;;
    llvm-project)
        clean-llvm-project
        ;;
    openshmem-runtime)
        clean-openshmem-runtime
        ;;
    all)
        clean-all
        ;;
    *)
        echo "Error: Unknown option '$1'"
        echo "Usage: $0 [incubator|upstream|clangir|llvm-project|openshmem-runtime|all]"
        exit 1
        ;;
esac
