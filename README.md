# mlirPyoclExec
Examples enabling OpenCL in MLIR via Python.

- Current available target : AMD GPU
- Requirement
  - MLIR with python binding https://mlir.llvm.org/docs/Bindings/Python/
    - Also need to copy ./bin/mlir-opt to the same path you run the test.
  - pyopencl
  - numpy
  - rocm OpenCL
