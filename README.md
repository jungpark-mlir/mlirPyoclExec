# mlirPyoclExec
Examples enabling OpenCL in MLIR via Python.

- Current available target : AMD GPU
- Requirement
  - MLIR with python binding https://mlir.llvm.org/docs/Bindings/Python/
    - Also need to copy ./bin/mlir-opt to the same path you run the test : not any more - fixed in MLIR and example is updated.
  - pyopencl
  - numpy
  - rocm OpenCL

- Planned work
  - Intergrate with host python callback per mgpu-* functions : Done, will be released soon. Stay tuned!
  - SPIRV path
