from mlir.ir import *
import mlir.dialects.builtin as builtin
import mlir.dialects.func as func
import mlir.dialects.linalg as linalg
from mlir.dialects.linalg.opdsl.lang import *

from mlir.passmanager import *

import numpy as np
import subprocess as sp
import pyopencl as cl

ctx = Context()

# =============================================================
# Creating a module performs elementwise addition using linalg
# =============================================================
def testMemrefAdd():
    with ctx, Location.unknown() as loc:
        m = builtin.ModuleOp()
        f32 = F32Type.get()

        with InsertionPoint(m.body):
            @func.FuncOp.from_py_func(
                MemRefType.get([1024], f32),
                MemRefType.get([1024], f32),
                MemRefType.get([1024], f32)
            )
            def vadd(bufA, bufB, outBuf):
                result = linalg.elemwise_binary(bufA, bufB, outs=[outBuf], fun=BinaryFn.add)
                return result
    return m
m0 = testMemrefAdd()
#print(m0)

# =============================================================
# An example pipeline lowering linalg ops to the gpu dialect.
# =============================================================
def gpu_frontend(module):
    with ctx:
        pm = PassManager("any")
        pm.add("convert-linalg-to-parallel-loops")
        pm.add("func.func(gpu-map-parallel-loops)")
        pm.add("func.func(convert-parallel-loops-to-gpu)")
        pm.add("convert-scf-to-cf")
        pm.add("canonicalize")
        pm.add("gpu-kernel-outlining")
        pm.add("canonicalize")
        pm.run(module.operation)
    return module

m1 = gpu_frontend(m0)
#print(m1)

# =============================================================
# An example pipeline lowering gpu module to llvm dialect.
# =============================================================
def gpu_to_llvm(module):
    with ctx:
        pm = PassManager("any")
        # gpu to rocdl covers *-to-llvm conversioin patterns.
        pm.add("gpu.module(strip-debuginfo, convert-gpu-to-rocdl{use-bare-ptr-memref-call-conv=true runtime=OpenCL})")
        pm.add("rocdl-attach-target{chip=gfx1102}")
        pm.add("gpu-to-llvm{use-bare-pointers-for-kernels=true}")
        pm.add("gpu-module-to-binary")
        pm.run(module.operation)
        return module

m2 = gpu_to_llvm(m1)
#print(m2)

# FIXME : find interface to access binary. (Possibly not needed if host part is handled in MLIR)
# For now, this code tries to extract binary from the string
gpu_obj = m2.body.operations[1].objects[0]
obj_str = str(gpu_obj)
x0 = obj_str.partition("bin = \"\\")
x1 = x0[2].partition("\">")
x1 = x1[0].rsplit("\\")
#print(x1)
merged = bytearray()
for chunk in x1:
    merged = merged + bytes.fromhex(chunk[:2])
    if len(chunk) > 2 :
        merged = merged + bytearray(chunk[2:], 'ascii')
compiled_bin = merged
#print(merged)
#f = open("./dump.bin", "wb")
#f.write(merged)
#f.close()

# =============================================================
# host function to invoke the binary, directly using pyopencl.
#
# N.B., Host code in this test is manually hard coded.
#       for example, local size was determined by the compiler.
# =============================================================
data_A = np.random.rand(1024).astype(np.float32)
data_B = np.random.rand(1024).astype(np.float32)
data_result = np.empty(shape=(1024), dtype=np.float32, order='C')

ctx = cl.Context(dev_type=cl.device_type.ALL)
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
buf_A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_A.flatten())
buf_B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_B.flatten())
buf_result = cl.Buffer(ctx, mf.WRITE_ONLY, data_result.nbytes)

devs = cl.get_platforms()[0].get_devices()
prg = cl.Program(ctx, devs, [compiled_bin]).build()
kernel = cl.Kernel(prg, "vadd_kernel")

kernel.set_args(buf_A, buf_B, buf_result)
gsize = (1024, 1, 1)
lsize = (1, 1, 1)
cl.enqueue_nd_range_kernel(queue, kernel, global_work_size=gsize, local_work_size=lsize)
cl.enqueue_copy(queue, data_result, buf_result)
queue.finish()

np_result = data_A + data_B

print("A     : ", data_A)
print("B     : ", data_B)
print("Validating A + B ...")
print("Numpy : ", np_result)
print("GPU   : ", data_result)
print("Pass  :", np.allclose(data_result, np_result, rtol=1e-8))
