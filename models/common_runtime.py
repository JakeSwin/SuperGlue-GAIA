#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically manages context initialization

# Helper for error checking (exceptions are automatically raised in PyCUDA for most errors)

class HostDeviceMem:
    """Pair of host and device memory, with host as a numpy array."""
    def __init__(self, size: int, dtype: np.dtype = np.uint8):
        dtype = np.dtype(dtype)
        nbytes = size * dtype.itemsize

        # Use pagelocked (pinned) host memory for best performance
        self._host = cuda.pagelocked_empty(size, dtype)
        # Device allocation
        self._device = cuda.mem_alloc(nbytes)
        self._nbytes = nbytes

    @property
    def host(self):
        return self._host

    @host.setter
    def host(self, data):
        if isinstance(data, np.ndarray):
            np.copyto(self.host[:data.size], data.flat, casting='safe')
        else:
            assert self.host.dtype == np.uint8
            self.host[:self.nbytes] = np.frombuffer(data, dtype=np.uint8)

    @property
    def device(self):
        return self._device

    @property
    def nbytes(self):
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{int(self.device):#x}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        self._device.free()
        # PyCUDA will release pagelocked host memory when the array is deleted

def allocate_buffers(engine: trt.ICudaEngine, profile_idx: int = None):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for binding in tensor_names:
        shape = engine.get_tensor_shape(binding) if profile_idx is None else engine.get_tensor_profile_shape(binding, profile_idx)[-1]
        shape_valid = np.all([s >= 0 for s in shape])
        if not shape_valid and profile_idx is None:
            raise ValueError(f"Binding {binding} has dynamic shape, but no profile was specified.")
        size = trt.volume(shape)
        trt_type = engine.get_tensor_dtype(binding)
        try:
            dtype = np.dtype(trt.nptype(trt_type))
            print(f"Binding: {binding}, shape: {shape}, dtype: {dtype}, num elements: {size}, nbytes: {size * dtype.itemsize}")
            if binding == "keypoints":
                binding_memory = HostDeviceMem(5000, dtype)
            elif binding == "scores":
                binding_memory = HostDeviceMem(5000, dtype)
            elif binding == "descriptors":
                binding_memory = HostDeviceMem(600000, dtype)
            else:
                binding_memory = HostDeviceMem(size, dtype)
        except TypeError:  # For unsupported types
            size = int(size * trt_type.itemsize)
            binding_memory = HostDeviceMem(size)
        # Device pointer as integer
        bindings.append(int(binding_memory.device))
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(binding_memory)
        else:
            outputs.append(binding_memory)
    return inputs, outputs, bindings, stream

def free_buffers(inputs, outputs, stream):
    for mem in inputs + outputs:
        mem.free()
    # PyCUDA stream resources are automatically cleaned up, but you can call stream.synchronize() if desired

# PyCUDA memory copy helpers
def memcpy_host_to_device(device_ptr, host_arr):
    cuda.memcpy_htod(device_ptr, host_arr)

def memcpy_device_to_host(host_arr, device_ptr):
    cuda.memcpy_dtoh(host_arr, device_ptr)

def _do_inference_base(inputs, outputs, stream, execute_async_func):
    # Transfer input data to the GPU (async with stream)
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    # Run inference
    execute_async_func()
    # Transfer predictions back from the GPU
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
    # Synchronize the stream
    stream.synchronize()
    return [out.host for out in outputs]

def do_inference(context, engine, bindings, inputs, outputs, stream):
    def execute_async_func():
        context.execute_async_v3(stream_handle=int(stream.handle))  # TensorRT expects integer stream handle
    num_io = engine.num_io_tensors
    for i in range(num_io):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    return _do_inference_base(inputs, outputs, stream, execute_async_func)
