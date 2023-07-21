/*
 * Copyright (c) 2023 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * OpenCL is a trademark of Apple Inc. used under license by Khronos.
 */

/// @file main.cpp
/// @brief Standalone test to check if the ProgramCache library can be consumed.

#include <program_cache/program_cache.hpp>

#include <CL/opencl.hpp>

#include <string_view>

namespace {
ocl::program_cache::program_cache_dispatch get_default_program_cache_dispatch()
{
    ocl::program_cache::program_cache_dispatch dispatch{};
    dispatch.clBuildProgram = &clBuildProgram;
    dispatch.clCreateContextFromType = &clCreateContextFromType;
    dispatch.clCreateProgramWithBinary = &clCreateProgramWithBinary;
    dispatch.clCreateProgramWithIL = &clCreateProgramWithIL;
    dispatch.clCreateProgramWithSource = &clCreateProgramWithSource;
    dispatch.clGetContextInfo = &clGetContextInfo;
    dispatch.clGetDeviceInfo = &clGetDeviceInfo;
    dispatch.clGetPlatformIDs = &clGetPlatformIDs;
    dispatch.clGetPlatformInfo = &clGetPlatformInfo;
    dispatch.clGetProgramBuildInfo = &clGetProgramBuildInfo;
    dispatch.clGetProgramInfo = &clGetProgramInfo;
    dispatch.clReleaseDevice = &clReleaseDevice;
    dispatch.clReleaseProgram = &clReleaseProgram;
    return dispatch;
}
} // namespace

int main()
{
    const auto context = cl::Context::getDefault();
    ocl::program_cache::program_cache program_cache(get_default_program_cache_dispatch(),
                                                    context());
    const std::string_view program_source = "kernel void foo(global int* i) { *i = 100; }";

    const cl::Program program(program_cache.fetch_or_build_source(program_source));
    cl::KernelFunctor<cl::Buffer> kernel_func(program, "foo");
    const cl::Buffer output(context, CL_MEM_WRITE_ONLY, sizeof(cl_int));

    kernel_func(cl::EnqueueArgs(cl::NDRange(1)), output);

    cl_int result{};
    cl::enqueueReadBuffer(output, true, 0, sizeof(result), &result);

    if (result != 100) return -1;
}