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

#ifndef OCL_PROGRAM_CACHE_LIB_TEST_UTILS_TEST_HPP_
#define OCL_PROGRAM_CACHE_LIB_TEST_UTILS_TEST_HPP_

#include <ocl_program_cache/common.hpp>

inline ocl::program_cache::program_cache_dispatch get_default_program_cache_dispatch()
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

#endif // OCL_PROGRAM_CACHE_LIB_TEST_UTILS_TEST_HPP_
