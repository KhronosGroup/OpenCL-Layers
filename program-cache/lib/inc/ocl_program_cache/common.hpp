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

#ifndef OCL_PROGRAM_CACHE_LIB_INC_OCL_PROGRAM_CACHE_COMMON_HPP_
#define OCL_PROGRAM_CACHE_LIB_INC_OCL_PROGRAM_CACHE_COMMON_HPP_

#include <CL/cl_icd.h>

#include <stdexcept>

namespace ocl::program_cache {

/// @brief Error thrown when the path for the cache cannot be accessed.
struct cache_access_error : public std::runtime_error
{
    cache_access_error(const std::string& what_arg)
        : std::runtime_error(what_arg)
    {}
};

/// @brief Error thrown when the OpenCL runtime returns an error.
struct opencl_error : public std::runtime_error
{
    opencl_error(cl_int error,
                 const std::string& prefix = "An OpenCL error occured: ")
        : std::runtime_error(prefix + std::to_string(error)), error_(error)
    {}

    cl_int err() const { return error_; }

private:
    cl_int error_;
};

/// @brief Error thrown when the passed OpenCL program could not be built.
struct opencl_build_error : public opencl_error
{
    opencl_build_error(cl_int error)
        : opencl_error(error, "An OpenCL kernel build error occured: ")
    {}
};

/// @brief Error thrown when a cl_program is passed which should have been
/// built previously.
struct unbuilt_program_error : public std::runtime_error
{
    unbuilt_program_error()
        : std::runtime_error("The passed program has not been built")
    {}
};

struct bad_opencl_version_format : public std::runtime_error
{
    bad_opencl_version_format()
        : std::runtime_error(
            "Got invalid OpenCL version string from the runtime")
    {}
};

struct preprocess_exception : public std::runtime_error
{
    preprocess_exception(const std::string& what): std::runtime_error(what) {}
};

struct program_cache_dispatch
{
    cl_api_clBuildProgram clBuildProgram{};
    cl_api_clCreateContextFromType clCreateContextFromType{};
    cl_api_clCreateProgramWithBinary clCreateProgramWithBinary{};
    cl_api_clCreateProgramWithIL clCreateProgramWithIL{};
    cl_api_clCreateProgramWithSource clCreateProgramWithSource{};
    cl_api_clGetContextInfo clGetContextInfo{};
    cl_api_clGetDeviceInfo clGetDeviceInfo{};
    cl_api_clGetPlatformIDs clGetPlatformIDs{};
    cl_api_clGetPlatformInfo clGetPlatformInfo{};
    cl_api_clGetProgramBuildInfo clGetProgramBuildInfo{};
    cl_api_clGetProgramInfo clGetProgramInfo{};
    cl_api_clReleaseDevice clReleaseDevice{};
    cl_api_clReleaseProgram clReleaseProgram{};
};


} // namespace ocl::program_cache

#endif // OCL_PROGRAM_CACHE_LIB_INC_OCL_PROGRAM_CACHE_COMMON_HPP_
