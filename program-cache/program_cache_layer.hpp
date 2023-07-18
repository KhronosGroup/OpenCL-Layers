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

/// @file program_cache_layer.hpp
/// @brief Declares the class \c program_cache_layer

#ifndef OCL_PROGRAM_CACHE_PROGRAM_CACHE_LAYER_HPP_
#define OCL_PROGRAM_CACHE_PROGRAM_CACHE_LAYER_HPP_

#include <ocl_program_cache/program_cache.hpp>

#include <CL/cl_layer.h>

#include <cassert>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

namespace ocl::program_cache::layer {

struct binary_program
{
};

/// @brief Encompasses all data and behaviour associated with the layer.
/// A singleton instance exists that is defined in program_cache_layer_surface.cpp.
/// @note The program cache needs both the source/IL code and the program build options for the
/// cache lookup. These two are provided by two OpenCL API functions: \c clCreateProgramWithSource /
/// \c clCreateProgramWithIL and \c clBuildProgram. Thereby, the creation of the \c cl_program
/// object must be delayed until \c clBuildProgram is called, and only at that point the program can
/// be looked up in the cache or created as new. To achieve this, this class returns mock \c
/// cl_program objects that map to entries in the internal \c program_entries_ table. These mock
/// objects must behave as valid \c cl_program objects, thereby all OpenCL API entry points
/// concerning \c cl_program are intercepted. If the wrapped \c cl_program already exists, the API
/// call can directly be translated to it. Otherwise, the \c program_entry contains the associated
/// data (e.g. source code, context), with which the API call can be answered as it was a properly
/// created \c cl_program.
class program_cache_layer {
public:
    explicit program_cache_layer(const _cl_icd_dispatch* tdispatch);

    cl_program clCreateProgramWithSource(cl_context context,
                                         cl_uint count,
                                         const char** strings,
                                         const size_t* lengths,
                                         cl_int* errcode_ret) noexcept;

    cl_program clCreateProgramWithIL(cl_context context,
                                     const void* il,
                                     size_t length,
                                     cl_int* errcode_ret) noexcept;

    cl_program clCreateProgramWithBinary(cl_context context,
                                         cl_uint num_devices,
                                         const cl_device_id* device_list,
                                         const size_t* lengths,
                                         const unsigned char** binaries,
                                         cl_int* binary_status,
                                         cl_int* errcode_ret) noexcept;

    cl_program clCreateProgramWithBuiltInKernels(cl_context context,
                                                 cl_uint num_devices,
                                                 const cl_device_id* device_list,
                                                 const char* kernel_names,
                                                 cl_int* errcode_ret) noexcept;

    cl_int clRetainProgram(cl_program program) noexcept;

    cl_int clReleaseProgram(cl_program program) noexcept;

    cl_int clSetProgramReleaseCallback(cl_program program,
                                       void(CL_CALLBACK* pfn_notify)(cl_program program,
                                                                     void* user_data),
                                       void* user_data) noexcept;

    cl_int clSetProgramSpecializationConstant(cl_program program,
                                              cl_uint spec_id,
                                              size_t spec_size,
                                              const void* spec_value) noexcept;

    cl_int clBuildProgram(cl_program program,
                          cl_uint num_devices,
                          const cl_device_id* device_list,
                          const char* options,
                          void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
                          void* user_data) noexcept;

    cl_int clCompileProgram(cl_program program,
                            cl_uint num_devices,
                            const cl_device_id* device_list,
                            const char* options,
                            cl_uint num_input_headers,
                            const cl_program* input_headers,
                            const char** header_include_names,
                            void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
                            void* user_data) noexcept;

    cl_program clLinkProgram(cl_context context,
                             cl_uint num_devices,
                             const cl_device_id* device_list,
                             const char* options,
                             cl_uint num_input_programs,
                             const cl_program* input_programs,
                             void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
                             void* user_data,
                             cl_int* errcode_ret) noexcept;

    cl_int clGetProgramInfo(cl_program program,
                            cl_program_info param_name,
                            size_t param_value_size,
                            void* param_value,
                            size_t* param_value_size_ret) noexcept;

    cl_int clGetProgramBuildInfo(cl_program program,
                                 cl_device_id device,
                                 cl_program_build_info param_name,
                                 size_t param_value_size,
                                 void* param_value,
                                 size_t* param_value_size_ret) noexcept;

    cl_kernel
    clCreateKernel(cl_program program, const char* kernel_name, cl_int* errcode_ret) noexcept;

    cl_int clCreateKernelsInProgram(cl_program program,
                                    cl_uint num_kernels,
                                    cl_kernel* kernels,
                                    cl_uint* num_kernels_ret) noexcept;

    cl_int clGetKernelInfo(cl_kernel kernel,
                           cl_kernel_info param_name,
                           size_t param_value_size,
                           void* param_value,
                           size_t* param_value_size_ret) noexcept;

private:
    /// @brief Entry that wraps a \c cl_program (potentially null) and is stored in the \c
    /// program_entries_ table.
    struct program_entry
    {
        explicit program_entry(cl_context context): context_(context)
        {
            assert(context != nullptr);
        }

        ~program_entry() = default;

        program_entry(const program_entry&) = delete;
        program_entry(program_entry&&) = default;

        program_entry& operator=(const program_entry&) = delete;
        program_entry& operator=(program_entry&&) = default;

        cl_uint reference_count_{ 1 };
        cl_context context_;
        cl_program program_{ nullptr };
        std::variant<binary_program, std::string, std::vector<char>> source_;
        std::string options_;
        void(CL_CALLBACK* release_notify_)(cl_program, void*){ nullptr };
        void* notify_user_data_{ nullptr };
        std::map<cl_uint, std::vector<unsigned char>> specialization_constants_;
        bool build_attempted_{ false };
    };

    bool is_il_program_supported(cl_context context) const;
    bool are_global_variables_supported(cl_context context) const;
    cl_int ensure_program(program_entry& entry);

    /// @brief OpenCL API dispatch table
    const _cl_icd_dispatch* tdispatch_;

    /// @brief ProgramCache object to use for caching
    program_cache program_cache_;

    /// @brief Mutex protecting \c next_program_idx_ and \c program_entries_
    std::recursive_mutex programs_mutex_;

    /// @brief Incrementing index that is reinterpreted as \c cl_program for the mock programs
    /// produced by the layer
    std::intptr_t next_program_idx_{ 1 };

    /// @brief Mapping from the program index to the entries
    std::map<std::intptr_t, program_entry> program_entries_;
};

} // namespace ocl::program_cache::layer

#endif // OCL_PROGRAM_CACHE_PROGRAM_CACHE_LAYER_HPP_
