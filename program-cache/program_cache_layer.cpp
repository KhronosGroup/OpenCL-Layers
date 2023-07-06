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

#include <ocl_program_cache/program_cache.hpp>

#include "lib/src/utils.hpp"

#include <CL/cl_layer.h>
#include <CL/opencl.hpp>

#include <cassert>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <string>
#include <variant>
#include <vector>

namespace {

namespace utils = ocl::program_cache::utils;

struct binary_program
{
};

struct program_entry
{
    cl_uint reference_count = 1;
    cl_context context{};
    cl_program program{};
    std::variant<binary_program, std::string, std::vector<char>> source =
        binary_program{};
    void(CL_CALLBACK* pfn_notify)(cl_program, void*);
    void* notify_user_data{};
    std::map<cl_uint, std::vector<unsigned char>> specialization_constants;
};

_cl_icd_dispatch dispatch{};
const _cl_icd_dispatch* tdispatch{};
std::unique_ptr<ocl::program_cache::program_cache> program_cache;

std::mutex programs_mutex;
std::intptr_t next_program_idx = 1;
std::map<std::size_t, program_entry> program_entries;

CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource_wrap(cl_context context,
                               cl_uint count,
                               const char** strings,
                               const size_t* lengths,
                               cl_int* errcode_ret)
{
    if (!strings)
    {
        if (errcode_ret) *errcode_ret = CL_INVALID_ARG_VALUE;
    }
    std::stringstream sstream;
    for (size_t i = 0; i < count; ++i)
    {
        if (lengths && lengths[i] > 0)
        {
            sstream << std::string_view(strings[i], lengths[i]);
        }
        else
        {
            sstream << std::string_view(strings[i]);
        }
    }
    program_entry entry{};
    entry.context = context;
    entry.source = sstream.str();

    std::lock_guard lock(programs_mutex);
    const auto index = next_program_idx++;
    program_entries[index] = std::move(entry);
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return reinterpret_cast<cl_program>(index);
}


CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithIL_wrap(
    cl_context context, const void* il, size_t length, cl_int* errcode_ret)
{
    if (!il)
    {
        if (errcode_ret) *errcode_ret = CL_INVALID_ARG_VALUE;
    }
    std::vector<char> source(length);
    std::copy_n(static_cast<const char*>(il), length, source.begin());
    program_entry entry{};
    entry.source = std::move(source);
    entry.context = context;

    std::lock_guard lock(programs_mutex);
    const auto index = next_program_idx++;
    program_entries[index] = std::move(entry);
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return reinterpret_cast<cl_program>(index);
}

CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithBinary_wrap(cl_context context,
                               cl_uint num_devices,
                               const cl_device_id* device_list,
                               const size_t* lengths,
                               const unsigned char** binaries,
                               cl_int* binary_status,
                               cl_int* errcode_ret)
{
    auto program = tdispatch->clCreateProgramWithBinary(
        context, num_devices, device_list, lengths, binaries, binary_status,
        errcode_ret);
    if (errcode_ret && *errcode_ret != CL_SUCCESS)
    {
        assert(program == NULL);
        return NULL;
    }
    program_entry entry{};
    entry.context = context;
    entry.program = program;

    std::lock_guard lock(programs_mutex);
    const auto index = next_program_idx++;
    program_entries[index] = std::move(entry);
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return reinterpret_cast<cl_program>(index);
}

CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithBuiltInKernels_wrap(cl_context context,
                                       cl_uint num_devices,
                                       const cl_device_id* device_list,
                                       const char* kernel_names,
                                       cl_int* errcode_ret)
{
    auto program = tdispatch->clCreateProgramWithBuiltInKernels(
        context, num_devices, device_list, kernel_names, errcode_ret);
    if (errcode_ret && *errcode_ret != CL_SUCCESS)
    {
        assert(program == NULL);
        return NULL;
    }

    program_entry entry;
    entry.context = context;
    entry.program = program;

    std::lock_guard lock(programs_mutex);
    const auto index = next_program_idx++;
    program_entries[index] = std::move(entry);
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return reinterpret_cast<cl_program>(index);
}


CL_API_ENTRY cl_int CL_API_CALL clRetainProgram_wrap(cl_program program)
{
    const auto index = reinterpret_cast<intptr_t>(program);
    std::lock_guard lock(programs_mutex);
    auto entry_it = program_entries.find(index);
    if (entry_it == program_entries.end())
    {
        return CL_INVALID_PROGRAM;
    }
    const auto error = tdispatch->clRetainProgram(entry_it->second.program);
    if (error == CL_SUCCESS)
    {
        entry_it->second.reference_count += 1;
    }
    return error;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram_wrap(cl_program program)
{
    const auto index = reinterpret_cast<intptr_t>(program);
    std::lock_guard lock(programs_mutex);
    auto entry_it = program_entries.find(index);
    if (entry_it == program_entries.end())
    {
        return CL_INVALID_PROGRAM;
    }
    const auto error = tdispatch->clReleaseProgram(entry_it->second.program);
    if (error != CL_SUCCESS)
    {
        return error;
    }
    entry_it->second.reference_count -= 1;
    if (entry_it->second.reference_count == 0)
    {
        if (entry_it->second.pfn_notify)
            entry_it->second.pfn_notify(program,
                                        entry_it->second.notify_user_data);
        program_entries.erase(entry_it);
    }
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clSetProgramReleaseCallback_wrap(
    cl_program program,
    void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    if (!pfn_notify)
    {
        return CL_INVALID_ARG_VALUE;
    }
    const auto index = reinterpret_cast<intptr_t>(program);
    std::lock_guard lock(programs_mutex);
    auto entry_it = program_entries.find(index);
    if (entry_it == program_entries.end())
    {
        return CL_INVALID_PROGRAM;
    }
    entry_it->second.pfn_notify = pfn_notify;
    entry_it->second.notify_user_data = user_data;
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clSetProgramSpecializationConstant_wrap(cl_program program,
                                        cl_uint spec_id,
                                        size_t spec_size,
                                        const void* spec_value)
{
    if (!spec_size || !spec_value)
    {
        return CL_INVALID_ARG_VALUE;
    }
    const auto index = reinterpret_cast<intptr_t>(program);
    std::lock_guard lock(programs_mutex);
    auto entry_it = program_entries.find(index);
    if (entry_it == program_entries.end())
    {
        return CL_INVALID_PROGRAM;
    }
    auto& spec_data = entry_it->second.specialization_constants[spec_id];
    spec_data.resize(spec_size);
    std::copy_n(static_cast<const unsigned char*>(spec_value), spec_size,
                spec_data.begin());
    return CL_SUCCESS;
}

cl_int ensure_program(program_entry& entry)
{
    cl_int error = CL_SUCCESS;
    if (entry.program == NULL)
    {
        std::visit(utils::overloads{
                       [](binary_program) { assert(false); },
                       [&](const std::string& source) {
                           const char* strings = source.c_str();
                           entry.program = tdispatch->clCreateProgramWithSource(
                               entry.context, 1, &strings, nullptr, &error);
                       },
                       [&](const std::vector<char>& il) {
                           entry.program = tdispatch->clCreateProgramWithIL(
                               entry.context, il.data(), il.size(), &error);
                       } },
                   entry.source);
    }
    return error;
}

CL_API_ENTRY cl_int CL_API_CALL clBuildProgram_wrap(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    const auto index = reinterpret_cast<intptr_t>(program);
    std::lock_guard lock(programs_mutex);
    auto entry_it = program_entries.find(index);
    if (entry_it == program_entries.end())
    {
        return CL_INVALID_PROGRAM;
    }
    // ToDo programCache
    if (const auto error = ensure_program(entry_it->second);
        error != CL_SUCCESS)
    {
        return error;
    }
    return tdispatch->clBuildProgram(entry_it->second.program, num_devices,
                                     device_list, options, pfn_notify,
                                     user_data);
}

CL_API_ENTRY cl_int CL_API_CALL clCompileProgram_wrap(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    cl_uint num_input_headers,
    const cl_program* input_headers,
    const char** header_include_names,
    void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
    const auto index = reinterpret_cast<intptr_t>(program);
    std::lock_guard lock(programs_mutex);
    auto entry_it = program_entries.find(index);
    if (entry_it == program_entries.end())
    {
        return CL_INVALID_PROGRAM;
    }
    if (const auto error = ensure_program(entry_it->second);
        error != CL_SUCCESS)
    {
        return error;
    }
    return tdispatch->clCompileProgram(entry_it->second.program, num_devices,
                                       device_list, options, num_input_headers,
                                       input_headers, header_include_names,
                                       pfn_notify, user_data);
}

CL_API_ENTRY cl_program CL_API_CALL clLinkProgram_wrap(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    cl_uint num_input_programs,
    const cl_program* input_programs,
    void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
    std::lock_guard lock(programs_mutex);
    std::vector<cl_program> unwrapped_programs;
    for (size_t program_idx = 0; program_idx < num_input_programs;
         ++program_idx)
    {
        const auto index =
            reinterpret_cast<intptr_t>(input_programs[program_idx]);
        auto entry_it = program_entries.find(index);
        if (entry_it == program_entries.end())
        {
            if (errcode_ret) *errcode_ret = CL_INVALID_PROGRAM;
            return NULL;
        }
        if (const auto error = ensure_program(entry_it->second);
            error != CL_SUCCESS)
        {
            if (errcode_ret) *errcode_ret = error;
            return NULL;
        }
        unwrapped_programs.push_back(entry_it->second.program);
    }
    return tdispatch->clLinkProgram(
        context, num_devices, device_list, options, num_input_programs,
        unwrapped_programs.data(), pfn_notify, user_data, errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clGetProgramInfo_wrap(cl_program program,
                      cl_program_info param_name,
                      size_t param_value_size,
                      void* param_value,
                      size_t* param_value_size_ret)
{
    const auto index = reinterpret_cast<intptr_t>(program);
    std::lock_guard lock(programs_mutex);
    auto entry_it = program_entries.find(index);
    if (entry_it == program_entries.end())
    {
        return CL_INVALID_PROGRAM;
    }
    switch (param_name)
    {
        case CL_PROGRAM_REFERENCE_COUNT:
            if (param_value && param_value_size < sizeof(cl_uint))
                return CL_INVALID_VALUE;
            if (param_value)
                *static_cast<cl_uint*>(param_value) =
                    entry_it->second.reference_count;
            if (param_value_size_ret) *param_value_size_ret = sizeof(cl_uint);
            return CL_SUCCESS;
        case CL_PROGRAM_CONTEXT:
            if (param_value && param_value_size < sizeof(cl_context))
                return CL_INVALID_VALUE;
            if (param_value)
                *static_cast<cl_context*>(param_value) =
                    entry_it->second.context;
            if (param_value_size_ret)
                *param_value_size_ret = sizeof(cl_context);
            return CL_SUCCESS;
        case CL_PROGRAM_NUM_DEVICES:
            if (auto unwrapped_program = entry_it->second.program;
                unwrapped_program)
            {
                return tdispatch->clGetProgramInfo(
                    unwrapped_program, CL_PROGRAM_NUM_DEVICES, param_value_size,
                    param_value, param_value_size_ret);
            }
            return tdispatch->clGetContextInfo(
                entry_it->second.context, CL_CONTEXT_NUM_DEVICES,
                param_value_size, param_value, param_value_size_ret);
        case CL_PROGRAM_DEVICES:
            if (auto unwrapped_program = entry_it->second.program;
                unwrapped_program)
            {
                return tdispatch->clGetProgramInfo(
                    unwrapped_program, CL_PROGRAM_DEVICES, param_value_size,
                    param_value, param_value_size_ret);
            }
            return tdispatch->clGetContextInfo(
                entry_it->second.context, CL_CONTEXT_DEVICES, param_value_size,
                param_value, param_value_size_ret);
        case CL_PROGRAM_SOURCE:
            if (std::holds_alternative<std::string>(entry_it->second.source))
            {
                const char* source =
                    std::get<std::string>(entry_it->second.source).c_str();
                const size_t source_size = std::strlen(source);
                if (param_value && param_value_size < source_size)
                    return CL_INVALID_VALUE;
                if (param_value)
                    std::strcpy(static_cast<char*>(param_value), source);
                if (param_value_size_ret) *param_value_size_ret = source_size;
            }
            else
            {
                if (param_value && param_value_size < 1)
                    return CL_INVALID_VALUE;
                if (param_value) *static_cast<char*>(param_value) = '\0';
                if (param_value_size_ret) *param_value_size_ret = 1;
            }
            return CL_SUCCESS;
        case CL_PROGRAM_IL:
            if (std::holds_alternative<std::vector<char>>(
                    entry_it->second.source))
            {
                const auto& il =
                    std::get<std::vector<char>>(entry_it->second.source);
                if (param_value && param_value_size < il.size())
                    return CL_INVALID_VALUE;
                if (param_value)
                    std::copy(il.begin(), il.end(),
                              static_cast<char*>(param_value));
                if (param_value_size_ret) *param_value_size_ret = il.size();
            }
            else if (param_value_size_ret)
            {
                *param_value_size_ret = 0;
            }
            return CL_SUCCESS;
        case CL_PROGRAM_BINARY_SIZES:
            if (auto unwrapped_program = entry_it->second.program;
                unwrapped_program)
            {
                return tdispatch->clGetProgramInfo(
                    unwrapped_program, CL_PROGRAM_BINARY_SIZES,
                    param_value_size, param_value, param_value_size_ret);
            }
            else if (param_value_size_ret)
            {
                *param_value_size_ret = 0;
            }
            return CL_SUCCESS;
        case CL_PROGRAM_BINARIES:
            if (auto unwrapped_program = entry_it->second.program;
                unwrapped_program)
            {
                return tdispatch->clGetProgramInfo(
                    unwrapped_program, CL_PROGRAM_BINARIES, param_value_size,
                    param_value, param_value_size_ret);
            }
            else if (param_value_size_ret)
            {
                *param_value_size_ret = 0;
            }
            return CL_SUCCESS;
        case CL_PROGRAM_NUM_KERNELS:
            if (auto unwrapped_program = entry_it->second.program;
                unwrapped_program)
            {
                return tdispatch->clGetProgramInfo(
                    unwrapped_program, CL_PROGRAM_NUM_KERNELS, param_value_size,
                    param_value, param_value_size_ret);
            }
            if (param_value && param_value_size < sizeof(size_t))
                return CL_INVALID_VALUE;
            if (param_value) *static_cast<size_t*>(param_value) = 0;
            if (param_value_size_ret) *param_value_size_ret = sizeof(size_t);
            return CL_SUCCESS;
        case CL_PROGRAM_KERNEL_NAMES:
            if (auto unwrapped_program = entry_it->second.program;
                unwrapped_program)
            {
                return tdispatch->clGetProgramInfo(
                    unwrapped_program, CL_PROGRAM_KERNEL_NAMES,
                    param_value_size, param_value, param_value_size_ret);
            }
            if (param_value && param_value_size < 1) return CL_INVALID_VALUE;
            if (param_value) *static_cast<char*>(param_value) = '\0';
            if (param_value_size_ret) *param_value_size_ret = 1;
            return CL_SUCCESS;
        case CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT:
            if (auto unwrapped_program = entry_it->second.program;
                unwrapped_program)
            {
                return tdispatch->clGetProgramInfo(
                    unwrapped_program, CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT,
                    param_value_size, param_value, param_value_size_ret);
            }
            if (param_value && param_value_size < sizeof(cl_bool))
                return CL_INVALID_VALUE;
            if (param_value) *static_cast<cl_bool*>(param_value) = CL_FALSE;
            if (param_value_size_ret) *param_value_size_ret = sizeof(cl_bool);
            return CL_SUCCESS;
        case CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT:
            if (auto unwrapped_program = entry_it->second.program;
                unwrapped_program)
            {
                return tdispatch->clGetProgramInfo(
                    unwrapped_program, CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT,
                    param_value_size, param_value, param_value_size_ret);
            }
            if (param_value && param_value_size < sizeof(cl_bool))
                return CL_INVALID_VALUE;
            if (param_value) *static_cast<cl_bool*>(param_value) = CL_FALSE;
            if (param_value_size_ret) *param_value_size_ret = sizeof(cl_bool);
            return CL_SUCCESS;

        default: return CL_INVALID_VALUE;
    }
}

CL_API_ENTRY cl_int CL_API_CALL
clGetProgramBuildInfo_wrap(cl_program program,
                           cl_device_id device,
                           cl_program_build_info param_name,
                           size_t param_value_size,
                           void* param_value,
                           size_t* param_value_size_ret)
{
    const auto index = reinterpret_cast<intptr_t>(program);
    std::lock_guard lock(programs_mutex);
    auto entry_it = program_entries.find(index);
    if (entry_it == program_entries.end())
    {
        return CL_INVALID_PROGRAM;
    }
    switch (param_name)
    {
        case CL_PROGRAM_BUILD_STATUS:
            if (auto unwrapped_program = entry_it->second.program;
                unwrapped_program)
            {
                return tdispatch->clGetProgramBuildInfo(
                    unwrapped_program, device, CL_PROGRAM_BUILD_STATUS,
                    param_value_size, param_value, param_value_size_ret);
            }
            if (param_value && param_value_size < sizeof(cl_build_status))
                return CL_INVALID_VALUE;
            if (param_value)
                *static_cast<cl_build_status*>(param_value) = CL_BUILD_NONE;
            if (param_value_size_ret)
                *param_value_size_ret = sizeof(cl_build_status);
            return CL_SUCCESS;
        case CL_PROGRAM_BUILD_OPTIONS:
            if (auto unwrapped_program = entry_it->second.program;
                unwrapped_program)
            {
                return tdispatch->clGetProgramBuildInfo(
                    unwrapped_program, device, CL_PROGRAM_BUILD_OPTIONS,
                    param_value_size, param_value, param_value_size_ret);
            }
            if (param_value && param_value_size < 1) return CL_INVALID_VALUE;
            if (param_value) *static_cast<char*>(param_value) = '\0';
            if (param_value_size_ret) *param_value_size_ret = 1;
            return CL_SUCCESS;
        case CL_PROGRAM_BUILD_LOG:
            if (auto unwrapped_program = entry_it->second.program;
                unwrapped_program)
            {
                return tdispatch->clGetProgramBuildInfo(
                    unwrapped_program, device, CL_PROGRAM_BUILD_LOG,
                    param_value_size, param_value, param_value_size_ret);
            }
            if (param_value && param_value_size < 1) return CL_INVALID_VALUE;
            if (param_value) *static_cast<char*>(param_value) = '\0';
            if (param_value_size_ret) *param_value_size_ret = 1;
            return CL_SUCCESS;
        case CL_PROGRAM_BINARY_TYPE:
            if (auto unwrapped_program = entry_it->second.program;
                unwrapped_program)
            {
                return tdispatch->clGetProgramBuildInfo(
                    unwrapped_program, device, CL_PROGRAM_BINARY_TYPE,
                    param_value_size, param_value, param_value_size_ret);
            }
            if (param_value
                && param_value_size < sizeof(cl_program_binary_type))
                return CL_INVALID_VALUE;
            if (param_value)
                *static_cast<cl_program_binary_type*>(param_value) =
                    CL_PROGRAM_BINARY_TYPE_NONE;
            if (param_value_size_ret)
                *param_value_size_ret = sizeof(cl_program_binary_type);
            return CL_SUCCESS;
        case CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE:
            if (auto unwrapped_program = entry_it->second.program;
                unwrapped_program)
            {
                return tdispatch->clGetProgramBuildInfo(
                    unwrapped_program, device,
                    CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE,
                    param_value_size, param_value, param_value_size_ret);
            }
            if (param_value && param_value_size < sizeof(size_t))
                return CL_INVALID_VALUE;
            if (param_value) *static_cast<size_t*>(param_value) = 0;
            if (param_value_size_ret) *param_value_size_ret = sizeof(size_t);
            return CL_SUCCESS;
        default: return CL_INVALID_VALUE;
    }
}

CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel_wrap(cl_program program,
                                                       const char* kernel_name,
                                                       cl_int* errcode_ret)
{
    const auto index = reinterpret_cast<intptr_t>(program);
    std::lock_guard lock(programs_mutex);
    auto entry_it = program_entries.find(index);
    if (entry_it == program_entries.end())
    {
        if (errcode_ret) *errcode_ret = CL_INVALID_PROGRAM;
        return NULL;
    }
    if (entry_it->second.program == NULL)
    {
        if (errcode_ret) *errcode_ret = CL_INVALID_PROGRAM_EXECUTABLE;
        return NULL;
    }
    return tdispatch->clCreateKernel(entry_it->second.program, kernel_name,
                                     errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clCreateKernelsInProgram_wrap(cl_program program,
                              cl_uint num_kernels,
                              cl_kernel* kernels,
                              cl_uint* num_kernels_ret)
{
    const auto index = reinterpret_cast<intptr_t>(program);
    std::lock_guard lock(programs_mutex);
    auto entry_it = program_entries.find(index);
    if (entry_it == program_entries.end())
    {
        return CL_INVALID_PROGRAM;
    }
    if (entry_it->second.program == NULL)
    {
        return CL_INVALID_PROGRAM_EXECUTABLE;
    }
    return tdispatch->clCreateKernelsInProgram(
        entry_it->second.program, num_kernels, kernels, num_kernels_ret);
}

void init_dispatch()
{
    dispatch = {};
    dispatch.clCreateProgramWithSource = &clCreateProgramWithSource_wrap;
    dispatch.clCreateProgramWithIL = &clCreateProgramWithIL_wrap;
    dispatch.clCreateProgramWithBinary = &clCreateProgramWithBinary_wrap;
    dispatch.clCreateProgramWithBuiltInKernels =
        &clCreateProgramWithBuiltInKernels_wrap;
    dispatch.clRetainProgram = &clRetainProgram_wrap;
    dispatch.clReleaseProgram = &clReleaseProgram_wrap;
    dispatch.clSetProgramReleaseCallback = &clSetProgramReleaseCallback_wrap;
    dispatch.clSetProgramSpecializationConstant =
        &clSetProgramSpecializationConstant_wrap;
    dispatch.clBuildProgram = &clBuildProgram_wrap;
    dispatch.clCompileProgram = &clCompileProgram_wrap;
    dispatch.clLinkProgram = &clLinkProgram_wrap;
    dispatch.clGetProgramInfo = &clGetProgramInfo_wrap;
    dispatch.clGetProgramBuildInfo = &clGetProgramBuildInfo_wrap;
    dispatch.clCreateKernel = &clCreateKernel_wrap;
    dispatch.clCreateKernelsInProgram = &clCreateKernelsInProgram_wrap;
}

} // namespace

CL_API_ENTRY cl_int CL_API_CALL clGetLayerInfo(cl_layer_info param_name,
                                               size_t param_value_size,
                                               void* param_value,
                                               size_t* param_value_size_ret)
{
    switch (param_name)
    {
        case CL_LAYER_API_VERSION:
            if (param_value)
            {
                if (param_value_size < sizeof(cl_layer_api_version))
                    return CL_INVALID_VALUE;
                *((cl_layer_api_version*)param_value) =
                    CL_LAYER_API_VERSION_100;
            }
            if (param_value_size_ret)
                *param_value_size_ret = sizeof(cl_layer_api_version);
            break;
        default: return CL_INVALID_VALUE;
    }
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clInitLayer(cl_uint num_entries,
            const struct _cl_icd_dispatch* target_dispatch,
            cl_uint* num_entries_out,
            const struct _cl_icd_dispatch** layer_dispatch_ret)
{
    // WHY WHY WHY???
    if (!target_dispatch || !layer_dispatch_ret || !num_entries_out
        || num_entries < sizeof(dispatch) / sizeof(dispatch.clGetPlatformIDs))
        return CL_INVALID_VALUE;

    try
    {
        program_cache = std::make_unique<ocl::program_cache::program_cache>();
    } catch (const std::exception& e)
    {
        return CL_INVALID_VALUE;
    }
    program_entries.clear();
    init_dispatch();

    tdispatch = target_dispatch;
    *layer_dispatch_ret = &dispatch;
    *num_entries_out = sizeof(dispatch) / sizeof(dispatch.clGetPlatformIDs);
    return CL_SUCCESS;
}
