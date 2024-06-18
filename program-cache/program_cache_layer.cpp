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

/// @file program_cache_layer.cpp
/// @brief Defines the class \c program_cache_layer

#include "program_cache_layer.hpp"

#include "./lib/src/utils.hpp"

#include <program_cache.hpp>

#include <CL/cl_layer.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <string_view>
#include <string>
#include <variant>
#include <vector>

namespace {
namespace pcl = ocl::program_cache::layer;

ocl::program_cache::program_cache_dispatch
init_program_cache_dispatch(const _cl_icd_dispatch* tdispatch)
{
    ocl::program_cache::program_cache_dispatch program_cache_dispatch{};
    program_cache_dispatch.clBuildProgram = tdispatch->clBuildProgram;
    program_cache_dispatch.clCreateContextFromType = tdispatch->clCreateContextFromType;
    program_cache_dispatch.clCreateProgramWithBinary = tdispatch->clCreateProgramWithBinary;
    program_cache_dispatch.clCreateProgramWithIL = tdispatch->clCreateProgramWithIL;
    program_cache_dispatch.clCreateProgramWithSource = tdispatch->clCreateProgramWithSource;
    program_cache_dispatch.clGetContextInfo = tdispatch->clGetContextInfo;
    program_cache_dispatch.clGetDeviceInfo = tdispatch->clGetDeviceInfo;
    program_cache_dispatch.clGetPlatformIDs = tdispatch->clGetPlatformIDs;
    program_cache_dispatch.clGetPlatformInfo = tdispatch->clGetPlatformInfo;
    program_cache_dispatch.clGetProgramBuildInfo = tdispatch->clGetProgramBuildInfo;
    program_cache_dispatch.clGetProgramInfo = tdispatch->clGetProgramInfo;
    program_cache_dispatch.clReleaseDevice = tdispatch->clReleaseDevice;
    program_cache_dispatch.clReleaseProgram = tdispatch->clReleaseProgram;
    program_cache_dispatch.clReleaseContext = tdispatch->clReleaseContext;
    return program_cache_dispatch;
}

} // namespace

pcl::program_cache_layer::program_cache_layer(const _cl_icd_dispatch* tdispatch)
    : tdispatch_(tdispatch), program_cache_(init_program_cache_dispatch(tdispatch_))
{
    assert(tdispatch_ != nullptr);
}

cl_program pcl::program_cache_layer::clCreateProgramWithSource(cl_context context,
                                                               cl_uint count,
                                                               const char** strings,
                                                               const size_t* lengths,
                                                               cl_int* errcode_ret) noexcept
{
    if (strings == nullptr)
    {
        if (errcode_ret) *errcode_ret = CL_INVALID_ARG_VALUE;
        return nullptr;
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
    program_entry entry(context);
    entry.source_ = sstream.str();

    const std::lock_guard lock(programs_mutex_);
    const auto index = next_program_idx_++;
    program_entries_.emplace(index, std::move(entry));
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return reinterpret_cast<cl_program>(index);
}

cl_program pcl::program_cache_layer::clCreateProgramWithIL(cl_context context,
                                                           const void* il,
                                                           size_t length,
                                                           cl_int* errcode_ret) noexcept
{
    if (il == nullptr)
    {
        if (errcode_ret) *errcode_ret = CL_INVALID_ARG_VALUE;
        return nullptr;
    }
    if (!is_il_program_supported(context))
    {
        if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
        return nullptr;
    }

    program_entry entry(context);
    entry.source_ =
        std::vector<char>(static_cast<const char*>(il), static_cast<const char*>(il) + length);

    const std::lock_guard lock(programs_mutex_);
    const auto index = next_program_idx_++;
    program_entries_.emplace(index, std::move(entry));
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return reinterpret_cast<cl_program>(index);
}

cl_program pcl::program_cache_layer::clCreateProgramWithBinary(cl_context context,
                                                               cl_uint num_devices,
                                                               const cl_device_id* device_list,
                                                               const size_t* lengths,
                                                               const unsigned char** binaries,
                                                               cl_int* binary_status,
                                                               cl_int* errcode_ret) noexcept
{
    auto program = tdispatch_->clCreateProgramWithBinary(context, num_devices, device_list, lengths,
                                                         binaries, binary_status, errcode_ret);
    if (errcode_ret && *errcode_ret != CL_SUCCESS)
    {
        assert(program == nullptr);
        return nullptr;
    }
    program_entry entry(context);
    entry.program_ = program;

    const std::lock_guard lock(programs_mutex_);
    const auto index = next_program_idx_++;
    program_entries_.emplace(index, std::move(entry));
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return reinterpret_cast<cl_program>(index);
}

cl_program
pcl::program_cache_layer::clCreateProgramWithBuiltInKernels(cl_context context,
                                                            cl_uint num_devices,
                                                            const cl_device_id* device_list,
                                                            const char* kernel_names,
                                                            cl_int* errcode_ret) noexcept
{
    auto program = tdispatch_->clCreateProgramWithBuiltInKernels(context, num_devices, device_list,
                                                                 kernel_names, errcode_ret);
    if (errcode_ret && *errcode_ret != CL_SUCCESS)
    {
        assert(program == nullptr);
        return nullptr;
    }

    program_entry entry(context);
    entry.program_ = program;

    const std::lock_guard lock(programs_mutex_);
    const auto index = next_program_idx_++;
    program_entries_.emplace(index, std::move(entry));
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return reinterpret_cast<cl_program>(index);
}

cl_int pcl::program_cache_layer::clRetainProgram(cl_program program) noexcept
{
    const auto index = reinterpret_cast<intptr_t>(program);
    const std::lock_guard lock(programs_mutex_);
    auto entry_it = program_entries_.find(index);
    if (entry_it == program_entries_.end())
    {
        return CL_INVALID_PROGRAM;
    }
    if (entry_it->second.program_)
    {
        const auto error = tdispatch_->clRetainProgram(entry_it->second.program_);
        if (error != CL_SUCCESS) return error;
    }
    entry_it->second.reference_count_ += 1;
    return CL_SUCCESS;
}

cl_int pcl::program_cache_layer::clReleaseProgram(cl_program program) noexcept
{
    const auto index = reinterpret_cast<intptr_t>(program);
    const std::lock_guard lock(programs_mutex_);
    auto entry_it = program_entries_.find(index);
    if (entry_it == program_entries_.end())
    {
        return CL_INVALID_PROGRAM;
    }
    if (entry_it->second.program_)
    {
        const auto error = tdispatch_->clReleaseProgram(entry_it->second.program_);
        if (error != CL_SUCCESS) return error;
    }
    entry_it->second.reference_count_ -= 1;
    if (entry_it->second.reference_count_ == 0)
    {
        program_entries_.erase(entry_it);
    }
    return CL_SUCCESS;
}

cl_int pcl::program_cache_layer::clSetProgramReleaseCallback(
    cl_program program,
    void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data) noexcept
{
    if (!pfn_notify)
    {
        return CL_INVALID_ARG_VALUE;
    }
    const auto index = reinterpret_cast<intptr_t>(program);
    const std::lock_guard lock(programs_mutex_);
    auto entry_it = program_entries_.find(index);
    if (entry_it == program_entries_.end())
    {
        return CL_INVALID_PROGRAM;
    }
    if (entry_it->second.program_ != nullptr)
    {
        const cl_int error = tdispatch_->clSetProgramReleaseCallback(entry_it->second.program_,
                                                                     pfn_notify, user_data);
        if (error != CL_SUCCESS) return error;
    }
    else if (!are_global_variables_supported(entry_it->second.context_))
    {
        return CL_INVALID_OPERATION;
    }
    entry_it->second.release_notify_ = pfn_notify;
    entry_it->second.notify_user_data_ = user_data;
    return CL_SUCCESS;
}

cl_int pcl::program_cache_layer::clSetProgramSpecializationConstant(cl_program program,
                                                                    cl_uint spec_id,
                                                                    size_t spec_size,
                                                                    const void* spec_value) noexcept
{
    if (spec_size == 0 || spec_value == nullptr)
    {
        return CL_INVALID_ARG_VALUE;
    }
    const auto index = reinterpret_cast<intptr_t>(program);
    const std::lock_guard lock(programs_mutex_);
    auto entry_it = program_entries_.find(index);
    if (entry_it == program_entries_.end())
    {
        return CL_INVALID_PROGRAM;
    }
    if (!is_il_program_supported(entry_it->second.context_)) return CL_INVALID_OPERATION;
    auto& spec_data = entry_it->second.specialization_constants_[spec_id];
    spec_data.resize(spec_size);
    std::copy_n(static_cast<const unsigned char*>(spec_value), spec_size, spec_data.begin());
    return CL_SUCCESS;
}

cl_int pcl::program_cache_layer::clBuildProgram(cl_program program,
                                                cl_uint num_devices,
                                                const cl_device_id* device_list,
                                                const char* options,
                                                void(CL_CALLBACK* pfn_notify)(cl_program program,
                                                                              void* user_data),
                                                void* user_data) noexcept
{
    if (num_devices > 0 && !device_list)
    {
        return CL_INVALID_ARG_VALUE;
    }
    const auto index = reinterpret_cast<intptr_t>(program);
    const std::lock_guard lock(programs_mutex_);
    auto entry_it = program_entries_.find(index);
    if (entry_it == program_entries_.end())
    {
        return CL_INVALID_PROGRAM;
    }
    entry_it->second.options_ = options ? options : std::string();
    entry_it->second.build_attempted_ = true;
    cl_int error = CL_SUCCESS;
    std::visit(
        utils::overloads{
            [&](binary_program) {
                error = tdispatch_->clBuildProgram(entry_it->second.program_, num_devices,
                                                   device_list, options, pfn_notify, user_data);
            },
            [&](const auto& source) {
                try
                {
                    const cl_context context = entry_it->second.context_;
                    std::vector<cl_device_id> devices(num_devices);
                    std::copy_n(device_list, num_devices, devices.begin());
                    cl_program new_program{};
                    if constexpr (std::is_same_v<std::decay_t<decltype(source)>, std::string>)
                    {
                        new_program = program_cache_.fetch_or_build_source(
                            source, context, devices, entry_it->second.options_);
                    }
                    else
                    {
                        new_program = program_cache_.fetch_or_build_il(source, context, devices,
                                                                       entry_it->second.options_);
                    }
                    if (entry_it->second.release_notify_)
                    {
                        error = tdispatch_->clSetProgramReleaseCallback(
                            new_program, entry_it->second.release_notify_,
                            entry_it->second.notify_user_data_);
                        if (error != CL_SUCCESS) return;
                    }
                    if (entry_it->second.program_)
                        tdispatch_->clReleaseProgram(entry_it->second.program_);
                    entry_it->second.program_ = new_program;
                } catch (const ocl::program_cache::opencl_error& err)
                {
                    error = err.err();
                } catch (...)
                {
                    error = CL_INVALID_BUILD_OPTIONS;
                }
                if (pfn_notify) pfn_notify(program, user_data);
            } },
        entry_it->second.source_);
    return error;
}

cl_int pcl::program_cache_layer::clCompileProgram(cl_program program,
                                                  cl_uint num_devices,
                                                  const cl_device_id* device_list,
                                                  const char* options,
                                                  cl_uint num_input_headers,
                                                  const cl_program* input_headers,
                                                  const char** header_include_names,
                                                  void(CL_CALLBACK* pfn_notify)(cl_program program,
                                                                                void* user_data),
                                                  void* user_data) noexcept
{
    const auto index = reinterpret_cast<intptr_t>(program);
    const std::lock_guard lock(programs_mutex_);
    auto entry_it = program_entries_.find(index);
    if (entry_it == program_entries_.end())
    {
        return CL_INVALID_PROGRAM;
    }
    if (const auto error = ensure_program(entry_it->second); error != CL_SUCCESS)
    {
        return error;
    }
    entry_it->second.options_ = options ? options : std::string();
    std::vector<cl_program> wrapped_input_headers;
    for (std::size_t header_idx = 0; header_idx < num_input_headers; ++header_idx)
    {
        const auto header_program_index = reinterpret_cast<intptr_t>(input_headers[header_idx]);
        const auto header_it = program_entries_.find(header_program_index);
        if (header_it == program_entries_.end()) return CL_INVALID_PROGRAM;
        if (const auto error = ensure_program(header_it->second); error != CL_SUCCESS)
        {
            return error;
        }
        wrapped_input_headers.push_back(header_it->second.program_);
    }
    return tdispatch_->clCompileProgram(entry_it->second.program_, num_devices, device_list,
                                        options, num_input_headers, wrapped_input_headers.data(),
                                        header_include_names, pfn_notify, user_data);
}

cl_program pcl::program_cache_layer::clLinkProgram(cl_context context,
                                                   cl_uint num_devices,
                                                   const cl_device_id* device_list,
                                                   const char* options,
                                                   cl_uint num_input_programs,
                                                   const cl_program* input_programs,
                                                   void(CL_CALLBACK* pfn_notify)(cl_program program,
                                                                                 void* user_data),
                                                   void* user_data,
                                                   cl_int* errcode_ret) noexcept
{
    const std::lock_guard lock(programs_mutex_);
    std::vector<cl_program> unwrapped_programs;
    for (size_t program_idx = 0; program_idx < num_input_programs; ++program_idx)
    {
        const auto index = reinterpret_cast<intptr_t>(input_programs[program_idx]);
        auto entry_it = program_entries_.find(index);
        if (entry_it == program_entries_.end() || entry_it->second.program_ == nullptr)
        {
            if (errcode_ret) *errcode_ret = CL_INVALID_PROGRAM;
            return nullptr;
        }
        unwrapped_programs.push_back(entry_it->second.program_);
    }
    const cl_program linked_program =
        tdispatch_->clLinkProgram(context, num_devices, device_list, options, num_input_programs,
                                  unwrapped_programs.data(), pfn_notify, user_data, errcode_ret);
    program_entry entry(context);
    entry.program_ = linked_program;
    const auto index = next_program_idx_++;
    program_entries_.emplace(index, std::move(entry));
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    return reinterpret_cast<cl_program>(index);
}

cl_int pcl::program_cache_layer::clGetProgramInfo(cl_program program,
                                                  cl_program_info param_name,
                                                  size_t param_value_size,
                                                  void* param_value,
                                                  size_t* param_value_size_ret) noexcept
{
    const auto index = reinterpret_cast<intptr_t>(program);
    const std::lock_guard lock(programs_mutex_);
    auto entry_it = program_entries_.find(index);
    if (entry_it == program_entries_.end())
    {
        return CL_INVALID_PROGRAM;
    }
    switch (param_name)
    {
        case CL_PROGRAM_REFERENCE_COUNT:
            if (param_value && param_value_size < sizeof(cl_uint)) return CL_INVALID_VALUE;
            if (param_value)
                *static_cast<cl_uint*>(param_value) = entry_it->second.reference_count_;
            if (param_value_size_ret) *param_value_size_ret = sizeof(cl_uint);
            return CL_SUCCESS;
        case CL_PROGRAM_CONTEXT:
            if (param_value && param_value_size < sizeof(cl_context)) return CL_INVALID_VALUE;
            if (param_value) *static_cast<cl_context*>(param_value) = entry_it->second.context_;
            if (param_value_size_ret) *param_value_size_ret = sizeof(cl_context);
            return CL_SUCCESS;
        case CL_PROGRAM_NUM_DEVICES:
            if (auto unwrapped_program = entry_it->second.program_; unwrapped_program)
            {
                return tdispatch_->clGetProgramInfo(unwrapped_program, CL_PROGRAM_NUM_DEVICES,
                                                    param_value_size, param_value,
                                                    param_value_size_ret);
            }
            return tdispatch_->clGetContextInfo(entry_it->second.context_, CL_CONTEXT_NUM_DEVICES,
                                                param_value_size, param_value,
                                                param_value_size_ret);
        case CL_PROGRAM_DEVICES:
            if (auto unwrapped_program = entry_it->second.program_; unwrapped_program)
            {
                return tdispatch_->clGetProgramInfo(unwrapped_program, CL_PROGRAM_DEVICES,
                                                    param_value_size, param_value,
                                                    param_value_size_ret);
            }
            return tdispatch_->clGetContextInfo(entry_it->second.context_, CL_CONTEXT_DEVICES,
                                                param_value_size, param_value,
                                                param_value_size_ret);
        case CL_PROGRAM_SOURCE:
            if (std::holds_alternative<std::string>(entry_it->second.source_))
            {
                const auto& source = std::get<std::string>(entry_it->second.source_);
                const size_t source_size = source.size() + 1;
                if (param_value && param_value_size < source_size) return CL_INVALID_VALUE;
                if (param_value)
                {
                    std::copy(source.begin(), source.end(), static_cast<char*>(param_value));
                    static_cast<char*>(param_value)[source.size()] = '\0';
                }
                if (param_value_size_ret) *param_value_size_ret = source_size;
            }
            else
            {
                if (param_value && param_value_size < 1) return CL_INVALID_VALUE;
                if (param_value) *static_cast<char*>(param_value) = '\0';
                if (param_value_size_ret) *param_value_size_ret = 1;
            }
            return CL_SUCCESS;
        case CL_PROGRAM_IL:
            if (std::holds_alternative<std::vector<char>>(entry_it->second.source_))
            {
                const auto& il = std::get<std::vector<char>>(entry_it->second.source_);
                if (param_value && param_value_size < il.size()) return CL_INVALID_VALUE;
                if (param_value) std::copy(il.begin(), il.end(), static_cast<char*>(param_value));
                if (param_value_size_ret) *param_value_size_ret = il.size();
            }
            else if (param_value_size_ret)
            {
                *param_value_size_ret = 0;
            }
            return CL_SUCCESS;
        case CL_PROGRAM_BINARY_SIZES:
            if (auto unwrapped_program = entry_it->second.program_; unwrapped_program)
            {
                return tdispatch_->clGetProgramInfo(unwrapped_program, CL_PROGRAM_BINARY_SIZES,
                                                    param_value_size, param_value,
                                                    param_value_size_ret);
            }
            else if (param_value_size_ret)
            {
                *param_value_size_ret = 0;
            }
            return CL_SUCCESS;
        case CL_PROGRAM_BINARIES:
            if (auto unwrapped_program = entry_it->second.program_; unwrapped_program)
            {
                return tdispatch_->clGetProgramInfo(unwrapped_program, CL_PROGRAM_BINARIES,
                                                    param_value_size, param_value,
                                                    param_value_size_ret);
            }
            else if (param_value_size_ret)
            {
                *param_value_size_ret = 0;
            }
            return CL_SUCCESS;
        case CL_PROGRAM_NUM_KERNELS:
            if (auto unwrapped_program = entry_it->second.program_; unwrapped_program)
            {
                return tdispatch_->clGetProgramInfo(unwrapped_program, CL_PROGRAM_NUM_KERNELS,
                                                    param_value_size, param_value,
                                                    param_value_size_ret);
            }
            if (param_value && param_value_size < sizeof(size_t)) return CL_INVALID_VALUE;
            if (param_value) *static_cast<size_t*>(param_value) = 0;
            if (param_value_size_ret) *param_value_size_ret = sizeof(size_t);
            return CL_SUCCESS;
        case CL_PROGRAM_KERNEL_NAMES:
            if (auto unwrapped_program = entry_it->second.program_; unwrapped_program)
            {
                return tdispatch_->clGetProgramInfo(unwrapped_program, CL_PROGRAM_KERNEL_NAMES,
                                                    param_value_size, param_value,
                                                    param_value_size_ret);
            }
            if (param_value && param_value_size < 1) return CL_INVALID_VALUE;
            if (param_value) *static_cast<char*>(param_value) = '\0';
            if (param_value_size_ret) *param_value_size_ret = 1;
            return CL_SUCCESS;
        case CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT:
            if (auto unwrapped_program = entry_it->second.program_; unwrapped_program)
            {
                return tdispatch_->clGetProgramInfo(
                    unwrapped_program, CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT, param_value_size,
                    param_value, param_value_size_ret);
            }
            if (param_value && param_value_size < sizeof(cl_bool)) return CL_INVALID_VALUE;
            if (param_value) *static_cast<cl_bool*>(param_value) = CL_FALSE;
            if (param_value_size_ret) *param_value_size_ret = sizeof(cl_bool);
            return CL_SUCCESS;
        case CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT:
            if (auto unwrapped_program = entry_it->second.program_; unwrapped_program)
            {
                return tdispatch_->clGetProgramInfo(
                    unwrapped_program, CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT, param_value_size,
                    param_value, param_value_size_ret);
            }
            if (param_value && param_value_size < sizeof(cl_bool)) return CL_INVALID_VALUE;
            if (param_value) *static_cast<cl_bool*>(param_value) = CL_FALSE;
            if (param_value_size_ret) *param_value_size_ret = sizeof(cl_bool);
            return CL_SUCCESS;

        default: return CL_INVALID_VALUE;
    }
}

cl_int pcl::program_cache_layer::clGetProgramBuildInfo(cl_program program,
                                                       cl_device_id device,
                                                       cl_program_build_info param_name,
                                                       size_t param_value_size,
                                                       void* param_value,
                                                       size_t* param_value_size_ret) noexcept
{
    const auto index = reinterpret_cast<intptr_t>(program);
    const std::lock_guard lock(programs_mutex_);
    auto entry_it = program_entries_.find(index);
    if (entry_it == program_entries_.end())
    {
        return CL_INVALID_PROGRAM;
    }
    switch (param_name)
    {
        case CL_PROGRAM_BUILD_STATUS:
            if (auto unwrapped_program = entry_it->second.program_; unwrapped_program)
            {
                return tdispatch_->clGetProgramBuildInfo(unwrapped_program, device,
                                                         CL_PROGRAM_BUILD_STATUS, param_value_size,
                                                         param_value, param_value_size_ret);
            }
            if (param_value && param_value_size < sizeof(cl_build_status)) return CL_INVALID_VALUE;
            if (param_value)
                *static_cast<cl_build_status*>(param_value) =
                    entry_it->second.build_attempted_ ? CL_BUILD_ERROR : CL_BUILD_NONE;
            if (param_value_size_ret) *param_value_size_ret = sizeof(cl_build_status);
            return CL_SUCCESS;
        case CL_PROGRAM_BUILD_OPTIONS: {
            const auto& options = entry_it->second.options_;
            const std::size_t options_size = options.size() + 1;
            if (param_value && param_value_size < options_size) return CL_INVALID_VALUE;
            if (param_value)
            {
                std::copy(options.begin(), options.end(), static_cast<char*>(param_value));
                static_cast<char*>(param_value)[options.size()] = '\0';
            }
            if (param_value_size_ret) *param_value_size_ret = options_size;
            return CL_SUCCESS;
        }
        case CL_PROGRAM_BUILD_LOG:
            if (auto unwrapped_program = entry_it->second.program_; unwrapped_program)
            {
                return tdispatch_->clGetProgramBuildInfo(unwrapped_program, device,
                                                         CL_PROGRAM_BUILD_LOG, param_value_size,
                                                         param_value, param_value_size_ret);
            }
            if (param_value && param_value_size < 1) return CL_INVALID_VALUE;
            if (param_value) *static_cast<char*>(param_value) = '\0';
            if (param_value_size_ret) *param_value_size_ret = 1;
            return CL_SUCCESS;
        case CL_PROGRAM_BINARY_TYPE:
            if (auto unwrapped_program = entry_it->second.program_; unwrapped_program)
            {
                return tdispatch_->clGetProgramBuildInfo(unwrapped_program, device,
                                                         CL_PROGRAM_BINARY_TYPE, param_value_size,
                                                         param_value, param_value_size_ret);
            }
            if (param_value && param_value_size < sizeof(cl_program_binary_type))
                return CL_INVALID_VALUE;
            if (param_value)
                *static_cast<cl_program_binary_type*>(param_value) = CL_PROGRAM_BINARY_TYPE_NONE;
            if (param_value_size_ret) *param_value_size_ret = sizeof(cl_program_binary_type);
            return CL_SUCCESS;
        case CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE:
            if (auto unwrapped_program = entry_it->second.program_; unwrapped_program)
            {
                return tdispatch_->clGetProgramBuildInfo(
                    unwrapped_program, device, CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE,
                    param_value_size, param_value, param_value_size_ret);
            }
            if (param_value && param_value_size < sizeof(size_t)) return CL_INVALID_VALUE;
            if (param_value) *static_cast<size_t*>(param_value) = 0;
            if (param_value_size_ret) *param_value_size_ret = sizeof(size_t);
            return CL_SUCCESS;
        default: return CL_INVALID_VALUE;
    }
}

cl_kernel pcl::program_cache_layer::clCreateKernel(cl_program program,
                                                   const char* kernel_name,
                                                   cl_int* errcode_ret) noexcept
{
    const auto index = reinterpret_cast<intptr_t>(program);
    const std::lock_guard lock(programs_mutex_);
    auto entry_it = program_entries_.find(index);
    if (entry_it == program_entries_.end())
    {
        if (errcode_ret) *errcode_ret = CL_INVALID_PROGRAM;
        return nullptr;
    }
    if (entry_it->second.program_ == nullptr)
    {
        if (errcode_ret) *errcode_ret = CL_INVALID_PROGRAM_EXECUTABLE;
        return nullptr;
    }
    return tdispatch_->clCreateKernel(entry_it->second.program_, kernel_name, errcode_ret);
}

cl_int pcl::program_cache_layer::clCreateKernelsInProgram(cl_program program,
                                                          cl_uint num_kernels,
                                                          cl_kernel* kernels,
                                                          cl_uint* num_kernels_ret) noexcept
{
    const auto index = reinterpret_cast<intptr_t>(program);
    const std::lock_guard lock(programs_mutex_);
    auto entry_it = program_entries_.find(index);
    if (entry_it == program_entries_.end())
    {
        return CL_INVALID_PROGRAM;
    }
    if (entry_it->second.program_ == nullptr)
    {
        return CL_INVALID_PROGRAM_EXECUTABLE;
    }
    return tdispatch_->clCreateKernelsInProgram(entry_it->second.program_, num_kernels, kernels,
                                                num_kernels_ret);
}

cl_int pcl::program_cache_layer::clGetKernelInfo(cl_kernel kernel,
                                                 cl_kernel_info param_name,
                                                 size_t param_value_size,
                                                 void* param_value,
                                                 size_t* param_value_size_ret) noexcept
{
    switch (param_name)
    {
        case CL_KERNEL_PROGRAM: {
            if (param_value && param_value_size < sizeof(cl_program)) return CL_INVALID_ARG_VALUE;
            if (param_value_size_ret) *param_value_size_ret = sizeof(cl_program);
            if (param_value == nullptr) return CL_SUCCESS;
            cl_program wrapped_program{};
            const cl_int error = tdispatch_->clGetKernelInfo(
                kernel, CL_KERNEL_PROGRAM, sizeof(wrapped_program), &wrapped_program, nullptr);
            if (error != CL_SUCCESS) return error;
            const std::lock_guard lock(programs_mutex_);
            const auto found_it =
                std::find_if(program_entries_.begin(), program_entries_.end(),
                             [wrapped_program](const auto& key_value) {
                                 return key_value.second.program_ == wrapped_program;
                             });
            assert(found_it != program_entries_.end());
            *static_cast<cl_program*>(param_value) = reinterpret_cast<cl_program>(found_it->first);
            return CL_SUCCESS;
        }
        case CL_KERNEL_ATTRIBUTES: {
            cl_program wrapped_program{};
            cl_int error = tdispatch_->clGetKernelInfo(
                kernel, CL_KERNEL_PROGRAM, sizeof(wrapped_program), &wrapped_program, nullptr);
            if (error != CL_SUCCESS) return error;
            const std::lock_guard lock(programs_mutex_);
            const auto found_it =
                std::find_if(program_entries_.begin(), program_entries_.end(),
                             [wrapped_program](const auto& key_value) {
                                 return key_value.second.program_ == wrapped_program;
                             });
            assert(found_it != program_entries_.end());
            if (std::holds_alternative<std::string>(found_it->second.source_))
            {
                // Since we might have a program that is read from the cache and
                // created with clCreateProgramWithBinary, we must recompile the
                // source to get the attributes.
                cl_program tmp_program{};
                cl_kernel tmp_kernel{};
                rebuild_kernel_from_source(kernel, found_it->second, tmp_kernel, tmp_program);
                error =
                    tdispatch_->clGetKernelInfo(tmp_kernel, CL_KERNEL_ATTRIBUTES, param_value_size,
                                                param_value, param_value_size_ret);
                tdispatch_->clReleaseKernel(tmp_kernel);
                tdispatch_->clReleaseProgram(tmp_program);
            }
            else
            {
                if (param_value && param_value_size < 1) return CL_INVALID_ARG_VALUE;
                if (param_value) *static_cast<char*>(param_value) = '\0';
                if (param_value_size_ret) *param_value_size_ret = 1;
            }
            return error;
        }
        default:
            return tdispatch_->clGetKernelInfo(kernel, param_name, param_value_size, param_value,
                                               param_value_size_ret);
    }
}

cl_int pcl::program_cache_layer::clGetKernelArgInfo(cl_kernel kernel,
                                                    cl_uint arg_index,
                                                    cl_kernel_arg_info param_name,
                                                    size_t param_value_size,
                                                    void* param_value,
                                                    size_t* param_value_size_ret) noexcept
{
    // If this layer is used, all kernels are created from programs that are built as a binary.
    // To acquire information about the kernel args, we must recompile the kernel from source.
    cl_program wrapped_program{};
    cl_int error = tdispatch_->clGetKernelInfo(kernel, CL_KERNEL_PROGRAM, sizeof(wrapped_program),
                                               &wrapped_program, nullptr);
    if (error != CL_SUCCESS) return error;
    const std::lock_guard lock(programs_mutex_);
    const auto found_it = std::find_if(program_entries_.begin(), program_entries_.end(),
                                       [wrapped_program](const auto& key_value) {
                                           return key_value.second.program_ == wrapped_program;
                                       });
    assert(found_it != program_entries_.end());

    if (std::holds_alternative<std::string>(found_it->second.source_))
    {
        cl_program tmp_program{};
        cl_kernel tmp_kernel{};
        rebuild_kernel_from_source(kernel, found_it->second, tmp_kernel, tmp_program);
        error = tdispatch_->clGetKernelArgInfo(tmp_kernel, arg_index, param_name, param_value_size,
                                               param_value, param_value_size_ret);
        tdispatch_->clReleaseKernel(tmp_kernel);
        tdispatch_->clReleaseProgram(tmp_program);
    }
    else
    {
        error = tdispatch_->clGetKernelArgInfo(kernel, arg_index, param_name, param_value_size,
                                               param_value, param_value_size_ret);
    }
    return error;
}

bool pcl::program_cache_layer::is_il_program_supported(cl_context context) const
{
    std::size_t num_devices{};
    cl_int error = tdispatch_->clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES,
                                                sizeof(num_devices), &num_devices, nullptr);
    if (error != CL_SUCCESS) return false;
    std::vector<cl_device_id> devices(num_devices);
    error = tdispatch_->clGetContextInfo(
        context, CL_CONTEXT_DEVICES, num_devices * sizeof(cl_device_id), devices.data(), nullptr);
    if (error != CL_SUCCESS) return false;
    for (const auto& device : devices)
    {
        std::size_t il_version_length{};
        error = tdispatch_->clGetDeviceInfo(device, CL_DEVICE_IL_VERSION, 0, nullptr,
                                            &il_version_length);
        if (error != CL_SUCCESS || il_version_length <= 1) return false;
    }
    return true;
}

bool pcl::program_cache_layer::are_global_variables_supported(cl_context context) const
{
    std::size_t num_devices{};
    cl_int error = tdispatch_->clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES,
                                                sizeof(num_devices), &num_devices, nullptr);
    if (error != CL_SUCCESS) return false;
    std::vector<cl_device_id> devices(num_devices);
    error = tdispatch_->clGetContextInfo(
        context, CL_CONTEXT_DEVICES, num_devices * sizeof(cl_device_id), devices.data(), nullptr);
    if (error != CL_SUCCESS) return false;

    for (const auto& device : devices)
    {
        std::size_t max_global_var_size{};
        error =
            tdispatch_->clGetDeviceInfo(device, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE,
                                        sizeof(max_global_var_size), &max_global_var_size, nullptr);
        if (error != CL_SUCCESS || max_global_var_size == 0) return false;
    }
    return true;
}

cl_int pcl::program_cache_layer::ensure_program(program_entry& entry)
{
    cl_int error = CL_SUCCESS;
    if (entry.program_ == nullptr)
    {
        std::visit(utils::overloads{ [](binary_program) { assert(false); },
                                     [&](const std::string& source) {
                                         const char* strings = source.c_str();
                                         entry.program_ = tdispatch_->clCreateProgramWithSource(
                                             entry.context_, 1, &strings, nullptr, &error);
                                     },
                                     [&](const std::vector<char>& il) {
                                         entry.program_ = tdispatch_->clCreateProgramWithIL(
                                             entry.context_, il.data(), il.size(), &error);
                                     } },
                   entry.source_);
    }
    return error;
}

cl_int pcl::program_cache_layer::rebuild_kernel_from_source(cl_kernel old_kernel,
                                                            const program_entry& entry,
                                                            cl_kernel& new_kernel,
                                                            cl_program& new_program) const
{
    std::size_t kernel_name_size{};
    cl_int error = tdispatch_->clGetKernelInfo(old_kernel, CL_KERNEL_FUNCTION_NAME, 0, nullptr,
                                               &kernel_name_size);
    if (error != CL_SUCCESS) return error;
    std::string kernel_name;
    kernel_name.resize(kernel_name_size);
    error = tdispatch_->clGetKernelInfo(old_kernel, CL_KERNEL_FUNCTION_NAME, kernel_name_size,
                                        kernel_name.data(), nullptr);
    if (error != CL_SUCCESS) return error;
    assert(std::holds_alternative<std::string>(entry.source_));
    const auto& source = std::get<std::string>(entry.source_);
    const char* source_str = source.data();
    const std::size_t source_size = source.size();
    new_program =
        tdispatch_->clCreateProgramWithSource(entry.context_, 1, &source_str, &source_size, &error);
    if (error != CL_SUCCESS) return error;
    error = tdispatch_->clBuildProgram(new_program, 0, nullptr, entry.options_.c_str(), nullptr,
                                       nullptr);
    if (error != CL_SUCCESS)
    {
        tdispatch_->clReleaseProgram(new_program);
        return error;
    }
    new_kernel = tdispatch_->clCreateKernel(new_program, kernel_name.c_str(), &error);
    if (error != CL_SUCCESS)
    {
        tdispatch_->clReleaseProgram(new_program);
    }
    return error;
}
