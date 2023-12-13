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

/// @file program_cache_layer_surface.cpp
/// @brief Defines the entry points for the layer.
///
/// A global instance of \c ocl::program_cache::layer::program_cache_layer encompasses all data and
/// behaviour associated with the layer. The functions in this file redirect the OpenCL API calls to
/// the layer object. Two functions provide the layer's details to the ICD Loader: \c clGetLayerInfo
/// and \c clInitLayer, which are part of this shared library's binary interface.

#include "program_cache_layer.hpp"

#include <CL/cl_layer.h>

#include <cassert>
#include <memory>

namespace {

std::unique_ptr<ocl::program_cache::layer::program_cache_layer> g_program_cache_layer;
_cl_icd_dispatch g_dispatch{};

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource_wrap(cl_context context,
                                                                   cl_uint count,
                                                                   const char** strings,
                                                                   const size_t* lengths,
                                                                   cl_int* errcode_ret)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clCreateProgramWithSource(context, count, strings, lengths,
                                                            errcode_ret);
}

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithIL_wrap(cl_context context,
                                                               const void* il,
                                                               size_t length,
                                                               cl_int* errcode_ret)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clCreateProgramWithIL(context, il, length, errcode_ret);
}

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithBinary_wrap(cl_context context,
                                                                   cl_uint num_devices,
                                                                   const cl_device_id* device_list,
                                                                   const size_t* lengths,
                                                                   const unsigned char** binaries,
                                                                   cl_int* binary_status,
                                                                   cl_int* errcode_ret)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clCreateProgramWithBinary(
        context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
}

CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithBuiltInKernels_wrap(cl_context context,
                                       cl_uint num_devices,
                                       const cl_device_id* device_list,
                                       const char* kernel_names,
                                       cl_int* errcode_ret)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clCreateProgramWithBuiltInKernels(
        context, num_devices, device_list, kernel_names, errcode_ret);
}


CL_API_ENTRY cl_int CL_API_CALL clRetainProgram_wrap(cl_program program)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clRetainProgram(program);
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram_wrap(cl_program program)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clReleaseProgram(program);
}

CL_API_ENTRY cl_int CL_API_CALL
clSetProgramReleaseCallback_wrap(cl_program program,
                                 void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
                                 void* user_data)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clSetProgramReleaseCallback(program, pfn_notify, user_data);
}

CL_API_ENTRY cl_int CL_API_CALL clSetProgramSpecializationConstant_wrap(cl_program program,
                                                                        cl_uint spec_id,
                                                                        size_t spec_size,
                                                                        const void* spec_value)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clSetProgramSpecializationConstant(program, spec_id, spec_size,
                                                                     spec_value);
}

CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram_wrap(cl_program program,
                    cl_uint num_devices,
                    const cl_device_id* device_list,
                    const char* options,
                    void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
                    void* user_data)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clBuildProgram(program, num_devices, device_list, options,
                                                 pfn_notify, user_data);
}

CL_API_ENTRY cl_int CL_API_CALL
clCompileProgram_wrap(cl_program program,
                      cl_uint num_devices,
                      const cl_device_id* device_list,
                      const char* options,
                      cl_uint num_input_headers,
                      const cl_program* input_headers,
                      const char** header_include_names,
                      void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
                      void* user_data)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clCompileProgram(program, num_devices, device_list, options,
                                                   num_input_headers, input_headers,
                                                   header_include_names, pfn_notify, user_data);
}

CL_API_ENTRY cl_program CL_API_CALL
clLinkProgram_wrap(cl_context context,
                   cl_uint num_devices,
                   const cl_device_id* device_list,
                   const char* options,
                   cl_uint num_input_programs,
                   const cl_program* input_programs,
                   void(CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
                   void* user_data,
                   cl_int* errcode_ret)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clLinkProgram(context, num_devices, device_list, options,
                                                num_input_programs, input_programs, pfn_notify,
                                                user_data, errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clGetProgramInfo_wrap(cl_program program,
                                                      cl_program_info param_name,
                                                      size_t param_value_size,
                                                      void* param_value,
                                                      size_t* param_value_size_ret)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clGetProgramInfo(program, param_name, param_value_size,
                                                   param_value, param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clGetProgramBuildInfo_wrap(cl_program program,
                                                           cl_device_id device,
                                                           cl_program_build_info param_name,
                                                           size_t param_value_size,
                                                           void* param_value,
                                                           size_t* param_value_size_ret)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clGetProgramBuildInfo(
        program, device, param_name, param_value_size, param_value, param_value_size_ret);
}

CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel_wrap(cl_program program,
                                                       const char* kernel_name,
                                                       cl_int* errcode_ret)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clCreateKernel(program, kernel_name, errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clCreateKernelsInProgram_wrap(cl_program program,
                                                              cl_uint num_kernels,
                                                              cl_kernel* kernels,
                                                              cl_uint* num_kernels_ret)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clCreateKernelsInProgram(program, num_kernels, kernels,
                                                           num_kernels_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clGetKernelInfo_wrap(cl_kernel kernel,
                                                     cl_kernel_info param_name,
                                                     size_t param_value_size,
                                                     void* param_value,
                                                     size_t* param_value_size_ret)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clGetKernelInfo(kernel, param_name, param_value_size, param_value,
                                                  param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clGetKernelArgInfo_wrap(cl_kernel kernel,
                                                        cl_uint arg_index,
                                                        cl_kernel_arg_info param_name,
                                                        size_t param_value_size,
                                                        void* param_value,
                                                        size_t* param_value_size_ret)
{
    assert(g_program_cache_layer != nullptr);
    return g_program_cache_layer->clGetKernelArgInfo(
        kernel, arg_index, param_name, param_value_size, param_value, param_value_size_ret);
}

void init_dispatch()
{
    g_dispatch = {};
    g_dispatch.clCreateProgramWithSource = &clCreateProgramWithSource_wrap;
    g_dispatch.clCreateProgramWithIL = &clCreateProgramWithIL_wrap;
    g_dispatch.clCreateProgramWithBinary = &clCreateProgramWithBinary_wrap;
    g_dispatch.clCreateProgramWithBuiltInKernels = &clCreateProgramWithBuiltInKernels_wrap;
    g_dispatch.clRetainProgram = &clRetainProgram_wrap;
    g_dispatch.clReleaseProgram = &clReleaseProgram_wrap;
    g_dispatch.clSetProgramReleaseCallback = &clSetProgramReleaseCallback_wrap;
    g_dispatch.clSetProgramSpecializationConstant = &clSetProgramSpecializationConstant_wrap;
    g_dispatch.clBuildProgram = &clBuildProgram_wrap;
    g_dispatch.clCompileProgram = &clCompileProgram_wrap;
    g_dispatch.clLinkProgram = &clLinkProgram_wrap;
    g_dispatch.clGetProgramInfo = &clGetProgramInfo_wrap;
    g_dispatch.clGetProgramBuildInfo = &clGetProgramBuildInfo_wrap;
    g_dispatch.clCreateKernel = &clCreateKernel_wrap;
    g_dispatch.clCreateKernelsInProgram = &clCreateKernelsInProgram_wrap;
    g_dispatch.clGetKernelInfo = &clGetKernelInfo_wrap;
    g_dispatch.clGetKernelArgInfo = &clGetKernelArgInfo_wrap;
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
                if (param_value_size < sizeof(cl_layer_api_version)) return CL_INVALID_VALUE;
                *static_cast<cl_layer_api_version*>(param_value) = CL_LAYER_API_VERSION_100;
            }
            if (param_value_size_ret) *param_value_size_ret = sizeof(cl_layer_api_version);
            break;
        default: return CL_INVALID_VALUE;
    }
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clInitLayer(cl_uint num_entries,
                                            const struct _cl_icd_dispatch* target_dispatch,
                                            cl_uint* num_entries_out,
                                            const struct _cl_icd_dispatch** layer_dispatch_ret)
{
    if (!target_dispatch || !layer_dispatch_ret || !num_entries_out
        || num_entries < sizeof(g_dispatch) / sizeof(g_dispatch.clGetPlatformIDs))
        return CL_INVALID_VALUE;

    init_dispatch();
    *layer_dispatch_ret = &g_dispatch;
    *num_entries_out = sizeof(g_dispatch) / sizeof(g_dispatch.clGetPlatformIDs);
    try
    {
        g_program_cache_layer =
            std::make_unique<ocl::program_cache::layer::program_cache_layer>(target_dispatch);
    } catch (const std::exception&)
    {
        return CL_INVALID_VALUE;
    }
    return CL_SUCCESS;
}
