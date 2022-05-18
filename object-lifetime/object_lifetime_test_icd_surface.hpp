#pragma once

#include <CL/cl_icd.h>  // cl_icd_dispatch

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo_wrap(
  cl_platform_id platform,
  cl_platform_info param_name,
  size_t param_value_size,
  void *param_value,
  size_t *param_value_size_ret);

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs_wrap(
  cl_platform_id platform,
  cl_device_type device_type,
  cl_uint num_entries,
  cl_device_id* devices,
  cl_uint* num_devices);

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo_wrap(
  cl_device_id device,
  cl_device_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret);

CL_API_ENTRY cl_int CL_API_CALL clCreateSubDevices_wrap(
  cl_device_id in_device,
  const cl_device_partition_property* properties,
  cl_uint num_devices,
  cl_device_id* out_devices,
  cl_uint* num_devices_ret);

CL_API_ENTRY cl_int CL_API_CALL clRetainDevice_wrap(
  cl_device_id device);

CL_API_ENTRY cl_int CL_API_CALL clReleaseDevice_wrap(
  cl_device_id device);

CL_API_ENTRY cl_context CL_API_CALL clCreateContext_wrap(
  const cl_context_properties* properties,
  cl_uint num_devices,
  const cl_device_id* devices,
  void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
  void* user_data,
  cl_int* errcode_ret);

CL_API_ENTRY cl_int CL_API_CALL clGetContextInfo_wrap(
  cl_context context,
  cl_context_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret);

CL_API_ENTRY cl_int CL_API_CALL clRetainContext_wrap(
  cl_context context);

CL_API_ENTRY cl_int CL_API_CALL clReleaseContext_wrap(
  cl_context context);

CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer_wrap(
  cl_context context,
  cl_mem_flags flags,
  size_t size,
  void* host_ptr,
  cl_int* errcode_ret);

CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueue_wrap(
  cl_context context,
  cl_device_id device,
  cl_command_queue_properties properties,
  cl_int* errcode_ret);

CL_API_ENTRY cl_mem CL_API_CALL clCreateSubBuffer_wrap(
  cl_mem buffer,
  cl_mem_flags flags,
  cl_buffer_create_type buffer_create_type,
  const void* buffer_create_info,
  cl_int* errcode_ret);

CL_API_ENTRY cl_int CL_API_CALL clRetainMemObject_wrap(
  cl_mem memobj);

CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject_wrap(
  cl_mem memobj);

CL_API_ENTRY cl_int CL_API_CALL clGetMemObjectInfo_wrap(
  cl_mem memobj,
  cl_mem_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret);

CL_API_ENTRY cl_int CL_API_CALL clGetCommandQueueInfo_wrap(
  cl_command_queue command_queue,
  cl_command_queue_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret);

CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel_wrap(
  cl_command_queue command_queue,
  cl_kernel kernel,
  cl_uint work_dim,
  const size_t* global_work_offset,
  const size_t* global_work_size,
  const size_t* local_work_size,
  cl_uint num_events_in_wait_list,
  const cl_event* event_wait_list,
  cl_event* event);

CL_API_ENTRY cl_int CL_API_CALL clRetainCommandQueue_wrap(
  cl_command_queue command_queue);

CL_API_ENTRY cl_int CL_API_CALL clReleaseCommandQueue_wrap(
  cl_command_queue command_queue);

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource_wrap(
  cl_context context,
  cl_uint count,
  const char** strings,
  const size_t* lengths,
  cl_int* errcode_ret);

CL_API_ENTRY cl_int CL_API_CALL clBuildProgram_wrap(
  cl_program program,
  cl_uint num_devices,
  const cl_device_id* device_list,
  const char* options,
  void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
  void* user_data);

CL_API_ENTRY cl_int CL_API_CALL clGetProgramInfo_wrap(
  cl_program program,
  cl_program_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret);

CL_API_ENTRY cl_int CL_API_CALL clRetainProgram_wrap(
  cl_program program);

CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram_wrap(
  cl_program program);

CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel_wrap(
  cl_program program,
  const char* kernel_name,
  cl_int* errcode_ret);

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg_wrap(
  cl_kernel kernel,
  cl_uint arg_index,
  size_t arg_size,
  const void* arg_value);

CL_API_ENTRY cl_kernel CL_API_CALL clCloneKernel_wrap(
  cl_kernel source_kernel,
  cl_int* errcode_ret);

CL_API_ENTRY cl_int CL_API_CALL clGetKernelInfo_wrap(
  cl_kernel kernel,
  cl_kernel_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret);

CL_API_ENTRY cl_int CL_API_CALL clRetainKernel_wrap(
  cl_kernel kernel);

CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel_wrap(
  cl_kernel kernel);

CL_API_ENTRY cl_event CL_API_CALL clCreateUserEvent_wrap(
  cl_context context,
  cl_int* errcode_ret);

CL_API_ENTRY cl_int CL_API_CALL clSetUserEventStatus_wrap(
  cl_event event,
  cl_int execution_status);

CL_API_ENTRY cl_int CL_API_CALL clGetEventInfo_wrap(
  cl_event event,
  cl_event_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret);

CL_API_ENTRY cl_int CL_API_CALL clWaitForEvents_wrap(
  cl_uint num_events,
  const cl_event* event_list);

CL_API_ENTRY cl_int CL_API_CALL clRetainEvent_wrap(
  cl_event event);

CL_API_ENTRY cl_int CL_API_CALL clReleaseEvent_wrap(
  cl_event event);

CL_API_ENTRY cl_sampler CL_API_CALL clCreateSampler_wrap(
  cl_context context,
  cl_bool normalized_coords,
  cl_addressing_mode addressing_mode,
  cl_filter_mode filter_mode,
  cl_int* errcode_ret);

CL_API_ENTRY cl_int CL_API_CALL clGetSamplerInfo_wrap(
  cl_sampler sampler,
  cl_sampler_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret);

CL_API_ENTRY cl_int CL_API_CALL clRetainSampler_wrap(
  cl_sampler sampler);

CL_API_ENTRY cl_int CL_API_CALL clReleaseSampler_wrap(
  cl_sampler sampler);

// Loader hooks

CL_API_ENTRY void* CL_API_CALL clGetExtensionFunctionAddress(
  const char* name);

CL_API_ENTRY cl_int CL_API_CALL
clIcdGetPlatformIDsKHR(
  cl_uint         num_entries,
  cl_platform_id* platforms,
  cl_uint*        num_platforms);
