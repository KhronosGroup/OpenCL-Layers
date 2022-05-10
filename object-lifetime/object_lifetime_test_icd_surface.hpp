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

CL_API_ENTRY cl_context clCreateContext_wrap(
  const cl_context_properties* properties,
  cl_uint num_devices,
  const cl_device_id* devices,
  void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
  void* user_data,
  cl_int* errcode_ret);

CL_API_ENTRY cl_int clGetContextInfo_wrap(
  cl_context context,
  cl_context_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret);

CL_API_ENTRY cl_int CL_API_CALL clRetainContext_wrap(
  cl_context context);

CL_API_ENTRY cl_int CL_API_CALL clReleaseContext_wrap(
  cl_context context);

// Loader hooks

CL_API_ENTRY void* CL_API_CALL clGetExtensionFunctionAddress(
  const char* name);

CL_API_ENTRY cl_int CL_API_CALL
clIcdGetPlatformIDsKHR(
  cl_uint         num_entries,
  cl_platform_id* platforms,
  cl_uint*        num_platforms);
