#include "object_lifetime_test_icd_surface.hpp"
#include "object_lifetime_test_icd.hpp"

#include <type_traits>  // std::remove_pointer_t
#include <algorithm>    // std::find_if
#include <mutex>        // std::lock_guard

template <typename T, typename F>
cl_int invoke_if_valid(T cl_object, F&& f, bool retain = false)
{
  using namespace lifetime;

  std::lock_guard<std::recursive_mutex> lock{lifetime::get_mutex<T>()};

  auto it = std::find_if(
    get_objects<T>().cbegin(),
    get_objects<T>().cend(),
    [&](const std::shared_ptr<std::remove_pointer_t<T>>& obj)
    {
      return cl_object == obj.get();
    }
  );

  cl_int result;
  if (it != get_objects<T>().cend())
    if ((*it)->is_valid(retain))
      result = f();
    else
      result = CL_INVALID<T>();
  else
    result = CL_INVALID<T>();
  return always_return_success ? CL_SUCCESS : result;
}

template<typename F>
cl_int invoke_if_valid(cl_platform_id platform, F&& f)
{
  using namespace lifetime;

  cl_int result;
  if (platform == &_platform)
    result = f();
  else
    result = CL_INVALID<cl_platform_id>();
  return always_return_success ? CL_SUCCESS : result;
}

template <typename T, typename F>
auto create_if_valid(T cl_object, cl_int* err, F&& f) -> decltype(f())
{
  using namespace lifetime;

  std::lock_guard<std::recursive_mutex> lock{lifetime::get_mutex<T>()};

  auto it = std::find_if(
    get_objects<T>().cbegin(),
    get_objects<T>().cend(),
    [&](const std::shared_ptr<std::remove_pointer_t<T>>& obj)
    {
      return cl_object == obj.get();
    }
  );

  if (it != get_objects<T>().cend())
    if ((*it)->is_valid(false))
    {
      auto object = f();
      if (always_return_success && err != nullptr)
        *err = CL_SUCCESS;
      return object;
    }
    else
    {
      if (err != nullptr)
        *err = always_return_success ? CL_SUCCESS : CL_INVALID<T>();
      return nullptr;
    }
  else
  {
    if (err != nullptr)
      *err = always_return_success ? CL_SUCCESS : CL_INVALID<T>();
    return nullptr;
  }
}

template<typename F>
auto create_if_valid(cl_platform_id platform, cl_int* err, F&& f) -> decltype(f())
{
  using namespace lifetime;

  if (platform == &_platform)
  {
    auto object = f();
    if (always_return_success && err != nullptr)
      *err = CL_SUCCESS;
    return object;
  }
  else
  {
    if (err != nullptr)
      *err = always_return_success ? CL_SUCCESS : CL_INVALID<cl_platform_id>();
    return nullptr;
  }
}

template <typename T>
T find_null_terminator(T first)
{
  T result = first;
  while(*result != static_cast<decltype(*first)>(0))
    result++;
  return result;
}

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo_wrap(
  cl_platform_id platform,
  cl_platform_info param_name,
  size_t param_value_size,
  void *param_value,
  size_t *param_value_size_ret)
{
  return invoke_if_valid(platform, [&]()
  {
    return platform->clGetPlatformInfo(
      param_name,
      param_value_size,
      param_value,
      param_value_size_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs_wrap(
  cl_platform_id platform,
  cl_device_type device_type,
  cl_uint num_entries,
  cl_device_id* devices,
  cl_uint* num_devices)
{
  return invoke_if_valid(platform, [&]()
  {
    return platform->clGetDeviceIDs(
      device_type,
      num_entries,
      devices,
      num_devices
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo_wrap(
  cl_device_id device,
  cl_device_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  return invoke_if_valid(device, [&]()
  {
    return device->clGetDeviceInfo(
      param_name,
      param_value_size,
      param_value,
      param_value_size_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clCreateSubDevices_wrap(
  cl_device_id in_device,
  const cl_device_partition_property* properties,
  cl_uint num_devices,
  cl_device_id* out_devices,
  cl_uint* num_devices_ret)
{
  return invoke_if_valid(in_device, [&]()
  {
    return in_device->clCreateSubDevices(
      properties,
      num_devices,
      out_devices,
      num_devices_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clRetainDevice_wrap(
  cl_device_id device)
{
  return invoke_if_valid(device, [&]()
  {
    return device->clRetainDevice();
  },
    true
  );
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseDevice_wrap(
  cl_device_id device)
{
  return invoke_if_valid(device, [&]()
  {
    return device->clReleaseDevice();
  });
}

CL_API_ENTRY cl_context CL_API_CALL clCreateContext_wrap(
  const cl_context_properties* properties,
  cl_uint num_devices,
  const cl_device_id* devices,
  void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
  void* user_data,
  cl_int* errcode_ret)
{
  return create_if_valid(devices[0], errcode_ret, [&]()
  {
    return devices[0]->clCreateContext(
      properties,
      num_devices,
      devices,
      pfn_notify,
      user_data,
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clGetContextInfo_wrap(
  cl_context context,
  cl_context_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  return invoke_if_valid(context, [&]()
  {
    return context->clGetContextInfo(
      param_name,
      param_value_size,
      param_value,
      param_value_size_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clRetainContext_wrap(
  cl_context context)
{
  return invoke_if_valid(context, [&]()
  {
    return context->clRetainContext();
  },
    true
  );
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseContext_wrap(
  cl_context context)
{
  return invoke_if_valid(context, [&]()
  {
    return context->clReleaseContext();
  });
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer_wrap(
  cl_context context,
  cl_mem_flags flags,
  size_t size,
  void* host_ptr,
  cl_int* errcode_ret)
{
  return create_if_valid(context, errcode_ret, [&]()
  {
    return context->clCreateBuffer(
      flags,
      size,
      host_ptr,
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateBufferWithProperties_wrap(
  cl_context context,
  const cl_mem_properties* properties,
  cl_mem_flags flags,
  size_t size,
  void* host_ptr,
  cl_int* errcode_ret)
{
  return create_if_valid(context, errcode_ret, [&]()
  {
    return context->clCreateBufferWithProperties(
      properties,
      flags,
      size,
      host_ptr,
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateImage_wrap(
  cl_context context,
  cl_mem_flags flags,
  const cl_image_format* image_format,
  const cl_image_desc* image_desc,
  void* host_ptr,
  cl_int* errcode_ret)
{
  return create_if_valid(context, errcode_ret, [&]()
  {
    return context->clCreateImage(
      flags,
      image_format,
      image_desc,
      host_ptr,
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateImageWithProperties_wrap(
  cl_context context,
  const cl_mem_properties* properties,
  cl_mem_flags flags,
  const cl_image_format* image_format,
  const cl_image_desc* image_desc,
  void* host_ptr,
  cl_int* errcode_ret)
{
  return create_if_valid(context, errcode_ret, [&]()
  {
    return context->clCreateImageWithProperties(
      properties,
      flags,
      image_format,
      image_desc,
      host_ptr,
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateImage2D_wrap(
  cl_context context,
  cl_mem_flags flags,
  const cl_image_format* image_format,
  size_t image_width,
  size_t image_height,
  size_t image_row_pitch,
  void* host_ptr,
  cl_int* errcode_ret)
{
  return create_if_valid(context, errcode_ret, [&]()
  {
    return context->clCreateImage2D(
      flags,
      image_format,
      image_width,
      image_height,
      image_row_pitch,
      host_ptr,
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateImage3D_wrap(
  cl_context context,
  cl_mem_flags flags,
  const cl_image_format* image_format,
  size_t image_width,
  size_t image_height,
  size_t image_depth,
  size_t image_row_pitch,
  size_t image_slice_pitch,
  void* host_ptr,
  cl_int* errcode_ret)
{
  return create_if_valid(context, errcode_ret, [&]()
  {
    return context->clCreateImage3D(
      flags,
      image_format,
      image_width,
      image_height,
      image_depth,
      image_row_pitch,
      image_slice_pitch,
      host_ptr,
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_mem CL_API_CALL clCreatePipe_wrap(
  cl_context context,
  cl_mem_flags flags,
  cl_uint pipe_packet_size,
  cl_uint pipe_max_packets,
  const cl_pipe_properties* properties,
  cl_int* errcode_ret)
{
  return create_if_valid(context, errcode_ret, [&]()
  {
    return context->clCreatePipe(
      flags,
      pipe_packet_size,
      pipe_max_packets,
      properties,
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueue_wrap(
  cl_context context,
  cl_device_id device,
  cl_command_queue_properties properties,
  cl_int* errcode_ret)
{
  return create_if_valid(context, errcode_ret, [&]()
  {
    return context->clCreateCommandQueue(
      device,
      properties,
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueueWithProperties_wrap(
  cl_context context,
  cl_device_id device,
  const cl_queue_properties* properties,
  cl_int* errcode_ret)
{
  return create_if_valid(context, errcode_ret, [&]()
  {
    return context->clCreateCommandQueueWithProperties(
      device,
      properties,
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel_wrap(
  cl_command_queue command_queue,
  cl_kernel kernel,
  cl_uint work_dim,
  const size_t* global_work_offset,
  const size_t* global_work_size,
  const size_t* local_work_size,
  cl_uint num_events_in_wait_list,
  const cl_event* event_wait_list,
  cl_event* event)
{
  return invoke_if_valid(command_queue, [&]()
  {
    return command_queue->clEnqueueNDRangeKernel(
      kernel,
      work_dim,
      global_work_offset,
      global_work_size,
      local_work_size,
      num_events_in_wait_list,
      event_wait_list,
      event
    );
  });
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateSubBuffer_wrap(
  cl_mem buffer,
  cl_mem_flags flags,
  cl_buffer_create_type buffer_create_type,
  const void* buffer_create_info,
  cl_int* errcode_ret)
{
  return create_if_valid(buffer, errcode_ret, [&]()
  {
    return buffer->clCreateSubBuffer(
      flags,
      buffer_create_type,
      buffer_create_info,
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clRetainMemObject_wrap(
  cl_mem memobj)
{
  return invoke_if_valid(memobj, [&]()
  {
    return memobj->clRetainMemObject();
  },
    true
  );
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject_wrap(
  cl_mem memobj)
{
  return invoke_if_valid(memobj, [&]()
  {
    return memobj->clReleaseMemObject();
  });
}

CL_API_ENTRY cl_int CL_API_CALL clGetMemObjectInfo_wrap(
  cl_mem memobj,
  cl_mem_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  return invoke_if_valid(memobj, [&]()
  {
    return memobj->clGetMemObjectInfo(
      param_name,
      param_value_size,
      param_value,
      param_value_size_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clGetCommandQueueInfo_wrap(
  cl_command_queue command_queue,
  cl_command_queue_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  return invoke_if_valid(command_queue, [&]()
  {
    return command_queue->clGetCommandQueueInfo(
      param_name,
      param_value_size,
      param_value,
      param_value_size_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clRetainCommandQueue_wrap(
  cl_command_queue command_queue)
{
  return invoke_if_valid(command_queue, [&]()
  {
    return command_queue->clRetainCommandQueue();
  },
    true
  );
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseCommandQueue_wrap(
  cl_command_queue command_queue)
{
  return invoke_if_valid(command_queue, [&]()
  {
    return command_queue->clReleaseCommandQueue();
  });
}

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource_wrap(
  cl_context context,
  cl_uint count,
  const char** strings,
  const size_t* lengths,
  cl_int* errcode_ret)
{
  return create_if_valid(context, errcode_ret, [&]()
  {
    return context->clCreateProgramWithSource(
      count,
      strings,
      lengths,
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clBuildProgram_wrap(
  cl_program program,
  cl_uint num_devices,
  const cl_device_id* device_list,
  const char* options,
  void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
  void* user_data)
{
  return invoke_if_valid(program, [&]()
  {
    return program->clBuildProgram(
      num_devices,
      device_list,
      options,
      pfn_notify,
      user_data
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clGetProgramInfo_wrap(
  cl_program program,
  cl_program_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  return invoke_if_valid(program, [&]()
  {
    return program->clGetProgramInfo(
      param_name,
      param_value_size,
      param_value,
      param_value_size_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clRetainProgram_wrap(
  cl_program program)
{
  return invoke_if_valid(program, [&]()
  {
    return program->clRetainProgram();
  },
    true
  );
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram_wrap(
  cl_program program)
{
  return invoke_if_valid(program, [&]()
  {
    return program->clReleaseProgram();
  });
}

CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel_wrap(
  cl_program program,
  const char* kernel_name,
  cl_int* errcode_ret)
{
  return create_if_valid(program, errcode_ret, [&]()
  {
    return program->clCreateKernel(
      kernel_name,
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clCreateKernelsInProgram_wrap(
  cl_program program,
  cl_uint num_kernels,
  cl_kernel* kernels,
  cl_uint* num_kernels_ret)
{
  return invoke_if_valid(program, [&]()
  {
    return program->clCreateKernelsInProgram(
      num_kernels,
      kernels,
      num_kernels_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg_wrap(
  cl_kernel kernel,
  cl_uint arg_index,
  size_t arg_size,
  const void* arg_value)
{
  return invoke_if_valid(kernel, [&]()
  {
    return kernel->clSetKernelArg(
      arg_index,
      arg_size,
      arg_value
    );
  });
}

CL_API_ENTRY cl_kernel CL_API_CALL clCloneKernel_wrap(
  cl_kernel source_kernel,
  cl_int* errcode_ret)
{
  return create_if_valid(source_kernel, errcode_ret, [&]()
  {
    return source_kernel->clCloneKernel(
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clGetKernelInfo_wrap(
  cl_kernel kernel,
  cl_kernel_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  return invoke_if_valid(kernel, [&]()
  {
    return kernel->clGetKernelInfo(
      param_name,
      param_value_size,
      param_value,
      param_value_size_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clRetainKernel_wrap(
  cl_kernel kernel)
{
  return invoke_if_valid(kernel, [&]()
  {
    return kernel->clRetainKernel();
  },
    true
  );
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel_wrap(
  cl_kernel kernel)
{
  return invoke_if_valid(kernel, [&]()
  {
    return kernel->clReleaseKernel();
  });
}

CL_API_ENTRY cl_event CL_API_CALL clCreateUserEvent_wrap(
  cl_context context,
  cl_int* errcode_ret)
{
  return create_if_valid(context, errcode_ret, [&]()
  {
    return context->clCreateUserEvent(
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clSetUserEventStatus_wrap(
  cl_event event,
  cl_int execution_status)
{
  return invoke_if_valid(event, [&]()
  {
    return event->clSetUserEventStatus(
      execution_status);
  });
}

CL_API_ENTRY cl_int CL_API_CALL clGetEventInfo_wrap(
  cl_event event,
  cl_event_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  return invoke_if_valid(event, [&]()
  {
    return event->clGetEventInfo(
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
  });
}

CL_API_ENTRY cl_int CL_API_CALL clWaitForEvents_wrap(
  cl_uint num_events,
  const cl_event* event_list)
{
  if (num_events != 0 && event_list != nullptr)
    return invoke_if_valid(event_list[0], [&]()
    {
      return event_list[0]->clWaitForEvents(
        num_events,
        event_list
      );
    });
  else
    return lifetime::always_return_success ? CL_SUCCESS : CL_INVALID_EVENT;
}

CL_API_ENTRY cl_int CL_API_CALL clRetainEvent_wrap(
  cl_event event)
{
  return invoke_if_valid(event, [&]()
  {
    return event->clRetainEvent();
  },
    true
  );
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseEvent_wrap(
  cl_event event)
{
  return invoke_if_valid(event, [&]()
  {
    return event->clReleaseEvent();
  });
}

CL_API_ENTRY cl_sampler CL_API_CALL clCreateSampler_wrap(
  cl_context context,
  cl_bool normalized_coords,
  cl_addressing_mode addressing_mode,
  cl_filter_mode filter_mode,
  cl_int* errcode_ret)
{
  return create_if_valid(context, errcode_ret, [&]()
  {
    return context->clCreateSampler(
      normalized_coords,
      addressing_mode,
      filter_mode,
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_sampler CL_API_CALL clCreateSamplerWithProperties_wrap(
  cl_context context,
  const cl_sampler_properties* sampler_properties,
  cl_int* errcode_ret)
{
  return create_if_valid(context, errcode_ret, [&]()
  {
    return context->clCreateSamplerWithProperties(
      sampler_properties,
      errcode_ret
    );
  });
}

CL_API_ENTRY cl_int CL_API_CALL clGetSamplerInfo_wrap(
  cl_sampler sampler,
  cl_sampler_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  return invoke_if_valid(sampler, [&]()
  {
    return sampler->clGetSamplerInfo(
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
  });
}

CL_API_ENTRY cl_int CL_API_CALL clRetainSampler_wrap(
  cl_sampler sampler)
{
  return invoke_if_valid(sampler, [&]()
  {
    return sampler->clRetainSampler();
  },
    true
  );
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseSampler_wrap(
  cl_sampler sampler)
{
  return invoke_if_valid(sampler, [&]()
  {
    return sampler->clReleaseSampler();
  });
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueFillBuffer_wrap(
  cl_command_queue command_queue,
    cl_mem,
    const void*,
    size_t,
    size_t,
    size_t,
    cl_uint,
    const cl_event*,
    cl_event*)
{
  return invoke_if_valid(command_queue, [&]()
  {
    return CL_SUCCESS;
  });
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyImageToBuffer_wrap(
    cl_command_queue command_queue,
    cl_mem,
    cl_mem,
    const size_t*,
    const size_t*,
    size_t,
    cl_uint,
    const cl_event*,
    cl_event*)
{
  return invoke_if_valid(command_queue, [&]()
  {
    return CL_SUCCESS;
  });
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyImage_wrap(
    cl_command_queue command_queue,
    cl_mem,
    cl_mem,
    const size_t*,
    const size_t*,
    const size_t*,
    cl_uint,
    const cl_event*,
    cl_event*)
{
  return invoke_if_valid(command_queue, [&]()
  {
    return CL_SUCCESS;
  });
}

CL_API_ENTRY cl_int CL_API_CALL clGetImageInfo_wrap(
    cl_mem image,
    cl_image_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  return invoke_if_valid(image, [&]()
  {
    return image->clGetImageInfo(param_name, param_value_size, param_value, param_value_size_ret);
  });
}

CL_API_ENTRY cl_int CL_API_CALL clGetSupportedImageFormats_wrap(
    cl_context context,
    cl_mem_flags flags,
    cl_mem_object_type image_type,
    cl_uint num_entries,
    cl_image_format* image_formats,
    cl_uint* num_image_formats)
{
  return invoke_if_valid(context, [&]()
  {
    return context->clGetSupportedImageFormats(flags, image_type, num_entries, image_formats, num_image_formats);
  });
}

// Loader hooks

CL_API_ENTRY void* CL_API_CALL clGetExtensionFunctionAddress(
  const char* name)
{
  using namespace lifetime;

  auto it = _extensions.find(name);
  if (it != _extensions.end())
    return it->second;
  else
    return nullptr;
}

CL_API_ENTRY cl_int CL_API_CALL
clIcdGetPlatformIDsKHR(
  cl_uint         num_entries,
  cl_platform_id* platforms,
  cl_uint*        num_platforms)
{
  using namespace lifetime;
  static constexpr cl_uint plat_count = 1;

  if (num_platforms)
    *num_platforms = plat_count;

  if ((platforms && num_entries > plat_count) ||
      (platforms && num_entries <= 0) ||
      (!platforms && num_entries >= 1))
  {
    return always_return_success ? CL_SUCCESS : CL_INVALID_VALUE;
  }

  if (platforms && num_entries == plat_count)
    platforms[0] = &_platform;

  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo(
  cl_platform_id platform,
  cl_platform_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  return clGetPlatformInfo_wrap(platform,
                                param_name,
                                param_value_size,
                                param_value,
                                param_value_size_ret);
}
