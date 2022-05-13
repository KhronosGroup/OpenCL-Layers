#include "object_lifetime_test_icd_surface.hpp"
#include "object_lifetime_test_icd.hpp"

#include <type_traits>  // std::remove_pointer_t
#include <algorithm>    // std::find_if

template <typename T, typename F>
cl_int invoke_if_valid(T cl_object, F&& f, bool retain = false)
{
  using namespace lifetime;

  auto it = std::find_if(
    get_objects<T>().cbegin(),
    get_objects<T>().cend(),
    [&](const std::shared_ptr<std::remove_pointer_t<T>>& obj)
    {
      return cl_object == obj.get();
    }
  );

  if (it != get_objects<T>().cend())
    if ((*it)->is_valid(retain))
      return f();
    else
      return CL_INVALID<T>();
  else
    return CL_INVALID<T>();
}

template<typename F>
cl_int invoke_if_valid(cl_platform_id platform, F&& f)
{
  using namespace lifetime;

  if (platform == &_platform)
    return f();
  else
    return CL_INVALID<cl_platform_id>();
}

template <typename T, typename F>
auto create_if_valid(T cl_object, cl_int* err, F&& f) -> decltype(f())
{
  using namespace lifetime;

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
      return f();
    else
    {
      if (err != nullptr)
        *err = CL_INVALID<T>();
      return nullptr;
    }
  else
  {
    if (err != nullptr)
      *err = CL_INVALID<T>();
    return nullptr;
  }
}

template<typename F>
auto create_if_valid(cl_platform_id platform, cl_int* err, F&& f) -> decltype(f())
{
  using namespace lifetime;

  if (platform == &_platform)
    return f();
  else
  {
    if (err != nullptr)
      *err = CL_INVALID<cl_platform_id>();
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

CL_API_ENTRY cl_context clCreateContext_wrap(
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

CL_API_ENTRY cl_int clGetContextInfo_wrap(
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
    return command_queue->clRetainCommandQueue();
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
    return CL_INVALID_VALUE;
  }

  if (platforms && num_entries == plat_count)
    platforms[0] = &_platform;

  return CL_SUCCESS;
}
