#ifndef CL_USE_DEPRECATED_OPENCL_1_0_APIS
#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_2_1_APIS
#define CL_USE_DEPRECATED_OPENCL_2_1_APIS
#endif

#ifndef CL_USE_DEPRECATED_OPENCL_2_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_2_APIS
#endif

#include <stdlib.h>
#include <string.h>
#include <CL/cl_layer.h>
#include <mutex>
#include <tuple>
#include <map>
#include <iostream>
#include <list>

typedef enum object_type_e {
  PLATFORM,
  DEVICE,
  SUB_DEVICE,
  CONTEXT,
  COMMAND_QUEUE,
  MEM,
  BUFFER,
  IMAGE,
  PIPE,
  PROGRAM,
  KERNEL,
  EVENT,
  SAMPLER,
  OBJECT_TYPE_MAX,
} object_type;

static const char * object_type_names[] = {
  "PLATFORM",
  "DEVICE",
  "SUB_DEVICE",
  "CONTEXT",
  "COMMAND_QUEUE",
  "MEM",
  "BUFFER",
  "IMAGE",
  "PIPE",
  "PROGRAM",
  "KERNEL",
  "EVENT",
  "SAMPLER"
};

static inline constexpr std::string_view rtrim(const std::string_view s) {
  return s.substr(0, s.size() - 5);
}

#define RTRIM_FUNC rtrim(__func__)

typedef std::tuple<object_type, ssize_t> type_count;

std::map<void*, type_count> objects;
std::map<void*, std::list<type_count>> deleted_objects;
std::mutex objects_mutex;

static void error_already_exist(const std::string_view &func, void *handle, object_type t, ssize_t ref_count) {
  std::cerr << "In " << func << " " <<
               object_type_names[t] <<
               ": " << handle <<
               " already exist with refcount: " << ref_count << "\n";
  std::cerr.flush();
}

static void error_ref_count(const std::string_view &func, void *handle, object_type t, ssize_t ref_count) {
  std::cerr << "In " << func << " " <<
               object_type_names[t] <<
               ": " << handle <<
               " used with refcount: " << ref_count << "\n";
  std::cerr.flush();
}

static void error_invalid_type(const std::string_view &func, void *handle, object_type t, object_type expect) {
  std::cerr << "In " << func << " " <<
               object_type_names[t] <<
               ": " << handle <<
               " was used whereas function expects: " <<
               object_type_names[expect] << "\n";
  std::cerr.flush();
}

static void error_does_not_exist(const std::string_view &func, void *handle, object_type t) {
  std::cerr << "In " << func << " " <<
               object_type_names[t] <<
               ": " << handle <<
               " was used but ";
  auto it = deleted_objects.find(handle);
  if (it == deleted_objects.end()) {
    std::cerr << "it does not exist" << "\n";
  } else {
    std::cerr << "it was recently deleted with type: " <<
                 object_type_names[std::get<0>(it->second.back())] << "\n";
  }
  std::cerr.flush();
}

static void error_invalid_release(const std::string_view &func, void *handle, object_type t) {
  std::cerr << "In " << func << " " <<
               object_type_names[t] <<
               ": " << handle <<
               " was released before being retained" << "\n";
  std::cerr.flush();
}

template<object_type T>
static void check_exists(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, T);
  } else if (std::get<0>(it->second) != T) {
    error_invalid_type(func, handle, std::get<0>(it->second), T);
  } else if (std::get<1>(it->second) <= 0) {
    error_ref_count(func, handle, T, std::get<1>(it->second));
  }
  objects_mutex.unlock();
}

template<>
void check_exists<PLATFORM>(const std::string_view &func, void *handle) {
  if (!handle)
    return;
  objects_mutex.lock();
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, PLATFORM);
  } else if (std::get<0>(it->second) != PLATFORM) {
    error_invalid_type(func, handle, std::get<0>(it->second), PLATFORM);
  }
  objects_mutex.unlock();
}

template<>
void check_exists<DEVICE>(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, DEVICE);
  } else if (std::get<0>(it->second) != DEVICE && std::get<0>(it->second) != SUB_DEVICE) {
    error_invalid_type(func, handle, std::get<0>(it->second), DEVICE);
  } else if (std::get<0>(it->second) == SUB_DEVICE && std::get<1>(it->second) <= 0) {
    error_ref_count(func, handle, SUB_DEVICE, std::get<1>(it->second));
  }
  objects_mutex.unlock();
}

template<>
void check_exists<MEM>(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, MEM);
  } else {
    object_type t = std::get<0>(it->second);
    switch (t) {
    case BUFFER:
    case IMAGE:
    case PIPE:
      if (std::get<1>(it->second) <= 0) {
        error_ref_count(func, handle, t, std::get<1>(it->second));
      }
      break;
    default:
      error_invalid_type(func, handle, std::get<0>(it->second), t);
    }
  }
  objects_mutex.unlock();
}

#define CHECK_EXISTS(type, handle) check_exists<type>(RTRIM_FUNC, handle)

template<object_type T>
static void check_exist_list(const std::string_view &func, cl_uint num_handles, void **handles) {
  if (!handles)
    return;
  objects_mutex.lock();
  for (cl_uint i = 0; i < num_handles; i++) {
    auto it = objects.find(handles[i]);
    if (it == objects.end()) {
      error_does_not_exist(func, handles[i], T);
    } else if (std::get<0>(it->second) != T) {
      error_invalid_type(func, handles[i], std::get<0>(it->second), T);
    } else if (std::get<1>(it->second) <= 0) {
      error_ref_count(func, handles[i], T, std::get<1>(it->second));
    }
  }
  objects_mutex.unlock();
}

template<>
void check_exist_list<DEVICE>(const std::string_view &func, cl_uint num_handles, void **handles) {
  if (!handles)
    return;
  objects_mutex.lock();
  for (cl_uint i = 0; i < num_handles; i++) {
    auto it = objects.find(handles[i]);
    if (it == objects.end()) {
      error_does_not_exist(func, handles[i], DEVICE);
    } else if (std::get<0>(it->second) != DEVICE && std::get<0>(it->second) != SUB_DEVICE) {
      error_invalid_type(func, handles[i], std::get<0>(it->second), DEVICE);
    } else if (std::get<0>(it->second) == SUB_DEVICE && std::get<1>(it->second) <= 0) {
      error_ref_count(func, handles[i], SUB_DEVICE, std::get<1>(it->second));
    }
  }
  objects_mutex.unlock();
}

#define CHECK_EXIST_LIST(type, num_handles, handles) check_exist_list<type>(RTRIM_FUNC, num_handles, (void **)handles)

template<object_type T>
static void check_creation_list(const std::string_view &func, cl_uint num_handles, void **handles) {
  objects_mutex.lock();
  for (cl_uint i = 0; i < num_handles; i++) {
    auto it = objects.find(handles[i]);
    if (it != objects.end()) {
      if (std::get<1>(it->second) > 0) {
        error_already_exist(func, handles[i], std::get<0>(it->second), std::get<1>(it->second));
        deleted_objects[handles[i]].push_back(it->second);
      }
    }
    objects[handles[i]] = type_count(T, 1);
  }
  objects_mutex.unlock();
}

template<>
void check_creation_list<DEVICE>(const std::string_view &func, cl_uint num_handles, void **handles) {
  objects_mutex.lock();
  for (cl_uint i = 0; i < num_handles; i++) {
    auto it = objects.find(handles[i]);
    if (it != objects.end() && std::get<0>(it->second) != DEVICE) {
      error_already_exist(func, handles[i], std::get<0>(it->second), std::get<1>(it->second));
      deleted_objects[handles[i]].push_back(it->second);
    }
    objects[handles[i]] = type_count(DEVICE, 0);
  }
  objects_mutex.unlock();
}

template<>
void check_creation_list<PLATFORM>(const std::string_view &func, cl_uint num_handles, void **handles) {
  objects_mutex.lock();
  for (cl_uint i = 0; i < num_handles; i++) {
    auto it = objects.find(handles[i]);
    if (it != objects.end() && std::get<0>(it->second) != PLATFORM) {
      error_already_exist(func, handles[i], std::get<0>(it->second), std::get<1>(it->second));
      deleted_objects[handles[i]].push_back(it->second);
    }
    objects[handles[i]] = type_count(PLATFORM, 0);
  }
  objects_mutex.unlock();
}

#define CHECK_CREATION_LIST(type, num_handles, handles) check_creation_list<type>(RTRIM_FUNC, num_handles, (void **)handles)

template<object_type T>
static void check_creation(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  auto it = objects.find(handle);
  if (it != objects.end()) {
    if (std::get<1>(it->second) > 0) {
      error_already_exist(func, handle, std::get<0>(it->second), std::get<1>(it->second));
      deleted_objects[handle].push_back(it->second);
    }
  }
  objects[handle] = type_count(T, 1);
  objects_mutex.unlock();
}

#define CHECK_CREATION(type, handle) check_creation<type>(RTRIM_FUNC, handle)

template<object_type T>
static void check_add_or_exists(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  auto it = objects.find(handle);
  if (it != objects.end()) {
    if (std::get<0>(it->second) != T) {
      if (std::get<1>(it->second) > 0) {
        error_already_exist(func, handle, std::get<0>(it->second), std::get<1>(it->second));
        deleted_objects[handle].push_back(it->second);
      }
      objects[handle] = type_count(T, 0);
    }
  } else {
    objects[handle] = type_count(T, 0);
  }
  objects_mutex.unlock();
}

#define CHECK_ADD_OR_EXISTS(type, handle) check_add_or_exists<type>(RTRIM_FUNC, handle)

template<object_type T>
static void check_release(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, T);
  } else {
    object_type t = std::get<0>(it->second);
    if (t == T) {
      std::get<1>(it->second) -= 1;
      if (std::get<1>(it->second) < 0) {
        error_invalid_release(func, handle, T);
      }
    } else {
      error_invalid_type(func, handle, t, T);
    }
  }
  objects_mutex.unlock();
}

template<>
void check_release<DEVICE>(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, DEVICE);
  } else {
    object_type t = std::get<0>(it->second);
    switch (t) {
    case DEVICE:
      break;
    case SUB_DEVICE:
      std::get<1>(it->second) -= 1;
      if (std::get<1>(it->second) < 0) {
        error_invalid_release(func, handle, SUB_DEVICE);
      }
      break;
    default:
      error_invalid_type(func, handle, t, DEVICE);
    }
  }
  objects_mutex.unlock();
}

template<>
void check_release<MEM>(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, MEM);
  } else {
    object_type t = std::get<0>(it->second);
    switch (t) {
    case BUFFER:
    case IMAGE:
    case PIPE:
      std::get<1>(it->second) -= 1;
      if (std::get<1>(it->second) == 0) {
        deleted_objects[handle].push_back(it->second);
        objects.erase(it);
      } else if (std::get<1>(it->second) < 0) {
        error_invalid_release(func, handle, t);
      }
      break;
    default:
      error_invalid_type(func, handle, t, MEM);
    }
  }
  objects_mutex.unlock();
}

#define CHECK_RELEASE(type, handle) check_release<type>(RTRIM_FUNC, handle);

template<object_type T>
static void check_retain(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, T);
  } else {
    object_type t = std::get<0>(it->second);
    if (t == T) {
      std::get<1>(it->second) += 1;
    } else {
      error_invalid_type(func, handle, t, T);
    }
  }
  objects_mutex.unlock();
}

template<>
void check_retain<DEVICE>(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, DEVICE);
  } else {
    object_type t = std::get<0>(it->second);
    switch (t) {
    case DEVICE:
      break;
    case SUB_DEVICE:
      std::get<1>(it->second) += 1;
      break;
    default:
      error_invalid_type(func, handle, t, DEVICE);
    }
  }
  objects_mutex.unlock();
}

template<>
void check_retain<MEM>(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, MEM);
  } else {
    object_type t = std::get<0>(it->second);
    switch (t) {
    case BUFFER:
    case IMAGE:
    case PIPE:
      std::get<1>(it->second) += 1;
      break;
    default:
      error_invalid_type(func, handle, t, MEM);
    }
  }
  objects_mutex.unlock();
}

#define CHECK_RETAIN(type, handle) check_retain<type>(RTRIM_FUNC, handle)

static struct _cl_icd_dispatch dispatch = {};

static const struct _cl_icd_dispatch *tdispatch;

CL_API_ENTRY cl_int CL_API_CALL
clGetLayerInfo(
    cl_layer_info  param_name,
    size_t         param_value_size,
    void          *param_value,
    size_t        *param_value_size_ret) {
  if (param_value_size && !param_value)
    return CL_INVALID_VALUE;
  if (!param_value && !param_value_size_ret)
    return CL_INVALID_VALUE;
  switch (param_name) {
  case CL_LAYER_API_VERSION:
    if (param_value_size < sizeof(cl_layer_api_version))
      return CL_INVALID_VALUE;
    if (param_value)
      *((cl_layer_api_version *)param_value) = CL_LAYER_API_VERSION_100;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(cl_layer_api_version);
    break;
  default:
    return CL_INVALID_VALUE;
  }
  return CL_SUCCESS;
}

static void _init_dispatch(void);

CL_API_ENTRY cl_int CL_API_CALL
clInitLayer(
    cl_uint                         num_entries,
    const struct _cl_icd_dispatch  *target_dispatch,
    cl_uint                        *num_entries_out,
    const struct _cl_icd_dispatch **layer_dispatch_ret) {
  if (!target_dispatch || !layer_dispatch_ret ||!num_entries_out || num_entries < sizeof(dispatch)/sizeof(dispatch.clGetPlatformIDs))
    return CL_INVALID_VALUE;

  tdispatch = target_dispatch;
  _init_dispatch();

  *layer_dispatch_ret = &dispatch;
  *num_entries_out = sizeof(dispatch)/sizeof(dispatch.clGetPlatformIDs);
  return CL_SUCCESS;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetPlatformIDs_wrap(
    cl_uint num_entries,
    cl_platform_id* platforms,
    cl_uint* num_platforms)
{
  cl_int result;
  cl_uint num_platforms_force;
  if (platforms && !num_platforms)
    num_platforms = &num_platforms_force;

  result = tdispatch->clGetPlatformIDs(
    num_entries,
    platforms,
    num_platforms);
  if (platforms && result == CL_SUCCESS && *num_platforms > 0)
    CHECK_CREATION_LIST(PLATFORM, *num_platforms, platforms);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo_wrap(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(PLATFORM, platform);
  return tdispatch->clGetPlatformInfo(
    platform,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs_wrap(
    cl_platform_id platform,
    cl_device_type device_type,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
  CHECK_EXISTS(PLATFORM, platform);

  cl_int result;
  cl_uint num_devices_force;
  if (devices && !num_devices)
    num_devices = &num_devices_force;

  result = tdispatch->clGetDeviceIDs(
    platform,
    device_type,
    num_entries,
    devices,
    num_devices);
  if (devices && result == CL_SUCCESS && *num_devices > 0)
    CHECK_CREATION_LIST(DEVICE, *num_devices, devices);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo_wrap(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(DEVICE, device);
  return tdispatch->clGetDeviceInfo(
    device,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
}

static CL_API_ENTRY cl_context CL_API_CALL clCreateContext_wrap(
    const cl_context_properties* properties,
    cl_uint num_devices,
    const cl_device_id* devices,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
  CHECK_EXIST_LIST(DEVICE, num_devices, devices);
  cl_context context = tdispatch->clCreateContext(
    properties,
    num_devices,
    devices,
    pfn_notify,
    user_data,
    errcode_ret);

  if (context)
    CHECK_CREATION(CONTEXT, context);
  return context;
}

static CL_API_ENTRY cl_context CL_API_CALL clCreateContextFromType_wrap(
    const cl_context_properties* properties,
    cl_device_type device_type,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
  cl_context context = tdispatch->clCreateContextFromType(
    properties,
    device_type,
    pfn_notify,
    user_data,
    errcode_ret);

  if (context)
    CHECK_CREATION(CONTEXT, context);
  return context;
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainContext_wrap(
    cl_context context)
{
  CHECK_RETAIN(CONTEXT, context);
  return tdispatch->clRetainContext(
    context);
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseContext_wrap(
    cl_context context)
{
  CHECK_RELEASE(CONTEXT, context);
  return tdispatch->clReleaseContext(
    context);
}

static inline object_type get_device_type(cl_device_id dev) {
  cl_device_id parent;
  cl_int res;
  res = tdispatch->clGetDeviceInfo(dev, CL_DEVICE_PARENT_DEVICE, sizeof(cl_device_id), &parent, NULL);
  if (res == CL_SUCCESS && parent)
    return SUB_DEVICE;
  res = tdispatch->clGetDeviceInfo(dev, CL_DEVICE_PARENT_DEVICE_EXT, sizeof(cl_device_id), &parent, NULL);
  if (res == CL_SUCCESS && parent)
    return SUB_DEVICE;
  return DEVICE;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetContextInfo_wrap(
    cl_context context,
    cl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(CONTEXT, context);
  
  cl_int result;
  size_t param_value_size_ret_force;
  if (param_name == CL_CONTEXT_DEVICES && param_value && !param_value_size_ret)
    param_value_size_ret = &param_value_size_ret_force;

  result = tdispatch->clGetContextInfo(
    context,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);

  if (param_name == CL_CONTEXT_DEVICES && param_value && result == CL_SUCCESS) {
    for (size_t i = 0; i < *param_value_size_ret/sizeof(cl_device_id); i++) {
      cl_device_id dev = ((cl_device_id *)param_value)[i];
      if (get_device_type(dev) == SUB_DEVICE)
        CHECK_ADD_OR_EXISTS(SUB_DEVICE, dev);
      else
        CHECK_ADD_OR_EXISTS(DEVICE, dev);
    }
  }
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainCommandQueue_wrap(
    cl_command_queue command_queue)
{
  CHECK_RETAIN(COMMAND_QUEUE, command_queue);
  return tdispatch->clRetainCommandQueue(command_queue);
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseCommandQueue_wrap(
    cl_command_queue command_queue)
{
  CHECK_RELEASE(COMMAND_QUEUE, command_queue);
  return tdispatch->clReleaseCommandQueue(command_queue);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetCommandQueueInfo_wrap(
    cl_command_queue command_queue,
    cl_command_queue_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(COMMAND_QUEUE, command_queue);

  cl_int result;
  result = tdispatch->clGetCommandQueueInfo(
    command_queue,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
  if (param_value && result == CL_SUCCESS) {
    if (param_name == CL_QUEUE_DEVICE) {
      cl_device_id dev = *(cl_device_id *)param_value;
      if (get_device_type(dev) == SUB_DEVICE)
        CHECK_ADD_OR_EXISTS(SUB_DEVICE, dev);
      else
        CHECK_ADD_OR_EXISTS(DEVICE, dev);
    } else if (param_name == CL_QUEUE_CONTEXT) {
      CHECK_ADD_OR_EXISTS(CONTEXT, *(cl_context *)param_value);
    }
  }
  return result;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer_wrap(
    cl_context context,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(CONTEXT, context);
  cl_mem mem = tdispatch->clCreateBuffer(
    context,
    flags,
    size,
    host_ptr,
    errcode_ret);
  if (mem)
    CHECK_CREATION(BUFFER, mem);
  return mem;
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainMemObject_wrap(
    cl_mem memobj)
{
  CHECK_RETAIN(MEM, memobj);
  return tdispatch->clRetainMemObject(memobj);
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject_wrap(
    cl_mem memobj)
{
  CHECK_RELEASE(MEM, memobj);
  return tdispatch->clReleaseMemObject(memobj);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetSupportedImageFormats_wrap(
    cl_context context,
    cl_mem_flags flags,
    cl_mem_object_type image_type,
    cl_uint num_entries,
    cl_image_format* image_formats,
    cl_uint* num_image_formats)
{
  CHECK_EXISTS(CONTEXT, context);
  return tdispatch->clGetSupportedImageFormats(
    context,
    flags,
    image_type,
    num_entries,
    image_formats,
    num_image_formats);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetMemObjectInfo_wrap(
    cl_mem memobj,
    cl_mem_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(MEM, memobj);

  cl_int result;
  result = tdispatch->clGetMemObjectInfo(
    memobj,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
  if (param_value && result == CL_SUCCESS) {
    if (param_name == CL_MEM_CONTEXT) {
      CHECK_ADD_OR_EXISTS(CONTEXT, *(cl_context *)param_value);
    } else if (param_name == CL_MEM_ASSOCIATED_MEMOBJECT && *(cl_mem *)param_value) {
      cl_int res;
      cl_mem_object_type t;
      cl_mem mem = *(cl_mem *)param_value;
      res = tdispatch->clGetMemObjectInfo(
        mem,
        CL_MEM_ASSOCIATED_MEMOBJECT,
        sizeof(cl_mem_object_type),
        &t, NULL);
      if (res == CL_SUCCESS) {
        switch (t) {
        case CL_MEM_OBJECT_BUFFER:
          CHECK_ADD_OR_EXISTS(BUFFER, mem);
          break;
        case CL_MEM_OBJECT_PIPE:
          CHECK_ADD_OR_EXISTS(PIPE, mem);
          break;
        default:
          CHECK_ADD_OR_EXISTS(IMAGE, mem);
        }
      }
    }
  }
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetImageInfo_wrap(
    cl_mem image,
    cl_image_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(IMAGE, image);

  cl_int result;
  result = tdispatch->clGetImageInfo(
    image,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
  if (param_value && result == CL_SUCCESS)
    if (param_name == CL_IMAGE_BUFFER)
      CHECK_ADD_OR_EXISTS(BUFFER, *(cl_mem *)param_value);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainSampler_wrap(
    cl_sampler sampler)
{
  CHECK_RETAIN(SAMPLER, sampler);
  return tdispatch->clRetainSampler(sampler);
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseSampler_wrap(
    cl_sampler sampler)
{
  CHECK_RELEASE(SAMPLER, sampler);
  return tdispatch->clReleaseSampler(sampler);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetSamplerInfo_wrap(
    cl_sampler sampler,
    cl_sampler_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(SAMPLER, sampler);
  cl_int result = tdispatch->clGetSamplerInfo(
    sampler,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
  if (param_value && result == CL_SUCCESS)
    if (param_name == CL_SAMPLER_CONTEXT)
      CHECK_ADD_OR_EXISTS(CONTEXT, *(cl_context *)param_value);
  return result;
}

static CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource_wrap(
    cl_context context,
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(CONTEXT, context);
  cl_program program = tdispatch->clCreateProgramWithSource(
    context,
    count,
    strings,
    lengths,
    errcode_ret);
  if (program)
    CHECK_CREATION(PROGRAM, program);
  return program;
}

static CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithBinary_wrap(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const size_t* lengths,
    const unsigned char** binaries,
    cl_int* binary_status,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(CONTEXT, context);
  CHECK_EXIST_LIST(DEVICE, num_devices, device_list);
  cl_program program = tdispatch->clCreateProgramWithBinary(
    context,
    num_devices,
    device_list,
    lengths,
    binaries,
    binary_status,
    errcode_ret);
  if (program)
    CHECK_CREATION(PROGRAM, program);
  return program;
}

static CL_API_ENTRY cl_int CL_API_CALL clBuildProgram_wrap(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
  CHECK_EXISTS(PROGRAM, program);
  CHECK_EXIST_LIST(DEVICE, num_devices, (void **)device_list);
  return tdispatch->clBuildProgram(
            program,
            num_devices,
            device_list,
            options,
            pfn_notify,
            user_data);
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainProgram_wrap(
    cl_program program)
{
  CHECK_RETAIN(PROGRAM, program);
  return tdispatch->clRetainProgram(program);
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram_wrap(
    cl_program program)
{
  CHECK_RELEASE(PROGRAM, program);
  return tdispatch->clReleaseProgram(program);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetProgramInfo_wrap(
    cl_program program,
    cl_program_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(PROGRAM, program);

  cl_int result;
  result = tdispatch->clGetProgramInfo(
    program,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
  if (param_value && result == CL_SUCCESS) {
    if (param_name == CL_PROGRAM_CONTEXT)
      CHECK_ADD_OR_EXISTS(CONTEXT, *(cl_context *)param_value);
    if (param_name == CL_PROGRAM_DEVICES) {
      for (size_t i = 0; i < *param_value_size_ret/sizeof(cl_device_id); i++) {
        cl_device_id dev = ((cl_device_id *)param_value)[i];
        if (get_device_type(dev) == SUB_DEVICE)
          CHECK_ADD_OR_EXISTS(SUB_DEVICE, dev);
        else
          CHECK_ADD_OR_EXISTS(DEVICE, dev);
      }
    }
  }
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetProgramBuildInfo_wrap(
    cl_program program,
    cl_device_id device,
    cl_program_build_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(PROGRAM, program);
  CHECK_EXISTS(DEVICE, device);
  return tdispatch->clGetProgramBuildInfo(
    program,
    device,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
}

static CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel_wrap(
    cl_program program,
    const char* kernel_name,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(PROGRAM, program);
  cl_kernel kernel = tdispatch->clCreateKernel(
    program,
    kernel_name,
    errcode_ret);
  if (kernel)
    CHECK_CREATION(KERNEL, kernel);
  return kernel;
}

static CL_API_ENTRY cl_int CL_API_CALL clCreateKernelsInProgram_wrap(
    cl_program program,
    cl_uint num_kernels,
    cl_kernel* kernels,
    cl_uint* num_kernels_ret)
{
  CHECK_EXISTS(PROGRAM, program);

  cl_int result;
  cl_uint num_kernels_ret_force;
  if (kernels && !num_kernels_ret)
    num_kernels_ret = &num_kernels_ret_force;
  result = tdispatch->clCreateKernelsInProgram(
    program,
    num_kernels,
    kernels,
    num_kernels_ret);
  if (kernels && result == CL_SUCCESS && *num_kernels_ret > 0)
    CHECK_CREATION_LIST(KERNEL, *num_kernels_ret, kernels);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainKernel_wrap(
    cl_kernel kernel)
{
  CHECK_RETAIN(KERNEL, kernel);
  return tdispatch->clRetainKernel(kernel);
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel_wrap(
    cl_kernel kernel)
{
  CHECK_RELEASE(KERNEL, kernel);
  return tdispatch->clReleaseKernel(kernel);
}

static CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg_wrap(
    cl_kernel kernel,
    cl_uint arg_index,
    size_t arg_size,
    const void* arg_value)
{
  CHECK_EXISTS(KERNEL, kernel);
  return tdispatch->clSetKernelArg(
    kernel,
    arg_index,
    arg_size,
    arg_value);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetKernelInfo_wrap(
    cl_kernel kernel,
    cl_kernel_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(KERNEL, kernel);
  cl_int result = tdispatch->clGetKernelInfo(
    kernel,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
  if (param_value && result == CL_SUCCESS) {
    if (param_name == CL_KERNEL_CONTEXT)
      CHECK_ADD_OR_EXISTS(CONTEXT, *(cl_context *)param_value);
    if (param_name == CL_KERNEL_PROGRAM)
      CHECK_ADD_OR_EXISTS(PROGRAM, *(cl_program *)param_value);
  }
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetKernelWorkGroupInfo_wrap(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_work_group_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(KERNEL, kernel);
  CHECK_EXISTS(DEVICE, device);
  return tdispatch->clGetKernelWorkGroupInfo(
    kernel,
    device,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
}

static CL_API_ENTRY cl_int CL_API_CALL clWaitForEvents_wrap(
    cl_uint num_events,
    const cl_event* event_list)
{
  CHECK_EXIST_LIST(EVENT, num_events, event_list);
  return tdispatch->clWaitForEvents(
    num_events,
    event_list);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetEventInfo_wrap(
    cl_event event,
    cl_event_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(EVENT, event);
  cl_int result = tdispatch->clGetEventInfo(
    event,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
  if (param_value && result == CL_SUCCESS) {
    if (param_name == CL_EVENT_CONTEXT)
      CHECK_ADD_OR_EXISTS(CONTEXT, *(cl_context *)param_value);
    if (param_name == CL_EVENT_COMMAND_QUEUE && *(cl_command_queue *)param_value)
      CHECK_ADD_OR_EXISTS(COMMAND_QUEUE, *(cl_command_queue *)param_value);
  }
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainEvent_wrap(
    cl_event event)
{
  CHECK_RETAIN(EVENT, event);
  return tdispatch->clRetainEvent(event);
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseEvent_wrap(
    cl_event event)
{
  CHECK_RELEASE(EVENT, event);
  return tdispatch->clReleaseEvent(event);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetEventProfilingInfo_wrap(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(EVENT, event);
  return tdispatch->clGetEventProfilingInfo(
    event,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
}

static CL_API_ENTRY cl_int CL_API_CALL clFlush_wrap(
    cl_command_queue command_queue)
{
  CHECK_EXISTS(COMMAND_QUEUE, command_queue);
  return tdispatch->clFlush(command_queue);
}

static CL_API_ENTRY cl_int CL_API_CALL clFinish_wrap(
    cl_command_queue command_queue)
{
  CHECK_EXISTS(COMMAND_QUEUE, command_queue);
  return tdispatch->clFinish(command_queue);
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer_wrap(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    size_t offset,
    size_t size,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(BUFFER, buffer);

  cl_int result = tdispatch->clEnqueueReadBuffer(
    command_queue,
    buffer,
    blocking_read,
    offset,
    size,
    ptr,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteBuffer_wrap(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    size_t offset,
    size_t size,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(BUFFER, buffer);
  cl_int result = tdispatch->clEnqueueWriteBuffer(
    command_queue,
    buffer,
    blocking_write,
    offset,
    size,
    ptr,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL
clCreateSubDevices_wrap(
    cl_device_id in_device,
    const cl_device_partition_property* properties,
    cl_uint num_devices,
    cl_device_id* out_devices,
    cl_uint* num_devices_ret)
{
  CHECK_EXISTS(DEVICE, in_device);

  cl_int result;
  cl_uint num_devices_ret_force;
  if (out_devices && !num_devices_ret)
    num_devices_ret = &num_devices_ret_force;

  result = tdispatch->clCreateSubDevices(
    in_device,
    properties,
    num_devices,
    out_devices,
    num_devices_ret);

  if (out_devices && result == CL_SUCCESS && *num_devices_ret > 0)
    CHECK_CREATION_LIST(SUB_DEVICE, *num_devices_ret, out_devices);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseDevice_wrap(
    cl_device_id device)
{
  CHECK_RELEASE(DEVICE, device);
  return tdispatch->clReleaseDevice(device);
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainDevice_wrap(
    cl_device_id device)
{
  CHECK_RETAIN(DEVICE, device);
  return tdispatch->clRetainDevice(device);
}

static CL_API_ENTRY cl_int CL_API_CALL
clCreateSubDevicesEXT_wrap(
    cl_device_id in_device,
    const cl_device_partition_property_ext* properties,
    cl_uint num_entries,
    cl_device_id* out_devices,
    cl_uint* num_devices)
{
  CHECK_EXISTS(DEVICE, in_device);

  cl_int result;
  cl_uint num_devices_force;
  if (out_devices && !num_devices)
    num_devices = &num_devices_force;

  result = tdispatch->clCreateSubDevicesEXT(
    in_device,
    properties,
    num_entries,
    out_devices,
    num_devices);
  if (out_devices && result == CL_SUCCESS && *num_devices > 0)
    CHECK_CREATION_LIST(SUB_DEVICE, *num_devices, out_devices);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseDeviceEXT_wrap(
    cl_device_id device)
{
  CHECK_RELEASE(DEVICE, device);
  return tdispatch->clReleaseDeviceEXT(device);
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainDeviceEXT_wrap(
    cl_device_id device)
{
  CHECK_RETAIN(DEVICE, device);
  return tdispatch->clRetainDeviceEXT(device);
}

static void _init_dispatch(void) {
  dispatch.clGetPlatformIDs = &clGetPlatformIDs_wrap;
  dispatch.clGetPlatformInfo = &clGetPlatformInfo_wrap;
  dispatch.clGetDeviceIDs = &clGetDeviceIDs_wrap;
  dispatch.clGetDeviceInfo = &clGetDeviceInfo_wrap;
  dispatch.clCreateContext = &clCreateContext_wrap;
  dispatch.clCreateContextFromType = &clCreateContextFromType_wrap;
  dispatch.clReleaseContext = &clReleaseContext_wrap;
  dispatch.clRetainContext = &clRetainContext_wrap;
  dispatch.clGetContextInfo = &clGetContextInfo_wrap;
  dispatch.clRetainCommandQueue = &clRetainCommandQueue_wrap;
  dispatch.clReleaseCommandQueue = &clReleaseCommandQueue_wrap;
  dispatch.clGetCommandQueueInfo = &clGetCommandQueueInfo_wrap;
  dispatch.clCreateBuffer = &clCreateBuffer_wrap;
  dispatch.clRetainMemObject = &clRetainMemObject_wrap;
  dispatch.clReleaseMemObject = &clReleaseMemObject_wrap;
  dispatch.clGetSupportedImageFormats = &clGetSupportedImageFormats_wrap;
  dispatch.clGetMemObjectInfo = &clGetMemObjectInfo_wrap;
  dispatch.clGetImageInfo = &clGetImageInfo_wrap;
  dispatch.clRetainSampler = &clRetainSampler_wrap;
  dispatch.clReleaseSampler = &clReleaseSampler_wrap;
  dispatch.clGetSamplerInfo = &clGetSamplerInfo_wrap;
  dispatch.clCreateProgramWithSource = &clCreateProgramWithSource_wrap;
  dispatch.clCreateProgramWithBinary = &clCreateProgramWithBinary_wrap;
  dispatch.clRetainProgram = &clRetainProgram_wrap;
  dispatch.clReleaseProgram = &clReleaseProgram_wrap;
  dispatch.clBuildProgram = &clBuildProgram_wrap;
  dispatch.clGetProgramInfo = &clGetProgramInfo_wrap;
  dispatch.clGetProgramBuildInfo = &clGetProgramBuildInfo_wrap;
  dispatch.clCreateKernel = &clCreateKernel_wrap;
  dispatch.clCreateKernelsInProgram = &clCreateKernelsInProgram_wrap;
  dispatch.clRetainKernel = &clRetainKernel_wrap;
  dispatch.clReleaseKernel = &clReleaseKernel_wrap;
  dispatch.clSetKernelArg = &clSetKernelArg_wrap;
  dispatch.clGetKernelInfo = &clGetKernelInfo_wrap;
  dispatch.clGetKernelWorkGroupInfo = &clGetKernelWorkGroupInfo_wrap;
  dispatch.clWaitForEvents = &clWaitForEvents_wrap;
  dispatch.clGetEventInfo = &clGetEventInfo_wrap;
  dispatch.clRetainEvent = &clRetainEvent_wrap;
  dispatch.clReleaseEvent = &clReleaseEvent_wrap;
  dispatch.clGetEventProfilingInfo = &clGetEventProfilingInfo_wrap;
  dispatch.clFlush = &clFlush_wrap;
  dispatch.clFinish = &clFinish_wrap;
  dispatch.clEnqueueReadBuffer = &clEnqueueReadBuffer_wrap;
  dispatch.clEnqueueWriteBuffer = &clEnqueueWriteBuffer_wrap;
  dispatch.clCreateSubDevicesEXT = &clCreateSubDevicesEXT_wrap;
  dispatch.clReleaseDeviceEXT = &clReleaseDeviceEXT_wrap;
  dispatch.clRetainDeviceEXT = &clRetainDeviceEXT_wrap;
  dispatch.clCreateSubDevices = &clCreateSubDevices_wrap;
  dispatch.clReleaseDevice = &clReleaseDevice_wrap;
  dispatch.clRetainDevice = &clRetainDevice_wrap;
}
