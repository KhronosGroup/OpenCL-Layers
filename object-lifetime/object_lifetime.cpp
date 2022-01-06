#include <stdlib.h>
#include <string.h>
#include <CL/cl_layer.h>
#include <mutex>
#include <tuple>
#include <map>
#include <iostream>
#include <list>
#include <string_view>

typedef enum object_type_e {
  OCL_PLATFORM,
  OCL_DEVICE,
  OCL_SUB_DEVICE,
  OCL_CONTEXT,
  OCL_COMMAND_QUEUE,
  OCL_MEM,
  OCL_BUFFER,
  OCL_IMAGE,
  OCL_PIPE,
  OCL_PROGRAM,
  OCL_KERNEL,
  OCL_EVENT,
  OCL_SAMPLER,
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

typedef std::tuple<object_type, cl_long> type_count;

std::map<void*, type_count> objects;
std::map<void*, std::list<type_count>> deleted_objects;
std::mutex objects_mutex;

static void error_already_exist(const std::string_view &func, void *handle, object_type t, cl_long ref_count) {
  std::cerr << "In " << func << " " <<
               object_type_names[t] <<
               ": " << handle <<
               " already exist with refcount: " << ref_count << "\n";
  std::cerr.flush();
}

static void error_ref_count(const std::string_view &func, void *handle, object_type t, cl_long ref_count) {
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
static inline void check_exists_no_lock(const std::string_view &func, void *handle) {
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, T);
  } else if (std::get<0>(it->second) != T) {
    error_invalid_type(func, handle, std::get<0>(it->second), T);
  } else if (std::get<1>(it->second) <= 0) {
    error_ref_count(func, handle, T, std::get<1>(it->second));
  }
}

template<object_type T>
static void check_exists(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  check_exists_no_lock<T>(func, handle);
  objects_mutex.unlock();
}

template<>
void check_exists_no_lock<OCL_PLATFORM>(const std::string_view &func, void *handle) {
  if(!handle)
    return;
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, OCL_PLATFORM);
  } else if (std::get<0>(it->second) != OCL_PLATFORM) {
    error_invalid_type(func, handle, std::get<0>(it->second), OCL_PLATFORM);
  }
}

template<>
void check_exists_no_lock<OCL_DEVICE>(const std::string_view &func, void *handle) {
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, OCL_DEVICE);
  } else if (std::get<0>(it->second) != OCL_DEVICE && std::get<0>(it->second) != OCL_SUB_DEVICE) {
    error_invalid_type(func, handle, std::get<0>(it->second), OCL_DEVICE);
  } else if (std::get<0>(it->second) == OCL_SUB_DEVICE && std::get<1>(it->second) <= 0) {
    error_ref_count(func, handle, OCL_SUB_DEVICE, std::get<1>(it->second));
  }
}

template<>
void check_exists_no_lock<OCL_MEM>(const std::string_view &func, void *handle) {
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, OCL_MEM);
  } else {
    object_type t = std::get<0>(it->second);
    switch (t) {
    case OCL_BUFFER:
    case OCL_IMAGE:
    case OCL_PIPE:
      if (std::get<1>(it->second) <= 0) {
        error_ref_count(func, handle, t, std::get<1>(it->second));
      }
      break;
    default:
      error_invalid_type(func, handle, std::get<0>(it->second), t);
    }
  }
}

#define CHECK_EXISTS(type, handle) check_exists<type>(RTRIM_FUNC, handle)

template<object_type T>
static void check_exist_list(const std::string_view &func, cl_uint num_handles, void **handles) {
  if (!handles)
    return;
  objects_mutex.lock();
  for (cl_uint i = 0; i < num_handles; i++) {
    check_exists_no_lock<T>(func, handles[i]);
  }
  objects_mutex.unlock();
}

#define CHECK_EXIST_LIST(type, num_handles, handles) check_exist_list<type>(RTRIM_FUNC, num_handles, (void **)handles)

template<object_type T>
static inline void check_creation_no_lock(const std::string_view &func, void *handle) {
  auto it = objects.find(handle);
  if (it != objects.end()) {
    if (std::get<1>(it->second) > 0) {
      error_already_exist(func, handle, std::get<0>(it->second), std::get<1>(it->second));
      deleted_objects[handle].push_back(it->second);
    }
  }
  objects[handle] = type_count(T, 1);
}

template<>
void check_creation_no_lock<OCL_DEVICE>(const std::string_view &func, void *handle) {
  auto it = objects.find(handle);
  if (it != objects.end() && std::get<0>(it->second) != OCL_DEVICE) {
    error_already_exist(func, handle, std::get<0>(it->second), std::get<1>(it->second));
    deleted_objects[handle].push_back(it->second);
  }
  objects[handle] = type_count(OCL_DEVICE, 0);
}

template<>
void check_creation_no_lock<OCL_PLATFORM>(const std::string_view &func, void *handle) {
  auto it = objects.find(handle);
  if (it != objects.end() && std::get<0>(it->second) != OCL_PLATFORM) {
    error_already_exist(func, handle, std::get<0>(it->second), std::get<1>(it->second));
    deleted_objects[handle].push_back(it->second);
  }
  objects[handle] = type_count(OCL_PLATFORM, 0);
}

template<object_type T>
static void check_creation(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  check_creation_no_lock<T>(func, handle);
  objects_mutex.unlock();
}

#define CHECK_CREATION(type, handle) check_creation<type>(RTRIM_FUNC, handle)

template<object_type T>
static void check_creation_list(const std::string_view &func, cl_uint num_handles, void **handles) {
  objects_mutex.lock();
  for (cl_uint i = 0; i < num_handles; i++) {
    check_creation_no_lock<T>(func, handles[i]);
  }
  objects_mutex.unlock();
}

#define CHECK_CREATION_LIST(type, num_handles, handles) check_creation_list<type>(RTRIM_FUNC, num_handles, (void **)handles)

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
void check_release<OCL_DEVICE>(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, OCL_DEVICE);
  } else {
    object_type t = std::get<0>(it->second);
    switch (t) {
    case OCL_DEVICE:
      break;
    case OCL_SUB_DEVICE:
      std::get<1>(it->second) -= 1;
      if (std::get<1>(it->second) < 0) {
        error_invalid_release(func, handle, OCL_SUB_DEVICE);
      }
      break;
    default:
      error_invalid_type(func, handle, t, OCL_DEVICE);
    }
  }
  objects_mutex.unlock();
}

template<>
void check_release<OCL_MEM>(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, OCL_MEM);
  } else {
    object_type t = std::get<0>(it->second);
    switch (t) {
    case OCL_BUFFER:
    case OCL_IMAGE:
    case OCL_PIPE:
      std::get<1>(it->second) -= 1;
      if (std::get<1>(it->second) == 0) {
        deleted_objects[handle].push_back(it->second);
        objects.erase(it);
      } else if (std::get<1>(it->second) < 0) {
        error_invalid_release(func, handle, t);
      }
      break;
    default:
      error_invalid_type(func, handle, t, OCL_MEM);
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
void check_retain<OCL_DEVICE>(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, OCL_DEVICE);
  } else {
    object_type t = std::get<0>(it->second);
    switch (t) {
    case OCL_DEVICE:
      break;
    case OCL_SUB_DEVICE:
      std::get<1>(it->second) += 1;
      break;
    default:
      error_invalid_type(func, handle, t, OCL_DEVICE);
    }
  }
  objects_mutex.unlock();
}

template<>
void check_retain<OCL_MEM>(const std::string_view &func, void *handle) {
  objects_mutex.lock();
  auto it = objects.find(handle);
  if (it == objects.end()) {
    error_does_not_exist(func, handle, OCL_MEM);
  } else {
    object_type t = std::get<0>(it->second);
    switch (t) {
    case OCL_BUFFER:
    case OCL_IMAGE:
    case OCL_PIPE:
      std::get<1>(it->second) += 1;
      break;
    default:
      error_invalid_type(func, handle, t, OCL_MEM);
    }
  }
  objects_mutex.unlock();
}

static void report() {
  objects_mutex.lock();
  std::cerr << "OpenCL objects leaks:\n";
  for (auto it = objects.begin(); it != objects.end(); ++it) {
    if (std::get<1>(it->second) > 0) {
      object_type t = std::get<0>(it->second);
      std::cerr << object_type_names[t] << " (" <<
                it->first << ") reference count: " <<
                std::get<1>(it->second) << "\n";
    }
  }
  objects.clear();
  deleted_objects.clear();
  objects_mutex.unlock();
}

#define CHECK_RETAIN(type, handle) check_retain<type>(RTRIM_FUNC, handle)

static struct _cl_icd_dispatch dispatch = {};

static const struct _cl_icd_dispatch *tdispatch;

  /* Layer API entry points */
CL_API_ENTRY cl_int CL_API_CALL
clGetLayerInfo(
    cl_layer_info  param_name,
    size_t         param_value_size,
    void          *param_value,
    size_t        *param_value_size_ret) {
  switch (param_name) {
  case CL_LAYER_API_VERSION:
    if (param_value) {
      if (param_value_size < sizeof(cl_layer_api_version))
        return CL_INVALID_VALUE;
      *((cl_layer_api_version *)param_value) = CL_LAYER_API_VERSION_100;
    }
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
  atexit(report);
  return CL_SUCCESS;
}

  /* helper functions */
static cl_platform_id context_properties_get_platform(const cl_context_properties *properties) {
  if (properties == NULL)
    return NULL;
  cl_platform_id platform = NULL;
  for (const cl_context_properties *property = properties; property && property[0]; property += 2)
    if ((cl_context_properties)CL_CONTEXT_PLATFORM == property[0])
      platform = (cl_platform_id)property[1];
  return platform;
}

static inline object_type get_device_type(cl_device_id dev) {
  cl_device_id parent;
  cl_int res;
  res = tdispatch->clGetDeviceInfo(dev, CL_DEVICE_PARENT_DEVICE, sizeof(cl_device_id), &parent, NULL);
  if (res == CL_SUCCESS && parent)
    return OCL_SUB_DEVICE;
  res = tdispatch->clGetDeviceInfo(dev, CL_DEVICE_PARENT_DEVICE_EXT, sizeof(cl_device_id), &parent, NULL);
  if (res == CL_SUCCESS && parent)
    return OCL_SUB_DEVICE;
  return OCL_DEVICE;
}

  /* OpenCL 1.0 */
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
    CHECK_CREATION_LIST(OCL_PLATFORM, *num_platforms, platforms);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo_wrap(
    cl_platform_id platform,
    cl_platform_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(OCL_PLATFORM, platform);
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
  CHECK_EXISTS(OCL_PLATFORM, platform);

  cl_uint num_devices_force;
  if (devices && !num_devices)
    num_devices = &num_devices_force;

  cl_int result = tdispatch->clGetDeviceIDs(
    platform,
    device_type,
    num_entries,
    devices,
    num_devices);
  if (devices && result == CL_SUCCESS && *num_devices > 0)
    CHECK_CREATION_LIST(OCL_DEVICE, *num_devices, devices);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo_wrap(
    cl_device_id device,
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(OCL_DEVICE, device);
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
  cl_platform_id platform = context_properties_get_platform(properties);
  CHECK_EXISTS(OCL_PLATFORM, platform);
  CHECK_EXIST_LIST(OCL_DEVICE, num_devices, devices);
  cl_context context = tdispatch->clCreateContext(
    properties,
    num_devices,
    devices,
    pfn_notify,
    user_data,
    errcode_ret);

  if (context)
    CHECK_CREATION(OCL_CONTEXT, context);
  return context;
}

static CL_API_ENTRY cl_context CL_API_CALL clCreateContextFromType_wrap(
    const cl_context_properties* properties,
    cl_device_type device_type,
    void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
  cl_platform_id platform = context_properties_get_platform(properties);
  CHECK_EXISTS(OCL_PLATFORM, platform);
  cl_context context = tdispatch->clCreateContextFromType(
    properties,
    device_type,
    pfn_notify,
    user_data,
    errcode_ret);

  if (context)
    CHECK_CREATION(OCL_CONTEXT, context);
  return context;
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainContext_wrap(
    cl_context context)
{
  CHECK_RETAIN(OCL_CONTEXT, context);
  return tdispatch->clRetainContext(
    context);
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseContext_wrap(
    cl_context context)
{
  CHECK_RELEASE(OCL_CONTEXT, context);
  return tdispatch->clReleaseContext(
    context);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetContextInfo_wrap(
    cl_context context,
    cl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  
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
      if (get_device_type(dev) == OCL_SUB_DEVICE)
        CHECK_ADD_OR_EXISTS(OCL_SUB_DEVICE, dev);
      else
        CHECK_ADD_OR_EXISTS(OCL_DEVICE, dev);
    }
  }
  return result;
}

static CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueue_wrap(
    cl_context context,
    cl_device_id device,
    cl_command_queue_properties properties,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_command_queue command_queue = tdispatch->clCreateCommandQueue(
    context,
    device,
    properties,
    errcode_ret);
  if (command_queue)
    CHECK_CREATION(OCL_COMMAND_QUEUE, command_queue);
  return command_queue;
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainCommandQueue_wrap(
    cl_command_queue command_queue)
{
  CHECK_RETAIN(OCL_COMMAND_QUEUE, command_queue);
  return tdispatch->clRetainCommandQueue(command_queue);
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseCommandQueue_wrap(
    cl_command_queue command_queue)
{
  CHECK_RELEASE(OCL_COMMAND_QUEUE, command_queue);
  return tdispatch->clReleaseCommandQueue(command_queue);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetCommandQueueInfo_wrap(
    cl_command_queue command_queue,
    cl_command_queue_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);

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
      if (get_device_type(dev) == OCL_SUB_DEVICE)
        CHECK_ADD_OR_EXISTS(OCL_SUB_DEVICE, dev);
      else
        CHECK_ADD_OR_EXISTS(OCL_DEVICE, dev);
    } else if (param_name == CL_QUEUE_CONTEXT) {
      CHECK_ADD_OR_EXISTS(OCL_CONTEXT, *(cl_context *)param_value);
    }
  }
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clSetCommandQueueProperty_wrap(
    cl_command_queue command_queue,
    cl_command_queue_properties properties,
    cl_bool enable,
    cl_command_queue_properties* old_properties)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  return tdispatch->clSetCommandQueueProperty(
    command_queue,
    properties,
    enable,
    old_properties);
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer_wrap(
    cl_context context,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem mem = tdispatch->clCreateBuffer(
    context,
    flags,
    size,
    host_ptr,
    errcode_ret);
  if (mem)
    CHECK_CREATION(OCL_BUFFER, mem);
  return mem;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateImage2D_wrap(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    size_t image_width,
    size_t image_height,
    size_t image_row_pitch,
    void* host_ptr,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem image = tdispatch->clCreateImage2D(
    context,
    flags,
    image_format,
    image_width,
    image_height,
    image_row_pitch,
    host_ptr,
    errcode_ret);
  if (image)
    CHECK_CREATION(OCL_IMAGE, image);
  return image;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateImage3D_wrap(
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
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem image = tdispatch->clCreateImage3D(
    context,
    flags,
    image_format,
    image_width,
    image_height,
    image_depth,
    image_row_pitch,
    image_slice_pitch,
    host_ptr,
    errcode_ret);
  if (image)
    CHECK_CREATION(OCL_IMAGE, image);
  return image;
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainMemObject_wrap(
    cl_mem memobj)
{
  CHECK_RETAIN(OCL_MEM, memobj);
  return tdispatch->clRetainMemObject(memobj);
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject_wrap(
    cl_mem memobj)
{
  CHECK_RELEASE(OCL_MEM, memobj);
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
  CHECK_EXISTS(OCL_CONTEXT, context);
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
  CHECK_EXISTS(OCL_MEM, memobj);

  cl_int result;
  result = tdispatch->clGetMemObjectInfo(
    memobj,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
  if (param_value && result == CL_SUCCESS) {
    if (param_name == CL_MEM_CONTEXT) {
      CHECK_ADD_OR_EXISTS(OCL_CONTEXT, *(cl_context *)param_value);
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
          CHECK_ADD_OR_EXISTS(OCL_BUFFER, mem);
          break;
        case CL_MEM_OBJECT_PIPE:
          CHECK_ADD_OR_EXISTS(OCL_PIPE, mem);
          break;
        default:
          CHECK_ADD_OR_EXISTS(OCL_IMAGE, mem);
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
  CHECK_EXISTS(OCL_IMAGE, image);

  cl_int result;
  result = tdispatch->clGetImageInfo(
    image,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
  if (param_value && result == CL_SUCCESS)
    if (param_name == CL_IMAGE_BUFFER)
      CHECK_ADD_OR_EXISTS(OCL_BUFFER, *(cl_mem *)param_value);
  return result;
}

static CL_API_ENTRY cl_sampler CL_API_CALL clCreateSampler_wrap(
    cl_context context,
    cl_bool normalized_coords,
    cl_addressing_mode addressing_mode,
    cl_filter_mode filter_mode,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_sampler sampler = tdispatch->clCreateSampler(
    context,
    normalized_coords,
    addressing_mode,
    filter_mode,
    errcode_ret);
  if (sampler)
    CHECK_CREATION(OCL_SAMPLER, sampler);
  return sampler;
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainSampler_wrap(
    cl_sampler sampler)
{
  CHECK_RETAIN(OCL_SAMPLER, sampler);
  return tdispatch->clRetainSampler(sampler);
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseSampler_wrap(
    cl_sampler sampler)
{
  CHECK_RELEASE(OCL_SAMPLER, sampler);
  return tdispatch->clReleaseSampler(sampler);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetSamplerInfo_wrap(
    cl_sampler sampler,
    cl_sampler_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(OCL_SAMPLER, sampler);
  cl_int result = tdispatch->clGetSamplerInfo(
    sampler,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
  if (param_value && result == CL_SUCCESS)
    if (param_name == CL_SAMPLER_CONTEXT)
      CHECK_ADD_OR_EXISTS(OCL_CONTEXT, *(cl_context *)param_value);
  return result;
}

static CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource_wrap(
    cl_context context,
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_program program = tdispatch->clCreateProgramWithSource(
    context,
    count,
    strings,
    lengths,
    errcode_ret);
  if (program)
    CHECK_CREATION(OCL_PROGRAM, program);
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
  CHECK_EXISTS(OCL_CONTEXT, context);
  CHECK_EXIST_LIST(OCL_DEVICE, num_devices, device_list);
  cl_program program = tdispatch->clCreateProgramWithBinary(
    context,
    num_devices,
    device_list,
    lengths,
    binaries,
    binary_status,
    errcode_ret);
  if (program)
    CHECK_CREATION(OCL_PROGRAM, program);
  return program;
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainProgram_wrap(
    cl_program program)
{
  CHECK_RETAIN(OCL_PROGRAM, program);
  return tdispatch->clRetainProgram(program);
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram_wrap(
    cl_program program)
{
  CHECK_RELEASE(OCL_PROGRAM, program);
  return tdispatch->clReleaseProgram(program);
}

static CL_API_ENTRY cl_int CL_API_CALL clBuildProgram_wrap(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
  CHECK_EXISTS(OCL_PROGRAM, program);
  CHECK_EXIST_LIST(OCL_DEVICE, num_devices, (void **)device_list);
  return tdispatch->clBuildProgram(
            program,
            num_devices,
            device_list,
            options,
            pfn_notify,
            user_data);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetProgramInfo_wrap(
    cl_program program,
    cl_program_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(OCL_PROGRAM, program);

  size_t param_value_size_ret_force;
  if (param_name == CL_PROGRAM_DEVICES && param_value && !param_value_size_ret)
    param_value_size_ret = &param_value_size_ret_force;

  cl_int result = tdispatch->clGetProgramInfo(
    program,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
  if (param_value && result == CL_SUCCESS) {
    if (param_name == CL_PROGRAM_CONTEXT)
      CHECK_ADD_OR_EXISTS(OCL_CONTEXT, *(cl_context *)param_value);
    if (param_name == CL_PROGRAM_DEVICES) {
      for (size_t i = 0; i < *param_value_size_ret/sizeof(cl_device_id); i++) {
        cl_device_id dev = ((cl_device_id *)param_value)[i];
        if (get_device_type(dev) == OCL_SUB_DEVICE)
          CHECK_ADD_OR_EXISTS(OCL_SUB_DEVICE, dev);
        else
          CHECK_ADD_OR_EXISTS(OCL_DEVICE, dev);
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
  CHECK_EXISTS(OCL_PROGRAM, program);
  CHECK_EXISTS(OCL_DEVICE, device);
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
  CHECK_EXISTS(OCL_PROGRAM, program);
  cl_kernel kernel = tdispatch->clCreateKernel(
    program,
    kernel_name,
    errcode_ret);
  if (kernel)
    CHECK_CREATION(OCL_KERNEL, kernel);
  return kernel;
}

static CL_API_ENTRY cl_int CL_API_CALL clCreateKernelsInProgram_wrap(
    cl_program program,
    cl_uint num_kernels,
    cl_kernel* kernels,
    cl_uint* num_kernels_ret)
{
  CHECK_EXISTS(OCL_PROGRAM, program);

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
    CHECK_CREATION_LIST(OCL_KERNEL, *num_kernels_ret, kernels);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainKernel_wrap(
    cl_kernel kernel)
{
  CHECK_RETAIN(OCL_KERNEL, kernel);
  return tdispatch->clRetainKernel(kernel);
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel_wrap(
    cl_kernel kernel)
{
  CHECK_RELEASE(OCL_KERNEL, kernel);
  return tdispatch->clReleaseKernel(kernel);
}

static CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg_wrap(
    cl_kernel kernel,
    cl_uint arg_index,
    size_t arg_size,
    const void* arg_value)
{
  CHECK_EXISTS(OCL_KERNEL, kernel);
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
  CHECK_EXISTS(OCL_KERNEL, kernel);
  cl_int result = tdispatch->clGetKernelInfo(
    kernel,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
  if (param_value && result == CL_SUCCESS) {
    if (param_name == CL_KERNEL_CONTEXT)
      CHECK_ADD_OR_EXISTS(OCL_CONTEXT, *(cl_context *)param_value);
    if (param_name == CL_KERNEL_PROGRAM)
      CHECK_ADD_OR_EXISTS(OCL_PROGRAM, *(cl_program *)param_value);
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
  CHECK_EXISTS(OCL_KERNEL, kernel);
  if (device)
    CHECK_EXISTS(OCL_DEVICE, device);
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
  CHECK_EXIST_LIST(OCL_EVENT, num_events, event_list);
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
  CHECK_EXISTS(OCL_EVENT, event);
  cl_int result = tdispatch->clGetEventInfo(
    event,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
  if (param_value && result == CL_SUCCESS) {
    if (param_name == CL_EVENT_CONTEXT)
      CHECK_ADD_OR_EXISTS(OCL_CONTEXT, *(cl_context *)param_value);
    if (param_name == CL_EVENT_COMMAND_QUEUE && *(cl_command_queue *)param_value)
      CHECK_ADD_OR_EXISTS(OCL_COMMAND_QUEUE, *(cl_command_queue *)param_value);
  }
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainEvent_wrap(
    cl_event event)
{
  CHECK_RETAIN(OCL_EVENT, event);
  return tdispatch->clRetainEvent(event);
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseEvent_wrap(
    cl_event event)
{
  CHECK_RELEASE(OCL_EVENT, event);
  return tdispatch->clReleaseEvent(event);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetEventProfilingInfo_wrap(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(OCL_EVENT, event);
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
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  return tdispatch->clFlush(command_queue);
}

static CL_API_ENTRY cl_int CL_API_CALL clFinish_wrap(
    cl_command_queue command_queue)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
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
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_BUFFER, buffer);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);

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
    CHECK_CREATION(OCL_EVENT, *event);
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
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_BUFFER, buffer);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
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
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBuffer_wrap(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    size_t src_offset,
    size_t dst_offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_BUFFER, src_buffer);
  CHECK_EXISTS(OCL_BUFFER, dst_buffer);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueCopyBuffer(
    command_queue,
    src_buffer,
    dst_buffer,
    src_offset,
    dst_offset,
    size,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadImage_wrap(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_read,
    const size_t* origin,
    const size_t* region,
    size_t row_pitch,
    size_t slice_pitch,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_IMAGE, image);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueReadImage(
    command_queue,
    image,
    blocking_read,
    origin,
    region,
    row_pitch,
    slice_pitch,
    ptr,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteImage_wrap(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_write,
    const size_t* origin,
    const size_t* region,
    size_t input_row_pitch,
    size_t input_slice_pitch,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_IMAGE, image);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueWriteImage(
    command_queue,
    image,
    blocking_write,
    origin,
    region,
    input_row_pitch,
    input_slice_pitch,
    ptr,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyImage_wrap(
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_image,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_IMAGE, src_image);
  CHECK_EXISTS(OCL_IMAGE, dst_image);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueCopyImage(
    command_queue,
    src_image,
    dst_image,
    src_origin,
    dst_origin,
    region,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyImageToBuffer_wrap(
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* region,
    size_t dst_offset,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_IMAGE, src_image);
  CHECK_EXISTS(OCL_BUFFER, dst_buffer);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueCopyImageToBuffer(
    command_queue,
    src_image,
    dst_buffer,
    src_origin,
    region,
    dst_offset,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBufferToImage_wrap(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_image,
    size_t src_offset,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_BUFFER, src_buffer);
  CHECK_EXISTS(OCL_IMAGE, dst_image);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueCopyBufferToImage(
    command_queue,
    src_buffer,
    dst_image,
    src_offset,
    dst_origin,
    region,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY void* CL_API_CALL clEnqueueMapBuffer_wrap(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_map,
    cl_map_flags map_flags,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_BUFFER, buffer);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  void *result = tdispatch->clEnqueueMapBuffer(
    command_queue,
    buffer,
    blocking_map,
    map_flags,
    offset,
    size,
    num_events_in_wait_list,
    event_wait_list,
    event,
    errcode_ret);
  if (result && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY void* CL_API_CALL clEnqueueMapImage_wrap(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_map,
    cl_map_flags map_flags,
    const size_t* origin,
    const size_t* region,
    size_t* image_row_pitch,
    size_t* image_slice_pitch,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_IMAGE, image);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  void *result = tdispatch->clEnqueueMapImage(
            command_queue,
            image,
            blocking_map,
            map_flags,
            origin,
            region,
            image_row_pitch,
            image_slice_pitch,
            num_events_in_wait_list,
            event_wait_list,
            event,
            errcode_ret);
  if (result && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueUnmapMemObject_wrap(
    cl_command_queue command_queue,
    cl_mem memobj,
    void* mapped_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_MEM, memobj);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueUnmapMemObject(
    command_queue,
    memobj,
    mapped_ptr,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel_wrap(
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
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_KERNEL, kernel);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueNDRangeKernel(
    command_queue,
    kernel,
    work_dim,
    global_work_offset,
    global_work_size,
    local_work_size,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueTask_wrap(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_KERNEL, kernel);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueTask(
    command_queue,
    kernel,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueNativeKernel_wrap(
    cl_command_queue command_queue,
    void (CL_CALLBACK* user_func)(void*),
    void* args,
    size_t cb_args,
    cl_uint num_mem_objects,
    const cl_mem* mem_list,
    const void** args_mem_loc,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  CHECK_EXIST_LIST(OCL_MEM, num_mem_objects, mem_list);
  cl_int result = tdispatch->clEnqueueNativeKernel(
    command_queue,
    user_func,
    args,
    cb_args,
    num_mem_objects,
    mem_list,
    args_mem_loc,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueMarker_wrap(
    cl_command_queue command_queue,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  cl_int result = tdispatch->clEnqueueMarker(
    command_queue,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueWaitForEvents_wrap(
    cl_command_queue command_queue,
    cl_uint num_events,
    const cl_event* event_list)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_EVENT, num_events, event_list);
  return tdispatch->clEnqueueWaitForEvents(
    command_queue,
    num_events,
    event_list);
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueBarrier_wrap(
    cl_command_queue command_queue)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  return tdispatch->clEnqueueBarrier(
    command_queue);
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLBuffer_wrap(
    cl_context context,
    cl_mem_flags flags,
    cl_GLuint bufobj,
    int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem buffer = tdispatch->clCreateFromGLBuffer(
    context,
    flags,
    bufobj,
    errcode_ret);
  if (buffer)
    CHECK_CREATION(OCL_BUFFER, buffer);
  return buffer;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLTexture2D_wrap(
    cl_context context,
    cl_mem_flags flags,
    cl_GLenum target,
    cl_GLint miplevel,
    cl_GLuint texture,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem image = tdispatch->clCreateFromGLTexture2D(
    context,
    flags,
    target,
    miplevel,
    texture,
    errcode_ret);
  if (image)
    CHECK_CREATION(OCL_IMAGE, image);
  return image;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLTexture3D_wrap(
    cl_context context,
    cl_mem_flags flags,
    cl_GLenum target,
    cl_GLint miplevel,
    cl_GLuint texture,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem image = tdispatch->clCreateFromGLTexture3D(
    context,
    flags,
    target,
    miplevel,
    texture,
    errcode_ret);
  if (image)
    CHECK_CREATION(OCL_IMAGE, image);
  return image;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLRenderbuffer_wrap(
    cl_context context,
    cl_mem_flags flags,
    cl_GLuint renderbuffer,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem image = tdispatch->clCreateFromGLRenderbuffer(
    context,
    flags,
    renderbuffer,
    errcode_ret);
  if (image)
    CHECK_CREATION(OCL_IMAGE, image);
  return image;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetGLObjectInfo_wrap(
    cl_mem memobj,
    cl_gl_object_type* gl_object_type,
    cl_GLuint* gl_object_name)
{
  CHECK_EXISTS(OCL_MEM, memobj);
  return tdispatch->clGetGLObjectInfo(
    memobj,
    gl_object_type,
    gl_object_name);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetGLTextureInfo_wrap(
    cl_mem memobj,
    cl_gl_texture_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(OCL_IMAGE, memobj);
  return tdispatch->clGetGLTextureInfo(
    memobj,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueAcquireGLObjects_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_MEM, num_objects, mem_objects);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueAcquireGLObjects(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseGLObjects_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_MEM, num_objects, mem_objects);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueReleaseGLObjects(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetGLContextInfoKHR_wrap(
    const cl_context_properties* properties,
    cl_gl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  cl_platform_id platform = context_properties_get_platform(properties);
  CHECK_EXISTS(OCL_PLATFORM, platform);

  size_t param_value_size_ret_force;
  if (param_value && !param_value_size_ret)
    param_value_size_ret = &param_value_size_ret_force;

  cl_int result = tdispatch->clGetGLContextInfoKHR(
    properties,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
  if (result == CL_SUCCESS && param_value) {
    if (CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR && param_value_size_ret != 0)
      CHECK_CREATION(OCL_DEVICE, *(cl_device_id *)param_value);
    if (CL_DEVICES_FOR_GL_CONTEXT_KHR)
      CHECK_CREATION_LIST(OCL_DEVICE, (cl_uint)(*param_value_size_ret/sizeof(cl_device_id)), param_value);
  }
  return result;
}

  /* cl_khr_d3d10_sharing */
#if defined(_WIN32)
static CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDsFromD3D10KHR_wrap(
    cl_platform_id platform,
    cl_d3d10_device_source_khr d3d_device_source,
    void* d3d_object,
    cl_d3d10_device_set_khr d3d_device_set,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
  CHECK_EXISTS(OCL_PLATFORM, platform);

  cl_uint num_devices_force;
  if (devices && !num_devices)
    num_devices = &num_devices_force;

  cl_int result = tdispatch->clGetDeviceIDsFromD3D10KHR(
    platform,
    d3d_device_source,
    d3d_object,
    d3d_device_set,
    num_entries,
    devices,
    num_devices);
  if (devices && result == CL_SUCCESS && *num_devices > 0)
    CHECK_CREATION_LIST(OCL_DEVICE, *num_devices, devices);
  return result;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromD3D10BufferKHR_wrap(
    cl_context context,
    cl_mem_flags flags,
    ID3D10Buffer* resource,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem buffer = tdispatch->clCreateFromD3D10BufferKHR(
    context,
    flags,
    resource,
    errcode_ret);
  if (buffer)
    CHECK_CREATION(OCL_BUFFER, buffer);
  return buffer;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromD3D10Texture2DKHR_wrap(
    cl_context context,
    cl_mem_flags flags,
    ID3D10Texture2D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem image = tdispatch->clCreateFromD3D10Texture2DKHR(
    context,
    flags,
    resource,
    subresource,
    errcode_ret);
  if (image)
    CHECK_CREATION(OCL_IMAGE, image);
  return image;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromD3D10Texture3DKHR_wrap(
    cl_context context,
    cl_mem_flags flags,
    ID3D10Texture3D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem image = tdispatch->clCreateFromD3D10Texture3DKHR(
    context,
    flags,
    resource,
    subresource,
    errcode_ret);
  if (image)
    CHECK_CREATION(OCL_IMAGE, image);
  return image;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueAcquireD3D10ObjectsKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_MEM, num_objects, mem_objects);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueAcquireD3D10ObjectsKHR(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseD3D10ObjectsKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_MEM, num_objects, mem_objects);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueReleaseD3D10ObjectsKHR(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}
#endif

  /* OpenCL 1.1 */
static CL_API_ENTRY cl_int CL_API_CALL clSetEventCallback_wrap(
    cl_event event,
    cl_int command_exec_callback_type,
    void (CL_CALLBACK* pfn_notify)(cl_event event, cl_int event_command_status, void *user_data),
    void* user_data)
{
  CHECK_EXISTS(OCL_EVENT, event);
  return tdispatch->clSetEventCallback(
    event,
    command_exec_callback_type,
    pfn_notify,
    user_data);
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateSubBuffer_wrap(
    cl_mem buffer,
    cl_mem_flags flags,
    cl_buffer_create_type buffer_create_type,
    const void* buffer_create_info,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_BUFFER, buffer);
  cl_mem sub_buffer = tdispatch->clCreateSubBuffer(
    buffer,
    flags,
    buffer_create_type,
    buffer_create_info,
    errcode_ret);
  if (sub_buffer)
    CHECK_CREATION(OCL_BUFFER, sub_buffer);
  return sub_buffer;
}

static CL_API_ENTRY cl_int CL_API_CALL clSetMemObjectDestructorCallback_wrap(
    cl_mem memobj,
    void (CL_CALLBACK* pfn_notify)(cl_mem memobj, void* user_data),
    void* user_data)
{
  CHECK_EXISTS(OCL_MEM, memobj);
  return tdispatch->clSetMemObjectDestructorCallback(
    memobj,
    pfn_notify,
    user_data);
}

static CL_API_ENTRY cl_event CL_API_CALL clCreateUserEvent_wrap(
    cl_context context,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_event event = tdispatch->clCreateUserEvent(
    context,
    errcode_ret);
  if (event)
    CHECK_CREATION(OCL_EVENT, event);
  return event;
}

static CL_API_ENTRY cl_int CL_API_CALL clSetUserEventStatus_wrap(
    cl_event event,
    cl_int execution_status)
{
  CHECK_EXISTS(OCL_EVENT, event);
  return tdispatch->clSetUserEventStatus(
    event,
    execution_status);
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBufferRect_wrap(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_read,
    const size_t* buffer_origin,
    const size_t* host_origin,
    const size_t* region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_BUFFER, buffer);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueReadBufferRect(
    command_queue,
    buffer,
    blocking_read,
    buffer_origin,
    host_origin,
    region,
    buffer_row_pitch,
    buffer_slice_pitch,
    host_row_pitch,
    host_slice_pitch,
    ptr,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteBufferRect_wrap(
    cl_command_queue command_queue,
    cl_mem buffer,
    cl_bool blocking_write,
    const size_t* buffer_origin,
    const size_t* host_origin,
    const size_t* region,
    size_t buffer_row_pitch,
    size_t buffer_slice_pitch,
    size_t host_row_pitch,
    size_t host_slice_pitch,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_BUFFER, buffer);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueWriteBufferRect(
    command_queue,
    buffer,
    blocking_write,
    buffer_origin,
    host_origin,
    region,
    buffer_row_pitch,
    buffer_slice_pitch,
    host_row_pitch,
    host_slice_pitch,
    ptr,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBufferRect_wrap(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_BUFFER, src_buffer);
  CHECK_EXISTS(OCL_BUFFER, dst_buffer);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueCopyBufferRect(
            command_queue,
            src_buffer,
            dst_buffer,
            src_origin,
            dst_origin,
            region,
            src_row_pitch,
            src_slice_pitch,
            dst_row_pitch,
            dst_slice_pitch,
            num_events_in_wait_list,
            event_wait_list,
            event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

  /* cl_ext_device_fission */
static CL_API_ENTRY cl_int CL_API_CALL
clCreateSubDevicesEXT_wrap(
    cl_device_id in_device,
    const cl_device_partition_property_ext* properties,
    cl_uint num_entries,
    cl_device_id* out_devices,
    cl_uint* num_devices)
{
  CHECK_EXISTS(OCL_DEVICE, in_device);

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
    CHECK_CREATION_LIST(OCL_SUB_DEVICE, *num_devices, out_devices);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainDeviceEXT_wrap(
    cl_device_id device)
{
  CHECK_RETAIN(OCL_DEVICE, device);
  return tdispatch->clRetainDeviceEXT(device);
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseDeviceEXT_wrap(
    cl_device_id device)
{
  CHECK_RELEASE(OCL_DEVICE, device);
  return tdispatch->clReleaseDeviceEXT(device);
}

  /* cl_khr_gl_event */
static CL_API_ENTRY cl_event CL_API_CALL clCreateEventFromGLsyncKHR_wrap(
    cl_context context,
    cl_GLsync sync,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_event event = tdispatch->clCreateEventFromGLsyncKHR(
    context,
    sync,
    errcode_ret);
  if (event)
    CHECK_CREATION(OCL_EVENT, event);
  return event;
}

  /* OpenCL 1.2 */
static CL_API_ENTRY cl_int CL_API_CALL
clCreateSubDevices_wrap(
    cl_device_id in_device,
    const cl_device_partition_property* properties,
    cl_uint num_devices,
    cl_device_id* out_devices,
    cl_uint* num_devices_ret)
{
  CHECK_EXISTS(OCL_DEVICE, in_device);

  cl_uint num_devices_ret_force;
  if (out_devices && !num_devices_ret)
    num_devices_ret = &num_devices_ret_force;

  cl_int result = tdispatch->clCreateSubDevices(
    in_device,
    properties,
    num_devices,
    out_devices,
    num_devices_ret);

  if (out_devices && result == CL_SUCCESS && *num_devices_ret > 0)
    CHECK_CREATION_LIST(OCL_SUB_DEVICE, *num_devices_ret, out_devices);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clRetainDevice_wrap(
    cl_device_id device)
{
  CHECK_RETAIN(OCL_DEVICE, device);
  return tdispatch->clRetainDevice(device);
}

static CL_API_ENTRY cl_int CL_API_CALL clReleaseDevice_wrap(
    cl_device_id device)
{
  CHECK_RELEASE(OCL_DEVICE, device);
  return tdispatch->clReleaseDevice(device);
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateImage_wrap(
    cl_context context,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  if (image_desc && image_desc->mem_object)
    CHECK_EXISTS(OCL_MEM, image_desc->mem_object);
  cl_mem image = tdispatch->clCreateImage(
    context,
    flags,
    image_format,
    image_desc,
    host_ptr,
    errcode_ret);
  if (image)
    CHECK_CREATION(OCL_IMAGE, image);
  return image;
}

static CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithBuiltInKernels_wrap(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* kernel_names,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_program program = tdispatch->clCreateProgramWithBuiltInKernels(
    context,
    num_devices,
    device_list,
    kernel_names,
    errcode_ret);
  if (program)
    CHECK_CREATION(OCL_PROGRAM, program);
  return program;
}

static CL_API_ENTRY cl_int CL_API_CALL clCompileProgram_wrap(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    cl_uint num_input_headers,
    const cl_program* input_headers,
    const char** header_include_names,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
  CHECK_EXISTS(OCL_PROGRAM, program);
  CHECK_EXIST_LIST(OCL_DEVICE, num_devices, device_list);
  return tdispatch->clCompileProgram(
    program,
    num_devices,
    device_list,
    options,
    num_input_headers,
    input_headers,
    header_include_names,
    pfn_notify,
    user_data);
}

static CL_API_ENTRY cl_program CL_API_CALL clLinkProgram_wrap(
    cl_context context,
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    cl_uint num_input_programs,
    const cl_program* input_programs,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  CHECK_EXIST_LIST(OCL_DEVICE, num_devices, device_list);
  CHECK_EXIST_LIST(OCL_PROGRAM, num_input_programs, input_programs);
  cl_program program = tdispatch->clLinkProgram(
    context,
    num_devices,
    device_list,
    options,
    num_input_programs,
    input_programs,
    pfn_notify,
    user_data,
    errcode_ret);
  if (program)
    CHECK_CREATION(OCL_PROGRAM, program);
  return program;
}

static CL_API_ENTRY cl_int CL_API_CALL clUnloadPlatformCompiler_wrap(
    cl_platform_id platform)
{
  CHECK_EXISTS(OCL_PLATFORM, platform);
  return tdispatch->clUnloadPlatformCompiler(
    platform);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetKernelArgInfo_wrap(
    cl_kernel kernel,
    cl_uint arg_index,
    cl_kernel_arg_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(OCL_KERNEL, kernel);
  return tdispatch->clGetKernelArgInfo(
    kernel,
    arg_index,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueFillBuffer_wrap(
    cl_command_queue command_queue,
    cl_mem buffer,
    const void* pattern,
    size_t pattern_size,
    size_t offset,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_BUFFER, buffer);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueFillBuffer(
    command_queue,
    buffer,
    pattern,
    pattern_size,
    offset,
    size,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueFillImage_wrap(
    cl_command_queue command_queue,
    cl_mem image,
    const void* fill_color,
    const size_t* origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXISTS(OCL_IMAGE, image);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueFillImage(
    command_queue,
    image,
    fill_color,
    origin,
    region,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueMigrateMemObjects_wrap(
    cl_command_queue command_queue,
    cl_uint num_mem_objects,
    const cl_mem* mem_objects,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_MEM, num_mem_objects, mem_objects);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueMigrateMemObjects(
    command_queue,
    num_mem_objects,
    mem_objects,
    flags,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueMarkerWithWaitList_wrap(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueMarkerWithWaitList(
    command_queue,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueBarrierWithWaitList_wrap(
    cl_command_queue command_queue,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueBarrierWithWaitList(
    command_queue,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY void* CL_API_CALL clGetExtensionFunctionAddressForPlatform_wrap(
    cl_platform_id platform,
    const char* func_name)
{
  CHECK_EXISTS(OCL_PLATFORM, platform);
  return tdispatch->clGetExtensionFunctionAddressForPlatform(
    platform,
    func_name);
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLTexture_wrap(
    cl_context context,
    cl_mem_flags flags,
    cl_GLenum target,
    cl_GLint miplevel,
    cl_GLuint texture,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem image = tdispatch->clCreateFromGLTexture(
    context,
    flags,
    target,
    miplevel,
    texture,
    errcode_ret);
  if (image)
    CHECK_CREATION(OCL_IMAGE, image);
  return image;
}

  /* cl_khr_d3d11_sharing */
  /* cl_khr_dx9_media_sharing */
#if defined(_WIN32)
static CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDsFromD3D11KHR_wrap(
    cl_platform_id platform,
    cl_d3d11_device_source_khr d3d_device_source,
    void* d3d_object,
    cl_d3d11_device_set_khr d3d_device_set,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
  CHECK_EXISTS(OCL_PLATFORM, platform);

  cl_uint num_devices_force;
  if (devices && !num_devices)
    num_devices = &num_devices_force;

  cl_int result = tdispatch->clGetDeviceIDsFromD3D11KHR(
    platform,
    d3d_device_source,
    d3d_object,
    d3d_device_set,
    num_entries,
    devices,
    num_devices);
  if (devices && result == CL_SUCCESS && *num_devices > 0)
    CHECK_CREATION_LIST(OCL_DEVICE, *num_devices, devices);
  return result;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromD3D11BufferKHR_wrap(
    cl_context context,
    cl_mem_flags flags,
    ID3D11Buffer* resource,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem buffer = tdispatch->clCreateFromD3D11BufferKHR(
    context,
    flags,
    resource,
    errcode_ret);
  if (buffer)
    CHECK_CREATION(OCL_BUFFER, buffer);
  return buffer;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromD3D11Texture2DKHR_wrap(
    cl_context context,
    cl_mem_flags flags,
    ID3D11Texture2D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem image = tdispatch->clCreateFromD3D11Texture2DKHR(
    context,
    flags,
    resource,
    subresource,
    errcode_ret);
  if (image)
    CHECK_CREATION(OCL_IMAGE, image);
  return image;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromD3D11Texture3DKHR_wrap(
    cl_context context,
    cl_mem_flags flags,
    ID3D11Texture3D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem image = tdispatch->clCreateFromD3D11Texture3DKHR(
    context,
    flags,
    resource,
    subresource,
    errcode_ret);
  if (image)
    CHECK_CREATION(OCL_IMAGE, image);
  return image;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromDX9MediaSurfaceKHR_wrap(
    cl_context context,
    cl_mem_flags flags,
    cl_dx9_media_adapter_type_khr adapter_type,
    void* surface_info,
    cl_uint plane,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem image = tdispatch->clCreateFromDX9MediaSurfaceKHR(
    context,
    flags,
    adapter_type,
    surface_info,
    plane,
    errcode_ret);
  if (image)
    CHECK_CREATION(OCL_IMAGE, image);
  return image;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueAcquireD3D11ObjectsKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_MEM, num_objects, mem_objects);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueAcquireD3D11ObjectsKHR(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseD3D11ObjectsKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_MEM, num_objects, mem_objects);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueReleaseD3D11ObjectsKHR(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDsFromDX9MediaAdapterKHR_wrap(
    cl_platform_id platform,
    cl_uint num_media_adapters,
    cl_dx9_media_adapter_type_khr* media_adapter_type,
    void* media_adapters,
    cl_dx9_media_adapter_set_khr media_adapter_set,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
  CHECK_EXISTS(OCL_PLATFORM, platform);

  cl_uint num_devices_force;
  if (devices && !num_devices)
    num_devices = &num_devices_force;

  cl_int result = tdispatch->clGetDeviceIDsFromDX9MediaAdapterKHR(
    platform,
    num_media_adapters,
    media_adapter_type,
    media_adapters,
    media_adapter_set,
    num_entries,
    devices,
    num_devices);
  if (devices && result == CL_SUCCESS && *num_devices > 0)
    CHECK_CREATION_LIST(OCL_DEVICE, *num_devices, devices);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueAcquireDX9MediaSurfacesKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_MEM, num_objects, mem_objects);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueAcquireDX9MediaSurfacesKHR(
            command_queue,
            num_objects,
            mem_objects,
            num_events_in_wait_list,
            event_wait_list,
            event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseDX9MediaSurfacesKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_MEM, num_objects, mem_objects);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueReleaseDX9MediaSurfacesKHR(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}
#endif

  /* cl_khr_egl_image */
static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromEGLImageKHR_wrap(
    cl_context context,
    CLeglDisplayKHR egldisplay,
    CLeglImageKHR eglimage,
    cl_mem_flags flags,
    const cl_egl_image_properties_khr* properties,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem image = tdispatch->clCreateFromEGLImageKHR(
    context,
    egldisplay,
    eglimage,
    flags,
    properties,
    errcode_ret);
  if (image)
    CHECK_CREATION(OCL_IMAGE, image);
  return image;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueAcquireEGLObjectsKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_MEM, num_objects, mem_objects);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueAcquireEGLObjectsKHR(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseEGLObjectsKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_MEM, num_objects, mem_objects);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueReleaseEGLObjectsKHR(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

  /* cl_khr_egl_event */
static CL_API_ENTRY cl_event CL_API_CALL clCreateEventFromEGLSyncKHR_wrap(
    cl_context context,
    CLeglSyncKHR sync,
    CLeglDisplayKHR display,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  return tdispatch->clCreateEventFromEGLSyncKHR(
    context,
    sync,
    display,
    errcode_ret);
}

  /* OpenCL 2.0 */
static CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueueWithProperties_wrap(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties* properties,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  CHECK_EXISTS(OCL_DEVICE, device);
  cl_command_queue command_queue = tdispatch->clCreateCommandQueueWithProperties(
    context,
    device,
    properties,
    errcode_ret);
  if (command_queue)
    CHECK_CREATION(OCL_COMMAND_QUEUE, command_queue);
  return command_queue;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreatePipe_wrap(
    cl_context context,
    cl_mem_flags flags,
    cl_uint pipe_packet_size,
    cl_uint pipe_max_packets,
    const cl_pipe_properties* properties,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem pipe = tdispatch->clCreatePipe(
    context,
    flags,
    pipe_packet_size,
    pipe_max_packets,
    properties,
    errcode_ret);
  if (pipe)
    CHECK_CREATION(OCL_PIPE, pipe);
  return pipe;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetPipeInfo_wrap(
    cl_mem pipe,
    cl_pipe_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(OCL_PIPE, pipe);
  return tdispatch->clGetPipeInfo(
    pipe,
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret);
}

static CL_API_ENTRY void* CL_API_CALL clSVMAlloc_wrap(
    cl_context context,
    cl_svm_mem_flags flags,
    size_t size,
    cl_uint alignment)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  return tdispatch->clSVMAlloc(
    context,
    flags,
    size,
    alignment);
}

static CL_API_ENTRY void CL_API_CALL clSVMFree_wrap(
    cl_context context,
    void* svm_pointer)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  tdispatch->clSVMFree(
    context,
    svm_pointer);
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMFree_wrap(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    void* svm_pointers[],
    void (CL_CALLBACK* pfn_free_func)(cl_command_queue queue, cl_uint num_svm_pointers, void* svm_pointers[], void* user_data),
    void* user_data,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueSVMFree(
    command_queue,
    num_svm_pointers,
    svm_pointers,
    pfn_free_func,
    user_data,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMMemcpy_wrap(
    cl_command_queue command_queue,
    cl_bool blocking_copy,
    void* dst_ptr,
    const void* src_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueSVMMemcpy(
    command_queue,
    blocking_copy,
    dst_ptr,
    src_ptr,
    size,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMMemFill_wrap(
    cl_command_queue command_queue,
    void* svm_ptr,
    const void* pattern,
    size_t pattern_size,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueSVMMemFill(
    command_queue,
    svm_ptr,
    pattern,
    pattern_size,
    size,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMMap_wrap(
    cl_command_queue command_queue,
    cl_bool blocking_map,
    cl_map_flags flags,
    void* svm_ptr,
    size_t size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueSVMMap(
    command_queue,
    blocking_map,
    flags,
    svm_ptr,
    size,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMUnmap_wrap(
    cl_command_queue command_queue,
    void* svm_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueSVMUnmap(
    command_queue,
    svm_ptr,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_sampler CL_API_CALL clCreateSamplerWithProperties_wrap(
    cl_context context,
    const cl_sampler_properties* sampler_properties,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_sampler sampler = tdispatch->clCreateSamplerWithProperties(
    context,
    sampler_properties,
    errcode_ret);
  if (sampler)
    CHECK_CREATION(OCL_SAMPLER, sampler);
  return sampler;
}

static CL_API_ENTRY cl_int CL_API_CALL clSetKernelArgSVMPointer_wrap(
    cl_kernel kernel,
    cl_uint arg_index,
    const void* arg_value)
{
  CHECK_EXISTS(OCL_KERNEL, kernel);
  return tdispatch->clSetKernelArgSVMPointer(
    kernel,
    arg_index,
    arg_value);
}

static CL_API_ENTRY cl_int CL_API_CALL clSetKernelExecInfo_wrap(
    cl_kernel kernel,
    cl_kernel_exec_info param_name,
    size_t param_value_size,
    const void* param_value)
{
  CHECK_EXISTS(OCL_KERNEL, kernel);
  return tdispatch->clSetKernelExecInfo(
    kernel,
    param_name,
    param_value_size,
    param_value);
}

  /* cl_khr_sub_groups */
static CL_API_ENTRY cl_int CL_API_CALL clGetKernelSubGroupInfoKHR_wrap(
    cl_kernel in_kernel,
    cl_device_id in_device,
    cl_kernel_sub_group_info param_name,
    size_t input_value_size,
    const void* input_value,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(OCL_KERNEL, in_kernel);
  if (in_device)
    CHECK_EXISTS(OCL_DEVICE, in_device);
  return tdispatch->clGetKernelSubGroupInfoKHR(
    in_kernel,
    in_device,
    param_name,
    input_value_size,
    input_value,
    param_value_size,
    param_value,
    param_value_size_ret);
}

  /* OpenCL 2.1 */
static CL_API_ENTRY cl_kernel CL_API_CALL clCloneKernel_wrap(
    cl_kernel source_kernel,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_KERNEL, source_kernel);
  cl_kernel kernel = tdispatch->clCloneKernel(
    source_kernel,
    errcode_ret);
  if (kernel)
    CHECK_CREATION(OCL_KERNEL, kernel);
  return kernel;
}

static CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithIL_wrap(
    cl_context context,
    const void* il,
    size_t length,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_program program = tdispatch->clCreateProgramWithIL(
    context,
    il,
    length,
    errcode_ret);
  if (program)
    CHECK_CREATION(OCL_PROGRAM, program);
  return program;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueSVMMigrateMem_wrap(
    cl_command_queue command_queue,
    cl_uint num_svm_pointers,
    const void** svm_pointers,
    const size_t* sizes,
    cl_mem_migration_flags flags,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  CHECK_EXIST_LIST(OCL_EVENT, num_events_in_wait_list, event_wait_list);
  cl_int result = tdispatch->clEnqueueSVMMigrateMem(
    command_queue,
    num_svm_pointers,
    svm_pointers,
    sizes,
    flags,
    num_events_in_wait_list,
    event_wait_list,
    event);
  if (result == CL_SUCCESS && event)
    CHECK_CREATION(OCL_EVENT, *event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clGetDeviceAndHostTimer_wrap(
    cl_device_id device,
    cl_ulong* device_timestamp,
    cl_ulong* host_timestamp)
{
  CHECK_EXISTS(OCL_DEVICE, device);
  return tdispatch->clGetDeviceAndHostTimer(
    device,
    device_timestamp,
    host_timestamp);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetHostTimer_wrap(
    cl_device_id device,
    cl_ulong* host_timestamp)
{
  CHECK_EXISTS(OCL_DEVICE, device);
  return tdispatch->clGetHostTimer(
    device,
    host_timestamp);
}

static CL_API_ENTRY cl_int CL_API_CALL clGetKernelSubGroupInfo_wrap(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_sub_group_info param_name,
    size_t input_value_size,
    const void* input_value,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  CHECK_EXISTS(OCL_KERNEL, kernel);
  if (device)
    CHECK_EXISTS(OCL_DEVICE, device);
  return tdispatch->clGetKernelSubGroupInfo(
    kernel,
    device,
    param_name,
    input_value_size,
    input_value,
    param_value_size,
    param_value,
    param_value_size_ret);
}

static CL_API_ENTRY cl_int CL_API_CALL clSetDefaultDeviceCommandQueue_wrap(
    cl_context context,
    cl_device_id device,
    cl_command_queue command_queue)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  CHECK_EXISTS(OCL_DEVICE, device);
  CHECK_EXISTS(OCL_COMMAND_QUEUE, command_queue);
  return tdispatch->clSetDefaultDeviceCommandQueue(
    context,
    device,
    command_queue);
}

  /* OpenCL 2.2 */
static CL_API_ENTRY cl_int CL_API_CALL clSetProgramReleaseCallback_wrap(
    cl_program program,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data)
{
  CHECK_EXISTS(OCL_PROGRAM, program);
  return tdispatch->clSetProgramReleaseCallback(
    program,
    pfn_notify,
    user_data);
}

static CL_API_ENTRY cl_int CL_API_CALL clSetProgramSpecializationConstant_wrap(
    cl_program program,
    cl_uint spec_id,
    size_t spec_size,
    const void* spec_value)
{
  CHECK_EXISTS(OCL_PROGRAM, program);
  return tdispatch->clSetProgramSpecializationConstant(
    program,
    spec_id,
    spec_size,
    spec_value);
}

  /* OpenCL 3.0 */
static CL_API_ENTRY cl_mem CL_API_CALL clCreateBufferWithProperties_wrap(
    cl_context context,
    const cl_mem_properties* properties,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  cl_mem buffer = tdispatch->clCreateBufferWithProperties(
    context,
    properties,
    flags,
    size,
    host_ptr,
    errcode_ret);
  if (buffer)
    CHECK_CREATION(OCL_BUFFER, buffer);
  return buffer;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateImageWithProperties_wrap(
    cl_context context,
    const cl_mem_properties* properties,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  if (image_desc && image_desc->mem_object)
    CHECK_EXISTS(OCL_MEM, image_desc->mem_object);
  cl_mem image = tdispatch->clCreateImageWithProperties(
    context,
    properties,
    flags,
    image_format,
    image_desc,
    host_ptr,
    errcode_ret);
  if (image)
    CHECK_CREATION(OCL_IMAGE, image);
  return image;
}

static CL_API_ENTRY cl_int CL_API_CALL clSetContextDestructorCallback_wrap(
    cl_context context,
    void (CL_CALLBACK* pfn_notify)(cl_context context, void* user_data),
    void* user_data)
{
  CHECK_EXISTS(OCL_CONTEXT, context);
  return tdispatch->clSetContextDestructorCallback(
    context,
    pfn_notify,
    user_data);
}

static void _init_dispatch(void) {
  dispatch.clGetPlatformIDs = &clGetPlatformIDs_wrap;
  dispatch.clGetPlatformInfo = &clGetPlatformInfo_wrap;
  dispatch.clGetDeviceIDs = &clGetDeviceIDs_wrap;
  dispatch.clGetDeviceInfo = &clGetDeviceInfo_wrap;
  dispatch.clCreateContext = &clCreateContext_wrap;
  dispatch.clCreateContextFromType = &clCreateContextFromType_wrap;
  dispatch.clRetainContext = &clRetainContext_wrap;
  dispatch.clReleaseContext = &clReleaseContext_wrap;
  dispatch.clGetContextInfo = &clGetContextInfo_wrap;
  dispatch.clCreateCommandQueue = &clCreateCommandQueue_wrap;
  dispatch.clRetainCommandQueue = &clRetainCommandQueue_wrap;
  dispatch.clReleaseCommandQueue = &clReleaseCommandQueue_wrap;
  dispatch.clGetCommandQueueInfo = &clGetCommandQueueInfo_wrap;
  dispatch.clSetCommandQueueProperty = &clSetCommandQueueProperty_wrap;
  dispatch.clCreateBuffer = &clCreateBuffer_wrap;
  dispatch.clCreateImage2D = &clCreateImage2D_wrap;
  dispatch.clCreateImage3D = &clCreateImage3D_wrap;
  dispatch.clRetainMemObject = &clRetainMemObject_wrap;
  dispatch.clReleaseMemObject = &clReleaseMemObject_wrap;
  dispatch.clGetSupportedImageFormats = &clGetSupportedImageFormats_wrap;
  dispatch.clGetMemObjectInfo = &clGetMemObjectInfo_wrap;
  dispatch.clGetImageInfo = &clGetImageInfo_wrap;
  dispatch.clCreateSampler = &clCreateSampler_wrap;
  dispatch.clRetainSampler = &clRetainSampler_wrap;
  dispatch.clReleaseSampler = &clReleaseSampler_wrap;
  dispatch.clGetSamplerInfo = &clGetSamplerInfo_wrap;
  dispatch.clCreateProgramWithSource = &clCreateProgramWithSource_wrap;
  dispatch.clCreateProgramWithBinary = &clCreateProgramWithBinary_wrap;
  dispatch.clRetainProgram = &clRetainProgram_wrap;
  dispatch.clReleaseProgram = &clReleaseProgram_wrap;
  dispatch.clBuildProgram = &clBuildProgram_wrap;
  dispatch.clUnloadCompiler = NULL;
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
  dispatch.clEnqueueCopyBuffer = &clEnqueueCopyBuffer_wrap;
  dispatch.clEnqueueReadImage = &clEnqueueReadImage_wrap;
  dispatch.clEnqueueWriteImage = &clEnqueueWriteImage_wrap;
  dispatch.clEnqueueCopyImage = &clEnqueueCopyImage_wrap;
  dispatch.clEnqueueCopyImageToBuffer = &clEnqueueCopyImageToBuffer_wrap;
  dispatch.clEnqueueCopyBufferToImage = &clEnqueueCopyBufferToImage_wrap;
  dispatch.clEnqueueMapBuffer = &clEnqueueMapBuffer_wrap;
  dispatch.clEnqueueMapImage = &clEnqueueMapImage_wrap;
  dispatch.clEnqueueUnmapMemObject = &clEnqueueUnmapMemObject_wrap;
  dispatch.clEnqueueNDRangeKernel = &clEnqueueNDRangeKernel_wrap;
  dispatch.clEnqueueTask = &clEnqueueTask_wrap;
  dispatch.clEnqueueNativeKernel = &clEnqueueNativeKernel_wrap;
  dispatch.clEnqueueMarker = &clEnqueueMarker_wrap;
  dispatch.clEnqueueWaitForEvents = &clEnqueueWaitForEvents_wrap;
  dispatch.clEnqueueBarrier = &clEnqueueBarrier_wrap;
  dispatch.clGetExtensionFunctionAddress = NULL;
  dispatch.clCreateFromGLBuffer = &clCreateFromGLBuffer_wrap;
  dispatch.clCreateFromGLTexture2D = &clCreateFromGLTexture2D_wrap;
  dispatch.clCreateFromGLTexture3D = &clCreateFromGLTexture3D_wrap;
  dispatch.clCreateFromGLRenderbuffer = &clCreateFromGLRenderbuffer_wrap;
  dispatch.clGetGLObjectInfo = &clGetGLObjectInfo_wrap;
  dispatch.clGetGLTextureInfo = &clGetGLTextureInfo_wrap;
  dispatch.clEnqueueAcquireGLObjects = &clEnqueueAcquireGLObjects_wrap;
  dispatch.clEnqueueReleaseGLObjects = &clEnqueueReleaseGLObjects_wrap;
  dispatch.clGetGLContextInfoKHR = &clGetGLContextInfoKHR_wrap;

  /* cl_khr_d3d10_sharing */
#if defined(_WIN32)
  dispatch.clGetDeviceIDsFromD3D10KHR = &clGetDeviceIDsFromD3D10KHR_wrap;
  dispatch.clCreateFromD3D10BufferKHR = &clCreateFromD3D10BufferKHR_wrap;
  dispatch.clCreateFromD3D10Texture2DKHR = &clCreateFromD3D10Texture2DKHR_wrap;
  dispatch.clCreateFromD3D10Texture3DKHR = &clCreateFromD3D10Texture3DKHR_wrap;
  dispatch.clEnqueueAcquireD3D10ObjectsKHR = &clEnqueueAcquireD3D10ObjectsKHR_wrap;
  dispatch.clEnqueueReleaseD3D10ObjectsKHR = &clEnqueueReleaseD3D10ObjectsKHR_wrap;
#else
  dispatch.clGetDeviceIDsFromD3D10KHR = NULL;
  dispatch.clCreateFromD3D10BufferKHR = NULL;
  dispatch.clCreateFromD3D10Texture2DKHR = NULL;
  dispatch.clCreateFromD3D10Texture3DKHR = NULL;
  dispatch.clEnqueueAcquireD3D10ObjectsKHR = NULL;
  dispatch.clEnqueueReleaseD3D10ObjectsKHR = NULL;
#endif

  /* OpenCL 1.1 */
  dispatch.clSetEventCallback = &clSetEventCallback_wrap;
  dispatch.clCreateSubBuffer = &clCreateSubBuffer_wrap;
  dispatch.clSetMemObjectDestructorCallback = &clSetMemObjectDestructorCallback_wrap;
  dispatch.clCreateUserEvent = &clCreateUserEvent_wrap;
  dispatch.clSetUserEventStatus = &clSetUserEventStatus_wrap;
  dispatch.clEnqueueReadBufferRect = &clEnqueueReadBufferRect_wrap;
  dispatch.clEnqueueWriteBufferRect = &clEnqueueWriteBufferRect_wrap;
  dispatch.clEnqueueCopyBufferRect = &clEnqueueCopyBufferRect_wrap;

  /* cl_ext_device_fission */
  dispatch.clCreateSubDevicesEXT = &clCreateSubDevicesEXT_wrap;
  dispatch.clRetainDeviceEXT = &clRetainDeviceEXT_wrap;
  dispatch.clReleaseDeviceEXT = &clReleaseDeviceEXT_wrap;

  /* cl_khr_gl_event */
  dispatch.clCreateEventFromGLsyncKHR = &clCreateEventFromGLsyncKHR_wrap;

  /* OpenCL 1.2 */
  dispatch.clCreateSubDevices = &clCreateSubDevices_wrap;
  dispatch.clRetainDevice = &clRetainDevice_wrap;
  dispatch.clReleaseDevice = &clReleaseDevice_wrap;
  dispatch.clCreateImage = &clCreateImage_wrap;
  dispatch.clCreateProgramWithBuiltInKernels = &clCreateProgramWithBuiltInKernels_wrap;
  dispatch.clCompileProgram = &clCompileProgram_wrap;
  dispatch.clLinkProgram = &clLinkProgram_wrap;
  dispatch.clUnloadPlatformCompiler = &clUnloadPlatformCompiler_wrap;
  dispatch.clGetKernelArgInfo = &clGetKernelArgInfo_wrap;
  dispatch.clEnqueueFillBuffer = &clEnqueueFillBuffer_wrap;
  dispatch.clEnqueueFillImage = &clEnqueueFillImage_wrap;
  dispatch.clEnqueueMigrateMemObjects = &clEnqueueMigrateMemObjects_wrap;
  dispatch.clEnqueueMarkerWithWaitList = &clEnqueueMarkerWithWaitList_wrap;
  dispatch.clEnqueueBarrierWithWaitList = &clEnqueueBarrierWithWaitList_wrap;
  dispatch.clGetExtensionFunctionAddressForPlatform = &clGetExtensionFunctionAddressForPlatform_wrap;
  dispatch.clCreateFromGLTexture = &clCreateFromGLTexture_wrap;

  /* cl_khr_d3d11_sharing */
  /* cl_khr_dx9_media_sharing */
#if defined(_WIN32)
  dispatch.clGetDeviceIDsFromD3D11KHR = &clGetDeviceIDsFromD3D11KHR_wrap;
  dispatch.clCreateFromD3D11BufferKHR = &clCreateFromD3D11BufferKHR_wrap;
  dispatch.clCreateFromD3D11Texture2DKHR = &clCreateFromD3D11Texture2DKHR_wrap;
  dispatch.clCreateFromD3D11Texture3DKHR = &clCreateFromD3D11Texture3DKHR_wrap;
  dispatch.clCreateFromDX9MediaSurfaceKHR = &clCreateFromDX9MediaSurfaceKHR_wrap;
  dispatch.clEnqueueAcquireD3D11ObjectsKHR = &clEnqueueAcquireD3D11ObjectsKHR_wrap;
  dispatch.clEnqueueReleaseD3D11ObjectsKHR = &clEnqueueReleaseD3D11ObjectsKHR_wrap;
  dispatch.clGetDeviceIDsFromDX9MediaAdapterKHR = &clGetDeviceIDsFromDX9MediaAdapterKHR_wrap;
  dispatch.clEnqueueAcquireDX9MediaSurfacesKHR = &clEnqueueAcquireDX9MediaSurfacesKHR_wrap;
  dispatch.clEnqueueReleaseDX9MediaSurfacesKHR = &clEnqueueReleaseDX9MediaSurfacesKHR_wrap;
#else
  dispatch.clGetDeviceIDsFromD3D11KHR = NULL;
  dispatch.clCreateFromD3D11BufferKHR = NULL;
  dispatch.clCreateFromD3D11Texture2DKHR = NULL;
  dispatch.clCreateFromD3D11Texture3DKHR = NULL;
  dispatch.clCreateFromDX9MediaSurfaceKHR = NULL;
  dispatch.clEnqueueAcquireD3D11ObjectsKHR = NULL;
  dispatch.clEnqueueReleaseD3D11ObjectsKHR = NULL;
  dispatch.clGetDeviceIDsFromDX9MediaAdapterKHR = NULL;
  dispatch.clEnqueueAcquireDX9MediaSurfacesKHR = NULL;
  dispatch.clEnqueueReleaseDX9MediaSurfacesKHR = NULL;
#endif

  /* cl_khr_egl_image */
  dispatch.clCreateFromEGLImageKHR = &clCreateFromEGLImageKHR_wrap;
  dispatch.clEnqueueAcquireEGLObjectsKHR = &clEnqueueAcquireEGLObjectsKHR_wrap;
  dispatch.clEnqueueReleaseEGLObjectsKHR = &clEnqueueReleaseEGLObjectsKHR_wrap;

  /* cl_khr_egl_event */
  dispatch.clCreateEventFromEGLSyncKHR = &clCreateEventFromEGLSyncKHR_wrap;

  /* OpenCL 2.0 */
  dispatch.clCreateCommandQueueWithProperties = &clCreateCommandQueueWithProperties_wrap;
  dispatch.clCreatePipe = &clCreatePipe_wrap;
  dispatch.clGetPipeInfo = &clGetPipeInfo_wrap;
  dispatch.clSVMAlloc = &clSVMAlloc_wrap;
  dispatch.clSVMFree = &clSVMFree_wrap;
  dispatch.clEnqueueSVMFree = &clEnqueueSVMFree_wrap;
  dispatch.clEnqueueSVMMemcpy = &clEnqueueSVMMemcpy_wrap;
  dispatch.clEnqueueSVMMemFill = &clEnqueueSVMMemFill_wrap;
  dispatch.clEnqueueSVMMap = &clEnqueueSVMMap_wrap;
  dispatch.clEnqueueSVMUnmap = &clEnqueueSVMUnmap_wrap;
  dispatch.clCreateSamplerWithProperties = &clCreateSamplerWithProperties_wrap;
  dispatch.clSetKernelArgSVMPointer = &clSetKernelArgSVMPointer_wrap;
  dispatch.clSetKernelExecInfo = &clSetKernelExecInfo_wrap;

  /* cl_khr_sub_groups */
  dispatch.clGetKernelSubGroupInfoKHR = &clGetKernelSubGroupInfoKHR_wrap;

  /* OpenCL 2.1 */
  dispatch.clCloneKernel = &clCloneKernel_wrap;
  dispatch.clCreateProgramWithIL = &clCreateProgramWithIL_wrap;
  dispatch.clEnqueueSVMMigrateMem = &clEnqueueSVMMigrateMem_wrap;
  dispatch.clGetDeviceAndHostTimer = &clGetDeviceAndHostTimer_wrap;
  dispatch.clGetHostTimer = &clGetHostTimer_wrap;
  dispatch.clGetKernelSubGroupInfo = &clGetKernelSubGroupInfo_wrap;
  dispatch.clSetDefaultDeviceCommandQueue = &clSetDefaultDeviceCommandQueue_wrap;

  /* OpenCL 2.2 */
  dispatch.clSetProgramReleaseCallback = &clSetProgramReleaseCallback_wrap;
  dispatch.clSetProgramSpecializationConstant = &clSetProgramSpecializationConstant_wrap;

  /* OpenCL 3.0 */
  dispatch.clCreateBufferWithProperties = &clCreateBufferWithProperties_wrap;
  dispatch.clCreateImageWithProperties = &clCreateImageWithProperties_wrap;
  dispatch.clSetContextDestructorCallback = &clSetContextDestructorCallback_wrap;
}
