#pragma once

#if defined(_WIN32)
#define NOMINMAX
#endif

#include <CL/cl_icd.h>  // cl_icd_dispatch

#include <vector>       // std::vector
#include <memory>       // std::unique_ptr
#include <string>       // std::string
#include <map>          // std::map<std::string, void*> _extensions
#include <set>          // std::set<std::shared_ptr<_cl_object>> _objects
#include <utility>      // std::make_pair

namespace lifetime
{
  extern bool report_implicit_ref_count_to_user,
              allow_using_released_objects,
              allow_using_inaccessible_objects;

  template <typename T> cl_int CL_INVALID();
  template <> inline cl_int CL_INVALID<cl_platform_id>() { return CL_INVALID_PLATFORM; }
  template <> inline cl_int CL_INVALID<cl_device_id>() { return CL_INVALID_DEVICE; }
  template <> inline cl_int CL_INVALID<cl_context>() { return CL_INVALID_CONTEXT; }
  template <> inline cl_int CL_INVALID<cl_mem>() { return CL_INVALID_MEM_OBJECT; }
  template <> inline cl_int CL_INVALID<cl_command_queue>() { return CL_INVALID_COMMAND_QUEUE; }
  template <> inline cl_int CL_INVALID<cl_program>() { return CL_INVALID_PROGRAM; }
  template <> inline cl_int CL_INVALID<cl_kernel>() { return CL_INVALID_KERNEL; }
  template <> inline cl_int CL_INVALID<cl_event>() { return CL_INVALID_EVENT; }
  template <> inline cl_int CL_INVALID<cl_sampler>() { return CL_INVALID_SAMPLER; }

  template <typename Object> struct object_parents;

  template <> struct object_parents<cl_device_id>
  {
    cl_device_id parent;
    void notify();
  };

  template <> struct object_parents<cl_context>
  {
    std::vector<cl_device_id> parent_devices;
    void notify();
  };

  template <> struct object_parents<cl_command_queue>
  {
    cl_device_id parent_device;
    cl_context parent_context;
    void notify();
  };

  template <> struct object_parents<cl_mem>
  {
    cl_mem parent_mem;
    cl_context parent_context;
    void notify();
  };

  template <> struct object_parents<cl_program>
  {
    cl_context parent_context;
    std::vector<cl_device_id> parent_devices;
    void notify();
  };

  template <> struct object_parents<cl_kernel>
  {
    cl_program parent_program;
    void notify();
  };

  template <> struct object_parents<cl_event>
  {
    cl_context parent_context;
    cl_command_queue parent_queue;
    void notify();
  };

  template <> struct object_parents<cl_sampler>
  {
    cl_context parent_context;
    void notify();
  };

  template <typename Object> struct ref_counted_object
  {
    cl_uint ref_count;
    cl_uint implicit_ref_count;
    object_parents<Object> parents;

    ref_counted_object(object_parents<Object> parents);
    ref_counted_object(const ref_counted_object&) = delete;
    ref_counted_object(ref_counted_object&&) = delete;
    ~ref_counted_object() = default;
    ref_counted_object &operator=(const ref_counted_object&) = delete;
    ref_counted_object &operator=(ref_counted_object&&) = delete;

    cl_int retain();
    cl_int release();

    cl_uint CL_OBJECT_REFERENCE_COUNT();

    bool is_valid(bool retain = false);

    void reference(int count = 1);
    void unreference();
  };

  template <typename Object>
  ref_counted_object<Object>::ref_counted_object(object_parents<Object> parents)
    : ref_count{ 1 }
    , implicit_ref_count{ 0 }
    , parents{ parents }
  {}

  template <typename Object>
  cl_int ref_counted_object<Object>::retain()
  {
    if (ref_count > 0 || (implicit_ref_count > 0 && allow_using_inaccessible_objects))
    {
      ++ref_count;
      return CL_SUCCESS;
    }
    else
      return CL_INVALID<Object>();
  }

  template <typename Object>
  cl_int ref_counted_object<Object>::release()
  {
    if (ref_count > 0)
    {
      --ref_count;
      if (ref_count == 0 && implicit_ref_count == 0)
        parents.notify();
      return CL_SUCCESS;
    }
    else
      return CL_INVALID<Object>();
  }

  template <typename Object>
  cl_uint ref_counted_object<Object>::CL_OBJECT_REFERENCE_COUNT()
  {
    if (report_implicit_ref_count_to_user)
      return ref_count + implicit_ref_count;
    else
      return ref_count;
  }

  template <typename Object>
  bool ref_counted_object<Object>::is_valid(bool retain)
  {
    if (ref_count > 0)
      return true; // More retains than releases
    else if (implicit_ref_count > 0 && (allow_using_inaccessible_objects || retain))
      return true; // User-visible ref_count may still be non-zero via implicit_ref_count
    else if (allow_using_released_objects)
      return true; // Runtimes releasing objects in a deferred fashion
    else
      return false;
  }

  template <typename Object>
  void ref_counted_object<Object>::reference(int count)
  {
    implicit_ref_count += count;
  }

  template <typename Object>
  void ref_counted_object<Object>::unreference()
  {
    --implicit_ref_count;
  }

  struct icd_compatible
  {
    cl_icd_dispatch* dispatch;

    icd_compatible();
    icd_compatible(const icd_compatible&) = delete;
    icd_compatible(icd_compatible&&) = delete;
    ~icd_compatible() = default;
    icd_compatible &operator=(const icd_compatible&) = delete;
    icd_compatible &operator=(icd_compatible&&) = delete;
  };
}

struct _cl_device_id
  : public lifetime::icd_compatible
  , public lifetime::ref_counted_object<cl_device_id>
{
  cl_platform_id platform;
  enum class device_kind
  {
    root,
    sub
  };
  device_kind kind;
  cl_version numeric_version;
  std::string profile,
              version,
              name,
              vendor,
              extensions;
  cl_uint cu_count;

  _cl_device_id() = delete;
  _cl_device_id(device_kind kind, cl_device_id parent = nullptr, cl_uint num_cu = 64);
  _cl_device_id(const _cl_device_id&) = delete;
  _cl_device_id(_cl_device_id&&) = delete;
  ~_cl_device_id() = default;
  _cl_device_id &operator=(const _cl_device_id&) = delete;
  _cl_device_id &operator=(_cl_device_id&&) = delete;

  void init_dispatch();

  cl_int clGetDeviceInfo(
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

  cl_int clCreateSubDevices(
    const cl_device_partition_property* properties,
    cl_uint num_devices,
    cl_device_id* out_devices,
    cl_uint* num_devices_ret);

  cl_int clRetainDevice();

  cl_int clReleaseDevice();

  cl_context clCreateContext(
  const cl_context_properties* properties,
  cl_uint num_devices,
  const cl_device_id* devices,
  void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
  void* user_data,
  cl_int* errcode_ret);
};

struct _cl_mem
  : public lifetime::icd_compatible
  , public lifetime::ref_counted_object<cl_mem>
{
  size_t _size;

  _cl_mem() = delete;
  _cl_mem(cl_mem mem_parent, cl_context context_parent, size_t size);
  _cl_mem(const _cl_mem&) = delete;
  _cl_mem(_cl_mem&&) = delete;
  ~_cl_mem() = default;
  _cl_mem &operator=(const _cl_mem&) = delete;
  _cl_mem &operator=(_cl_mem&&) = delete;

  void init_dispatch();

  cl_mem clCreateSubBuffer(
    cl_mem_flags flags,
    cl_buffer_create_type buffer_create_type,
    const void* buffer_create_info,
    cl_int* errcode_ret);

  cl_int clGetMemObjectInfo(
    cl_mem_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

  cl_int clRetainMemObject();

  cl_int clReleaseMemObject();
};

struct _cl_command_queue
  : public lifetime::icd_compatible
  , public lifetime::ref_counted_object<cl_command_queue>
{
  size_t _size;

  _cl_command_queue() = delete;
  _cl_command_queue(cl_device_id parent_device, cl_context parent_context);
  _cl_command_queue(const _cl_command_queue&) = delete;
  _cl_command_queue(_cl_command_queue&&) = delete;
  ~_cl_command_queue() = default;
  _cl_command_queue &operator=(const _cl_command_queue&) = delete;
  _cl_command_queue &operator=(_cl_command_queue&&) = delete;

  void init_dispatch();

  cl_int clGetCommandQueueInfo(
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

  cl_int clEnqueueNDRangeKernel(
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    const size_t* local_work_size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event);

  cl_int clRetainCommandQueue();

  cl_int clReleaseCommandQueue();
};

struct _cl_context
  : public lifetime::icd_compatible
  , public lifetime::ref_counted_object<cl_context>
{
  _cl_context() = delete;
  _cl_context(const cl_device_id* first_device, const cl_device_id* last_device);
  _cl_context(const _cl_context&) = delete;
  _cl_context(_cl_context&&) = delete;
  ~_cl_context() = default;
  _cl_context &operator=(const _cl_context&) = delete;
  _cl_context &operator=(_cl_context&&) = delete;

  void init_dispatch();

  cl_int clGetContextInfo(
    cl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

  cl_int clRetainContext();

  cl_int clReleaseContext();

  cl_mem clCreateBuffer(
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret);

  cl_mem clCreateBufferWithProperties(
    const cl_mem_properties* properties,
    cl_mem_flags flags,
    size_t size,
    void* host_ptr,
    cl_int* errcode_ret);

  cl_mem clCreateImage(
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret);

  cl_mem clCreateImageWithProperties(
    const cl_mem_properties* properties,
    cl_mem_flags flags,
    const cl_image_format* image_format,
    const cl_image_desc* image_desc,
    void* host_ptr,
    cl_int* errcode_ret);

  cl_mem clCreateImage2D(
    cl_mem_flags flags,
    const cl_image_format* image_format,
    size_t image_width,
    size_t image_height,
    size_t image_row_pitch,
    void* host_ptr,
    cl_int* errcode_ret);

  cl_mem clCreateImage3D(
    cl_mem_flags flags,
    const cl_image_format* image_format,
    size_t image_width,
    size_t image_height,
    size_t image_depth,
    size_t image_row_pitch,
    size_t image_slice_pitch,
    void* host_ptr,
    cl_int* errcode_ret);

  cl_command_queue clCreateCommandQueue(
    cl_device_id device,
    cl_command_queue_properties properties,
    cl_int* errcode_ret);

  cl_program clCreateProgramWithSource(
    cl_uint count,
    const char** strings,
    const size_t* lengths,
    cl_int* errcode_ret);

  cl_event clCreateUserEvent(
    cl_int* errcode_ret);

  cl_sampler clCreateSampler(
    cl_bool normalized_coords,
    cl_addressing_mode addressing_mode,
    cl_filter_mode filter_mode,
    cl_int* errcode_ret);
};

struct _cl_program
  : public lifetime::icd_compatible
  , public lifetime::ref_counted_object<cl_program>
{
  _cl_program() = delete;
  _cl_program(const cl_context parent_context, const cl_device_id* first_device = nullptr, const cl_device_id* last_device = nullptr);
  _cl_program(const _cl_program&) = delete;
  _cl_program(_cl_program&&) = delete;
  ~_cl_program() = default;
  _cl_program &operator=(const _cl_program&) = delete;
  _cl_program &operator=(_cl_program&&) = delete;

  cl_int clBuildProgram(
    cl_uint num_devices,
    const cl_device_id* device_list,
    const char* options,
    void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
    void* user_data);

  cl_int clGetProgramInfo(
    cl_program_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

  cl_int clRetainProgram();

  cl_int clReleaseProgram();

  cl_kernel clCreateKernel(
    const char* kernel_name,
    cl_int* errcode_ret);
};

struct _cl_kernel
  : public lifetime::icd_compatible
  , public lifetime::ref_counted_object<cl_kernel>
{
  _cl_kernel() = delete;
  _cl_kernel(const cl_program parent_program);
  _cl_kernel(const _cl_kernel&) = delete;
  _cl_kernel(_cl_kernel&&) = delete;
  ~_cl_kernel() = default;
  _cl_kernel &operator=(const _cl_kernel&) = delete;
  _cl_kernel &operator=(_cl_kernel&&) = delete;

  cl_int clGetKernelInfo(
    cl_kernel_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

  cl_int clSetKernelArg(
    cl_uint arg_index,
    size_t arg_size,
    const void* arg_value);

  cl_int clRetainKernel();

  cl_int clReleaseKernel();

  cl_kernel clCloneKernel(
    cl_int* errcode_ret);
};

struct _cl_event
  : public lifetime::icd_compatible
  , public lifetime::ref_counted_object<cl_event>
{
  _cl_event() = delete;
  _cl_event(const cl_context parent_context, const cl_command_queue parent_queue);
  _cl_event(const _cl_event&) = delete;
  _cl_event(_cl_event&&) = delete;
  ~_cl_event() = default;
  _cl_event &operator=(const _cl_event&) = delete;
  _cl_event &operator=(_cl_event&&) = delete;

  cl_int clGetEventInfo(
    cl_event_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

  cl_int clWaitForEvents(
    cl_uint num_events,
    const cl_event* event_list);

  cl_int clSetUserEventStatus(
    cl_int execution_status);

  cl_int clRetainEvent();

  cl_int clReleaseEvent();
};

struct _cl_sampler
  : public lifetime::icd_compatible
  , public lifetime::ref_counted_object<cl_sampler>
{
  _cl_sampler() = delete;
  _cl_sampler(const cl_context parent_context);
  _cl_sampler(const _cl_sampler&) = delete;
  _cl_sampler(_cl_sampler&&) = delete;
  ~_cl_sampler() = default;
  _cl_sampler &operator=(const _cl_sampler&) = delete;
  _cl_sampler &operator=(_cl_sampler&&) = delete;

  cl_int clGetSamplerInfo(
    cl_sampler_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret);

  cl_int clRetainSampler();

  cl_int clReleaseSampler();
};

struct _cl_platform_id
  : public lifetime::icd_compatible
{
  cl_version numeric_version;
  std::string profile,
              version,
              name,
              vendor,
              extensions,
              suffix;

  // Objects anchored here solely for the purpose of not being
  // subjected to static initialization order issues
  std::set<std::shared_ptr<_cl_device_id>> _devices;
  std::set<std::shared_ptr<_cl_context>> _contexts;
  std::set<std::shared_ptr<_cl_mem>> _mems;
  std::set<std::shared_ptr<_cl_command_queue>> _queues;
  std::set<std::shared_ptr<_cl_program>> _programs;
  std::set<std::shared_ptr<_cl_kernel>> _kernels;
  std::set<std::shared_ptr<_cl_event>> _events;
  std::set<std::shared_ptr<_cl_sampler>> _samplers;

  _cl_platform_id();
  _cl_platform_id(const _cl_platform_id&) = delete;
  _cl_platform_id(_cl_platform_id&&) = delete;
  ~_cl_platform_id() = default;
  _cl_platform_id &operator=(const _cl_platform_id&) = delete;
  _cl_platform_id &operator=(_cl_platform_id&&) = delete;

  void init_dispatch();

  cl_int clGetPlatformInfo(
    cl_platform_info  param_name,
    size_t            param_value_size,
    void *            param_value,
    size_t *          param_value_size_ret);

  cl_int clGetDeviceIDs(
    cl_device_type device_type,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices);
};

namespace lifetime
{
  extern std::map<std::string, void*> _extensions;
  extern _cl_platform_id _platform;
  extern cl_icd_dispatch _dispatch;

  template <typename T> std::set<std::shared_ptr<std::remove_pointer_t<T>>>& get_objects();
  template <> inline std::set<std::shared_ptr<_cl_device_id>>& get_objects<cl_device_id>() { return _platform._devices; }
  template <> inline std::set<std::shared_ptr<_cl_context>>& get_objects<cl_context>() { return _platform._contexts; }
  template <> inline std::set<std::shared_ptr<_cl_mem>>& get_objects<cl_mem>() { return _platform._mems; }
  template <> inline std::set<std::shared_ptr<_cl_command_queue>>& get_objects<cl_command_queue>() { return _platform._queues; }
  template <> inline std::set<std::shared_ptr<_cl_program>>& get_objects<cl_program>() { return _platform._programs; }
  template <> inline std::set<std::shared_ptr<_cl_kernel>>& get_objects<cl_kernel>() { return _platform._kernels; }
  template <> inline std::set<std::shared_ptr<_cl_event>>& get_objects<cl_event>() { return _platform._events; }
  template <> inline std::set<std::shared_ptr<_cl_sampler>>& get_objects<cl_sampler>() { return _platform._samplers; }
}
