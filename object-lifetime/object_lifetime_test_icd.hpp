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

  template <typename Object> struct object_parents;

  template <> struct object_parents<cl_device_id>
  {
    cl_device_id parent;
    void notify();
  };

  template <> struct object_parents<cl_context>
  {
    cl_device_id parent;
    void notify();
  };

  template <> struct object_parents<cl_command_queue>
  {
    cl_device_id parent_device;
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

    bool is_valid(bool retain);

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
    if (ref_count > 0)
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
    std::unique_ptr<cl_icd_dispatch> scoped_dispatch;

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
};

struct _cl_context
  : public lifetime::icd_compatible
  , public lifetime::ref_counted_object<cl_context>
{
  _cl_context();
  _cl_context(const _cl_context&) = delete;
  _cl_context(_cl_context&&) = delete;
  ~_cl_context() = default;
  _cl_context &operator=(const _cl_context&) = delete;
  _cl_context &operator=(_cl_context&&) = delete;
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

  std::set<std::shared_ptr<_cl_device_id>> _devices;
  std::set<std::shared_ptr<_cl_context>> _contexts;

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

  template <typename T> std::set<std::shared_ptr<std::remove_pointer_t<T>>>& get_objects();
  template <> inline std::set<std::shared_ptr<_cl_device_id>>& get_objects<cl_device_id>() { return _platform._devices; }
  template <> inline std::set<std::shared_ptr<_cl_context>>& get_objects<cl_context>() { return _platform._contexts; }
}
