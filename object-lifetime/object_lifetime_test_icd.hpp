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

    operator bool();

    void reference();
    void unreference();
  };

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
  std::string profile;
  std::string name;
  std::string vendor;
  std::string extensions;
  cl_uint cu_count;

  _cl_device_id() = default;
  _cl_device_id(device_kind kind, cl_uint num_cu = 64);
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

  bool report_implicit_ref_count_to_user,
       allow_using_released_objects,
       allow_using_inaccessible_objects;

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
  extern std::set<std::shared_ptr<_cl_device_id>> _devices;
  extern std::set<std::shared_ptr<_cl_context>> _contexts;

  template <typename T> cl_int CL_INVALID;
  template <> cl_int CL_INVALID<cl_platform_id> = CL_INVALID_PLATFORM;
  template <> cl_int CL_INVALID<cl_device_id> = CL_INVALID_DEVICE;
  template <> cl_int CL_INVALID<cl_context> = CL_INVALID_CONTEXT;

  template <typename T> T& _objects;
  template <> std::set<std::shared_ptr<_cl_device_id>>& _objects<cl_device_id> = _devices;
  template <> std::set<std::shared_ptr<_cl_context>>& _objects<cl_context> = _contexts;
}
