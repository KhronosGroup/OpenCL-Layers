#include "object_lifetime_test_icd.hpp"
#include "object_lifetime_test_icd_surface.hpp"

namespace lifetime
{
  void object_parents<cl_device_id>::notify()
  {
    parent->unreference();
  }

  void object_parents<cl_context>::notify()
  {
    parent->unreference();
  }

  void object_parents<cl_command_queue>::notify()
  {
    parent_device->unreference();
    parent_context->unreference();
  }

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
      return CL_INVALID<Object>;
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
      return CL_INVALID<Object>;
  }

  template <typename Object>
  cl_uint ref_counted_object<Object>::CL_OBJECT_REFERENCE_COUNT()
  {
    if (_platform.report_implicit_ref_count_to_user)
      return ref_count + implicit_ref_count;
    else
      return ref_count;
  }

  template <typename Object>
  ref_counted_object<Object>::operator bool()
  {
    if (ref_count > 0)
      return true;
    else if (implicit_ref_count > 0 && _platform.allow_using_inaccessible_objects)
      return true;
    else if (_platform.allow_using_released_objects)
      return true;
    else
      return false;
  }

  template <typename Object>
  void ref_counted_object<Object>::reference()
  {
    ++implicit_ref_count;
  }

  template <typename Object>
  void ref_counted_object<Object>::unreference()
  {
    --implicit_ref_count;
  }

  icd_compatible::icd_compatible()
    : scoped_dispatch{ std::make_unique<cl_icd_dispatch>() }
    , dispatch{ scoped_dispatch.get() }
  {}
}

#include <utils.hpp>

#include <string>
#include <algorithm>
#include <iterator>

_cl_device_id::_cl_device_id(device_kind kind, cl_uint num_cu)
  : icd_compatible{}
  , ref_counted_object{ lifetime::object_parents<cl_device_id>{
    kind == device_kind::root ? nullptr : this} }
  , kind{ kind }
  , profile{ "FULL_PROFILE" }
  , name{ "Test Device" }
  , vendor{ "The Khronos Group" }
  , extensions{ "" }
  , cu_count{ num_cu }
{
  init_dispatch();
}

void _cl_device_id::init_dispatch()
{
  dispatch->clGetDeviceInfo = clGetDeviceInfo_wrap;
  dispatch->clCreateSubDevices = clCreateSubDevices_wrap;
  dispatch->clRetainDevice = clRetainDevice_wrap;
  dispatch->clReleaseDevice = clReleaseDevice_wrap;
}

cl_int _cl_device_id::clGetDeviceInfo(
    cl_device_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  if (param_value_size == 0 && param_value != NULL)
    return CL_INVALID_VALUE;

  std::vector<char> result;
  switch(param_name)
  {
    case CL_DEVICE_TYPE:
    {
      auto dev_type = CL_DEVICE_TYPE_CUSTOM;
      std::copy(
        reinterpret_cast<char*>(&dev_type),
        reinterpret_cast<char*>(&dev_type) + sizeof(dev_type),
        std::back_inserter(result));
      break;
    }
    case CL_DEVICE_NAME:
      std::copy(name.begin(), name.end(), std::back_inserter(result));
      result.push_back('\0');
      break;
    case CL_DEVICE_VENDOR:
      std::copy(vendor.begin(), vendor.end(), std::back_inserter(result));
      result.push_back('\0');
      break;
    case CL_DEVICE_PROFILE:
      std::copy(profile.begin(), profile.end(), std::back_inserter(result));
      result.push_back('\0');
      break;
    case CL_DEVICE_MAX_COMPUTE_UNITS:
      std::copy(
        reinterpret_cast<char*>(&cu_count),
        reinterpret_cast<char*>(&cu_count) + sizeof(cu_count),
        std::back_inserter(result));
      break;
    case CL_DEVICE_EXTENSIONS:
      std::copy(extensions.begin(), extensions.end(), std::back_inserter(result));
      result.push_back('\0');
      break;
    case CL_DEVICE_PARTITION_PROPERTIES:
    {
      cl_device_partition_property partition_types = CL_DEVICE_PARTITION_EQUALLY;
      std::copy(
        reinterpret_cast<char*>(&partition_types),
        reinterpret_cast<char*>(&partition_types) + sizeof(partition_types),
        std::back_inserter(result));
      break;
    }
    case CL_DEVICE_REFERENCE_COUNT:
    {
      cl_uint tmp = CL_OBJECT_REFERENCE_COUNT();
      std::copy(
        reinterpret_cast<char*>(&tmp),
        reinterpret_cast<char*>(&tmp) + sizeof(tmp),
        std::back_inserter(result));
      break;
    }
    default:
      return CL_INVALID_VALUE;
  }

  if (param_value_size_ret)
    *param_value_size_ret = result.size();

  if (param_value_size && param_value_size < result.size())
    return CL_INVALID_VALUE;

  if (param_value)
  {
    std::copy(result.begin(), result.end(), static_cast<char*>(param_value));
  }

  return CL_SUCCESS;
}

cl_int _cl_device_id::clCreateSubDevices(
  const cl_device_partition_property* properties,
  cl_uint num_devices,
  cl_device_id* out_devices,
  cl_uint* num_devices_ret)
{
  if (!properties)
    return CL_INVALID_VALUE;

  std::vector<cl_device_partition_property> props;
  const cl_device_partition_property* it = properties;
  while(*it != 0)
    props.push_back(*it++);

  switch(props.front())
  {
    case CL_DEVICE_PARTITION_EQUALLY:
    {
      cl_uint n = static_cast<cl_uint>(properties[1]);
      if (num_devices * n > cu_count)
        return CL_INVALID_DEVICE_PARTITION_COUNT;

      std::vector<std::shared_ptr<_cl_device_id>> result(
        num_devices,
        std::make_shared<_cl_device_id>(
          _cl_device_id::device_kind::sub,
          n
        )
      );
      implicit_ref_count += num_devices;

      if (num_devices_ret)
        *num_devices_ret = static_cast<cl_uint>(result.size());

      if (out_devices && num_devices < result.size())
        return CL_INVALID_VALUE;

      if (out_devices)
      {
        std::transform(
          result.cbegin(),
          result.cbegin(),
          out_devices,
          [](const std::shared_ptr<_cl_device_id>& dev){ return dev.get(); }
        );
      }

      break;
    }
    default:
      return CL_INVALID_VALUE;
  }
}

cl_int _cl_device_id::clRetainDevice()
{
  if(kind == device_kind::root)
    return CL_SUCCESS;
  else
    return retain();
}

cl_int _cl_device_id::clReleaseDevice()
{
  if(kind == device_kind::root)
    return CL_SUCCESS;
  else
    return release();
}

_cl_platform_id::_cl_platform_id()
  : numeric_version{ CL_MAKE_VERSION(3, 0, 0) }
  , version{}
  , vendor{ "Khronos" }
  , profile{ "FULL_PROFILE" }
  , name{ "Object Lifetime Layer Test ICD" }
  , extensions{ "cl_khr_icd" }
  , suffix{ "khronos" }
  , report_implicit_ref_count_to_user{ false }
  , allow_using_released_objects{ false }
  , allow_using_inaccessible_objects{ false }
{
  scoped_dispatch = std::make_unique<cl_icd_dispatch>();
  dispatch = scoped_dispatch.get();
  init_dispatch();

  std::string ICD_VERSION;
  if (ocl_layer_utils::detail::get_environment("OBJECT_LIFETIME_ICD_VERSION", ICD_VERSION))
  {
    auto icd_version = std::atoi(ICD_VERSION.c_str());
    numeric_version = CL_MAKE_VERSION(
      icd_version / 100,
      icd_version % 100 / 10,
      icd_version % 10
    );
  }

  version = std::string{"OpenCL "} +
    std::to_string(CL_VERSION_MAJOR(numeric_version)) +
    "." +
    std::to_string(CL_VERSION_MINOR(numeric_version)) +
    " Mock";

  std::string REPORT_IMPLICIT_REF_COUNT_TO_USER;
  if (ocl_layer_utils::detail::get_environment("REPORT_IMPLICIT_REF_COUNT_TO_USER", REPORT_IMPLICIT_REF_COUNT_TO_USER))
  {
    if (CL_VERSION_MAJOR(numeric_version) == 1)
      std::exit(-1); // conflating implicit ref counts with regular ones is 2.0+ behavior
    report_implicit_ref_count_to_user = std::atoi(REPORT_IMPLICIT_REF_COUNT_TO_USER.c_str());
  }

  std::string ALLOW_USING_RELEASED_OBJECTS;
  if (ocl_layer_utils::detail::get_environment("ALLOW_USING_RELEASED_OBJECTS", ALLOW_USING_RELEASED_OBJECTS))
  {
    allow_using_released_objects = std::atoi(ALLOW_USING_RELEASED_OBJECTS.c_str());
  }

  std::string ALLOW_USING_INACCESSIBLE_OBJECTS;
  if (ocl_layer_utils::detail::get_environment("ALLOW_USING_INACCESSIBLE_OBJECTS", ALLOW_USING_INACCESSIBLE_OBJECTS))
  {
    if (CL_VERSION_MAJOR(numeric_version) != 3)
      std::exit(-1); // the definition of inaccessible objects only exists in 3.0+
    allow_using_inaccessible_objects = std::atoi(ALLOW_USING_INACCESSIBLE_OBJECTS.c_str());
  }

  lifetime::_devices.insert(std::make_shared<_cl_device_id>(_cl_device_id::device_kind::root));
}

void _cl_platform_id::init_dispatch()
{
  dispatch->clGetPlatformInfo = clGetPlatformInfo_wrap;
  dispatch->clGetDeviceIDs = clGetDeviceIDs_wrap;
}

cl_int _cl_platform_id::clGetPlatformInfo(
    cl_platform_info  param_name,
    size_t            param_value_size,
    void *            param_value,
    size_t *          param_value_size_ret)
{
  if (param_value_size == 0 && param_value != NULL)
    return CL_INVALID_VALUE;

  std::string result;
  switch(param_name) {
    case CL_PLATFORM_PROFILE:
      result = profile;
      break;
    case CL_PLATFORM_VERSION:
      result = version;
      break;
    case CL_PLATFORM_NAME:
      result = name;
      break;
    case CL_PLATFORM_VENDOR:
      result = vendor;
      break;
    case CL_PLATFORM_EXTENSIONS:
      result = extensions;
      break;
    case CL_PLATFORM_ICD_SUFFIX_KHR:
      result = suffix;
      break;
    default:
      return CL_INVALID_VALUE;
  }

  if (param_value_size_ret)
    *param_value_size_ret = result.length()+1;

  if (param_value_size && param_value_size < result.length()+1)
    return CL_INVALID_VALUE;

  if (param_value)
  {
    std::copy(result.begin(), result.end(), static_cast<char*>(param_value));
    static_cast<char*>(param_value)[result.length()] = '\0';
  }

  return CL_SUCCESS;
}

cl_int _cl_platform_id::clGetDeviceIDs(
    cl_device_type device_type,
    cl_uint num_entries,
    cl_device_id* devices,
    cl_uint* num_devices)
{
  using namespace lifetime;

  if (num_entries == 0 && devices != NULL)
    return CL_INVALID_VALUE;

  const bool asking_for_custom =
    device_type == CL_DEVICE_TYPE_CUSTOM ||
    device_type == CL_DEVICE_TYPE_DEFAULT ||
    device_type == CL_DEVICE_TYPE_ALL;

  if (num_devices)
    *num_devices = asking_for_custom ?
      static_cast<cl_uint>(std::count_if(
        _devices.cbegin(),
        _devices.cend(),
        [](const std::shared_ptr<_cl_device_id>& dev)
        {
          return dev->kind == _cl_device_id::device_kind::root;
        })) :
       0;

  if(devices && asking_for_custom)
  {
    std::vector<std::shared_ptr<_cl_device_id>> result;
    std::copy_if(
      _devices.cbegin(),
      _devices.cend(),
      std::back_inserter(result),
      [](const std::shared_ptr<_cl_device_id>& dev)
      {
        return dev->kind == _cl_device_id::device_kind::root;
      }
    );

    std::transform(
      result.cbegin(),
      result.cbegin() + std::min<size_t>(num_entries, result.size()),
      devices,
      [](const std::shared_ptr<_cl_device_id>& dev){ return dev.get(); }
    );
  }

  return CL_SUCCESS;
}

namespace lifetime
{
  std::map<std::string, void*> _extensions{
    std::make_pair("clIcdGetPlatformIDsKHR", reinterpret_cast<void*>(clIcdGetPlatformIDsKHR))
  };
  _cl_platform_id _platform;
  std::set<std::shared_ptr<_cl_device_id>> _devices;
  std::set<std::shared_ptr<_cl_context>> _contexts;
}
