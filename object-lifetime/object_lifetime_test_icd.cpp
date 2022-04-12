#include "object_lifetime_test_icd.hpp"

namespace lifetime
{
  _cl_platform_id _platform;
  std::map<std::string, void*> _extensions{
    std::make_pair("clIcdGetPlatformIDsKHR", reinterpret_cast<void*>(clIcdGetPlatformIDsKHR))
  };
}

#include <utils.hpp>

#include <string>
#include <algorithm>

CL_API_ENTRY void * CL_API_CALL
clGetExtensionFunctionAddress(const char *name)
{
  using namespace lifetime;

  auto it = _extensions.find(name);
  if (it != _extensions.end())
    return it->second;
  else
    return nullptr;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformInfo_wrap(
  cl_platform_id platform,
  cl_platform_info param_name,
  size_t param_value_size,
  void *param_value,
  size_t *param_value_size_ret)
{
  return platform->clGetPlatformInfo(
    param_name,
    param_value_size,
    param_value,
    param_value_size_ret
  );
}

_cl_platform_id::_cl_platform_id()
  : dispatch{ nullptr }
  , scoped_dispatch{ }
  , numeric_version{}
  , version{}
  , vendor{ "Khronos" }
  , profile{ "FULL_PROFILE" }
  , name{ "Object Lifetime Layer Test ICD" }
  , extensions{ "cl_khr_icd" }
  , suffix{ "khronos" }
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
  else
    numeric_version = CL_MAKE_VERSION(1, 2, 0);

  version = std::string{"OpenCL "} +
    std::to_string(CL_VERSION_MAJOR(numeric_version)) +
    "." +
    std::to_string(CL_VERSION_MINOR(numeric_version)) +
    " Mock";
}

void _cl_platform_id::init_dispatch()
{
  dispatch->clGetPlatformInfo = clGetPlatformInfo_wrap;
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

CL_API_ENTRY cl_int CL_API_CALL
clIcdGetPlatformIDsKHR(cl_uint         num_entries,
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
