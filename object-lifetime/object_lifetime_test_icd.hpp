#pragma once

#if defined(_WIN32)
#define NOMINMAX
#endif

#include <CL/cl_icd.h>  // cl_icd_dispatch

#include <memory>       // std::unique_ptr
#include <string>       // std::string
#include <map>          // std::map<std::string, void*> _extensions
#include <utility>      // std::make_pair

struct _cl_platform_id {
  cl_icd_dispatch* dispatch;
  // NOTE: On all tested compiler/STL implementations std::unique_ptr
  //       is the same size as a raw pointer using the default deleter
  //       however this isn't guaranteed by the spec.
  std::unique_ptr<cl_icd_dispatch> scoped_dispatch;

  cl_version numeric_version;
  std::string profile;
  std::string version;
  std::string name;
  std::string vendor;
  std::string extensions;
  std::string suffix;

  _cl_platform_id();
  _cl_platform_id(const _cl_platform_id&) = delete;
  _cl_platform_id(_cl_platform_id&&) = delete;
  ~_cl_platform_id() = default;
  _cl_platform_id &operator=(const _cl_platform_id&) = delete;
  _cl_platform_id &operator=(_cl_platform_id &&) = delete;

  void init_dispatch(void);

  cl_int clGetPlatformInfo(
    cl_platform_info  param_name,
    size_t            param_value_size,
    void *            param_value,
    size_t *          param_value_size_ret);
};

namespace lifetime
{
  extern _cl_platform_id _platform;
  extern std::map<std::string, void*> _extensions;
}
