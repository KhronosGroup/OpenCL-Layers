#pragma once

#include <CL/cl_layer.h>
#include "utils.hpp"
#include <vector>

namespace layer {
  constexpr const static cl_version FALLBACK_VERSION = CL_MAKE_VERSION(3, 0, 0);

  struct layer_settings {
    enum class DebugLogType { StdOut, StdErr, File };

    static layer_settings load();

    DebugLogType log_type = DebugLogType::StdErr;
    std::string log_filename;
    bool transparent = false;
  };

  extern layer_settings settings;
  extern ocl_layer_utils::stream_ptr log_stream;
}

// auxilary functions
std::vector<cl_device_id> get_devices(cl_kernel kernel);
std::vector<cl_device_id> get_devices(cl_context context);
std::vector<cl_device_id> get_devices(cl_program program);
size_t pixel_size(const cl_image_format * image_format);

void init_dispatch();

extern struct _cl_icd_dispatch dispatch;

extern const struct _cl_icd_dispatch *tdispatch;

cl_version get_object_version(cl_platform_id platform);
cl_version get_object_version(cl_device_id device);
cl_version get_object_version(cl_context context);
cl_version get_object_version(cl_command_queue queue);
cl_version get_object_version(cl_mem mem);
cl_version get_object_version(cl_sampler sampler);
cl_version get_object_version(cl_program program);
cl_version get_object_version(cl_kernel kernel);
cl_version get_object_version(cl_event event);

cl_platform_id get_context_properties_platform(const cl_context_properties * properties);
