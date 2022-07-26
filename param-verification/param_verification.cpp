#include "param_verification.hpp"
#include <fstream>
#include <memory>

struct _cl_icd_dispatch dispatch = {};

const struct _cl_icd_dispatch *tdispatch;

namespace layer {
  ocl_layer_utils::stream_ptr log_stream;
  layer_settings settings;

  void init_output_stream() {
    switch(settings.log_type) {
    case layer_settings::DebugLogType::StdOut:
      log_stream.reset(&std::cout);
      break;
    case layer_settings::DebugLogType::StdErr:
      log_stream.reset(&std::cerr);
      break;
    case layer_settings::DebugLogType::File:
      log_stream.reset(new std::ofstream(settings.log_filename));
      if (log_stream->fail()) {
        log_stream.reset(&std::cerr);
        *log_stream << "param_verification failed to open specified output stream: "
                    << settings.log_filename << ". Falling back to stderr." << '\n';
      }
      break;
    }
  }

  layer_settings layer_settings::load() {
    const auto settings_from_file = ocl_layer_utils::load_settings();
    const auto parser =
      ocl_layer_utils::settings_parser("param_verification", settings_from_file);

    auto result = layer_settings{};
    const auto debug_log_values =
      std::map<std::string, DebugLogType>{{"stdout", DebugLogType::StdOut},
                                          {"stderr", DebugLogType::StdErr},
                                          {"file", DebugLogType::File}};
    parser.get_enumeration("log_sink", debug_log_values, result.log_type);
    parser.get_filename("log_filename", result.log_filename);
    parser.get_bool("transparent", result.transparent);

    return result;
  }
}

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

CL_API_ENTRY cl_int CL_API_CALL
clInitLayer(
    cl_uint                         num_entries,
    const struct _cl_icd_dispatch  *target_dispatch,
    cl_uint                        *num_entries_out,
    const struct _cl_icd_dispatch **layer_dispatch_ret) {
  if (!target_dispatch || !layer_dispatch_ret || !num_entries_out || num_entries < sizeof(dispatch) / sizeof(dispatch.clGetPlatformIDs))
    return CL_INVALID_VALUE;

  layer::settings = layer::layer_settings::load();
  layer::init_output_stream();

  tdispatch = target_dispatch;
  init_dispatch();

  *layer_dispatch_ret = &dispatch;
  *num_entries_out = sizeof(dispatch)/sizeof(dispatch.clGetPlatformIDs);
  return CL_SUCCESS;
}

cl_version get_object_version(cl_platform_id platform) {
  if (!platform)
    return layer::FALLBACK_VERSION;
  size_t version_len;
  cl_int res;
  res = tdispatch->clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, nullptr, &version_len);
  if (res != CL_SUCCESS)
    return layer::FALLBACK_VERSION;

  auto version_str = std::make_unique<char[]>(version_len);
  res = tdispatch->clGetPlatformInfo(platform, CL_PLATFORM_VERSION, version_len, version_str.get(), nullptr);
  if (res != CL_SUCCESS)
    return layer::FALLBACK_VERSION;

  cl_version version;
  if (!ocl_layer_utils::parse_cl_version_string(version_str.get(), &version)) {
    return layer::FALLBACK_VERSION;
  }
  return version;
}

cl_version get_object_version(cl_device_id device) {
  if (!device)
    return layer::FALLBACK_VERSION;
  cl_platform_id platform;
  cl_int res = tdispatch->clGetDeviceInfo(
    device,
    CL_DEVICE_PLATFORM,
    sizeof(cl_platform_id),
    &platform,
    nullptr);
  if (res != CL_SUCCESS) {
    return layer::FALLBACK_VERSION;
  }

  return get_object_version(platform);
}

cl_version get_object_version(cl_context context) {
  if (!context)
    return layer::FALLBACK_VERSION;
  // Note: need to query all devices, even if we only need one.
  size_t devices_size;
  cl_int res = tdispatch->clGetContextInfo(
    context,
    CL_CONTEXT_DEVICES,
    0,
    NULL,
    &devices_size);
  if (res != CL_SUCCESS || devices_size == 0) {
    return layer::FALLBACK_VERSION;
  }

  auto devices = std::make_unique<cl_device_id[]>(devices_size / sizeof(cl_device_id));
  res = tdispatch->clGetContextInfo(
    context,
    CL_CONTEXT_DEVICES,
    devices_size,
    (void*)devices.get(),
    NULL);
  if (res != CL_SUCCESS) {
    return layer::FALLBACK_VERSION;
  }

  return get_object_version(devices[0]); // platform should be the same for all devices in the context
}

cl_version get_object_version(cl_command_queue queue) {
  if (!queue)
    return layer::FALLBACK_VERSION;
  cl_context context;
  cl_int res = tdispatch->clGetCommandQueueInfo(
    queue,
    CL_QUEUE_CONTEXT,
    sizeof(cl_context),
    &context,
    nullptr);
  if (res != CL_SUCCESS) {
    return layer::FALLBACK_VERSION;
  }

  return get_object_version(context);
}

cl_version get_object_version(cl_mem mem) {
  if (!mem)
    return layer::FALLBACK_VERSION;
  cl_context context;
  cl_int res = tdispatch->clGetMemObjectInfo(
    mem,
    CL_MEM_CONTEXT,
    sizeof(cl_context),
    &context,
    nullptr);
  if (res != CL_SUCCESS) {
    return layer::FALLBACK_VERSION;
  }

  return get_object_version(context);
}

cl_version get_object_version(cl_sampler sampler) {
  if (!sampler)
    return layer::FALLBACK_VERSION;
  cl_context context;
  cl_int res = tdispatch->clGetSamplerInfo(
    sampler,
    CL_SAMPLER_CONTEXT,
    sizeof(cl_context),
    &context,
    nullptr);
  if (res != CL_SUCCESS) {
    return layer::FALLBACK_VERSION;
  }

  return get_object_version(context);
}

cl_version get_object_version(cl_program program) {
  if (!program)
    return layer::FALLBACK_VERSION;
  cl_context context;
  cl_int res = tdispatch->clGetProgramInfo(
    program,
    CL_PROGRAM_CONTEXT,
    sizeof(cl_context),
    &context,
    nullptr);
  if (res != CL_SUCCESS) {
    return layer::FALLBACK_VERSION;
  }

  return get_object_version(context);
}

cl_version get_object_version(cl_kernel kernel) {
  if (!kernel)
    return layer::FALLBACK_VERSION;
  cl_context context;
  cl_int res = tdispatch->clGetKernelInfo(
    kernel,
    CL_KERNEL_CONTEXT,
    sizeof(cl_context),
    &context,
    nullptr);
  if (res != CL_SUCCESS) {
    return layer::FALLBACK_VERSION;
  }

  return get_object_version(context);
}

cl_version get_object_version(cl_event event) {
  if (!event)
    return layer::FALLBACK_VERSION;
  cl_context context;
  cl_int res = tdispatch->clGetEventInfo(
    event,
    CL_EVENT_CONTEXT,
    sizeof(cl_context),
    &context,
    nullptr);
  if (res != CL_SUCCESS) {
    return layer::FALLBACK_VERSION;
  }

  return get_object_version(context);
}

cl_platform_id get_context_properties_platform(const cl_context_properties * properties) {
  // Keep logic in sync with khrIcdContextPropertiesGetPlatform from OpenCL-ICD-Loader.
  cl_platform_id platform = nullptr;
  for (const cl_context_properties *property = properties; property && property[0]; property += 2) {
    if ((cl_context_properties) CL_CONTEXT_PLATFORM == property[0]) {
      platform = (cl_platform_id) property[1];
    }
  }
  if (!platform) {
    // Fetch the first platform reported by clGetPlatformIDs.
    cl_uint num_platforms;
    cl_int status = tdispatch->clGetPlatformIDs(
      1,
      &platform,
      &num_platforms);
    if (status != CL_SUCCESS || num_platforms == 0) {
      platform = nullptr;
    }
  }
  return platform;
}
