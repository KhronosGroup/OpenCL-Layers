bool object_is_valid(cl_platform_id platform) {
  size_t size;
  cl_int res = tdispatch->clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &size);
  return res == CL_SUCCESS;
}

bool object_is_valid(cl_device_id device) {
  cl_device_type type;
  cl_int res = tdispatch->clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, nullptr);
  return res == CL_SUCCESS;
}

bool object_is_valid(cl_context context) {
  cl_uint refcount;
  cl_int res = tdispatch->clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint), &refcount, nullptr);
  return res == CL_SUCCESS;
}

bool object_is_valid(cl_command_queue command_queue) {
  cl_uint refcount;
  cl_int res = tdispatch->clGetCommandQueueInfo(command_queue, CL_QUEUE_REFERENCE_COUNT, sizeof(cl_uint), &refcount, nullptr);
  return res == CL_SUCCESS;
}

bool object_is_valid(cl_mem mem) {
  cl_uint refcount;
  cl_int res = tdispatch->clGetMemObjectInfo(mem, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &refcount, nullptr);
  return res == CL_SUCCESS;
}

bool object_is_valid(cl_mem mem, cl_mem_object_type type) {
  cl_mem_object_type curr_type = 0;
  cl_int res = tdispatch->clGetMemObjectInfo(mem, CL_MEM_TYPE, sizeof(cl_mem_object_type), &curr_type, nullptr);
  return (res == CL_SUCCESS) && (curr_type == type);
}

bool object_is_valid(cl_sampler sampler) {
  cl_uint refcount;
  cl_int res = tdispatch->clGetSamplerInfo(sampler, CL_SAMPLER_REFERENCE_COUNT, sizeof(cl_uint), &refcount, nullptr);
  return res == CL_SUCCESS;
}

bool object_is_valid(cl_program program) {
  cl_uint refcount;
  cl_int res = tdispatch->clGetProgramInfo(program, CL_PROGRAM_REFERENCE_COUNT, sizeof(cl_uint), &refcount, nullptr);
  return res == CL_SUCCESS;
}

bool object_is_valid(cl_kernel kernel) {
  cl_uint refcount;
  cl_int res = tdispatch->clGetKernelInfo(kernel, CL_KERNEL_REFERENCE_COUNT, sizeof(cl_uint), &refcount, nullptr);
  return res == CL_SUCCESS;
}

bool object_is_valid(cl_event event) {
  cl_uint refcount;
  cl_int res = tdispatch->clGetEventInfo(event, CL_EVENT_REFERENCE_COUNT, sizeof(cl_uint), &refcount, nullptr);
  return res == CL_SUCCESS;
}

bool object_is_valid(cl_event event, cl_command_type type) {
  cl_command_type curr_type = 0;
  cl_int res = tdispatch->clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(cl_command_type), &curr_type, nullptr);
  return (res == CL_SUCCESS) && (curr_type == type);
}

template<cl_uint property>
return_type<property> query(cl_platform_id platform)
{
  cl_version version = get_object_version(platform);
  return_type<property> a;
  memset(&a, 0, sizeof(return_type<property>));
  if (!enum_violation(version, "cl_platform_info", property)) {
    tdispatch->clGetPlatformInfo(platform, property, sizeof(a), &a, NULL);
  } else {
    *layer::log_stream << "Invalid platform query in query(cl_platform_id). This is a bug in the param_verification layer." << std::endl;
    exit(-1);
  }
  return a;
}

template<cl_uint property>
return_type<property> query(cl_device_id device)
{
  cl_version version = get_object_version(device);
  return_type<property> a;
  memset(&a, 0, sizeof(return_type<property>));
  if (!enum_violation(version, "cl_device_info", property)) {
    tdispatch->clGetDeviceInfo(device, property, sizeof(a), &a, NULL);
  } else if (!enum_violation(version, "cl_platform_info", property)) {
    cl_platform_id p;
    tdispatch->clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &p, NULL);
    tdispatch->clGetPlatformInfo(p, property, sizeof(a), &a, NULL);
  } else {
    *layer::log_stream << "Invalid device query in query(cl_device_id). This is a bug in the param_verification layer." << std::endl;
    exit(-1);
  }
  return a;
}

template<cl_uint property>
return_type<property> query(cl_context context)
{
  cl_version version = get_object_version(context);
  return_type<property> a;
  memset(&a, 0, sizeof(return_type<property>));
  if (!enum_violation(version, "cl_context_info", property)) {
    tdispatch->clGetContextInfo(context, property, sizeof(a), &a, NULL);
  } else {
    *layer::log_stream << "Invalid context query in query(cl_context). This is a bug in the param_verification layer." << std::endl;
    exit(-1);
  }
  return a;
}

template<cl_uint property>
return_type<property> query(cl_command_queue queue)
{
  cl_version version = get_object_version(queue);
  return_type<property> a;
  memset(&a, 0, sizeof(return_type<property>));
  if (!enum_violation(version, "cl_command_queue_info", property)) {
    tdispatch->clGetCommandQueueInfo(queue, property, sizeof(a), &a, NULL);
  } else if (!enum_violation(version, "cl_device_info", property)) {
    cl_device_id d;
    tdispatch->clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(d), &d, NULL);
    tdispatch->clGetDeviceInfo(d, property, sizeof(a), &a, NULL);
  } else {
    *layer::log_stream << "Invalid command queue query in query(cl_command_queue). This is a bug in the param_verification layer." << std::endl;
    exit(-1);
  }
  return a;
}

template<cl_uint property>
return_type<property> query(cl_mem object)
{
  cl_version version = get_object_version(object);
  return_type<property> a;
  memset(&a, 0, sizeof(return_type<property>));
  if (!enum_violation(version, "cl_mem_info", property)) {
    tdispatch->clGetMemObjectInfo(object, property, sizeof(a), &a, NULL);
  } else if (!enum_violation(version, "cl_image_info", property)) {
    tdispatch->clGetImageInfo(object, property, sizeof(a), &a, NULL);
  } else if (!enum_violation(version, "cl_pipe_info", property)) {
    tdispatch->clGetPipeInfo(object, property, sizeof(a), &a, NULL);
  } else {
    *layer::log_stream << "Invalid mem object query in query(cl_mem). This is a bug in the param_verification layer." << std::endl;
    exit(-1);
  }
  return a;
}

template<cl_uint property>
return_type<property> query(cl_program program)
{
  cl_version version = get_object_version(program);
  return_type<property> a;
  memset(&a, 0, sizeof(return_type<property>));
  if (!enum_violation(version, "cl_program_info", property)) {
    tdispatch->clGetProgramInfo(program, property, sizeof(a), &a, NULL);
  } else {
    *layer::log_stream << "Invalid program query in query(cl_program). This is a bug in the param_verification layer." << std::endl;
    exit(-1);
  }
  return a;
}

template<cl_uint property>
return_type<property> query(cl_kernel kernel)
{
  cl_version version = get_object_version(kernel);
  return_type<property> a;
  memset(&a, 0, sizeof(return_type<property>));
  if (!enum_violation(version, "cl_kernel_info", property)) {
    tdispatch->clGetKernelInfo(kernel, property, sizeof(a), &a, NULL);
  } else {
    *layer::log_stream << "Invalid kernel query in query(cl_kernel). This is a bug in the param_verification layer." << std::endl;
    exit(-1);
  }
  return a;
}

template<cl_uint property>
return_type<property> query(cl_event event)
{
  cl_version version = get_object_version(event);
  return_type<property> a;
  memset(&a, 0, sizeof(return_type<property>));
  if (!enum_violation(version, "cl_event_info", property)) {
    tdispatch->clGetEventInfo(event, property, sizeof(a), &a, NULL);
  } else if (!enum_violation(version, "cl_command_queue_info", property)) {
    cl_command_queue q;
    tdispatch->clGetEventInfo(event, CL_EVENT_COMMAND_QUEUE, sizeof(q), &q, NULL);
    tdispatch->clGetCommandQueueInfo(q, property, sizeof(a), &a, NULL);
  } else {
    *layer::log_stream << "Invalid event query in query(cl_event). This is a bug in the param_verification layer." << std::endl;
    exit(-1);
  }
  return a;
}
