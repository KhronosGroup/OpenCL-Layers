// version that checks number of subdevices for clCreateSubDevices
template<typename T>
bool list_violation(
  cl_version version,
  const char * name,
  T param,
  cl_device_id device,
  cl_uint num_devices)
{
  if (strcmp(name, "cl_device_partition_property") == 0)
  { // clCreateSubDevices
    cl_uint cu;
    cl_uint sd;

    // only single partition scheme is allowed
    size_t pos = 0;
    cl_uint curr_cu = 0;
    cl_uint curr_sd = 0;

    switch (param[0]) {
      case CL_DEVICE_PARTITION_EQUALLY:
        tdispatch->clGetDeviceInfo(device,
          CL_DEVICE_MAX_COMPUTE_UNITS,
          sizeof(cl_uint),
          &cu,
          NULL);

        if ((param[1] <= 0) || (static_cast<cl_uint>(param[1]) > cu) || (param[2] != 0))
          return true;
        if (cu / param[1] > num_devices)
          return true;
        return false;

      case CL_DEVICE_PARTITION_BY_COUNTS:
        tdispatch->clGetDeviceInfo(device,
          CL_DEVICE_MAX_COMPUTE_UNITS,
          sizeof(cl_uint),
          &cu,
          NULL);

        tdispatch->clGetDeviceInfo(device,
          CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
          sizeof(cl_uint),
          &sd,
          NULL);

        ++pos;
        while ((param[pos] != 0) && (param[pos] != CL_DEVICE_PARTITION_BY_COUNTS_LIST_END))
        {
          curr_cu += (cl_uint)param[pos];
          curr_sd++;
          if ((param[pos] < 0) || (curr_cu > cu) || (curr_sd > sd))
            return true;
          ++pos;
        }

        ++pos;
        if (param[pos] != 0)
          return true;
        if (curr_sd > num_devices)
          return true;
        return false;

      case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
        if (bitfield_violation(version, "cl_device_affinity_domain", param[1]) || 
            (param[1] == 0) || (param[2] != 0))
          return true;
        return false;

      default:
        return true;
    }
  }

  *layer::log_stream << "Wrong list:" << name << ", expected cl_device_partition_property."
    << std::endl << "This is a bug in the param_verification layer." << std::endl;
  return true;
}

// version that checks device limits for clCreateSubDevices
// and max size of device queue and support for for clCreateCommandQueueWithProperties
template<typename T>
bool list_violation(
  cl_version version,
  const char * name,
  T param,
  cl_device_id device)
{
  if (strcmp(name, "cl_device_partition_property") == 0)
  { // clCreateSubDevices
    cl_uint cu;
    cl_uint sd;

    // only single partition scheme is allowed
    size_t pos = 0;
    cl_uint curr_cu = 0;
    cl_uint curr_sd = 0;

    switch (param[0]) {
      case CL_DEVICE_PARTITION_EQUALLY:
        tdispatch->clGetDeviceInfo(device,
          CL_DEVICE_MAX_COMPUTE_UNITS,
          sizeof(cl_uint),
          &cu,
          NULL);

        if ((param[1] <= 0) || (static_cast<cl_uint>(param[1]) > cu) || (param[2] != 0))
          return true;
        return false;

      case CL_DEVICE_PARTITION_BY_COUNTS:
        tdispatch->clGetDeviceInfo(device,
          CL_DEVICE_MAX_COMPUTE_UNITS,
          sizeof(cl_uint),
          &cu,
          NULL);

        tdispatch->clGetDeviceInfo(device,
          CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
          sizeof(cl_uint),
          &sd,
          NULL);

        ++pos;
        while ((param[pos] != 0) && (param[pos] != CL_DEVICE_PARTITION_BY_COUNTS_LIST_END))
        {
          curr_cu += (cl_uint)param[pos];
          curr_sd++;
          if ((param[pos] < 0) || (curr_cu > cu) || (curr_sd > sd) || (curr_sd > cu))
            return true;
          ++pos;
        }

        ++pos;
        if (param[pos] != 0)
          return true;
        return false;

      case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
        if (bitfield_violation(version, "cl_device_affinity_domain", param[1]) || 
            (param[1] == 0) || (param[2] != 0))
          return true;
        return false;

      default:
        return true;
    }
  }

  if (strcmp(name, "cl_queue_properties") == 0)
  { // clCreateCommandQueueWithProperties - min 2.0
    if (param == NULL)
      return false;

    // any order of properties is allowed
    size_t pos = 0;
    // and not once ???
    cl_uint qs = 0;
    tdispatch->clGetDeviceInfo(device,
      CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE,
      sizeof(cl_uint),
      &qs,
      NULL);
    cl_uint curr_qs = 0;
    cl_device_device_enqueue_capabilities ddec = 0;
    if (version >= CL_MAKE_VERSION(3, 0, 0))
      tdispatch->clGetDeviceInfo(device,
        CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES,
        sizeof(cl_device_device_enqueue_capabilities),
        &ddec,
        NULL);
    cl_command_queue_properties qp = 0;

    while (param[pos] != 0)
    {
      switch (param[pos]) {
        case CL_QUEUE_PROPERTIES:
          ++pos;
          qp = param[pos];
          ++pos;
          if (bitfield_violation(version, "cl_command_queue_properties", qp))
            return true;
          if ((qp & CL_QUEUE_ON_DEVICE) && !(qp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE))
            return true;
          if ((qp & CL_QUEUE_ON_DEVICE_DEFAULT) && !(qp & CL_QUEUE_ON_DEVICE))
            return true;
          if ((version >= CL_MAKE_VERSION(3, 0, 0)) && 
              (qp & CL_QUEUE_ON_DEVICE) && !(ddec & CL_DEVICE_QUEUE_SUPPORTED))
            return true;
          break;

        case CL_QUEUE_SIZE:
          ++pos;
          curr_qs = (cl_uint)param[pos];
          ++pos;
          if (curr_qs > qs)
            return true;
          break;

        default:
          return true;
      }
    }

    if ((curr_qs > 0) && !(qp & CL_QUEUE_ON_DEVICE))
      return true;
    return false;
  }

  *layer::log_stream << "Wrong list:" << name << ", expected cl_device_partition_property or cl_queue_properties."
    << std::endl << "This is a bug in the param_verification layer." << std::endl;
   return true;
}

// version that checks platform for clCreateContext and clCreateContextFromType
template<typename T>
bool list_violation(
  cl_version version,
  const char * name,
  T param,
  void * user_data)
{
  (void)version;
  // dummy param to separate the case
  (void)user_data;

  if (strcmp(name, "cl_context_properties") == 0)
  { // clCreateContext
    if (param == NULL)
      return false;

    // any order of properties is allowed
    size_t pos = 0;
    // but only once
    cl_uint cp_num = 0;
    cl_uint cius_num = 0;

    while (param[pos] != 0)
    {
      switch (param[pos]) {
        case CL_CONTEXT_PLATFORM:
          if (!object_is_valid((cl_platform_id)param[pos+1]))
            return true;
          pos += 2;
          ++cp_num;
          if (cp_num > 1)
            return true;
          break;

        case CL_CONTEXT_INTEROP_USER_SYNC:
          pos += 2;
          ++cius_num;
          if (cius_num > 1)
            return true;
          break;

        default:
          return true;
      }
    }

    return false;
  }

  *layer::log_stream << "Wrong list:" << name << ", expected cl_context_properties."
    << std::endl << "This is a bug in the param_verification layer." << std::endl;
  return true;
}

// base version
template<typename T>
bool list_violation(cl_version version, const char * name, T param)
{
  if (strcmp(name, "cl_device_partition_property") == 0)
  { // clCreateSubDevices

    // only single partition scheme is allowed
    size_t pos = 0;

    switch (param[0]) {
      case CL_DEVICE_PARTITION_EQUALLY:
        if ((param[1] == 0) || (param[2] != 0))
          return true;
        return false;

      case CL_DEVICE_PARTITION_BY_COUNTS:
        ++pos;
        while ((param[pos] != 0) && (param[pos] != CL_DEVICE_PARTITION_BY_COUNTS_LIST_END))
        {
          ++pos;
        }

        ++pos;
        if (param[pos] != 0)
          return true;
        return false;

      case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
        if (bitfield_violation(version, "cl_device_affinity_domain", param[1]) || 
            (param[1] == 0) || (param[2] != 0))
          return true;
        return false;

      default:
        return true;
    }
  }

  if (strcmp(name, "cl_context_properties") == 0)
  { // clCreateContext
    if (param == NULL)
      return false;

    // any order of properties is allowed
    size_t pos = 0;
    // but only once
    cl_uint cp_num = 0;
    cl_uint cius_num = 0;

    while (param[pos] != 0)
    {
      switch (param[pos]) {
        case CL_CONTEXT_PLATFORM:
          pos += 2;
          ++cp_num;
          if (cp_num > 1)
            return true;
          break;

        case CL_CONTEXT_INTEROP_USER_SYNC:
          pos += 2;
          ++cius_num;
          if (cius_num > 1)
            return true;
          break;

        default:
          return true;
      }
    }

    return false;
  }

  if (strcmp(name, "cl_queue_properties") == 0)
  { // clCreateCommandQueueWithProperties - min 2.0
    if (param == NULL)
      return false;

    // any order of properties is allowed
    size_t pos = 0;
    // and not once ???
    cl_ulong curr_qs = 0;
    cl_command_queue_properties qp = 0;

    while (param[pos] != 0)
    {
      switch (param[pos]) {
        case CL_QUEUE_PROPERTIES:
          ++pos;
          qp = param[pos];
          ++pos;
          if (bitfield_violation(version, "cl_command_queue_properties", qp))
            return true;
          if ((qp & CL_QUEUE_ON_DEVICE) && !(qp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE))
            return true;
          if ((qp & CL_QUEUE_ON_DEVICE_DEFAULT) && !(qp & CL_QUEUE_ON_DEVICE))
            return true;
          break;

        case CL_QUEUE_SIZE:
          ++pos;
          curr_qs = param[pos];
          ++pos;
          break;

        default:
          return true;
      }
    }

    if ((curr_qs > 0) && !(qp & CL_QUEUE_ON_DEVICE))
      return true;
    return false;
  }

  if (strcmp(name, "cl_mem_properties") == 0)
  { // clCreateBufferWithProperties
    if (param == NULL)
      return false;
    // no properties yet
    if (param[0] == 0)
      return false;
    return true;
  }

  if (strcmp(name, "cl_sampler_properties") == 0)
  { // clCreateBufferWithProperties
    if (param == NULL)
      return false;

    // any order of properties is allowed
    size_t pos = 0;
    // but only once
    cl_uint snc_num = 0;
    cl_uint sam_num = 0;
    cl_uint sfm_num = 0;

    while (param[pos] != 0)
    {
      switch (param[pos]) {
        case CL_SAMPLER_NORMALIZED_COORDS:
          pos += 2;
          ++snc_num;
          if (snc_num > 1)
            return true;
          break;

        case CL_SAMPLER_ADDRESSING_MODE:
          ++pos;
          ++sam_num;
          if (sam_num > 1)
            return true;

          if (enum_violation(version, "cl_addressing_mode", param[pos]))
            return true;

          ++pos;
          break;

        case CL_SAMPLER_FILTER_MODE:
          ++pos;
          ++sfm_num;
          if (sfm_num > 1)
            return true;

          if (enum_violation(version, "cl_filter_mode", param[pos]))
            return true;

          ++pos;
          break;

        default:
          return true;
      }
    }

    return false;
  }

  *layer::log_stream << "Bad list:" << name << "."
    << std::endl << "This is a bug in the param_verification layer." << std::endl;
  return true;
}

std::vector<cl_device_id> get_devices(cl_context context)
{
  // suppose minimum OpenCL 1.1
  size_t nd = 0;
  tdispatch->clGetContextInfo(
    context,
    CL_CONTEXT_NUM_DEVICES,
    sizeof(nd),
    &nd,
    NULL);

  std::vector<cl_device_id> devices(nd);
  tdispatch->clGetContextInfo(
    context,
    CL_CONTEXT_DEVICES,
    nd * sizeof(cl_device_id),
    devices.data(),
    NULL);

  return devices;
}

std::vector<cl_device_id> get_devices(cl_program program)
{
  size_t nd = 0;
  tdispatch->clGetProgramInfo(
    program,
    CL_PROGRAM_NUM_DEVICES,
    sizeof(nd),
    &nd,
    NULL);

  std::vector<cl_device_id> devices(nd);
  tdispatch->clGetProgramInfo(
    program,
    CL_PROGRAM_DEVICES,
    nd * sizeof(cl_device_id),
    devices.data(),
    nullptr);

  return devices;
}

std::vector<cl_device_id> get_devices(cl_kernel kernel)
{
  cl_program pr;
  tdispatch->clGetKernelInfo(
    kernel,
    CL_KERNEL_PROGRAM,
    sizeof(pr),
    &pr,
    NULL);
  size_t nd;
  tdispatch->clGetProgramInfo(
    pr,
    CL_PROGRAM_NUM_DEVICES,
    sizeof(nd),
    &nd,
    NULL);
  std::vector<cl_device_id> devices(nd);
  tdispatch->clGetProgramInfo(
    pr,
    CL_PROGRAM_DEVICES,
    nd * sizeof(cl_device_id),
    devices.data(),
    NULL);
  // remove all devices for which the program is not built
  devices.erase(
    std::remove_if(
      devices.begin(),
      devices.end(),
      [pr](cl_device_id d) {
        cl_build_status bs;
        tdispatch->clGetProgramBuildInfo(
          pr,
          d,
          CL_PROGRAM_BUILD_STATUS,
          sizeof(bs),
          &bs,
          NULL);
        return (bs != CL_BUILD_SUCCESS); }
      ),
    devices.end());

  return devices;
}

//template<typename T1, typename T2>
//bool object_not_in(T1 object, T2 in);

// device should belong to context
bool object_not_in(cl_device_id device, cl_context context)
{
  std::vector<cl_device_id> devices = get_devices(context);
  size_t nd = devices.size();

  for (size_t i = 0; i < nd; ++i)
    if (device == devices[i])
      return false;

  return true;
}

// command queue and buffer should belong to the same context
bool object_not_in(cl_command_queue command_queue, cl_mem buffer)
{
  cl_context c_context;
  tdispatch->clGetCommandQueueInfo(
    command_queue,
    CL_QUEUE_CONTEXT,
    sizeof(cl_context),
    &c_context,
    NULL);
  cl_context b_context;
  tdispatch->clGetMemObjectInfo(
    buffer,
    CL_MEM_CONTEXT,
    sizeof(cl_context),
    &b_context,
    NULL);

  if (b_context == c_context)
      return false;
  return true;
}

// events and command queue should belong to the same context
bool object_not_in(cl_event event, cl_command_queue command_queue)
{
  cl_context e_context;
  tdispatch->clGetEventInfo(
    event,
    CL_EVENT_CONTEXT,
    sizeof(cl_context),
    &e_context,
    NULL);
  cl_context c_context;
  tdispatch->clGetCommandQueueInfo(
    command_queue,
    CL_QUEUE_CONTEXT,
    sizeof(cl_context),
    &c_context,
    NULL);

  if (e_context == c_context)
      return false;
  return true;
}

// mem objects and command queue should belong to the same context
bool object_not_in(cl_mem object, cl_command_queue command_queue)
{
  cl_context m_context;
  tdispatch->clGetMemObjectInfo(
    object,
    CL_MEM_CONTEXT,
    sizeof(cl_context),
    &m_context,
    NULL);
  cl_context c_context;
  tdispatch->clGetCommandQueueInfo(
    command_queue,
    CL_QUEUE_CONTEXT,
    sizeof(cl_context),
    &c_context,
    NULL);

  if (m_context == c_context)
      return false;
  return true;
}

// command queue and kernel should belong to the same context
bool object_not_in(cl_command_queue command_queue, cl_kernel kernel)
{
  cl_context c_context;
  tdispatch->clGetCommandQueueInfo(
    command_queue,
    CL_QUEUE_CONTEXT,
    sizeof(cl_context),
    &c_context,
    NULL);
  cl_context k_context;
  tdispatch->clGetKernelInfo(
    kernel,
    CL_KERNEL_CONTEXT,
    sizeof(cl_context),
    &k_context,
    NULL);

  if (k_context == c_context)
      return false;
  return true;
}

// events should belong to the same context
bool object_not_in(cl_event event1, cl_event event2)
{
  cl_context e_context;
  tdispatch->clGetEventInfo(
    event1,
    CL_EVENT_CONTEXT,
    sizeof(cl_context),
    &e_context,
    NULL);
  cl_context c_context;
  tdispatch->clGetEventInfo(
    event2,
    CL_EVENT_CONTEXT,
    sizeof(cl_context),
    &c_context,
    NULL);

  if (e_context == c_context)
      return false;
  return true;
}

// command_queue should be on device
bool object_not_in(cl_command_queue command_queue, cl_device_id device)
{
  cl_device_id q_device;
  tdispatch->clGetCommandQueueInfo(
    command_queue,
    CL_QUEUE_DEVICE,
    sizeof(cl_device_id),
    &q_device,
    NULL);

  if (device == q_device)
    return false;
  return true;
}

// device should belong to the program
bool object_not_in(cl_device_id device, cl_program program)
{
  std::vector<cl_device_id> devices = get_devices(program);
  size_t nd = devices.size();

  for (size_t i = 0; i < nd; ++i)
  {
    if (device == devices[i])
      return false;
  }

  return true;
}

// device should belong to the kernel
bool object_not_in(cl_device_id device, cl_kernel kernel)
{
  std::vector<cl_device_id> devices = get_devices(kernel);
  size_t nd = devices.size();

  for (size_t i = 0; i < nd; ++i)
  {
    if (device == devices[i])
      return false;
  }

  return true;
}

// kernel must be built for the device of command_queue
bool object_not_in(cl_kernel kernel, cl_command_queue command_queue)
{
  std::vector<cl_device_id> devices = get_devices(kernel);
  size_t nd = devices.size();

  cl_device_id d;
  tdispatch->clGetCommandQueueInfo(
    command_queue,
    CL_QUEUE_DEVICE,
    sizeof(d),
    &d,
    NULL);

  for (size_t i = 0; i < nd; ++i)
  {
    if (d == devices[i])
      return false;
  }

  return true;
}

template<typename T1, typename T2>
bool any_object_not_in(T1 * objects, size_t n, T2 in)
{
  for (size_t i = 0; i < n; ++i)
    if (object_not_in(objects[i], in))
      return true;
  return false;
}

template<cl_uint property>
bool for_all(const cl_device_id * devices, const size_t nd,
  std::function<bool(return_type<property>)> check)
{
  return_type<property> a;
  bool res = true;
  for (size_t i = 0; i < nd; ++i)
  {
   tdispatch->clGetDeviceInfo(
      devices[i],
      property,
      sizeof(a),
      &a,
      NULL);
    res = res && check(a);
  }

  return res;
}

template<cl_uint property>
bool for_all(cl_context context, std::function<bool(return_type<property>)> check)
{
  std::vector<cl_device_id> devices = get_devices(context);
  size_t nd = devices.size();

  return_type<property> a;
  bool res = true;
  for (size_t i = 0; i < nd; ++i)
  {
   tdispatch->clGetDeviceInfo(
      devices[i],
      property,
      sizeof(a),
      &a,
      NULL);
    res = res && check(a);
  }

  return res;
}

template<cl_uint property>
bool for_all(cl_program program, std::function<bool(return_type<property>)> check)
{
  std::vector<cl_device_id> devices = get_devices(program);
  size_t nd = devices.size();

  return_type<property> a;
  bool res = true;
  for (size_t i = 0; i < nd; ++i)
  {
   tdispatch->clGetDeviceInfo(
      devices[i],
      property,
      sizeof(a),
      &a,
      NULL);
    res = res && check(a);
  }

  return res;
}

template<cl_uint property>
bool for_all(cl_kernel kernel, std::function<bool(return_type<property>)> check)
{
  std::vector<cl_device_id> devices = get_devices(kernel);
  size_t nd = devices.size();

  return_type<property> a;
  bool res = true;
  for (size_t i = 0; i < nd; ++i)
  {
   tdispatch->clGetDeviceInfo(
      devices[i],
      property,
      sizeof(a),
      &a,
      NULL);
    res = res && check(a);
  }

  return res;
}

template<cl_uint property>
bool for_any(const cl_device_id * devices, const size_t nd,
  std::function<bool(return_type<property>)> check)
{
  return_type<property> a;
  bool res = false;
  for (size_t i = 0; i < nd; ++i)
  {
    tdispatch->clGetDeviceInfo(
      devices[i],
      property,
      sizeof(a),
      &a,
      NULL);
    res = res || check(a);
  }

  return res;
}

template<cl_uint property>
bool for_any(cl_context context, std::function<bool(return_type<property>)> check)
{
  std::vector<cl_device_id> devices = get_devices(context);
  size_t nd = devices.size();

  return_type<property> a;
  bool res = false;
  for (size_t i = 0; i < nd; ++i)
  {
    tdispatch->clGetDeviceInfo(
      devices[i],
      property,
      sizeof(a),
      &a,
      NULL);
    res = res || check(a);
  }

  return res;
}

template<cl_uint property>
bool for_any(cl_program program, std::function<bool(return_type<property>)> check)
{
  std::vector<cl_device_id> devices = get_devices(program);
  size_t nd = devices.size();

  return_type<property> a;
  bool res = false;
  for (size_t i = 0; i < nd; ++i)
  {
    tdispatch->clGetDeviceInfo(
      devices[i],
      property,
      sizeof(a),
      &a,
      NULL);
    res = res || check(a);
  }

  return res;
}

// adopted from Appendix D of OpenCL 3.0 standard
bool check_copy_overlap(
  const size_t src_origin[],
  const size_t dst_origin[],
  const size_t region[],
  const size_t row_pitch,
  const size_t slice_pitch)
{
  const size_t real_row_pitch = row_pitch > 0 ? row_pitch : region[0];
  const size_t real_slice_pitch = slice_pitch > 0 ? slice_pitch : region[1] * real_row_pitch;

  const size_t slice_size = (region[1] - 1) * real_row_pitch + region[0];
  const size_t block_size = (region[2] - 1) * real_slice_pitch + slice_size;
  const size_t src_start = 
    src_origin[2] * real_slice_pitch + src_origin[1] * real_row_pitch + src_origin[0];
  const size_t src_end = src_start + block_size;
  const size_t dst_start = 
    dst_origin[2] * real_slice_pitch + dst_origin[1] * real_row_pitch + dst_origin[0];
  const size_t dst_end = dst_start + block_size;

  // No overlap if dst ends before src starts or if src ends before dst starts.
  if ( (dst_end <= src_start) || (src_end <= dst_start) ) {
    return false;
  }

  // No overlap if region[0] for dst or src fits in the gap between region[0] and row_pitch.
  {
    const size_t src_dx = src_origin[0] % real_row_pitch;
    const size_t dst_dx = dst_origin[0] % real_row_pitch;
    if ( ((dst_dx >= src_dx + region[0]) && 
          (dst_dx + region[0] <= src_dx + real_row_pitch)) ||
         ((src_dx >= dst_dx + region[0]) && 
          (src_dx + region[0] <= dst_dx + real_row_pitch)) )
    {
      return false;
    }
}

  // No overlap if region[1] for dst or src fits in the gap between region[1] and slice_pitch.
  {
    const size_t src_dy = 
      (src_origin[1] * real_row_pitch + src_origin[0]) % real_slice_pitch;
    const size_t dst_dy = 
      (dst_origin[1] * real_row_pitch + dst_origin[0]) % real_slice_pitch;
    if ( ((dst_dy >= src_dy + slice_size) && 
          (dst_dy + slice_size <= src_dy + real_slice_pitch)) ||
         ((src_dy >= dst_dy + slice_size) && 
          (src_dy + slice_size <= dst_dy + real_slice_pitch)) )
    {
      return false;
    }
  }

  // Otherwise src and dst overlap.
  return true;
}

// check overlap for clEnqueueSVMMemcpy
bool check_copy_overlap(
  void * dst_ptr,
  const void * src_ptr,
  size_t size)
{
  uint8_t * dst = static_cast<uint8_t *>(dst_ptr);
  void * temp = const_cast<void *>(src_ptr);
  uint8_t * src = static_cast<uint8_t *>(temp);

  if ((src < dst + 1) && (dst < src + size))
    return true;
  if ((dst < src + 1) && (src < dst + size))
    return true;

  return false;
}

// check if ptr is not aligned to align
bool not_aligned(
  void * ptr,
  size_t align)
{
  if (std::align(align, align, ptr, align) == nullptr)
    return true;

  return false;
}

template<typename T>
std::remove_cv_t<T> mult(T * array, size_t elements)
{
  std::remove_cv_t<T> res = array[0];
  for (size_t i = 1; i < elements; ++i)
    res *= array[i];

  return res;
}
