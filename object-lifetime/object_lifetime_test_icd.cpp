#include "object_lifetime_test_icd.hpp"
#include "object_lifetime_test_icd_surface.hpp"

namespace lifetime
{
  bool report_implicit_ref_count_to_user,
       allow_using_released_objects,
       allow_using_inaccessible_objects;

  void object_parents<cl_device_id>::notify()
  {
    if (parent)
      parent->unreference();
  }

  void object_parents<cl_context>::notify()
  {
    for (auto& parent : parent_devices)
      parent->unreference();
  }

  void object_parents<cl_command_queue>::notify()
  {
    parent_device->unreference();
    parent_context->unreference();
  }

  void object_parents<cl_mem>::notify()
  {
    if (parent_mem)
      parent_mem->unreference();
    parent_context->unreference();
  }

  void object_parents<cl_program>::notify()
  {
    parent_context->unreference();
    for (auto& parent : parent_devices)
      parent->unreference();
  }

  void object_parents<cl_kernel>::notify()
  {
    parent_program->unreference();
  }

  void object_parents<cl_event>::notify()
  {
    parent_context->unreference();
    if (parent_queue)
      parent_queue->unreference();
  }

  void object_parents<cl_sampler>::notify()
  {
    parent_context->unreference();
  }

  icd_compatible::icd_compatible()
    : dispatch{ &_dispatch }
  {}
}

#include <utils.hpp>

#include <string>
#include <algorithm>
#include <iterator>

_cl_device_id::_cl_device_id(device_kind kind, cl_device_id parent, cl_uint num_cu)
  : icd_compatible{}
  , ref_counted_object<cl_device_id>{ lifetime::object_parents<cl_device_id>{ parent } }
  , platform{ &lifetime::_platform }
  , kind{ kind }
  , profile{ "FULL_PROFILE" }
  , version { lifetime::_platform.version }
  , name{ "Test Device" }
  , vendor{ "The Khronos Group" }
  , extensions{ "" }
  , cu_count{ num_cu }
{}

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
     case CL_DEVICE_PLATFORM:
      std::copy(
        reinterpret_cast<char*>(&platform),
        reinterpret_cast<char*>(&platform) + sizeof(platform),
        std::back_inserter(result));
      break;
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
    case CL_DEVICE_VERSION:
      std::copy(version.begin(), version.end(), std::back_inserter(result));
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
    case CL_DEVICE_PARENT_DEVICE:
      std::copy(
        reinterpret_cast<char*>(&parents.parent),
        reinterpret_cast<char*>(&parents.parent) + sizeof(parents.parent),
        std::back_inserter(result));
      break;
    case CL_DEVICE_REFERENCE_COUNT:
    {
      cl_uint tmp = CL_OBJECT_REFERENCE_COUNT();
      std::copy(
        reinterpret_cast<char*>(&tmp),
        reinterpret_cast<char*>(&tmp) + sizeof(tmp),
        std::back_inserter(result));
      break;
    }
    case CL_DEVICE_NUMERIC_VERSION:
      std::copy(
        reinterpret_cast<char*>(&numeric_version),
        reinterpret_cast<char*>(&numeric_version) + sizeof(numeric_version),
        std::back_inserter(result));
      break;
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
          this,
          n
        )
      );
      reference(num_devices);

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

      for (const auto& dev : result)
        lifetime::get_objects<cl_device_id>().insert(dev);

      return CL_SUCCESS;
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

cl_context _cl_device_id::clCreateContext(
  const cl_context_properties*,
  cl_uint num_devices,
  const cl_device_id* devices,
  void (CL_CALLBACK*)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
  void*,
  cl_int* errcode_ret)
{
  bool all_devices_are_ours = std::all_of(
    devices,
    devices + num_devices,
    [dev_plat = (cl_platform_id)nullptr](const cl_device_id& device) mutable
    {
      device->dispatch->clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &dev_plat, nullptr);
      return dev_plat == &lifetime::_platform;
    }
  );

  if (!all_devices_are_ours)
  {
    if (errcode_ret)
      *errcode_ret = CL_INVALID_DEVICE;
    return nullptr;
  }

  std::for_each(
    devices,
    devices + num_devices,
    [](const cl_device_id& device)
    {
      device->reference();
    }
  );

  auto result = lifetime::get_objects<cl_context>().insert(
    std::make_shared<_cl_context>(
      devices,
      devices + num_devices
    )
  );

  if(result.second)
  {
    if (errcode_ret)
      *errcode_ret = CL_SUCCESS;
    return result.first->get();
  }
  else
  {
    std::exit(-1);
  }
}

_cl_mem::_cl_mem(cl_mem mem_parent, cl_context context_parent, size_t size)
  : icd_compatible{}
  , ref_counted_object<cl_mem>{ lifetime::object_parents<cl_mem>{ mem_parent, context_parent } }
  , _size{ size }
{}

cl_mem _cl_mem::clCreateSubBuffer(
  cl_mem_flags,
  cl_buffer_create_type buffer_create_type,
  const void* buffer_create_info,
  cl_int* errcode_ret)
{
  if (buffer_create_type != CL_BUFFER_CREATE_TYPE_REGION ||
      buffer_create_info == nullptr)
  {
    if (errcode_ret)
      *errcode_ret = CL_INVALID_VALUE;
    return nullptr;
  }

  const cl_buffer_region* region_info =
    reinterpret_cast<const cl_buffer_region*>(buffer_create_info);

  if (region_info->origin + region_info->size > this->_size)
  {
    if (errcode_ret)
      *errcode_ret = CL_INVALID_BUFFER_SIZE;
    return nullptr;
  }

  auto result = lifetime::get_objects<cl_mem>().insert(
    std::make_shared<_cl_mem>(
      this,
      parents.parent_context,
      region_info->size
    )
  );

  if(result.second)
  {
    if (errcode_ret)
      *errcode_ret = CL_SUCCESS;
    return result.first->get();
  }
  else
  {
    std::exit(-1);
  }
}

cl_int _cl_mem::clGetMemObjectInfo(
    cl_mem_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  if (param_value_size == 0 && param_value != NULL)
    return CL_INVALID_VALUE;

  std::vector<char> result;
  switch(param_name)
  {
    case CL_MEM_SIZE:
    {
      std::copy(
        reinterpret_cast<char*>(&_size),
        reinterpret_cast<char*>(&_size) + sizeof(_size),
        std::back_inserter(result));
      break;
    }
    case CL_MEM_CONTEXT:
      std::copy(
        reinterpret_cast<char*>(&parents.parent_context),
        reinterpret_cast<char*>(&parents.parent_context) + sizeof(parents.parent_context),
        std::back_inserter(result));
      break;
    case CL_MEM_REFERENCE_COUNT:
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

cl_int _cl_mem::clRetainMemObject()
{
  return retain();
}

cl_int _cl_mem::clReleaseMemObject()
{
  return release();
}

_cl_command_queue::_cl_command_queue(cl_device_id parent_device, cl_context parent_context)
  : icd_compatible{}
  , ref_counted_object<cl_command_queue>{ lifetime::object_parents<cl_command_queue>{ parent_device, parent_context } }
{}

cl_int _cl_command_queue::clGetCommandQueueInfo(
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
    case CL_QUEUE_CONTEXT:
      std::copy(
        reinterpret_cast<char*>(&parents.parent_context),
        reinterpret_cast<char*>(&parents.parent_context) + sizeof(parents.parent_context),
        std::back_inserter(result));
      break;
    case CL_QUEUE_DEVICE:
      std::copy(
        reinterpret_cast<char*>(&parents.parent_context),
        reinterpret_cast<char*>(&parents.parent_context) + sizeof(parents.parent_context),
        std::back_inserter(result));
      break;
    case CL_MEM_REFERENCE_COUNT:
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

cl_int _cl_command_queue::clRetainCommandQueue()
{
  return retain();
}

cl_int _cl_command_queue::clReleaseCommandQueue()
{
  return release();
}

_cl_context::_cl_context(const cl_device_id* first_device, const cl_device_id* last_device)
  : icd_compatible{}
  , ref_counted_object<cl_context>{ lifetime::object_parents<cl_context>{ {first_device, last_device} } }
{}

cl_int _cl_context::clGetContextInfo(
    cl_context_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  if (param_value_size == 0 && param_value != NULL)
    return CL_INVALID_VALUE;

  std::vector<char> result;
  switch(param_name)
  {
    case CL_CONTEXT_REFERENCE_COUNT:
    {
      cl_uint tmp = CL_OBJECT_REFERENCE_COUNT();
      std::copy(
        reinterpret_cast<char*>(&tmp),
        reinterpret_cast<char*>(&tmp) + sizeof(tmp),
        std::back_inserter(result));
      break;
    }
    case CL_CONTEXT_NUM_DEVICES:
    {
      cl_uint tmp = static_cast<cl_uint>(parents.parent_devices.size());
      std::copy(
        reinterpret_cast<char*>(&tmp),
        reinterpret_cast<char*>(&tmp) + sizeof(tmp),
        std::back_inserter(result));
      break;
    }
    case CL_CONTEXT_DEVICES:
      std::copy(
        reinterpret_cast<char*>(&(*parents.parent_devices.begin())),
        reinterpret_cast<char*>(&(*parents.parent_devices.end())),
        std::back_inserter(result));
      break;
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

cl_int _cl_context::clRetainContext()
{
  return retain();
}

cl_int _cl_context::clReleaseContext()
{
  return release();
}

cl_mem _cl_context::clCreateBuffer(
  cl_mem_flags,
  size_t size,
  void*,
  cl_int* errcode_ret)
{
  auto result = lifetime::get_objects<cl_mem>().insert(
    std::make_shared<_cl_mem>(
      nullptr,
      this,
      size
    )
  );

  if(result.second)
  {
    if (errcode_ret)
      *errcode_ret = CL_SUCCESS;
    return result.first->get();
  }
  else
  {
    std::exit(-1);
  }
}

cl_command_queue _cl_context::clCreateCommandQueue(
  cl_device_id device,
  cl_command_queue_properties,
  cl_int* errcode_ret)
{
  if (std::find(
    parents.parent_devices.cbegin(),
    parents.parent_devices.cend(),
    device
  ) == parents.parent_devices.cend()
  )
  {
    if (errcode_ret)
      *errcode_ret = CL_INVALID_DEVICE;
    return nullptr;
  }

  auto result = lifetime::get_objects<cl_command_queue>().insert(
    std::make_shared<_cl_command_queue>(
      device,
      this
    )
  );

  if(result.second)
  {
    if (errcode_ret)
      *errcode_ret = CL_SUCCESS;
    return result.first->get();
  }
  else
  {
    std::exit(-1);
  }
}

cl_program _cl_context::clCreateProgramWithSource(
  cl_uint,
  const char**,
  const size_t*,
  cl_int* errcode_ret)
{
  auto result = lifetime::get_objects<cl_program>().insert(
    std::make_shared<_cl_program>(
      this,
      parents.parent_devices.data(),
      parents.parent_devices.data() + parents.parent_devices.size()
    )
  );

  if(result.second)
  {
    if (errcode_ret)
      *errcode_ret = CL_SUCCESS;
    return result.first->get();
  }
  else
  {
    std::exit(-1);
  }
}

cl_event _cl_context::clCreateUserEvent(
    cl_int* errcode_ret)
{
  auto result = lifetime::get_objects<cl_event>().insert(
    std::make_shared<_cl_event>(
      this,
      nullptr
    )
  );

  if(result.second)
  {
    if (errcode_ret)
      *errcode_ret = CL_SUCCESS;
    return result.first->get();
  }
  else
  {
    std::exit(-1);
  }
}

cl_sampler _cl_context::clCreateSampler(
    cl_bool,
    cl_addressing_mode,
    cl_filter_mode,
    cl_int* errcode_ret)
{
  auto result = lifetime::get_objects<cl_sampler>().insert(
    std::make_shared<_cl_sampler>(
      this
    )
  );

  if(result.second)
  {
    if (errcode_ret)
      *errcode_ret = CL_SUCCESS;
    return result.first->get();
  }
  else
  {
    std::exit(-1);
  }
}

_cl_program::_cl_program(
  const cl_context parent_context,
  const cl_device_id* first_device,
  const cl_device_id* last_device
)
  : icd_compatible{}
  , ref_counted_object<cl_program>{ lifetime::object_parents<cl_program>{ parent_context, {first_device, last_device} } }
{}

cl_int _cl_program::clBuildProgram(
  cl_uint num_devices,
  const cl_device_id* device_list,
  const char*,
  void (CL_CALLBACK*)(cl_program program, void* user_data),
  void*)
{
  bool all_devices_are_in_context = std::all_of(
    device_list,
    device_list + num_devices,
    [this](const cl_device_id& device) mutable
    {
      return std::find(
        parents.parent_devices.cbegin(),
        parents.parent_devices.cend(),
        device
      ) != parents.parent_devices.cend();
    }
  );

  if (!all_devices_are_in_context)
  {
    return CL_INVALID_DEVICE;
  }

  return CL_SUCCESS;
}

cl_int _cl_program::clGetProgramInfo(
  cl_program_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  if (param_value_size == 0 && param_value != NULL)
    return CL_INVALID_VALUE;

  std::vector<char> result;
  switch(param_name)
  {
    case CL_PROGRAM_REFERENCE_COUNT:
    {
      cl_uint tmp = CL_OBJECT_REFERENCE_COUNT();
      std::copy(
        reinterpret_cast<char*>(&tmp),
        reinterpret_cast<char*>(&tmp) + sizeof(tmp),
        std::back_inserter(result));
      break;
    }
    case CL_PROGRAM_CONTEXT:
      std::copy(
        reinterpret_cast<char*>(&parents.parent_context),
        reinterpret_cast<char*>(&parents.parent_context + sizeof(parents.parent_context)),
        std::back_inserter(result));
      break;
    case CL_PROGRAM_NUM_DEVICES:
    {
      cl_uint tmp = static_cast<cl_uint>(parents.parent_devices.size());
      std::copy(
        reinterpret_cast<char*>(&tmp),
        reinterpret_cast<char*>(&tmp) + sizeof(tmp),
        std::back_inserter(result));
      break;
    }
    case CL_PROGRAM_DEVICES:
      std::copy(
        reinterpret_cast<char*>(&(*parents.parent_devices.begin())),
        reinterpret_cast<char*>(&(*parents.parent_devices.end())),
        std::back_inserter(result));
      break;
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

cl_int _cl_program::clRetainProgram()
{
  return retain();
}

cl_int _cl_program::clReleaseProgram()
{
  return release();
}

cl_kernel _cl_program::clCreateKernel(
  const char*,
  cl_int* errcode_ret)
{
  auto result = lifetime::get_objects<cl_kernel>().insert(
    std::make_shared<_cl_kernel>(
      this
    )
  );

  if(result.second)
  {
    if (errcode_ret)
      *errcode_ret = CL_SUCCESS;
    return result.first->get();
  }
  else
  {
    std::exit(-1);
  }
}

_cl_kernel::_cl_kernel(const cl_program parent_program)
  : icd_compatible{}
  , ref_counted_object<cl_kernel>{ lifetime::object_parents<cl_kernel>{ parent_program } }
{}

cl_int _cl_kernel::clGetKernelInfo(
  cl_kernel_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  if (param_value_size == 0 && param_value != NULL)
    return CL_INVALID_VALUE;

  std::vector<char> result;
  switch(param_name)
  {
    case CL_KERNEL_REFERENCE_COUNT:
    {
      cl_uint tmp = CL_OBJECT_REFERENCE_COUNT();
      std::copy(
        reinterpret_cast<char*>(&tmp),
        reinterpret_cast<char*>(&tmp) + sizeof(tmp),
        std::back_inserter(result));
      break;
    }
    case CL_KERNEL_PROGRAM:
      std::copy(
        reinterpret_cast<char*>(&parents.parent_program),
        reinterpret_cast<char*>(&parents.parent_program + sizeof(parents.parent_program)),
        std::back_inserter(result));
      break;
    case CL_KERNEL_CONTEXT:
    {
      cl_context tmp = parents.parent_program->parents.parent_context;
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

cl_int _cl_kernel::clRetainKernel()
{
  return retain();
}

cl_int _cl_kernel::clReleaseKernel()
{
  return release();
}

cl_kernel _cl_kernel::clCloneKernel(
  cl_int* errcode_ret)
{
  auto result = lifetime::get_objects<cl_kernel>().insert(
    std::make_shared<_cl_kernel>(
      parents.parent_program
    )
  );

  if(result.second)
  {
    if (errcode_ret)
      *errcode_ret = CL_SUCCESS;
    return result.first->get();
  }
  else
  {
    std::exit(-1);
  }
}

_cl_event::_cl_event(const cl_context parent_context, const cl_command_queue parent_queue)
  : icd_compatible{}
  , ref_counted_object<cl_event>{ lifetime::object_parents<cl_event>{ parent_context, parent_queue } }
{}

cl_int _cl_event::clGetEventInfo(
  cl_event_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  if (param_value_size == 0 && param_value != NULL)
    return CL_INVALID_VALUE;

  std::vector<char> result;
  switch(param_name)
  {
    case CL_EVENT_COMMAND_QUEUE:
      if (parents.parent_queue)
        std::copy(
          reinterpret_cast<char*>(&parents.parent_queue),
          reinterpret_cast<char*>(&parents.parent_queue + sizeof(parents.parent_queue)),
          std::back_inserter(result));
      else
      {
        cl_command_queue tmp = NULL;
        std::copy(
          reinterpret_cast<char*>(&tmp),
          reinterpret_cast<char*>(&tmp + sizeof(tmp)),
          std::back_inserter(result));
      }
      break;
    case CL_EVENT_CONTEXT:
    {
      std::copy(
        reinterpret_cast<char*>(&parents.parent_context),
        reinterpret_cast<char*>(&parents.parent_context) + sizeof(parents.parent_context),
        std::back_inserter(result));
      break;
    }
    case CL_EVENT_REFERENCE_COUNT:
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

cl_int _cl_event::clRetainEvent()
{
  return retain();
}

cl_int _cl_event::clReleaseEvent()
{
  return release();
}

_cl_sampler::_cl_sampler(const cl_context parent_context)
  : icd_compatible{}
  , ref_counted_object<cl_sampler>{ lifetime::object_parents<cl_sampler>{ parent_context } }
{}

cl_int _cl_sampler::clGetSamplerInfo(
  cl_event_info param_name,
  size_t param_value_size,
  void* param_value,
  size_t* param_value_size_ret)
{
  if (param_value_size == 0 && param_value != NULL)
    return CL_INVALID_VALUE;

  std::vector<char> result;
  switch(param_name)
  {
    case CL_SAMPLER_REFERENCE_COUNT:
    {
      cl_uint tmp = CL_OBJECT_REFERENCE_COUNT();
      std::copy(
        reinterpret_cast<char*>(&tmp),
        reinterpret_cast<char*>(&tmp) + sizeof(tmp),
        std::back_inserter(result));
      break;
    }
    case CL_SAMPLER_CONTEXT:
    {
      std::copy(
        reinterpret_cast<char*>(&parents.parent_context),
        reinterpret_cast<char*>(&parents.parent_context) + sizeof(parents.parent_context),
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

cl_int _cl_sampler::clRetainSampler()
{
  return retain();
}

cl_int _cl_sampler::clReleaseSampler()
{
  return release();
}

_cl_platform_id::_cl_platform_id()
  : numeric_version{ CL_MAKE_VERSION(3, 0, 0) }
  , profile{ "FULL_PROFILE" }
  , version{}
  , name{ "Object Lifetime Layer Test ICD" }
  , vendor{ "Khronos" }
  , extensions{ "cl_khr_icd cl_khr_extended_versioning" }
  , suffix{ "khronos" }
  , _contexts{}
  , _devices{}
  , _mems{}
  , _queues{}
{
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

  using namespace lifetime;

  std::string REPORT_IMPLICIT_REF_COUNT_TO_USER;
  if (ocl_layer_utils::detail::get_environment("REPORT_IMPLICIT_REF_COUNT_TO_USER", REPORT_IMPLICIT_REF_COUNT_TO_USER))
  {
    if (CL_VERSION_MAJOR(numeric_version) == 1)
      std::exit(-1); // conflating implicit ref counts with regular ones is 2.0+ behavior
    report_implicit_ref_count_to_user = true;
  }
  else
    report_implicit_ref_count_to_user = false;

  std::string ALLOW_USING_RELEASED_OBJECTS;
  if (ocl_layer_utils::detail::get_environment("ALLOW_USING_RELEASED_OBJECTS", ALLOW_USING_RELEASED_OBJECTS))
  {
    allow_using_released_objects = true;
  }
  else
    allow_using_released_objects = false;

  std::string ALLOW_USING_INACCESSIBLE_OBJECTS;
  if (ocl_layer_utils::detail::get_environment("ALLOW_USING_INACCESSIBLE_OBJECTS", ALLOW_USING_INACCESSIBLE_OBJECTS))
  {
    allow_using_inaccessible_objects = true;
  }
  else
    allow_using_inaccessible_objects = false;

  _devices.insert(std::make_shared<_cl_device_id>(_cl_device_id::device_kind::root));
}

void _cl_platform_id::init_dispatch()
{
  dispatch->clGetPlatformInfo = clGetPlatformInfo_wrap;
  dispatch->clGetDeviceIDs = clGetDeviceIDs_wrap;
  dispatch->clGetDeviceInfo = clGetDeviceInfo_wrap;
  dispatch->clCreateSubDevices = clCreateSubDevices_wrap;
  dispatch->clRetainDevice = clRetainDevice_wrap;
  dispatch->clReleaseDevice = clReleaseDevice_wrap;
  dispatch->clCreateContext = clCreateContext_wrap;
  dispatch->clGetContextInfo = clGetContextInfo_wrap;
  dispatch->clRetainContext = clRetainContext_wrap;
  dispatch->clReleaseContext = clReleaseContext_wrap;
  dispatch->clCreateBuffer = clCreateBuffer_wrap;
  dispatch->clCreateCommandQueue = clCreateCommandQueue_wrap;
  dispatch->clCreateSubBuffer = clCreateSubBuffer_wrap;
  dispatch->clRetainMemObject = clRetainMemObject_wrap;
  dispatch->clReleaseMemObject = clReleaseMemObject_wrap;
  dispatch->clGetMemObjectInfo = clGetMemObjectInfo_wrap;
  dispatch->clGetCommandQueueInfo = clGetCommandQueueInfo_wrap;
  dispatch->clRetainCommandQueue = clRetainCommandQueue_wrap;
  dispatch->clReleaseCommandQueue = clReleaseCommandQueue_wrap;
  dispatch->clCreateProgramWithSource = clCreateProgramWithSource_wrap;
  dispatch->clBuildProgram = clBuildProgram_wrap;
  dispatch->clGetProgramInfo = clGetProgramInfo_wrap;
  dispatch->clRetainProgram = clRetainProgram_wrap;
  dispatch->clReleaseProgram = clReleaseProgram_wrap;
  dispatch->clCreateKernel = clCreateKernel_wrap;
  dispatch->clCloneKernel = clCloneKernel_wrap;
  dispatch->clGetKernelInfo = clGetKernelInfo_wrap;
  dispatch->clRetainKernel = clRetainKernel_wrap;
  dispatch->clReleaseKernel = clReleaseKernel_wrap;
  dispatch->clCreateUserEvent = clCreateUserEvent_wrap;
  dispatch->clGetEventInfo = clGetEventInfo_wrap;
  dispatch->clRetainEvent = clRetainEvent_wrap;
  dispatch->clReleaseEvent = clReleaseEvent_wrap;
  dispatch->clCreateSampler = clCreateSampler_wrap;
  dispatch->clGetSamplerInfo = clGetSamplerInfo_wrap;
  dispatch->clRetainSampler = clRetainSampler_wrap;
  dispatch->clReleaseSampler = clReleaseSampler_wrap;
}

cl_int _cl_platform_id::clGetPlatformInfo(
    cl_platform_info  param_name,
    size_t            param_value_size,
    void *            param_value,
    size_t *          param_value_size_ret)
{
  if (param_value_size == 0 && param_value != NULL)
    return CL_INVALID_VALUE;

  std::vector<char> result;
  switch(param_name) {
    case CL_PLATFORM_PROFILE:
      std::copy(profile.begin(), profile.end(), std::back_inserter(result));
      result.push_back('\0');
      break;
    case CL_PLATFORM_VERSION:
      std::copy(version.begin(), version.end(), std::back_inserter(result));
      result.push_back('\0');
      break;
    case CL_PLATFORM_NAME:
      std::copy(name.begin(), name.end(), std::back_inserter(result));
      result.push_back('\0');
      break;
    case CL_PLATFORM_VENDOR:
      std::copy(vendor.begin(), vendor.end(), std::back_inserter(result));
      result.push_back('\0');
      break;
    case CL_PLATFORM_EXTENSIONS:
      std::copy(extensions.begin(), extensions.end(), std::back_inserter(result));
      result.push_back('\0');
      break;
    case CL_PLATFORM_ICD_SUFFIX_KHR:
      std::copy(suffix.begin(), suffix.end(), std::back_inserter(result));
      result.push_back('\0');
      break;
    case CL_PLATFORM_NUMERIC_VERSION:
      std::copy(
        reinterpret_cast<char*>(&numeric_version),
        reinterpret_cast<char*>(&numeric_version) + sizeof(numeric_version),
        std::back_inserter(result));
      break;
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
  cl_icd_dispatch _dispatch;
}
