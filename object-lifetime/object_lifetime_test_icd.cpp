#include "object_lifetime_test_icd.hpp"
#include "object_lifetime_test_icd_surface.hpp"

#include <algorithm>

namespace lifetime
{
  bool report_implicit_ref_count_to_user,
       allow_using_released_objects,
       allow_using_inaccessible_objects,
       always_return_success;

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

  template <typename T, typename... Args> auto create_or_exit(cl_int* errcode_ret, Args&& ...args)
  {
    auto result = get_objects<T>().insert(
      std::make_shared<std::remove_pointer_t<T>>(
        args...
      )
    );

    if(result.second)
    {
      if (errcode_ret)
        *errcode_ret = CL_SUCCESS;
      return result.first->get();
    }
    else
      std::exit(-1);
  }
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
      cl_uint tmp = kind == device_kind::sub ? CL_OBJECT_REFERENCE_COUNT() : 1;
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
    case CL_DEVICE_AVAILABLE:
    {
      cl_bool available = CL_TRUE;
      std::copy(
        reinterpret_cast<char*>(&available),
        reinterpret_cast<char*>(&available) + sizeof(available),
        std::back_inserter(result));
      break;
    }
    case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
    {
      cl_ulong max_mem_alloc_size = 0x1000; // Arbitrary value
      std::copy(
        reinterpret_cast<char*>(&max_mem_alloc_size),
        reinterpret_cast<char*>(&max_mem_alloc_size) + sizeof(max_mem_alloc_size),
        std::back_inserter(result));
      break;
    }
    case CL_DEVICE_MEM_BASE_ADDR_ALIGN:
    {
      cl_uint align = 16;
      std::copy(
        reinterpret_cast<char*>(&align),
        reinterpret_cast<char*>(&align) + sizeof(align),
        std::back_inserter(result));
      break;
    }
    case CL_DEVICE_IMAGE_SUPPORT:
    {
      cl_bool support = true;
      std::copy(
        reinterpret_cast<char*>(&support),
        reinterpret_cast<char*>(&support) + sizeof(support),
        std::back_inserter(result));
      break;
    }
    case CL_DEVICE_IMAGE2D_MAX_WIDTH:
    case CL_DEVICE_IMAGE2D_MAX_HEIGHT:
    case CL_DEVICE_IMAGE3D_MAX_WIDTH:
    case CL_DEVICE_IMAGE3D_MAX_HEIGHT:
    case CL_DEVICE_IMAGE3D_MAX_DEPTH:
    case CL_DEVICE_IMAGE_MAX_BUFFER_SIZE:
    case CL_DEVICE_IMAGE_MAX_ARRAY_SIZE:
    {
      size_t max_size = 4096;
      std::copy(
        reinterpret_cast<char*>(&max_size),
        reinterpret_cast<char*>(&max_size) + sizeof(max_size),
        std::back_inserter(result));
      break;
    }
    case CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE:
    case CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE:
    {
      cl_uint max_size = 256 * 1024;
      std::copy(
        reinterpret_cast<char*>(&max_size),
        reinterpret_cast<char*>(&max_size) + sizeof(max_size),
        std::back_inserter(result));
      break;
    }
    case CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES:
    {
      cl_device_device_enqueue_capabilities caps = CL_DEVICE_QUEUE_SUPPORTED | CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT;
      std::copy(
        reinterpret_cast<char*>(&caps),
        reinterpret_cast<char*>(&caps) + sizeof(caps),
        std::back_inserter(result));
      break;
    }
    case CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT:
    {
      cl_bool support = CL_TRUE;
      std::copy(
        reinterpret_cast<char*>(&support),
        reinterpret_cast<char*>(&support) + sizeof(support),
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
      cl_uint ret_count = cu_count / n;
      if (num_devices < ret_count)
        return CL_INVALID_DEVICE_PARTITION_COUNT;

      if (num_devices_ret)
        *num_devices_ret = ret_count;

      if (out_devices && num_devices < ret_count)
        return CL_INVALID_VALUE;

      if (out_devices)
      {
        std::generate_n(
          out_devices,
          num_devices,
          [n = static_cast<cl_uint>(properties[1]), this]()
          {
            return lifetime::create_or_exit<cl_device_id>(
              nullptr,
              _cl_device_id::device_kind::sub,
              this,
              n);
          }
        );
        reference(ret_count);
      }

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

  return lifetime::create_or_exit<cl_context>(
    errcode_ret,
    devices,
    devices + num_devices
  );
}

_cl_mem::_cl_mem(cl_mem mem_parent,
                 cl_context context_parent,
                 cl_mem_object_type type,
                 cl_mem_flags flags,
                 size_t origin,
                 size_t size)
  : icd_compatible{}
  , ref_counted_object<cl_mem>{ lifetime::object_parents<cl_mem>{ mem_parent, context_parent } }
  , _type{ type }
  , _flags{ flags }
{
  _properties.buffer = { origin, size };
}

_cl_mem::_cl_mem(cl_mem mem_parent,
                 cl_context context_parent,
                 cl_mem_object_type type,
                 cl_mem_flags flags,
                 cl_image_format format,
                 cl_image_desc desc)
  : icd_compatible{}
  , ref_counted_object<cl_mem>{ lifetime::object_parents<cl_mem>{ mem_parent, context_parent } }
  , _type{ type }
  , _flags{ flags }
{
  _properties.image = { format, desc };
}

cl_mem _cl_mem::clCreateSubBuffer(
  cl_mem_flags flags,
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

  if (_type != CL_MEM_OBJECT_BUFFER)
  {
    if (errcode_ret)
      *errcode_ret = CL_INVALID_VALUE;
    return nullptr;
  }

  const cl_buffer_region* region_info =
    reinterpret_cast<const cl_buffer_region*>(buffer_create_info);

  if (region_info->origin + region_info->size > this->_properties.buffer.size)
  {
    if (errcode_ret)
      *errcode_ret = CL_INVALID_BUFFER_SIZE;
    return nullptr;
  }

  reference();
  return lifetime::create_or_exit<cl_mem>(
    errcode_ret,
    this,
    parents.parent_context,
    CL_MEM_OBJECT_BUFFER,
    flags,
    region_info->origin,
    region_info->size
  );
}

static size_t image_format_size(cl_image_format format)
{
  size_t channels;
  switch (format.image_channel_order)
  {
    case CL_R:
    case CL_A:
    case CL_DEPTH:
    case CL_LUMINANCE:
    case CL_INTENSITY:
      channels = 1;
      break;
    case CL_RG:
    case CL_RA:
    case CL_Rx:
      channels = 2;
      break;
    case CL_RGB:
    case CL_RGx:
      channels = 3;
      break;
    default:
      channels = 4;
      break;
  }

  size_t type_size;
  switch (format.image_channel_data_type)
  {
    case CL_SNORM_INT8:
    case CL_UNORM_INT8:
    case CL_SIGNED_INT8:
    case CL_UNSIGNED_INT8:
      type_size = 1;
      break;
    case CL_SNORM_INT16:
    case CL_UNORM_INT16:
    case CL_UNORM_SHORT_555:
    case CL_UNORM_SHORT_565:
    case CL_SIGNED_INT16:
    case CL_UNSIGNED_INT16:
    case CL_HALF_FLOAT:
      type_size = 2;
      break;
    default:
      type_size = 4;
      break;
  }

  return type_size * channels;
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
      size_t size;
      if (_type == CL_MEM_OBJECT_BUFFER)
      {
        size = _properties.buffer.size;
      }
      else
      {
        size = image_format_size(_properties.image.format)
             * _properties.image.desc.image_array_size
             * _properties.image.desc.image_depth;
        if (_properties.image.desc.image_slice_pitch)
          size *= _properties.image.desc.image_slice_pitch;
        else
        {
          size *= _properties.image.desc.image_height;
          if (_properties.image.desc.image_row_pitch)
            size *= _properties.image.desc.image_row_pitch;
          else
            size *= _properties.image.desc.image_width;
        }
      }
      std::copy(
        reinterpret_cast<char*>(&size),
        reinterpret_cast<char*>(&size) + sizeof(size),
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
    case CL_MEM_OFFSET:
    {
      std::copy(
        reinterpret_cast<char*>(&_properties.buffer.origin),
        reinterpret_cast<char*>(&_properties.buffer.origin) + sizeof(_properties.buffer.origin),
        std::back_inserter(result));
      break;
    }
    case CL_MEM_TYPE:
    {
      std::copy(
        reinterpret_cast<char*>(&_type),
        reinterpret_cast<char*>(&_type) + sizeof(_type),
        std::back_inserter(result));
      break;
    }
    case CL_MEM_FLAGS:
    {
      std::copy(
        reinterpret_cast<char*>(&_flags),
        reinterpret_cast<char*>(&_flags) + sizeof(_flags),
        std::back_inserter(result));
      break;
    }
    case CL_MEM_ASSOCIATED_MEMOBJECT:
    {
      std::copy(
        reinterpret_cast<char*>(&parents.parent_mem),
        reinterpret_cast<char*>(&parents.parent_mem) + sizeof(parents.parent_mem),
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

cl_int _cl_mem::clGetImageInfo(
    cl_image_info param_name,
    size_t param_value_size,
    void* param_value,
    size_t* param_value_size_ret)
{
  if (param_value_size == 0 && param_value != NULL)
    return CL_INVALID_VALUE;

  if (_type == CL_MEM_OBJECT_BUFFER || _type == CL_MEM_OBJECT_PIPE)
    return CL_INVALID_VALUE;

  std::vector<char> result;
  switch(param_name)
  {
    case CL_IMAGE_FORMAT:
    {
      std::copy(
        reinterpret_cast<char*>(&_properties.image.format),
        reinterpret_cast<char*>(&_properties.image.format) + sizeof(_properties.image.format),
        std::back_inserter(result));
      break;
    }
    case CL_IMAGE_WIDTH:
    {
      std::copy(
        reinterpret_cast<char*>(&_properties.image.desc.image_width),
        reinterpret_cast<char*>(&_properties.image.desc.image_width) + sizeof(_properties.image.desc.image_width),
        std::back_inserter(result));
      break;
    }
    case CL_IMAGE_HEIGHT:
    {
      std::copy(
        reinterpret_cast<char*>(&_properties.image.desc.image_height),
        reinterpret_cast<char*>(&_properties.image.desc.image_height) + sizeof(_properties.image.desc.image_height),
        std::back_inserter(result));
      break;
    }
    case CL_IMAGE_DEPTH:
    {
      std::copy(
        reinterpret_cast<char*>(&_properties.image.desc.image_depth),
        reinterpret_cast<char*>(&_properties.image.desc.image_depth) + sizeof(_properties.image.desc.image_depth),
        std::back_inserter(result));
      break;
    }
    case CL_IMAGE_ARRAY_SIZE:
    {
      std::copy(
        reinterpret_cast<char*>(&_properties.image.desc.image_array_size),
        reinterpret_cast<char*>(&_properties.image.desc.image_array_size) + sizeof(_properties.image.desc.image_array_size),
        std::back_inserter(result));
      break;
    }
    case CL_IMAGE_ELEMENT_SIZE:
    {
      size_t element_size = image_format_size(_properties.image.format);
      std::copy(
        reinterpret_cast<char*>(&element_size),
        reinterpret_cast<char*>(&element_size) + sizeof(element_size),
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

_cl_command_queue::_cl_command_queue(cl_device_id parent_device, cl_context parent_context, const cl_queue_properties* first, const cl_queue_properties* last)
  : icd_compatible{}
  , ref_counted_object<cl_command_queue>{ lifetime::object_parents<cl_command_queue>{ parent_device, parent_context } }
  , _size{ 0 }
  , _props( first, last )
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
        reinterpret_cast<char*>(&parents.parent_device),
        reinterpret_cast<char*>(&parents.parent_device) + sizeof(parents.parent_device),
        std::back_inserter(result));
      break;
    case CL_QUEUE_REFERENCE_COUNT:
    {
      cl_uint tmp = CL_OBJECT_REFERENCE_COUNT();
      std::copy(
        reinterpret_cast<char*>(&tmp),
        reinterpret_cast<char*>(&tmp) + sizeof(tmp),
        std::back_inserter(result));
      break;
    }
    case CL_QUEUE_PROPERTIES:
    {
       cl_command_queue_properties props = 0;
       auto it = std::find(_props.cbegin(), _props.cend(), CL_QUEUE_PROPERTIES);
       if (it != _props.cend())
       {
         props = *(++it);
       }
       std::copy(
        reinterpret_cast<char*>(&props),
        reinterpret_cast<char*>(&props) + sizeof(props),
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

cl_int _cl_command_queue::clEnqueueNDRangeKernel(
  cl_kernel kernel,
  cl_uint,
  const size_t*,
  const size_t*,
  const size_t*,
  cl_uint num_events_in_wait_list,
  const cl_event* event_wait_list,
  cl_event* event)
{
  cl_context kernel_ctx = nullptr;
  kernel->dispatch->clGetKernelInfo(kernel, CL_KERNEL_CONTEXT, sizeof(cl_context), &kernel_ctx, nullptr);
  if (kernel_ctx != parents.parent_context)
    return CL_INVALID_KERNEL;

  if (!kernel->is_valid())
    return CL_INVALID_KERNEL;

  if ((num_events_in_wait_list == 0 && event_wait_list != nullptr) ||
      (num_events_in_wait_list != 0 && event_wait_list == nullptr))
    return CL_INVALID_EVENT_WAIT_LIST;

  bool all_events_are_ours_and_valid = std::all_of(
    event_wait_list,
    event_wait_list + num_events_in_wait_list,
    [event_ctx = (cl_context)nullptr, this](const cl_event& event) mutable
    {
      if (event->dispatch->clGetEventInfo(event, CL_EVENT_CONTEXT, sizeof(cl_context), &event_ctx, nullptr) != CL_SUCCESS)
        return false;

      if (event_ctx != parents.parent_context)
        return false;

      return event->is_valid();
    }
  );

  if (all_events_are_ours_and_valid && event)
  {
    reference();
    *event = lifetime::create_or_exit<cl_event>(
      nullptr,
      parents.parent_context,
      this
    );
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
        reinterpret_cast<char*>(parents.parent_devices.data()),
        reinterpret_cast<char*>(parents.parent_devices.data() + parents.parent_devices.size()),
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
  cl_mem_flags flags,
  size_t size,
  void*,
  cl_int* errcode_ret)
{
  reference();
  return lifetime::create_or_exit<cl_mem>(
    errcode_ret,
    nullptr,
    this,
    CL_MEM_OBJECT_BUFFER,
    flags,
    0,
    size
  );
}

cl_mem _cl_context::clCreateBufferWithProperties(
  const cl_mem_properties*,
  cl_mem_flags flags,
  size_t size,
  void*,
  cl_int* errcode_ret)
{
  reference();
  return lifetime::create_or_exit<cl_mem>(
    errcode_ret,
    nullptr,
    this,
    CL_MEM_OBJECT_BUFFER,
    flags,
    0,
    size
  );
}

cl_mem _cl_context::clCreateImage(
  cl_mem_flags flags,
  const cl_image_format* format,
  const cl_image_desc* desc,
  void*,
  cl_int* errcode_ret)
{
  reference();
  return lifetime::create_or_exit<cl_mem>(
    errcode_ret,
    nullptr,
    this,
    desc->image_type,
    flags,
    *format,
    *desc
  );
}

cl_mem _cl_context::clCreateImageWithProperties(
    const cl_mem_properties*,
    cl_mem_flags flags,
    const cl_image_format* format,
    const cl_image_desc* desc,
    void*,
    cl_int* errcode_ret)
{
  reference();
  if (desc)
  {
    if (desc->buffer)
    {
      auto it = std::find_if(
        lifetime::get_objects<cl_mem>().cbegin(),
        lifetime::get_objects<cl_mem>().cend(),
        [&](const std::shared_ptr<_cl_mem>& mem)
        {
          return mem->parents.parent_context == this;
        }
      );

      if (it != lifetime::get_objects<cl_mem>().cend())
      {
        (*it)->implicit_ref_count++;
        return lifetime::create_or_exit<cl_mem>(
          errcode_ret,
          (*it).get(),
          this,
          desc->image_type,
          flags,
          *format,
          *desc
        );
      }
    }
  }
  return lifetime::create_or_exit<cl_mem>(
    errcode_ret,
    nullptr,
    this,
    desc->image_type,
    flags,
    *format,
    *desc
  );
}

cl_mem _cl_context::clCreateImage2D(
  cl_mem_flags flags,
  const cl_image_format* format,
  size_t width,
  size_t height,
  size_t row_pitch,
  void*,
  cl_int* errcode_ret)
{
  reference();
  return lifetime::create_or_exit<cl_mem>(
    errcode_ret,
    nullptr,
    this,
    CL_MEM_OBJECT_IMAGE2D,
    flags,
    *format,
    cl_image_desc{
      CL_MEM_OBJECT_IMAGE2D,
      width,
      height,
      1,
      1,
      row_pitch,
      0,
      0,
      0,
      { nullptr }
    }
  );
}

cl_mem _cl_context::clCreateImage3D(
  cl_mem_flags flags,
  const cl_image_format* format,
  size_t width,
  size_t height,
  size_t depth,
  size_t row_pitch,
  size_t slice_pitch,
  void*,
  cl_int* errcode_ret)
{
  reference();
  return lifetime::create_or_exit<cl_mem>(
    errcode_ret,
    nullptr,
    this,
    CL_MEM_OBJECT_IMAGE3D,
    flags,
    *format,
    cl_image_desc{
      CL_MEM_OBJECT_IMAGE2D,
      width,
      height,
      depth,
      1,
      row_pitch,
      slice_pitch,
      0,
      0,
      { nullptr }
    }
  );
}

cl_mem _cl_context::clCreatePipe(
  cl_mem_flags flags,
  cl_uint,
  cl_uint,
  const cl_pipe_properties*,
  cl_int* errcode_ret)
{
  reference();
  return lifetime::create_or_exit<cl_mem>(
    errcode_ret,
    nullptr,
    this,
    CL_MEM_OBJECT_PIPE,
    flags,
    0,
    0
  );
}

cl_command_queue _cl_context::clCreateCommandQueue(
  cl_device_id device,
  cl_command_queue_properties properties,
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

  reference();
  return lifetime::create_or_exit<cl_command_queue>(
    errcode_ret,
    device,
    this,
    &properties,
    &properties + 1
  );
}

cl_command_queue _cl_context::clCreateCommandQueueWithProperties(
  cl_device_id device,
  const cl_queue_properties* properties,
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

  std::vector<cl_queue_properties> props;
  {
    const cl_queue_properties* it = properties;
    while(*it != 0)
      props.push_back(*it++);
  }

  auto CL_QUEUE_ON_DEVICE_DEFAULT_finder = [](
    std::vector<cl_queue_properties>::const_iterator first,
    std::vector<cl_queue_properties>::const_iterator last
  ) -> bool
  {
    auto it = std::find(first, last, CL_QUEUE_PROPERTIES);
    if (it == last)
      return false;
    else
      return *(++it) & CL_QUEUE_ON_DEVICE_DEFAULT;
  };

  if (CL_QUEUE_ON_DEVICE_DEFAULT_finder(props.cbegin(), props.cend()))
  {
    // From all queues on platform, find one which:
    //   1. points to the same device AND
    //   2. is in the same context AND
    //   3. has the property CL_QUEUE_ON_DEVICE_DEFAULT
    // If such queue is found, retain and return it
    // Else create new queue
    auto it = std::find_if(
      lifetime::get_objects<cl_command_queue>().cbegin(),
      lifetime::get_objects<cl_command_queue>().cend(),
      [&](const std::shared_ptr<_cl_command_queue>& queue)
      {
        bool points_to_same_device = queue->parents.parent_device == device;
        bool is_in_the_same_context = queue->parents.parent_context == this;
        bool has_prop_CL_QUEUE_ON_DEVICE_DEFAULT =
          CL_QUEUE_ON_DEVICE_DEFAULT_finder(queue->_props.cbegin(), queue->_props.cend());

        return points_to_same_device &&
          is_in_the_same_context &&
          has_prop_CL_QUEUE_ON_DEVICE_DEFAULT;
      }
    );

    if (it != lifetime::get_objects<cl_command_queue>().cend())
    {
      (*it)->retain();
      if (errcode_ret)
        *errcode_ret = CL_SUCCESS;
      return it->get();
    }
    else
    {
      reference();
      return lifetime::create_or_exit<cl_command_queue>(
        errcode_ret,
        device,
        this,
        props.data(),
        props.data() + props.size()
      );
    }
  }
  else
  {
    reference();
    return lifetime::create_or_exit<cl_command_queue>(
      errcode_ret,
      device,
      this,
      props.data(),
      props.data() + props.size()
    );
  }
}

cl_program _cl_context::clCreateProgramWithSource(
  cl_uint,
  const char**,
  const size_t*,
  cl_int* errcode_ret)
{
  reference();
  return lifetime::create_or_exit<cl_program>(
    errcode_ret,
    this,
    parents.parent_devices.data(),
    parents.parent_devices.data() + parents.parent_devices.size()
  );
}

cl_event _cl_context::clCreateUserEvent(
    cl_int* errcode_ret)
{
  reference();
  return lifetime::create_or_exit<cl_event>(
    errcode_ret,
    this,
    nullptr
  );
}

cl_sampler _cl_context::clCreateSampler(
    cl_bool,
    cl_addressing_mode,
    cl_filter_mode,
    cl_int* errcode_ret)
{
  reference();
  return lifetime::create_or_exit<cl_sampler>(
    errcode_ret,
    this
  );
}

cl_sampler _cl_context::clCreateSamplerWithProperties(
  const cl_sampler_properties*,
  cl_int* errcode_ret)
{
  reference();
  return lifetime::create_or_exit<cl_sampler>(
    errcode_ret,
    this
  );
}

cl_int _cl_context::clGetSupportedImageFormats(
  cl_mem_flags flags,
  cl_mem_object_type image_type,
  cl_uint num_entries,
  cl_image_format* image_formats,
  cl_uint* num_image_formats)
{
  (void) flags;
  (void) image_type;

  if (num_entries == 0 && image_formats != nullptr)
    return CL_INVALID_VALUE;

  std::vector<cl_image_format> supported_formats =
  {
    { CL_R,     CL_UNORM_INT8 },
    { CL_R,     CL_UNORM_INT8 },
    { CL_R,     CL_UNORM_INT16 },
    { CL_R,     CL_SNORM_INT8 },
    { CL_R,     CL_SNORM_INT16 },
    { CL_R,     CL_SIGNED_INT8 },
    { CL_R,     CL_SIGNED_INT16 },
    { CL_R,     CL_SIGNED_INT32 },
    { CL_R,     CL_UNSIGNED_INT8 },
    { CL_R,     CL_UNSIGNED_INT16 },
    { CL_R,     CL_UNSIGNED_INT32 },
    { CL_R,     CL_HALF_FLOAT },
    { CL_R,     CL_FLOAT },
    { CL_DEPTH, CL_UNORM_INT16 },
    { CL_DEPTH, CL_FLOAT },
    { CL_RG,    CL_UNORM_INT8 },
    { CL_RG,    CL_UNORM_INT16 },
    { CL_RG,    CL_SNORM_INT8 },
    { CL_RG,    CL_SNORM_INT16 },
    { CL_RG,    CL_SIGNED_INT8 },
    { CL_RG,    CL_SIGNED_INT16 },
    { CL_RG,    CL_SIGNED_INT32 },
    { CL_RG,    CL_UNSIGNED_INT8 },
    { CL_RG,    CL_UNSIGNED_INT16 },
    { CL_RG,    CL_UNSIGNED_INT32 },
    { CL_RG,    CL_HALF_FLOAT },
    { CL_RG,    CL_FLOAT },
    { CL_RGBA,  CL_UNORM_INT8 },
    { CL_RGBA,  CL_UNORM_INT16 },
    { CL_RGBA,  CL_SNORM_INT8 },
    { CL_RGBA,  CL_SNORM_INT16 },
    { CL_RGBA,  CL_SIGNED_INT8 },
    { CL_RGBA,  CL_SIGNED_INT16 },
    { CL_RGBA,  CL_SIGNED_INT32 },
    { CL_RGBA,  CL_UNSIGNED_INT8 },
    { CL_RGBA,  CL_UNSIGNED_INT16 },
    { CL_RGBA,  CL_UNSIGNED_INT32 },
    { CL_RGBA,  CL_HALF_FLOAT },
    { CL_RGBA,  CL_FLOAT },
    { CL_BGRA,  CL_UNORM_INT8 },
    { CL_sRGBA, CL_UNORM_INT8 },
  };

  if (num_image_formats)
    *num_image_formats = static_cast<cl_uint>(supported_formats.size());

  if (image_formats)
  {
    std::copy(
      supported_formats.begin(),
      supported_formats.begin() + std::min(static_cast<size_t>(num_entries), supported_formats.size()),
      image_formats
    );
  }

  return CL_SUCCESS;
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
        reinterpret_cast<char*>(&parents.parent_context) + sizeof(parents.parent_context),
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
        reinterpret_cast<char*>(parents.parent_devices.data()),
        reinterpret_cast<char*>(parents.parent_devices.data() + parents.parent_devices.size()),
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
  reference();
  return lifetime::create_or_exit<cl_kernel>(
    errcode_ret,
    this
  );
}

cl_int _cl_program::clCreateKernelsInProgram(
  cl_uint num_kernels,
  cl_kernel* kernels,
  cl_uint* num_kernels_ret)
{
  // The proper counting of kernels requires a C compiler. kernel entry points
  // may be declared/defined both, they may be '__kernel' or 'kernel' and be
  // subjected to the preprocessor.
  static constexpr cl_uint kernel_count = 3;

  if (num_kernels_ret)
    *num_kernels_ret = kernel_count;

  if (kernels && num_kernels < kernel_count)
    return CL_INVALID_VALUE;

  if(kernels)
  {
    std::vector<std::shared_ptr<_cl_kernel>> result;
    std::generate_n(
      std::back_inserter(result),
      3,
      [=](){ return std::make_shared<_cl_kernel>(this); }
    );

    std::copy(
      result.cbegin(),
      result.cend(),
      std::inserter(
        lifetime::get_objects<cl_kernel>(),
        lifetime::get_objects<cl_kernel>().end()
      )
    );

    std::transform(
      result.cbegin(),
      result.cend(),
      kernels,
      [](const std::shared_ptr<_cl_kernel>& kernel){ return kernel.get(); }
    );
    reference(kernel_count);
  }

  return CL_SUCCESS;
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
        reinterpret_cast<char*>(&parents.parent_program) + sizeof(parents.parent_program),
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

cl_int _cl_kernel::clSetKernelArg(
  cl_uint,
  size_t,
  const void*)
{
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
  return lifetime::create_or_exit<cl_kernel>(
    errcode_ret,
    parents.parent_program
  );
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
          reinterpret_cast<char*>(&parents.parent_queue) + sizeof(parents.parent_queue),
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

cl_int _cl_event::clWaitForEvents(
  cl_uint num_events,
  const cl_event* event_list)
{
  bool all_events_are_in_the_same_context = std::all_of(
    event_list,
    event_list + num_events,
    [this_ctx = parents.parent_context](const cl_event& event)
    {
      cl_context event_ctx = nullptr;
      if (event->dispatch->clGetEventInfo(event, CL_EVENT_CONTEXT, sizeof(cl_context), &event_ctx, nullptr) != CL_SUCCESS)
        return false;

      return event_ctx == this_ctx;
    }
  );

  if (!all_events_are_in_the_same_context)
    return CL_INVALID_CONTEXT;

  bool all_events_are_valid = std::all_of(
    event_list,
    event_list + num_events,
    [](const cl_event& event){ return event->is_valid(); }
  );

  if (all_events_are_valid)
    return CL_SUCCESS;
  else
    return CL_INVALID_EVENT;
}

cl_int _cl_event::clSetUserEventStatus(
  cl_int execution_status)
{
  if (parents.parent_queue != nullptr)
    return CL_INVALID_EVENT;

  if (execution_status != CL_COMPLETE ||
      execution_status < 0)
    return CL_INVALID_VALUE;

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
  , _devices{}
  , _contexts{}
  , _mems{}
  , _queues{}
  , _programs{}
  , _kernels{}
  , _events{}
  , _samplers{}
  , _global_mutex{}
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

  std::string ALWAYS_RETURN_SUCCESS;
  if (ocl_layer_utils::detail::get_environment("ALWAYS_RETURN_SUCCESS", ALWAYS_RETURN_SUCCESS))
  {
    always_return_success = true;
  }
  else
    always_return_success = false;

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
  dispatch->clCreateBufferWithProperties = clCreateBufferWithProperties_wrap;
  dispatch->clCreateImage = clCreateImage_wrap;
  dispatch->clCreateImageWithProperties = clCreateImageWithProperties_wrap;
  dispatch->clCreateImage2D = clCreateImage2D_wrap;
  dispatch->clCreateImage3D = clCreateImage3D_wrap;
  dispatch->clCreatePipe = clCreatePipe_wrap;
  dispatch->clCreateCommandQueue = clCreateCommandQueue_wrap;
  dispatch->clCreateCommandQueueWithProperties = clCreateCommandQueueWithProperties_wrap;
  dispatch->clCreateSubBuffer = clCreateSubBuffer_wrap;
  dispatch->clRetainMemObject = clRetainMemObject_wrap;
  dispatch->clReleaseMemObject = clReleaseMemObject_wrap;
  dispatch->clGetMemObjectInfo = clGetMemObjectInfo_wrap;
  dispatch->clGetCommandQueueInfo = clGetCommandQueueInfo_wrap;
  dispatch->clEnqueueNDRangeKernel = clEnqueueNDRangeKernel_wrap;
  dispatch->clRetainCommandQueue = clRetainCommandQueue_wrap;
  dispatch->clReleaseCommandQueue = clReleaseCommandQueue_wrap;
  dispatch->clCreateProgramWithSource = clCreateProgramWithSource_wrap;
  dispatch->clBuildProgram = clBuildProgram_wrap;
  dispatch->clGetProgramInfo = clGetProgramInfo_wrap;
  dispatch->clRetainProgram = clRetainProgram_wrap;
  dispatch->clReleaseProgram = clReleaseProgram_wrap;
  dispatch->clCreateKernel = clCreateKernel_wrap;
  dispatch->clCreateKernelsInProgram = clCreateKernelsInProgram_wrap;
  dispatch->clSetKernelArg = clSetKernelArg_wrap;
  dispatch->clCloneKernel = clCloneKernel_wrap;
  dispatch->clGetKernelInfo = clGetKernelInfo_wrap;
  dispatch->clRetainKernel = clRetainKernel_wrap;
  dispatch->clReleaseKernel = clReleaseKernel_wrap;
  dispatch->clCreateUserEvent = clCreateUserEvent_wrap;
  dispatch->clSetUserEventStatus = clSetUserEventStatus_wrap;
  dispatch->clGetEventInfo = clGetEventInfo_wrap;
  dispatch->clWaitForEvents = clWaitForEvents_wrap;
  dispatch->clRetainEvent = clRetainEvent_wrap;
  dispatch->clReleaseEvent = clReleaseEvent_wrap;
  dispatch->clCreateSampler = clCreateSampler_wrap;
  dispatch->clCreateSamplerWithProperties = clCreateSamplerWithProperties_wrap;
  dispatch->clGetSamplerInfo = clGetSamplerInfo_wrap;
  dispatch->clRetainSampler = clRetainSampler_wrap;
  dispatch->clReleaseSampler = clReleaseSampler_wrap;
  dispatch->clEnqueueFillBuffer = clEnqueueFillBuffer_wrap;
  dispatch->clEnqueueCopyImageToBuffer = clEnqueueCopyImageToBuffer_wrap;
  dispatch->clEnqueueCopyImage = clEnqueueCopyImage_wrap;
  dispatch->clGetSupportedImageFormats = clGetSupportedImageFormats_wrap;
  dispatch->clGetImageInfo = clGetImageInfo_wrap;
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
        lifetime::get_objects<cl_device_id>().cbegin(),
        lifetime::get_objects<cl_device_id>().cend(),
        [](const std::shared_ptr<_cl_device_id>& dev)
        {
          return dev->kind == _cl_device_id::device_kind::root;
        })) :
       0;

  if(devices && asking_for_custom)
  {
    std::vector<std::shared_ptr<_cl_device_id>> result;
    std::copy_if(
      lifetime::get_objects<cl_device_id>().cbegin(),
      lifetime::get_objects<cl_device_id>().cend(),
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
