// auxilary functions

bool is_3D_image_fits(
  const cl_image_desc * const image_desc,
  cl_context context)
{
  std::vector<cl_device_id> devices = get_devices(context);
  size_t nd = devices.size();

  size_t width, height, depth;
  for (size_t i = 0; i < nd; ++i)
  {
    tdispatch->clGetDeviceInfo(
      devices[i],
      CL_DEVICE_IMAGE3D_MAX_WIDTH,
      sizeof(size_t),
      &width,
      NULL);
    tdispatch->clGetDeviceInfo(
      devices[i],
      CL_DEVICE_IMAGE3D_MAX_HEIGHT,
      sizeof(size_t),
      &height,
      NULL);
    tdispatch->clGetDeviceInfo(
      devices[i],
      CL_DEVICE_IMAGE3D_MAX_DEPTH,
      sizeof(size_t),
      &depth,
      NULL);
    if ((image_desc->image_width  <= width ) &&
        (image_desc->image_height <= height) &&
        (image_desc->image_depth  <= depth ))
      return true;
  }

  return false;
}

bool is_2D_image_fits(
  const cl_image_desc * const image_desc,
  cl_context context)
{
  std::vector<cl_device_id> devices = get_devices(context);
  size_t nd = devices.size();

  size_t width, height;
  for (size_t i = 0; i < nd; ++i)
  {
    tdispatch->clGetDeviceInfo(
      devices[i],
      CL_DEVICE_IMAGE2D_MAX_WIDTH,
      sizeof(size_t),
      &width,
      NULL);
    tdispatch->clGetDeviceInfo(
      devices[i],
      CL_DEVICE_IMAGE2D_MAX_HEIGHT,
      sizeof(size_t),
      &height,
      NULL);
    if ((image_desc->image_width  <= width ) &&
        (image_desc->image_height <= height))
      return true;
  }

  return false;
}

bool is_1D_image_fits(
  const cl_image_desc * const image_desc,
  cl_context context)
{
  std::vector<cl_device_id> devices = get_devices(context);
  size_t nd = devices.size();

  size_t width;
  for (size_t i = 0; i < nd; ++i)
  {
    tdispatch->clGetDeviceInfo(
      devices[i],
      CL_DEVICE_IMAGE2D_MAX_WIDTH,
      sizeof(size_t),
      &width,
      NULL);
    if (image_desc->image_width <= width)
      return true;
  }

  return false;
}

bool is_2D_array_fits(
  const cl_image_desc * const image_desc,
  cl_context context)
{
  std::vector<cl_device_id> devices = get_devices(context);
  size_t nd = devices.size();

  size_t width, height, size;
  for (size_t i = 0; i < nd; ++i)
  {
    tdispatch->clGetDeviceInfo(
      devices[i],
      CL_DEVICE_IMAGE2D_MAX_WIDTH,
      sizeof(size_t),
      &width,
      NULL);
    tdispatch->clGetDeviceInfo(
      devices[i],
      CL_DEVICE_IMAGE2D_MAX_HEIGHT,
      sizeof(size_t),
      &height,
      NULL);
    tdispatch->clGetDeviceInfo(
      devices[i],
      CL_DEVICE_IMAGE_MAX_ARRAY_SIZE,
      sizeof(size_t),
      &size,
      NULL);
    if ((image_desc->image_width      <= width ) &&
        (image_desc->image_height     <= height) &&
        (image_desc->image_array_size <= size  ))
      return true;
  }

  return false;
}

bool is_1D_array_fits(
  const cl_image_desc * const image_desc,
  cl_context context)
{
  std::vector<cl_device_id> devices = get_devices(context);
  size_t nd = devices.size();

  size_t width, size;
  for (size_t i = 0; i < nd; ++i)
  {
    tdispatch->clGetDeviceInfo(
      devices[i],
      CL_DEVICE_IMAGE2D_MAX_WIDTH,
      sizeof(size_t),
      &width,
      NULL);
    tdispatch->clGetDeviceInfo(
      devices[i],
      CL_DEVICE_IMAGE_MAX_ARRAY_SIZE,
      sizeof(size_t),
      &size,
      NULL);
    if ((image_desc->image_width      <= width) &&
        (image_desc->image_array_size <= size ))
      return true;
  }

  return false;
}

bool is_1D_buffer_fits(
  const cl_image_desc * const image_desc,
  cl_context context)
{
  std::vector<cl_device_id> devices = get_devices(context);
  size_t nd = devices.size();

  size_t width;
  for (size_t i = 0; i < nd; ++i)
  {
    tdispatch->clGetDeviceInfo(
      devices[i],
      CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
      sizeof(size_t),
      &width,
      NULL);
    if (image_desc->image_width <= width)
      return true;
  }

  return false;
}

cl_uint max_pitch_al(cl_context context)
{
  std::vector<cl_device_id> devices = get_devices(context);
  size_t nd = devices.size();

  cl_uint res = 0;

  // find maximum row pitch alignment size in pixels for 2D images
  // created from a buffer for all devices in the context
  cl_uint size;
  for (size_t i = 0; i < nd; ++i)
  {
    tdispatch->clGetDeviceInfo( // give 0 for devices not supporting such image creation
      devices[i],
      CL_DEVICE_IMAGE_PITCH_ALIGNMENT,
      sizeof(cl_uint),
      &size,
      NULL);
    if (size > res)
      res = size;
  }

  return res;
}

cl_uint max_base_al(cl_context context)
{
  std::vector<cl_device_id> devices = get_devices(context);
  size_t nd = devices.size();

  cl_uint res = 0;

  // find maximum base address alignment size in pixels for 2D images
  // created from a buffer for all devices in the context
  cl_uint size;
  for (size_t i = 0; i < nd; ++i)
  {
    tdispatch->clGetDeviceInfo( // give 0 for devices not supporting such image creation
      devices[i],
      CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT,
      sizeof(cl_uint),
      &size,
      NULL);
    if (size > res)
      res = size;
  }

  return res;
}

size_t pixel_size(const cl_image_format * image_format)
{
  size_t channels = 0;
  switch (image_format->image_channel_order) {
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

// For 3 channels we can safely use 4 for pixel_size as all supported data types
// are powers of 2 in bit size (see below) and we are interesed in
// the next or equal power of two of (channels * channel_size)*8 bits:
// "The number of bits per element determined by the image_channel_data_type
// and image_channel_order must be a power of two."
    case CL_RGB:
    case CL_RGx:
    case CL_sRGB:
      channels = 4;
      break;

    case CL_RGBA:
    case CL_ARGB:
    case CL_BGRA:
    case CL_ABGR:
    case CL_RGBx:
    case CL_sRGBA:
    case CL_sBGRA:
    case CL_sRGBx:
      channels = 4;
      break;

    default:
      printf("Wrong image channel order!\n");
      return 0;
  }

  size_t channel_size = 0;
  switch (image_format->image_channel_data_type) {
    case CL_SNORM_INT8:
    case CL_UNORM_INT8:
    case CL_SIGNED_INT8:
    case CL_UNSIGNED_INT8:
      channel_size = 1;
      break;

    case CL_SNORM_INT16:
    case CL_UNORM_INT16:
    case CL_UNORM_SHORT_565:
    case CL_UNORM_SHORT_555:
    case CL_SIGNED_INT16:
    case CL_UNSIGNED_INT16:
    case CL_HALF_FLOAT:
      channel_size = 2;
      break;

    case CL_UNORM_INT_101010:
    case CL_UNORM_INT_101010_2:
    case CL_SIGNED_INT32:
    case CL_UNSIGNED_INT32:
    case CL_FLOAT:
      channel_size = 4;
      break;

    default:
      printf("Wrong image channel data type!\n");
      return 0;
  }

  return channels * channel_size;
}

size_t pixel_size(const cl_mem image)
{
  cl_image_format format;
  tdispatch->clGetImageInfo(
    image,
    CL_IMAGE_FORMAT,
    sizeof(cl_image_format),
    &format,
    NULL);

  return pixel_size(&format);
}

// check if the descriptor of image being created is compatible with
// one of the image from which the current image is being created
bool is_compatible_image(const cl_image_desc * const image_desc, const cl_image_format * image_format)
{
  cl_image_format format;
  tdispatch->clGetImageInfo(
    image_desc->mem_object,
    CL_IMAGE_FORMAT,
    sizeof(cl_image_format),
    &format,
    NULL);

  if (format.image_channel_data_type != image_format->image_channel_data_type)
    return false;
  // image channel order must be compatible
  switch (image_format->image_channel_order) {
    case CL_sBGRA:
      if (format.image_channel_order != CL_BGRA)
        return false;
      break;

    case CL_BGRA:
      if (format.image_channel_order != CL_sBGRA)
        return false;
      break;

    case CL_sRGBA:
      if (format.image_channel_order != CL_RGBA)
        return false;
      break;

    case CL_RGBA:
      if (format.image_channel_order != CL_sRGBA)
        return false;
      break;

    case CL_sRGB:
      if (format.image_channel_order != CL_RGB)
        return false;
      break;

    case CL_RGB:
      if (format.image_channel_order != CL_sRGB)
        return false;
      break;

    case CL_sRGBx:
      if (format.image_channel_order != CL_RGBx)
        return false;
      break;

    case CL_RGBx:
      if (format.image_channel_order != CL_sRGBx)
        return false;
      break;

    case CL_DEPTH:
      if (format.image_channel_order != CL_R)
        return false;
      break;

    default:
      return false;
  }

  size_t size = 0;

  tdispatch->clGetImageInfo(
    image_desc->mem_object,
    CL_IMAGE_WIDTH,
    sizeof(size_t),
    &size,
    NULL);
  if (size != image_desc->image_width)
    return false;

  tdispatch->clGetImageInfo(
    image_desc->mem_object,
    CL_IMAGE_HEIGHT,
    sizeof(size_t),
    &size,
    NULL);
  if (size != image_desc->image_height)
    return false;

  tdispatch->clGetImageInfo(
    image_desc->mem_object,
    CL_IMAGE_DEPTH,
    sizeof(size_t),
    &size,
    NULL);
  if (size != 0)
    return false;

  tdispatch->clGetImageInfo(
    image_desc->mem_object,
    CL_IMAGE_ARRAY_SIZE,
    sizeof(size_t),
    &size,
    NULL);
  if (size != 0)
    return false;

  size_t image_row_pitch = image_desc->image_width * pixel_size(image_format);
  if ((image_desc->image_row_pitch < image_row_pitch) &&
    (image_desc->image_row_pitch != 0))
    return false;
  if (image_desc->image_row_pitch != 0)
    image_row_pitch = image_desc->image_row_pitch;
  tdispatch->clGetImageInfo(
    image_desc->mem_object,
    CL_IMAGE_ROW_PITCH,
    sizeof(size_t),
    &size,
    NULL);
  if (size != image_row_pitch)
    return false;

  // num_mip_levels and num_samples are always the same and = 0 - no need to check
  // mem_object can be different

  return true;
}

size_t buffer_size(cl_mem buffer)
{
  size_t size = 0;
  tdispatch->clGetMemObjectInfo(
    buffer,
    CL_MEM_SIZE,
    sizeof(size_t),
    &size,
    NULL);
  return size;
}

////////////////////
// main functions //
////////////////////

// the order of function invocations should follow the XML and is important
// as functions rely on the correctness of objects checked previously

// 5.2.1

// check validity of structure
bool struct_violation(
  cl_version,
  const void * buffer_create_info,
  cl_buffer_create_type buffer_create_type)
{
  (void)buffer_create_info;

  if (buffer_create_type == CL_BUFFER_CREATE_TYPE_REGION) {
    //const cl_buffer_region * sb = static_cast<const cl_buffer_region *>(buffer_create_info);
    // no possible violations
    return false;
  }

  return true;
}

// check if out-of-bounds
bool struct_violation(
  cl_version,
  const void * buffer_create_info,
  cl_mem buffer)
{
  const cl_buffer_region * sb = static_cast<const cl_buffer_region *>(buffer_create_info);
  size_t buffer_size;
  tdispatch->clGetMemObjectInfo(
    buffer,
    CL_MEM_SIZE,
    sizeof(size_t),
    &buffer_size,
    NULL);
  if (sb->origin + sb->size > buffer_size)
    return true;

  return false;
}

// check if size = 0
bool struct_violation(
  cl_version,
  const void * buffer_create_info)
{
  const cl_buffer_region * sb = static_cast<const cl_buffer_region *>(buffer_create_info);
  if (sb->size == 0)
    return true;

  return false;
}

// check if there are no devices in context associated with buffer
// for which the origin field of the cl_buffer_region structure
// passed in buffer_create_info is aligned to the CL_DEVICE_MEM_BASE_ADDR_ALIGN value
bool struct_violation(
  cl_version,
  const void * buffer_create_info,
  cl_buffer_create_type buffer_create_type,
  cl_mem buffer)
{
  if (buffer_create_type == CL_BUFFER_CREATE_TYPE_REGION) {
    cl_context context;
    tdispatch->clGetMemObjectInfo(
      buffer,
      CL_MEM_CONTEXT,
      sizeof(cl_context),
      &context,
      NULL);

    const cl_buffer_region * sb = static_cast<const cl_buffer_region *>(buffer_create_info);
    return for_all<CL_DEVICE_MEM_BASE_ADDR_ALIGN>(context, [sb](cl_uint align) {
      return (sb->origin % align != 0);
    });
  }

  return true;
}

// 5.3.1.1. Image Format Descriptor

// check image_format violation
bool struct_violation(
  cl_version version,
  const cl_image_format * const image_format)
{
  if (enum_violation(version, "cl_channel_order", image_format->image_channel_order))
    return true;
  if (enum_violation(version, "cl_channel_type", image_format->image_channel_data_type))
    return true;

  if (((image_format->image_channel_data_type == CL_UNORM_SHORT_555) ||
     (image_format->image_channel_data_type == CL_UNORM_SHORT_565) ||
     (image_format->image_channel_data_type == CL_UNORM_INT_101010)) &&
    !((image_format->image_channel_order == CL_RGB) || (image_format->image_channel_order == CL_RGBx)))
    return true;
  if ((image_format->image_channel_data_type == CL_UNORM_INT_101010_2) &&
    !(image_format->image_channel_order == CL_RGBA))
    return true;

  return false;
}

// check correctness of 2D image creation from buffer
bool struct_violation(
  cl_version,
  const cl_image_format * const image_format,
  cl_context context,
  const cl_image_desc * const image_desc)
{
  // if 2D image is created from the buffer
  if ((image_desc->image_type == CL_MEM_OBJECT_IMAGE2D) &&
    (image_desc->mem_object != NULL) &&
    object_is_valid(image_desc->mem_object, CL_MEM_OBJECT_BUFFER))
  {
    // calculate image_row_pitch and check if it is too low
    size_t image_row_pitch = image_desc->image_width * pixel_size(image_format);
    if ((image_desc->image_row_pitch < image_row_pitch) &&
      (image_desc->image_row_pitch != 0))
      return true;
    if (image_desc->image_row_pitch != 0)
      image_row_pitch = image_desc->image_row_pitch;
    // check if it is multiple of a pixel
    if (image_row_pitch % pixel_size(image_format) != 0)
      return true;
    // check if aligned properly
    if (image_row_pitch % max_pitch_al(context) != 0)
      return true;

    // check if base pointer is aligned properly
    cl_mem_flags flags;
    tdispatch->clGetMemObjectInfo(
      image_desc->buffer,
      CL_MEM_FLAGS,
      sizeof(cl_mem_flags),
      &flags,
      NULL);
    if (flags | CL_MEM_USE_HOST_PTR)
    {
      void * host_ptr;
      tdispatch->clGetMemObjectInfo(
        image_desc->buffer,
        CL_MEM_HOST_PTR,
        sizeof(void *),
        &host_ptr,
        NULL);
      //if ((uintptr_t)host_ptr % max_base_al(context) != 0)
      size_t space = max_base_al(context);
      if (std::align(space, space, host_ptr, space) == nullptr)
        return true;
    }
  }

  return false;
}

// check correctness of 2D image creation from 2D image
bool struct_violation(
  cl_version,
  const cl_image_format * const image_format,
  const cl_image_desc * const image_desc)
{
  // if 2D image is created from 2D image
  if ((image_desc->image_type == CL_MEM_OBJECT_IMAGE2D) &&
    (image_desc->mem_object != NULL) &&
    object_is_valid(image_desc->mem_object, CL_MEM_OBJECT_IMAGE2D))
  {
    if (!is_compatible_image(image_desc, image_format))
      return true;
  }

  return false;
}

// check if there are no devices that support image_format in the context
bool struct_violation(
  cl_version,
  const cl_image_format * const image_format,
  cl_context context,
  cl_mem_flags flags,
  const cl_image_desc * const image_desc)
{
  cl_uint num_image_format = 0;
  tdispatch->clGetSupportedImageFormats(
    context,
    flags & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_KERNEL_READ_AND_WRITE),
    image_desc->image_type,
    0,
    NULL,
    &num_image_format);
  std::vector<cl_image_format> image_formats(num_image_format);
  tdispatch->clGetSupportedImageFormats(
    context,
    flags & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_KERNEL_READ_AND_WRITE),
    image_desc->image_type,
    num_image_format,
    image_formats.data(),
    NULL);

  for (auto a : image_formats) {
    if ((image_format->image_channel_order == a.image_channel_order) &&
        (image_format->image_channel_data_type == a.image_channel_data_type))
      return false;
  }

  return true;
}

// check if there are no devices that support image_format in the context
// for clCreateImage2D and clCreateImage3D
bool struct_violation(
  cl_version,
  const cl_image_format * const image_format,
  cl_context context,
  cl_mem_flags flags,
  cl_mem_object_type image_type)
{
  cl_uint num_image_format = 0;
  tdispatch->clGetSupportedImageFormats(
    context,
    flags & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_KERNEL_READ_AND_WRITE),
    image_type,
    0,
    NULL,
    &num_image_format);
  std::vector<cl_image_format> image_formats(num_image_format);
  tdispatch->clGetSupportedImageFormats(
    context,
    flags & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_KERNEL_READ_AND_WRITE),
    image_type,
    num_image_format,
    image_formats.data(),
    NULL);

  for (auto a : image_formats) {
    if ((image_format->image_channel_order == a.image_channel_order) &&
        (image_format->image_channel_data_type == a.image_channel_data_type))
      return false;
  }

  return true;
}

// check if 2D image does not fit
bool struct_violation(
  cl_version,
  cl_context context,
  size_t image_width,
  size_t image_height)
{
  cl_image_desc a;
  a.image_width = image_width;
  a.image_height = image_height;

  return !is_2D_image_fits(&a, context);
}

// check if 3D image does not fit
bool struct_violation(
  cl_version,
  cl_context context,
  size_t image_width,
  size_t image_height,
  size_t image_depth)
{
  cl_image_desc a;
  a.image_width = image_width;
  a.image_height = image_height;
  a.image_depth = image_depth;

  return !is_3D_image_fits(&a, context);
}

// 5.3.1.2. Image Descriptor

// check all besides checked below
bool struct_violation(
  cl_version,
  const cl_image_desc * const image_desc,
  const cl_image_format * image_format, 
  void * host_ptr)
{
  // check image sizes (upper limits are checked in next function)
  if (image_desc->image_type == CL_MEM_OBJECT_IMAGE3D)
    if (image_desc->image_depth == 0)
      return true;

  if (image_desc->image_type == CL_MEM_OBJECT_IMAGE3D ||
      image_desc->image_type == CL_MEM_OBJECT_IMAGE2D ||
      image_desc->image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY)
    if (image_desc->image_height == 0)
      return true;

  if (image_desc->image_type == CL_MEM_OBJECT_IMAGE3D ||
      image_desc->image_type == CL_MEM_OBJECT_IMAGE2D ||
      image_desc->image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY ||
      image_desc->image_type == CL_MEM_OBJECT_IMAGE1D ||
      image_desc->image_type == CL_MEM_OBJECT_IMAGE1D_BUFFER ||
      image_desc->image_type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
  {
    if (image_desc->image_width == 0)
      return true;
  }
  else // wrong image type
    return true;

  // check array size
  if (image_desc->image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY ||
      image_desc->image_type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
    if (image_desc->image_array_size == 0)
      return true;

  // check image_row_pitch
  if ((host_ptr == NULL) && (image_desc->image_row_pitch != 0))
    return true;
  if ((host_ptr != NULL) && (image_desc->image_row_pitch > 1) &&
    (image_desc->image_row_pitch < image_desc->image_width * pixel_size(image_format)))
    return true;
  if (image_desc->image_row_pitch % pixel_size(image_format) != 0)
    return true;
  size_t image_row_pitch =
    std::max(image_desc->image_row_pitch, image_desc->image_width * pixel_size(image_format));

  // check image_slice_pitch
  if ((host_ptr == NULL) && (image_desc->image_slice_pitch != 0))
    return true;
  if (host_ptr != NULL)
    switch (image_desc->image_type) {
      case CL_MEM_OBJECT_IMAGE3D:
      case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        if ((image_desc->image_slice_pitch > 1) && 
            (image_desc->image_slice_pitch < image_row_pitch * image_desc->image_height))
          return true;
        break;

      case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        if ((image_desc->image_slice_pitch > 1) && 
            (image_desc->image_slice_pitch < image_row_pitch))
          return true;
        break;
    }
  if (image_desc->image_slice_pitch % image_row_pitch != 0)
    return true;

  // check image creation from buffer or other image
  switch (image_desc->image_type) {
    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
      if (image_desc->mem_object != NULL)
        if (!object_is_valid(image_desc->mem_object, CL_MEM_OBJECT_BUFFER) ||
            (buffer_size(image_desc->mem_object) < image_desc->image_width * pixel_size(image_format)))
          return true;
      break;

    case CL_MEM_OBJECT_IMAGE2D:
      if (image_desc->mem_object != NULL)
        if ((!object_is_valid(image_desc->mem_object, CL_MEM_OBJECT_BUFFER) ||
            (buffer_size(image_desc->mem_object) < image_row_pitch * image_desc->image_height)) &&
            (!object_is_valid(image_desc->mem_object, CL_MEM_OBJECT_IMAGE2D)))
          return true;
      break;

    default:
      if (image_desc->mem_object != NULL)
        return true;
  }

  if ((image_desc->num_mip_levels != 0) || (image_desc->num_samples != 0))
    return true;

  return false;
}

// check image sizes to fit into some device of the context
bool struct_violation(
  cl_version,
  const cl_image_desc * const image_desc, 
  cl_context context)
{
  switch (image_desc->image_type) {
    case CL_MEM_OBJECT_IMAGE3D:
      return !is_3D_image_fits(image_desc, context);

    case CL_MEM_OBJECT_IMAGE2D:
      return !is_2D_image_fits(image_desc, context);

    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
      return !is_2D_array_fits(image_desc, context);

    case CL_MEM_OBJECT_IMAGE1D:
      return !is_1D_image_fits(image_desc, context);

    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
      return !is_1D_array_fits(image_desc, context);

    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
      return !is_1D_buffer_fits(image_desc, context);

    default: // unknown type
      return true;
  }
}

// check memory flags
bool struct_violation(
  cl_version,
  const cl_image_desc * const image_desc,
  cl_mem_flags flags)
{
  // check image creation from buffer or other image
  switch (image_desc->image_type) {
    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
    case CL_MEM_OBJECT_IMAGE2D:
      if (image_desc->mem_object != NULL) {
        cl_mem_flags old_flags;
        tdispatch->clGetMemObjectInfo(
          image_desc->mem_object,
          CL_MEM_FLAGS,
          sizeof(cl_mem_flags),
          &old_flags,
          NULL);

        if ((old_flags | CL_MEM_WRITE_ONLY) && 
            ((flags | CL_MEM_READ_WRITE) || (flags | CL_MEM_READ_ONLY)))
          return true;
        if ((old_flags | CL_MEM_READ_ONLY) && 
            ((flags | CL_MEM_READ_WRITE) || (flags | CL_MEM_WRITE_ONLY)))
          return true;
        if ((flags | CL_MEM_USE_HOST_PTR) || 
            (flags | CL_MEM_ALLOC_HOST_PTR) ||
            (flags | CL_MEM_COPY_HOST_PTR))
          return true;
        if ((old_flags | CL_MEM_HOST_WRITE_ONLY) && 
            (flags | CL_MEM_HOST_READ_ONLY))
          return true;
        if ((old_flags | CL_MEM_HOST_READ_ONLY) && 
            (flags | CL_MEM_HOST_WRITE_ONLY))
          return true;
        if ((old_flags | CL_MEM_HOST_NO_ACCESS) && 
            ((flags | CL_MEM_HOST_READ_ONLY) || (flags | CL_MEM_HOST_WRITE_ONLY)))
          return true;
      }
  }

  return false;
}

// check size of host_ptr
// rely on correctness of image_row_pitch and image_slice_pitch
// sizes from Table 15 of OpenCL 3.0 specification are used
bool struct_violation(
  cl_version,
  const cl_image_desc * const image_desc, 
  void * host_ptr,
  const cl_image_format * image_format)
{
  if (host_ptr != NULL) {
    size_t image_row_pitch = 
      std::max(image_desc->image_row_pitch, 
        image_desc->image_width * pixel_size(image_format));
/*    size_t image_line_size = 
      image_desc->image_width * pixel_size(image_format);

    switch (image_desc->image_type) {
      case CL_MEM_OBJECT_IMAGE1D:
      case CL_MEM_OBJECT_IMAGE1D_BUFFER:
        return array_len_ls(host_ptr, image_line_size);

      case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        return array_len_ls(host_ptr, 
          (image_desc->image_array_size - 1) * 
            std::max(image_desc->image_slice_pitch, image_row_pitch)
          + image_line_size);

      case CL_MEM_OBJECT_IMAGE2D:
        return array_len_ls(host_ptr, 
          (image_desc->image_height - 1) * image_row_pitch
          + image_line_size);

      case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        return array_len_ls(host_ptr, 
          (image_desc->image_array_size - 1) * 
            std::max(image_desc->image_slice_pitch, 
              image_desc->image_height * image_row_pitch)
          + (image_desc->image_height - 1) * image_row_pitch
          + image_line_size);

      case CL_MEM_OBJECT_IMAGE3D:
        return array_len_ls(host_ptr, 
          (image_desc->image_depth - 1) * 
            std::max(image_desc->image_slice_pitch, 
              image_row_pitch * image_desc->image_height)
          + (image_desc->image_height - 1) * image_row_pitch
          + image_line_size);

      default:
        return true;
    }
*/

    switch (image_desc->image_type) {
      case CL_MEM_OBJECT_IMAGE1D:
      case CL_MEM_OBJECT_IMAGE1D_BUFFER:
        return array_len_ls(host_ptr, image_row_pitch);

      case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        return array_len_ls(host_ptr, 
          image_desc->image_array_size * 
            std::max(image_desc->image_slice_pitch, image_row_pitch));

      case CL_MEM_OBJECT_IMAGE2D:
        return array_len_ls(host_ptr, 
          image_desc->image_height * image_row_pitch);

      case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        return array_len_ls(host_ptr, 
          image_desc->image_array_size * 
            std::max(image_desc->image_slice_pitch, 
              image_desc->image_height * image_row_pitch));

      case CL_MEM_OBJECT_IMAGE3D:
        return array_len_ls(host_ptr, 
          image_desc->image_depth * 
            std::max(image_desc->image_slice_pitch, 
              image_row_pitch * image_desc->image_height));

      default:
        return true;
    }
  }
  return false;
}

// check impossibility of 2D image creation from a buffer
bool struct_violation(
  cl_version version,
  const cl_image_format * const,
  const cl_image_desc * const image_desc,
  cl_context context)
{
  // if 2D image is created from the buffer
  if ((image_desc->image_type == CL_MEM_OBJECT_IMAGE2D) &&
    (image_desc->mem_object != NULL) &&
    object_is_valid(image_desc->mem_object, CL_MEM_OBJECT_BUFFER))
  {
    // before 2.0 there is no possibility to do it
    if (CL_VERSION_MAJOR(version) < 2)
      return true;
    // check if all devices in the context does not support creation
    return for_all<CL_DEVICE_IMAGE_PITCH_ALIGNMENT>(context, 
      [=](return_type<CL_DEVICE_IMAGE_PITCH_ALIGNMENT> query)
      { return query == 0; });
  }

  return false;
}

// 5.3.3

// check if image format for image is not supported by device associated with queue
bool struct_violation(
  cl_version,
  cl_mem image,
  cl_command_queue queue)
{
  bool res = true;

  cl_context c;
  tdispatch->clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(c), &c, NULL);

  if (c != NULL)
  {
    cl_mem_flags fl = 0;
    tdispatch->clGetMemObjectInfo(image, CL_MEM_FLAGS, sizeof(fl), &fl, NULL);
    fl &= (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_KERNEL_READ_AND_WRITE);
    
    cl_mem_object_type it = 0;
    tdispatch->clGetMemObjectInfo(image, CL_MEM_TYPE, sizeof(it), &it, NULL);
    
    cl_image_format imf;
    tdispatch->clGetImageInfo(image, CL_IMAGE_FORMAT, sizeof(imf), &imf, NULL);

    cl_uint n = 0;
    tdispatch->clGetSupportedImageFormats(c, fl, it, 0, NULL, &n);

    std::vector<cl_image_format> image_formats(n);
    tdispatch->clGetSupportedImageFormats(c, fl, it, n, image_formats.data(), NULL);

    for (auto a : image_formats) {
      if ((imf.image_channel_order == a.image_channel_order) &&
          (imf.image_channel_data_type == a.image_channel_data_type))
      {
        res = false;
        break;
      }
    }
  }

  return res;
}

// check if two images have the same format
bool struct_violation(
  cl_version,
  cl_mem image1,
  cl_mem image2)
{
    cl_image_format imf1;
    tdispatch->clGetImageInfo(image1, CL_IMAGE_FORMAT, sizeof(imf1), &imf1, NULL);

    cl_image_format imf2;
    tdispatch->clGetImageInfo(image2, CL_IMAGE_FORMAT, sizeof(imf2), &imf2, NULL);

    if ((imf1.image_channel_order == imf2.image_channel_order) &&
          (imf1.image_channel_data_type == imf2.image_channel_data_type))
      return false;
    
    return true;
}

// 5.3.4

// check the correct size of fill_color for clEnqueueFillImage
bool struct_violation(
  cl_version,
  cl_mem image,
  const void* fill_color)
{
  cl_image_format imf;
  tdispatch->clGetImageInfo(image, CL_IMAGE_FORMAT, sizeof(imf), &imf, NULL);

  size_t size = 0;
  if (imf.image_channel_order == CL_DEPTH)
  {
    //  fill color is a single floating point value
    size = sizeof(float);
  }
  else
  {
    switch (imf.image_channel_data_type) {
      // fill color is a four component signed integer value 
      case CL_SIGNED_INT8:
        size = 4;
        break;
      case CL_SIGNED_INT16:
        size = 8;
        break;
      case CL_SIGNED_INT32:
        size = 16;
        break;

      // fill color is a four component unsigned integer value
      case CL_UNSIGNED_INT8:
        size = 4;
        break;
      case CL_UNSIGNED_INT16:
        size = 8;
        break;
      case CL_UNSIGNED_INT32:
        size = 16;
        break;

      default:
        // fill color is a four component RGBA floating-point color value
        size = sizeof(float) * 4;
    }
  }

  return array_len_ls(fill_color, size);
}

// check fine-grained SVM for clSetKernelExecInfo
bool struct_violation(
  cl_version,
  cl_kernel kernel,
  cl_kernel_exec_info param_name,
  const void * param_value)
{
  if ((param_name == CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM) &&
      (*static_cast<const cl_bool *>(param_value) == CL_TRUE))
  {
      std::vector<cl_device_id> devices = get_devices(kernel);

      for (auto a : devices)
        if (query<CL_DEVICE_SVM_CAPABILITIES>(a) & 
            CL_DEVICE_SVM_FINE_GRAIN_BUFFER)
          return false;

    return true;
  }

  return false;
}

// check total number of workitems in group for clEnqueueNDRangeKernel
bool struct_violation(
  cl_version,
  cl_kernel kernel,
  cl_command_queue command_queue,
  cl_uint work_dim,
  const size_t * local_work_size)
{
  if (local_work_size != NULL)
  {
    size_t kwgs = 0;
    tdispatch->clGetKernelWorkGroupInfo(
      kernel,
      query<CL_QUEUE_DEVICE>(command_queue),
      CL_KERNEL_WORK_GROUP_SIZE,
      sizeof(size_t),
      &kwgs,
      NULL);

    size_t wgs = 1;
    for (cl_uint i = 0; i < work_dim; ++i)
      wgs *= local_work_size[i];

    return (wgs > kwgs);
  }

  return false;
}

// check non-uniform workgroups for clEnqueueNDRangeKernel
bool struct_violation(
  cl_version version,
  cl_command_queue command_queue,
  cl_kernel kernel,
  cl_uint work_dim,
  const size_t * global_work_size,
  const size_t * local_work_size)
{
  if (((CL_VERSION_MAJOR(version) < 2) ||
       ((CL_VERSION_MAJOR(version) >= 3) && 
        (query<CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT>(command_queue) == CL_FALSE)))
      && (local_work_size != nullptr))
  {
    size_t cwgs[3] = {0,0,0};
    tdispatch->clGetKernelInfo(
      kernel,
      CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
      sizeof(cwgs),
      &cwgs,
      NULL);
    for (cl_uint i = 0; i < work_dim; ++i)
      if ((cwgs[i] > 0) && (cwgs[i] != local_work_size[i]))
        return true;

    for (cl_uint i = 0; i < work_dim; ++i)
      if (global_work_size[i] % local_work_size[i] != 0)
        return true;
  }

  return false;
}

// check subgroups for clEnqueueNDRangeKernel
bool struct_violation(
  cl_version version,
  cl_command_queue command_queue,
  cl_kernel kernel,
  cl_uint work_dim,
  const size_t * local_work_size)
{
  if ((CL_VERSION_MAJOR(version)*100 +
       CL_VERSION_MINOR(version) >= 201) &&
      (local_work_size != nullptr))
  {
    size_t cnsg = 0;
    tdispatch->clGetKernelSubGroupInfo(
      kernel,
      query<CL_QUEUE_DEVICE>(command_queue),
      CL_KERNEL_COMPILE_NUM_SUB_GROUPS,
      0,
      NULL,
      sizeof(size_t),
      &cnsg,
      NULL);

    if (cnsg != 0)
    {
      size_t sg = 0;
      tdispatch->clGetKernelSubGroupInfo(
        kernel,
        query<CL_QUEUE_DEVICE>(command_queue),
        CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE,
        work_dim * sizeof(size_t),
        local_work_size,
        sizeof(size_t),
        &sg,
        NULL);

      if (sg != cnsg)
        return true;
    }
  }

  return false;
}

// check max local_work_size
bool struct_violation(
  cl_version,
  cl_command_queue command_queue,
  cl_uint work_dim,
  const size_t * local_work_size)
{
  if (local_work_size != nullptr)
  {
    size_t n = 0;
    tdispatch->clGetDeviceInfo(
      query<CL_QUEUE_DEVICE>(command_queue),
      CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
      sizeof(size_t),
      &n,
      NULL);

    std::vector<size_t> mwis(n);
    tdispatch->clGetDeviceInfo(
      query<CL_QUEUE_DEVICE>(command_queue),
      CL_DEVICE_MAX_WORK_ITEM_SIZES,
      n * sizeof(size_t),
      mwis.data(),
      NULL);

    for (cl_uint i = 0; i < work_dim; ++i)
      if (mwis[i] < local_work_size[i])
        return true;
  }

  return false;
}

// check workgroups for clEnqueueTask
bool struct_violation(
  cl_version version,
  cl_command_queue command_queue,
  cl_kernel kernel)
{
  {
    size_t cwgs[3] = {0,0,0};
    tdispatch->clGetKernelInfo(
      kernel,
      CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
      sizeof(cwgs),
      &cwgs,
      NULL);

    for (cl_uint i = 0; i < 3; ++i)
      if ((cwgs[i] > 0) && (cwgs[i] != 1))
        return true;
  }

  if (version >= CL_MAKE_VERSION(2, 1, 0))
  {
    size_t cnsg = 0;
    tdispatch->clGetKernelSubGroupInfo(
      kernel,
      query<CL_QUEUE_DEVICE>(command_queue),
      CL_KERNEL_COMPILE_NUM_SUB_GROUPS,
      0,
      NULL,
      sizeof(size_t),
      &cnsg,
      NULL);

    if (cnsg != 0)
    {
      size_t local_work_size[3] = {1, 1, 1};
      size_t sg = 0;
      tdispatch->clGetKernelSubGroupInfo(
        kernel,
        query<CL_QUEUE_DEVICE>(command_queue),
        CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE,
        3 * sizeof(size_t),
        local_work_size,
        sizeof(size_t),
        &sg,
        NULL);

      if (sg != cnsg)
        return true;
    }
  }

  return false;
}

// check many devices in kernel
// for clGetKernelWorkGroupInfo and clGetKernelSubGroupInfo
bool struct_violation(
  cl_version,
  cl_kernel kernel,
  cl_device_id device)
{
  if (device == NULL)
  {
    std::vector<cl_device_id> devices = get_devices(kernel);
    if (devices.size() > 1)
      return true;
  }

  return false;
}

// check subgroups non-supported for clGetKernelSubGroupInfo
bool struct_violation(
  cl_version,
  cl_device_id device,
  cl_kernel kernel)
{
  if (device == NULL)
  {
    std::vector<cl_device_id> devices = get_devices(kernel);
    if (query<CL_DEVICE_MAX_NUM_SUB_GROUPS>(devices[0]) == 0)
      return true;
  }

  return false;
}

// check memory objects for clEnqueueNativeKernel
bool struct_violation(
  cl_version,
  const cl_mem * mem_list,
  cl_uint num_mem_objects)
{
  if (num_mem_objects > 0)
  {
    for (cl_uint i = 0; i < num_mem_objects; ++i)
      if ((mem_list[i] != NULL) && 
          !object_is_valid(mem_list[i], CL_MEM_OBJECT_BUFFER))
        return true;
  }

  return false;
}
