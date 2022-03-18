#include <stdlib.h>
#include <string.h>
#include <CL/cl_layer.h>
#include <mutex>
#include <tuple>
#include <map>
#include <vector>
#include <cstdio>
#include <list>

static struct _cl_icd_dispatch dispatch = {};

static const struct _cl_icd_dispatch *tdispatch;

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

static void _init_dispatch(void);

CL_API_ENTRY cl_int CL_API_CALL
clInitLayer(
    cl_uint                         num_entries,
    const struct _cl_icd_dispatch  *target_dispatch,
    cl_uint                        *num_entries_out,
    const struct _cl_icd_dispatch **layer_dispatch_ret) {
  if (!target_dispatch ||
      !layer_dispatch_ret ||
      !num_entries_out ||
      num_entries < sizeof(dispatch)/sizeof(dispatch.clGetPlatformIDs))
    return CL_INVALID_VALUE;

  tdispatch = target_dispatch;
  _init_dispatch();

  *layer_dispatch_ret = &dispatch;
  *num_entries_out = sizeof(dispatch)/sizeof(dispatch.clGetPlatformIDs);
  return CL_SUCCESS;
}

enum image_type {
  CL_GL_IMAGE = 0,
  CL_EGL_IMAGE,
  CL_D3D10_IMAGE,
  CL_D3D11_IMAGE,
  CL_DX9_IMAGE
};

static std::map<image_type, const char *> image_type_names = {
  {CL_GL_IMAGE, "GL"},
  {CL_EGL_IMAGE, "EGL"},
  {CL_D3D10_IMAGE, "D3D10"},
  {CL_D3D11_IMAGE, "D3D11"},
  {CL_DX9_IMAGE, "DX9"}
};

static std::map<cl_command_type, const char *> commands_type_names = {
  {CL_COMMAND_READ_IMAGE, "ReadImage"},
  {CL_COMMAND_WRITE_IMAGE, "WriteImage"},
  {CL_COMMAND_COPY_IMAGE, "CopyImage"},
  {CL_COMMAND_COPY_IMAGE_TO_BUFFER, "CopyImageToBuffer"},
  {CL_COMMAND_COPY_BUFFER_TO_IMAGE, "CopyBufferToImage"},
  {CL_COMMAND_MAP_IMAGE, "MapImage"},
  {CL_COMMAND_UNMAP_MEM_OBJECT, "UnmapMemObject"},
  {CL_COMMAND_FILL_IMAGE, "EnqueueFillImage"}
};

struct image_desc {
  cl_mem buffer;
  image_type type;
};

static std::map<cl_mem, image_desc> objects;
static std::map<cl_kernel, std::map<cl_uint, cl_mem>> kernel_image_arguments;
static std::mutex objects_mutex;


static cl_int zero = 0;
static cl_int one = 1;


static void CL_CALLBACK buff_destructor(cl_mem memobj, void* user_data) {
  (void)memobj;
  objects_mutex.lock();
  auto iter = objects.find((cl_mem)user_data);
  if (iter != objects.end()) {
    objects.erase(iter);
    objects_mutex.unlock();
  } else // this should technically not occur
    objects_mutex.unlock();
}

static void CL_CALLBACK image_destructor(cl_mem memobj, void* user_data) {
  (void)memobj;
  tdispatch->clReleaseMemObject((cl_mem)user_data);
}

// associate images with a cl buffer to store the aquired state
static inline void register_image(
    cl_context context,
    cl_mem image,
    image_type type)
{
  cl_int err;
  cl_mem buff = tdispatch->clCreateBuffer(
    context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
  if (CL_SUCCESS == err && buff) {
    // Destroy the map entry when the object is destroyed.
    tdispatch->clSetMemObjectDestructorCallback(
      buff,
      buff_destructor,
      image);
    // Destroy the buffer when the image is destroyed
    // If it is in a queue it will be kept alive untill callbacks are finished
    tdispatch->clSetMemObjectDestructorCallback(
      image,
      image_destructor,
      buff);
    objects_mutex.lock();
    objects[image] = {buff, type};
    objects_mutex.unlock();
  }
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLTexture2D_wrap(
    cl_context context,
    cl_mem_flags flags,
    cl_GLenum target,
    cl_GLint miplevel,
    cl_GLuint texture,
    cl_int* errcode_ret)
{
  cl_mem image = tdispatch->clCreateFromGLTexture2D(
    context,
    flags,
    target,
    miplevel,
    texture,
    errcode_ret);

  if (image)
    register_image(context, image, CL_GL_IMAGE);
  return image;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLTexture3D_wrap(
    cl_context context,
    cl_mem_flags flags,
    cl_GLenum target,
    cl_GLint miplevel,
    cl_GLuint texture,
    cl_int* errcode_ret)
{
  cl_mem image = tdispatch->clCreateFromGLTexture3D(
    context,
    flags,
    target,
    miplevel,
    texture,
    errcode_ret);

  if (image)
    register_image(context, image, CL_GL_IMAGE);
  return image;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLRenderbuffer_wrap(
    cl_context context,
    cl_mem_flags flags,
    cl_GLuint renderbuffer,
    cl_int* errcode_ret)
{
  cl_mem image = tdispatch->clCreateFromGLRenderbuffer(
    context,
    flags,
    renderbuffer,
    errcode_ret);

  if (image)
    register_image(context, image, CL_GL_IMAGE);
  return image;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromGLTexture_wrap(
    cl_context context,
    cl_mem_flags flags,
    cl_GLenum target,
    cl_GLint miplevel,
    cl_GLuint texture,
    cl_int* errcode_ret)
{
  cl_mem image = tdispatch->clCreateFromGLTexture(
    context,
    flags,
    target,
    miplevel,
    texture,
    errcode_ret);

  if (image)
    register_image(context, image, CL_GL_IMAGE);
  return image;
}

#if defined(_WIN32)
static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromD3D10Texture2DKHR_wrap(
    cl_context context,
    cl_mem_flags flags,
    ID3D10Texture2D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
  cl_mem image = tdispatch->clCreateFromD3D10Texture2DKHR(
    context,
    flags,
    resource,
    subresource,
    errcode_ret);

  if (image)
    register_image(context, image, CL_D3D10_IMAGE);
  return image;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromD3D10Texture3DKHR_wrap(
    cl_context context,
    cl_mem_flags flags,
    ID3D10Texture3D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
  cl_mem image = tdispatch->clCreateFromD3D10Texture3DKHR(
    context,
    flags,
    resource,
    subresource,
    errcode_ret);

  if (image)
    register_image(context, image, CL_D3D10_IMAGE);
  return image;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromD3D11Texture2DKHR_wrap(
    cl_context context,
    cl_mem_flags flags,
    ID3D11Texture2D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
  cl_mem image = tdispatch->clCreateFromD3D11Texture2DKHR(
    context,
    flags,
    resource,
    subresource,
    errcode_ret);

  if (image)
    register_image(context, image, CL_D3D11_IMAGE);
  return image;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromD3D11Texture3DKHR_wrap(
    cl_context context,
    cl_mem_flags flags,
    ID3D11Texture3D* resource,
    UINT subresource,
    cl_int* errcode_ret)
{
  cl_mem image = tdispatch->clCreateFromD3D11Texture3DKHR(
    context,
    flags,
    resource,
    subresource,
    errcode_ret);

  if (image)
    register_image(context, image, CL_D3D11_IMAGE);
  return image;
}

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromDX9MediaSurfaceKHR_wrap(
    cl_context context,
    cl_mem_flags flags,
    cl_dx9_media_adapter_type_khr adapter_type,
    void* surface_info,
    cl_uint plane,
    cl_int* errcode_ret)
{
  cl_mem image = tdispatch->clCreateFromDX9MediaSurfaceKHR(
    context,
    flags,
    adapter_type,
    surface_info,
    plane,
    errcode_ret);

  if (image)
    register_image(context, image, CL_DX9_IMAGE);
  return image;
}

#endif

static CL_API_ENTRY cl_mem CL_API_CALL clCreateFromEGLImageKHR_wrap(
    cl_context context,
    CLeglDisplayKHR egldisplay,
    CLeglImageKHR eglimage,
    cl_mem_flags flags,
    const cl_egl_image_properties_khr* properties,
    cl_int* errcode_ret)
{
  cl_mem image = tdispatch->clCreateFromEGLImageKHR(
    context,
    egldisplay,
    eglimage,
    flags,
    properties,
    errcode_ret);

  if (image)
    register_image(context, image, CL_EGL_IMAGE);
  return image;
}


// Enqueue a write into each of the associated mem objects
// to change the status
static inline void enqueue_write_chain(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_event first_event,
    cl_int *value,
    cl_event *event)
{
  // clEnqueueMarkerWithEventWait List is not always available
  // so create a denpendency chain starting by the previous Acquire
  std::vector<cl_event> events;
  events.push_back(first_event);
  for (cl_uint i = 0; i < num_objects; i++) {
    image_desc desc;
    objects_mutex.lock();
    auto iter = objects.find(mem_objects[i]);
    if (iter != objects.end()) {
      desc = iter->second;
      objects_mutex.unlock();
    } else {
      // this can occur wehen the object is not a texture
      // (a buffer for instance)
      objects_mutex.unlock();
      continue;
    }
    cl_mem buff = desc.buffer;
    cl_event evt = events.back();
    if (event)
      events.push_back(NULL);
    cl_int err = tdispatch->clEnqueueWriteBuffer(
      command_queue, buff, CL_FALSE, 0, sizeof(*value), value,
      1, &evt, event ? &events.back() : event);
    if (CL_SUCCESS != err && event)
      events.pop_back();
  }
  if (event) {
    *event = events.back();
    events.pop_back();
    for (auto it = events.begin(); it != events.end(); it++)
      tdispatch->clReleaseEvent(*it);
  } else
    tdispatch->clReleaseEvent(first_event);
}

// Enqueue a write into each of the associated mem objects
// to change the status to acquired
static inline void set_objects_status_aquired(
  cl_command_queue command_queue,
  cl_uint num_objects,
  const cl_mem* mem_objects,
  cl_event first_event,
  cl_event *event)
{
  enqueue_write_chain(
    command_queue,
    num_objects,
    mem_objects,
    first_event,
    &one,
    event);
}

// Enqueue a write into each of the associated mem objects
// to change the status to released
static inline void set_objects_status_released(
  cl_command_queue command_queue,
  cl_uint num_objects,
  const cl_mem* mem_objects,
  cl_event first_event,
  cl_event *event)
{
  enqueue_write_chain(
    command_queue,
    num_objects,
    mem_objects,
    first_event,
    &zero,
    event);
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueAcquireGLObjects_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  cl_event first_event;
  cl_int result = tdispatch->clEnqueueAcquireGLObjects(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    &first_event);

  if (result == CL_SUCCESS)
    set_objects_status_aquired(
      command_queue,
      num_objects,
      mem_objects,
      first_event,
      event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseGLObjects_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  cl_event first_event;
  cl_int result = tdispatch->clEnqueueReleaseGLObjects(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    &first_event);

  if (result == CL_SUCCESS)
    set_objects_status_released(
      command_queue,
      num_objects,
      mem_objects,
      first_event,
      event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueAcquireEGLObjectsKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  cl_event first_event;
  cl_int result = tdispatch->clEnqueueAcquireEGLObjectsKHR(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    &first_event);

  if (result == CL_SUCCESS)
    set_objects_status_aquired(
      command_queue,
      num_objects,
      mem_objects,
      first_event,
      event);
  return result;
}

// Here we could change the aquire status before the release,
// but that would be assuming the release is enqueued correctly
// so keep the same pattern as before
static CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseEGLObjectsKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  cl_event first_event;
  cl_int result = tdispatch->clEnqueueReleaseEGLObjectsKHR(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    &first_event);

  if (result == CL_SUCCESS)
    set_objects_status_released(
      command_queue,
      num_objects,
      mem_objects,
      first_event,
      event);
  return result;
}

#if defined(_WIN32)
static CL_API_ENTRY cl_int CL_API_CALL clEnqueueAcquireD3D10ObjectsKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  cl_event first_event;
  cl_int result = tdispatch->clEnqueueAcquireD3D10ObjectsKHR(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    &first_event);

  if (result == CL_SUCCESS)
    set_objects_status_aquired(
      command_queue,
      num_objects,
      mem_objects,
      first_event,
      event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseD3D10ObjectsKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  cl_event first_event;
  cl_int result = tdispatch->clEnqueueReleaseD3D10ObjectsKHR(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    &first_event);

  if (result == CL_SUCCESS)
    set_objects_status_released(
      command_queue,
      num_objects,
      mem_objects,
      first_event,
      event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueAcquireD3D11ObjectsKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  cl_event first_event;
  cl_int result = tdispatch->clEnqueueAcquireD3D11ObjectsKHR(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    &first_event);

  if (result == CL_SUCCESS)
    set_objects_status_aquired(
      command_queue,
      num_objects,
      mem_objects,
      first_event,
      event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseD3D11ObjectsKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  cl_event first_event;
  cl_int result = tdispatch->clEnqueueReleaseD3D11ObjectsKHR(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    &first_event);

  if (result == CL_SUCCESS)
    set_objects_status_released(
      command_queue,
      num_objects,
      mem_objects,
      first_event,
      event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueAcquireDX9MediaSurfacesKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  cl_event first_event;
  cl_int result = tdispatch->clEnqueueAcquireDX9MediaSurfacesKHR(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    &first_event);

  if (result == CL_SUCCESS)
    set_objects_status_aquired(
      command_queue,
      num_objects,
      mem_objects,
      first_event,
      event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseDX9MediaSurfacesKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  cl_event first_event;
  cl_int result = tdispatch->clEnqueueReleaseDX9MediaSurfacesKHR(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    &first_event);

  if (result == CL_SUCCESS)
    set_objects_status_released(
      command_queue,
      num_objects,
      mem_objects,
      first_event,
      event);
  return result;
}
#endif

struct image_state {
  cl_mem image;
  cl_int acquired;
  cl_command_type command_type;
  image_type type;
};

static inline cl_int enqueue_image_state_copy(
    cl_command_queue command_queue,
    cl_mem image,
    image_state *state,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  image_desc desc;
  objects_mutex.lock();
  auto iter = objects.find(image);
  if (iter != objects.end()) {
    desc = iter->second;
    objects_mutex.unlock();
  } else {
    // this should technically not occur
    objects_mutex.unlock();
    return CL_INVALID_MEM_OBJECT;
  }
  cl_mem buffer = desc.buffer;
  state->type = desc.type;
  return tdispatch->clEnqueueReadBuffer(
    command_queue,
    buffer,
    CL_FALSE,
    0,
    sizeof(state->acquired),
    &state->acquired,
    num_events_in_wait_list,
    event_wait_list,
    event);
}

static void CL_CALLBACK status_check_notify(
    cl_event event,
    cl_int event_command_status,
    void *user_data) {
  (void)event;
  image_state *state = (image_state *)user_data;
  if (!state->acquired && CL_SUCCESS == event_command_status)
    fprintf(stderr, "%s Image %p was used in %s before being acquired\n",
      image_type_names[state->type], (void*)(state->image),
      commands_type_names[state->command_type]);
  delete state;
}

static void CL_CALLBACK status_cleanup_notify(
    cl_event event,
    cl_int event_command_status,
    void *user_data) {
  (void)event;
  (void)event_command_status;
  image_state *state = (image_state *)user_data;
  delete state;
}

static inline void multiple_images_command_pre_enqueue(
    cl_command_type command_type,
    cl_command_queue command_queue,
    std::vector<cl_mem> images,
    cl_uint &num_events_in_wait_list,
    const cl_event * &event_wait_list,
    std::vector<cl_event> &events,
    std::vector<image_state *> &states)
{
  for (auto image: images) {
    image_state * state = new image_state();
    state->image = image;
    state->acquired = 0;
    state->command_type = command_type;
    events.push_back(NULL);
    cl_int result = enqueue_image_state_copy(
      command_queue,
      image,
      state,
      num_events_in_wait_list,
      event_wait_list,
      &events.back());
    if (CL_SUCCESS == result && events.back()) {
      states.push_back(state);
    } else {
      delete state;
      events.pop_back();
    }
  }

  if (events.size() > 0) {
    num_events_in_wait_list = events.size();
    event_wait_list = events.data();
  }
}

static inline cl_int simple_command_pre_enqueue(
    cl_command_type type,
    cl_command_queue command_queue,
    cl_mem image,
    cl_uint &num_events_in_wait_list,
    const cl_event * &event_wait_list,
    cl_event &event,
    image_state * &state)
{
  state = new image_state();
  state->image = image;
  state->acquired = 0;
  state->command_type = type;

  cl_int result = enqueue_image_state_copy(
    command_queue,
    image,
    state,
    num_events_in_wait_list,
    event_wait_list,
    &event);

  if (CL_SUCCESS == result && event) {
    num_events_in_wait_list = 1;
    event_wait_list = &event;
  } else {
    delete state;
  }
  return result;
}

static inline void multiple_images_command_post_enqueue(
    cl_int result,
    std::vector<cl_event> &events,
    std::vector<image_state *> &states)
{
  for (size_t i = 0; i < events.size(); i++) {
    auto event = events[i];
    auto state = states[i];
    if (CL_SUCCESS == result)
      tdispatch->clSetEventCallback(
        event,
        CL_COMPLETE,
        status_check_notify,
        state);
    else
      tdispatch->clSetEventCallback(
        event,
        CL_COMPLETE,
        status_cleanup_notify,
        state);
    tdispatch->clReleaseEvent(event);
  }
}

static inline void simple_command_post_enqueue(
    cl_int result_pre,
    cl_int result,
    cl_event event,
    image_state *state)
{
  if (CL_SUCCESS == result_pre) {
    if (CL_SUCCESS == result)
      tdispatch->clSetEventCallback(
        event,
        CL_COMPLETE,
        status_check_notify,
        state);
    else
      tdispatch->clSetEventCallback(
        event,
        CL_COMPLETE,
        status_cleanup_notify,
        state);
    tdispatch->clReleaseEvent(event);
  }
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadImage_wrap(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_read,
    const size_t* origin,
    const size_t* region,
    size_t row_pitch,
    size_t slice_pitch,
    void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  cl_event first_event;
  image_state *state;
  cl_int result_pre = simple_command_pre_enqueue(
    CL_COMMAND_READ_IMAGE,
    command_queue, image, num_events_in_wait_list,
    event_wait_list, first_event, state);

  cl_int result = tdispatch->clEnqueueReadImage(
    command_queue,
    image,
    blocking_read,
    origin,
    region,
    row_pitch,
    slice_pitch,
    ptr,
    num_events_in_wait_list,
    event_wait_list,
    event);

  simple_command_post_enqueue(
    result_pre, result, first_event, state);

  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteImage_wrap(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_write,
    const size_t* origin,
    const size_t* region,
    size_t input_row_pitch,
    size_t input_slice_pitch,
    const void* ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  cl_event first_event;
  image_state *state;
  cl_int result_pre = simple_command_pre_enqueue(
    CL_COMMAND_WRITE_IMAGE,
    command_queue, image, num_events_in_wait_list,
    event_wait_list, first_event, state);

  cl_int result = tdispatch->clEnqueueWriteImage(
    command_queue,
    image,
    blocking_write,
    origin,
    region,
    input_row_pitch,
    input_slice_pitch,
    ptr,
    num_events_in_wait_list,
    event_wait_list,
    event);

  simple_command_post_enqueue(
    result_pre, result, first_event, state);

  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyImage_wrap(
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_image,
    const size_t* src_origin,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  std::vector<cl_mem> images = {src_image, dst_image};
  std::vector<cl_event> events;
  std::vector<image_state *> states;

  multiple_images_command_pre_enqueue(
    CL_COMMAND_COPY_IMAGE,
    command_queue, images, num_events_in_wait_list, event_wait_list,
    events, states);

  cl_int result = tdispatch->clEnqueueCopyImage(
    command_queue,
    src_image,
    dst_image,
    src_origin,
    dst_origin,
    region,
    num_events_in_wait_list,
    event_wait_list,
    event);

  multiple_images_command_post_enqueue(result, events, states);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyImageToBuffer_wrap(
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_buffer,
    const size_t* src_origin,
    const size_t* region,
    size_t dst_offset,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  cl_event first_event;
  image_state *state;
  cl_int result_pre = simple_command_pre_enqueue(
    CL_COMMAND_COPY_IMAGE_TO_BUFFER,
    command_queue, src_image, num_events_in_wait_list,
    event_wait_list, first_event, state);

  cl_int result = tdispatch->clEnqueueCopyImageToBuffer(
    command_queue,
    src_image,
    dst_buffer,
    src_origin,
    region,
    dst_offset,
    num_events_in_wait_list,
    event_wait_list,
    event);

  simple_command_post_enqueue(
    result_pre, result, first_event, state);

  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBufferToImage_wrap(
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_image,
    size_t src_offset,
    const size_t* dst_origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  cl_event first_event;
  image_state *state;
  cl_int result_pre = simple_command_pre_enqueue(
    CL_COMMAND_COPY_BUFFER_TO_IMAGE,
    command_queue, dst_image, num_events_in_wait_list,
    event_wait_list, first_event, state);

  cl_int result = tdispatch->clEnqueueCopyBufferToImage(
    command_queue,
    src_buffer,
    dst_image,
    src_offset,
    dst_origin,
    region,
    num_events_in_wait_list,
    event_wait_list,
    event);

  simple_command_post_enqueue(
    result_pre, result, first_event, state);

  return result;
}

static CL_API_ENTRY void* CL_API_CALL clEnqueueMapImage_wrap(
    cl_command_queue command_queue,
    cl_mem image,
    cl_bool blocking_map,
    cl_map_flags map_flags,
    const size_t* origin,
    const size_t* region,
    size_t* image_row_pitch,
    size_t* image_slice_pitch,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event,
    cl_int* errcode_ret)
{
  cl_event first_event;
  image_state *state;
  cl_int      errcode_tmp;
  cl_int result_pre = simple_command_pre_enqueue(
    CL_COMMAND_MAP_IMAGE,
    command_queue, image, num_events_in_wait_list,
    event_wait_list, first_event, state);

  if (!errcode_ret)
    errcode_ret = &errcode_tmp;
  void *result = tdispatch->clEnqueueMapImage(
            command_queue,
            image,
            blocking_map,
            map_flags,
            origin,
            region,
            image_row_pitch,
            image_slice_pitch,
            num_events_in_wait_list,
            event_wait_list,
            event,
            errcode_ret);

  simple_command_post_enqueue(
    result_pre, *errcode_ret, first_event, state);

  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueUnmapMemObject_wrap(
    cl_command_queue command_queue,
    cl_mem memobj,
    void* mapped_ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  cl_event first_event;
  image_state *state;
  cl_int result_pre = simple_command_pre_enqueue(
    CL_COMMAND_UNMAP_MEM_OBJECT,
    command_queue, memobj, num_events_in_wait_list,
    event_wait_list, first_event, state);

  cl_int result = tdispatch->clEnqueueUnmapMemObject(
    command_queue,
    memobj,
    mapped_ptr,
    num_events_in_wait_list,
    event_wait_list,
    event);

  simple_command_post_enqueue(
    result_pre, result, first_event, state);

  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueFillImage_wrap(
    cl_command_queue command_queue,
    cl_mem image,
    const void* fill_color,
    const size_t* origin,
    const size_t* region,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  cl_event first_event;
  image_state *state;
  cl_int result_pre = simple_command_pre_enqueue(
    CL_COMMAND_FILL_IMAGE,
    command_queue, image, num_events_in_wait_list,
    event_wait_list, first_event, state);

  cl_int result = tdispatch->clEnqueueFillImage(
    command_queue,
    image,
    fill_color,
    origin,
    region,
    num_events_in_wait_list,
    event_wait_list,
    event);

  simple_command_post_enqueue(
    result_pre, result, first_event, state);

  return result;
}

static inline void register_kernel(cl_kernel kernel) {
  objects_mutex.lock();
  auto iter = kernel_image_arguments.find(kernel);
  if (iter != kernel_image_arguments.end())
    kernel_image_arguments.erase(iter);
  objects_mutex.unlock();
}

static CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel_wrap(
    cl_program program,
    const char* kernel_name,
    cl_int* errcode_ret)
{
  cl_kernel kernel = tdispatch->clCreateKernel(
    program,
    kernel_name,
    errcode_ret);
  if (kernel)
    register_kernel(kernel);
  return kernel;
}

static inline void clone_kernel(cl_kernel source_kernel, cl_kernel kernel) {
  objects_mutex.lock();
  auto iter = kernel_image_arguments.find(source_kernel);
  if (iter != kernel_image_arguments.end())
    kernel_image_arguments[kernel] = iter->second;
  objects_mutex.unlock();
}

static CL_API_ENTRY cl_kernel CL_API_CALL clCloneKernel_wrap(
    cl_kernel source_kernel,
    cl_int* errcode_ret)
{
  cl_kernel kernel = tdispatch->clCloneKernel(
    source_kernel,
    errcode_ret);
  if (kernel)
    clone_kernel(source_kernel, kernel);
  return kernel;
}

static CL_API_ENTRY cl_int CL_API_CALL clCreateKernelsInProgram_wrap(
    cl_program program,
    cl_uint num_kernels,
    cl_kernel* kernels,
    cl_uint* num_kernels_ret)
{
  cl_uint num_kernels_ret_force;
  if (kernels && !num_kernels_ret)
    num_kernels_ret = &num_kernels_ret_force;
  cl_int result = tdispatch->clCreateKernelsInProgram(
    program,
    num_kernels,
    kernels,
    num_kernels_ret);
  if (kernels && result == CL_SUCCESS && *num_kernels_ret > 0) {
    for (cl_uint i = 0; i < *num_kernels_ret; i++)
      register_kernel(kernels[i]);
  }
  return result;
}

static void register_kernel_argument(
    cl_kernel kernel,
    cl_uint arg_index,
    size_t arg_size,
    const void* arg_value)
{
  if (sizeof(cl_mem) == arg_size) {
    cl_mem memobj = *(cl_mem *)arg_value;
    objects_mutex.lock();
    auto iter = objects.find(memobj);
    if (iter != objects.end()) {
      auto arguments = kernel_image_arguments[kernel];
      arguments[arg_index] = memobj;
      objects_mutex.unlock();
    } else {
      objects_mutex.unlock();
    }
  }
}

static CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg_wrap(
    cl_kernel kernel,
    cl_uint arg_index,
    size_t arg_size,
    const void* arg_value)
{
  cl_int result = tdispatch->clSetKernelArg(
    kernel,
    arg_index,
    arg_size,
    arg_value);
  if (CL_SUCCESS == result)
    register_kernel_argument(
      kernel,
      arg_index,
      arg_size,
      arg_value);
  return result;
}

static void collect_image_arguments(
    cl_kernel kernel,
    std::vector<cl_mem> &images)
{
  objects_mutex.lock();
  auto iter = kernel_image_arguments.find(kernel);
  if (iter != kernel_image_arguments.end()) {
    for (auto it = iter->second.begin(); it != iter->second.end(); ++it)
      images.push_back(it->second);
  }
  objects_mutex.unlock();
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel_wrap(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    const size_t* local_work_size,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  std::vector<cl_mem> images;
  std::vector<cl_event> events;
  std::vector<image_state *> states;

  collect_image_arguments(kernel, images);
  multiple_images_command_pre_enqueue(
    CL_COMMAND_NDRANGE_KERNEL,
    command_queue, images, num_events_in_wait_list, event_wait_list,
    events, states);

  cl_int result = tdispatch->clEnqueueNDRangeKernel(
    command_queue,
    kernel,
    work_dim,
    global_work_offset,
    global_work_size,
    local_work_size,
    num_events_in_wait_list,
    event_wait_list,
    event);

  multiple_images_command_post_enqueue(result, events, states);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueTask_wrap(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  std::vector<cl_mem> images;
  std::vector<cl_event> events;
  std::vector<image_state *> states;

  collect_image_arguments(kernel, images);
  multiple_images_command_pre_enqueue(
    CL_COMMAND_NDRANGE_KERNEL,
    command_queue, images, num_events_in_wait_list, event_wait_list,
    events, states);

  cl_int result = tdispatch->clEnqueueTask(
    command_queue,
    kernel,
    num_events_in_wait_list,
    event_wait_list,
    event);

  multiple_images_command_post_enqueue(result, events, states);
  return result;
}

static void _init_dispatch(void) {
  dispatch.clCreateFromGLTexture2D             = &clCreateFromGLTexture2D_wrap;
  dispatch.clCreateFromGLTexture3D             = &clCreateFromGLTexture3D_wrap;
  dispatch.clCreateFromGLRenderbuffer          = &clCreateFromGLRenderbuffer_wrap;
  dispatch.clCreateFromGLTexture               = &clCreateFromGLTexture_wrap;
  dispatch.clEnqueueAcquireGLObjects           = &clEnqueueAcquireGLObjects_wrap;
  dispatch.clEnqueueReleaseGLObjects           = &clEnqueueReleaseGLObjects_wrap;
#if defined(_WIN32)
  dispatch.clCreateFromD3D10Texture2DKHR       = &clCreateFromD3D10Texture2DKHR_wrap;
  dispatch.clCreateFromD3D10Texture3DKHR       = &clCreateFromD3D10Texture3DKHR_wrap;
  dispatch.clEnqueueAcquireD3D10ObjectsKHR     = &clEnqueueAcquireD3D10ObjectsKHR_wrap;
  dispatch.clEnqueueReleaseD3D10ObjectsKHR     = &clEnqueueReleaseD3D10ObjectsKHR_wrap;
  dispatch.clCreateFromD3D11Texture2DKHR       = &clCreateFromD3D11Texture2DKHR_wrap;
  dispatch.clCreateFromD3D11Texture3DKHR       = &clCreateFromD3D11Texture3DKHR_wrap;
  dispatch.clEnqueueAcquireD3D11ObjectsKHR     = &clEnqueueAcquireD3D11ObjectsKHR_wrap;
  dispatch.clEnqueueReleaseD3D11ObjectsKHR     = &clEnqueueReleaseD3D11ObjectsKHR_wrap;
  dispatch.clCreateFromDX9MediaSurfaceKHR      = &clCreateFromDX9MediaSurfaceKHR_wrap;
  dispatch.clEnqueueAcquireDX9MediaSurfacesKHR = &clEnqueueAcquireDX9MediaSurfacesKHR_wrap;
  dispatch.clEnqueueReleaseDX9MediaSurfacesKHR = &clEnqueueReleaseDX9MediaSurfacesKHR_wrap;
#endif
  dispatch.clCreateFromEGLImageKHR             = &clCreateFromEGLImageKHR_wrap;
  dispatch.clEnqueueAcquireEGLObjectsKHR       = &clEnqueueAcquireEGLObjectsKHR_wrap;
  dispatch.clEnqueueReleaseEGLObjectsKHR       = &clEnqueueReleaseEGLObjectsKHR_wrap;
  dispatch.clEnqueueReadImage                  = &clEnqueueReadImage_wrap;
  dispatch.clEnqueueWriteImage                 = &clEnqueueWriteImage_wrap;
  dispatch.clEnqueueCopyImage                  = &clEnqueueCopyImage_wrap;
  dispatch.clEnqueueCopyBufferToImage          = &clEnqueueCopyBufferToImage_wrap;
  dispatch.clEnqueueCopyImageToBuffer          = &clEnqueueCopyImageToBuffer_wrap;
  dispatch.clEnqueueMapImage                   = &clEnqueueMapImage_wrap;
  dispatch.clEnqueueUnmapMemObject             = &clEnqueueUnmapMemObject_wrap;
  dispatch.clEnqueueFillImage                  = &clEnqueueFillImage_wrap;
  dispatch.clCreateKernel                      = &clCreateKernel_wrap;
  dispatch.clCloneKernel                       = &clCloneKernel_wrap;
  dispatch.clCreateKernelsInProgram            = &clCreateKernelsInProgram_wrap;
  dispatch.clSetKernelArg                      = &clSetKernelArg_wrap;
  dispatch.clEnqueueNDRangeKernel              = &clEnqueueNDRangeKernel_wrap;
  dispatch.clEnqueueTask                       = &clEnqueueTask_wrap;
}
