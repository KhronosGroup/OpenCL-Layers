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
  if (!target_dispatch || !layer_dispatch_ret ||!num_entries_out || num_entries < sizeof(dispatch)/sizeof(dispatch.clGetPlatformIDs))
    return CL_INVALID_VALUE;

  tdispatch = target_dispatch;
  _init_dispatch();

  *layer_dispatch_ret = &dispatch;
  *num_entries_out = sizeof(dispatch)/sizeof(dispatch.clGetPlatformIDs);
  return CL_SUCCESS;
}

typedef std::tuple<cl_mem, cl_long> object_count;
static std::map<cl_mem, object_count> objects;
static std::mutex objects_mutex;

static cl_int zero = 0;
static cl_int one = 1;


void CL_CALLBACK buff_destructor(cl_mem memobj, void* user_data) {
  (void)memobj;
  objects_mutex.lock();
  auto iter = objects.find((cl_mem)user_data);
  if (iter != objects.end()) {
    objects.erase(iter);
    objects_mutex.unlock();
  } else // this should technically not occur
    objects_mutex.unlock();
}

void CL_CALLBACK image_destructor(cl_mem memobj, void* user_data) {
  (void)memobj;
  tdispatch->clReleaseMemObject((cl_mem)user_data);
}

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

  // associate images with a cl buffer to store the aquired state
  if (image) {
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
      objects[image] = std::make_tuple(buff, 1);
      objects_mutex.unlock();
    }
  }
  return image;
}


// Enqueue a write into each of the associated mem objects
// to change the status
static inline void enqueueWriteChain(
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
    object_count oc;
    objects_mutex.lock();
    auto iter = objects.find(mem_objects[i]);
    if (iter != objects.end()) {
      oc = iter->second;
      objects_mutex.unlock();
    } else {
      // this should technically not occur
      objects_mutex.unlock();
      continue;
    }
    cl_mem buff = std::get<0>(oc);
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

  // Enqueue a write into each of the associated mem objects
  // to change the status to acquired
  if (result == CL_SUCCESS)
    enqueueWriteChain(
      command_queue,
      num_objects,
      mem_objects,
      first_event,
      &one,
      event);
  return result;
}

static CL_API_ENTRY cl_int CL_API_CALL clEnqueueReleaseEGLObjectsKHR_wrap(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem* mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  // Here we could change the aquire status before the release,
  // but that would be assuming the release is enqueued correctly
  // so keep the same pattern as before
  cl_event first_event;
  cl_int result = tdispatch->clEnqueueReleaseEGLObjectsKHR(
    command_queue,
    num_objects,
    mem_objects,
    num_events_in_wait_list,
    event_wait_list,
    &first_event);

  // Enqueue a write into each of the associated mem objects
  // to change the status to released
  if (result == CL_SUCCESS)
    enqueueWriteChain(
      command_queue,
      num_objects,
      mem_objects,
      first_event,
      &zero,
      event);
  return result;
}

static inline cl_int enqueueImageStateCopy(
    cl_command_queue command_queue,
    cl_mem image,
    cl_int *ptr,
    cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list,
    cl_event* event)
{
  object_count oc;
  objects_mutex.lock();
  auto iter = objects.find(image);
  if (iter != objects.end()) {
    oc = iter->second;
    objects_mutex.unlock();
  } else {
    // this should technically not occur
    objects_mutex.unlock();
    return CL_INVALID_MEM_OBJECT;
  }
  cl_mem buffer = std::get<0>(oc);
  return tdispatch->clEnqueueReadBuffer(
    command_queue,
    buffer,
    CL_FALSE,
    0,
    sizeof(*ptr),
    ptr,
    num_events_in_wait_list,
    event_wait_list,
    event);
}

struct image_state {
  cl_mem image;
  cl_int acquired;
};

static void CL_CALLBACK status_check_notify(
    cl_event event,
    cl_int event_command_status,
    void *user_data) {
  (void)event;
  image_state *state = (image_state *)user_data;
  if (!state->acquired && CL_SUCCESS == event_command_status)
    fprintf(stderr, "EGL Image %p was used before being acquired\n", (void*)(state->image));
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

static inline cl_int simple_command_pre_enqueue(
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

  cl_int result = enqueueImageStateCopy(
    command_queue,
    image,
    &state->acquired,
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

static inline void multiple_images_command_pre_enqueue(
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
    events.push_back(NULL);
    cl_int result = enqueueImageStateCopy(
      command_queue,
      image,
      &state->acquired,
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

static inline void multiple_images_post_enqueue(
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

  multiple_images_post_enqueue(result, events, states);
  return result;
}

static void _init_dispatch(void) {
  dispatch.clCreateFromEGLImageKHR = &clCreateFromEGLImageKHR_wrap;
  dispatch.clEnqueueAcquireEGLObjectsKHR = &clEnqueueAcquireEGLObjectsKHR_wrap;
  dispatch.clEnqueueReleaseEGLObjectsKHR = &clEnqueueReleaseEGLObjectsKHR_wrap;
  dispatch.clEnqueueReadImage            = &clEnqueueReadImage_wrap;
  dispatch.clEnqueueWriteImage           = &clEnqueueWriteImage_wrap;
  dispatch.clEnqueueCopyBufferToImage    = &clEnqueueCopyBufferToImage_wrap;
  dispatch.clEnqueueCopyImageToBuffer    = &clEnqueueCopyImageToBuffer_wrap;
  dispatch.clEnqueueCopyImage            = &clEnqueueCopyImage_wrap;
}
