#include "param_verification_test.hpp"

int main(int argc, char* argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  param_verification_test::setup(argc, argv, CL_MAKE_VERSION(1, 2, 0), platform, device);

  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
  cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, &status);
  EXPECT_SUCCESS(status);

  // CL_INVALID_VALUE if devices is NULL
  clCreateContext(properties, 1, nullptr, nullptr, nullptr, &status); // devices is NULL
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  // CL_INVALID_VALUE if num_devices is 0
  clCreateContext(properties, 0, &device, nullptr, nullptr, &status); // size is greater than CL_DEVICE_MAX_MEM_ALLOC_SIZE for all devices in context
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  size_t buffer_size = 128;
  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, nullptr, &status);
  EXPECT_SUCCESS(status);

  cl_command_queue queue = clCreateCommandQueue(context,
                                                device,
                                                0,
                                                &status);
  EXPECT_SUCCESS(status);

  cl_int pattern = 0;
  // sanity check
  status = clEnqueueFillBuffer(queue,
                               buffer,
                               &pattern,
                               sizeof(cl_int),
                               0,
                               buffer_size,
                               0,
                               nullptr,
                               nullptr);
  EXPECT_SUCCESS(status);

  // CL_INVALID_VALUE if pattern is NULL
  status = clEnqueueFillBuffer(queue,
                               buffer,
                               nullptr,
                               sizeof(cl_int),
                               0,
                               buffer_size,
                               0,
                               nullptr,
                               nullptr); // pattern is NULL
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  // CL_INVALID_VALUE if pattern_size is 0 or not a power of 2 < 128
  status = clEnqueueFillBuffer(queue,
                               buffer,
                               &pattern,
                               0,
                               0,
                               buffer_size,
                               0,
                               nullptr,
                               nullptr); // pattern_size is not one of { 1, 2, 4, 8, 16, 32, 64, 128 }
  EXPECT_ERROR(status, CL_INVALID_VALUE);
  status = clEnqueueFillBuffer(queue,
                               buffer,
                               &pattern,
                               3,
                               0,
                               buffer_size,
                               0,
                               nullptr,
                               nullptr); // pattern_size is not one of { 1, 2, 4, 8, 16, 32, 64, 128 }
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  // CL_INVALID_EVENT_WAIT_LIST if event_wait_list is NULL and num_events_in_wait_list > 0
  status = clEnqueueFillBuffer(queue,
                               buffer,
                               &pattern,
                               sizeof(cl_int),
                               0,
                               buffer_size,
                               10,
                               nullptr,
                               nullptr); // event_wait_list is NULL and num_events_in_wait_list > 0
  EXPECT_ERROR(status, CL_INVALID_EVENT_WAIT_LIST);

  // CL_INVALID_EVENT_WAIT_LIST if event_wait_list is not NULL and num_events_in_wait_list is 0
  cl_event event;
  status = clEnqueueFillBuffer(queue,
                               buffer,
                               &pattern,
                               sizeof(cl_int),
                               0,
                               buffer_size,
                               0,
                               &event,
                               nullptr); // event_wait_list is not NULL and num_events_in_wait_list is 0
  EXPECT_ERROR(status, CL_INVALID_EVENT_WAIT_LIST);

  EXPECT_SUCCESS(clReleaseMemObject(buffer));
  EXPECT_SUCCESS(clReleaseContext(context));

  return param_verification_test::finalize();
}
