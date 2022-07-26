#include "param_verification_test.hpp"
#include <array>
#include <algorithm>
#include <memory>

int main(int argc, char* argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  param_verification_test::setup(argc, argv, CL_MAKE_VERSION(1, 2, 0), platform, device);

  // Make some extra devices so we can test
  constexpr const cl_uint num_sub_devices = 2;

  {
    size_t size;
    EXPECT_SUCCESS(clGetDeviceInfo(device, CL_DEVICE_PARTITION_PROPERTIES, 0, nullptr, &size));
    size_t len = size / sizeof(cl_device_partition_property);
    auto properties = std::make_unique<cl_device_partition_property[]>(len);
    EXPECT_SUCCESS(clGetDeviceInfo(device, CL_DEVICE_PARTITION_PROPERTIES, size, properties.get(), nullptr));

    if (std::find(properties.get(), properties.get() + len, CL_DEVICE_PARTITION_EQUALLY) == properties.get() + len) {
      LAYERS_TEST_LOG() << "test device does not support CL_DEVICE_PARTITION_EQUALLY" << std::endl;
      param_verification_test::TEST_CONTEXT.fail();
      return param_verification_test::finalize();
    }
  }

  cl_uint num_cus;
  EXPECT_SUCCESS(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_cus, nullptr));
  if (num_cus < num_sub_devices) {
    LAYERS_TEST_LOG() << "test device does not have enough compute units" << std::endl;
    param_verification_test::TEST_CONTEXT.fail();
    return param_verification_test::finalize();
  }

  cl_uint num_sub_cus = num_cus / num_sub_devices;
  cl_device_partition_property props[] = {CL_DEVICE_PARTITION_EQUALLY, static_cast<cl_device_partition_property>(num_sub_cus), 0};
  std::array<cl_device_id, num_sub_devices> sub_devices;
  EXPECT_SUCCESS(clCreateSubDevices(device,
                                    props,
                                    static_cast<cl_uint>(sub_devices.size()),
                                    sub_devices.data(),
                                    nullptr));
  cl_device_id device_a = sub_devices[0];
  cl_device_id device_b = sub_devices[1];

  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
  cl_context context_a = clCreateContext(properties, 1, &device_a, nullptr, nullptr, &status);
  EXPECT_SUCCESS(status);

  cl_context context_b = clCreateContext(properties, 1, &device_b, nullptr, nullptr, &status);
  EXPECT_SUCCESS(status);

  // Sanity check
  cl_command_queue queue_a = clCreateCommandQueue(context_a,
                                                  device_a,
                                                  0,
                                                  &status);
  EXPECT_SUCCESS(status);

  // CL_INVALID_DEVICE if device is not a valid device or is not associated with context.
  clCreateCommandQueue(context_a,
                       device_b,
                       0,
                       &status); // device is not associated with context
  EXPECT_ERROR(status, CL_INVALID_DEVICE);

  cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };
  cl_image_desc desc = {
    CL_MEM_OBJECT_IMAGE2D, // image_type
    4,                     // image_width
    4,                     // image_height
    1,                     // image_depth
    1,                     // image_array size
    0,                     // image_row_pitch
    0,                     // image_slice_pitch
    0,                     // num_mip_levels
    0,                     // num_samples
    { nullptr }            // mem_object
  };
  cl_mem image_a = clCreateImage(context_a,
                                 CL_MEM_READ_WRITE,
                                 &format,
                                 &desc,
                                 nullptr,
                                 &status);
  EXPECT_SUCCESS(status);

  cl_mem buffer_a = clCreateBuffer(context_a,
                                   CL_MEM_READ_WRITE,
                                   64,
                                   nullptr,
                                   &status);
  EXPECT_SUCCESS(status);

  cl_mem buffer_b = clCreateBuffer(context_b,
                                   CL_MEM_READ_WRITE,
                                   64,
                                   nullptr,
                                   &status);

  cl_event event_a0 = clCreateUserEvent(context_a, &status);
  EXPECT_SUCCESS(status);
  cl_event event_a1 = clCreateUserEvent(context_a, &status);
  EXPECT_SUCCESS(status);
  std::array<cl_event, 2> events_a = { event_a0, event_a1 };

  cl_event event_b = clCreateUserEvent(context_b, &status);
  EXPECT_SUCCESS(status);

  auto enqueue_copy = [&](cl_mem src_image,
                          cl_mem dst_buffer,
                          size_t num_events,
                          const cl_event* events) {
    size_t src_origin[] = {0, 0, 0};
    size_t region[] = {1, 1, 1};
    return clEnqueueCopyImageToBuffer(queue_a,
                                      src_image,
                                      dst_buffer,
                                      src_origin,
                                      region,
                                      0,
                                      static_cast<cl_uint>(num_events),
                                      events,
                                      nullptr);
  };

  // Sanity check
  status = enqueue_copy(image_a, buffer_a, events_a.size(), events_a.data());
  EXPECT_SUCCESS(status);

  // CL_INVALID_CONTEXT if the context associated with command_queue, src_image or dst_buffer is not the same.
  status = enqueue_copy(image_a, buffer_b, events_a.size(), events_a.data()); // context associated with command_queue and dst_buffer are not the same
  EXPECT_ERROR(status, CL_INVALID_CONTEXT);

  // CL_INVALID_CONTEXT if the context associated with command_queue and events in event_wait_list are not the same.
  std::array<cl_event, 3> all_events = { event_a1, event_b, event_a0 };
  status = enqueue_copy(image_a, buffer_a, all_events.size(), all_events.data()); // the context associated with command_queue and events in event_wait_list are not the same
  EXPECT_ERROR(status, CL_INVALID_CONTEXT);

  EXPECT_SUCCESS(clReleaseEvent(event_a0));
  EXPECT_SUCCESS(clReleaseEvent(event_a1));
  EXPECT_SUCCESS(clReleaseEvent(event_b));
  EXPECT_SUCCESS(clReleaseMemObject(image_a));
  EXPECT_SUCCESS(clReleaseCommandQueue(queue_a));
  EXPECT_SUCCESS(clReleaseContext(context_a));
  EXPECT_SUCCESS(clReleaseContext(context_b));
  for (auto sub_device : sub_devices) {
    EXPECT_SUCCESS(clReleaseDevice(sub_device));
  }

  return param_verification_test::finalize();
}
