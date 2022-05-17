#include "object_lifetime_test.hpp"
#include <memory>
#include <algorithm>
#include <array>

int main(int argc, char *argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  layer_test::setup(argc, argv, CL_MAKE_VERSION(1, 2, 0), platform, device);

  cl_context context = layer_test::createContext(platform, device);

  constexpr const cl_uint num_sub_devices = 2;
  constexpr const cl_uint num_sub_sub_devices_per_sub_device = 2;
  constexpr const cl_uint min_cus = num_sub_devices * num_sub_sub_devices_per_sub_device;
  constexpr const cl_uint num_sub_sub_devices = num_sub_devices * num_sub_sub_devices_per_sub_device;
  static_assert(num_sub_sub_devices >= 4, "test requires at least 4 sub-sub-devices");

  {
    size_t size;
    EXPECT_SUCCESS(clGetDeviceInfo(device, CL_DEVICE_PARTITION_PROPERTIES, 0, nullptr, &size));
    size_t len = size / sizeof(cl_device_partition_property);
    auto properties = std::make_unique<cl_device_partition_property[]>(len);
    EXPECT_SUCCESS(clGetDeviceInfo(device, CL_DEVICE_PARTITION_PROPERTIES, size, properties.get(), nullptr));

    if (std::find(properties.get(), properties.get() + len, CL_DEVICE_PARTITION_EQUALLY) == properties.get() + len) {
      LAYER_TEST_LOG() << "test device does not support CL_DEVICE_PARTITION_EQUALLY" << std::endl;
      layer_test::TEST_CONTEXT.fail();
      return layer_test::finalize();
    }
  }

  cl_uint num_cus;
  EXPECT_SUCCESS(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_cus, nullptr));
  if (num_cus < min_cus) {
    LAYER_TEST_LOG() << "test device does not have enough compute units" << std::endl;
    layer_test::TEST_CONTEXT.fail();
    return layer_test::finalize();
  }

  cl_uint num_sub_cus = num_cus / num_sub_devices;
  cl_device_partition_property props[] = {CL_DEVICE_PARTITION_EQUALLY, static_cast<cl_device_partition_property>(num_sub_cus), 0};
  std::array<cl_device_id, num_sub_devices> sub_devices;
  EXPECT_SUCCESS(clCreateSubDevices(device,
                                    props,
                                    static_cast<cl_uint>(sub_devices.size()),
                                    sub_devices.data(),
                                    nullptr));

  for (auto sub_device : sub_devices) {
    EXPECT_REF_COUNT(sub_device, 1, 0);
  }

  // Create two sub-sub-devices for each sub-device
  std::array<cl_device_id, num_sub_sub_devices> sub_sub_devices;
  for (size_t i = 0; i < num_sub_devices; ++i) {
    cl_device_partition_property sub_props[] = {CL_DEVICE_PARTITION_EQUALLY, static_cast<cl_device_partition_property>(num_sub_cus / 2), 0};
    EXPECT_SUCCESS(clCreateSubDevices(sub_devices[i],
                                      sub_props,
                                      num_sub_sub_devices_per_sub_device,
                                      &sub_sub_devices[i * num_sub_sub_devices_per_sub_device],
                                      nullptr));
  }

  // Check ref counts of sub-sub-devices
  for (size_t i = 0; i < num_sub_sub_devices; ++i) {
    EXPECT_REF_COUNT(sub_sub_devices[i], 1, 0);
  }

  // Check ref counts of and delete sub-devices.
  for (size_t i = 0; i < num_sub_devices; ++i) {
    EXPECT_REF_COUNT(sub_devices[i], 1, num_sub_sub_devices_per_sub_device);
    EXPECT_SUCCESS(clReleaseDevice(sub_devices[i]));
    EXPECT_REF_COUNT(sub_devices[i], 0, num_sub_sub_devices_per_sub_device);
  }

  {
    // Root devices should always have a ref count of 1, which is unaffected by retains and releases
    EXPECT_REF_COUNT(device, 1, 0);

    EXPECT_SUCCESS(clReleaseDevice(device));
    EXPECT_REF_COUNT(device, 1, 0);

    EXPECT_SUCCESS(clRetainDevice(device));
    EXPECT_REF_COUNT(device, 1, 0);
  }

  {
    // Release the first device and check if the rest is okay.
    EXPECT_SUCCESS(clReleaseDevice(sub_sub_devices[0]));
    EXPECT_DESTROYED(sub_sub_devices[0]); // recently deleted with type: SUB_DEVICE
    EXPECT_REF_COUNT(sub_devices[0], 0, num_sub_sub_devices_per_sub_device - 1);

    for (cl_uint i = 1; i < num_sub_sub_devices; ++i) {
      EXPECT_REF_COUNT(sub_sub_devices[i], 1, 0);
    }

    for (cl_uint i = 1; i < num_sub_devices; ++i) {
      EXPECT_REF_COUNT(sub_devices[i], 0, num_sub_sub_devices_per_sub_device);
    }
  }

  {
    // Check that device dependencies keep the device alive.
    cl_context context1 = layer_test::createContext(platform, sub_sub_devices[1]);
    EXPECT_SUCCESS(clReleaseDevice(sub_sub_devices[1]));
    EXPECT_REF_COUNT(sub_sub_devices[1], 0, 1);

    EXPECT_SUCCESS(clReleaseContext(context1));
    EXPECT_DESTROYED(context1); // recently destroyed with type: CONTEXT
    EXPECT_DESTROYED(sub_sub_devices[1]); // recently destroyed with type: SUB_DEVICE

    for (cl_uint i = 2; i < num_sub_devices; ++i) {
      EXPECT_REF_COUNT(sub_sub_devices[i], 1, 0);
    }
  }

  {
    cl_uint used_devices = 2;
    cl_uint remaining_devices = num_sub_sub_devices - used_devices;
    cl_int status;
    // Create a context with the remaining devices, check that the context keeps them both alive.
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform), 0};
    cl_context context2 = clCreateContext(properties,
                                          remaining_devices,
                                          &sub_sub_devices[used_devices],
                                          nullptr,
                                          nullptr,
                                          &status);
    EXPECT_SUCCESS(status);
    EXPECT_REF_COUNT(context2, 1, 0);
    for (cl_uint i = used_devices; i < num_sub_sub_devices; ++i) {
      EXPECT_SUCCESS(clReleaseDevice(sub_sub_devices[i]));
      EXPECT_REF_COUNT(sub_sub_devices[i], 0, 1);
    }

    EXPECT_SUCCESS(clReleaseContext(context2));
    EXPECT_DESTROYED(context2);

    for (cl_uint i = used_devices; i < num_sub_sub_devices; ++i) {
      EXPECT_DESTROYED(sub_sub_devices[i]); // recently destroyed with type: SUB_DEVICE (2x)
    }
  }

  for (cl_uint i = 0; i < num_sub_devices; ++i) {
    EXPECT_DESTROYED(sub_devices[i]); // recently destroyed with type: SUB_DEVICE (2x)
  }

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_DESTROYED(context); // recently destroyed with type: CONTEXT

  return layer_test::finalize();
}

