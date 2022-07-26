#include "param_verification_test.hpp"
#include <array>
#include <algorithm>
#include <memory>

int main(int argc, char* argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  param_verification_test::setup(argc, argv, CL_MAKE_VERSION(2, 0, 0), platform, device);

  cl_uint queue_on_device_max_size;
  EXPECT_SUCCESS(clGetDeviceInfo(device,
                                 CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE,
                                 sizeof(queue_on_device_max_size),
                                 &queue_on_device_max_size,
                                 nullptr));
  if (queue_on_device_max_size == 0) {
    LAYERS_TEST_LOG() << "Test requires CL_DEVICE_QUEUE_SUPPORTED" << std::endl;
    param_verification_test::TEST_CONTEXT.fail();
    return param_verification_test::finalize();
  }

  if (param_verification_test::TEST_CONTEXT.version >= CL_MAKE_VERSION(3, 0, 0)) {
    cl_device_device_enqueue_capabilities caps;
    EXPECT_SUCCESS(clGetDeviceInfo(device,
                                   CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES,
                                   sizeof(caps),
                                   &caps,
                                   nullptr));
    if ((caps & CL_DEVICE_QUEUE_SUPPORTED) == 0) {
      LAYERS_TEST_LOG() << "Test requires CL_DEVICE_QUEUE_SUPPORTED" << std::endl;
      param_verification_test::TEST_CONTEXT.fail();
      return param_verification_test::finalize();
    }
  }

  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
  cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, &status);
  EXPECT_SUCCESS(status);

  // CL_INVALID_PROPERTY if context property name in properties is not a supported property name
  cl_context_properties incorrect_properties[] = {123456, (cl_context_properties) platform, 0};
  clCreateContext(incorrect_properties, 1, &device, nullptr, nullptr, &status); // context property name in properties is not a supported property name or the same property name is specified more than once
  EXPECT_ERROR(status, CL_INVALID_PROPERTY);

  // CL_INVALID_PROPERTY if the same property name is specified more than once
  cl_context_properties duplicate_properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform, CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
  clCreateContext(duplicate_properties, 1, &device, nullptr, nullptr, &status); // context property name in properties is not a supported property name or the same property name is specified more than once
  EXPECT_ERROR(status, CL_INVALID_PROPERTY);

  // Sanity check
  cl_sampler_properties sampler_properties[] = { 0 };
  cl_sampler sampler = clCreateSamplerWithProperties(context,
                                                     sampler_properties,
                                                     &status);
  EXPECT_SUCCESS(status);

  // CL_INVALID_VALUE if the property name in sampler_properties is not a supported property name.
  clCreateSamplerWithProperties(context,
                                reinterpret_cast<cl_sampler_properties*>(properties),
                                &status); // the property name in sampler_properties is not a supported property name, if the value specified for a supported property name is not valid, or if the same property name is specified more than once
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  // CL_INVALID_VALUE if the value specified for a supported property name is not valid
  cl_sampler_properties invalid_sampler_properties[] = { CL_SAMPLER_FILTER_MODE, 123456, 0 };
  clCreateSamplerWithProperties(context,
                                invalid_sampler_properties,
                                &status); // the property name in sampler_properties is not a supported property name, if the value specified for a supported property name is not valid, or if the same property name is specified more than once
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  cl_queue_properties queue_properties[] = { CL_QUEUE_PROPERTIES,
                                             CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE,
                                             CL_QUEUE_SIZE, queue_on_device_max_size,
                                             0 };
  cl_command_queue queue = clCreateCommandQueueWithProperties(context,
                                                              device,
                                                              queue_properties,
                                                              &status);
  EXPECT_SUCCESS(status);

  // CL_INVALID_VALUE if the values specified in properties are not valid
  // Note: CL_QUEUE_ON_DEVICE requires CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
  cl_queue_properties invalid_queue_properties[] = { CL_QUEUE_ON_DEVICE };
  clCreateCommandQueueWithProperties(context,
                                     device,
                                     invalid_queue_properties,
                                     &status); // values specified in properties are not valid
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  cl_queue_properties more_invalid_queue_properties[] = { CL_QUEUE_PROPERTIES,
                                                          CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE,
                                                          CL_QUEUE_SIZE, queue_on_device_max_size + 1,
                                                          0 };
  clCreateCommandQueueWithProperties(context,
                                     device,
                                     more_invalid_queue_properties,
                                     &status); // values specified in properties are valid but are not supported by the device
  EXPECT_ERROR(status, CL_INVALID_QUEUE_PROPERTIES);

  EXPECT_SUCCESS(clReleaseCommandQueue(queue));
  EXPECT_SUCCESS(clReleaseSampler(sampler));
  EXPECT_SUCCESS(clReleaseContext(context));

  return param_verification_test::finalize();
}
