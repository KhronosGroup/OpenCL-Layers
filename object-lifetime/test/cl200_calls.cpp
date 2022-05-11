#include "object_lifetime_test.hpp"

int main(int argc, char *argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  layer_test::setup(argc, argv, CL_MAKE_VERSION(2, 0, 0), platform, device);

  cl_context context = layer_test::createContext(platform, device);

  cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT , 0};
  cl_command_queue queue_a = clCreateCommandQueueWithProperties(context,
                                                                device,
                                                                props,
                                                                &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(queue_a, 1, 0);

  // Creating extra queues with CL_QUEUE_ON_DEVICE_DEFAULT will return the same queue and increment its reference count.
  cl_command_queue queue_b = clCreateCommandQueueWithProperties(context,
                                                                device,
                                                                props,
                                                                &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(queue_a, 2, 0);
  EXPECT_REF_COUNT(queue_b, 2, 0);

  EXPECT_SUCCESS(clReleaseCommandQueue(queue_a));
  EXPECT_REF_COUNT(queue_a, 1, 0);
  EXPECT_REF_COUNT(queue_b, 1, 0);

  EXPECT_SUCCESS(clReleaseCommandQueue(queue_b));
  EXPECT_DESTROYED(queue_a);
  EXPECT_DESTROYED(queue_b);

  cl_sampler sampler = clCreateSamplerWithProperties(context, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(sampler, 1, 0);
  EXPECT_SUCCESS(clReleaseSampler(sampler));
  EXPECT_DESTROYED(sampler);

  cl_mem pipe = clCreatePipe(context, CL_MEM_READ_WRITE, 8, 16, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(pipe, 1, 0);
  EXPECT_SUCCESS(clReleaseMemObject(pipe));
  EXPECT_DESTROYED(pipe);

  return layer_test::finalize();
}

