#include "object_lifetime_test.hpp"

int main(int argc, char *argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  layer_test::setup(argc, argv, CL_MAKE_VERSION(1, 1, 0), platform, device);

  cl_context context = layer_test::createContext(platform, device);

  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 32, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(buffer, 1, 0);

  cl_sampler sampler = clCreateSampler(context, CL_TRUE, CL_ADDRESS_CLAMP, CL_FILTER_LINEAR, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(sampler, 1, 0);

  cl_event event = clCreateUserEvent(context, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(event, 1, 0);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 0, 2);

  EXPECT_SUCCESS(clReleaseMemObject(buffer));
  EXPECT_DESTROYED(buffer); // recently deleted with type: BUFFER
  EXPECT_REF_COUNT(context, 0, 1);

  // Leak sampler and event

  return layer_test::finalize();
}
