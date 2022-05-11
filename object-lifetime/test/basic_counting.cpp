#include "object_lifetime_test.hpp"

int main(int argc, char *argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  layer_test::setup(argc, argv, CL_MAKE_VERSION(1, 1, 0), platform, device);

  cl_context context = layer_test::createContext(platform, device);

  // Create some buffers to test with
  cl_mem buffer_a = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(buffer_a, 1, 0);
  EXPECT_REF_COUNT(context, 1, 1);

  cl_mem buffer_b = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 1, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(buffer_a, 1, 0);
  EXPECT_REF_COUNT(context, 1, 2);

  EXPECT_SUCCESS(clRetainContext(context));
  EXPECT_REF_COUNT(buffer_a, 1, 0);
  EXPECT_REF_COUNT(buffer_b, 1, 0);
  EXPECT_REF_COUNT(context, 2, 2);

  EXPECT_SUCCESS(clRetainMemObject(buffer_a));
  EXPECT_REF_COUNT(buffer_a, 2, 0);
  EXPECT_REF_COUNT(buffer_b, 1, 0);
  EXPECT_REF_COUNT(context, 2, 2);

  EXPECT_SUCCESS(clRetainMemObject(buffer_b));
  EXPECT_REF_COUNT(buffer_a, 2, 0);
  EXPECT_REF_COUNT(buffer_b, 2, 0);
  EXPECT_REF_COUNT(context, 2, 2);

  EXPECT_SUCCESS(clRetainContext(context));
  EXPECT_REF_COUNT(context, 3, 2);

  EXPECT_SUCCESS(clReleaseMemObject(buffer_b));
  EXPECT_REF_COUNT(buffer_a, 2, 0);
  EXPECT_REF_COUNT(buffer_b, 1, 0);
  EXPECT_REF_COUNT(context, 3, 2);

  EXPECT_SUCCESS(clReleaseMemObject(buffer_b));
  EXPECT_REF_COUNT(buffer_a, 2, 0);
  EXPECT_DESTROYED(buffer_b);
  EXPECT_REF_COUNT(context, 3, 1);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 2, 1);
  EXPECT_REF_COUNT(buffer_a, 2, 0);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 1, 1);
  EXPECT_REF_COUNT(buffer_a, 2, 0);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 0, 1);
  EXPECT_REF_COUNT(buffer_a, 2, 0);

  EXPECT_SUCCESS(clReleaseMemObject(buffer_a));
  EXPECT_REF_COUNT(buffer_a, 1, 0);
  EXPECT_REF_COUNT(context, 0, 1);

  EXPECT_SUCCESS(clReleaseMemObject(buffer_a));
  EXPECT_DESTROYED(buffer_a);
  EXPECT_DESTROYED(context);

  return layer_test::finalize();
}
