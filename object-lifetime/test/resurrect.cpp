#include "object_lifetime_test.hpp"

int main(int argc, char *argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  object_lifetime_test::setup(argc, argv, CL_MAKE_VERSION(3, 0, 0), platform, device);

  cl_context context = object_lifetime_test::createContext(platform, device);

  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 32, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(buffer, 1, 0);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 0, 1); // used with implicit refcount: 1

  // Try to 'resurrect' the context
  EXPECT_SUCCESS(clRetainContext(context));
  EXPECT_REF_COUNT(context, 1, 1);
  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 0, 1); // used with implicit refcount: 1

  EXPECT_SUCCESS(clReleaseMemObject(buffer));
  EXPECT_DESTROYED(buffer); // recently deleted with type: BUFFER
  EXPECT_DESTROYED(context); // recently deleted with type: CONTEXT

  return object_lifetime_test::finalize();
}
