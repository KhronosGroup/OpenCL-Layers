#include "object_lifetime_test.hpp"

int main(int argc, char *argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  layer_test::setup(argc, argv, CL_MAKE_VERSION(1, 1, 0), platform, device);

  cl_context context = layer_test::createContext(platform, device);

  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(buffer, 1, 0);
  EXPECT_REF_COUNT(context, 1, 1);

  cl_uint refcount;
  status = clGetContextInfo(reinterpret_cast<cl_context>(buffer),
                            CL_CONTEXT_REFERENCE_COUNT,
                            sizeof(refcount),
                            &refcount,
                            nullptr); // BUFFER was used whereas function expects: CONTEXT
  EXPECT_ERROR(status, CL_INVALID_CONTEXT);


  EXPECT_SUCCESS(clReleaseMemObject(buffer));
  EXPECT_DESTROYED(buffer);

  EXPECT_SUCCESS(clReleaseContext(context)); // recently deleted with type: BUFFER
  EXPECT_DESTROYED(context); // recently deleted with type: CONTEXT

  return layer_test::finalize();
}
