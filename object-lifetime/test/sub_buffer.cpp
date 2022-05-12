#include "object_lifetime_test.hpp"

int main(int argc, char *argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  layer_test::setup(argc, argv, CL_MAKE_VERSION(1, 1, 0), platform, device);

  cl_context context = layer_test::createContext(platform, device);

  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 64, nullptr, &status);
  EXPECT_SUCCESS(status);

  // Release context, the buffer should keep it alive.
  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 0, 1);

  cl_buffer_region sub_region = {0, 32};
  cl_mem sub_buffer = clCreateSubBuffer(buffer,
                                        CL_MEM_READ_WRITE,
                                        CL_BUFFER_CREATE_TYPE_REGION,
                                        static_cast<const void*>(&sub_region),
                                        &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(sub_buffer, 1, 0);
  EXPECT_REF_COUNT(buffer, 1, 1);

  // Release buffer, the child should keep it alive
  EXPECT_SUCCESS(clReleaseMemObject(buffer));
  EXPECT_REF_COUNT(sub_buffer, 1, 0);
  EXPECT_REF_COUNT(buffer, 0, 1);
  EXPECT_REF_COUNT(context, 0, 1);

  // Release the entire chain by releasing sub_buffer
  EXPECT_SUCCESS(clReleaseMemObject(sub_buffer));
  EXPECT_DESTROYED(sub_buffer); // recently deleted with type: BUFFER
  EXPECT_DESTROYED(buffer); // recently deleted with type: BUFFER
  EXPECT_DESTROYED(context); // recently deleted with type: CONTEXT

  return layer_test::finalize();
}
