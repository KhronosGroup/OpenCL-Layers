#include "object_lifetime_test.hpp"

int main(int argc, char *argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  object_lifetime_test::setup(argc, argv, CL_MAKE_VERSION(1, 1, 0), platform, device);

  cl_context context = object_lifetime_test::createContext(platform, device);

  cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };
  cl_mem image_2d = clCreateImage2D(context, CL_MEM_READ_ONLY, &format, 1, 1, 0, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(image_2d, 1, 0);
  EXPECT_REF_COUNT(context, 1, 1);

  cl_mem image_3d = clCreateImage3D(context, CL_MEM_READ_ONLY, &format, 1, 1, 2, 0, 0, nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(image_3d, 1, 0);
  EXPECT_REF_COUNT(context, 1, 2);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 0, 2); // used with implicit refcount: 2

  // Try to free the context, which is now only implicitly retained.
  EXPECT_ERROR(clReleaseContext(context), CL_INVALID_CONTEXT); // released before being retained
  // That should not influence the implicit ref count
  EXPECT_REF_COUNT(context, 0, 2); // used with implicit refcount: 2

  EXPECT_SUCCESS(clReleaseMemObject(image_2d));
  EXPECT_DESTROYED(image_2d); // recently deleted with type: IMAGE
  EXPECT_REF_COUNT(context, 0, 1); // used with implicit refcount: 1

  EXPECT_SUCCESS(clReleaseMemObject(image_3d));
  EXPECT_DESTROYED(image_3d); // recently deleted with type: IMAGE
  EXPECT_DESTROYED(context); // recently deleted with type: CONTEXT

  // Double free should fail.
  EXPECT_ERROR(clReleaseMemObject(image_2d), CL_INVALID_MEM_OBJECT); // recently deleted with type: IMAGE
  EXPECT_ERROR(clReleaseContext(context), CL_INVALID_CONTEXT); // recently deleted with type: CONTEXT

  return object_lifetime_test::finalize();
}
