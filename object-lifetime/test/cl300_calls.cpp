#include "object_lifetime_test.hpp"

int main(int argc, char *argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  layer_test::setup(argc, argv, CL_MAKE_VERSION(3, 0, 0), platform, device);

  cl_context context = layer_test::createContext(platform, device);

  cl_mem buffer = clCreateBufferWithProperties(context,
                                               nullptr,
                                               CL_MEM_READ_WRITE,
                                               4,
                                               nullptr,
                                               &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(context, 1, 1);
  EXPECT_REF_COUNT(buffer, 1, 0);

  cl_image_desc desc = {
    CL_MEM_OBJECT_IMAGE1D_BUFFER, // image_type
    4,                            // image_width
    1,                            // image_height
    1,                            // image_depth
    1,                            // image_array size
    0,                            // image_row_pitch
    0,                            // image_slice_pitch
    0,                            // num_mip_levels
    0,                            // num_samples
    buffer                        // mem_object
  };
  cl_image_format format = {CL_R, CL_UNSIGNED_INT8};
  cl_mem image = clCreateImageWithProperties(context,
                                             nullptr,
                                             CL_MEM_READ_WRITE,
                                             &format,
                                             &desc,
                                             nullptr,
                                             &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(image, 1, 0);
  EXPECT_REF_COUNT(buffer, 1, 1);

  EXPECT_SUCCESS(clReleaseMemObject(buffer));
  EXPECT_REF_COUNT(buffer, 0, 1);
  EXPECT_SUCCESS(clReleaseMemObject(image));
  EXPECT_DESTROYED(image);
  EXPECT_DESTROYED(buffer);
  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_DESTROYED(context);

  return layer_test::finalize();
}

