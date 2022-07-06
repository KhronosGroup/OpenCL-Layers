#include "param_verification_test.hpp"
#include <array>
#include <algorithm>
#include <memory>

int main(int argc, char* argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  param_verification_test::setup(argc, argv, CL_MAKE_VERSION(1, 2, 0), platform, device);

  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
  cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, &status);
  EXPECT_SUCCESS(status);

  auto create_image = [&](cl_image_format format, cl_image_desc desc) {
    return clCreateImage(context,
                         CL_MEM_READ_WRITE,
                         &format,
                         &desc,
                         nullptr,
                         &status);
  };

  auto create_image_from_pixels = [&](cl_image_format format, cl_image_desc desc, void* pixels) {
    return clCreateImage(context,
                         CL_MEM_READ_WRITE,
                         &format,
                         &desc,
                         pixels,
                         &status);
  };

  // Sanity check
  cl_mem image_a = create_image({ CL_RGBA,
                                  CL_UNORM_INT8 },
                                { CL_MEM_OBJECT_IMAGE2D, // image_type
                                  4,                     // image_width
                                  4,                     // image_height
                                  1,                     // image_depth
                                  1,                     // image_array size
                                  0,                     // image_row_pitch
                                  0,                     // image_slice_pitch
                                  0,                     // num_mip_levels
                                  0,                     // num_samples
                                  { nullptr } });        // mem_object
  EXPECT_SUCCESS(status);

  // CL_INVALID_IMAGE_FORMAT_DESCRIPTOR or CL_INVALID_IMAGE_DESCRIPTOR
  // if image_format or image_desc is not valid in some way.

  // values specified in image_format are not valid
  create_image({ 123456,
                 CL_UNORM_INT8 },
               { CL_MEM_OBJECT_IMAGE2D, // image_type
                 4,                     // image_width
                 4,                     // image_height
                 1,                     // image_depth
                 1,                     // image_array size
                 0,                     // image_row_pitch
                 0,                     // image_slice_pitch
                 0,                     // num_mip_levels
                 0,                     // num_samples
                 { nullptr } });        // mem_object
  EXPECT_ERROR(status, CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);

  // values specified in image_format are not valid
  create_image({ CL_RGBA,
                 CL_UNORM_SHORT_565 },  // Note: CL_UNORM_SHORT requires CL_RGB or CL_RGBx
               { CL_MEM_OBJECT_IMAGE2D, // image_type
                 4,                     // image_width
                 4,                     // image_height
                 1,                     // image_depth
                 1,                     // image_array size
                 0,                     // image_row_pitch
                 0,                     // image_slice_pitch
                 0,                     // num_mip_levels
                 0,                     // num_samples
                 { nullptr } });        // mem_object
  EXPECT_ERROR(status, CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);

  // values specified in image_desc are not valid
  create_image({ CL_RGBA,
                 CL_UNORM_INT8 },
               { 123456,                // image_type
                 4,                     // image_width
                 4,                     // image_height
                 1,                     // image_depth
                 1,                     // image_array size
                 0,                     // image_row_pitch
                 0,                     // image_slice_pitch
                 0,                     // num_mip_levels
                 0,                     // num_samples
                 { nullptr } });        // mem_object
  EXPECT_ERROR(status, CL_INVALID_IMAGE_DESCRIPTOR);

  // values specified in image_desc are not valid
  create_image({ CL_RGBA,
                 CL_UNORM_INT8 },
               { CL_MEM_OBJECT_IMAGE2D, // image_type
                 0,                     // image_width
                 4,                     // image_height
                 1,                     // image_depth
                 1,                     // image_array size
                 0,                     // image_row_pitch
                 0,                     // image_slice_pitch
                 0,                     // num_mip_levels
                 0,                     // num_samples
                 { nullptr } });        // mem_object
  EXPECT_ERROR(status, CL_INVALID_IMAGE_DESCRIPTOR);

  // values specified in image_desc are not valid
  create_image({ CL_RGBA,
                 CL_UNORM_INT8 },
               { CL_MEM_OBJECT_IMAGE2D, // image_type
                 4,                     // image_width
                 4,                     // image_height
                 1,                     // image_depth
                 1,                     // image_array size
                 0,                     // image_row_pitch
                 0,                     // image_slice_pitch
                 0,                     // num_mip_levels
                 1,                     // num_samples
                 { nullptr } });        // mem_object
  EXPECT_ERROR(status, CL_INVALID_IMAGE_DESCRIPTOR);

  // values specified in image_desc are not valid
  create_image({ CL_RGBA,
                 CL_UNORM_INT8 },
               { CL_MEM_OBJECT_IMAGE1D, // image_type
                 4,                     // image_width
                 1,                     // image_height
                 1,                     // image_depth
                 1,                     // image_array size
                 0,                     // image_row_pitch
                 0,                     // image_slice_pitch
                 0,                     // num_mip_levels
                 1,                     // num_samples
                 { image_a } });        // mem_object
  EXPECT_ERROR(status, CL_INVALID_IMAGE_DESCRIPTOR);

  // values specified in image_desc are not valid
  create_image({ CL_RGBA,
                 CL_UNORM_INT8 },
               { CL_MEM_OBJECT_IMAGE2D, // image_type
                 4,                     // image_width
                 4,                     // image_height
                 1,                     // image_depth
                 1,                     // image_array size
                 1,                     // image_row_pitch
                 0,                     // image_slice_pitch
                 0,                     // num_mip_levels
                 0,                     // num_samples
                 { nullptr } });        // mem_object
  EXPECT_ERROR(status, CL_INVALID_IMAGE_DESCRIPTOR);

  cl_uint pixels = 0xFF00FF00;
  create_image_from_pixels({ CL_RGBA,
                             CL_UNORM_INT8 },
                           { CL_MEM_OBJECT_IMAGE2D, // image_type
                             4,                     // image_width
                             4,                     // image_height
                             1,                     // image_depth
                             1,                     // image_array size
                             1,                     // image_row_pitch
                             0,                     // image_slice_pitch
                             0,                     // num_mip_levels
                             0,                     // num_samples
                             { nullptr } },         // mem_object
                             &pixels); // values specified in image_desc are not valid
  EXPECT_ERROR(status, CL_INVALID_IMAGE_DESCRIPTOR);

  // a 2D image is created from a 2D image object but image descriptors are not compatible
  create_image({ CL_RGBA,
                 CL_UNORM_INT8 },
               { CL_MEM_OBJECT_IMAGE2D, // image_type
                 4,                     // image_width
                 3,                     // image_height
                 1,                     // image_depth
                 1,                     // image_array size
                 0,                     // image_row_pitch
                 0,                     // image_slice_pitch
                 0,                     // num_mip_levels
                 0,                     // num_samples
                 { image_a } });        // mem_object
  EXPECT_ERROR(status, CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);


  cl_command_queue queue = clCreateCommandQueue(context,
                                                device,
                                                0,
                                                &status);
  EXPECT_SUCCESS(status);

  cl_mem image_b = create_image({ CL_RG,
                                  CL_UNORM_INT16 },
                                { CL_MEM_OBJECT_IMAGE2D, // image_type
                                  4,                     // image_width
                                  4,                     // image_height
                                  1,                     // image_depth
                                  1,                     // image_array size
                                  0,                     // image_row_pitch
                                  0,                     // image_slice_pitch
                                  0,                     // num_mip_levels
                                  0,                     // num_samples
                                  { nullptr } });        // mem_object
  EXPECT_SUCCESS(status);

  // CL_IMAGE_FORMAT_MISMATCH if src_image and dst_image do not use the same image format.
  size_t origin[] = {0, 0, 0};
  size_t region[] = {4, 4, 1};
  status = clEnqueueCopyImage(queue,
                              image_a,
                              image_b,
                              origin,
                              origin,
                              region,
                              0,
                              nullptr,
                              nullptr); // src_image and dst_image do not use the same image format
  EXPECT_ERROR(status, CL_IMAGE_FORMAT_MISMATCH);

  EXPECT_SUCCESS(clReleaseCommandQueue(queue));
  EXPECT_SUCCESS(clReleaseMemObject(image_a));
  EXPECT_SUCCESS(clReleaseContext(context));

  return param_verification_test::finalize();
}
