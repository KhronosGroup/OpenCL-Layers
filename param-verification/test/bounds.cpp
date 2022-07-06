#include "param_verification_test.hpp"
#include <array>

int main(int argc, char* argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  param_verification_test::setup(argc, argv, CL_MAKE_VERSION(1, 1, 0), platform, device);

  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
  cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, &status);
  EXPECT_SUCCESS(status);

  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 64, nullptr, &status);
  EXPECT_SUCCESS(status);

  // CL_INVALID_BUFFER_SIZE if the size is zero
  cl_buffer_region zero_region = { 0, 0 };
  clCreateSubBuffer(buffer,
                    CL_MEM_READ_WRITE,
                    CL_BUFFER_CREATE_TYPE_REGION,
                    &zero_region,
                    &status); // the size field of the cl_buffer_region structure passed in buffer_create_info is 0
  EXPECT_ERROR(status, CL_INVALID_BUFFER_SIZE);

  // CL_INVALID_VALUE if the region is out of bounds
  cl_buffer_region too_large_region = { 0, 128 };
  clCreateSubBuffer(buffer,
                    CL_MEM_READ_WRITE,
                    CL_BUFFER_CREATE_TYPE_REGION,
                    &too_large_region,
                    &status); // the region specified by the cl_buffer_region structure passed in buffer_create_info is out of bounds in buffer
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  // Also check what happens if the size is right but offset + size is larger
  cl_buffer_region invalid_end_region = { 48, 32 };
  clCreateSubBuffer(buffer,
                    CL_MEM_READ_WRITE,
                    CL_BUFFER_CREATE_TYPE_REGION,
                    &invalid_end_region,
                    &status); // the region specified by the cl_buffer_region structure passed in buffer_create_info is out of bounds in buffer
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  // And if the start of the buffer is just out of bounds
  cl_buffer_region invalid_start_region = { 128, 32 };
  clCreateSubBuffer(buffer,
                    CL_MEM_READ_WRITE,
                    CL_BUFFER_CREATE_TYPE_REGION,
                    &invalid_start_region,
                    &status); // the region specified by the cl_buffer_region structure passed in buffer_create_info is out of bounds in buffer
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };
  cl_image_desc desc = {
    CL_MEM_OBJECT_IMAGE2D, // image_type
    4,                     // image_width
    4,                     // image_height
    1,                     // image_depth
    1,                     // image_array size
    0,                     // image_row_pitch
    0,                     // image_slice_pitch
    0,                     // num_mip_levels
    0,                     // num_samples
    { nullptr }            // mem_object
  };
  cl_mem src_image = clCreateImage(context,
                                   CL_MEM_READ_WRITE,
                                   &format,
                                   &desc,
                                   nullptr,
                                   &status);
  EXPECT_SUCCESS(status);

  cl_mem dst_image = clCreateImage(context,
                                   CL_MEM_READ_WRITE,
                                   &format,
                                   &desc,
                                   nullptr,
                                   &status);
  EXPECT_SUCCESS(status);

  cl_command_queue queue = clCreateCommandQueue(context,
                                                device,
                                                0,
                                                &status);
  EXPECT_SUCCESS(status);

  auto enqueue_copy = [&](
      std::array<size_t, 3> src_origin,
      std::array<size_t, 3> dst_origin,
      std::array<size_t, 3> region) {
    return clEnqueueCopyImage(queue,
                              src_image,
                              dst_image,
                              src_origin.data(),
                              dst_origin.data(),
                              region.data(),
                              0,
                              nullptr,
                              nullptr);
  };

  // Sanity check
  status = enqueue_copy({0, 0, 0}, {0, 0, 0}, {4, 4, 1});
  EXPECT_SUCCESS(status);

  // CL_INVALID_VALUE if the elements outside dimensions of the image are not 0/1
  status = enqueue_copy({0, 0, 1}, {0, 0, 0}, {4, 4, 1}); // wrong origin or region values for 2D image or 1D image array src_image
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  status = enqueue_copy({0, 0, 0}, {0, 0, 0}, {4, 4, 0}); // some region array element is 0
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  // CL_INVALID_VALUE if the origins are out of range
  status = enqueue_copy({1, 0, 0}, {0, 0, 0}, {4, 4, 1}); // the region being read specified by origin and region is out of bounds for 2D image src_image
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  status = enqueue_copy({0, 4, 0}, {0, 0, 0}, {4, 4, 1}); // the region being read specified by origin and region is out of bounds for 2D image src_image
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  status = enqueue_copy({0, 0, 0}, {3, 3, 0}, {1, 2, 1}); // the region being read specified by origin and region is out of bounds for 2D image dst_image
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  EXPECT_SUCCESS(clReleaseMemObject(src_image));
  EXPECT_SUCCESS(clReleaseMemObject(dst_image));
  EXPECT_SUCCESS(clReleaseMemObject(buffer));
  EXPECT_SUCCESS(clReleaseContext(context));

  return param_verification_test::finalize();
}
