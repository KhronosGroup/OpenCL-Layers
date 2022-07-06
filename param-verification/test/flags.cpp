#include "param_verification_test.hpp"

int main(int argc, char* argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  param_verification_test::setup(argc, argv, CL_MAKE_VERSION(1, 2, 0), platform, device);

  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
  cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, &status);
  EXPECT_SUCCESS(status);

  cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };
  cl_image_desc desc = {
    CL_MEM_OBJECT_IMAGE2D, // image_type
    2,                     // image_width
    2,                     // image_height
    1,                     // image_depth
    1,                     // image_array size
    0,                     // image_row_pitch
    0,                     // image_slice_pitch
    0,                     // num_mip_levels
    0,                     // num_samples
    { nullptr }            // mem_object
  };

  // Sanity check
  cl_mem image = clCreateImage(context,
                               CL_MEM_READ_WRITE,
                               &format,
                               &desc,
                               nullptr,
                               &status);
  EXPECT_SUCCESS(status);
  EXPECT_SUCCESS(clReleaseMemObject(image));

  // CL_INVALID_VALUE if flags are not valid
  clCreateImage(context,
                static_cast<cl_mem_flags>(1 << 30),
                &format,
                &desc,
                nullptr,
                &status); // values specified in flags are not valid
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  // CL_MEM_READ_WRITE and CL_MEM_WRITE_ONLY are mutually exclusive
  clCreateImage(context,
                CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY,
                &format,
                &desc,
                nullptr,
                &status); // values specified in flags are not compatible
  EXPECT_ERROR(status, CL_INVALID_VALUE);

    // CL_MEM_USE_HOST_PTR is only valid if host_ptr is not null
  clCreateImage(context,
                CL_MEM_USE_HOST_PTR,
                &format,
                &desc,
                nullptr,
                &status); // host_ptr is NULL and CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR are set in flags
  EXPECT_ERROR(status, CL_INVALID_HOST_PTR);

  // Other way around also
  cl_uint pixel = 0xFF00FF00;
  clCreateImage(context,
                CL_MEM_READ_ONLY,
                &format,
                &desc,
                &pixel,
                &status); // host_ptr is not NULL but CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are not set in flags

  cl_mem buffer = clCreateBuffer(context,
                                 CL_MEM_READ_ONLY,
                                 128,
                                 nullptr,
                                 &status);
  EXPECT_SUCCESS(status);

  // Sanity check
  cl_buffer_region region = { 0, 128 };
  cl_mem sub_buffer = clCreateSubBuffer(buffer,
                                        CL_MEM_READ_ONLY,
                                        CL_BUFFER_CREATE_TYPE_REGION,
                                        &region,
                                        &status);

  EXPECT_SUCCESS(clReleaseMemObject(sub_buffer));

  // CL_INVALID_VALUE if the value specified in buffer_create_type is not valid
  clCreateSubBuffer(buffer,
                    CL_MEM_READ_ONLY,
                    123456,
                    &region,
                    &status); // the value specified in buffer_create_type is not valid
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  // CL_INVALID_VALUE if buffer was created with CL_MEM_READ_ONLY and flags specifies CL_MEM_READ_WRITE or CL_MEM_WRITE_ONLY
  clCreateSubBuffer(buffer,
                    CL_MEM_READ_WRITE,
                    CL_BUFFER_CREATE_TYPE_REGION,
                    &region,
                    &status); // TODO
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  EXPECT_SUCCESS(clReleaseMemObject(buffer));

  size_t size;
  status = clGetContextInfo(context,
                            123456,
                            0,
                            nullptr,
                            &size);
  EXPECT_ERROR(status, CL_INVALID_VALUE);

  EXPECT_SUCCESS(clReleaseContext(context));

  return param_verification_test::finalize();
}
