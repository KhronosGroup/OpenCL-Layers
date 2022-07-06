#include "param_verification_test.hpp"

int main(int argc, char* argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  param_verification_test::setup(argc, argv, CL_MAKE_VERSION(1, 1, 0), platform, device);

  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
  cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, &status);
  EXPECT_SUCCESS(status);

  cl_ulong max_size;
  EXPECT_SUCCESS(clGetDeviceInfo(device,
                                 CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                                 sizeof(cl_ulong),
                                 &max_size,
                                 nullptr));

  // Sanity check
  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 64, nullptr, &status);
  EXPECT_SUCCESS(status);

  // CL_INVALID_BUFFER_SIZE if size is 0 or if size is greater than CL_DEVICE_MAX_MEM_ALLOC_SIZE for all devices in context.
  // queryies CL_DEVICE_MAX_MEM_ALLOC_SIZE from device, via context
  clCreateBuffer(context, CL_MEM_READ_WRITE, 0, nullptr, &status); // size is 0
  EXPECT_ERROR(status, CL_INVALID_BUFFER_SIZE);

  clCreateBuffer(context, CL_MEM_READ_WRITE, static_cast<size_t>(max_size) + 1, nullptr, &status); // size is greater than CL_DEVICE_MAX_MEM_ALLOC_SIZE for all devices in context
  EXPECT_ERROR(status, CL_INVALID_BUFFER_SIZE);

  cl_mem large_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, static_cast<size_t>(max_size), nullptr, &status);
  EXPECT_SUCCESS(status);
  EXPECT_SUCCESS(clReleaseMemObject(large_buffer));

  cl_buffer_region region = {0, 32};
  cl_mem sub_buffer = clCreateSubBuffer(buffer,
                                        CL_MEM_READ_WRITE,
                                        CL_BUFFER_CREATE_TYPE_REGION,
                                        &region,
                                        &status);
  EXPECT_SUCCESS(status);

  // CL_INVALID_MEM_OBJECT is buffer is a sub-buffer
  clCreateSubBuffer(sub_buffer,
                    CL_MEM_READ_WRITE,
                    CL_BUFFER_CREATE_TYPE_REGION,
                    &region,
                    &status); // buffer is a sub-buffer object
  EXPECT_ERROR(status, CL_INVALID_MEM_OBJECT);

  // CL_MISALIGNED_SUB_BUFFER_OFFSET if there are no devices in context associated with buffer for which the origin field
  // of the cl_buffer_region structure passed in buffer_create_info is aligned to the CL_DEVICE_MEM_BASE_ADDR_ALIGN value.
  // Note: minimum align is something like 16
  cl_buffer_region invalid_region = {1, 32};
  clCreateSubBuffer(buffer,
                    CL_MEM_READ_WRITE,
                    CL_BUFFER_CREATE_TYPE_REGION,
                    &invalid_region,
                    &status);
  EXPECT_ERROR(status, CL_MISALIGNED_SUB_BUFFER_OFFSET);

  EXPECT_SUCCESS(clReleaseMemObject(sub_buffer));
  EXPECT_SUCCESS(clReleaseMemObject(buffer));
  EXPECT_SUCCESS(clReleaseContext(context));

  return param_verification_test::finalize();
}
