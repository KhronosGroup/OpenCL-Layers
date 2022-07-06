#include "param_verification_test.hpp"
#include <array>
#include <algorithm>
#include <memory>

int main(int argc, char* argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  param_verification_test::setup(argc, argv, CL_MAKE_VERSION(1, 2, 0), platform, device);

  cl_context_properties invalid_properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) 123456, 0};
  cl_context context = clCreateContext(invalid_properties, 1, &device, nullptr, nullptr, &status); // platform value specified in properties is not a valid platform
  EXPECT_ERROR(status, CL_INVALID_PLATFORM);

  clCreateBuffer(context,
                 CL_MEM_READ_ONLY,
                 128,
                 nullptr,
                 &status); // context is not a valid context
  EXPECT_ERROR(status, CL_INVALID_CONTEXT);

  EXPECT_ERROR(clReleaseContext(context), CL_INVALID_CONTEXT); // context is not a vaild context

  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
  context = clCreateContext(properties, 1, &device, nullptr, nullptr, &status);
  EXPECT_SUCCESS(status);

  EXPECT_SUCCESS(clReleaseContext(context));

  clCreateSampler(context,
                 CL_TRUE,
                 CL_ADDRESS_CLAMP,
                 CL_FILTER_NEAREST,
                 &status); // context is not a valid context
  EXPECT_ERROR(status, CL_INVALID_CONTEXT);

  return param_verification_test::finalize();
}

