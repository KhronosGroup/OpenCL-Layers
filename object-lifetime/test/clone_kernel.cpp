#include "object_lifetime_test.hpp"

int main(int argc, char *argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  layer_test::setup(argc, argv, CL_MAKE_VERSION(2, 1, 0), platform, device);

  cl_context context = layer_test::createContext(platform, device);

  const char* source = "kernel void test_kernel(sampler_t sampler) {}";
  size_t length = strlen(source);
  cl_program program = clCreateProgramWithSource(context,
                                                 1,
                                                 &source,
                                                 &length,
                                                 &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(program, 1, 0);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 0, 1); // used with implicit refcount: 1
  EXPECT_SUCCESS(clBuildProgram(program,
                                1,
                                &device,
                                nullptr,
                                nullptr,
                                nullptr));

  cl_kernel kernel = clCreateKernel(program, "test_kernel", &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(kernel, 1, 0);
  EXPECT_SUCCESS(clReleaseProgram(program));
  EXPECT_REF_COUNT(program, 0, 1); // used with implicit refcount: 1
  EXPECT_SUCCESS(clRetainKernel(kernel));
  EXPECT_REF_COUNT(kernel, 2, 0);

  cl_kernel clone = clCloneKernel(kernel, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(kernel, 2, 0);
  EXPECT_REF_COUNT(clone, 1, 0);
  EXPECT_REF_COUNT(program, 0, 2); // used with implicit refcount: 2
  EXPECT_SUCCESS(clReleaseKernel(kernel));
  EXPECT_SUCCESS(clReleaseKernel(kernel));
  EXPECT_DESTROYED(kernel); // recently deleted with type: KERNEL
  EXPECT_REF_COUNT(clone, 1, 0);
  EXPECT_REF_COUNT(program, 0, 1); // used with implicit refcount: 1

  EXPECT_SUCCESS(clReleaseKernel(clone));
  EXPECT_DESTROYED(clone); // recently deleted with type: KERNEL
  EXPECT_DESTROYED(program); // recently deleted with type: PROGRAM
  EXPECT_DESTROYED(context); // recently deleted with type: CONTEXT

  return layer_test::finalize();
}
