#include "object_lifetime_test.hpp"

int main(int argc, char *argv[]) {
  cl_platform_id platform;
  cl_device_id device;
  cl_int status;
  layer_test::setup(argc, argv, CL_MAKE_VERSION(1, 1, 0), platform, device);

  cl_context context = layer_test::createContext(platform, device);

  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(context, 1, 1);
  EXPECT_REF_COUNT(queue, 1, 0);

  const char* source = "kernel void test_kernel(sampler_t sampler) {}";
  size_t length = strlen(source);
  cl_program program = clCreateProgramWithSource(context,
                                                 1,
                                                 &source,
                                                 &length,
                                                 &status);
  EXPECT_SUCCESS(status);
  EXPECT_SUCCESS(clRetainProgram(program));
  EXPECT_REF_COUNT(context, 1, 2);
  EXPECT_REF_COUNT(program, 2, 0);

  cl_sampler sampler = clCreateSampler(context,
                                       CL_TRUE,
                                       CL_ADDRESS_REPEAT,
                                       CL_FILTER_LINEAR,
                                       &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(sampler, 1, 0);
  EXPECT_REF_COUNT(context, 1, 3);

  EXPECT_SUCCESS(clBuildProgram(program,
                                1,
                                &device,
                                nullptr,
                                nullptr,
                                nullptr));

  cl_kernel kernel = clCreateKernel(program, "test_kernel", &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(kernel, 1, 0);
  EXPECT_REF_COUNT(program, 2, 1);

  cl_event top_of_pipe = clCreateUserEvent(context, &status);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(top_of_pipe, 1, 0);
  EXPECT_REF_COUNT(context, 1, 4);

  EXPECT_SUCCESS(clReleaseContext(context));
  EXPECT_REF_COUNT(context, 0, 4); // used with implicit refcount: 4

  EXPECT_SUCCESS(clRetainEvent(top_of_pipe));
  EXPECT_REF_COUNT(top_of_pipe, 2, 0);

  // Setting kernel arguments should not affect reference counts.
  EXPECT_SUCCESS(clRetainSampler(sampler));
  EXPECT_REF_COUNT(sampler, 2, 0);
  EXPECT_SUCCESS(clSetKernelArg(kernel, 0, sizeof(cl_sampler), static_cast<const void*>(&sampler)));
  EXPECT_REF_COUNT(sampler, 2, 0);
  EXPECT_SUCCESS(clReleaseSampler(sampler));
  EXPECT_REF_COUNT(sampler, 1, 0);

  size_t global_work_size = 2;
  size_t local_work_size = 1;
  cl_event bottom_of_pipe;
  EXPECT_SUCCESS(clEnqueueNDRangeKernel(queue,
                                        kernel,
                                        1,
                                        nullptr,
                                        &global_work_size,
                                        &local_work_size,
                                        1,
                                        &top_of_pipe,
                                        &bottom_of_pipe));
  EXPECT_REF_COUNT(bottom_of_pipe, 1, 0);
  EXPECT_REF_COUNT(top_of_pipe, 2, 0);
  EXPECT_SUCCESS(clSetUserEventStatus(top_of_pipe, CL_COMPLETE));
  EXPECT_SUCCESS(clWaitForEvents(1, &bottom_of_pipe));

  EXPECT_SUCCESS(clRetainKernel(kernel));
  EXPECT_REF_COUNT(kernel, 2, 0);
  EXPECT_SUCCESS(clReleaseKernel(kernel));
  EXPECT_SUCCESS(clReleaseKernel(kernel));
  EXPECT_DESTROYED(kernel); // recently deleted with type: KERNEL

  EXPECT_SUCCESS(clReleaseProgram(program));
  EXPECT_REF_COUNT(program, 1, 0);
  EXPECT_SUCCESS(clReleaseProgram(program));
  EXPECT_DESTROYED(program); // recently deleted with type: PROGRAM
  EXPECT_SUCCESS(clReleaseSampler(sampler));
  EXPECT_DESTROYED(sampler); // recently deleted with type: SAMPLER
  EXPECT_SUCCESS(clReleaseCommandQueue(queue));
  EXPECT_DESTROYED(queue); // recently deleted with type: COMMAND_QUEUE
  EXPECT_SUCCESS(clReleaseEvent(top_of_pipe));
  EXPECT_SUCCESS(clReleaseEvent(top_of_pipe));
  EXPECT_DESTROYED(top_of_pipe); // recently deleted with type: EVENT
  EXPECT_SUCCESS(clReleaseEvent(bottom_of_pipe));
  EXPECT_DESTROYED(bottom_of_pipe); // recently deleted with type: EVENT
  EXPECT_DESTROYED(context); // recently deleted with type: CONTEXT

  return layer_test::finalize();
}

