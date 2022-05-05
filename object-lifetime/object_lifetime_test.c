#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <stdio.h>  // printf
#include <stdlib.h> // malloc
#include <stdint.h> // UINTMAX_MAX

void expectSuccess(cl_int err, const char * name) {
  if (err != CL_SUCCESS) {
    printf("ERROR: %s (%i)\n", name, err);
    exit( err );
  } else {
    printf("SUCCESS: %s\n", name);
  }
}

void expectErr(cl_int err, const char *name) {
  if (err != CL_SUCCESS) {
    printf("ERROR: %s (%i)\n", name, err);
  } else {
    printf("SUCCESS: %s\n", name);
    printf("Expected error in %s, but no error was set.\n", name);
    exit(-2);
  }
}

void logError(cl_int err, const char * name) {
  if (err != CL_SUCCESS) {
    printf("ERROR: %s (%i)\n", name, err);
  } else {
    printf("SUCCESS: %s\n", name);
  }
}

int main() {
  cl_int status = CL_SUCCESS;
  cl_uint numPlatforms = 0;

  cl_platform_id platform;
  status = clGetPlatformIDs(1, &platform, &numPlatforms);
  expectSuccess(status, "clGetPlatformIDs");
  if (numPlatforms == 0) {
    printf("No OpenCL platform detected.\n");
    exit(-1);
  }

  cl_device_id device;
  cl_uint      numDevices;
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, &numDevices);
  expectSuccess(status, "clGetDeviceIDs");
  if(numDevices == 0) {
    printf("No OpenCL device found.\n");
    exit(-1);
  }

  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform), 0};

  cl_context context = clCreateContext(properties, 1, &device, NULL, NULL, &status);
  expectSuccess(status, "clCreateContext");

  // Create a buffer from the context
  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &status);
  expectSuccess(status, "clCreateBuffer");

  // Release the context, but the buffer should keep it alive
  status = clReleaseContext(context);
  expectSuccess(status, "clReleaseContext");

  // Use the context
  cl_uint ref_count = 0;
  status = clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT,
                            sizeof(cl_uint), &ref_count, NULL);
  expectSuccess(status, "clGetContextInfo");

  // Release the buffer, this should also release the context
  status = clReleaseMemObject(buffer);
  expectSuccess(status, "clReleaseMemObject");

  // Use the context, this should fail as the context should be already deleted at this point
  status = clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT,
                            sizeof(cl_uint), &ref_count, NULL);
  logError(status, "clGetContextInfo");

  // Try to release the context again
  status = clReleaseContext(context);
  logError(status, "clReleaseContext");

  fflush(stdout);

  return 0;
}
