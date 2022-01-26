#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <stdio.h>  // printf
#include <stdlib.h> // malloc
#include <stdint.h> // UINTMAX_MAX

void expectSuccess(cl_int err, const char * name)
{
    if (err != CL_SUCCESS)
    {
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

void logError(cl_int err, const char * name)
{
    if (err != CL_SUCCESS)
    {
        printf("ERROR: %s (%i)\n", name, err);
    } else {
        printf("SUCCESS: %s\n", name);
    }
}

int main()
{
    cl_int CL_err = CL_SUCCESS;
    cl_uint numPlatforms = 0;

    cl_platform_id platform;
    CL_err = clGetPlatformIDs(1, &platform, &numPlatforms);
    expectSuccess(CL_err, "clGetPlatformIDs");
    if (numPlatforms == 0)
    {
        printf("No OpenCL platform detected.\n");
        exit(-1);
    }
    
    cl_device_id device;
    cl_uint      numDevices;
    CL_err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, &numDevices);
    expectSuccess(CL_err, "clGetDeviceIDs");
    if(numDevices == 0) {
        printf("No OpenCL device found.\n");
        exit(-1);
    }

    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform), 0};

    cl_context context =
        clCreateContext(properties, 1, &device, NULL, NULL, &CL_err);
    expectSuccess(CL_err, "clCreateContext");

    CL_err = clReleaseContext(context);
    expectSuccess(CL_err, "clCreateContext");

    CL_err = clReleaseContext(context);
    logError(CL_err, "clCreateContext");

    fflush(stdout);

    return 0;
}
