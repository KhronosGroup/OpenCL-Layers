#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <stdio.h>  // printf
#include <stdlib.h> // malloc
#include <stdint.h> // UINTMAX_MAX

void checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS)
    {
        printf("ERROR: %s (%i)\n", name, err);
        exit( err );
    }
}

int main()
{
    cl_int CL_err = CL_SUCCESS;
    cl_uint numPlatforms = 0;
#if defined(_WIN32)
    size_t var_size;
    errno_t err = getenv_s(&var_size, NULL, 0, "OPENCL_LAYERS");
    if(var_size == 0 || err != 0) return -1;
    char* var = (char*)malloc(var_size);
    err = getenv_s(&var_size, var, var_size, "OPENCL_LAYERS");
    if (err != 0) return -1;
#else
    char* var = getenv("OPENCL_LAYERS");
#endif
    if(var != NULL) printf("OPENCL_LAYERS: %s\n", var);

    CL_err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(CL_err, "clGetPlatformIDs(numPlatforms)");

    if (numPlatforms == 0)
    {
        printf("No OpenCL platform detected.\n");
        exit( -1 );
    }
    printf("Found %u platform(s)\n\n", numPlatforms);
    fflush(NULL);

    return 0;
}
