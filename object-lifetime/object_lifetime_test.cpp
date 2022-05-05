#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>

struct TestConfig {
  // The version to base testing off.
  cl_version version;
  // Whether CL_*_REFERENCE_COUNT reports implicit or explicit reference count.
  bool ref_count_includes_implicit;
  // Whether released objects can still be used.
  bool use_released_objects;
  // Whether inaccessible objects that are still implicitly referenced can be used.
  bool use_inaccessible_objects;
};

bool parseArgs(TestConfig &cfg, int argc, char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--ref-count-includes-implicit") == 0) {
      cfg.ref_count_includes_implicit = true;
    } else if (std::strcmp(arg, "--use-released-objects") == 0) {
      cfg.use_released_objects = true;
    } else if (std::strcmp(arg, "--use-inaccessible-objects") == 0) {
      cfg.use_inaccessible_objects = true;
    } else {
      fprintf(stderr, "ERROR: invalid argument %s\n", arg);
      return false;
    }
  }

  return true;
}

void expectSuccess(cl_int err, const char * name) {
  if (err != CL_SUCCESS) {
    printf("ERROR: %s (%i)\n", name, err);
    exit(err);
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

namespace ocl_utils {
  cl_int getRefCount(cl_context handle, cl_uint& ref_count) {
    return clGetContextInfo(handle, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_device_id handle, cl_uint& ref_count) {
    return clGetDeviceInfo(handle, CL_DEVICE_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_command_queue handle, cl_uint& ref_count) {
    return clGetCommandQueueInfo(handle, CL_QUEUE_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_mem handle, cl_uint& ref_count) {
    return clGetMemObjectInfo(handle, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_sampler handle, cl_uint& ref_count) {
    return clGetSamplerInfo(handle, CL_SAMPLER_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_program handle, cl_uint& ref_count) {
    return clGetProgramInfo(handle, CL_PROGRAM_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_kernel handle, cl_uint& ref_count) {
    return clGetKernelInfo(handle, CL_KERNEL_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_event handle, cl_uint& ref_count) {
    return clGetEventInfo(handle, CL_EVENT_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  template <typename T>
  cl_int CL_INVALID;

  template <>
  constexpr const cl_int CL_INVALID<cl_context> = CL_INVALID_CONTEXT;

  template <>
  constexpr const cl_int CL_INVALID<cl_device_id> = CL_INVALID_DEVICE;

  template <>
  constexpr const cl_int CL_INVALID<cl_command_queue> = CL_INVALID_COMMAND_QUEUE;

  template <>
  constexpr const cl_int CL_INVALID<cl_mem> = CL_INVALID_MEM_OBJECT;

  template <>
  constexpr const cl_int CL_INVALID<cl_sampler> = CL_INVALID_SAMPLER;

  template <>
  constexpr const cl_int CL_INVALID<cl_program> = CL_INVALID_PROGRAM;

  template <>
  constexpr const cl_int CL_INVALID<cl_kernel> = CL_INVALID_KERNEL;

  template <>
  constexpr const cl_int CL_INVALID<cl_event> = CL_INVALID_EVENT;
}

template <typename Handle>
void expectRefCount(TestConfig &cfg, Handle handle, cl_uint expected_explicit_count, cl_uint expected_implicit_count) {
  cl_uint actual_ref_count;
  cl_int status = ocl_utils::getRefCount(handle, actual_ref_count);
  if (expected_explicit_count == 0) {
    // If the expected explicit ref count is zero:
    // In OpenCL 1, the object is deallocated and so we should get CL_INVALID.
    // - If use_released_objects is true, we expect a refcount of zero and CL_SUCCESS.
    // - use_inaccessible_objects does not apply.
    // In OpenCL 2:
    // - If the object still has implicit references, we expect CL_SUCCESS.
    //   - use_released_objects and use_inaccessible_objects do not apply.
    // - Otherwise:
    //   - if use_release_objects is true, we expect CL_SUCCESS and zero references.
    //   - Else, CL_INVALID.
    // - use_inaccessible_objects does not apply.
    // In OpenCL 3, the object may have implicit references, but it is inaccessible and so we should get CL_INVALID.
    // - If the implicit reference count is zero and use_release_objects is true, we expect CL_SUCCESS and zero references.
    // - Otherwise if implicit reference count is nonzero and use_inaccessible_objects is true, we expect CL_SUCCESS.
    // - Otherwise we expect CL_INVALID.

    if (CL_VERSION_MAJOR(cfg.version) == 1) {
      if (cfg.use_released_objects) {
        assert(status == CL_SUCCESS);
        assert(actual_ref_count == 0);
      } else {
        assert(status == ocl_utils::CL_INVALID<Handle>);
      }
    } else if (CL_VERSION_MAJOR(cfg.version) == 2 && CL_VERSION_MINOR(cfg.version) == 0) {
      if (expected_implicit_count != 0) {
        assert(status == CL_SUCCESS);
        assert(actual_ref_count == expected_implicit_count);
      } else if (cfg.use_released_objects) {
        assert(status == CL_SUCCESS);
        assert(actual_ref_count == 0);
      } else {
        assert(status == ocl_utils::CL_INVALID<Handle>);
      }
    } else /* 2.1, 2.2 and 3.0 */ {
      if (expected_implicit_count == 0 && cfg.use_released_objects) {
        assert(status == CL_SUCCESS);
        assert(actual_ref_count == 0);
      } else if (expected_implicit_count != 0 && cfg.use_inaccessible_objects) {
        assert(status == CL_SUCCESS);
        assert(actual_ref_count == expected_explicit_count);
      } else {
        assert(status == ocl_utils::CL_INVALID<Handle>);
      }
    }
  } else {
    // If the expected explicit ref count is nonzero:
    // In OpenCL 1.1 and 1.2, we expect that the explicit ref count is the actual ref count.
    // In OpenCL 2.0 and above, the actual ref count includes the implicit count as well.
    if (cfg.ref_count_includes_implicit) {
      assert(actual_ref_count == expected_explicit_count + expected_implicit_count);
    } else {
      assert(actual_ref_count == expected_explicit_count);
    }
  }
}

int main(int argc, char *argv[]) {
  TestConfig cfg = { 0, false, false, false };
  parseArgs(cfg, argc, argv);

  cl_int status = CL_SUCCESS;
  cl_uint numPlatforms = 0;

  cl_platform_id platform;
  status = clGetPlatformIDs(1, &platform, &numPlatforms);
  expectSuccess(status, "clGetPlatformIDs");
  if (numPlatforms == 0) {
    printf("No OpenCL platform detected.\n");
    exit(-1);
  }

  status = clGetPlatformInfo(platform, CL_PLATFORM_NUMERIC_VERSION, sizeof(cl_version), &cfg.version, nullptr);
  expectSuccess(status, "clGetPlatformInfo");

  cl_device_id device;
  cl_uint      numDevices;
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, &numDevices);
  expectSuccess(status, "clGetDeviceIDs");
  if(numDevices == 0) {
    printf("No OpenCL device found.\n");
    exit(-1);
  }

  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform), 0};

  cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, &status);
  expectSuccess(status, "clCreateContext");

  // Create a buffer from the context
  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &status);
  expectSuccess(status, "clCreateBuffer");

  // Release the context, but the buffer should keep it alive
  status = clReleaseContext(context);
  expectSuccess(status, "clReleaseContext");
  expectRefCount(cfg, context, 0, 1);

  // Release the buffer, this should also release the context
  status = clReleaseMemObject(buffer);
  expectSuccess(status, "clReleaseMemObject");
  expectRefCount(cfg, buffer, 0, 0);
  expectRefCount(cfg, context, 0, 0);

  // Try to release the context again
  status = clReleaseContext(context);
  expectErr(status, "clReleaseContext");

  fflush(stdout);

  return 0;
}
