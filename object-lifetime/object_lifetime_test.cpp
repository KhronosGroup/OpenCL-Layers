#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <iostream>
#include <string>
#include <memory>
#include <cstdlib>
#include <cstdint>
#include <cstring>

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

TestConfig TEST_CONFIG = { 0, false, false, false };

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

namespace ocl_utils {
  cl_int getRefCount(cl_context handle, cl_uint& ref_count) {
    return clGetContextInfo(handle, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_device_id handle, cl_uint &ref_count) {
    return clGetDeviceInfo(handle, CL_DEVICE_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_command_queue handle, cl_uint &ref_count) {
    return clGetCommandQueueInfo(handle, CL_QUEUE_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_mem handle, cl_uint &ref_count) {
    return clGetMemObjectInfo(handle, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_sampler handle, cl_uint &ref_count) {
    return clGetSamplerInfo(handle, CL_SAMPLER_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_program handle, cl_uint &ref_count) {
    return clGetProgramInfo(handle, CL_PROGRAM_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_kernel handle, cl_uint &ref_count) {
    return clGetKernelInfo(handle, CL_KERNEL_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
  }

  cl_int getRefCount(cl_event handle, cl_uint &ref_count) {
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

std::ostream& reportError(const char* file, int line) {
  return std::cout << file << ":" << line << ": ";
}

#define REPORT_ERROR() reportError(__FILE__, __LINE__)

void expectSuccess(const char* file, int line, cl_int status) {
  if (status != CL_SUCCESS) {
    reportError(file, line) << "expected success, got " << status << std::endl;
    exit(-1);
  }
}

#define EXPECT_SUCCESS(status) ::expectSuccess(__FILE__, __LINE__, status)

void expectError(const char* file, int line, cl_int status, cl_int expected) {
  if (status == CL_SUCCESS) {
    reportError(file, line) << "expected error " << expected << ", got success" << std::endl;
    exit(-1);
  } else if (status != expected) {
    reportError(file, line) << "expected error " << expected << ", got " << status << std::endl;
    exit(-1);
  }
}

#define EXPECT_ERROR(status, expected) ::expectError(__FILE__, __LINE__, status, expected)

template <typename Handle>
void expectRefCount(const char* file,
                    int line,
                    Handle handle,
                    cl_uint expected_explicit_count,
                    cl_uint expected_implicit_count)
{
  cl_uint actual_ref_count;
  cl_int status = ocl_utils::getRefCount(handle, actual_ref_count);

  auto expect_not_destroyed = [&](cl_uint expected_ref_count) {
    if (status == ocl_utils::CL_INVALID<Handle>) {
      reportError(file, line) << "expected that object was not destroyed, but it was" << std::endl;
      exit(-1);
    } else if (status != CL_SUCCESS) {
      reportError(file, line) << "expected that object was not destroyed, got unexpected error " << status << std::endl;
      exit(-1);
    } else if (actual_ref_count != expected_ref_count) {
      reportError(file, line) << "expected ref count " << expected_ref_count << ", got " << actual_ref_count << std::endl;
      exit(-1);
    }
  };

  auto expect_destroyed = [&] {
    if (status == CL_SUCCESS) {
      reportError(file, line) << "expected that object was destroyed, but it is not" << std::endl;
      exit(-1);
    } else if (status != ocl_utils::CL_INVALID<Handle>) {
      reportError(file, line) << "expected that object was destroyed, got unexpect error " << status << std::endl;
      exit(-1);
    }
  };

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

    if (CL_VERSION_MAJOR(TEST_CONFIG.version) == 1) {
      if (TEST_CONFIG.use_released_objects) {
        expect_not_destroyed(0);
      } else {
        expect_destroyed();
      }
    } else if (CL_VERSION_MAJOR(TEST_CONFIG.version) == 2 && CL_VERSION_MINOR(TEST_CONFIG.version) == 0) {
      if (expected_implicit_count != 0) {
        expect_not_destroyed(expected_implicit_count);
      } else if (TEST_CONFIG.use_released_objects) {
        expect_not_destroyed(0);
      } else {
        expect_destroyed();
      }
    } else /* 2.1, 2.2 and 3.0 */ {
      if (expected_implicit_count == 0 && TEST_CONFIG.use_released_objects) {
        expect_not_destroyed(0);
      } else if (expected_implicit_count != 0 && TEST_CONFIG.use_inaccessible_objects) {
        expect_not_destroyed(expected_implicit_count);
      } else {
        expect_destroyed();
      }
    }
  } else {
    // If the expected explicit ref count is nonzero:
    // In OpenCL 1.1 and 1.2, we expect that the explicit ref count is the actual ref count.
    // In OpenCL 2.0 and above, the actual ref count includes the implicit count as well.
    if (TEST_CONFIG.ref_count_includes_implicit) {
      expect_not_destroyed(expected_explicit_count + expected_implicit_count);
    } else {
      expect_not_destroyed(expected_explicit_count);
    }
  }
}

#define EXPECT_REF_COUNT(handle, explicit_count, implicit_count) \
  ::expectRefCount(__FILE__, __LINE__, handle, explicit_count, implicit_count)

#define EXPECT_DESTROYED(handle) EXPECT_REF_COUNT(handle, 0, 0)

int main(int argc, char *argv[]) {
  parseArgs(TEST_CONFIG, argc, argv);

  cl_int status = CL_SUCCESS;
  cl_uint num_platforms = 0;
  EXPECT_SUCCESS(clGetPlatformIDs(0, nullptr, &num_platforms));

  if (num_platforms == 0) {
    REPORT_ERROR() << "No OpenCL platform detected" << std::endl;
    exit(-1);
  }

  auto platforms = std::make_unique<cl_platform_id[]>(num_platforms);
  EXPECT_SUCCESS(clGetPlatformIDs(num_platforms, platforms.get(), nullptr));

  // Select the mock icd.
  cl_platform_id platform;
  bool           test_icd_platform_found = false;
  for (cl_uint i = 0; i < num_platforms; ++i) {
    size_t name_size;
    EXPECT_SUCCESS(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &name_size));
    auto name = std::make_unique<char[]>(name_size);
    EXPECT_SUCCESS(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, name_size, name.get(), nullptr));
    if (std::strcmp(name.get(), "Object Lifetime Layer Test ICD") == 0) {
      platform = platforms[i];
      test_icd_platform_found = true;
    }
  }
  if (!test_icd_platform_found) {
    REPORT_ERROR() << "OpenCL Test ICD platform not found" << std::endl;
    exit(-1);
  }

  EXPECT_SUCCESS(clGetPlatformInfo(platform, CL_PLATFORM_NUMERIC_VERSION, sizeof(cl_version), &TEST_CONFIG.version, nullptr));

  cl_device_id device;
  cl_uint      numDevices;
  EXPECT_SUCCESS(clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, &numDevices));
  if(numDevices == 0) {
    REPORT_ERROR() << "No OpenCL device found" << std::endl;
    exit(-1);
  }

  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform), 0};

  cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, &status);
  EXPECT_SUCCESS(status);

  // Create a buffer from the context
  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &status);
  EXPECT_SUCCESS(status);

  // Release the context, but the buffer should keep it alive
  status = clReleaseContext(context);
  EXPECT_SUCCESS(status);
  EXPECT_REF_COUNT(context, 0, 1);

  // Release the buffer, this should also release the context
  status = clReleaseMemObject(buffer);
  EXPECT_SUCCESS(status);
  EXPECT_DESTROYED(buffer);
  EXPECT_DESTROYED(context);

  // Try to release the context again
  status = clReleaseContext(context);
  EXPECT_ERROR(status, CL_INVALID_CONTEXT);

  return 0;
}
