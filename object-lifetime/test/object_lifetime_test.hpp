#ifndef _OBJECT_LIFETIME_TEST_HPP
#define _OBJECT_LIFETIME_TEST_HPP

#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <iostream>
#include <cstring>

namespace layer_test {
  bool parseArgs(int argc, char* argv[]);

  void setup(int argc,
             char* argv[],
             cl_version required_version,
             cl_platform_id& platform,
             cl_device_id& device);

  int finalize();

  struct TestContext {
    // The version to base testing off.
    cl_version version = CL_MAKE_VERSION(1, 1, 0);
    // Whether CL_*_REFERENCE_COUNT reports implicit or explicit reference count.
    bool ref_count_includes_implicit = false;
    // Whether querying released objects (with an expected total refcount of 0) is expected
    // to return CL_INVALID or the actual ref count.
    bool use_released_objects = false;
    // Whether inaccessible objects that are still implicitly referenced can be used.
    bool use_inaccessible_objects = false;
    // The OpenCL platform to use.
    const char* platform = "Object Lifetime Layer Test ICD";
    // Keeps track number of failed assertions during this test.
    size_t failed_checks = 0;

    void fail() {
      ++this->failed_checks;
    }
  };

  extern TestContext TEST_CONTEXT;

  namespace ocl_utils {
    inline cl_int getRefCount(cl_context handle, cl_uint& ref_count) {
      return clGetContextInfo(handle, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
    }

    inline cl_int getRefCount(cl_device_id handle, cl_uint &ref_count) {
      return clGetDeviceInfo(handle, CL_DEVICE_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
    }

    inline cl_int getRefCount(cl_command_queue handle, cl_uint &ref_count) {
      return clGetCommandQueueInfo(handle, CL_QUEUE_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
    }

    inline cl_int getRefCount(cl_mem handle, cl_uint &ref_count) {
      return clGetMemObjectInfo(handle, CL_MEM_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
    }

    inline cl_int getRefCount(cl_sampler handle, cl_uint &ref_count) {
      return clGetSamplerInfo(handle, CL_SAMPLER_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
    }

    inline cl_int getRefCount(cl_program handle, cl_uint &ref_count) {
      return clGetProgramInfo(handle, CL_PROGRAM_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
    }

    inline cl_int getRefCount(cl_kernel handle, cl_uint &ref_count) {
      return clGetKernelInfo(handle, CL_KERNEL_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
    }

    inline cl_int getRefCount(cl_event handle, cl_uint &ref_count) {
      return clGetEventInfo(handle, CL_EVENT_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr);
    }

    template <typename T>
    cl_int CL_INVALID();

    template <>
    inline cl_int CL_INVALID<cl_platform_id>() { return CL_INVALID_PLATFORM; }

    template <>
    inline cl_int CL_INVALID<cl_context>() { return CL_INVALID_CONTEXT; }

    template <>
    inline cl_int CL_INVALID<cl_device_id>() { return CL_INVALID_DEVICE; }

    template <>
    inline cl_int CL_INVALID<cl_command_queue>() { return CL_INVALID_COMMAND_QUEUE; }

    template <>
    inline cl_int CL_INVALID<cl_mem>() { return CL_INVALID_MEM_OBJECT; }

    template <>
    inline cl_int CL_INVALID<cl_sampler>() { return CL_INVALID_SAMPLER; }

    template <>
    inline cl_int CL_INVALID<cl_program>() { return CL_INVALID_PROGRAM; }

    template <>
    inline cl_int CL_INVALID<cl_kernel>() { return CL_INVALID_KERNEL; }

    template <>
    inline cl_int CL_INVALID<cl_event>() { return CL_INVALID_EVENT; }
  }

  inline std::ostream& log(const char* file, int line) {
    return std::cout << file << ":" << line << ": ";
  }

  #define LAYER_TEST_LOG() ::layer_test::log(__FILE__, __LINE__)

  inline void expectSuccess(const char* file, int line, cl_int status) {
    if (status != CL_SUCCESS) {
      log(file, line) << "expected success, got " << status << std::endl;
      TEST_CONTEXT.fail();
    }
  }

  #define EXPECT_SUCCESS(status) ::layer_test::expectSuccess(__FILE__, __LINE__, status)

  inline void expectError(const char* file, int line, cl_int status, cl_int expected) {
    if (status == CL_SUCCESS) {
      log(file, line) << "expected error " << expected << ", got success" << std::endl;
      TEST_CONTEXT.fail();
    } else if (status != expected) {
      log(file, line) << "expected error " << expected << ", got " << status << std::endl;
      TEST_CONTEXT.fail();
    }
  }

  #define EXPECT_ERROR(status, expected) ::layer_test::expectError(__FILE__, __LINE__, status, expected)

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
      if (status == ocl_utils::CL_INVALID<Handle>()) {
        log(file, line) << "expected that object was not destroyed, but it was" << std::endl;
        TEST_CONTEXT.fail();
      } else if (status != CL_SUCCESS) {
        log(file, line) << "expected that object was not destroyed, got error " << status << std::endl;
        TEST_CONTEXT.fail();
      } else if (actual_ref_count != expected_ref_count) {
        log(file, line) << "expected ref count " << expected_ref_count << ", got " << actual_ref_count << std::endl;
        TEST_CONTEXT.fail();
      }
    };

    auto expect_destroyed = [&] {
      if (TEST_CONTEXT.use_released_objects) {
        expect_not_destroyed(0);
      } else if (status == CL_SUCCESS) {
        log(file, line) << "expected that object was destroyed, but it has " << actual_ref_count << " references remaining" << std::endl;
        TEST_CONTEXT.fail();
      } else if (status != ocl_utils::CL_INVALID<Handle>()) {
        log(file, line) << "expected that object was destroyed, got error " << status << std::endl;
        TEST_CONTEXT.fail();
      }
    };

    if (expected_explicit_count == 0) {
      // If the expected explicit ref count is zero:
      // In OpenCL 1, the object is deallocated and so we should get CL_INVALID.
      // - if ref_count_includes_implicit, it is deallocated when the total refcount hits zero.
      // In OpenCL 2:
      // - If the object still has implicit references, we expect CL_SUCCESS.
      // - Otherwise, CL_INVALID.
      // In OpenCL 3, the object may have implicit references, but it is inaccessible and so we should get CL_INVALID.
      // - If implicit reference count is nonzero and use_inaccessible_objects is true, we expect CL_SUCCESS.
      // - Otherwise we expect CL_INVALID.

      if (TEST_CONTEXT.version >= CL_MAKE_VERSION(1, 1, 0) && TEST_CONTEXT.version <= CL_MAKE_VERSION(1, 2, 0)) {
        if (expected_implicit_count > 0 && TEST_CONTEXT.ref_count_includes_implicit) {
          expect_not_destroyed(expected_implicit_count);
        } else {
          expect_destroyed();
        }
      } else if (TEST_CONTEXT.version == CL_MAKE_VERSION(2, 0, 0)) {
        if (expected_implicit_count > 0 && TEST_CONTEXT.ref_count_includes_implicit) {
          expect_not_destroyed(expected_implicit_count);
        } else if (expected_implicit_count > 0) {
          expect_not_destroyed(0);
        } else {
          expect_destroyed();
        }
      } else if (TEST_CONTEXT.version >= CL_MAKE_VERSION(2, 1, 0)) {
        if (expected_implicit_count > 0 && TEST_CONTEXT.use_inaccessible_objects) {
          if (TEST_CONTEXT.ref_count_includes_implicit) {
            expect_not_destroyed(expected_implicit_count);
          } else {
            expect_not_destroyed(0);
          }
        } else {
          expect_destroyed();
        }
      }
    } else {
      // If the expected explicit ref count is nonzero:
      // In OpenCL 1.1 and 1.2, we expect that the explicit ref count is the actual ref count.
      // In OpenCL 2.0 and above, the actual ref count includes the implicit count as well (if that is enbled).
      if (TEST_CONTEXT.ref_count_includes_implicit) {
        expect_not_destroyed(expected_explicit_count + expected_implicit_count);
      } else {
        expect_not_destroyed(expected_explicit_count);
      }
    }
  }

  #define EXPECT_REF_COUNT(handle, explicit_count, implicit_count) \
    ::layer_test::expectRefCount(__FILE__, __LINE__, handle, explicit_count, implicit_count)

  #define EXPECT_DESTROYED(handle) EXPECT_REF_COUNT(handle, 0, 0)

  inline cl_context createContext(cl_platform_id platform, cl_device_id device) {
    cl_int status;
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform), 0};
    cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, &status);
    EXPECT_SUCCESS(status);
    EXPECT_REF_COUNT(context, 1, 0);

    return context;
  }
}

#endif
